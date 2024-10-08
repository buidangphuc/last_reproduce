# lstm_vae_trajectory.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast
import os

# =========================
# 1. Chuẩn Bị Dữ Liệu
# =========================

class TrajectoryDataset(Dataset):
    def __init__(self, csv_file, mean=None, std=None):
        """
        csv_file: Đường dẫn tới file CSV chứa dữ liệu.
        mean, std: Nếu đã tính trước, cung cấp để chuẩn hóa.
        """
        self.dataframe = pd.read_csv(csv_file)
        self.trajectories = self.process_data(self.dataframe['Data'])
        
        # Tính mean và std nếu chưa có
        if len(self.trajectories) == 0:
            raise ValueError("No valid trajectories found in the dataset.")
        
        all_points = np.vstack(self.trajectories)
        if mean is None or std is None:
            self.mean = all_points.mean(axis=0)
            self.std = all_points.std(axis=0)
        else:
            self.mean = mean
            self.std = std
        
        # Chuẩn hóa dữ liệu
        self.trajectories = [(traj - self.mean) / self.std for traj in self.trajectories]

    def process_data(self, data_series):
        """
        Chuyển đổi chuỗi dữ liệu từ cột 'Data' thành danh sách các numpy arrays với [latitude, longitude].
        """
        trajectories = []
        for idx, data_str in enumerate(data_series):
            # Sử dụng ast.literal_eval để chuyển chuỗi thành list
            try:
                data_list = ast.literal_eval(data_str)
                # Loại bỏ thời gian và giữ lại [latitude, longitude]
                traj = np.array([[point[2], point[1]] for point in data_list], dtype=np.float32)
                # Kiểm tra shape
                if traj.ndim != 2 or traj.shape[1] != 2:
                    print(f"Invalid trajectory shape at index {idx}: {traj.shape}")
                    continue
                trajectories.append(traj)
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing data at index {idx}: {e}")
                continue
        return trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj = torch.tensor(traj, dtype=torch.float32)  # Kích thước: [seq_len, 2]
        return traj, traj  # Input và target đều là cùng một chuỗi

def load_data(csv_file):
    """
    Tạo Dataset và DataLoader từ file CSV.
    """
    dataset = TrajectoryDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    return dataset, dataloader

# =========================
# 2. Định Nghĩa Mô Hình LSTM-VAE
# =========================

class LSTM_VAE(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, latent_dim=32, num_layers=1):
        super(LSTM_VAE, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder LSTM
        self.fc_decode = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
        # Lớp Linear để chuyển đổi output trở lại hidden_dim
        self.output_to_hidden = nn.Linear(input_dim, hidden_dim)

    def encode(self, x):
        # x: [batch, seq_len, input_dim]
        _, (h_n, _) = self.encoder_lstm(x)
        h_n = h_n[-1]  # Lấy hidden state của lớp cuối cùng
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        # z: [batch, latent_dim]
        hidden = self.fc_decode(z)  # [batch, hidden_dim]
        hidden = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)  # [num_layers, batch, hidden_dim]
        cell = torch.zeros_like(hidden)  # [num_layers, batch, hidden_dim]

        # Khởi tạo input cho decoder (zeros)
        decoder_input = torch.zeros(z.size(0), 1, self.hidden_dim).to(z.device)  # [batch, 1, hidden_dim]

        outputs = []
        input_step = decoder_input

        for _ in range(seq_len):
            out, (hidden, cell) = self.decoder_lstm(input_step, (hidden, cell))
            out = self.output_layer(out)  # [batch, 1, input_dim]
            outputs.append(out)
            # Chuyển đổi output trở lại hidden_dim trước khi đưa vào decoder_lstm
            input_step = self.output_to_hidden(out)  # [batch, 1, hidden_dim]

        outputs = torch.cat(outputs, dim=1)  # [batch, seq_len, input_dim]
        return outputs

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        seq_len = x.size(1)
        x_recon = self.decode(z, seq_len)
        return x_recon, mu, logvar

# =========================
# 3. Huấn Luyện Mô Hình
# =========================

def train_model(model, dataloader, num_epochs=100, learning_rate=1e-3, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_recon = nn.MSELoss()

    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0
        for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
            input_seq = input_seq.to(device)  # [1, seq_len, 2]
            target_seq = target_seq.to(device)

            optimizer.zero_grad()

            recon_seq, mu, logvar = model(input_seq)

            # Kiểm tra shape của input_seq và recon_seq
            if recon_seq.shape != input_seq.shape:
                print(f"Shape mismatch at epoch {epoch}, batch {batch_idx}:")
                print(f"Input shape: {input_seq.shape}")
                print(f"Reconstructed shape: {recon_seq.shape}")
                continue  # Bỏ qua loss nếu shape không khớp

            # Reconstruction loss
            loss_recon = criterion_recon(recon_seq, target_seq)

            # KL Divergence
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_divergence /= input_seq.size(0)

            # Tổng loss
            loss = loss_recon + kl_divergence

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

    print("Đã hoàn thành huấn luyện mô hình.")
    return model

# =========================
# 4. Đánh Giá Mô Hình
# =========================

def evaluate_model(model, dataset, mean, std, device='cpu', save_plots=False, plot_dir='plots'):
    model.to(device)
    model.eval()
    os.makedirs(plot_dir, exist_ok=True)
    
    total_mse = 0
    count = 0
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            input_seq, target_seq = dataset[idx]
            input_seq = input_seq.unsqueeze(0).to(device)  # [1, seq_len, 2]
            recon_seq, _, _ = model(input_seq)
            recon_seq = recon_seq.squeeze(0).cpu().numpy()
            target_seq = target_seq.numpy()

            # Chuyển đổi ngược lại từ chuẩn hóa
            recon_seq = recon_seq * std + mean
            target_seq = target_seq * std + mean

            # Kiểm tra shape
            if recon_seq.shape != target_seq.shape:
                print(f"Shape mismatch for trajectory {idx + 1}:")
                print(f"Original shape: {target_seq.shape}")
                print(f"Reconstructed shape: {recon_seq.shape}")
                continue  # Bỏ qua việc vẽ nếu shape không khớp

            # Tính MSE cho trajectory này
            mse = np.mean((recon_seq - target_seq) ** 2)
            total_mse += mse
            count += 1

            # Vẽ trajectory gốc và trajectory được tái tạo
            plt.figure(figsize=(8, 6))
            plt.plot(target_seq[:, 0], target_seq[:, 1], 'b-', label='Original Trajectory')
            plt.plot(recon_seq[:, 0], recon_seq[:, 1], 'r--', label='Reconstructed Trajectory')
            plt.legend()
            plt.title(f"Trajectory {idx + 1} - MSE: {mse:.4f}")
            plt.xlabel("Latitude")
            plt.ylabel("Longitude")

            if save_plots:
                plt.savefig(os.path.join(plot_dir, f"trajectory_{idx + 1}.png"))
                plt.close()
            else:
                plt.show()
    
    if count > 0:
        average_mse = total_mse / count
        print(f"Average MSE across {count} trajectories: {average_mse:.4f}")
    else:
        print("No trajectories were evaluated.")

# =========================
# 5. Chạy Mô Hình
# =========================

def main():
    # Đặt biến môi trường để tránh lỗi OpenMP (tạm thời)
    # Lưu ý: Đây là giải pháp không an toàn và chỉ nên sử dụng khi không thể khắc phục xung đột thư viện
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Thiết lập thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    # Đường dẫn tới file CSV của bạn
    csv_file = 'data/1_patel_hurricane_2vs3_train.csv'  # Thay thế bằng đường dẫn thực tế

    # Kiểm tra xem file CSV có tồn tại không
    if not os.path.exists(csv_file):
        print(f"File CSV '{csv_file}' không tồn tại. Vui lòng kiểm tra đường dẫn.")
        return

    # Tạo Dataset và DataLoader
    dataset, dataloader = load_data(csv_file)
    mean = dataset.mean
    std = dataset.std

    # Kiểm tra shape của một trajectory mẫu
    if len(dataset) > 0:
        sample_input, sample_target = dataset[0]
        print(f"Sample input shape: {sample_input.shape}")  # [seq_len, 2]
        print(f"Sample target shape: {sample_target.shape}")  # [seq_len, 2]
    else:
        print("Dataset is empty.")
        return

    # Khởi tạo mô hình
    model = LSTM_VAE(input_dim=2, hidden_dim=64, latent_dim=32, num_layers=1)

    # Huấn luyện mô hình
    num_epochs = 10
    learning_rate = 1e-3
    print("Bắt đầu huấn luyện mô hình...")
    trained_model = train_model(model, dataloader, num_epochs=num_epochs, learning_rate=learning_rate, device=device)

    # Đánh giá mô hình
    print("Đánh giá mô hình trên dữ liệu huấn luyện...")
    evaluate_model(trained_model, dataset, mean, std, device=device, save_plots=False)

    # Lưu mô hình đã huấn luyện
    torch.save(trained_model.state_dict(), "lstm_vae_trajectory.pth")
    print("Đã lưu mô hình vào 'lstm_vae_trajectory.pth'.")

if __name__ == "__main__":
    main()
