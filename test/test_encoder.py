import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from vae_model import LSTM_VAE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast  # Để chuyển đổi chuỗi sang list

def parse_trajectory(data_str):
    """
    Chuyển đổi chuỗi trajectory thành list của list float.

    Args:
        data_str (str): Chuỗi chứa trajectory, ví dụ: "[[0.0, 108.8, 30.0], ...]"

    Returns:
        list: List của list float.
    """
    try:
        # Sử dụng ast.literal_eval để an toàn chuyển đổi chuỗi thành list
        trajectory = ast.literal_eval(data_str)
        # Loại bỏ các giá trị không cần thiết nếu có (ví dụ: thời gian)
        # Nếu dữ liệu là [time, lat, lon], bạn có thể chỉ lấy [lat, lon]
        # Giả sử dữ liệu là [time, lat, lon], chúng ta sẽ lấy [lat, lon]
        trajectory = [[point[1], point[2]] for point in trajectory]
        return trajectory
    except Exception as e:
        print(f"Error parsing trajectory: {e}")
        return []

def load_data(csv_file):
    """
    Tải dữ liệu từ file CSV và chuyển đổi thành tensor.

    Args:
        csv_file (str): Đường dẫn tới file CSV.

    Returns:
        list of torch.Tensor: Danh sách các trajectory với kích thước (1, seq_len, input_size).
        list of int: Danh sách độ dài thực tế của từng trajectory.
    """
    df = pd.read_csv(csv_file)

    trajectories = []
    seq_lengths = []
    for idx, row in df.iterrows():
        traj = parse_trajectory(row['Data'])
        if traj:
            traj_tensor = torch.tensor(traj, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, input_size)
            trajectories.append(traj_tensor)
            seq_lengths.append(len(traj))

    if not trajectories:
        raise ValueError("No valid trajectories found in the dataset.")

    return trajectories, seq_lengths

def plot_trajectories(original, reconstructed, sample_idx=0, hurricane_name=""):
    """
    Vẽ đồ thị trajectory gốc và trajectory sau khi decode.

    Args:
        original (np.ndarray): Trajectory gốc với kích thước (1, seq_len, 2).
        reconstructed (np.ndarray): Trajectory sau khi decode với kích thước tương tự.
        sample_idx (int): Chỉ số của mẫu trong danh sách để vẽ.
        hurricane_name (str): Tên hurricane để hiển thị trên đồ thị.
    """
    plt.figure(figsize=(8, 6))
    original_traj = original[0, :, :]
    reconstructed_traj = reconstructed[0, :, :]

    # Giả sử cột 0 là latitude và cột 1 là longitude
    plt.plot(original_traj[:, 1], original_traj[:, 0], 'o-', label='Original Trajectory', color='blue')
    plt.plot(reconstructed_traj[:, 1], reconstructed_traj[:, 0], 'x--', label='Reconstructed Trajectory', color='red')

    plt.title(f'Trajectory Comparison for Sample {sample_idx} ({hurricane_name})')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_lstm_vae_trajectory(csv_file):
    # Bước 1: Tải dữ liệu
    trajectories, seq_lengths = load_data(csv_file)
    print(f"Loaded {len(trajectories)} trajectories.")

    # Bước 2: Khởi tạo mô hình
    hidden_size = 32
    latent_size = 16
    num_layers = 2
    input_size = 2  # Lat và Lon
    model = LSTM_VAE(input_size=input_size,
                    hidden_size=hidden_size,
                    latent_size=latent_size,
                    num_layers=num_layers)

    # Bước 3: Đưa mô hình lên thiết bị phù hợp (CPU hoặc GPU)
    device = model.device
    model.to(device)

    # Bước 4: Đặt mô hình ở chế độ đánh giá
    model.eval()

    # Bước 5: Chạy từng trajectory qua mô hình mà không sử dụng padding
    for i, traj in enumerate(trajectories):
        traj = traj.to(device)  # (1, seq_len, input_size)
        length = seq_lengths[i]

        with torch.no_grad():
            x_hat, mean, log_var = model(traj, torch.tensor([length]).to(device))  # Kích thước batch=1

        # Chuyển dữ liệu về CPU và numpy để xử lý
        traj_np = traj.cpu().numpy()
        x_hat_np = x_hat.cpu().numpy()

        # Kiểm tra sự khác biệt giữa đầu vào và đầu ra
        difference = np.abs(traj_np - x_hat_np)
        max_diff = np.max(difference)
        mean_diff = np.mean(difference)
        print(f"Sample {i}: Max difference = {max_diff:.6f}, Mean difference = {mean_diff:.6f}")

        if max_diff < 1e-5:
            print(f"Sample {i}: Đầu vào và đầu ra gần như giống nhau.")
        else:
            print(f"Sample {i}: Đầu vào và đầu ra khác nhau như mong đợi.")

        # Vẽ trajectory
        hurricane_name = df.iloc[i]['HurricaneName'] if i < len(df) else "Unknown"
        plot_trajectories(traj_np, x_hat_np, sample_idx=i, hurricane_name=hurricane_name)

    # Bước 6: Thêm một số kiểm tra cuối cùng
    print("Kiểm thử hoàn thành!")

if __name__ == "__main__":
    # Đường dẫn tới file CSV chứa dữ liệu train
    csv_file = 'data/1_patel_hurricane_2vs3_train.csv'
    test_lstm_vae_trajectory(csv_file)
