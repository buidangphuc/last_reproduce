import torch
from torch import nn
import torch.optim as optim

class LSTM_VAE(nn.Module):
    def __init__(self, input_dim=2, latent_dim=2, hidden_dim=64):  # Updated input_dim to 2 for lat, lon
        super(LSTM_VAE, self).__init__()
        # LSTM Encoder
        self.lstm_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # LSTM Decoder
        self.lstm_decoder = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, input_dim)  # Ensure this matches input_dim=2

    def encoder(self, x):
        _, (h_n, _) = self.lstm_encoder(x)  # Use the final hidden state
        mean = self.fc_mean(h_n[-1])
        logvar = self.fc_logvar(h_n[-1])
        return mean, logvar
    
    def decoder(self, z, seq_len):
        # Repeat the latent vector for each time step
        z_repeated = z.unsqueeze(1).repeat(1, seq_len, 1)
        decoded_seq, _ = self.lstm_decoder(z_repeated)
        return self.fc_out(decoded_seq)
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x, seq_len):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        return self.decoder(z, seq_len), mean, logvar

# Train the VAE
def train_vae(train_data, vae, optimizer, epochs=10):
    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        for ts in train_data:
            optimizer.zero_grad()
            seq_len = ts.shape[1]
            recon, mean, logvar = vae(ts, seq_len)
            loss = vae_loss(recon, ts, mean, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_data)}')

# Loss function for VAE (Reconstruction + KL Divergence)
def vae_loss(recon_x, x, mean, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return BCE + KLD
