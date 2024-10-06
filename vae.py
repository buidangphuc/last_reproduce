import torch
from torch import nn
import torch.optim as optim

class LSTM_VAE(nn.Module):
    """
    Variational Autoencoder (VAE) using LSTM layers for encoding and decoding time-series data.
    """
    
    def __init__(self, input_dim=2, latent_dim=128, hidden_dim=64):
        super(LSTM_VAE, self).__init__()
        self.lstm_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.lstm_decoder = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def encoder(self, x_packed):
        packed_output, (h_n, _) = self.lstm_encoder(x_packed)
        mean = self.fc_mean(h_n[-1])
        logvar = self.fc_logvar(h_n[-1])
        return mean, logvar
    
    def decoder(self, z, seq_lengths):
        z = z.unsqueeze(1)  # Add time-step dimension
        z_repeated = z.repeat(1, max(seq_lengths), 1)  # Repeat across time-step dimension
        decoded_seq, _ = self.lstm_decoder(z_repeated)
        return self.fc_out(decoded_seq)


    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x_padded, seq_lengths):
        x_packed = nn.utils.rnn.pack_padded_sequence(x_padded, seq_lengths, batch_first=True, enforce_sorted=False)
        mean, logvar = self.encoder(x_packed)
        z = self.reparameterize(mean, logvar)
        return self.decoder(z, seq_lengths), mean, logvar

# Loss function for VAE
def vae_loss(recon_x, x, mean, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return BCE + KLD
