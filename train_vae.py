# train_vae.py

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_loader import collate_fn  # Import collate_fn directly
from vae_model import LSTM_VAE  # Ensure correct model import

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loss_function(recon_sequences, sequences, seq_lengths, mu, logvar):
    """
    Computes the VAE loss function.
    recon_sequences: Reconstructed sequences from the VAE (batch_size, max_seq_len, input_size)
    sequences: Original sequences (batch_size, max_seq_len, input_size)
    seq_lengths: Lengths of each sequence in the batch
    mu: Mean from the encoder
    logvar: Log variance from the encoder
    """
    # Compute reconstruction loss (MSE) only for the valid timesteps
    batch_size, max_seq_len, input_size = sequences.size()
    mask = torch.arange(max_seq_len).unsqueeze(0).expand(batch_size, max_seq_len).to(device) < seq_lengths.unsqueeze(1)
    mask = mask.unsqueeze(2).expand(-1, -1, input_size)
    
    recon_loss = F.mse_loss(recon_sequences * mask.float(), sequences * mask.float(), reduction='sum') / batch_size
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    return recon_loss + kl_loss

def train_vae(model, dataset, num_epochs=20, batch_size=16, learning_rate=1e-3):
    """
    Trains the LSTM-VAE model.
    """
    # Use the imported collate_fn directly
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for sequences, seq_lengths, _ in dataloader:
            sequences = sequences.to(device)
            seq_lengths = seq_lengths.to(device)
            optimizer.zero_grad()
            recon_sequences, mu, logvar = model(sequences, seq_lengths)
            loss = loss_function(recon_sequences, sequences, seq_lengths, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    return model
