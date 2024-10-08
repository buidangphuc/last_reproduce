# tests/test_vae_model.py

import unittest
import torch
from vae_model import VAE
from data_loader import TrajectoryDataset, collate_fn

class TestVAEModel(unittest.TestCase):
    def test_vae_forward(self):
        train_dataset = TrajectoryDataset('data/train.csv')
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn)

        input_size = 2
        hidden_size = 64
        latent_dim = 16
        output_size = 2
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = VAE(input_size, hidden_size, latent_dim, output_size).to(device)

        sequences, seq_lengths, _ = next(iter(dataloader))
        sequences = sequences.to(device)
        seq_lengths = seq_lengths.to(device)
        recon_sequences, mu, logvar = model(sequences, seq_lengths)

        self.assertEqual(recon_sequences.shape, sequences.shape, "Reconstructed sequences shape mismatch.")
        self.assertEqual(mu.shape[0], sequences.shape[0], "Mu shape mismatch.")
        self.assertEqual(logvar.shape[0], sequences.shape[0], "Logvar shape mismatch.")
        print('VAE forward pass successful.')

if __name__ == '__main__':
    unittest.main()
