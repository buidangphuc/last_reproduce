# test_vae.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from vae_model import LSTM_VAE
from data_loader import TrajectoryDataset, collate_fn
import os

def test_decoder_generation(model_path, data_path, num_samples=10, latent_dim=16, output_dir='plots/generated_trajectories'):
    """
    Tests the VAE decoder's ability to generate new trajectories from latent vectors.

    Args:
        model_path (str): Path to the trained VAE model weights (.pth file).
        data_path (str): Path to the dataset (train.csv or test.csv) for inverse transformation.
        num_samples (int): Number of trajectories to generate.
        latent_dim (int): Dimension of the latent space.
        output_dir (str): Directory to save generated trajectory plots.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the dataset to access inverse_transform
    dataset = TrajectoryDataset(data_path)
    
    # Initialize the model (ensure parameters match the trained model)
    input_size = 2  # Longitude and Latitude
    hidden_size = 64
    num_layers = 1
    model = LSTM_VAE(input_size=input_size,
                    hidden_size=hidden_size,
                    latent_size=latent_dim,
                    num_layers=num_layers).to(device)
    
    # Load the trained model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample latent vectors from standard normal
    z_samples = torch.randn(num_samples, latent_dim).to(device)
    
    # For each z, generate a trajectory
    generated_trajectories = []
    with torch.no_grad():
        # Create dummy decoder inputs: zeros
        # Assuming max_seq_len=60 as per your LSTM_VAE class
        max_seq_len = 60
        decoder_input = torch.zeros(num_samples, max_seq_len, input_size).to(device)
        # Pack the decoder input
        packed_decoder_input = torch.nn.utils.rnn.pack_padded_sequence(
            decoder_input, torch.full((num_samples,), max_seq_len, dtype=torch.int64), 
            batch_first=True, enforce_sorted=False
        )
        # Decode
        generated_x_hat = model.decoder(z_samples, packed_decoder_input, hidden=None)
        # generated_x_hat: (batch_size, max_seq_len, input_size)
        generated_x_hat = generated_x_hat.cpu().numpy()
        # Inverse transform
        for i in range(num_samples):
            trajectory = dataset.inverse_transform(generated_x_hat[i])
            generated_trajectories.append(trajectory)
    
    # Plot generated trajectories
    for i, traj in enumerate(generated_trajectories):
        plt.figure(figsize=(8,6))
        plt.plot(traj[:, 0], traj[:, 1], marker='o')
        plt.title(f'Generated Trajectory {i+1}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'generated_trajectory_{i+1}.png'))
        plt.close()
    
    print(f"Generated {num_samples} trajectories and saved to '{output_dir}'.")

if __name__ == '__main__':
    # Example usage
    # Path to the trained model (ensure this path is correct)
    trained_model_path = 'models/lstm_vae_trained.pth'  # Update this path as needed
    # Path to the dataset (train.csv or test.csv)
    data_file_path = os.path.join('data', '1_patel_hurricane_2vs3_test.csv')  # Ensure this file exists
    # Number of samples to generate
    num_generated_samples = 10
    # Latent dimension (should match the trained model's latent_dim)
    latent_dimension = 16
    # Output directory
    output_directory = os.path.join('plots', 'generated_trajectories')
    
    test_decoder_generation(model_path=trained_model_path,
                            data_path=data_file_path,
                            num_samples=num_generated_samples,
                            latent_dim=latent_dimension,
                            output_dir=output_directory)
