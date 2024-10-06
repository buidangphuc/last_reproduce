import numpy as np
import torch


class LatentNeighborhoodGenerator:
    
    def __init__(self, vae, blackbox_model):
        self.vae = vae
        self.blackbox_model = blackbox_model

    def generate_latent_neighborhood(self, z, num_neighbors=10, perturbation_scale=0.1):
        z_np = z.detach().cpu().numpy()
        latent_neighbors = []

        for _ in range(num_neighbors):
            perturbation = np.random.normal(0, perturbation_scale, size=z_np.shape)
            z_perturbed = z_np + perturbation
            latent_neighbors.append(z_perturbed)

        latent_neighbors_np = np.array(latent_neighbors)
        return latent_neighbors_np

    def decode_latent_neighborhood(self, latent_neighbors, seq_lengths):
        decoded_sequences = []
        for z_perturbed in latent_neighbors:
            z_perturbed = np.expand_dims(z_perturbed, axis=0)  # Add batch dimension
            z_repeated = np.repeat(z_perturbed, max(seq_lengths), axis=1)  # Repeat across time-step dimension
            z_repeated_tensor = torch.tensor(z_repeated, dtype=torch.float32)  # Convert to tensor before decoding
            
            # Ensure z_repeated_tensor has the shape (batch_size, seq_len, latent_dim)
            assert z_repeated_tensor.shape[2] == self.vae.latent_dim, f"Expected latent_dim: {self.vae.latent_dim}, but got {z_repeated_tensor.shape[2]}"
            
            decoded_seq = self.vae.decoder(z_repeated_tensor, seq_lengths)
            decoded_sequences.append(decoded_seq.detach().cpu().numpy())  # Convert back to NumPy array
        return np.array(decoded_sequences)


    def classify_neighborhood(self, decoded_sequences):
        decoded_np = decoded_sequences.reshape(decoded_sequences.shape[0], -1)
        predictions = self.blackbox_model.predict(decoded_np)
        return predictions

    def run(self, z, seq_lengths, num_neighbors=10, perturbation_scale=0.1):
        latent_neighbors = self.generate_latent_neighborhood(z, num_neighbors, perturbation_scale)
        decoded_sequences = self.decode_latent_neighborhood(latent_neighbors, seq_lengths)
        predictions = self.classify_neighborhood(decoded_sequences)
        return predictions
