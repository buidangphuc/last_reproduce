# tests/test_explain_instance.py

import unittest
from vae_model import VAE
from data_loader import TrajectoryDataset, collate_fn
from train_vae import train_vae
from explain_instance import explain_instance
import torch

class TestExplainInstance(unittest.TestCase):
    def test_explain_instance(self):
        train_dataset = TrajectoryDataset('data/train.csv')
        test_dataset = TrajectoryDataset('data/test.csv')
        input_size = 2
        hidden_size = 64
        latent_dim = 16
        output_size = 2
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = VAE(input_size, hidden_size, latent_dim, output_size).to(device)

        # Train the VAE for a few epochs
        model = train_vae(model, train_dataset, num_epochs=1, batch_size=2, learning_rate=1e-3)

        # Generate explanations for the first test instance
        explanations = explain_instance(model, test_dataset, index=0)

        self.assertIn('saliency_map', explanations, "Saliency map missing in explanations.")
        self.assertIn('exemplars', explanations, "Exemplars missing in explanations.")
        self.assertIn('counterexemplars', explanations, "Counterexemplars missing in explanations.")
        self.assertIn('decision_rules', explanations, "Decision rules missing in explanations.")
        print('Explanation generated successfully.')

if __name__ == '__main__':
    unittest.main()
