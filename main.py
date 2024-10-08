# main.py

import torch
from data_loader import TrajectoryDataset, collate_fn
from vae_model import LSTM_VAE
from train_vae import train_vae
from explain_instance import explain_instance
import matplotlib.pyplot as plt
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # Define directories
    data_dir = 'data'
    plots_dir = os.path.join('plots', 'explanation_plots')
    models_dir = 'models'  # Define models directory

    # Print current working directory for debugging
    print("Current Working Directory:", os.getcwd())

    # Create necessary directories if they don't exist
    for directory in [plots_dir, models_dir]:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Directory '{directory}' is ready.")
        except Exception as e:
            print(f"Failed to create directory '{directory}'. Error: {e}")
            return  # Exit the script if directory creation fails

    # Load datasets
    print("Loading training data...")
    train_dataset = TrajectoryDataset(os.path.join(data_dir, '1_patel_hurricane_2vs3_train.csv'))
    print(f"Training data loaded: {len(train_dataset)} samples.")

    print("Loading test data...")
    test_dataset = TrajectoryDataset(os.path.join(data_dir, '1_patel_hurricane_2vs3_test.csv'))
    print(f"Test data loaded: {len(test_dataset)} samples.")

    # Define model parameters
    input_size = 2  # Longitude and Latitude
    hidden_size = 64
    latent_dim = 16
    num_layers = 1  # Adjust based on your preference

    # Initialize the model
    print("Initializing the LSTM-VAE model...")
    model = LSTM_VAE(input_size=input_size,
                    hidden_size=hidden_size,
                    latent_size=latent_dim,
                    num_layers=num_layers).to(device)

    # Train the VAE
    print("Starting VAE training...")
    model = train_vae(model, train_dataset, num_epochs=10, batch_size=16, learning_rate=1e-3)
    print("VAE training completed.")

    # Save the trained model
    model_path = os.path.join(models_dir, 'lstm_vae_trained.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved at '{model_path}'.")

    # Select a random test instance for explanation
    import random
    random_index = random.randint(0, len(test_dataset) - 1)
    print(f"Generating explanation for test instance at index {random_index}...")

    # Generate explanations
    explanations = explain_instance(model, test_dataset, index=random_index, output_dir=plots_dir)
    print("Explanations generated and plots saved.")

if __name__ == '__main__':
    main()
