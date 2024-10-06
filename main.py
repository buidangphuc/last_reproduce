import torch
import numpy as np
from vae import LSTM_VAE, vae_loss
from neighen import LatentNeighborhoodGenerator 
from counter_generator import CounterGenerator
from latent_tree import LatentDecisionTreeExtractor
from shapelet import ShapeletExtractor
from plot import ShapeletPlotter
import random
import pandas as pd
from model import KNN

def read_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    
    # Remove time and keep latitude and longitude
    df['Data'] = df['Data'].apply(lambda x: np.array(eval(x))[:, 1:])  # Latitude and longitude only
    
    # Convert each trajectory to PyTorch tensor
    time_series_tensors = [torch.tensor(ts, dtype=torch.float32).unsqueeze(0) for ts in df['Data'].values]
    
    return time_series_tensors, df['Label'].values

def run_all_flow(train_file, test_file):
    # Step 1: Read and prepare train and test data
    train_data_tensors, train_labels = read_and_prepare_data(train_file)
    test_data_tensors, test_labels = read_and_prepare_data(test_file)
    print(test_data_tensors)
    
    # Initialize the VAE model
    vae = LSTM_VAE(input_dim=2, latent_dim=128, hidden_dim=64)  # Latitude and longitude only
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
    
    # Train the VAE
    for epoch in range(2):  # You can adjust the number of epochs
        total_loss = 0
        for trajectory in train_data_tensors:
            optimizer.zero_grad()
            seq_len = trajectory.size(1)
            recon, mean, logvar = vae(trajectory, [seq_len])  # Pass sequence length
            loss = vae_loss(recon, trajectory, mean, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/10], Loss: {total_loss/len(train_data_tensors)}")
    
    # Flatten the data for KNN (ensure that we use the raw time-series data, not the latent representations)
    X_train = [ts.squeeze(0).numpy() for ts in train_data_tensors]  # Use the time-series data for KNN with DTW
    X_test = [ts.squeeze(0).numpy() for ts in test_data_tensors]

    # Step 2: Use your custom KNN as the black-box model
    knn_model = KNN(k=5)  # Example k=5, you can tune this
    knn_model.fit(X_train, train_labels)

    # Select a random test sample for explanation
    random_test_sample = random.choice(test_data_tensors)
    test_sample_tensor = random_test_sample  # Already a tensor

    # Step 3: Encode the test sample using the VAE encoder
    latent_test_sample, _ = vae.encoder(test_sample_tensor)
    
    # Step 4: Generate and classify neighborhood using LatentNeighborhoodGenerator
    latent_neighgen = LatentNeighborhoodGenerator(vae, knn_model)  # Initialize LatentNeighborhoodGenerator
    predictions = latent_neighgen.run(latent_test_sample.squeeze(0), [test_sample_tensor.size(1)])
    print(f"Neighborhood Predictions: {predictions}")
    
    # Step 5: Generate counter-exemplars using the CounterGenerator
    counter_generator = CounterGenerator(blackbox=vae)
    explanation = counter_generator.explain(latent_test_sample.squeeze(0).numpy(), z_label=1, n=500)  # z_label=1 (example)
    
    exemplars = torch.tensor(explanation['exemplars'], dtype=torch.float32)
    counter_exemplars = torch.tensor(explanation['counter_exemplars'], dtype=torch.float32)
    
    # Step 6: Train the latent decision tree using LatentDecisionTreeExtractor
    latent_tree_extractor = LatentDecisionTreeExtractor(vae)  # Initialize latent tree extractor
    tree_model = latent_tree_extractor.train_latent_tree(latent_test_sample.squeeze(0).cpu().numpy(), predictions)  # Train decision tree

    # Step 7: Extract factual and counterfactual rules from the decision tree
    factual_rule = latent_tree_extractor.extract_factual_rule(sample_idx=0)
    counterfactual_rule = latent_tree_extractor.extract_counterfactual_rule(sample_idx=0)
    
    print("Factual Rule:", factual_rule)
    print("Counterfactual Rule:", counterfactual_rule)
    
    # Step 8: Learn shapelets and extract rules
    shapelet_extractor = ShapeletExtractor(exemplars, counter_exemplars)
    shapelet_rules = shapelet_extractor.run(latent_test_sample.squeeze(0), predictions, num_shapelets=5)
    print("Extracted Shapelet Rules:")
    for rule in shapelet_rules:
        print(rule)
    
    # Step 9: Plot the shapelets on the original trajectory
    shapelet_plotter = ShapeletPlotter()
    shapelets = shapelet_extractor.learn_shapelets(num_shapelets=2)
    shapelet_plotter.plot_multiple_shapelets_on_trajectory(np.array(random_test_sample.squeeze(0)), shapelets, [(3, 6), (10, 13)])

if __name__ == "__main__":
    # Paths to your train and test CSV files
    train_file = "1_patel_hurricane_2vs3_train.csv"
    test_file = "1_patel_hurricane_2vs3_test.csv"
    
    run_all_flow(train_file, test_file)