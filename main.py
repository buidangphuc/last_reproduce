import torch
from vae import LSTM_VAE, train_vae
from explanation import explain_sample
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data (latitude, longitude)
def read_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    
    # Remove time and keep latitude and longitude
    df['Data'] = df['Data'].apply(lambda x: np.array(eval(x))[:, 1:])  # Latitude and longitude only
    
    # Convert each trajectory to PyTorch tensor
    time_series_tensors = [torch.tensor(ts, dtype=torch.float32).unsqueeze(0) for ts in df['Data'].values]
    
    return time_series_tensors, df['Label'].values

# Load train and test data
train_file_path = '1_patel_hurricane_2vs3_train.csv'
test_file_path = '1_patel_hurricane_2vs3_test.csv'

train_data, train_labels = read_and_prepare_data(train_file_path)
test_data, test_labels = read_and_prepare_data(test_file_path)

# Initialize the VAE
vae = LSTM_VAE(input_dim=2, latent_dim=2, hidden_dim=64)
optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

# Train the VAE
train_vae(train_data, vae, optimizer, epochs=10)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
latent_train = [vae.encoder(ts)[0].detach().numpy() for ts in train_data]
knn.fit(np.vstack(latent_train), train_labels)

# Randomly select one test sample for explanation
random_idx = np.random.randint(0, len(test_data))
sample = test_data[random_idx]

# Generate explanation and highlight points
explanation, highlighted_points = explain_sample(sample, vae, knn, num_neighbors=10)

# Display explanation
print(f"Explanation for sample {random_idx}:")
print(explanation.explain())

# Display the highlighted points on the trajectory
print(f"Highlighted important points on the trajectory: {highlighted_points}")

def visualize_hurricane_trajectory(sample, highlighted_points=None):
    # Convert the tensor to a numpy array
    sample_np = sample.squeeze().detach().numpy()
    latitudes = sample_np[:, 0]
    longitudes = sample_np[:, 1]

    # Plot the full trajectory in blue
    plt.figure(figsize=(10, 6))
    plt.plot(longitudes, latitudes, label="Trajectory", marker="o", color="blue", linestyle="-")

    # Highlight the important points (if provided) in red
    if highlighted_points is not None and len(highlighted_points) > 0:
        important_latitudes = latitudes[highlighted_points]
        important_longitudes = longitudes[highlighted_points]
        plt.scatter(important_longitudes, important_latitudes, color="red", s=100, zorder=5, label="Important Points")

    # Adding labels and title
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Hurricane Trajectory with Highlighted Points")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
# visualize_hurricane_trajectory(sample, highlighted_points)
visualize_hurricane_trajectory(sample, highlighted_points)
