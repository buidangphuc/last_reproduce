import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

class LastsExplanation:
    def __init__(self, neighgen_explanation, surrogate_explanation):
        self.neighgen_explanation = neighgen_explanation
        self.surrogate_explanation = surrogate_explanation

    def explain(self):
        return self.neighgen_explanation, self.surrogate_explanation

    def highlight_important_points(self, trajectory, feature_names):
        """
        Highlight the important points on the trajectory based on the explanation.
        
        Parameters:
        - trajectory: Original trajectory data (numpy array or tensor) [seq_len, 2]
        - feature_names: Names of the features (latitude, longitude) 
        
        Returns:
        - highlighted_points: List of indices in the trajectory that are important
        """
        highlighted_points = []
        
        # Surrogate explanation (decision tree)
        tree = self.surrogate_explanation

        # Get the important features and their thresholds from the decision tree
        important_features = tree.tree_.feature  # Indices of features (latitude/longitude)
        thresholds = tree.tree_.threshold  # Thresholds for each feature
        
        # Loop through the trajectory and find points that cross these thresholds
        for idx, point in enumerate(trajectory):
            for feature_idx in range(len(feature_names)):
                if feature_idx < important_features.size and important_features[feature_idx] != -2:  # -2 indicates a leaf node
                    threshold = thresholds[important_features[feature_idx]]
                    # If the point in the trajectory crosses the threshold, mark it as important
                    if (point[feature_idx] > threshold) or (point[feature_idx] < threshold):
                        highlighted_points.append(idx)
        
        return highlighted_points

# Train Decision Tree Surrogate
def train_decision_tree(latent_vectors, labels):
    latent_vectors_2d = latent_vectors.reshape(latent_vectors.shape[0], -1)
    clf = DecisionTreeClassifier()
    clf.fit(latent_vectors_2d, labels)
    return clf

# Function to explain a sample and highlight points
def explain_sample(sample, vae, knn, num_neighbors=10):
    sample_latent, _ = vae.encoder(sample)
    neighgen_explanation = generate_neighbors(sample_latent, knn, num_neighbors)
    
    # Predict labels for neighbors and train surrogate model (decision tree)
    latent_vectors, predicted_labels = neighgen_explanation
    tree_converter = train_decision_tree(latent_vectors, predicted_labels)
    
    surrogate_explanation = tree_converter
    explanation = LastsExplanation(neighgen_explanation, surrogate_explanation)
    
    # Highlight important points on the trajectory
    feature_names = ["latitude", "longitude"]
    highlighted_points = explanation.highlight_important_points(sample.squeeze().detach().numpy(), feature_names)
    
    return explanation, highlighted_points

# Neighborhood Generation (latent space)
def generate_neighbors(latent_vector, knn, num_neighbors=10):
    neighbors = knn.kneighbors(latent_vector.detach().numpy().reshape(1, -1), n_neighbors=num_neighbors)
    return neighbors[0], neighbors[1]  # Return neighbor latent vectors and their labels

# Visualize the trajectory and highlight important points
def visualize_highlighted_points(trajectory, highlighted_points):
    """
    Visualizes the trajectory and marks the highlighted important points.
    
    Parameters:
    - trajectory: Original trajectory data [seq_len, 2]
    - highlighted_points: List of indices of important points
    """
    trajectory = np.array(trajectory)
    
    # Plot the full trajectory
    plt.plot(trajectory[:, 0], trajectory[:, 1], label='Trajectory', color='blue', marker='o')
    
    # Highlight the important points
    if highlighted_points:
        highlighted = trajectory[highlighted_points]
        plt.scatter(highlighted[:, 0], highlighted[:, 1], color='red', label='Important Points', marker='x', s=100)
    
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title('Trajectory with Important Points Highlighted')
    plt.legend()
    plt.show()

