import numpy as np
from sklearn.tree import DecisionTreeClassifier
from shapelet_tree import ShapeletTree
import matplotlib.pyplot as plt

class LastsExplanation:
    def __init__(self, neighgen_explanation, surrogate_explanation, shapelet_explanation=None):
        self.neighgen_explanation = neighgen_explanation
        self.surrogate_explanation = surrogate_explanation
        self.shapelet_explanation = shapelet_explanation 

    def explain(self):
        explanations = {
            "neighborhood": self.neighgen_explanation,
            "surrogate": self.surrogate_explanation,
        }
        if self.shapelet_explanation:
            explanations["shapelet"] = self.shapelet_explanation
        return explanations

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

# Train ShapeletTree Surrogate
def train_shapelet_tree(latent_vectors, labels):
    shapelet_tree = ShapeletTree(
        random_state=0, 
        shapelet_model_kwargs={"l": 0.1, "r": 2, "optimizer": "sgd", "n_shapelets_per_size": "heuristic", "max_iter": 100}
    )
    shapelet_tree.fit(latent_vectors, labels)
    return shapelet_tree

# Function to explain a sample and highlight points
def explain_sample(sample, vae, knn, num_neighbors=10):
    # Encode the sample to latent space using VAE
    sample_latent, _ = vae.encoder(sample)
    neighgen_explanation = generate_neighbors(sample_latent, knn, num_neighbors)
    
    # Predict labels for neighbors and train surrogate model (decision tree)
    latent_vectors, predicted_labels = neighgen_explanation
    tree_converter = train_decision_tree(latent_vectors, predicted_labels)
    
    # Train ShapeletTree on the same latent data for enhanced explanations
    shapelet_tree = train_shapelet_tree(latent_vectors, predicted_labels)
    
    surrogate_explanation = tree_converter
    shapelet_explanation = shapelet_tree  # Adding the shapelet-based explanation
    
    explanation = LastsExplanation(neighgen_explanation, surrogate_explanation, shapelet_explanation)
    
    # Highlight important points on the trajectory
    feature_names = ["latitude", "longitude"]
    highlighted_points = explanation.highlight_important_points(sample.squeeze().detach().numpy(), feature_names)
    
    return explanation, highlighted_points

# Neighborhood Generation (latent space)
def generate_neighbors(latent_vector, knn, num_neighbors=10):
    neighbors = knn.kneighbors(latent_vector.detach().numpy().reshape(1, -1), n_neighbors=num_neighbors)
    return neighbors[0], neighbors[1]  # Return neighbor latent vectors and their labels
