from scipy.spatial.distance import euclidean
from collections import Counter
from fastdtw import fastdtw
import numpy as np
def dtw_distance(x1, x2):
    distance, _ = fastdtw(x1, x2, dist=euclidean)
    return distance

class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _predict(self, x):
        # Compute the distances using DTW
        distances = [dtw_distance(x, x_train) for x_train in self.X_train]
        # Get the k-nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_neighbors = [self.y_train[i] for i in k_indices]
        # Majority vote
        most_common = Counter(k_nearest_neighbors).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels).reshape(-1, 1)