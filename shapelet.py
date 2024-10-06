import torch
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class ShapeletExtractor:
    """
    Learn shapelets from exemplars and counter-exemplars, transform data, and extract decision rules.
    """
    
    def __init__(self, exemplars, counter_exemplars):
        self.exemplars = exemplars
        self.counter_exemplars = counter_exemplars

    def learn_shapelets(self, num_shapelets=5):
        shapelets = []
        for i in range(num_shapelets):
            exemplar = self.exemplars[i % len(self.exemplars)]
            counter_exemplar = self.counter_exemplars[i % len(self.counter_exemplars)]
            shapelet = torch.abs(exemplar - counter_exemplar)
            shapelets.append(shapelet)
        return shapelets

    def transform_with_shapelets(self, shapelets, latent_vectors):
        transformed_data = []
        for latent_vector in latent_vectors:
            matches = [torch.dist(latent_vector, shapelet) for shapelet in shapelets]
            transformed_data.append(matches)
        return torch.tensor(transformed_data)

    def train_shapelet_tree(self, transformed_data, labels):
        transformed_np = transformed_data.numpy()
        shapelet_tree = DecisionTreeClassifier()
        shapelet_tree.fit(transformed_np, labels)
        return shapelet_tree

    def extract_shapelet_rules(self, shapelet_tree):
        rules = []
        tree_ = shapelet_tree.tree_

        def recurse(node, depth=0):
            indent = "  " * depth
            if tree_.feature[node] != -2:
                name = f"shapelet_{tree_.feature[node]}"
                threshold = tree_.threshold[node]
                rules.append(f"{indent}if {name} <= {threshold:.2f}:")
                recurse(tree_.children_left[node], depth + 1)
                rules.append(f"{indent}else:  # if {name} > {threshold:.2f}")
                recurse(tree_.children_right[node], depth + 1)
            else:
                rules.append(f"{indent}return {tree_.value[node].argmax()}")

        recurse(0)
        return rules

    def run(self, latent_vectors, labels, num_shapelets=5):
        shapelets = self.learn_shapelets(num_shapelets)
        transformed_data = self.transform_with_shapelets(shapelets, latent_vectors)
        shapelet_tree = self.train_shapelet_tree(transformed_data, labels)
        rules = self.extract_shapelet_rules(shapelet_tree)
        return rules
