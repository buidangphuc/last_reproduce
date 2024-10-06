from sklearn.tree import DecisionTreeClassifier
from tree import SklearnDecisionTreeConverter

class LatentDecisionTreeExtractor:
    """
    Learn and extract rules from a decision tree trained on latent space representations.
    """
    def __init__(self, vae, max_depth=5):
        self.vae = vae
        self.classifier = DecisionTreeClassifier(max_depth=max_depth)

    def train_latent_tree(self, latent_vectors, labels):
        latent_np = latent_vectors.cpu().numpy()
        self.classifier.fit(latent_np, labels)
        self.tree_converter = SklearnDecisionTreeConverter(self.classifier)
        return self.tree_converter

    def extract_factual_rule(self, sample_idx):
        leaf_idx = self.classifier.apply(self.vae.latent_space[sample_idx].cpu().numpy().reshape(1, -1))[0]
        factual_rule = self.tree_converter.get_factual_rule_by_idx(leaf_idx)
        return factual_rule

    def extract_counterfactual_rule(self, sample_idx):
        leaf_idx = self.classifier.apply(self.vae.latent_space[sample_idx].cpu().numpy().reshape(1, -1))[0]
        factual_node = self.tree_converter._get_node_by_idx(leaf_idx)
        counterfactual_rule = self.tree_converter.get_counterfactual_rule(factual_node)
        return counterfactual_rule
