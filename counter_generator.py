import numpy as np

class CounterGenerator:
    """
    Generate counter-exemplars in the latent space.
    """
    
    def __init__(self, blackbox, n_search=10000, n_batch=1000):
        self.blackbox = blackbox
        self.n_search = n_search
        self.n_batch = n_batch

    def select_counter_exemplars(self, z, z_label, n=500):
        """
        Select counter-exemplars around the latent vector `z`.
        """
        Z_neighborhood = np.random.normal(loc=z, scale=1.0, size=(n, z.shape[1]))
        predicted_labels = self.blackbox.predict(Z_neighborhood)
        counter_exemplars = Z_neighborhood[predicted_labels != z_label]
        return counter_exemplars

    def explain(self, z, z_label, n=500):
        """
        Generate an explanation for the latent vector `z`.
        """
        counter_exemplars = self.select_counter_exemplars(z, z_label, n=n)
        explanation = {'exemplars': z, 'counter_exemplars': counter_exemplars}
        return explanation
