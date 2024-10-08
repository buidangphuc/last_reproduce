# tests/test_shapelet_transform.py

import unittest
from pyts.transformation import ShapeletTransform
import numpy as np

class TestShapeletTransform(unittest.TestCase):
    def test_shapelet_transform(self):
        # Create dummy data
        X = np.random.rand(10, 100)  # 10 samples, 100 time steps
        y = np.random.randint(2, size=10)  # Binary labels

        # Initialize ShapeletTransform
        try:
            shapelet_transform = ShapeletTransform(
                n_shapelets=10,
                window_sizes=[5, 10],
                random_state=42
            )
            X_transformed = shapelet_transform.fit_transform(X, y)
            self.assertIsNotNone(X_transformed, "ShapeletTransform did not transform data.")
            print('ShapeletTransform initialized and transformed data successfully.')
        except TypeError as e:
            self.fail(f"ShapeletTransform initialization failed: {e}")

if __name__ == '__main__':
    unittest.main()
