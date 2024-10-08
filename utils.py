# utils.py

import numpy as np

def black_box_model_predict(trajectory):
    """
    Placeholder for the actual black-box model prediction.
    Replace this function with your actual classification model.
    """
    # Example: Simple rule based on average longitude
    avg_longitude = np.mean(trajectory[:, 0])
    if avg_longitude > 0.5:
        return 3  # Class label 3
    else:
        return 2  # Class label 2
