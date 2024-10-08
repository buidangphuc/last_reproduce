# explain_instance.py

import torch
import numpy as np
from utils import black_box_model_predict
from sklearn.tree import DecisionTreeClassifier, export_text
from pyts.transformation import ShapeletTransform
import matplotlib.pyplot as plt
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def find_counterfactual(z, target_label, model, surrogate, dataset, seq_len, max_iter=1000, step_size=0.1):
    """
    Finds a counterfactual latent vector that is predicted as the target_label using the surrogate model.
    """
    z_cf = z.clone().detach().requires_grad_(True)
    optimizer_cf = torch.optim.Adam([z_cf], lr=step_size)
    for iteration in range(max_iter):
        optimizer_cf.zero_grad()
        x_cf = model.decoder(z_cf.unsqueeze(0), torch.tensor([seq_len]).to(device))
        x_cf_np = x_cf.detach().cpu().numpy()[0][:seq_len]
        x_cf_denorm = dataset.inverse_transform(x_cf_np)
        x_cf_flat = torch.FloatTensor(x_cf_denorm.flatten()).unsqueeze(0).to(device)
        output = surrogate.predict(x_cf_flat.cpu().numpy())
        if output == target_label:
            print(f'Counterfactual found at iteration {iteration}.')
            break
        # Dummy loss since surrogate is not differentiable (DecisionTree)
        # To implement gradient-based search, use a differentiable surrogate
        # Here, we terminate if no improvement
        loss = torch.tensor(0.0, requires_grad=True)
        loss.backward()
        optimizer_cf.step()
    else:
        print('Counterfactual not found within the maximum iterations.')
    return z_cf.detach()

def generate_neighborhood(z_original, z_counterfactual, num_samples=100):
    """
    Generates a neighborhood of latent vectors between the original and counterfactual.
    """
    alphas = torch.linspace(0, 1, steps=num_samples).unsqueeze(1).to(device)
    z_original = z_original.unsqueeze(0)
    z_counterfactual = z_counterfactual.unsqueeze(0)
    zs = (1 - alphas) * z_original + alphas * z_counterfactual
    return zs

def decode_and_classify(zs, model, dataset, seq_len):
    """
    Decodes latent vectors back to trajectories and classifies them using the black-box model.
    """
    trajectories = []
    labels = []
    seq_lengths = torch.tensor([seq_len] * len(zs)).to(device)
    with torch.no_grad():
        recon_sequences, _, _ = model(zs, seq_lengths)
        recon_sequences = recon_sequences.cpu().numpy()
        for i in range(len(zs)):
            x_decoded_i = recon_sequences[i][:seq_len]
            x_denorm = dataset.inverse_transform(x_decoded_i)
            trajectories.append(x_denorm)
            label = black_box_model_predict(x_denorm)
            labels.append(label)
    return trajectories, labels

def select_exemplars(trajectories, labels, x_original_denorm, original_label, num_exemplars=3):
    """
    Selects exemplars and counterexemplars based on distance to the original trajectory.
    """
    distances = [np.linalg.norm(t - x_original_denorm) for t in trajectories]
    sorted_indices = np.argsort(distances)
    exemplars = []
    counterexemplars = []
    for idx in sorted_indices:
        if labels[idx] == original_label and len(exemplars) < num_exemplars:
            exemplars.append(trajectories[idx])
        elif labels[idx] != original_label and len(counterexemplars) < num_exemplars:
            counterexemplars.append(trajectories[idx])
        if len(exemplars) == num_exemplars and len(counterexemplars) == num_exemplars:
            break
    return exemplars, counterexemplars

def explain_instance(model, dataset, index, output_dir='plots/explanation_plots'):
    """
    Generates explanations for a specific instance in the dataset, including visualization.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    x_original, seq_len_original, y_original = dataset[index]
    x_original = x_original.unsqueeze(0).to(device)
    seq_len_original = torch.tensor([seq_len_original]).to(device)
    y_original = y_original

    model.eval()
    with torch.no_grad():
        recon_sequences, mu, log_var = model(x_original, seq_len_original)
        z_original = mu.squeeze(0)

    # Define target label (e.g., the other class)
    possible_labels = [2, 3]  # Adjust based on your actual labels
    target_label = next(l for l in possible_labels if l != y_original)

    # Prepare data for surrogate training
    # Generate a small neighborhood for surrogate training
    zs_neighborhood = generate_neighborhood(z_original, z_original, num_samples=100)  # Placeholder
    
    # Decode trajectories in neighborhood
    trajectories_neighborhood, labels_neighborhood = decode_and_classify(zs_neighborhood, model, dataset, seq_len_original.item())

    # Flatten trajectories for ShapeletTransform
    flattened_trajectories = [t.flatten() for t in trajectories_neighborhood]
    labels_array = np.array(labels_neighborhood)

    # Initialize ShapeletTransform
    try:
        shapelet_transform = ShapeletTransform(
            n_shapelets=10,
            window_sizes=[5, 10],
            random_state=42
        )
    except TypeError:
        # Fallback for older pyts versions
        shapelet_transform = ShapeletTransform(
            n_shapelets=10,
            min_shapelet_length=5,
            max_shapelet_length=10,
            random_state=42
        )

    # Fit and transform the data
    X_transformed = shapelet_transform.fit_transform(flattened_trajectories, labels_array)

    # Train surrogate decision tree
    decision_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    decision_tree.fit(X_transformed, labels_array)

    # Extract decision rules
    rules = export_text(decision_tree, feature_names=[f'Shapelet_{i}' for i in range(X_transformed.shape[1])])

    # Find counterfactual
    # Since DecisionTreeClassifier is not differentiable, we'll find a counterfactual by selecting a sample from target class
    counter_indices = np.where(labels_array == target_label)[0]
    if len(counter_indices) > 0:
        chosen_cf_index = np.random.choice(counter_indices)
        z_counterfactual = zs_neighborhood[chosen_cf_index]
    else:
        print("No counterfactual found in the neighborhood.")
        return

    # Decode counterfactual trajectory
    with torch.no_grad():
        x_cf_decoded, _, _ = model(z_counterfactual.unsqueeze(0), torch.tensor([seq_len_original.item()]).to(device))
        x_cf_np = x_cf_decoded.cpu().numpy()[0][:seq_len_original.item()]
        x_cf_denorm = dataset.inverse_transform(x_cf_np)

    # Original trajectory (denormalized)
    x_original_denorm = dataset.inverse_transform(x_original.cpu().numpy()[0][:seq_len_original.item()])

    # Compute saliency map
    saliency_map = np.abs(x_original_denorm - x_cf_denorm)

    # Select exemplars and counterexemplars
    exemplars, counterexemplars = select_exemplars(trajectories_neighborhood, labels_neighborhood, x_original_denorm, y_original)

    # Visualize the original trajectory, counterfactual, and important shapelets
    plt.figure(figsize=(10, 6))
    plt.plot(x_original_denorm[:, 0], x_original_denorm[:, 1], 'b-o', label='Original Trajectory')
    plt.plot(x_cf_denorm[:, 0], x_cf_denorm[:, 1], 'r--x', label='Counterfactual Trajectory')

    # Highlight important shapelets based on decision tree rules
    # Note: ShapeletTransform does not directly provide shapelet positions in the original trajectory
    # To map shapelets back, we need to find the best match positions
    # For demonstration, we'll select the top shapelets based on the decision tree importance

    # Get feature importances from the decision tree
    feature_importances = decision_tree.feature_importances_
    top_shapelet_indices = feature_importances.argsort()[-3:][::-1]  # Top 3 shapelets

    for shapelet_idx in top_shapelet_indices:
        shapelet = shapelet_transform.shapelets_[shapelet_idx]  # Shape: (n_shapelet_samples, shapelet_length)
        # Find the best match in the original trajectory
        best_match = None
        min_distance = float('inf')
        for i in range(len(x_original_denorm) - shapelet.shape[1] + 1):
            window = x_original_denorm[i:i+shapelet.shape[1], 0]  # Using longitude for simplicity
            distance = np.linalg.norm(window - shapelet)
            if distance < min_distance:
                min_distance = distance
                best_match = i
        if best_match is not None:
            end = best_match + shapelet.shape[1]
            plt.plot(x_original_denorm[best_match:end, 0],
                     x_original_denorm[best_match:end, 1],
                     'g-', linewidth=3, label='Important Shapelet' if shapelet_idx == top_shapelet_indices[0] else "")
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Trajectory with Important Shapelets Highlighted')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'instance_{index}_explanation.png'))
    plt.show()

    # Print decision rules
    print("Decision Rules from Surrogate Model:\n")
    print(rules)

    # Optionally, save saliency map
    np.save(os.path.join(output_dir, f'instance_{index}_saliency_map.npy'), saliency_map)
