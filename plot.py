import matplotlib.pyplot as plt

class ShapeletPlotter:
    """
    Plot shapelets on a given trajectory.
    """
    
    def plot_shapelet_on_trajectory(self, trajectory, shapelet, start_idx, end_idx):
        plt.figure(figsize=(10, 6))
        plt.plot(trajectory[:, 0], trajectory[:, 1], label="Trajectory", color='blue')
        plt.plot(trajectory[start_idx:end_idx, 0], trajectory[start_idx:end_idx, 1], label="Shapelet", color='red', linewidth=2.5)
        plt.scatter(trajectory[start_idx:end_idx, 0], trajectory[start_idx:end_idx, 1], color='red', label="Shapelet Points")
        plt.xlabel("Latitude")
        plt.ylabel("Longitude")
        plt.legend()
        plt.show()

    def plot_multiple_shapelets_on_trajectory(self, trajectory, shapelets, positions):
        plt.figure(figsize=(10, 6))
        plt.plot(trajectory[:, 0], trajectory[:, 1], label="Trajectory", color='blue')
        for i, (shapelet, (start_idx, end_idx)) in enumerate(zip(shapelets, positions)):
            plt.plot(trajectory[start_idx:end_idx, 0], trajectory[start_idx:end_idx, 1], label=f"Shapelet {i+1}", linewidth=2.5)
            plt.scatter(trajectory[start_idx:end_idx, 0], trajectory[start_idx:end_idx, 1], label=f"Shapelet Points {i+1}")
        plt.xlabel("Latitude")
        plt.ylabel("Longitude")
        plt.legend()
        plt.show()
