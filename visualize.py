import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_npz(npz_path):
    data = np.load(npz_path)
    positions = data['positions']
    colors = data['colors']

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=colors,
        s=1,
        alpha=0.5
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Initialized Gaussians")
    plt.show()

# Run it like this:
# visualize_npz('output_gaussians.npz')

if __name__ == "__main__":
    import sys
    input_path = sys.argv[1]
    visualize_npz(input_path)