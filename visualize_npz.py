import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load .npz file
data = np.load("final.npy")

# Extract arrays
positions = data['positions']
colors = data['colors']
scales = data['scale']      # shape (N, 3)
opacity = data['opacity']   # shape (N,)

# Compute scalar size from 3D scale vectors (use average or norm)
scales_scalar = np.mean(scales, axis=1)

# Normalize colors if needed
if colors.max() > 1.0:
    colors = colors / 255.0

# Add alpha channel using opacity
if colors.shape[1] == 3:
    colors = np.concatenate([colors, opacity[:, np.newaxis]], axis=1)

# Normalize scale for visualization
scales_visual = 300 * (scales_scalar / np.max(scales_scalar))  # adjust multiplier for size

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    positions[:, 0], positions[:, 1], positions[:, 2],
    c=colors,
    s=scales_visual,
    depthshade=False
)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Point Cloud with Color, Scale, and Opacity')
plt.show()
