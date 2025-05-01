from convert_3dmodel_binary import read_points
from read_write_model import read_points3D_binary
import os
import numpy as np


def create_gaussian(input_path):
    positions = []
    colors = []
    scale = []
    opacity = []
    # from github: https://github.com/colmap/colmap/blob/main/scripts/python/read_write_model.py
    
    points3d = read_points3D_binary(input_path)
    for point in points3d.values():
        positions.append(point.xyz)
        colors.append(np.array(point.rgb) / 255.0)
        scale.append([0.01, 0.01, 0.01])
        opacity.append(0.1)
    
    return positions, colors, scale, opacity
    
def gaussian_pathway(input, output_path = None):
    positions, colors, scale, opacity = create_gaussian(input_path=input)
    positions = np.array(positions, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32)
    scale = np.array(scale, dtype=np.float32)
    opacity = np.array(opacity, dtype=np.float32)
    if output_path == None:
        output_path = './output_guassians.npz'
    np.savez(output_path, positions = positions,
             colors = colors, 
             scale = scale, 
             opacity = opacity)


if __name__ == "__main__":
    import sys
    input_path = sys.argv[1]
    gaussian_pathway(input_path)