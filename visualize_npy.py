from plyfile import PlyData, PlyElement
import numpy as np

def npy_to_ply_with_plyfile(npy_path, ply_path):
    # 1) load
    gaussians = np.load(npy_path, allow_pickle=True)
    names = gaussians.dtype.names
    N = len(gaussians)

    # 2) build a flat array of dicts
    vertices = []
    for g in gaussians:
        d = {}
        for name in names:
            vals = g[name]
            # rename alpha → opacity
            key = "opacity" if name == "alpha" else name
            if isinstance(vals, np.ndarray):
                for i, v in enumerate(vals):
                    d[f"{key}_{i}"] = float(v)
            else:
                d[key] = float(vals)
        vertices.append(d)

    # 3) infer dtype
    #    plyfile wants a structured numpy array
    #    with a field for each key in d
    dtype = []
    for k, v in vertices[0].items():
        dtype.append((k, 'f4'))
    arr = np.zeros(N, dtype=dtype)
    for i, v in enumerate(vertices):
        for k, val in v.items():
            arr[i][k] = val

    # 4) write
    el = PlyElement.describe(arr, 'vertex')
    PlyData([el], text=True).write(ply_path)