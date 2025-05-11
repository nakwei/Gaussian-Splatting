import numpy as np
import sys

_kind_map = {
    'f4': 'float',
    'f8': 'double',
    'i4': 'int',
    'i8': 'int',
    'u1': 'uchar',
    'u2': 'ushort',
    'u4': 'uint',
}

def ply_type(dt):
    code = dt.str.lstrip('<>')  # e.g. 'f4'
    return _kind_map.get(code, 'float')

def npy_to_ply(npy_path, ply_path):
    # Load your structured array
    gaussians = np.load("final.npy", allow_pickle=True)
    
    names = gaussians.dtype.names
    N = len(gaussians)
    
    with open(ply_path, 'w') as f:
        # Write PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        
        for name in names:
            prop = "opacity" if name == "alpha" else name
            dt, _ = gaussians.dtype.fields[name]
            if dt.shape:  # vector field
                for i in range(dt.shape[0]):
                    f.write(f"property {ply_type(dt.base)} {prop}_{i}\n")
            else:         # scalar field
                f.write(f"property {ply_type(dt)} {prop}\n")
        
        f.write("end_header\n")
        
        # Write data rows
        for g in gaussians:
            vals = []
            for name in names:
                v = g[name]
                if isinstance(v, np.ndarray):
                    vals.extend(v.tolist())
                else:
                    vals.append(float(v))
            f.write(" ".join(map(str, vals)) + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python npy_to_ply.py input.npy output.ply")
    else:
        npy_to_ply(sys.argv[1], sys.argv[2])
