import numpy as np
import struct
import os

def reorder_mapping(R: int):
    """Return mapping such that rest_read = rest_ply[mapping].

    This is the mapping produced by the reshape/transpose logic in gau_io.py.
    """
    arr = np.arange(R, dtype=np.int32)
    reshaped = arr.reshape(1, 3, R // 3)
    reordered = reshaped.transpose(0, 2, 1).reshape(-1)
    return reordered

def convert_npy_to_ply_bin(npy_path: str, ply_path: str):
    """
    Convert a Gaussian Splat .npy file (as produced by gsplat)
    to a binary‐little‐endian PLY that Meshlab/CloudCompare can open.
    """
    gs = np.load(npy_path)
    num = len(gs)

    sh_dim = gs['sh'].shape[1]
    sh_rest_dim = sh_dim - 3
    assert sh_rest_dim % 3 == 0, "SH rest dimension must be divisible by 3"

    # Build inverse mapping that undo the ordering used in gau_io.load_ply
    mapping = reorder_mapping(sh_rest_dim)
    inverse_mapping = np.zeros_like(mapping)
    inverse_mapping[mapping] = np.arange(sh_rest_dim)

    # Compose header ---------------------------------------------------------
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {num}",
        "property float x",
        "property float y",
        "property float z",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float opacity",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
    ]
    for i in range(sh_rest_dim):
        header_lines.append(f"property float f_rest_{i}")
    header_lines.append("end_header\n")
    header_blob = ("\n".join(header_lines)).encode("ascii")

    # Prepare all arrays -----------------------------------------------------
    pws = gs['pw']                        # (N, 3)
    rots = gs['rot']                      # (N, 4)
    scales_raw = np.log(gs['scale'] + 1e-8)  # inverse of exp() used in loader
    # logistic inverse for opacity
    alpha_clipped = np.clip(gs['alpha'], 1e-7, 1 - 1e-7)
    alphas_raw = np.log(alpha_clipped / (1 - alpha_clipped))
    sh = gs['sh']                         # (N, sh_dim)
    # reorder SH rest coefficients into the layout expected by PLY
    rest_ply = sh[:, 3:][:, inverse_mapping]

    # Write file -------------------------------------------------------------
    with open(ply_path, "wb") as f:
        f.write(header_blob)

        fmt_row = "<" + "f" * (3 + 4 + 3 + 1 + sh_dim)
        pack = struct.Struct(fmt_row).pack

        for i in range(num):
            row = (
                *pws[i],
                *rots[i],
                *scales_raw[i],
                float(alphas_raw[i]),
                *sh[i, :3],
                *rest_ply[i],
            )
            f.write(pack(*row))

    print(f"Converted {num} gaussians from {npy_path} -> {ply_path}")

# ---------------------------------------------------------------------------
src = "final-5.npy"
dst = "final5.ply"

if not os.path.exists(dst):
    convert_npy_to_ply_bin(src, dst)
else:
    print("PLY already exists – skipping conversion.")


