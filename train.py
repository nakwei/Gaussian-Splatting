import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from read_write_model import *
from gs_utils import *
from gsmodel import *


def save_training_params(fn: str, training_params: dict):
    """Serialize Gaussian Splat training parameters to a compact ``.npy``.

    All auxiliary operations (opacity sigmoid, scale exp, quaternion
    normalization, SH concatenation, dtype creation) are performed here so the
    caller only needs to hand in the raw training‑time tensors.

    Parameters
    ----------
    fn : str
        Output filename (``.npy``).
    training_params : dict
        Dictionary returned by ``get_training_params``.
    """

    import torch.nn.functional as F

    def _dtype(sh_dim: int):
        return [
            ('pw', '<f4', (3,)),  
            ('rot', '<f4', (4,)),  
            ('scale', '<f4', (3,)),
            ('opacity', '<f4'),
            ('sh', '<f4', (sh_dim,)),
        ]


    pw = training_params["points_world"]                     # (N,3)
    sh = torch.cat((training_params["low_spherical_harmonics"],
                    training_params["high_spherical_harmonics"]), dim=1)  # (N, C)
    opacity = torch.sigmoid(training_params["raw_opacity"]).squeeze()     # (N,)
    scales = torch.exp(training_params["scales_raw"])                     # (N,3)
    rots = F.normalize(training_params["rots_raw"], dim=1)                 # (N,4)


    pw, sh, opacity, scales, rots = [x.detach().cpu().numpy() for x in (pw, sh, opacity, scales, rots)]

    gs = np.rec.fromarrays([pw, rots, scales, opacity, sh], dtype=_dtype(sh.shape[1]))
    np.save(fn, gs)



torch.autograd.set_detect_anomaly(True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="the path of dataset")
    args = parser.parse_args()


    gs = read_points_bin_as_gau(Path(args.path, "sparse/0/points3D.bin"))

    training_params, adam_params = get_training_params(gs)
    cameras, images = get_cameras_and_images(args.path)

    optimizer = optim.Adam(adam_params, lr=0.000, eps=1e-15)  

    #tune epochs
    epochs = 100 
    n_imgs = len(images)


    cam_centers = torch.stack([cam.twc for cam in cameras])
    scene_radius = torch.linalg.norm(cam_centers - cam_centers.mean(0), dim=1).max().item() * 1.1

    model = GSModel(scene_radius, n_imgs * epochs)

    # Training loop 
    for epoch in range(epochs):
        idxs = np.random.permutation(n_imgs)
        epoch_loss = 0.0

        for i in idxs:
            cam = cameras[i]
            img_gt = images[i]

            img_pred = model(*training_params.values(), cam)
            loss = gau_loss(img_pred, img_gt)

            loss.backward()
            model.update_density_info()
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            model.update_pws_lr(optimizer)

            epoch_loss += loss.item()

        epoch_loss /= n_imgs
        print(f"epoch:{epoch} avg_loss:{epoch_loss:.6f}")

        with torch.no_grad():
            if 1 < epoch <= 50:
                if epoch % 5 == 0:
                    print("updating gaussian density …")
                    model.update_gaussian_density(training_params, optimizer)
                if epoch % 15 == 0:
                    print("resetting gaussian alpha …")
                    model.reset_alpha(training_params, optimizer)

    #save
    save_training_params('data/final.npy', training_params)
