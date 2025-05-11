import torch
import numpy as np
import torch.optim as optim
from pathlib import Path
from read_write_model import *
from borrowed import *
from gsmodel import *


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="dataset root")
    parser.add_argument("--downscale", type=float, default=1.0,
                        help="divide width/height by this factor when loading images (e.g. 2 → 50% size)")

    args = parser.parse_args()

    gs = read_points_bin_as_gau(Path(args.path, "sparse/0/points3D.bin"))
    training_params, adam_params = get_training_params(gs)

    cameras, images = get_cameras_and_images(args.path, downscale=args.downscale)

    optimizer = optim.Adam(adam_params, lr=0.000, eps=1e-15)
    epochs = 100
    n_imgs = len(images)

    twcs = torch.stack([cam.twc for cam in cameras])
    scene_size = float(torch.max(torch.linalg.norm(twcs - torch.mean(twcs, axis=0), dim=1))) * 1.1
    model = GSModel(scene_size, n_imgs * epochs)

    for epoch in range(epochs):
        order = np.random.permutation(n_imgs)
        epoch_loss = 0.0

        for idx in order:
            cam = cameras[idx]
            img_gt = images[idx]
            img_pred = model(*training_params.values(), cam)
            loss = gau_loss(img_pred, img_gt)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            model.update_density_info()
            model.update_pws_lr(optimizer)
            epoch_loss += loss.item()

        print(f"epoch {epoch:03d}  avg_loss {epoch_loss / n_imgs:.6f}")

        with torch.no_grad():
            if 1 < epoch <= 50:
                if epoch % 5 == 0:
                    model.update_gaussian_density(training_params, optimizer)
                if epoch % 15 == 0:
                    model.reset_alpha(training_params, optimizer)

    save_training_params("data/final.npy", training_params)
