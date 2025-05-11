import os
import torch
import numpy as np
from math import exp
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
import faiss
from read_write_model import *


class Camera:
    def __init__(self, id, width, height, fx, fy, cx, cy, Rcw, tcw, path):
        self.id = id
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.Rcw = Rcw
        self.tcw = tcw
        self.twc = -torch.linalg.inv(Rcw) @ tcw
        self.path = path



def get_training_params(gs):
    pws = torch.from_numpy(gs['pw']).type(
        torch.float32).to('cuda').requires_grad_()
    rots_raw = torch.from_numpy(gs['rot']).type(
        # the unactivated scales
        torch.float32).to('cuda').requires_grad_()
    scales_raw = get_scales_raw(torch.from_numpy(gs['scale']).type(
        torch.float32).to('cuda')).requires_grad_()
    # the unactivated alphas
    alphas_raw = get_alphas_raw(torch.from_numpy(gs['alpha'][:, np.newaxis]).type(
        torch.float32).to('cuda')).requires_grad_()
    shs = torch.from_numpy(gs['sh']).type(
        torch.float32).to('cuda')
    low_shs = shs[:, :3]
    high_shs = torch.ones_like(low_shs).repeat(1, 15) * 0.001
    high_shs[:, :shs[:, 3:].shape[1]] = shs[:, 3:]
    low_shs = low_shs.requires_grad_()
    high_shs = high_shs.requires_grad_()
    params = {"pws": pws, "low_shs": low_shs, "high_shs": high_shs,
                "alphas_raw": alphas_raw, "scales_raw": scales_raw, "rots_raw": rots_raw}

    adam_params = [
        {'params': [params['pws']], 'lr': 0.001, "name": "pws"},
        {'params': [params['low_shs']],
            'lr': 0.001, "name": "low_shs"},
        {'params': [params['high_shs']],
            'lr': 0.001/20, "name": "high_shs"},
        {'params': [params['alphas_raw']],
            'lr': 0.05, "name": "alphas_raw"},
        {'params': [params['scales_raw']],
            'lr': 0.005, "name": "scales_raw"},
        {'params': [params['rots_raw']], 'lr': 0.001, "name": "rots_raw"}]

    return params, adam_params


def get_scales_raw(x):
    if isinstance(x, float):
        return np.log(x)
    else:
        return torch.log(x)


#we can rewrite our own version of this but i think we use for now:
#(could/should go in read_write_model.py):
def read_points_bin_as_gau(path_to_model_file):
    """
    read colmap points file as inital gaussians
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        pws = np.zeros([num_points, 3])
        shs = np.zeros([num_points, 3])
        for i in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            pws[i] = np.array(binary_point_line_properties[1:4])
            shs[i] = (np.array(binary_point_line_properties[4:7]) /
                      255 - 0.5) / (0.28209479177387814)
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
        rots = np.zeros([num_points, 4])
        rots[:, 0] = 1
        alphas = np.ones([num_points]) * 0.8
        pws = pws.astype(np.float32)
        rots = rots.astype(np.float32)
        alphas = alphas.astype(np.float32)
        shs = shs.astype(np.float32)

        N, D = pws.shape
        index = faiss.IndexFlatL2(D)
        index.add(pws)
        distances, indices = index.search(pws, 2)
        distances = np.clip(distances[:, 1], 0.01, 3)
        scales = distances[:, np.newaxis].repeat(3, 1)

        dtypes = [('pw', '<f4', (3,)),
                  ('rot', '<f4', (4,)),
                  ('scale', '<f4', (3,)),
                  ('alpha', '<f4'),
                  ('sh', '<f4', (3,))]

        gs = np.rec.fromarrays(
            [pws, rots, scales, alphas, shs], dtype=dtypes)

        return gs
    

#not borrowed function but mostly borrowed code:

def get_cameras_and_images(path, *, downscale: float = 1.0, device: str = "cuda"):
    """Return Camera list **and** list of GPU tensors for training.

    If `downscale > 1`, each photo is resized on the CPU via Pillow/LANCZOS
    before converting to a float32 tensor on the GPU.  Using `downscale = 2`
    halves width & height (¼ pixels) and usually cuts memory 4× with negligible
    visual impact.
    """
    camera_params = read_cameras_binary(os.path.join(Path(path, "sparse/0"), "cameras.bin"))
    image_params = read_images_binary(os.path.join(Path(path, "sparse/0"), "images.bin"))

    cameras = []
    images = []

    for img_par in image_params.values():
        cam_spec = camera_params[img_par.camera_id]
        img_path = Path(path, "images", img_par.name)

        img = Image.open(img_path).convert("RGB")
        if downscale != 1.0:
            w, h = img.size
            img = img.resize((int(w / downscale), int(h / downscale)), Image.LANCZOS)

        # scale intrinsics to match possible resize
        w_scale = img.width / cam_spec.width
        h_scale = img.height / cam_spec.height
        fx = cam_spec.params[0] * w_scale
        fy = cam_spec.params[1] * h_scale
        cx = cam_spec.params[2] * w_scale
        cy = cam_spec.params[3] * h_scale

        Rcw = torch.from_numpy(img_par.qvec2rotmat()).to(device).float()
        tcw = torch.from_numpy(img_par.tvec).to(device).float()

        cameras.append(
            Camera(img_par.id, img.width, img.height, fx, fy, cx, cy, Rcw, tcw, str(img_path))
        )
        images.append(TF.to_tensor(img).to(device))  # float32 in [0,1]

    return cameras, images

def get_alphas_raw(x):
    """
    inverse of sigmoid
    """
    if isinstance(x, float):
        return np.log(x/(1-x))
    else:
        return torch.log(x/(1-x))
    





#next section is all for calculating loss

def gau_loss(image, gt_image, loss_lambda=0.2):
    loss_l1 = torch.abs((image - gt_image)).mean()
    loss_ssim = 1.0 - ssim(image, gt_image)
    return (1.0 - loss_lambda) * loss_l1 + loss_lambda * loss_ssim

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.shape[-3]
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(
        img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(
        img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size //
                       2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    

def rotate_vector_by_quaternion(q, v):
    q = torch.nn.functional.normalize(q)
    u = q[:, 1:, np.newaxis]
    s = q[:, 0, np.newaxis, np.newaxis]
    v = v[:, :,  np.newaxis]
    v_prime = 2.0 * u * (u.permute(0, 2, 1) @ v) +\
        v * (s*s - (u.permute(0, 2, 1) @ u)) +\
        2.0 * torch.linalg.cross(u, v, dim=1) * s
    return v_prime.squeeze()

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    this file is copied from Plenoxels
    https://github.com/sxyu/svox2/blob/ee80e2c4df8f29a407fda5729a494be94ccf9234/opt/util/util.py#L78

    Continuous learning rate decay function. Adapted from JaxNeRF

    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.

    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def get_alphas(x):
    return torch.sigmoid(x)

def get_scales(x):
    return torch.exp(x)

def get_rots(x):
    return torch.nn.functional.normalize(x)


def get_shs(low_shs, high_shs):
    return torch.cat((low_shs, high_shs), dim=1)


_kind_map = {
    'f4': 'float',
    'f8': 'double',
    'i4': 'int',
    'i8': 'int',
    'u1': 'uchar',
    'u2': 'ushort',
    'u4': 'uint',
}

def _ply_type(dt):
    code = dt.str.lstrip('<>')    # e.g. 'f4'
    return _kind_map.get(code, 'float')
def write_gaussians_as_ply(fn: str, arr: np.recarray) -> None:
    """
    Export a Gaussian‑splat recarray to an ASCII PLY that follows the
    Graph‑DECO header convention:

        x y z
        rot_0 rot_1 rot_2 rot_3       (w, x, y, z)
        scale_0 scale_1 scale_2
        opacity
        f_dc_0 f_dc_1 f_dc_2          (degree‑0 SH RGB)
        f_rest_0 … f_rest_44          (remaining coeffs, if any)

    Any viewer that supports Gaussian splats (Polycam, Luma, Blender add‑on,
    CloudCompare plug‑in, etc.) will load this file without warnings.
    """
    names = arr.dtype.names
    n     = len(arr)

    # ------------------------------------------------------------------ #
    # 1) build a label map that matches Graph‑DECO                         #
    # ------------------------------------------------------------------ #
    sh_dim = arr['sh'].shape[1]  # number of SH coefficients per gaussian

    label_map = {
        'pw'   : ('x', 'y', 'z'),
        'rot'  : tuple(f'rot_{i}'   for i in range(4)),
        'scale': tuple(f'scale_{i}' for i in range(3)),
        'alpha': ('opacity',),
        'sh'   : (
            ('f_dc_0', 'f_dc_1', 'f_dc_2') +
            tuple(f'f_rest_{i}' for i in range(max(0, sh_dim - 3)))
        ),
    }

    # ------------------------------------------------------------------ #
    # 2) write header                                                    #
    # ------------------------------------------------------------------ #
    with open(fn, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")

        for name in names:
            labels = label_map.get(name, (name,))
            dt, _  = arr.dtype.fields[name]
            base   = _ply_type(dt.base if dt.shape else dt)

            for lab in labels:
                f.write(f"property {base} {lab}\n")

        f.write("end_header\n")

        # ------------------------------------------------------------------ #
        # 3) write body                                                     #
        # ------------------------------------------------------------------ #
        for g in arr:
            row = []
            for name in names:
                v = g[name]
                row.extend(v.tolist() if isinstance(v, np.ndarray) else [float(v)])
            f.write(" ".join(map(str, row)) + "\n")

# dtype generator remains unchanged
def gsdata_type(sh_dim:int):
    """Return a NumPy structured dtype for one Gaussian record."""
    return [('pw',   '<f4', (3,)),      # position (x,y,z)
            ('rot',  '<f4', (4,)),      # rotation quaternion (w,x,y,z)
            ('scale','<f4', (3,)),      # xyz scale (standard GSplat)
            ('alpha','<f4'),            # opacity before sigmoid
            ('sh',   '<f4', (sh_dim,))] # RGB spherical‑harmonic coeffs





def save_training_params(fn:str, params:dict):
    """
    Convert torch tensors in *params* to CPU numpy arrays and write either
    a *.npy* (default) or an ASCII *.ply* depending on the file extension.
    """
    # unpack & move to numpy -------------------------------------------------
    pws    = params["pws"].detach().cpu().numpy()
    shs    = torch.cat((params["low_shs"], params["high_shs"]), 1
                      ).detach().cpu().numpy()
    alphas = torch.sigmoid(params["alphas_raw"]).detach().cpu().numpy().squeeze()
    scales = torch.exp(params["scales_raw"]).detach().cpu().numpy()
    rots   = torch.nn.functional.normalize(params["rots_raw"]
                          ).detach().cpu().numpy()

    gaussians = np.rec.fromarrays(
        [pws, rots, scales, alphas, shs],
        dtype = gsdata_type(shs.shape[1])
    )

    ext = os.path.splitext(fn)[1].lower()
    if ext == '.ply':
        write_gaussians_as_ply(fn, gaussians)
    else:                             # fallback keeps your old workflow
        if ext != '.npy':
            print(f"[save_training_params] Unrecognised extension '{ext}', "
                  "falling back to .npy")
        np.save(fn, gaussians)
