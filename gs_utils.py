
import torch
import os
import numpy as np
from math import exp
import torch.nn.functional as F
import torchvision
from pathlib import Path
import faiss
from read_write_model import *
from torch.autograd import Variable
from PIL import Image



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
    points_world = torch.from_numpy(gs['pw']).type(
        torch.float32).to('cuda').requires_grad_()
    rots_raw = torch.from_numpy(gs['rot']).type(

        torch.float32).to('cuda').requires_grad_()
    scales_raw = get_scales_raw(torch.from_numpy(gs['scale']).type(
        torch.float32).to('cuda')).requires_grad_()

    raw_opacity = get_opacity_raw(torch.from_numpy(gs['alpha'][:, np.newaxis]).type(
        torch.float32).to('cuda')).requires_grad_()
    spherical_harmonics = torch.from_numpy(gs['sh']).type(
        torch.float32).to('cuda')
    low_spherical_harmonics = spherical_harmonics[:, :3]
    high_spherical_harmonics = torch.ones_like(low_spherical_harmonics).repeat(1, 15) * 0.001
    high_spherical_harmonics[:, :spherical_harmonics[:, 3:].shape[1]] = spherical_harmonics[:, 3:]
    low_spherical_harmonics = low_spherical_harmonics.requires_grad_()
    high_spherical_harmonics = high_spherical_harmonics.requires_grad_()
    params = {"points_world": points_world, "low_spherical_harmonics": low_spherical_harmonics, "high_spherical_harmonics": high_spherical_harmonics,
                "raw_opacity": raw_opacity, "scales_raw": scales_raw, "rots_raw": rots_raw}

    adam_params = [
        {'params': [params['points_world']], 'lr': 0.001, "name": "points_world"},
        {'params': [params['low_spherical_harmonics']],
            'lr': 0.001, "name": "low_spherical_harmonics"},
        {'params': [params['high_spherical_harmonics']],
            'lr': 0.001/20, "name": "high_spherical_harmonics"},
        {'params': [params['raw_opacity']],
            'lr': 0.05, "name": "raw_opacity"},
        {'params': [params['scales_raw']],
            'lr': 0.005, "name": "scales_raw"},
        {'params': [params['rots_raw']], 'lr': 0.001, "name": "rots_raw"}]

    return params, adam_params


def get_scales_raw(x):
    if isinstance(x, float):
        return np.log(x)
    else:
        return torch.log(x)


def read_points_bin_as_gau(path_to_model_file):
    """
    read colmap points file as inital gaussians
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        points_world = np.zeros([num_points, 3])
        spherical_harmonics = np.zeros([num_points, 3])
        for i in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            points_world[i] = np.array(binary_point_line_properties[1:4])
            points_world[i] = (np.array(binary_point_line_properties[4:7]) /
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
        opacity = np.ones([num_points]) * 0.8
        points_world = points_world.astype(np.float32)
        rots = rots.astype(np.float32)
        opacity = opacity.astype(np.float32)
        spherical_harmonics = spherical_harmonics.astype(np.float32)

        N, D = points_world.shape
        index = faiss.IndexFlatL2(D)
        index.add(points_world)
        distances, indices = index.search(points_world, 2)
        distances = np.clip(distances[:, 1], 0.01, 3)
        scales = distances[:, np.newaxis].repeat(3, 1)

        dtypes = [('pw', '<f4', (3,)),
                  ('rot', '<f4', (4,)),
                  ('scale', '<f4', (3,)),
                  ('opacity', '<f4'),
                  ('sh', '<f4', (3,))]

        gs = np.rec.fromarrays(
            [points_world, rots, scales, opacity, spherical_harmonics], dtype=dtypes)

        return gs
    

def get_cameras_and_images(path):
    device = 'cuda'

    camera_params = read_cameras_binary(os.path.join(Path(path, "sparse/0"), "cameras.bin"))
    image_params = read_images_binary(os.path.join(Path(path, "sparse/0"), "images.bin"))
    cameras = []
    images = []
    for image_param in image_params.values():
        i = image_param.camera_id
        camera_param = camera_params[i]
        im_path = str(Path(path, "images", image_param.name))
        image = Image.open(im_path)

        w_scale = image.width/camera_param.width
        h_scale = image.height/camera_param.height
        fx = camera_param.params[0] * w_scale
        fy = camera_param.params[1] * h_scale
        cx = camera_param.params[2] * w_scale
        cy = camera_param.params[3] * h_scale
        Rcw = torch.from_numpy(image_param.qvec2rotmat()).to(device).to(torch.float32)
        tcw = torch.from_numpy(image_param.tvec).to(device).to(torch.float32)
        camera = Camera(image_param.id, image.width, image.height, fx, fy, cx, cy, Rcw, tcw, im_path)
        image = torchvision.transforms.functional.to_tensor(image).to(device).to(torch.float32)

        cameras.append(camera)
        images.append(image)

    return cameras, images

def get_opacity_raw(x):
    """
    inverse of sigmoid
    """
    if isinstance(x, float):
        return np.log(x/(1-x))
    else:
        return torch.log(x/(1-x))
    

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

    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):

            return 0.0
        if lr_delay_steps > 0:

            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper