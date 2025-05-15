import torch
import gsplatcu as gsc   
from borrowed import *
 


class GSFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        context,
        points_world,
        spherical_harmonics,
        opacity,
        scales,
        rots,
        uv,
        cam,
    ):
        # pw->camera frame->image
        uv, points_camera, depths, jacobian_uv_wrt_points = gsc.project(
            points_world, cam.Rcw, cam.tcw, cam.fx, cam.fy, cam.cx, cam.cy, True)

        # find 3d gaussian
        covariance_3d, jacobian_covar3d_wrt_rot, jacobian_covar3d_wrt_scale = gsc.computeCov3D(
            rots, scales, depths, True)

        # 2d gaussian
        covariance_2d, jacobian_covar2d_wrt_covar3d, jacobian_covar2d_wrt_poscam = gsc.computeCov2D(
            covariance_3d, points_camera, cam.Rcw, depths, cam.fx, cam.fy, cam.width, cam.height, True)

        # color
        colors, jacobian_color_wrt_sh, jacobian_color_wrt_ptsworld = gsc.sh2Color(spherical_harmonics, points_world, cam.twc, True)

        # 2d Gaussian -> image
        inv_covar2d, areas, jacobian_invcovar2d_wrt_covar2d = gsc.inverseCov2D(covariance_2d, depths, True)
        image, pixel_contribution, final_weights, tile_patch_range, patch_gaussian_ids =\
            gsc.splat(cam.height, cam.width,
                      uv, inv_covar2d, opacity, depths, colors, areas)

        
        context.cam = cam
        context.save_for_backward(uv, inv_covar2d, opacity,
                              depths, colors, pixel_contribution, final_weights,
                              tile_patch_range, patch_gaussian_ids,
                              jacobian_invcovar2d_wrt_covar2d, jacobian_covar2d_wrt_covar3d,
                              jacobian_covar3d_wrt_rot, jacobian_covar3d_wrt_scale, jacobian_color_wrt_sh,
                              jacobian_uv_wrt_points, jacobian_covar2d_wrt_poscam, jacobian_color_wrt_ptsworld)
        return image, depths > 0.2

    @staticmethod
    def backward(context, gradient_loss_wrt_image, _):
        # Retrieve the saved tensors and static parameters
        cam = context.cam
        uv, inv_covar2d, opacity, \
            depths, colors, pixel_contribution, final_weights,\
            tile_patch_range, patch_gaussian_ids,\
            jacobian_invcovar2d_wrt_covar2d, jacobian_covar2d_wrt_covar3d,\
            jacobian_covar3d_wrt_rot, jacobian_covar3d_wrt_scale, jacobian_color_wrt_sh,\
            jacobian_uv_wrt_points, jacobian_covar2d_wrt_poscam, jacobian_color_wrt_ptsworld = context.saved_tensors

        
        gradient_loss_wrt_uv, gradient_loss_wrt_inv_covar2d, dloss_opacity, gradient_loss_wrt_colors =\
            gsc.splatB(cam.height, cam.width, uv, inv_covar2d, opacity,
                       depths, colors, pixel_contribution, final_weights,
                       tile_patch_range, patch_gaussian_ids, gradient_loss_wrt_image)

        dpc_dpws = cam.Rcw
        gradient_loss_wrt_covar2d = gradient_loss_wrt_inv_covar2d @ jacobian_invcovar2d_wrt_covar2d
        gradient_loss_wrt_rot = gradient_loss_wrt_covar2d @ jacobian_covar2d_wrt_covar3d @ jacobian_covar3d_wrt_rot
        gradient_loss_wrt_scale = gradient_loss_wrt_covar2d @ jacobian_covar2d_wrt_covar3d @ jacobian_covar3d_wrt_scale
        gradient_loss_wrt_sh = (gradient_loss_wrt_colors.permute(0, 2, 1) @
                      jacobian_color_wrt_sh).permute(0, 2, 1).squeeze()

        gradient_loss_wrt_sh = gradient_loss_wrt_sh.reshape(gradient_loss_wrt_sh.shape[0], -1)
        gradient_loss_wrt_ptsworld = gradient_loss_wrt_uv @ jacobian_uv_wrt_points @ dpc_dpws + \
            gradient_loss_wrt_colors @ jacobian_color_wrt_ptsworld + \
            gradient_loss_wrt_covar2d @ jacobian_covar2d_wrt_poscam @ dpc_dpws

        return gradient_loss_wrt_ptsworld.squeeze(),\
            gradient_loss_wrt_sh,\
            dloss_opacity.squeeze().unsqueeze(1),\
            gradient_loss_wrt_scale.squeeze(),\
            gradient_loss_wrt_rot.squeeze(),\
            gradient_loss_wrt_uv.squeeze(),\
            None


def get_training_params(gs):
    points_world = torch.from_numpy(gs['pw']).type(
        torch.float32).to('cuda').requires_grad_()
    rots_raw = torch.from_numpy(gs['rot']).type(
        # unactivated scales
        torch.float32).to('cuda').requires_grad_()
    scales_raw = get_scales_raw(torch.from_numpy(gs['scale']).type(
        torch.float32).to('cuda')).requires_grad_()
    # unactivated opacity
    raw_opacity = get_opacity_raw(torch.from_numpy(gs['opacity'][:, np.newaxis]).type(
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


def update_params(optimizer, params, new_params):
    for group in optimizer.param_groups:
        param_new = new_params[group["name"]]
        state = optimizer.state.get(group['params'][0], None)
        if state is not None:
            state["exp_avg"] = torch.cat(
                (state["exp_avg"], torch.zeros_like(param_new)), dim=0)
            state["exp_avg_sq"] = torch.cat(
                (state["exp_avg_sq"], torch.zeros_like(param_new)), dim=0)
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter(
                torch.cat((group["params"][0], param_new), dim=0).requires_grad_(True))
            optimizer.state[group['params'][0]] = state
            params[group["name"]] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(
                torch.cat((group["params"][0], param_new), dim=0).requires_grad_(True))
            params[group["name"]] = group["params"][0]


def prune_params(optimizer, params, mask):
    for group in optimizer.param_groups:
        state = optimizer.state.get(group['params'][0], None)
        if state is not None:
            state["exp_avg"] = state["exp_avg"][mask]
            state["exp_avg_sq"] = state["exp_avg_sq"][mask]
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter(
                (group["params"][0][mask].requires_grad_(True)))
            optimizer.state[group['params'][0]] = state
            params[group["name"]] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(
                group["params"][0][mask].requires_grad_(True))
            params[group["name"]] = group["params"][0]


class GSModel(torch.nn.Module):
    def __init__(self, sense_size, max_steps):
        super().__init__()
        self.sample_count = None
        self.grad_accum = None
        self.cam = None
        self.grad_threshold = 4e-7
        self.scale_threshold = 0.01 * sense_size
        self.opacity_threshold = 0.005
        self.big_threshold = 0.1 * sense_size
        self.reset_opacity_val = 0.01
        self.iteration = 0
        self.pws_lr_scheduler = get_expon_lr_func(lr_init=1e-4 * sense_size,
                                                  lr_final=1e-6 * sense_size,
                                                  lr_delay_mult=0.01,
                                                  max_steps=max_steps)

    def forward(
            self,
            points_world,
            low_spherical_harmonics,
            high_spherical_harmonics,
            raw_opacity,
            scales_raw,
            rots_raw,
            cam):
        self.cam = cam
      

        self.uv = torch.zeros([points_world.shape[0], 2], dtype=torch.float32,
                              device='cuda', requires_grad=True)
        # Limit the value of opacity: 0 < opacity < 1
        opacity = get_opacity(raw_opacity)
        # Limit the value of scales > 0
        scales = get_scales(scales_raw)
        # Limit the value of rot, normal of rots is 1
        rots = get_rots(rots_raw)

        spherical_harmonics = get_spherical_harmonics(low_spherical_harmonics, high_spherical_harmonics)

        # apply GSfunction (forward)
        image, self.mask = GSFunction.apply(points_world, spherical_harmonics, opacity, scales, rots, self.uv, cam)

        return image

    def update_density_info(self):
        """
        calculate average grad of image points.
        # do it after backward
        """
        gradient_loss_wrt_uv = self.uv.grad
        with torch.no_grad():
            grad = torch.norm(gradient_loss_wrt_uv, dim=-1, keepdim=True)

            if self.sample_count is None:
                self.grad_accum = grad
                self.sample_count = self.mask.to(torch.int32)
            else:
                self.sample_count += self.mask
                self.grad_accum[self.mask] += grad[self.mask]
        del self.uv.grad
        del self.mask

    def update_gaussian_density(self, params, optimizer):
        # prune too small or too big gaussian
        selected_by_small_opacity = params["raw_opacity"].squeeze() < get_opacity_raw(self.opacity_threshold)
        selected_by_big_scale = torch.max(params["scales_raw"], axis=1)[0] > get_scales_raw(self.big_threshold)
        selected_for_prune = torch.logical_or(selected_by_small_opacity, selected_by_big_scale)
        selected_for_remain = torch.logical_not(selected_for_prune)
        prune_params(optimizer, params, selected_for_remain)

        grads = self.grad_accum.squeeze()[selected_for_remain] / self.sample_count[selected_for_remain]
        grads[grads.isnan()] = 0.0

        points_world = params["points_world"]
        low_spherical_harmonics = params["low_spherical_harmonics"]
        high_spherical_harmonics = params["high_spherical_harmonics"]
        opacity = get_opacity(params["raw_opacity"])
        scales = get_scales(params["scales_raw"])
        rots = get_rots(params["rots_raw"])

        selected_by_grad = grads >= self.grad_threshold
        selected_by_scale = torch.max(scales, axis=1)[0] <= self.scale_threshold

        selected_for_clone = torch.logical_and(selected_by_grad, selected_by_scale)
        selected_for_split = torch.logical_and(selected_by_grad, torch.logical_not(selected_by_scale))

        # clone gaussians
        pws_cloned = points_world[selected_for_clone]
        low_spherical_harmonics_cloned = low_spherical_harmonics[selected_for_clone]
        high_spherical_harmonics_cloned = high_spherical_harmonics[selected_for_clone]
        opacity_cloned = opacity[selected_for_clone]
        scales_cloned = scales[selected_for_clone]
        rots_cloned = rots[selected_for_clone]

        rots_splited = rots[selected_for_split]
        means = torch.zeros((rots_splited.size(0), 3), device="cuda")
        stds = scales[selected_for_split]
        samples = torch.normal(mean=means, std=stds)
        # sampling new pw for splited gaussian
        pws_splited = points_world[selected_for_split] + \
            rotate_vector_by_quaternion(rots_splited, samples)
        opacity_splited = opacity[selected_for_split]
        scales[selected_for_split] = scales[selected_for_split] * 0.6  # splited gaussian will go smaller
        scales_splited = scales[selected_for_split]
        low_spherical_harmonics_splited = low_spherical_harmonics[selected_for_split]
        high_spherical_harmonics_splited = high_spherical_harmonics[selected_for_split]

        new_params = {"points_world": torch.cat([pws_cloned, pws_splited]),
                      "low_spherical_harmonics": torch.cat([low_spherical_harmonics_cloned, low_spherical_harmonics_splited]),
                      "high_spherical_harmonics": torch.cat([high_spherical_harmonics_cloned, high_spherical_harmonics_splited]),
                      "raw_opacity": get_opacity_raw(torch.cat([opacity_cloned, opacity_splited])),
                      "scales_raw": get_scales_raw(torch.cat([scales_cloned, scales_splited])),
                      "rots_raw": torch.cat([rots_cloned, rots_splited])}


        update_params(optimizer, params, new_params)
        print("---------------------")
        print("gaussian density update report")
        prune_n = int(torch.sum(selected_for_prune))
        clone_n = int(torch.sum(selected_for_clone))
        split_n = int(torch.sum(selected_for_split))
        print("pruned num: ", prune_n)
        print("cloned num: ", clone_n)
        print("splited num: ", split_n)
        print("total gaussian number: ", params['points_world'].shape[0])
        print("---------------------")
        self.grad_accum = None
        self.sample_count = None

    def reset_opacity(self, params, optimizer):
        reset_raw_opacity_val = get_opacity_raw(self.reset_opacity_val)
        rest_mask = params['raw_opacity'] > reset_raw_opacity_val
        params['raw_opacity'][rest_mask] = torch.ones_like(
            params['raw_opacity'])[rest_mask] * reset_raw_opacity_val
        opacity_param = list(
            filter(lambda x: x["name"] == "raw_opacity", optimizer.param_groups))[0]
        state = optimizer.state.get(opacity_param['params'][0], None)
        state["exp_avg"] = torch.zeros_like(params['raw_opacity'])
        state["exp_avg_sq"] = torch.zeros_like(params['raw_opacity'])
        del optimizer.state[opacity_param['params'][0]]
        optimizer.state[opacity_param['params'][0]] = state

    def update_pws_lr(self, optimizer):
        pws_lr = self.pws_lr_scheduler(self.iteration)
        pws_param = list(
            filter(lambda x: x["name"] == "points_world", optimizer.param_groups))[0]
        pws_param['lr'] = pws_lr
        self.iteration += 1
