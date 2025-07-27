import torch
from tqdm import tqdm
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from utils.knn_utils import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.sss_model import SSS
from bvh import RayTracer
from utils.sh_utils import eval_sh
import torch.nn.functional as F


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.normal_activation = lambda x: torch.nn.functional.normalize(x, dim=-1, eps=1e-3)
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.base_color_activation = torch.sigmoid # FIXME: This might be really bad 
        self.roughness_activation = torch.sigmoid
        self.metallic_activation = torch.sigmoid
        self.subsurface_activation = torch.sigmoid


    def __init__(self):
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._normals = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self._sss = None
        self.sss_optimizer = None
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        # PBR 
        self._base_color = torch.empty(0)
        self._roughness = torch.empty(0)
        self._metallic = torch.empty(0)
        self.subsurfaceness = torch.empty(0)

        self._visibility_dc = torch.empty(0)
        self._visibility_rest = torch.empty(0)


    def capture(self):
        return (
            self._xyz,
            self._scaling,
            self._rotation,
            self._opacity,
            self._normals,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self._sss.state_dict(),
            self.vdgs_optimizer.state_dict(),
            self.spatial_lr_scale,
            self._base_color,
            self._roughness,
            self._metallic,
            self._subsurfaceness,
            self._visibility_dc,
            self._visibility_rest,
        )
    
    def restore(self, model_args, training_args):
        (self._xyz, 
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._normals,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self._base_color,
        self._roughness,
        self._metallic,
        self._subsurfaceness,
        self._visibility_dc,
        self._visibility_rest) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self._sss.load_state_dict(opt_dict)
        self.sss_optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_base_color(self):
        return self.base_color_activation(self._base_color)

    @property
    def get_roughness(self):
        return self.roughness_activation(self._roughness)

    @property
    def get_metallic(self):
        return self.metallic_activation(self._metallic)

    @property
    def get_subsurfaceness(self):
        return self.subsurface_activation(self._subsurfaceness)

    @property
    def get_normals(self):
        return self.normal_activation(self._normals)
    
    @property
    def get_visibility(self):
        """SH"""
        visibility_dc = self._visibility_dc
        visibility_rest = self._visibility_rest
        return torch.cat((visibility_dc, visibility_rest), dim=1)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def get_inverse_covariance(self, scaling_modifier=1):
        return self.covariance_activation(1 / self.get_scaling, 1 / scaling_modifier, self.get_rotation)

    def finetune_visibility(self, iterations=1000):
        visibility_sh_lr = 1e-2
        optimizer = torch.optim.Adam([
            {'params': [self._visibility_dc], 'lr': visibility_sh_lr},
            {'params': [self._visibility_rest], 'lr': visibility_sh_lr}
        ])
        means3D = self.get_xyz
        opacity = self.get_opacity[:, 0]
        scaling = self.get_scaling
        rotation = self.get_rotation
        normal = self.get_normals
        cov_inv = self.get_inverse_covariance()
        tbar = tqdm(range(iterations), desc="Finetuning visibility shs")
        raytracer = RayTracer(means3D, scaling, rotation)
        visibility_shs_view = self.get_visibility.transpose(1, 2)
        vis_sh_degree = np.sqrt(visibility_shs_view.shape[-1]) - 1
        rays_o = means3D
        for iteration in tbar:
            rays_d = torch.randn_like(rays_o)
            mask = (rays_d * normal).sum(-1) < 0
            rays_d[mask] *= -1
            sample_sh2vis = eval_sh(vis_sh_degree, visibility_shs_view, rays_d)
            sample_vis = torch.clamp(sample_sh2vis + 0.5, 0.0, 1.0)
            trace_results = raytracer.trace_visibility(
                rays_o,
                rays_d,
                means3D,
                cov_inv,
                opacity,
                normal)
            visibility = trace_results["visibility"]
            loss = F.l1_loss(visibility, sample_vis)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        normals = torch.randn((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda") 
        normals = torch.nn.functional.normalize(normals, p=2, dim=1)

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._normals = nn.Parameter(normals.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        base_color = torch.zeros((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")
        roughness = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        metallic = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        subsurfaceness = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")

        self._base_color = nn.Parameter(base_color.requires_grad_(True))
        self._roughness = nn.Parameter(roughness.requires_grad_(True))
        self._metallic = nn.Parameter(metallic.requires_grad_(True))
        self._subsurfaceness = nn.Parameter(subsurfaceness.requires_grad_(True))

        visibility = torch.zeros((self._xyz.shape[0], 1, 4 ** 2)).float().cuda()
        self._visibility_dc = nn.Parameter(visibility[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._visibility_rest = nn.Parameter(visibility[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        if training_args.visibility_rest_lr < 0:
            training_args.visibility_rest_lr = training_args.visibility_lr / 20.0
            
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._normals], 'lr': training_args.normals_lr, "name": "normals"},
            {'params': [self._base_color], 'lr': training_args.base_color_lr, "name": "base_color"},
            {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
            {'params': [self._metallic], 'lr': training_args.metallic_lr, "name": "metallic"},
            {'params': [self._subsurfaceness], 'lr': training_args.subsurfaceness_lr, "name": "subsurfaceness"},
            {'params': [self._visibility_dc], 'lr': training_args.visibility_lr, "name": "visibility_dc"},
            {'params': [self._visibility_rest], 'lr': training_args.visibility_rest_lr, "name": "visibility_rest"},
        ]
        
        # SSS MLP
        self._sss = SSS(net_width=training_args.sss_width).to("cuda")
        self.sss_optimizer = torch.optim.Adam(self._sss.parameters(), lr=training_args.sss_lr)
        self.sss_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.sss_optimizer, gamma=0.9999)
        print(self._sss)
        
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        # Print info about the VDGS network the layers etc. 

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr

                if iteration > 1 and iteration <= 30_000: 
                    self.sss_scheduler.step()

                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        
        for i in range(self._base_color.shape[1]):
            l.append('base_color_{}'.format(i))
        l.append('roughness')
        l.append('metallic')
        l.append('subsurfaceness')
        for i in range(self._visibility_dc.shape[1] * self._visibility_dc.shape[2]):
                l.append('visibility_dc_{}'.format(i))
        for i in range(self._visibility_rest.shape[1] * self._visibility_rest.shape[2]):
            l.append('visibility_rest_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals =  self._normals.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        base_color = self._base_color.detach().cpu().numpy()
        roughness = self._roughness.detach().cpu().numpy()
        metallic = self._metallic.detach().cpu().numpy()
        subsurfaceness = self._subsurfaceness.detach().cpu().numpy()

        visibility_dc = self._visibility_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        visibility_rest = self._visibility_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, opacities, scale, rotation, base_color, roughness, metallic, subsurfaceness, visibility_dc, visibility_rest), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_roughness(self):
        roughness_new = torch.ones_like(self.get_roughness) * 0.5
        optimizable_tensors = self.replace_tensor_to_optimizer(roughness_new, "roughness")
        self._roughness = optimizable_tensors["roughness"]

    def reset_metallic(self):
        metallic_new = torch.zeros_like(self.get_metallic)
        optimizable_tensors = self.replace_tensor_to_optimizer(metallic_new, "metallic")
        self._metallic = optimizable_tensors["metallic"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        normals = np.stack((np.asarray(plydata.elements[0]["nx"]), np.asarray(plydata.elements[0]["ny"]), np.asarray(plydata.elements[0]["nz"])), axis=1)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._normals = nn.Parameter(torch.tensor(normals, dtype=torch.float, device="cuda").requires_grad_(True))

        base_color_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("base_color")]
        base_color_names = sorted(base_color_names, key=lambda x: int(x.split('_')[-1]))
        base_color = np.zeros((xyz.shape[0], len(base_color_names)))
        for idx, attr_name in enumerate(base_color_names):
            base_color[:, idx] = np.asarray(plydata.elements[0][attr_name])

        roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]
        metallic = np.asarray(plydata.elements[0]["metallic"])[..., np.newaxis]
        subsurfaceness = np.asarray(plydata.elements[0]["subsurfaceness"])[..., np.newaxis]

        self._base_color = nn.Parameter(torch.tensor(base_color, dtype=torch.float, device="cuda").requires_grad_(True))
        self._roughness = nn.Parameter(torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))
        self._metallic = nn.Parameter(torch.tensor(metallic, dtype=torch.float, device="cuda").requires_grad_(True))
        self._subsurfaceness = nn.Parameter(torch.tensor(subsurfaceness, dtype=torch.float, device="cuda").requires_grad_(True))

        visibility_dc = np.zeros((xyz.shape[0], 1, 1))
        visibility_dc[:, 0, 0] = np.asarray(plydata.elements[0]["visibility_dc_0"])
        extra_visibility_names = [p.name for p in plydata.elements[0].properties if  p.name.startswith("visibility_rest_")]
        extra_visibility_names = sorted(extra_visibility_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_visibility_names) == 4 ** 2 - 1
        visibility_extra = np.zeros((xyz.shape[0], len(extra_visibility_names)))
        for idx, attr_name in enumerate(extra_visibility_names):
            visibility_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        visibility_extra = visibility_extra.reshape((visibility_extra.shape[0], 1, 4 ** 2 - 1))
        self._visibility_dc = nn.Parameter(torch.tensor(visibility_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._visibility_rest = nn.Parameter(torch.tensor(visibility_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._normals = optimizable_tensors["normals"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        self._base_color = optimizable_tensors["base_color"]
        self._roughness = optimizable_tensors["roughness"]
        self._metallic = optimizable_tensors["metallic"]
        self._subsurfaceness = optimizable_tensors["subsurfaceness"]
        self._visibility_dc = optimizable_tensors["visibility_dc"]
        self._visibility_rest = optimizable_tensors["visibility_rest"]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_opacities, new_scaling, new_rotation, new_normals, new_base_color, new_roughness, new_metallic, new_subsurfaceness, new_visibility_dc, new_visibility_rest):
        d = {
            "xyz": new_xyz,
            "opacity": new_opacities,
            "scaling" : new_scaling,
            "rotation" : new_rotation,
            "normals" : new_normals,
            "base_color": new_base_color,
            "roughness": new_roughness,
            "metallic": new_metallic,
            "subsurfaceness": new_subsurfaceness,
            "visibility_dc": new_visibility_dc,
            "visibility_rest": new_visibility_rest
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._normals = optimizable_tensors["normals"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self._base_color = optimizable_tensors["base_color"]
        self._roughness = optimizable_tensors["roughness"]
        self._metallic = optimizable_tensors["metallic"]
        self._subsurfaceness = optimizable_tensors["subsurfaceness"]

        self._visibility_dc = optimizable_tensors["visibility_dc"]
        self._visibility_rest = optimizable_tensors["visibility_rest"]

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_normals = self._normals[selected_pts_mask].repeat(N,1)

        new_base_color = self._base_color[selected_pts_mask].repeat(N, 1)
        new_roughness = self._roughness[selected_pts_mask].repeat(N, 1)
        new_metallic = self._metallic[selected_pts_mask].repeat(N, 1)
        new_subsurfaceness = self._subsurfaceness[selected_pts_mask].repeat(N, 1)

        new_visibility_dc = self._visibility_dc[selected_pts_mask].repeat(N, 1, 1)
        new_visibility_rest = self._visibility_rest[selected_pts_mask].repeat(N, 1, 1)

        self.densification_postfix(new_xyz, new_opacity, new_scaling, new_rotation, new_normals, new_base_color, new_roughness, new_metallic, new_subsurfaceness, new_visibility_dc, new_visibility_rest)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_normals = self._normals[selected_pts_mask]

        new_base_color = self._base_color[selected_pts_mask]
        new_roughness = self._roughness[selected_pts_mask]
        new_metallic = self._metallic[selected_pts_mask]
        new_subsurfaceness = self._subsurfaceness[selected_pts_mask]

        new_visibility_dc = self._visibility_dc[selected_pts_mask]
        new_visibility_rest = self._visibility_rest[selected_pts_mask]

        self.densification_postfix(new_xyz, new_opacities, new_scaling, new_rotation, new_normals, new_base_color, new_roughness, new_metallic, new_subsurfaceness, new_visibility_dc, new_visibility_rest)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1