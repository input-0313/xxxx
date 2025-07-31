#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation


class GaussianModel:
    def __init__(self, sh_degree: int):

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)

        # Initialize an empty tensor for normals; will be properly set later.
        self._normals = torch.empty(0)

        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

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
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
      
    @property
    def get_normals(self):
        return self._normals

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, normals=None):
        self.spatial_lr_scale = 5
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # Initialize normals as zeros, then estimate initial normals based on scaling and rotation.
        self._normals = nn.Parameter(torch.zeros_like(fused_point_cloud).requires_grad_(True))
        # Immediately estimate initial normals using the shortest axis method at initialization.
        self.update_normals(iteration=0)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.spatial_lr_scale = 5

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr * self.spatial_lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        # Add normals parameter to the optimizer if available
        if len(self._normals) > 0:
            l.append({'params': [self._normals], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "normals"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()

        # Use the computed normals when saving to PLY
        normals = self._normals.detach().cpu().numpy()

        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, og_number_points=-1):
        self.og_number_points = og_number_points
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # Initialize normals as zeros
        normals = np.zeros_like(xyz)
        # If the PLY file contains normal attributes, load them
        if "nx" in plydata.elements[0].properties:
            normals[:, 0] = np.asarray(plydata.elements[0]["nx"])
            normals[:, 1] = np.asarray(plydata.elements[0]["ny"])
            normals[:, 2] = np.asarray(plydata.elements[0]["nz"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        # Load normals as a trainable parameter (either loaded from file or initialized as zeros)
        self._normals = nn.Parameter(torch.tensor(normals, dtype=torch.float, device="cuda").requires_grad_(True))
        
        self.active_sh_degree = self.max_sh_degree

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
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # Update normals parameter after pruning.
        # If normals are managed by the optimizer, retrieve the pruned tensor from optimizer.
        if "normals" in optimizable_tensors:
            self._normals = optimizable_tensors["normals"]
        else:
            # If not, manually prune the normals tensor.
            self._normals = nn.Parameter(self._normals[valid_points_mask].clone().detach().requires_grad_(True))

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_normals=None):
        # If no normals are provided for new points, initialize them as unit vectors along the z-axis.
        if new_normals is None:
            new_normals = torch.zeros_like(new_xyz)
            new_normals[:, 2] = 1.0  # Default normal direction is along z-axis
        
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}
        
        # Add normals if available
        if len(self._normals) > 0:
            d["normals"] = new_normals

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        if "normals" in optimizable_tensors:
            self._normals = optimizable_tensors["normals"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        # Inherit normals: for each new split Gaussian, copy the normal from the original selected Gaussian.
        new_normals = self._normals[selected_pts_mask].repeat(N, 1) if len(self._normals) > 0 else None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_normals)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        # Inherit normals: for each new clone Gaussian, copy the normal from the original selected Gaussian.
        new_normals = self._normals[selected_pts_mask] if len(self._normals) > 0 else None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_normals)

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


    def densify_and_prune_with_curvature(self, max_grad, min_opacity, extent, max_screen_size, 
                                        curvature_prune_ratio=0.02, grad_threshold_low=0.00005, 
                                        max_candidates=10000, low_curvature_threshold=0.5):
        """
        1. Remove low-gradient + low-curvature redundant points (reduce point cloud size)
        2. Densify on the smaller point cloud (lower cost)
        3. Traditional pruning (final cleanup)
        Args:
            max_grad: Gradient threshold for densification
            min_opacity: Opacity threshold for pruning
            extent: Scene extent
            max_screen_size: Screen size threshold
            curvature_prune_ratio: Max ratio of points to prune by curvature
            grad_threshold_low: Low gradient threshold for candidate selection
            max_candidates: Max number of candidates (avoid OOM)
            low_curvature_threshold: Curvature threshold for pruning
        """
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        grad_norms = torch.norm(grads, dim=-1)
    
        # Step 1: remove redundant points
        # Step 1.1: Optimization activity assessment
        low_grad_mask = grad_norms < grad_threshold_low
        low_grad_count = low_grad_mask.sum().item()

        # Limit candidate number to avoid OOM                                  
        if low_grad_count > max_candidates:
            # Get indices and values of all low-gradient points
            low_grad_indices = torch.where(low_grad_mask)[0]
            low_grad_values = grad_norms[low_grad_indices]
            
            # Sort by gradient, select the lowest max_candidates points
            _, sorted_indices = torch.sort(low_grad_values)
            selected_indices = low_grad_indices[sorted_indices[:max_candidates]]
            
            # Rebuild candidate mask
            low_grad_mask = torch.zeros_like(low_grad_mask)
            low_grad_mask[selected_indices] = True
            low_grad_count = max_candidates
        
        # Step 1.2: Geometric complexity assessment - Only compute curvature for candidate points
        curvatures = torch.zeros(len(self._xyz), device=self._xyz.device)
        if low_grad_count > 0:
            low_grad_curvatures = self.compute_curvatures_for_subset(low_grad_mask, k_neighbors=10)
            curvatures[low_grad_mask] = low_grad_curvatures
        
        if len(curvatures) == 0 or low_grad_count == 0:
            # No candidates, skip curvature pruning
            pass
        else:
            # Step 1.3: Identify truly redundant points (low gradient & low curvature)
            candidate_curvatures = curvatures[low_grad_mask]
            if candidate_curvatures.max() > candidate_curvatures.min():
                candidate_curvatures_norm = (candidate_curvatures - candidate_curvatures.min()) / (candidate_curvatures.max() - candidate_curvatures.min())
            else:
                candidate_curvatures_norm = candidate_curvatures
    
            # Find points with normalized curvature below threshold
            low_curvature_mask = candidate_curvatures_norm < low_curvature_threshold
            high_curvature_mask = candidate_curvatures_norm >= low_curvature_threshold
    
            # Get indices of redundant points (low gradient & low curvature)
            candidate_indices = torch.where(low_grad_mask)[0]
            low_curvature_indices = candidate_indices[low_curvature_mask]
            
            redundant_mask = torch.zeros(len(self._xyz), dtype=torch.bool, device=self._xyz.device)
            redundant_mask[low_curvature_indices] = True
    
            redundant_count = redundant_mask.sum().item()
    
            # Step 1.4: Prune the most redundant points, limit by ratio
            if redundant_count > 0:
                num_to_prune = min(redundant_count, int(len(self._xyz) * curvature_prune_ratio))
                
                if num_to_prune > 0:
                    redundant_indices = torch.where(redundant_mask)[0]
                    redundant_curvatures = curvatures[redundant_indices]
                    
                    # Prune the points with the lowest curvature
                    _, sorted_indices = torch.sort(redundant_curvatures)
                    prune_indices = redundant_indices[sorted_indices[:num_to_prune]]
                    
                    curvature_prune_mask = torch.zeros(len(self._xyz), dtype=torch.bool, device=self._xyz.device)
                    curvature_prune_mask[prune_indices] = True
                    
                    before_curvature_prune = self.get_xyz.shape[0]
                    self.prune_points(curvature_prune_mask)
        
        # Step 2: Densification on the cleaned point cloud
        current_grads = self.xyz_gradient_accum / self.denom
        current_grads[current_grads.isnan()] = 0.0
      
        self.densify_and_clone(current_grads, max_grad, extent)
        self.densify_and_split(current_grads, max_grad, extent)
    
        # Step 3: Final cleanup - remove low-quality points
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
    
        if prune_mask.sum().item() > 0:
            self.prune_points(prune_mask)
            
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1

    def update_normals(self, iteration, d_xyz=0.0, d_rotation=0.0, d_scaling=0.0):
        """
        Update the normal vectors for all Gaussians.
        In the static reconstruction stage (iteration <= 3000), normals are initialized using the shortest axis of the ellipsoid
        In the dynamic deformation stage, normals are updated using a coordinate system transformation to maintain consistency with local geometry

        Args:
            iteration: Current training iteration
            d_xyz: Deformation offset for positions (used in dynamic stage)
            d_rotation: Deformation offset for rotations (used in dynamic stage)
            d_scaling: Deformation offset for scaling (used in dynamic stage)

        Notes:
            The update is performed every 30 iterations to reduce computational overhead.
            Normals are always normalized to unit vectors.
            In the dynamic stage, the update is performed in chunks to save memory for large point clouds.
        """
        # Reduce update frequency for efficiency
        if iteration % 30 != 0:
            return
        
        if len(self._normals) == 0:
            self._normals = nn.Parameter(torch.zeros_like(self._xyz).requires_grad_(True))
        
        # Static stage: estimate normals by shortest axis
        if iteration <= 3000:
            scales = self.get_scaling
            rotations = self.get_rotation
            
            rot_matrices = build_rotation(rotations)
            min_axis_indices = torch.argmin(scales, dim=1).unsqueeze(1)
            
            unit_vectors = torch.zeros_like(scales, device=scales.device)
            unit_vectors.scatter_(1, min_axis_indices, 1.0)
            
            normals = torch.bmm(rot_matrices, unit_vectors.unsqueeze(-1)).squeeze(-1)
            normals = torch.nn.functional.normalize(normals, dim=1)
            
            self._normals.data = normals
            
        # Dynamic stage: update normals using coordinate transformation
        elif torch.is_tensor(d_xyz) and torch.any(d_xyz != 0):
            original_scaling = self.get_scaling
            original_rotation = self.get_rotation
            batch_size = original_scaling.shape[0]
            device = original_scaling.device
            
            deformed_scaling = original_scaling + d_scaling if torch.is_tensor(d_scaling) else original_scaling
            deformed_rotation = self.rotation_activation(original_rotation + d_rotation) if torch.is_tensor(d_rotation) else original_rotation
            
            original_rot_matrices = build_rotation(original_rotation)
            deformed_rot_matrices = build_rotation(deformed_rotation)
            
            # Process in chunks to save memory
            chunk_size = min(100000, batch_size)
            chunks = (batch_size + chunk_size - 1) // chunk_size
            
            transformed_normals = torch.zeros_like(self._normals)
            
            for i in range(chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, batch_size)
                chunk_indices = slice(start_idx, end_idx)
                
                # Find shortest and longest axes before and after deformation
                min_axes_orig = torch.argmin(original_scaling[chunk_indices], dim=1)
                max_axes_orig = torch.argmax(original_scaling[chunk_indices], dim=1)
                
                min_axes_def = torch.argmin(deformed_scaling[chunk_indices], dim=1)
                max_axes_def = torch.argmax(deformed_scaling[chunk_indices], dim=1)
                
                batch_indices = torch.arange(end_idx-start_idx, device=device).unsqueeze(1).repeat(1, 3)
                
                # Gather axis vectors
                min_indices_orig = min_axes_orig.unsqueeze(1).unsqueeze(2).expand(-1, 3, 1)
                max_indices_orig = max_axes_orig.unsqueeze(1).unsqueeze(2).expand(-1, 3, 1)
                
                min_indices_def = min_axes_def.unsqueeze(1).unsqueeze(2).expand(-1, 3, 1)
                max_indices_def = max_axes_def.unsqueeze(1).unsqueeze(2).expand(-1, 3, 1)
                
                v_s_orig = torch.gather(original_rot_matrices[chunk_indices], 2, min_indices_orig).squeeze(2)
                v_l_orig = torch.gather(original_rot_matrices[chunk_indices], 2, max_indices_orig).squeeze(2)
                
                v_s_def = torch.gather(deformed_rot_matrices[chunk_indices], 2, min_indices_def).squeeze(2)
                v_l_def = torch.gather(deformed_rot_matrices[chunk_indices], 2, max_indices_def).squeeze(2)
                
                # Compute third axis by cross product
                v_t_orig = torch.linalg.cross(v_s_orig, v_l_orig)
                v_t_def = torch.linalg.cross(v_s_def, v_l_def)
                
                # Normalize all axes
                v_s_orig = torch.nn.functional.normalize(v_s_orig, dim=1)
                v_l_orig = torch.nn.functional.normalize(v_l_orig, dim=1)
                v_t_orig = torch.nn.functional.normalize(v_t_orig, dim=1)
                
                v_s_def = torch.nn.functional.normalize(v_s_def, dim=1)
                v_l_def = torch.nn.functional.normalize(v_l_def, dim=1)
                v_t_def = torch.nn.functional.normalize(v_t_def, dim=1)
                
                # Build orthogonal bases before and after deformation
                U = torch.stack([v_s_orig, v_l_orig, v_t_orig], dim=2)
                V = torch.stack([v_s_def, v_l_def, v_t_def], dim=2)
                
                # Compute rotation matrix between bases
                R = torch.bmm(V, U.transpose(1, 2))
                
                # Apply rotation to normals
                chunk_normals = torch.bmm(R, self._normals[chunk_indices].unsqueeze(-1)).squeeze(-1)
                
                # Ensure normals point outward
                dot_product = torch.sum(chunk_normals * v_s_def, dim=1, keepdim=True)
                chunk_normals = torch.where(dot_product < 0, -chunk_normals, chunk_normals)
                
                # Normalize normals
                chunk_normals = torch.nn.functional.normalize(chunk_normals, dim=1)

                transformed_normals[chunk_indices] = chunk_normals
          
            self._normals.data = transformed_normals


    def compute_curvatures_for_subset(self, subset_mask, k_neighbors=10):
        '''
        Compute curvature for a subset of points (low-gradient points)
        '''
        if len(self._xyz) == 0 or subset_mask.sum() == 0:
            return torch.tensor([])
        
        # Get all points and normals
        all_points = self.get_xyz
        all_normals = self.get_normals
        
        # Select candidate points and their normals
        candidate_points = all_points[subset_mask]  # [M, 3]
        candidate_normals = all_normals[subset_mask]  # [M, 3]
        n_candidates = len(candidate_points)
        
        if n_candidates == 0:
            return torch.tensor([])
        
        if n_candidates < k_neighbors:
            k_neighbors = max(1, n_candidates - 1)
        
        if k_neighbors <= 0:
            return torch.zeros(n_candidates, device=candidate_points.device)
        
        # Compute pairwise distances between candidates and all points
        dists = torch.cdist(candidate_points, all_points)  # [M, N]
        
        # For each candidate, find k nearest neighbors in the whole point cloud
        _, nn_indices = torch.topk(-dists, k_neighbors, dim=1)  # [M, k]
        
        # Compute curvature based on normal differences
        curvatures = torch.zeros(n_candidates, device=candidate_points.device)
        for i in range(n_candidates):
            center_normal = candidate_normals[i:i+1]  # [1, 3]
            neighbor_normals = all_normals[nn_indices[i]]  # [k, 3]
            
            # θ is the angle between center and neighbor normals
            normal_diff = 1.0 - torch.abs(torch.sum(center_normal * neighbor_normals, dim=1))
            curvatures[i] = torch.mean(normal_diff)
        
        return curvatures
    

    def get_storage_info(self):
        """
        Compute and return storage and memory usage statistics for the Gaussian model.
        All memory statistics are computed in bytes and converted to megabytes (MB) where appropriate.
        Auxiliary tensors include gradient accumulators and other optional attributes.
        """
        storage_info = {}
        
        num_points = self.get_xyz.shape[0]
        storage_info['num_points'] = num_points
        
        def tensor_memory(tensor):
            if tensor.numel() == 0:
                return 0
            return tensor.numel() * tensor.element_size()
        
        xyz_memory = tensor_memory(self._xyz)
        storage_info['xyz_memory_mb'] = xyz_memory / (1024 * 1024)
        
        features_dc_memory = tensor_memory(self._features_dc)
        features_rest_memory = tensor_memory(self._features_rest)
        features_total_memory = features_dc_memory + features_rest_memory
        storage_info['features_memory_mb'] = features_total_memory / (1024 * 1024)
        
        scaling_memory = tensor_memory(self._scaling)
        storage_info['scaling_memory_mb'] = scaling_memory / (1024 * 1024)
        
        rotation_memory = tensor_memory(self._rotation)
        storage_info['rotation_memory_mb'] = rotation_memory / (1024 * 1024)
        
        opacity_memory = tensor_memory(self._opacity)
        storage_info['opacity_memory_mb'] = opacity_memory / (1024 * 1024)
        
        normals_memory = tensor_memory(self._normals)
        storage_info['normals_memory_mb'] = normals_memory / (1024 * 1024)
        
        aux_memory = 0
        if hasattr(self, 'xyz_gradient_accum') and self.xyz_gradient_accum.numel() > 0:
            gradient_memory = tensor_memory(self.xyz_gradient_accum)
            aux_memory += gradient_memory
        
        if hasattr(self, 'denom') and self.denom.numel() > 0:
            denom_memory = tensor_memory(self.denom)
            aux_memory += denom_memory
        
        if hasattr(self, 'max_radii2D') and self.max_radii2D.numel() > 0:
            radii_memory = tensor_memory(self.max_radii2D)
            aux_memory += radii_memory
        
        storage_info['auxiliary_memory_mb'] = aux_memory / (1024 * 1024)
        
        total_memory = (xyz_memory + features_total_memory + scaling_memory + 
                       rotation_memory + opacity_memory + normals_memory + aux_memory)
  
        storage_info['total_memory_mb'] = total_memory / (1024 * 1024)
        
        if num_points > 0:
            storage_info['memory_per_point_bytes'] = total_memory / num_points
        else:
            storage_info['memory_per_point_bytes'] = 0
        
        return storage_info
