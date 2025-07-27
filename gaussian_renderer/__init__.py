import torch
import math
import numpy as np
from gs_sss_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from bvh import RayTracer
from utils.graphics_utils import fov2focal
from torch.nn import functional as F


def render_equation(base_color, metalness, roughness, subsurfaceness, normal, residual, light_direction, view_direction, incident_light, debug=False, exr=False):
    """
    Render equation for the PBR shading model in pytorch

    Args:
        base_color: (B, 3)
        metalness: (B, 1)
        roughness: (B, 1)
        subsurfaceness: (B, 1)
        normal: (B, 3)
        residual: (B, 3)
        light_direction: (B, 3) 
        view_direction: (B, 3)
        incident_light: (B, 3) 
    """

    # Clamp roughness 
    roughness = torch.clamp(roughness, 0.001, 0.99)

    # Normalize normal
    normal = normal / (normal.norm(dim=-1, keepdim=True) + 1e-5)

    # Compute halfway vector and dot products
    half_vector = F.normalize(light_direction + view_direction, dim=-1)
    dot_nl = torch.clamp(torch.sum(normal * light_direction, dim=-1, keepdim=True), 0.001, 1.0)
    dot_nv = torch.clamp(torch.abs(torch.sum(normal * view_direction, dim=-1, keepdim=True)), 0.001, 1.0) 
    dot_nh = torch.clamp(torch.sum(normal * half_vector, dim=-1, keepdim=True), 0.0, 1.0)
    dot_vh = torch.clamp(torch.sum(view_direction * half_vector, dim=-1, keepdim=True), 0.0, 1.0)

    # Fresnel equation using Schlick's approximation
    def fresnel_schlick(dot_vh, f0):
        exponent = (-5.55473 * dot_vh -6.98316) * dot_vh
        result = f0 + (1.0 - f0) * pow(2.0, exponent)
        return result

    # Calculate NDF (Normal Distribution Function) using GGX/Trowbridge-Reitz
    def ndf_ggx(roughness, dot_nh):
        a = roughness * roughness
        a2 = a * a
        dot_nh2 = dot_nh * dot_nh
        denom = (dot_nh2 * (a2 - 1.0) + 1.0)
        return a2 / (torch.pi * denom * denom + 1e-5)

    # Geometry function using Smith's method with Schlick-GGX
    def geometry_smith(roughness, dot_nv, dot_nl):
        r = roughness + 1.0
        k = r * r / 8.0
        g1 = dot_nv / (dot_nv * (1.0 - k) + k + 1e-5)
        g2 = dot_nl / (dot_nl * (1.0 - k) + k + 1e-5)
        return g1 * g2

    # Fresnel at normal incidence
    dialectric_spec = torch.ones_like(base_color) * 0.04
    c_diff = torch.lerp(torch.zeros_like(base_color), base_color * (1 - 0.04), 1.0 - metalness)
    f0 = torch.lerp(dialectric_spec, base_color, metalness)

    # Specular term
    Fr = fresnel_schlick(dot_vh, f0)
    D = ndf_ggx(roughness, dot_nh)
    G = geometry_smith(roughness, dot_nv, dot_nl)
    specular = (D * Fr * G) / (4.0 * dot_nv * dot_nl + 1e-5)

    # Diffuse term (Lambertian reflection)
    diffuse = (1.0 - Fr) * c_diff / torch.pi 

    # PBR equation
    pbr = dot_nl * incident_light * (diffuse + specular)

    # Combine the components using the subsurface weight as blending factor
    # Set subsurfacenes to 0 
    # subsurfaceness = torch.zeros_like(subsurfaceness)
    pbr_combined = (1.0 - subsurfaceness) * pbr + subsurfaceness * residual

    if not exr:
        pbr_combined = torch.clamp(pbr_combined, 0.0, 1.0)

    extra = {
        'diffuse': diffuse * incident_light * dot_nl,
        'specular': specular * incident_light * dot_nl,
        'pbr': pbr
    }

    return pbr_combined, extra


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, debug=False, evaluate=False, iteration=-1, simulate_point_light=False, override=None, exr=False):
    """
    Render the scene using the Gaussian model
    
    Args:
        viewpoint_camera: dict containing camera parameters
        pc: GaussianModel object
        pipe: Pipeline object
        bg_color: background color
        scaling_modifier: scaling modifier
        debug: debug flag
        evaluate: evaluate flag
        iteration: current iteration
        simulate_point_light: simulate point light
        override: override parameters
        exr: exr flag
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera["FoVx"] * 0.5)
    tanfovy = math.tan(viewpoint_camera["FoVy"] * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera["image_height"]),
        image_width=int(viewpoint_camera["image_width"]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=viewpoint_camera["cx"],
        cy=viewpoint_camera["cy"],
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera["world_view_transform"],
        projmatrix=viewpoint_camera["full_proj_transform"],
        sh_degree=1,
        campos=viewpoint_camera["camera_center"],
        prefiltered=False,
        backward_geometry=True,
        computer_pseudo_normal=True,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    normals = pc.get_normals
    positions = pc.get_xyz

    base_color = pc.get_base_color
    metalness = pc.get_metallic
    roughness = pc.get_roughness
    subsurfaceness = pc.get_subsurfaceness
    visibility = pc.get_visibility

    if override is not None:
        if "base_color" in override:
            base_color_list = override["base_color"]
            if "transition" in override:
                transition = override["transition"]
                base_color_new = torch.tensor(base_color_list, device=base_color.device, dtype=base_color.dtype).unsqueeze(0).repeat(base_color.shape[0], 1)
                base_color = torch.lerp(base_color, base_color_new, transition)
            else:
                base_color = torch.tensor(base_color_list, device=base_color.device, dtype=base_color.dtype).unsqueeze(0).repeat(base_color.shape[0], 1)
        if "metalness" in override:
            metalness_list = override["metalness"]
            if "transition" in override:
                transition = override["transition"]
                metalness_new = torch.tensor(metalness_list, device=metalness.device, dtype=metalness.dtype).unsqueeze(0).repeat(metalness.shape[0], 1)
                metalness = torch.lerp(metalness, metalness_new, transition)
            else:
                metalness = torch.tensor(metalness_list, device=metalness.device, dtype=metalness.dtype).unsqueeze(0).repeat(metalness.shape[0], 1)
        if "roughness" in override:
            roughness_list = override["roughness"]
            if "transition" in override:
                transition = override["transition"]
                roughness_new = torch.tensor(roughness_list, device=roughness.device, dtype=roughness.dtype).unsqueeze(0).repeat(roughness.shape[0], 1)
                roughness = torch.lerp(roughness, roughness_new, transition)
            else:
                roughness = torch.tensor(roughness_list, device=roughness.device, dtype=roughness.dtype).unsqueeze(0).repeat(roughness.shape[0], 1)
        if "subsurfaceness" in override:
            subsurfaceness_list = override["subsurfaceness"]
            if "transition" in override:
                transition = override["transition"]
                subsurfaceness_new = torch.tensor(subsurfaceness_list, device=subsurfaceness.device, dtype=subsurfaceness.dtype).unsqueeze(0).repeat(subsurfaceness.shape[0], 1)
                subsurfaceness = torch.lerp(subsurfaceness, subsurfaceness_new, transition)
            else:
                subsurfaceness = torch.tensor(subsurfaceness_list, device=subsurfaceness.device, dtype=subsurfaceness.dtype).unsqueeze(0).repeat(subsurfaceness.shape[0], 1)
        if "opacity" in override:
            opacity_list = override["opacity"]
            if "transition" in override:
                transition = override["transition"]
                opacity_new = torch.tensor(opacity_list, device=opacity.device, dtype=opacity.dtype).unsqueeze(0).repeat(opacity.shape[0], 1)
                opacity = opacity * torch.lerp(torch.ones_like(opacity), opacity_new, transition)
            else:
                opacity = opacity * torch.tensor(opacity_list, device=opacity.device, dtype=opacity.dtype).unsqueeze(0).repeat(opacity.shape[0], 1)

    # Freeze roughness for 10 000 iterations
    if iteration < 10_000 and iteration != -1:
        roughness = torch.ones_like(roughness) * 0.5

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    if override is not None:
        if "scales" in override:
            scales_list = override["scales"]
            if "transition" in override:
                transition = override["transition"]
                scales_new = torch.tensor(scales_list, device=scales.device, dtype=scales.dtype).unsqueeze(0).repeat(scales.shape[0], 1)
                scales = torch.lerp(scales, scales_new, transition)
            else:
                scales = scales * torch.tensor(scales_list, device=scales.device, dtype=scales.dtype).unsqueeze(0).repeat(scales.shape[0], 1)


    # View Direction
    camera_position_sss = viewpoint_camera["camera_center"].repeat(means3D.shape[0], 1)
    view_direction_sss = (camera_position_sss - means3D)
    view_direction_normalized_sss = view_direction_sss / (view_direction_sss.norm(dim=-1, keepdim=True) + 1e-5)

    # Light Direction
    light_position_sss = viewpoint_camera["light_position"].repeat(means3D.shape[0], 1)
    light_direction_sss = (light_position_sss - means3D)
    light_direction_normalized_sss = light_direction_sss / (view_direction_sss.norm(dim=-1, keepdim=True) + 1e-5)
    light_distance_sss = light_direction_sss.norm(dim=-1, keepdim=True)

    # Light Intensity
    light_intensity_sss = viewpoint_camera["light_intensity"].repeat(means3D.shape[0], 1)

    # Visibility
    visibility_shs_view = visibility.transpose(1, 2).view(-1, 1, 4 ** 2)
    
    # Flip the light direction YZ 
    light_direction_normalized_switched  = light_direction_normalized_sss.clone()
    # light_direction_normalized_switched[:, 1] = -light_direction_normalized_switched[:, 1]
    # light_direction_normalized_switched[:, 2] = -light_direction_normalized_switched[:, 2]
    visibilities = eval_sh(3, visibility_shs_view, light_direction_normalized_switched)
    visibilities = torch.clamp(visibilities + 0.5, 0.0, 1.0)

    # Visibility
    # Evaluate visibility using raytracer
    with torch.no_grad():
        cov_inv = pc.get_inverse_covariance()
        raytracer = RayTracer(means3D, pc.get_scaling, pc.get_rotation)
        trace_results = raytracer.trace_visibility(means3D, light_direction_sss, means3D, cov_inv, opacity, normals)
        visibilities = trace_results["visibility"]

    # Residual
    residual, incident_light = pc._sss(positions, rotations, scales, view_direction_normalized_sss, light_direction_normalized_sss, normals, visibilities, light_distance_sss, iteration=iteration)
    if not exr:
        residual = torch.clamp(residual, 0, 1)

    # Tint the residual
    residual = residual * light_intensity_sss
    
    if override is not None:
        if "residual_color" in override:
            residual_color_list = override["residual_color"]

            if "transition" in override:
                transition = override["transition"]
                residual_color_new = torch.tensor(residual_color_list, device=residual.device, dtype=residual.dtype).unsqueeze(0).repeat(residual.shape[0], 1)
                residual = residual * torch.lerp(torch.ones_like(residual), residual_color_new, transition)
            else:
                residual *= torch.tensor(residual_color_list, device=residual.device, dtype=residual.dtype).unsqueeze(0).repeat(residual.shape[0], 1)
        

    # Add PBR to color
    torch.manual_seed(0)
    colors_precomp = torch.rand_like(base_color)
    features = torch.cat([normals, base_color, metalness, roughness, subsurfaceness, visibilities, incident_light, residual], dim=1)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    (num_rendered, num_contrib, rendered_image, rendered_opacity, rendered_depth, 
        rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, radii) = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        features=features)
    
    rendered_color_specular = None
    rendered_color_diffuse = None

    rendered_gaussians = rendered_image
    rendered_normal = rendered_feature[:3, :, :]
    rendered_base_color = rendered_feature[3:6, :, :]
    rendered_metalness = rendered_feature[6:7, :, :]
    rendered_roughness = rendered_feature[7:8, :, :]
    rendered_subsurfaceness = rendered_feature[8:9, :, :]
    rendered_visibility = rendered_feature[9:10, :, :]
    rendered_incident_light = rendered_feature[10:13, :, :]
    rendered_residual = rendered_feature[13:16, :, :]

    # This is just for testing purposes
    # rendered_roughness = torch.ones_like(rendered_roughness) * 0.1
    # rendered_metalness = torch.ones_like(rendered_metalness) * 0.05
    # rendered_base_color = torch.ones_like(rendered_base_color) * 1.0
    # rendered_incident_light = torch.ones_like(rendered_incident_light) * 1.0
    # rendered_subsurfaceness = torch.ones_like(rendered_subsurfaceness) * 0.0

    with torch.no_grad():
        # FIXME: This code should be writen in a more efficient explicit way 
        def ndc_2_cam(ndc_xyz, intrinsic, W, H):
            """
            Convert normalized device coordinates to camera space

            Args:
                ndc_xyz: (B, N, C, H, W, 3)
                intrinsic: (B, 3, 3)
                W: image width
                H: image height
            """

            inv_scale = torch.tensor([[W - 1, H - 1]], device=ndc_xyz.device)
            cam_z = ndc_xyz[..., 2:3]
            cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
            cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)
            cam_xyz = cam_xyz @ torch.inverse(intrinsic[0, ...].t())
            return cam_xyz

        def depth2point_cam(sampled_depth, ref_intrinsic):
            """
            Convert depth image to camera space points

            Args:
                sampled_depth: (B, N, C, H, W)
                ref_intrinsic: (B, 3, 3)
            """

            B, N, C, H, W = sampled_depth.shape
            valid_z = sampled_depth
            valid_x = torch.arange(W, dtype=torch.float32, device=sampled_depth.device) / (W - 1)
            valid_y = torch.arange(H, dtype=torch.float32, device=sampled_depth.device) / (H - 1)
            valid_y, valid_x = torch.meshgrid(valid_y, valid_x, indexing='ij')
            # B,N,H,W
            valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
            valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
            ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(B, N, C, H, W, 3)  # 1, 1, 5, 512, 640, 3
            cam_xyz = ndc_2_cam(ndc_xyz, ref_intrinsic, W, H) # 1, 1, 5, 512, 640, 3
            return ndc_xyz, cam_xyz
        
        def depth2point_world(depth_image, intrinsic_matrix, extrinsic_matrix):
            """
            Convert depth image to world space points

            Args:
                depth_image: (H, W)
                intrinsic_matrix: (3, 3)
                extrinsic_matrix: (4, 4)
            """
            _, xyz_cam = depth2point_cam(depth_image[None,None,None,...], intrinsic_matrix[None,...])
            xyz_cam = xyz_cam.reshape(-1,3)
            xyz_world = torch.cat([xyz_cam, torch.ones_like(xyz_cam[...,0:1])], axis=-1) @ torch.inverse(extrinsic_matrix).transpose(0,1)
            xyz_world = xyz_world[...,:3]
            xyz_world = xyz_world.reshape(*depth_image.shape, 3)
            xyz_world = xyz_world.permute(2, 0, 1)
            return xyz_world

        def get_calib_matrix_nerf(viewpoint_camera):
            """
            Get calibration matrix

            Args:
                viewpoint_camera: dict containing camera parameters
            """
            FoVx = viewpoint_camera["FoVx"]
            image_width = viewpoint_camera["image_width"]
            image_height = viewpoint_camera["image_height"]
            world_view_transform = viewpoint_camera["world_view_transform"]

            focal = fov2focal(FoVx, image_width)  # original focal length
            intrinsic_matrix = torch.tensor([[focal, 0, image_width / 2], [0, focal, image_height / 2], [0, 0, 1]], device=world_view_transform.device).float()
            extrinsic_matrix = world_view_transform.transpose(0,1).contiguous() # cam2world
            return intrinsic_matrix, extrinsic_matrix

        intrinsic_matrix, extrinsic_matrix = get_calib_matrix_nerf(viewpoint_camera)
        rendered_surface_xyz = depth2point_world(rendered_depth[0], intrinsic_matrix, extrinsic_matrix) 

        # Flatten image space
        rendered_surface_xyz_flat = rendered_surface_xyz.permute(1, 2, 0).reshape(-1, 3)

        # Light Direction
        light_position = viewpoint_camera["light_position"].repeat(rendered_surface_xyz_flat.shape[0], 1)
        light_direction = (light_position - rendered_surface_xyz_flat)
        light_direction_normalized = light_direction / (light_direction.norm(dim=-1, keepdim=True) + 1e-5)

        # View Direction
        camera_position = viewpoint_camera["camera_center"].repeat(rendered_surface_xyz_flat.shape[0], 1)
        view_direction = (camera_position - rendered_surface_xyz_flat)
        view_direction_normalized = view_direction / (view_direction.norm(dim=-1, keepdim=True) + 1e-5)

        # Light Distance 
        if simulate_point_light == True:
            light_distance = torch.norm(light_direction, dim=-1, keepdim=True)
            rendered_incident_light = 40.0 / (light_distance * light_distance + 1e-5) 
            rendered_incident_light = rendered_incident_light.repeat(1, 3).reshape(viewpoint_camera["image_height"], viewpoint_camera["image_width"], 3).permute(2, 0, 1) * (rendered_visibility > 0.1).float()
        
        # Light Intensity
        light_intensity = viewpoint_camera["light_intensity"]
        light_intensity = light_intensity.repeat(rendered_surface_xyz_flat.shape[0], 1).reshape(viewpoint_camera["image_height"], viewpoint_camera["image_width"], 3).permute(2, 0, 1)
        rendered_incident_light = rendered_incident_light * light_intensity

    # Flatten all values 
    rendered_base_color_flat = rendered_base_color.permute(1, 2, 0).reshape(-1, 3)
    rendered_metalness_flat = rendered_metalness.permute(1, 2, 0).reshape(-1, 1)
    rendered_roughness_flat = rendered_roughness.permute(1, 2, 0).reshape(-1, 1)
    rendered_subsurfaceness_flat = rendered_subsurfaceness.permute(1, 2, 0).reshape(-1, 1)
    rendered_normal_flat = rendered_normal.permute(1, 2, 0).reshape(-1, 3)
    rendered_residual_flat = rendered_residual.permute(1, 2, 0).reshape(-1, 3)
    rendered_subsurfaceness_flat = rendered_subsurfaceness.permute(1, 2, 0).reshape(-1, 1)
    rendered_incident_light_flat = rendered_incident_light.permute(1, 2, 0).reshape(-1, 3)

    pbr_combined, extra = render_equation(
        rendered_base_color_flat,
        rendered_metalness_flat,
        rendered_roughness_flat,
        rendered_subsurfaceness_flat, 
        rendered_normal_flat,
        rendered_residual_flat,
        light_direction_normalized,
        view_direction_normalized,
        rendered_incident_light_flat,
        debug=debug
    )

    rendered_image = pbr_combined.view(rendered_surface_xyz.shape[1], rendered_surface_xyz.shape[2], 3).permute(2, 0, 1)
    rendered_color_specular = extra["specular"].view(rendered_surface_xyz.shape[1], rendered_surface_xyz.shape[2], 3).permute(2, 0, 1)
    rendered_color_diffuse = extra["diffuse"].view(rendered_surface_xyz.shape[1], rendered_surface_xyz.shape[2], 3).permute(2, 0, 1)
    rendered_pbr = extra["pbr"].view(rendered_surface_xyz.shape[1], rendered_surface_xyz.shape[2], 3).permute(2, 0, 1)

    if override is not None:
        if "random_color_scale" in override:
            random_color_scale = override["random_color_scale"]

            if "transition" in override:
                transition = override["transition"]
                random_color_scale_new = torch.tensor(random_color_scale, device=rendered_image.device, dtype=rendered_image.dtype).unsqueeze(0).repeat(rendered_image.shape[0], 1)
                random_color_scale = torch.lerp(random_color_scale, random_color_scale_new, transition)
            else:
                rendered_image = (1.0 - random_color_scale) * rendered_image + random_color_scale * rendered_gaussians

    return {"render": rendered_image,
            "normals": rendered_normal,
            "pseudo_normals": rendered_pseudo_normal,
            "surface_xyz": rendered_surface_xyz,
            "opacity": rendered_opacity,
            "depth": rendered_depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "num_rendered": num_rendered,
            "num_contrib": num_contrib,
            "base_color": rendered_base_color,
            "metalness": rendered_metalness,
            "roughness": rendered_roughness,
            "subsurfaceness": rendered_subsurfaceness,
            "pbr": rendered_pbr,
            "residual": rendered_residual,
            "visibility": rendered_visibility,
            "incident_light": rendered_incident_light,
            "color_specular": rendered_color_specular,
            "color_diffuse": rendered_color_diffuse,
            }
