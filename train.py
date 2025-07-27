import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
from itertools import cycle
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
import torchvision
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
from utils.loss_utils import ssim, bilateral_smooth_loss
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import warnings
from bvh import RayTracer
from utils.sh_utils import eval_sh
from utils.color_utils import srgb_to_linear, linear_to_srgb

# Suppress specific warnings from torchmetrics for lpips
warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")
warnings.filterwarnings("ignore", message="Arguments other than a weight enum or `None` for 'weights' are deprecated")

def training(dataset, opt, pipe, render_iteraions, testing_iterations, testing_size, saving_iterations, checkpoint_iterations, checkpoint, debug_from, use_wandb, evaluate):
    torch.manual_seed(0)
    
    first_iter = 0

    wandb_writer = prepare_output_and_logger(dataset, use_wandb)

    gaussians: GaussianModel = GaussianModel()
    scene: Scene = Scene(dataset, gaussians)

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # Finetune visibility
    # gaussians.finetune_visibility()

    # Data Loading
    train_dataset = scene.getTrainCameras()
    print("Batch size: {}".format(opt.batch_size))
    dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, sampler=RandomSampler(train_dataset, replacement=True), num_workers=4, pin_memory=True, drop_last=True)
    data_iterator = cycle(dataloader)

    if evaluate:
        test_dataset = scene.getTestCameras()

    # Sample fixed viewpoints for testing
    idx_train, idx_test = None, None
    test_count = testing_size
    
    if evaluate:
        idx_train = torch.randperm(len(train_dataset))[:test_count]
        idx_test = torch.randperm(len(test_dataset))[:test_count]


    # Debug images folder
    debug_dir = os.path.join(dataset.model_path, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    # viewpoint_stack = None
    camera_batch = None
    batch_index = 0

    # Losses
    lpips = LearnedPerceptualImagePatchSimilarity().cuda()

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if camera_batch is None or iteration % len(camera_batch["uid"]) == 0:
            batch_index = 0
            camera_batch = next(data_iterator)
           
        viewpoint_cam = {key: value[batch_index] if not isinstance(value, torch.Tensor) else value[batch_index].cuda() for key, value in camera_batch.items()}
        batch_index += 1

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        is_debug_iteration = iteration % render_iteraions == 0
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, debug=is_debug_iteration, iteration=iteration)

        image = render_pkg["render"]
        radii = render_pkg["radii"]
        normals = render_pkg["normals"]
        opacity = render_pkg["opacity"]
        pseudo_normals = render_pkg["pseudo_normals"]
        gt_image = viewpoint_cam["original_image"][0:3, :, :]
        image_mask = viewpoint_cam["image_mask"][0:3, :, :]
        visibility_filter = render_pkg["visibility_filter"]
        viewspace_point_tensor = render_pkg["viewspace_points"]

        # sRGB to linear RGB
        gt_image = srgb_to_linear(gt_image)

        # Loss
        loss_dict = {}

        loss_l1 = l1_loss(image, gt_image)
        loss_dict["l1"] = loss_l1.item()

        loss_ssim = ssim(image, gt_image)
        loss_dict["ssim"] = loss_ssim.item()

        loss = (1.0 - opt.lambda_dssim) * loss_l1 + opt.lambda_dssim * (1.0 - loss_ssim)
        loss_dict["original"] = loss.item()

        image_lpips_input = (2.0 * linear_to_srgb(image).unsqueeze(0) -1).clamp(-1, 1)
        gt_image_lpips_input = (2.0 * linear_to_srgb(gt_image).unsqueeze(0) -1).clamp(-1, 1)

        loss_lpips = lpips(image_lpips_input, gt_image_lpips_input)
        loss_dict["lpips"] = loss_lpips.item()
        loss = loss + opt.lambda_lpips * loss_lpips

        # Normal loss computation
        loss_normal = F.mse_loss(normals, pseudo_normals.detach())
        loss_dict["normal"] = loss_normal.item()
        loss = loss + opt.lambda_normal * loss_normal

        """ # PBR loss computation
        loss_pbr = F.mse_loss(render_pkg["pbr"], gt_image)
        loss_dict["pbr"] = loss_pbr.item()
        loss = loss + 0.5 * loss_pbr """


        """ if opt.lambda_incident_light > 0:
            # Retrieve necessary tensors
            visibility = render_pkg["visibility"].detach()  
            incident_light = render_pkg["incident_light"]   
            surface_xyz = render_pkg["surface_xyz"]         
            light_position = viewpoint_cam["light_position"]  

            # Compute the vector from surface points to the light position
            diff = surface_xyz - light_position.view(3, 1, 1)  
            # Compute the squared distance with epsilon to avoid division by zero
            epsilon = 1e-6
            distance_sq = torch.sum(diff ** 2, dim=0) + epsilon  

            # Compute the expected incident light using the inverse square law
            light_intensity = 40.0
            expected_incident_light = visibility * (light_intensity / distance_sq)

            # Expand expected_incident_light to match the shape of incident_light
            expected_incident_light = expected_incident_light.expand_as(incident_light)

            # Compute the MSE loss between the expected and actual incident light
            loss_incident = F.mse_loss(incident_light, expected_incident_light)
            loss_dict["incident_light"] = loss_incident.item()

            # Add the incident light loss to the total loss
            loss = loss + opt.lambda_incident_light * loss_incident """


        if opt.lambda_mask_entropy > 0:
            o = opacity.clamp(1e-6, 1 - 1e-6)
            loss_mask_entropy = -(image_mask * torch.log(o) + (1 - image_mask) * torch.log(1 - o)).mean()
            loss_dict["mask_entropy"] = loss_mask_entropy.item()
            loss = loss + opt.lambda_mask_entropy * loss_mask_entropy

        if opt.lambda_base_color_smooth > 0:
            base_color = render_pkg["base_color"]
            loss_base_color_smooth = bilateral_smooth_loss(base_color, gt_image, image_mask)
            loss_dict["base_color_smooth"] = loss_base_color_smooth.item()
            loss = loss + opt.lambda_base_color_smooth * loss_base_color_smooth

        if iteration < 10_000 and iteration != -1:
            if opt.lambda_metallic_smooth > 0:
                metallic = render_pkg["metalness"]
                loss_metallic_smooth = bilateral_smooth_loss(metallic, gt_image, image_mask)
                loss_dict["metallic_smooth"] = loss_metallic_smooth.item()
                loss = loss + opt.lambda_metallic_smooth * loss_metallic_smooth

            if opt.lambda_roughness_smooth > 0:
                roughness = render_pkg["roughness"]
                loss_roughness_smooth = bilateral_smooth_loss(roughness, gt_image, image_mask)
                loss_dict["roughness_smooth"] = loss_roughness_smooth.item()
                loss = loss + opt.lambda_roughness_smooth * loss_roughness_smooth

        if opt.lambda_subsurfaceness_smooth > 0:
            subsurface = render_pkg["subsurfaceness"]
            loss_subsurface_smooth = bilateral_smooth_loss(subsurface, gt_image, image_mask)
            loss_dict["subsurface_smooth"] = loss_subsurface_smooth.item()
            loss = loss + opt.lambda_subsurfaceness_smooth * loss_subsurface_smooth

        if opt.lambda_base_color > 0:
            value_img = torch.max(gt_image * image_mask, dim=0, keepdim=True)[0]
            shallow_enhance = gt_image * image_mask
            shallow_enhance = 1 - (1 - shallow_enhance) * (1 - shallow_enhance)

            specular_enhance = gt_image * image_mask
            specular_enhance = specular_enhance * specular_enhance

            k = 5
            specular_weight = 1 / (1 + torch.exp(-k * (value_img - 0.5)))
            target_img = (specular_weight * specular_enhance + (1 - specular_weight) * shallow_enhance)

            base_color = render_pkg["base_color"]
            loss_base_color = F.l1_loss(target_img, base_color)
            lambda_base_color = opt.lambda_base_color  # * max(0, 1 - float(dict_params["iteration"]) / (opt.base_color_guide_iter_num+1e-8))
            loss_dict["base_color"] = loss_base_color.item()

            loss = loss + lambda_base_color * loss_base_color
        
        """ if opt.lambda_visibility > 0:
            num = 10000
            means3D = gaussians.get_xyz
            visibility = gaussians.get_visibility
            normal = gaussians.get_normals
            opacity = gaussians.get_opacity

            rand_idx = torch.randperm(means3D.shape[0])[:num]
            rand_visibility_shs_view = visibility.transpose(1, 2).view(-1, 1, 4 ** 2)[rand_idx]
            rand_rays_o = means3D[rand_idx]
            rand_rays_d = torch.randn_like(rand_rays_o)
            cov_inv = gaussians.get_inverse_covariance()
            rand_normal = normal[rand_idx]
            mask = (rand_rays_d * rand_normal).sum(-1) < 0
            rand_rays_d[mask] *= -1

            # Scale rays
            rand_rays_d *= 10.0 # 10 units long

            sample_sh2vis = eval_sh(3, rand_visibility_shs_view, rand_rays_d)
            sample_vis = torch.clamp(sample_sh2vis + 0.5, 0.0, 1.0)
            raytracer = RayTracer(means3D, gaussians.get_scaling, gaussians.get_rotation)
            trace_results = raytracer.trace_visibility(
                rand_rays_o, rand_rays_d, means3D,
                cov_inv, opacity, normal)

            rand_ray_visibility = trace_results["visibility"]
            loss_visibility = F.l1_loss(rand_ray_visibility, sample_vis)
            loss_dict["loss_visibility"] = loss_visibility.item()
            loss = loss + opt.lambda_visibility * loss_visibility """

        # Adding new PBR regularizer
        """ if opt.lambda_pbr_regularizer > 0:
            with torch.no_grad():
                visible_area = (render_pkg["incident_light"] > 0.0).float()

            # Convert PBR to grayscale
            pbr = render_pkg["pbr"]
            pbr_gray = 0.2989 * pbr[0, :, :] + 0.5870 * pbr[1, :, :] + 0.1140 * pbr[2, :, :]

            # PBR is not allowed to be zero in lit areas
            pbr_penalty = F.relu(1e-3 - pbr_gray)  # Penalize small values close to zero
            loss_pbr_regularizer = (pbr_penalty * visible_area).mean()
            loss_dict["pbr_regularizer"] = loss_pbr_regularizer.item()
            loss = loss + opt.lambda_pbr_regularizer * loss_pbr_regularizer """

        
        loss_dict["total"] = loss.item()

        # Backprop
        loss.backward()
        iter_end.record()

    
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Roughness reset 
            if iteration == 15_000:
                # gaussians.reset_roughness()
                # gaussians.reset_metallic()
                pass

            # Log and save
            training_report(wandb_writer, iteration, render_iteraions, loss_dict, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), opt, render_pkg, viewpoint_cam, debug_dir, idx_train, idx_test, evaluate)
            if (iteration in saving_iterations):
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                   gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                gaussians.sss_optimizer.step()
                gaussians.sss_optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args, use_wandb):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create wandb logger
    if use_wandb:
        wandb.init(
            project="gs_sss", 
            name=args.model_path.split("/")[-1],
            dir=args.model_path, 
            config=vars(args)
        )
        return wandb


def training_report(
    wandb_writer, 
    iteration, 
    render_iteraions, 
    loss_dict, 
    elapsed, 
    testing_iterations, 
    scene : Scene, 
    renderFunc, 
    renderArgs,
    opt, 
    render_pkg, 
    viewpoint_cam, 
    debug_dir, 
    idx_train, 
    idx_test,
    evaluate
): 
    
    # Add train_loss_patches infront of loss_dict keys
    train_information ={}
    for key in list(loss_dict.keys()):
        train_information["train_loss_patches/" + key] = loss_dict[key]

    train_information["iter_time"] = elapsed
    train_information["point_count"] = scene.gaussians.get_xyz.shape[0]
    train_information["learning_rate"] = scene.gaussians.sss_optimizer.param_groups[0]["lr"]
    train_information["incident_light_max"] = torch.max(render_pkg["incident_light"]).item()

    
    if wandb_writer:
        wandb_writer.log(train_information, step=iteration)

    
    if iteration % render_iteraions == 0 and evaluate: 
        def convertTensorToPILImage(tensor):
            tensor = tensor.clamp(0, 1)
            return torchvision.transforms.ToPILImage()(tensor)

        wandb_images = []

        image = render_pkg["render"]
        image = linear_to_srgb(image).clamp(0, 1)

        torchvision.utils.save_image(image, os.path.join(debug_dir, '{0:05d}'.format(iteration) + ".png"))
        wandb_images.append(wandb.Image(convertTensorToPILImage(image), caption="Render", grouping="render"))

        gt_image = viewpoint_cam["original_image"][0:3, :, :]
        torchvision.utils.save_image(gt_image, os.path.join(debug_dir, '{0:05d}'.format(iteration) + "_gt.png"))
        wandb_images.append(wandb.Image(convertTensorToPILImage(gt_image), caption="Ground Truth", grouping="gt"))

        difference = torch.abs(image - gt_image)
        torchvision.utils.save_image(difference, os.path.join(debug_dir, '{0:05d}'.format(iteration) + "_diff.png"))
        wandb_images.append(wandb.Image(convertTensorToPILImage(difference), caption="Difference", grouping="diff"))

        if "normals" in render_pkg:
            normals = render_pkg["normals"]
            normals = (normals + 1) / 2
            normals = normals.clamp(0, 1)
            torchvision.utils.save_image(normals, os.path.join(debug_dir, '{0:05d}'.format(iteration) + "_normals.png"))
            normals = convertTensorToPILImage(normals)
            wandb_images.append(wandb.Image(normals, caption="Normals", grouping="normals"))

        if "base_color" in render_pkg:
            base_color = linear_to_srgb(render_pkg["base_color"]).clamp(0, 1)
            torchvision.utils.save_image(base_color, os.path.join(debug_dir, '{0:05d}'.format(iteration) + "_base_color.png"))
            base_color = convertTensorToPILImage(base_color)
            wandb_images.append(wandb.Image(base_color, caption="Base Color", grouping="base_color"))

        if "roughness" in render_pkg:
            roughness = render_pkg["roughness"].clamp(0, 1)
            torchvision.utils.save_image(roughness, os.path.join(debug_dir, '{0:05d}'.format(iteration) + "_roughness.png"))
            roughness = convertTensorToPILImage(roughness)
            wandb_images.append(wandb.Image(roughness, caption="Roughness", grouping="roughness"))

        if "metalness" in render_pkg:
            metalness = render_pkg["metalness"].clamp(0, 1)
            torchvision.utils.save_image(metalness, os.path.join(debug_dir, '{0:05d}'.format(iteration) + "_metalness.png"))
            metalness = convertTensorToPILImage(metalness)
            wandb_images.append(wandb.Image(metalness, caption="Metalness", grouping="metalness"))

        if "subsurfaceness" in render_pkg:
            subsurfaceness = render_pkg["subsurfaceness"].clamp(0, 1)
            torchvision.utils.save_image(subsurfaceness, os.path.join(debug_dir, '{0:05d}'.format(iteration) + "_subsurfaceness.png"))
            subsurfaceness = convertTensorToPILImage(subsurfaceness)
            wandb_images.append(wandb.Image(subsurfaceness, caption="Subsurfaceness", grouping="subsurfaceness"))
        
        if "pbr" in render_pkg:
            pbr = linear_to_srgb(render_pkg["pbr"]).clamp(0, 1)
            torchvision.utils.save_image(pbr, os.path.join(debug_dir, '{0:05d}'.format(iteration) + "_pbr.png"))
            pbr = convertTensorToPILImage(pbr)
            wandb_images.append(wandb.Image(pbr, caption="PBR", grouping="pbr"))

        if "residual" in render_pkg:
            residual = render_pkg["residual"].clamp(0, 1)
            torchvision.utils.save_image(residual, os.path.join(debug_dir, '{0:05d}'.format(iteration) + "_residual.png"))
            residual = convertTensorToPILImage(residual)
            wandb_images.append(wandb.Image(residual, caption="Residual", grouping="residual"))

        if "visibility" in render_pkg:
            visibility = render_pkg["visibility"].clamp(0, 1)
            light_position = viewpoint_cam["light_position"]
            light_position_x = light_position[0].item()
            light_position_y = light_position[1].item()
            light_position_z = light_position[2].item()
            torchvision.utils.save_image(visibility, os.path.join(debug_dir, '{0:05d}'.format(iteration) + "_visibility_" + "X: " + '{0:.2f}'.format(light_position_x) + "_Y: " + '{0:.2f}'.format(light_position_y) + "_Z: " + '{0:.2f}'.format(light_position_z) + ".png"))
            visibility = convertTensorToPILImage(visibility)
            wandb_images.append(wandb.Image(visibility, caption="Visibility", grouping="visibility"))

        if "incident_light" in render_pkg:
            incident_light = render_pkg["incident_light"]
            max_incident_light = torch.max(incident_light)
            min_incident_light = torch.min(incident_light)
            incident_light = incident_light / max_incident_light
            torchvision.utils.save_image(incident_light, os.path.join(debug_dir, '{0:05d}'.format(iteration) + "_incident_light_" + "MAX: " + '{0:.2f}'.format(max_incident_light.item()) + ", MIN: " + '{0:.2f}'.format(min_incident_light.item()) + ".png"))
            incident_light = convertTensorToPILImage(incident_light)
            wandb_images.append(wandb.Image(incident_light, caption="Incident Light", grouping="incident_light"))

        if "color_diffuse" in render_pkg:
            color_diffuse = linear_to_srgb(render_pkg["color_diffuse"]).clamp(0, 1)
            torchvision.utils.save_image(color_diffuse, os.path.join(debug_dir, '{0:05d}'.format(iteration) + "_color_diffuse.png"))
            color_diffuse = convertTensorToPILImage(color_diffuse)
            wandb_images.append(wandb.Image(color_diffuse, caption="Color Diffuse", grouping="color_diffuse"))

        if "color_specular" in render_pkg:
            color_specular = linear_to_srgb(render_pkg["color_specular"]).clamp(0, 1)
            torchvision.utils.save_image(color_specular, os.path.join(debug_dir, '{0:05d}'.format(iteration) + "_color_specular.png"))
            color_specular = convertTensorToPILImage(color_specular)
            wandb_images.append(wandb.Image(color_specular, caption="Color Specular", grouping="color_specular"))

        if "opacity" in render_pkg:
            opacity = render_pkg["opacity"].clamp(0, 1)
            torchvision.utils.save_image(opacity, os.path.join(debug_dir, '{0:05d}'.format(iteration) + "_opacity.png"))
            opacity = convertTensorToPILImage(opacity)
            wandb_images.append(wandb.Image(opacity, caption="Opacity", grouping="opacity"))

        if "image_mask" in viewpoint_cam:
            mask = viewpoint_cam["image_mask"][0:3, :, :]
            torchvision.utils.save_image(mask, os.path.join(debug_dir, '{0:05d}'.format(iteration) + "_mask.png"))
            mask = convertTensorToPILImage(mask)
            wandb_images.append(wandb.Image(mask, caption="Mask", grouping="mask"))

        # Calculate PSNR
        psnr_val = psnr(image, gt_image)
        ssim_val = ssim(image, gt_image)

        if wandb_writer:
            wandb_writer.log({
                "train_render/images": wandb_images, 
                "train_render/psnr": psnr_val.item(), 
                "train_render/ssim": ssim_val.item(),
                "train_render/min_pbr": torch.min(render_pkg["pbr"]).item(),
                "train_render/max_pbr": torch.max(render_pkg["pbr"]).item(),
            }, step=iteration)


    # Report test and samples of training set
    if iteration in testing_iterations and evaluate:

        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()

        def evaluate_on_dataset(dataset, sample_idx, name, iteration, wandb_writer, renderFunc, renderArgs, opt):
            dataloader = DataLoader(dataset, batch_size=50, sampler=SubsetRandomSampler(sample_idx), num_workers=4, pin_memory=True, drop_last=True)
        
            data_iterator = tqdm(dataloader, desc="Evaluation on " + name + " set", leave=False)
            values ={"psnr": 0.0, "loss": 0.0, "ssim": 0.0, "l1": 0.0}
            for camera in data_iterator:
                batch_size = len(camera["uid"])
        
                # Iterate over each item in the batch
                item_iterator = tqdm(range(batch_size), desc="Processing items", leave=False)
                for i in item_iterator: 
                    view = {key: value[i] if not isinstance(value, torch.Tensor) else value[i].cuda() for key, value in camera.items()}

                    idx = view["uid"]
                    maps = renderFunc(view, scene.gaussians, renderArgs[0], renderArgs[1], debug=False, evaluate=True, iteration=iteration)

                    image = maps["render"]
                    gt_image = view["original_image"][0:3, :, :]
                    gt_image = srgb_to_linear(gt_image)
                    Ll1 = l1_loss(image, gt_image)
                    ssim_val = ssim(image, gt_image)
                    psnr_val = psnr(image, gt_image)
                    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)

                    data_iterator.set_postfix({"Loss": f"{loss:.{7}f}", "PSNR": f"{psnr_val:.{7}f}", "SSIM": f"{ssim_val:.{7}f}", "L1": f"{Ll1:.{7}f}"})

                    values["psnr"] += psnr_val
                    values["loss"] += loss
                    values["ssim"] += ssim_val
                    values["l1"] += Ll1

            values["psnr"] /= sample_idx.shape[0]
            values["loss"] /= sample_idx.shape[0]
            values["ssim"] /= sample_idx.shape[0]
            values["l1"] /= sample_idx.shape[0]

            if wandb_writer:
                wandb_writer.log({f"evaluation/{name}_psnr": values["psnr"], f"evaluation/{name}_loss": values["loss"], f"evaluation/{name}_ssim": values["ssim"], f"evaluation/{name}_l1": values["l1"]}, step=iteration)

        evaluate_on_dataset(train_cameras, idx_train, "train", iteration, wandb_writer, renderFunc, renderArgs, opt)
        evaluate_on_dataset(test_cameras, idx_test, "test", iteration, wandb_writer, renderFunc, renderArgs, opt)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--render_iterations", type=int, default=1000)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 5_000, 7_000, 10_000, 12_000, 15_000, 20_000, 30_000, 40_000, 50_000, 60_000, 90_000, 120_000])
    parser.add_argument("--testing_size", type=int, default=50)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 5_000, 7_000, 10_000, 12_000, 15_000, 20_000, 30_000, 40_000, 50_000, 60_000, 90_000, 120_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Set up training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.render_iterations, args.test_iterations, args.testing_size, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.wandb, args.eval)

    
    # All done
    print("\nTraining complete.")
