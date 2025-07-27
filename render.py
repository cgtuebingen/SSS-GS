import torch
from scene import Scene
import os
import json 
import cv2
import subprocess
from tqdm import tqdm
import numpy as np
from os import makedirs
from gaussian_renderer import render
import torchvision
from torch.utils.data import DataLoader
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def save_image(tensor, model_path, name, iteration, idx, map_name, exr=False):
    if exr:
        tensor = tensor.permute(1, 2, 0).cpu().numpy()
        tensor = cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)
        path = os.path.join(model_path, name, "ours_{}".format(iteration), map_name)
        makedirs(path, exist_ok=True)
        cv2.imwrite(os.path.join(path, '{0:05d}'.format(idx) + ".exr"), tensor)
        # print("Saved image to", os.path.join(path, '{0:05d}'.format(idx) + ".exr"))
    else:
        tensor = tensor.clamp(0, 1)
        path = os.path.join(model_path, name, "ours_{}".format(iteration), map_name)
        makedirs(path, exist_ok=True)
        torchvision.utils.save_image(tensor, os.path.join(path, '{0:05d}'.format(idx) + ".png"))


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, uses_gt_images, maps = None, exr=False):
    rendered_lookup = {}
    
    dataloader = DataLoader(views, shuffle=False, num_workers=4, batch_size=50, prefetch_factor=2,  persistent_workers=True, pin_memory=True)

    data_iterator = tqdm(dataloader, desc="Rendering cameras")
    counter = 0
    for camera in data_iterator:  
        batch_size = len(camera["uid"])
    
        # Iterate over each item in the batch
        item_iterator = tqdm(range(batch_size), desc="Processing items", leave=False)
        for i in item_iterator: # TODO: This is not efficient but the code is not designed to handle batched rendering
            # Construct a dictionary for the current item by extracting the ith element from each key
            view = {key: value[i] if not isinstance(value, torch.Tensor) else value[i].cuda() for key, value in camera.items()}

            idx = view["uid"]
            image_name = view["image_name"]
            rendered_lookup['{0:05d}'.format(idx)] = image_name

            # steps = 50.0
            #scale = min(counter / steps, 1.0)
            # random_color_scale = 1.0 - min(counter / steps, 1.0)
            # override ={"scales": [scale, scale, scale], "random_color_scale": random_color_scale}

            start_step = 20
            end_step = 70 

            transition = min(max((counter - start_step) / (end_step - start_step), 0.0), 1.0)

            # override_wax = {"residual_color": [0.8, 0.0, 0.0], "base_color": [0.6, 0.2, 0.2], "roughness": 0.4, "metallness": 0.2, "subsurfaceness": 0.8}
            #override_glass = { "roughness": 0.2, "metallness": 0.0, "opacity": 0.12, "residual_color": [0.5, 0.5, 0.4]}
            #override_skin = {"residual_color": [0.5, 0.2, 0.2], "base_color": [1.0, 0.9, 0.9], "roughness": 0.9, "metallness": 0.0, "subsurfaceness": 0.8}

            override = {"residual_color": [0.5, 0.2, 0.2], "base_color": [1.0, 0.9, 0.9], "roughness": 0.9, "metallness": 0.0, "subsurfaceness": 0.8, "transition": transition}

            render_pkg = render(view, gaussians, pipeline, background, debug=True, exr=exr, override=override)

            counter += 1


            # Rendered image
            rendered_image = render_pkg["render"].clamp(0,1)
            save_image(rendered_image, model_path, name, iteration, idx, "renders", exr=exr)

            if uses_gt_images:
                # Original image
                gt_image = view["original_image"][0:3, :, :]
                save_image(gt_image, model_path, name, iteration, idx, "gt", exr=exr)
                
                difference = torch.abs(rendered_image - gt_image)
                save_image( difference, model_path, name, iteration, idx, "differences", exr=exr)

                if "image_mask" in view: 
                    mask = view["image_mask"][0:3, :, :]
                    save_image(mask, model_path, name, iteration, idx, "mask", exr=exr)
            

            if maps is not None:
                for map_name in maps:
                    if map_name in render_pkg:
                        rendered_map = render_pkg[map_name]

                        if map_name == "normals":
                            rendered_map = (rendered_map + 1) / 2

                        if map_name == "depth" and not exr:
                            nonzero_depth = rendered_map > 0
                            rendered_depth_selected = rendered_map[nonzero_depth]
                            rendered_map = (rendered_map - rendered_depth_selected.min()) / (rendered_depth_selected.max() - rendered_depth_selected.min())

                        if map_name == "incident_light" and not exr:
                            max_incident_light = torch.max(rendered_map)
                            rendered_map = rendered_map / max_incident_light

                        save_image(rendered_map, model_path, name, iteration, idx, map_name, exr=exr)
    
    # Save the rendered lookup
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "rendered_lookup.json"), 'w') as f:
        json.dump(rendered_lookup, f)
                
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, name, mode, uses_gt_images=False, custom_transforms_file=None, maps=None, exr=False):
    with torch.no_grad():
        gaussians = GaussianModel()
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, custom_transforms_file=custom_transforms_file, uses_gt_images=uses_gt_images)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        
        if mode == "train" or mode == "both":
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, uses_gt_images=uses_gt_images, maps=maps, exr=exr)
        if mode == "test" or mode == "both":
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, uses_gt_images=uses_gt_images, maps=maps, exr=exr)
        if mode == "custom":
            render_set(dataset.model_path, name, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, uses_gt_images=uses_gt_images, maps=maps, exr=exr)

        

def render_videos(model_path, name, iteration, output_dir='videos', fps=24.0, uses_gt_images=False, maps=None): 
    """Create videos from rendered images using ffmpeg."""
    base_path = os.path.join(model_path, name, f"ours_{iteration}")

    categories = ['renders']

    if uses_gt_images:
        categories += ['gt', 'differences', 'mask']

    if maps is not None:
        categories += maps

    makedirs(os.path.join(base_path, output_dir), exist_ok=True)

    for category in categories:
        input_path = os.path.join(base_path, category, '%05d.png')
        output_path = os.path.join(base_path, output_dir, f"{category}.mp4")
        ffmpeg_command = [
            '/bin/ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-framerate', str(fps),  # Frame rate
            '-i', input_path,  # Input file format
            '-c:v', 'libx264',  # Video codec
            '-profile:v', 'high',  # H.264 profile
            '-crf', '20',  # Constant rate factor (quality)
            '-pix_fmt', 'yuv420p',  # Pixel format
            '-start_number', '0',  # Start number for file sequence
            output_path  # Output file
        ]
        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)




if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_videos", action="store_true")
    parser.add_argument("--from_existing_images", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--maps", default="none", type=str)
    
    parser.add_argument("--mode", default="train", type=str, choices=["train", "test", "both", "custom"])
    parser.add_argument("--name", default="custom", type=str)
    parser.add_argument("--custom_transforms_file", default=None, type=str)
    parser.add_argument("--no_gt_images", action="store_true")
    parser.add_argument("--exr", action="store_true")
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Properties
    uses_gt_images = not args.no_gt_images

    # For mode custom transforms file is required
    if args.mode == "custom" and args.custom_transforms_file is None:
        raise ValueError("Custom mode requires a transforms file to be specified.")

    maps = None
    if args.maps == "all":
        maps = ["normals", "depth", "base_color", "roughness", "metalness", "pbr", "residual", "visibility", "color_diffuse", "color_specular", "opacity", "incident_light", "subsurfaceness"]
    if args.maps == "none":
        maps = None
    if args.maps != "all" and args.maps != "none":
        maps = args.maps.split(",")

    if not args.from_existing_images:
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.name, args.mode, uses_gt_images=uses_gt_images, custom_transforms_file=args.custom_transforms_file, maps=maps, exr=args.exr)

    

    # Create videos from the rendered images with ffmpeg
    if not args.skip_videos:
        if args.mode == "train" or args.mode == "both":
            render_videos(args.model_path, "train", args.iteration, uses_gt_images=uses_gt_images, maps=maps)
        if args.mode == "test" or args.mode == "both":
            render_videos(args.model_path, "test", args.iteration, uses_gt_images=uses_gt_images, maps=maps)
        if args.mode == "custom":
            render_videos(args.model_path, args.name, args.iteration, uses_gt_images=uses_gt_images, maps=maps)
