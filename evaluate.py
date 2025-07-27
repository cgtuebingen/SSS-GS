from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import torch
from scene import Scene
import os
import subprocess
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from torch.utils.data import DataLoader
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

import json
from tqdm import tqdm
from argparse import ArgumentParser
import time
import contextlib
import warnings

# Suppress specific warnings from torchmetrics for lpips
warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")
warnings.filterwarnings("ignore", message="Arguments other than a weight enum or `None` for 'weights' are deprecated")


@contextlib.contextmanager
def timer(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"{label}: {end - start:.6f} seconds")

device = torch.device("cuda:0")

def evaluate(dataset: ModelParams, mode: str, iteration: int , pipeline : PipelineParams, speedtest=False):
    # Set up metrics
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)

    gaussians = GaussianModel()
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    # Path 
    scene_dir = os.path.join(dataset.model_path, "ours_{}".format(scene.loaded_iter))
    makedirs(scene_dir, exist_ok=True)
     
    views = None 
    if mode == "train":
        views = scene.getTrainCameras()
    elif mode == "test":
        views = scene.getTestCameras()

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    dataloader = DataLoader(views, shuffle=False, num_workers=4, batch_size=50, prefetch_factor=2,  persistent_workers=True, pin_memory=True)
    
    ssims = []
    psnrs = []
    lpipss = []

    ssim_sum, psnr_sum, lpips_sum, total_samples = 0, 0, 0, 0

    per_view_dict = {}

    data_iterator = tqdm(dataloader, desc="Evaluating cameras")
    for camera in data_iterator:  
        batch_size = len(camera["uid"])
    
        # Iterate over each item in the batch
        item_iterator = tqdm(range(batch_size), desc="Processing items", leave=False)
        for i in item_iterator: # TODO: This is not efficient but the code is not designed to handle batched rendering
            # Construct a dictionary for the current item by extracting the ith element from each key
            view = {key: value[i] if not isinstance(value, torch.Tensor) else value[i].cuda() for key, value in camera.items()}
            filename = view["image_name"]

            maps = render(view, gaussians, pipeline, background, evaluate=True)

            if speedtest:
                continue

            gt_img = view["original_image"][0:3, :, :].clamp(0, 1)
            render_img = maps["render"].clamp(0, 1)

            gt_img = gt_img.unsqueeze(0)
            render_img = render_img.unsqueeze(0)

            # Compute metrics
            current_ssim = ssim(render_img, gt_img).item()
            current_psnr = psnr(render_img, gt_img).item()
            current_lpips = lpips(render_img, gt_img).item()

            # Update sums and counts
            ssim_sum += current_ssim
            psnr_sum += current_psnr
            lpips_sum += current_lpips
            total_samples += 1

            # Append to lists
            ssims.append(current_ssim)
            psnrs.append(current_psnr)
            lpipss.append(current_lpips)

            data_iterator.set_description(
                "SSIM: {:.4f}, PSNR: {:.4f}, LPIPS: {:.4f}, SAMPLES: {:d}".format(
                    ssim_sum / total_samples,
                    psnr_sum / total_samples,
                    lpips_sum / total_samples,
                    total_samples
                )
            )

            # Append to lists with filename as key
            if filename not in per_view_dict:  # Initialize if not exists
                per_view_dict[filename] = {}
                per_view_dict[filename]["SSIM"] = current_ssim
                per_view_dict[filename]["PSNR"] = current_psnr
                per_view_dict[filename]["LPIPS"] = current_lpips

    ssim_mean = torch.tensor(ssims).mean().item()
    psnr_mean = torch.tensor(psnrs).mean().item()
    lpips_mean = torch.tensor(lpipss).mean().item()

    print("  SSIM : {:>12.7f}".format(ssim_mean))
    print("  PSNR : {:>12.7f}".format(psnr_mean))
    print("  LPIPS: {:>12.7f}".format(lpips_mean))

    full_dict={"SSIM": ssim_mean, "PSNR": psnr_mean, "LPIPS": lpips_mean}

    with open(os.path.join(scene_dir, f"evaluation_{mode}.json"), 'w') as fp:
        json.dump(full_dict, fp, indent=True)
    with open(os.path.join(scene_dir, f"per_view_evaluation_{mode}.json"), 'w') as fp:
        json.dump(per_view_dict, fp, indent=True)

if __name__ == "__main__":
    parser = ArgumentParser(description="Script for evaluating image quality metrics")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--mode', '-mode', required=True, type=str, choices=["train", "test", "both"])
    parser.add_argument('--batch_size', '-b', type=int, default=10)
    parser.add_argument('--speedtest', action='store_true')

    args = get_combined_args(parser)
    print("Evaluating " + args.model_path)

    if args.mode == "both":
        for mode in ["train", "test"]:
            with torch.no_grad():
                evaluate(model.extract(args), mode, args.iteration, pipeline.extract(args), args.speedtest)
    else:
        with torch.no_grad():
            evaluate(model.extract(args), args.mode, args.iteration, pipeline.extract(args), args.speedtest)

