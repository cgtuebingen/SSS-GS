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

class ImageDataset(Dataset):
    def __init__(self, gt_dir, renders_dir):
        self.gt_images = sorted([os.path.join(gt_dir, file) for file in os.listdir(gt_dir)])
        self.render_images = sorted([os.path.join(renders_dir, file) for file in os.listdir(renders_dir)])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.gt_images)

    def __getitem__(self, idx):
        gt_image = self.transform(Image.open(self.gt_images[idx]).convert("RGB"))
        render_image = self.transform(Image.open(self.render_images[idx]).convert("RGB"))
        filename = os.path.basename(self.gt_images[idx]).split(".")[0]
        return gt_image, render_image, filename

def evaluate(model_paths, mode, batch_size):
    # Set up metrics
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)

    for scene_dir in model_paths:
        print("Scene:", scene_dir)

        full_dict = {}
        per_view_dict = {}

        mode_dir = Path(scene_dir) / mode

        for method in os.listdir(mode_dir):
            print("Method:", method)

            method_dir = mode_dir / method
            gt_dir = method_dir / "gt"
            renders_dir = method_dir / "renders"

            dataset = ImageDataset(gt_dir, renders_dir)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

            ssims = []
            psnrs = []
            lpipss = []

            ssim_sum, psnr_sum, lpips_sum, total_samples = 0, 0, 0, 0

            if method not in per_view_dict:
                per_view_dict[method] = {}

            with torch.no_grad():
                # Create a tqdm iterator with an initial description
                data_iterator = tqdm(dataloader, desc="Computing metrics")
                for gt_images, render_images, filenames in data_iterator:
                    gt_images, render_images = gt_images.to(device), render_images.to(device)

                    for gt_img, render_img, filename in zip(gt_images, render_images, filenames):
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
                        if filename not in per_view_dict[method]:  # Initialize if not exists
                            per_view_dict[method][filename] = {}
                        per_view_dict[method][filename]["SSIM"] = current_ssim
                        per_view_dict[method][filename]["PSNR"] = current_psnr
                        per_view_dict[method][filename]["LPIPS"] = current_lpips

            ssim_mean = torch.tensor(ssims).mean().item()
            psnr_mean = torch.tensor(psnrs).mean().item()
            lpips_mean = torch.tensor(lpipss).mean().item()

            print("  SSIM : {:>12.7f}".format(ssim_mean))
            print("  PSNR : {:>12.7f}".format(psnr_mean))
            print("  LPIPS: {:>12.7f}".format(lpips_mean))

            full_dict[method] = {"SSIM": ssim_mean, "PSNR": psnr_mean, "LPIPS": lpips_mean}

            with open(os.path.join(scene_dir, f"results_{method}_{mode}.json"), 'w') as fp:
                json.dump(full_dict[method], fp, indent=True)
            with open(os.path.join(scene_dir, f"per_view_results_{method}_{mode}.json"), 'w') as fp:
                json.dump(per_view_dict[method], fp, indent=True)

if __name__ == "__main__":
    parser = ArgumentParser(description="Script for evaluating image quality metrics")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str)
    parser.add_argument('--mode', '-mode', required=True, type=str, choices=["train", "test"])
    parser.add_argument('--batch_size', '-b', type=int, default=10)

    args = parser.parse_args()
    evaluate(args.model_paths, args.mode, args.batch_size)
