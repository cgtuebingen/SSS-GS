""" Dataset. """
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from arguments import ModelParams
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixCenterShift, fov2focal, focal2fov


class CameraDataset(Dataset):
    def __init__(self, cam_infos, resolution_scale=1.0, args: ModelParams = None, uses_gt_images=True):
        self.args = args
        self.cam_infos = cam_infos
        self.use_gt_images = uses_gt_images
        
        if not uses_gt_images:
            print("[ INFO ] Using synthetic images, no ground truth images will be loaded")
            return

        first_image = Image.open(cam_infos[0].image_path)
        orig_w, orig_h = first_image.size # Assuming all images have the same size
        self.original_resolution = (orig_w, orig_h)

        self.scale = 1.0
        self.resolution = None
        if args.resolution in [1, 2, 4, 8]:
            self.resolution = round(orig_h/(resolution_scale * args.resolution)), round(orig_w/(resolution_scale * args.resolution))
        else:  # should be a type that converts to float
            if args.resolution == -1:
                if orig_w > 1600:
                    global WARNED
                    if not WARNED:
                        print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                            "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                        WARNED = True
                    global_down = orig_w / 1600
                else:
                    global_down = 1
            else:
                global_down = orig_w / args.resolution

            self.scale = float(global_down) * float(resolution_scale)
            self.resolution = (int(orig_h / self.scale), int(orig_w / self.scale))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.resolution)
        ])

    def __len__(self):
        return len(self.cam_infos)

    def __getitem__(self, idx):
        cam_info = self.cam_infos[idx]

        if self.use_gt_images:
            gt_image = Image.open(cam_info.image_path)
            loaded_mask = None


            if gt_image.mode == "RGBA":
                loaded_mask = gt_image.split()[3]
                loaded_mask = self.transform(loaded_mask)
                loaded_mask = loaded_mask.repeat(3, 1, 1)

            if gt_image.size != self.original_resolution:
                raise ValueError(f"Image {cam_info.image_path} has size {gt_image.size} but expected {self.original_resolution}, please check your input images, all images must have the same resolution.")

            gt_image = self.transform(gt_image.convert("RGB"))
            original_image = gt_image.clamp(0.0, 1.0)
            image_width = original_image.shape[2]
            image_height = original_image.shape[1]

            if loaded_mask is not None:
                original_image *= loaded_mask
            else:
                original_image *= torch.ones((1, image_height, image_width))

        else:
            image_width = cam_info.width
            image_height = cam_info.height
            loaded_mask = torch.ones((3, image_height, image_width))
            original_image = torch.zeros((3, image_height, image_width))
        
        znear = 0.01
        zfar = 100.0

        trans = np.array([0.0, 0.0, 0.0])
        scale = 1.0 # FIXME: This is hardcoded, should be a parameter

        world_view_transform = torch.tensor(getWorld2View2(cam_info.R, cam_info.T, trans, scale)).transpose(0, 1)
        
        if cam_info.fy is None:
            projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=cam_info.FovX, fovY=cam_info.FovY).transpose(0, 1)
        else:
            projection_matrix = getProjectionMatrixCenterShift(znear, zfar, cam_info.cx, cam_info.cy, cam_info.fx, cam_info.fy, image_width, image_height).transpose(0, 1)


        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        light_position = torch.tensor(cam_info.light_position).float()

        if cam_info.light_intensity is not None:
            light_intensity = torch.tensor(cam_info.light_intensity).float()
        else:
            light_intensity = torch.tensor([1.0, 1.0, 1.0]).float()

        # Correct coordinate sytem X,Y,Z => X,-Y,-Z
        # FIXME: This might be a Blender specific fix, should be generalized
        light_position[1] = -light_position[1]
        light_position[2] = -light_position[2]

        scale_cx = cam_info.cx
        scale_cy = cam_info.cy
        scale_fx = cam_info.fx
        scale_fy = cam_info.fy

        if cam_info.cx is not None and cam_info.cy is not None:
            scale_cx /= scale
            scale_cy /= scale
        else:
            scale_cx = (image_width * 0.5) / scale
            scale_cy = (image_height * 0.5) / scale

        if cam_info.fx is not None and cam_info.fy is not None:    
            scale_fx /= scale
            scale_fy /= scale
        else:
            scale_fx = fov2focal(cam_info.FovX, image_width) / scale
            scale_fy = fov2focal(cam_info.FovY, image_height) / scale

        camera = {
            "uid": idx, 
            "colmap_id": cam_info.uid,
            "R": cam_info.R,
            "T": cam_info.T,
            "FoVx": cam_info.FovX,
            "FoVy": cam_info.FovY,
            "fx": scale_fx,
            "fy": scale_fy,
            "cx": scale_cx,
            "cy": scale_cy,
            "image_name": cam_info.image_name,
            "original_image": original_image,
            "image_mask": loaded_mask,
            "image_width": image_width,
            "image_height": image_height,
            "znear": znear,
            "zfar": zfar,
            "scale": scale,
            "world_view_transform": world_view_transform,
            "projection_matrix": projection_matrix,
            "full_proj_transform": full_proj_transform,
            "camera_center": camera_center,
            "light_position": light_position,
            "light_intensity": light_intensity,
        }

        return camera