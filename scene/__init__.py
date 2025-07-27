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

import os
import random
import json
from PIL import Image
from torch import load, save
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.dataset import CameraDataset
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], custom_transforms_file=None, uses_gt_images=True):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.train_datasets = {}
        self.test_datasets = {}

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))


        if args.alternative_transforms_path is not None:
            scene_info = sceneLoadTypeCallbacks["Blender"](args.alternative_transforms_path, args.white_background, args.eval, custom_transforms_file=custom_transforms_file, uses_gt_images=uses_gt_images)
        else:
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, custom_transforms_file=custom_transforms_file, uses_gt_images=uses_gt_images)

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []

            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            
            if custom_transforms_file is  None:
                for id, cam in enumerate(camlist):
                    json_cams.append(camera_to_JSON(id, cam))
                with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                    json.dump(json_cams, file)


        # if shuffle:
        #     random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
        #    random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Creating Training Datasets")
            self.train_datasets[resolution_scale] = CameraDataset(scene_info.train_cameras, resolution_scale, args, uses_gt_images=uses_gt_images)

            if args.eval and custom_transforms_file == None: 
                print("Creating Test Datasets")
                self.test_datasets[resolution_scale] = CameraDataset(scene_info.test_cameras, resolution_scale, args, uses_gt_images=uses_gt_images)
            else:
                self.test_datasets[resolution_scale] = None

        if self.loaded_iter:
            print("Loading Model ...")
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians._sss = load(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "mlp_model.pt"))
            print("Loaded model at iteration {}".format(self.loaded_iter))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        save(self.gaussians._sss, os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration), "mlp_model.pt"))
        
    def getTrainCameras(self, scale=1.0):
        return self.train_datasets[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_datasets[scale]