import os
import sys
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    light_position: np.array
    light_intensity: np.array = None
    FovY: np.array = None
    FovX: np.array = None
    fx: np.array = None
    fy: np.array = None
    cx: np.array = None
    cy: np.array = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", uses_gt_images=True): 
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        # Check if an exlude_list.txt file exists
        exclude_list = []
        exclude_file = os.path.join(path, "exclude_list.txt")
        if os.path.exists(exclude_file):
            with open(exclude_file, "r") as f:
                exclude_list = f.read().splitlines()
                exclude_list = [x.replace(".png", "") for x in exclude_list]

        excluded_images = 0

        
        fovx = None
        if "camera_angle_x" in contents:
            fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            
            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            
            # Background color
            # bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
            # file_path = frame["file_path"]   


            # Load image width and height
            if "width" not in frame or "height" not in frame:
                raise ValueError("Width and height must be part of the frame, this is a new addition to save time by not loading all images.")

            width = frame["width"]
            height = frame["height"]

            # Focal Lenght
            if "fl_x" in frame and "fl_y" in frame:
                fl_x = frame["fl_x"]
                fl_y = frame["fl_y"]

            # Principal Point
            cx, cy = None, None
            if "cx" in frame and "cy" in frame:
                cx = frame["cx"]
                cy = frame["cy"]

            if fovx is not None:
                FovY = focal2fov(fov2focal(fovx, width), height)
                FovX = fovx
            else:
                FovY = focal2fov(fl_y, height)
                FovX = focal2fov(fl_x, width)

            light_positions = frame["light_positions"]

            light_intensities = None
            if "light_intensities" in frame:
                light_intensities = frame["light_intensities"]

            for light_idx, light_position in enumerate(light_positions):
                light_intensity = None
                if light_intensities is not None:
                    light_intensity = light_intensities[light_idx]

                image_path = ""
                image_name = ""
                if uses_gt_images:
                    file_path = frame["file_paths"][light_idx]
                
                    if file_path.split("/")[-1] in exclude_list:
                        excluded_images += 1
                        continue

                    image_path = os.path.join(path, file_path + extension)
                    image_name = Path(file_path).stem

                light_position = np.array(light_position)

                # Convert OpenGL/Blender position to COLMAP position
                light_position[1] *= -1  # Flip Y axis
                light_position[2] *= -1  # Flip Z axis

                # image = Image.open(image_path)
                # im_data = np.array(image.convert("RGBA"))

                # norm_data = im_data / 255.0
                # arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

                

                cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, width=width, height=height, cx=cx, cy=cy,
                    image_path=image_path, image_name=image_name, light_position=light_position, light_intensity=light_intensity))

    return cam_infos, (excluded_images, len(exclude_list))

def readBlenderDatasetInfo(path, white_background, eval, extension=".png", custom_transforms_file=None, uses_gt_images=True):

    if custom_transforms_file is not None:
        print("Reading Custom Transforms")
        train_cam_infos, _ = readCamerasFromTransforms(path, custom_transforms_file, white_background, extension, uses_gt_images)
        test_cam_infos = []
    else: 
        print("Reading Training Transforms")
        train_cam_infos, excluded_train = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, uses_gt_images)
        print("Reading Test Transforms")
        test_cam_infos, excluded_test = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, uses_gt_images)
    
        excluded = (excluded_train[0] + excluded_test[0], excluded_train[1])
        if excluded[1] > 0:
            print(f"Excluded {excluded[0]} images from {excluded[1]} marked images from the dataset.")

        if not eval:
            train_cam_infos.extend(test_cam_infos)
            test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path) and custom_transforms_file is None:
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Blender" : readBlenderDatasetInfo
}