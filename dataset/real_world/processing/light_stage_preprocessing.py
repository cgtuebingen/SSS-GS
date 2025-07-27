"""preprocessing pipeline for light stage data

it includes the following steps:

- colmap reconstruction
- matting
- copy mattes for all light directions

Determine mask prompt (or use points).

Determine scaling factor for colmap scene.


example:
python light_stage_preprocessing.py --input_dir /ceph/datasets/sss_light_stage/red_car_2/ --output_dir /ceph/datasets/sss_light_stage/preprocessed/red_car_2/ --colmap_path ~/.local/bin/bin/colmap --prompt "toy car" --shift_scene_z -0.5 --demosaic --gpu 0, 

python light_stage_preprocessing.py --input_dir /ceph/datasets/sss_light_stage/red_car_2/ --output_dir /ceph/datasets/sss_light_stage/preprocessed/red_car_2/ --colmap_path ~/.local/bin/bin/colmap --prompt "toy car" --shift_scene_z -0.5 --demosaic --gpu 0

car 2nd stage:
--light_stage_reconstructed_distance 0.931582728913541 --stage_2

candle green 2nd stage:
--light_stage_reconstructed_distance 0.9343099479915645 --stage_2

E.g. output /ceph/datasets/sss_light_stage/preprocessed/red_car_2

# Marmelade
python light_stage_preprocessing.py --input_dir /ceph/datasets/sss_light_stage/marmelade/ --output_dir /ceph/datasets/sss_light_stage/preprocessed/marmelade/ --colmap_path ~/.local/bin/bin/colmap --prompt "marmelade jar" --shift_scene_z -0.5 --demosaic --gpu 1



"""

import argparse
import glob
import json
import multiprocessing
import os
import shutil
import subprocess
from typing import List, Optional, Tuple, Union

import cv2
import imageio.v2 as imageio
import numpy as np
import pycolmap
import tqdm
import utils.reflectance_utils
from colour_demosaicing import (
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_DDFAPD,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="path to the input directory containing the light stage data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="path to the output directory where the preprocessed data will be stored. Subdirectories for colmap reconstruction and mattes will be created.",
    )
    parser.add_argument(
        "--colmap_path",
        type=str,
    )
    parser.add_argument(
        "--matte_anything_path",
        default="~/projects/gsss/Matte-Anything",
        type=str,
        help="Path to Matte Anything project.",
    )
    parser.add_argument("--gpu", default=0, type=int, help="GPU to use.")
    parser.add_argument(
        "--high_quality",
        default=False,
        action="store_true",
        help="Use high quality settings for colmap reconstruction.",
    )
    parser.add_argument(
        "--invert_matte",
        action="store_true",
        help="Invert the matte from Matte Anything.",
    )
    parser.add_argument(
        "--downscale_factor",
        default="1",
        type=str,
        help="Downscale factor for Matte Anything.",
    )
    parser.add_argument(
        "--prompt", default="3D object", type=str, help="Prompt for Matte Anything."
    )
    parser.add_argument(
        "--light_stage_reconstructed_distance",
        default=-1.0,
        type=float,
        help="Distance of the box side with colored markers as scaling reference.",
    )
    parser.add_argument(
        "--resize_to",
        default=800,
        type=int,
        help="Resize images to this resolution on the longest side.",
    )
    parser.add_argument(
        "--global_scale",
        type=float,
        default=3.0,
        help="Global scale of scene.",
    )
    parser.add_argument(
        "--shift_scene_z",
        type=float,
        default=0.0,
        help="Shift scene along z-axis. (this is most likely the height in real world.)",
    )
    parser.add_argument(
        "--shift_scene_y",
        type=float,
        default=0.0,
        help="Shift scene along y-axis.",
    )
    parser.add_argument(
        "--demosaic",
        action="store_true",
        help="Demosaic input images.",
    )
    parser.add_argument(
        "--stage_2",
        action="store_true",
        help="Run stage 2 of the pipeline including light alignment.",
    )
    parser.add_argument(
        "--light_stage_calibration_path",
        type=str,
        default="",
        help="Path to the light stage calibration file.",
    )
    parser.add_argument(
        "--all_on_dataset",
        action="store_true",
        help="Use all_on images for dataset.",
    )
    parser.add_argument(
        "--flare_threshold",
        type=float,
        default=0.2,
        help="Threshold for flare detection. Ratio of image pixels.",
    )
    return parser.parse_args()


def check_output_dir(output_dir):
    if os.path.exists(output_dir):
        return True
    else:
        return False


def manual_colmap_pipeline(input_dir, output_dir, colmap_bin=None, high_quality=False):
    """
    Manually perform steps for automatic colmap sparse reconstruction.

    """
    colmap_dir = os.path.join(output_dir, "colmap")

    if check_output_dir(colmap_dir):
        print(f"INFO: Found existing colmap reconstruction at {colmap_dir}. Skipping.")
        return colmap_dir

    os.makedirs(colmap_dir, exist_ok=True)
    if colmap_bin is None:
        colmap_bin = shutil.which("colmap")
        if colmap_bin is None:
            raise ValueError("colmap binary not found in PATH")
    log_file = os.path.join(colmap_dir, "colmap.log")

    # Feature extraction.
    database_path = os.path.join(colmap_dir, "database.db")

    print("Running colmap, look at log file for details.", log_file)

    with open(log_file, "w") as f:
        if high_quality:
            subprocess.run(
                [
                    " ".join(
                        [
                            colmap_bin,
                            "feature_extractor",
                            "--database_path",
                            database_path,
                            "--image_path",
                            input_dir,
                            "--ImageReader.camera_model OPENCV",
                            "--ImageReader.single_camera 1",
                            "--SiftExtraction.estimate_affine_shape 1",
                            "--SiftExtraction.domain_size_pooling 1",
                            "--SiftMatching.guided_matching 1",
                            "--SiftExtraction.max_num_features 20000",
                            "--tri_ignore_two_view_tracks 0",
                        ]
                    )
                ],
                shell=True,
                stdout=f,
                stderr=subprocess.STDOUT,
            )

        else:
            subprocess.run(
                [
                    " ".join(
                        [
                            colmap_bin,
                            "feature_extractor",
                            "--database_path",
                            database_path,
                            "--image_path",
                            input_dir,
                            "--ImageReader.camera_model OPENCV",
                            "--ImageReader.single_camera 1",
                        ]
                    )
                ],
                shell=True,
                stdout=f,
                stderr=subprocess.STDOUT,
            )

    with open(log_file, "a") as f:
        subprocess.run(
            [
                " ".join(
                    [colmap_bin, "exhaustive_matcher", "--database_path", database_path]
                )
            ],
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    os.makedirs(os.path.join(colmap_dir, "sparse"), exist_ok=True)
    with open(log_file, "a") as f:
        subprocess.run(
            [
                " ".join(
                    [
                        colmap_bin,
                        "mapper",
                        "--database_path",
                        database_path,
                        "--image_path",
                        input_dir,
                        "--output_path",
                        os.path.join(colmap_dir, "sparse"),
                    ]
                )
            ],
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    # Verify that the reconstruction was successful.
    if not os.path.exists(os.path.join(colmap_dir, "sparse", "0")):
        raise ValueError("colmap reconstruction failed")

    return colmap_dir


def colmap_reconstruction(input_dir, output_dir, colmap_bin=None, high_quality=False):
    colmap_dir = os.path.join(output_dir, "colmap")

    if check_output_dir(colmap_dir):
        print(f"INFO: Found existing colmap reconstruction at {colmap_dir}. Skipping.")
        return colmap_dir

    os.makedirs(colmap_dir, exist_ok=True)
    if colmap_bin is None:
        colmap_bin = shutil.which("colmap")
        if colmap_bin is None:
            raise ValueError("colmap binary not found in PATH")
    log_file = os.path.join(colmap_dir, "colmap.log")
    with open(log_file, "w") as f:
        subprocess.run(
            [
                " ".join(
                    [
                        colmap_bin,
                        "automatic_reconstructor",
                        "--image_path",
                        input_dir,
                        "--workspace_path",
                        colmap_dir,
                        "--quality extreme" if high_quality else "",
                        "--single_camera 1",  # shared intrinsics
                    ]
                )
            ],
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    # Verify that the reconstruction was successful.
    if not os.path.exists(os.path.join(colmap_dir, "sparse", "0")):
        raise ValueError("colmap reconstruction failed")

    # Todo check logfile for potential errors.
    with open(log_file, "r") as f:
        log_content = f.read()
        if "failed" in log_content.lower():
            raise ValueError("Colmap reconstruction failed for some images.")

    return colmap_dir


def copy_all_on_images(input_dir, output_dir, ext="png"):
    all_on_dir = os.path.join(output_dir, "all_on_images")
    if check_output_dir(all_on_dir):
        print(f"INFO: Found existing image folder at {all_on_dir}. Skipping.")
        return all_on_dir
    os.makedirs(all_on_dir, exist_ok=True)
    for img_path in glob.glob(os.path.join(input_dir, f"*all_on.{ext}")):
        shutil.copy2(img_path, all_on_dir)
    return all_on_dir


def generate_mattes(
    input_dir,
    output_dir,
    matte_anything_path,
    prompt="3D object",
    downscale_factor="1",
    invert_matte: bool = False,
):
    matte_dir = os.path.join(output_dir, "matte")
    if check_output_dir(os.path.join(matte_dir, "alpha")):
        print(f"INFO: Found existing mattes at {matte_dir}. Skipping.")
        return matte_dir
    os.makedirs(matte_dir, exist_ok=True)
    # Run Matte Anything for all images in the input directory.

    log_file = os.path.join(matte_dir, "matte_anything.log")
    with open(log_file, "w") as f:
        subprocess.run(
            [
                " ".join(
                    [
                        f"cd {matte_anything_path} &&",
                        "python",
                        os.path.join(matte_anything_path, "matte_image_folder.py"),
                        "--input_folder",
                        input_dir,
                        "--output_folder",
                        matte_dir,
                        "--apply_mask",
                        "--prompt",
                        f'"{prompt}"',
                        "--downscale",
                        downscale_factor,
                        "--invert" if invert_matte else "",
                    ]
                )
            ],
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    # Verify that the mattes were generated successfully.
    if not os.path.exists(os.path.join(matte_dir, "alpha")):
        raise ValueError("Matte Anything failed")

    return matte_dir


def apply_matte_to_image(img_path, matte_path, output_path):
    image_np = imageio.imread(img_path) / 255.0
    mask_np = imageio.imread(matte_path) / 255.0

    # Apply the mask to the image
    masked_image_np = np.zeros_like(image_np)
    if len(mask_np.shape) == 2 and len(image_np.shape) == 3:
        mask_np = mask_np[..., None]
    masked_image_np = image_np * mask_np + (1 - mask_np) * masked_image_np

    # Save the masked image
    # Add alpha channel.
    masked_image_np = np.concatenate([masked_image_np, mask_np], axis=-1)
    imageio.imwrite(output_path, (masked_image_np * 255).astype(np.uint8))


def copy_and_apply_matte(
    skip_all_on, img_path, ext, matte_dir, light_dir, comp_dir=None, comp_out_dir=None
):
    if skip_all_on and img_path.endswith(f"all_on.{ext}"):
        return

    basename = os.path.basename(img_path)
    file_name, ext = os.path.splitext(basename)
    file_name = "_".join(file_name.split("_")[:4])
    source_path = glob.glob(os.path.join(matte_dir, f"{file_name}*"))
    if len(source_path) == 0 or len(source_path) > 1:
        print("WARNING: No matching matte found for", img_path)
        return

    if not os.path.exists(os.path.join(light_dir, basename)):
        shutil.copy2(source_path[0], os.path.join(light_dir, basename))

    if img_path.endswith(f"all_on.{ext}"):
        if not os.path.exists(os.path.join(comp_out_dir, basename)):
            shutil.copy2(
                os.path.join(comp_dir, basename), os.path.join(comp_out_dir, basename)
            )

    if comp_dir is not None and comp_out_dir is not None:
        if not os.path.exists(os.path.join(comp_out_dir, basename)):
            apply_matte_to_image(
                img_path,
                source_path[0],
                os.path.join(comp_out_dir, basename),
            )


def copy_matte_to_all_light_dirs(
    matte_dir, output_dir, input_dir, ext="png", skip_all_on=True, comp_dir=None
):
    light_dir = output_dir
    # if check_output_dir(light_dir):
    #     print(f"INFO: Found existing mattes at {light_dir}. Skipping.")
    #     return
    os.makedirs(light_dir, exist_ok=True)

    if comp_dir is not None:
        comp_out_dir = os.path.join(os.path.dirname(output_dir.rstrip("/")), "masked")
        os.makedirs(comp_out_dir, exist_ok=True)

    with multiprocessing.Pool(processes=16) as pool:
        for _ in list(
            tqdm.tqdm(
                pool.starmap(
                    copy_and_apply_matte,
                    [
                        (
                            skip_all_on,
                            img_path,
                            ext,
                            matte_dir,
                            light_dir,
                            comp_dir,
                            comp_out_dir,
                        )
                        for img_path in glob.glob(os.path.join(input_dir, f"*.{ext}"))
                    ],
                ),
                total=len(glob.glob(os.path.join(input_dir, f"*.{ext}"))),
            )
        ):
            pass
    # for img_path in glob.glob(os.path.join(input_dir, f"*.{ext}")):
    #     copy_and_apply_matte(skip_all_on, img_path, ext, matte_dir, light_dir, comp_dir)


def read_image_gammacorrect(img_path, gamma=2.2):
    img = imageio.imread(img_path)
    img = img.astype(np.float32) / 255.0
    img = img**gamma
    return img


def write_image_gammacorrect(img, img_path, gamma=2.2):
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    img = img ** (1 / gamma)
    img = (img * 255.0).astype(np.uint8)
    imageio.imwrite(img_path, img)


def read_demosaic_save(in_path, out_path, dark_frame=None):
    # img = imageio.imread(in_path)
    # img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
    img = read_image_gammacorrect(in_path)
    if len(img.shape) > 2 and img.shape[-1] > 1:
        img = img[..., 0]

    if dark_frame is not None:
        assert (
            img.dtype == dark_frame.dtype
        ), "Dark frame and image have different types."
        if img.dtype == np.uint8:
            img = img.astype(np.float64) / 255.0
            dark_frame = dark_frame.astype(np.float64) / 255.0
        img = np.clip(img - dark_frame, 0, 1)

    if img.dtype != np.uint8:
        img = (img * 255.0).astype(np.uint8)

    # rgb_img = cv.cvtColor(img, cv.COLOR_BayerRG2RGB_MHT)
    # img_demosaiced = demosaicing_CFA_Bayer_Malvar2004(CFA=img, pattern="RGGB")
    # img = np.clip(img, 0, 1)
    # img_demosaiced = demosaicing_CFA_Bayer_Malvar2004(CFA=img, pattern="RGGB")
    # img = img.astype(np.float64) / 255.0
    # img_demosaiced = demosaicing_CFA_Bayer_bilinear(CFA=img, pattern="RGGB")
    # img_demosaiced = demosaicing_CFA_Bayer_DDFAPD(CFA=img, pattern="RGGB")
    # img_demosaiced = img_demosaiced * 255.0
    img_demosaiced = cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR_VNG)

    write_image_gammacorrect(img_demosaiced, out_path)
    return


def demosaic_images(image_dir, output_dir):
    # Demosaic images.
    demosaiced_dir = os.path.join(output_dir, "demosaiced")
    if check_output_dir(demosaiced_dir):
        print(f"INFO: Found existing images at {demosaiced_dir}. Skipping.")
        return demosaiced_dir
    os.makedirs(demosaiced_dir, exist_ok=True)
    img_paths = []
    for ext in [".png", ".jpg", ".jpeg"]:
        img_paths.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))

    dark_frame = glob.glob(os.path.join(image_dir, "*dark*"))
    if len(dark_frame) == 0:
        dark_frame = None
    else:
        dark_frame = dark_frame[0]
        img_paths.remove(dark_frame)
        dark_frame = read_image_gammacorrect(dark_frame)

    # Demosaic in parallel.
    print("Demosacing images...")

    # for img_path in tqdm.tqdm(img_paths):
    #     read_demosaic_save(
    #         img_path,
    #         os.path.join(demosaiced_dir, os.path.basename(img_path)),
    #         dark_frame,
    #     )
    with multiprocessing.Pool() as pool:
        for _ in list(
            tqdm.tqdm(
                pool.starmap(
                    read_demosaic_save,
                    [
                        (
                            in_path,
                            os.path.join(demosaiced_dir, os.path.basename(in_path)),
                            dark_frame,
                        )
                        for in_path in img_paths
                    ],
                ),
                total=len(img_paths),
            )
        ):
            pass

    return demosaiced_dir


def get_best_colmap_reconstruction(colmap_dir: str) -> str:
    # Get the best reconstruction from the colmap directory.
    reconstructions = glob.glob(os.path.join(colmap_dir, "sparse", "*"))
    if len(reconstructions) == 0:
        raise ValueError("No colmap reconstruction found.")
    if len(reconstructions) == 1:
        return reconstructions[0]
    # Find the best reconstruction.
    best_reconstruction = None
    num_views_registered = 0
    for reconstruction in reconstructions:
        cm_reconstruction = pycolmap.Reconstruction(reconstruction)
        if num_views_registered < len(cm_reconstruction.images):
            best_reconstruction = reconstruction
            num_views_registered = len(cm_reconstruction.images)

    return best_reconstruction


def undistort_opencv(
    img_path, output_dir, camera_matrix, camera_matrix_new, dist_coeffs, new_image_size
):
    # Undistort images using OpenCV.
    if os.path.exists(os.path.join(output_dir, os.path.basename(img_path))):
        return

    # print("camera matrix", camera_matrix)

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("WARN: Could not load image", img_path)
        return
    h, w = img.shape[:2]
    # print(h, w)
    # new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    #     camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    #     # camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    # )
    new_camera_matrix = camera_matrix_new
    # print(new_camera_matrix)

    mapx, mapy = cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix,
        (new_image_size[1], new_image_size[0]),
        cv2.CV_32FC1,
    )
    img_undistorted = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # img_undistorted = cv2.undistort(
    #     img, camera_matrix, dist_coeffs, None, new_camera_matrix
    # )

    # x, y, w, h = roi
    # img_undistorted = img_undistorted[y : y + h, x : x + w]
    cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), img_undistorted)


def undistort_images(image_dir, output_dir, colmap_dir, colmap_bin=None):
    # Undistort images.
    if colmap_bin is None:
        colmap_bin = shutil.which("colmap")
        if colmap_bin is None:
            raise ValueError("colmap binary not found in PATH")
    undistorted_dir = os.path.join(output_dir, "undistorted")
    colmap_reconstruction_dir = get_best_colmap_reconstruction(colmap_dir)
    if check_output_dir(undistorted_dir):
        print(f"INFO: Found existing undistorted at {undistorted_dir}. Skipping.")
    else:
        os.makedirs(undistorted_dir, exist_ok=True)
        log_file = os.path.join(colmap_dir, "undistort.log")

        print(
            "Undistorting images using colmap reconstruction from",
            colmap_reconstruction_dir,
        )

        with open(log_file, "a") as f:
            subprocess.run(
                [
                    " ".join(
                        [
                            colmap_bin,
                            "image_undistorter",
                            "--image_path",
                            image_dir,
                            "--input_path",
                            colmap_reconstruction_dir,
                            "--output_path",
                            undistorted_dir,
                            "--output_type",
                            "COLMAP",
                        ]
                    )
                ],
                shell=True,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
    # Now also undistort the additional images using the same intrinsics.

    print("Undistorting additional images...")
    # Get intrinsics:
    reconstruction = pycolmap.Reconstruction(colmap_reconstruction_dir)
    images = reconstruction.images
    image_v = [v for k, v in images.items()][0]
    camera = reconstruction.cameras[image_v.camera_id]
    intrinsics = camera.calibration_matrix()

    rec_undistorted = pycolmap.Reconstruction(os.path.join(undistorted_dir, "sparse"))
    camera2 = rec_undistorted.cameras[image_v.camera_id]
    new_intrinsics = camera2.calibration_matrix()

    new_size = imageio.imread(
        glob.glob(os.path.join(undistorted_dir, "images", "theta*.png"))[0]
    ).shape[:2]

    # All parameters.
    params = camera.params[4:]

    with multiprocessing.Pool() as pool:
        pool.starmap(
            undistort_opencv,
            [
                (
                    img_path,
                    os.path.join(undistorted_dir, "images"),
                    intrinsics,
                    new_intrinsics,
                    params,
                    new_size,
                )
                for img_path in glob.glob(os.path.join(image_dir, "*"))
            ],
        )

    return undistorted_dir


def convert_to_3dgs_format(colmap_dir, output_dir, image_dir: str = "masked"):
    # Convert colmap reconstruction to 3DGS format.
    # sparse_dir = os.path.join("sparse", "0")
    sparse_dir = "sparse"

    dataset_dir = os.path.join(output_dir, "dataset")
    if check_output_dir(dataset_dir):
        print(f"INFO: Found existing dataset at {dataset_dir}. Skipping.")
        return dataset_dir
    print("Creating 3DGS style dataset folder...")
    os.makedirs(os.path.join(output_dir, "dataset", "images"), exist_ok=True)
    target_path = dataset_dir
    src_path_images = os.path.abspath(os.path.join(output_dir, image_dir))
    src_path_colmap = os.path.abspath(os.path.join(colmap_dir))

    # Symlink images.
    for img_path in glob.glob(os.path.join(src_path_images, "*")):
        os.symlink(
            img_path, os.path.join(target_path, "images", os.path.basename(img_path))
        )

    # Symlink sparse data.
    os.makedirs(os.path.join(target_path, sparse_dir), exist_ok=True)
    for file in os.listdir(os.path.join(src_path_colmap, sparse_dir)):
        if file == "0":
            continue
        source_file = os.path.join(src_path_colmap, sparse_dir, file)
        destination_file = os.path.join(target_path, sparse_dir, file)
        os.symlink(source_file, destination_file)

    return target_path


def get_light_positions(dataset_dir, light_pos_file: str = None):
    # Write light positions to json.
    if os.path.isfile(os.path.join(dataset_dir, "light_positions.json")):
        print(f"INFO: Found existing light positions at {dataset_dir}. Skipping.")
        return

    print("Load light positions.")

    if light_pos_file is None:
        light_positions, _ = utils.reflectance_utils.load_light_positions()
    else:
        light_positions, _ = utils.reflectance_utils.load_light_positions(
            light_pos_file
        )

    image_files = os.listdir(os.path.join(dataset_dir, "images"))
    light_metadata = {}

    for img_file in image_files:
        if not "board" in os.path.basename(img_file):
            print("WARNING: No light position found for", img_file)
            continue
        board_id = os.path.splitext(os.path.basename(img_file))[0].split("_")[-1]
        light_metadata[os.path.basename(img_file)] = light_positions[board_id]

    # Write to json.
    with open(os.path.join(dataset_dir, "light_positions.json"), "w") as f:
        json.dump(light_metadata, f, indent=4)


def transform(points, transform):
    # Transform all points by the given transformation matrix.
    return (make_homogeneous(points) @ transform.T)[..., :3]


def pad_poses(p: np.ndarray) -> np.ndarray:
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.0], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p: np.ndarray) -> np.ndarray:
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def make_homogeneous(vector):
    return np.concatenate([vector, np.ones_like(vector[..., :1])], -1)


def make_heterogeneous(vector):
    return np.divide(
        vector[..., :-1],
        vector[..., -1:],
        out=np.zeros_like(vector[..., :-1]),
        where=vector[..., -1:] != 0,
    )


def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def recenter_poses(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Recenter poses around the origin."""
    cam2world = average_pose(poses)
    transform = np.linalg.inv(pad_poses(cam2world))
    poses = transform @ pad_poses(poses)
    return unpad_poses(poses), transform


def average_pose(poses: np.ndarray) -> np.ndarray:
    """New pose using average position, z-axis, and up vector of input poses."""
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    cam2world = viewmatrix(z_axis, up, position)
    return cam2world


def matrix_vector(matrix, vector):
    return np.sum(vector[..., None, :] * matrix, -1)


def stable_invert_c2w(c2w):
    r = c2w[..., :3, :3]
    t = c2w[..., :3, 3]

    idxs = list(range(len(r.shape)))
    transpose_idxs = [*idxs[:-2], idxs[-1], idxs[-2]]

    r_inv = np.transpose(r, transpose_idxs)
    t_inv = matrix_vector(r_inv, t)

    w2c = np.concatenate([r_inv, -t_inv[..., None]], -1)
    w2c = pad_poses(w2c)

    return w2c


# Relative transformation between cameras.
def get_relative_transform(reference_c2w, target_c2w):
    return target_c2w @ np.linalg.inv(reference_c2w)


def get_colmap_camera(reconstruction, camera_id):
    """Retrieve camera intrinsics from a colmap reconstruction."""
    camera = reconstruction.cameras[camera_id]
    w, h, params = camera.width, camera.height, camera.params

    if camera.model == pycolmap.CameraModelId.SIMPLE_RADIAL:
        f, cx, cy, *_ = params
        fx = f
        fy = f
    elif camera.model == pycolmap.CameraModelId.PINHOLE:
        fx, fy, cx, cy = params
    return w, h, fx, fy, cx, cy


def resize_img(img_path, resize_to, img_resized_dir):
    img_path = os.path.realpath(img_path)
    img = imageio.imread(img_path)
    # Resize longest size to resize_to.
    img_shape = img.shape
    if img_shape[0] > img_shape[1]:
        target_resolution = (resize_to, int(resize_to * img_shape[1] / img_shape[0]))
    else:
        target_resolution = (int(resize_to * img_shape[0] / img_shape[1]), resize_to)

    img = cv2.resize(img, (target_resolution[1], target_resolution[0]))
    imageio.imwrite(os.path.join(img_resized_dir, os.path.basename(img_path)), img)
    img_resized = os.path.join(img_resized_dir, os.path.basename(img_path))
    scaling_factor = np.array(target_resolution) / img_shape[0:2]
    return scaling_factor, img_resized


def align_light_positions(
    light_stage_calibration_path,
    colmap_dir: str,
    out_path: str,
    auto_scale: bool = True,
    recenter: bool = False,
    scale_factor: float = 0.23157780025228164,
    reconstructed_distance: float = -1.0,
    resize_to: int = -1,
    global_scale: float = 1.0,
    shift_scene_z: float = 0.0,
    shift_scene_y: float = 0.0,
    all_on_dataset: bool = False,
):
    """
    Based on an existing colmap reconstruction and a text file containing
    the light stage positions, the light positions are aligned to the reconstruction
    based on precomputed correspondences relative to the first camera pose theta90,phi0.
    """
    # Precomputed relative transform, see 'test_light_positions.ipynb' for details.
    light_to_colmap_relative = np.array(
        [
            [-0.03353816, -0.04968083, -0.99820189, -0.16935734],
            [-0.99939386, 0.01099358, 0.03303105, 0.03042503],
            [0.0093328, 0.99870464, -0.05001942, 1.00527308],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    measured_distance = 0.215

    # load colmap reconstruction
    reconstruction_path = os.path.join(colmap_dir, "sparse")
    if not os.path.isfile(os.path.join(reconstruction_path, "images.bin")):
        reconstruction_path = os.path.join(colmap_dir, "sparse/0")
    reconstruction = pycolmap.Reconstruction(reconstruction_path)

    if reconstructed_distance > 0:
        scale_factor = measured_distance / reconstructed_distance

    # Load light stage calibration.
    c2w = {}
    for image_id, image in reconstruction.images.items():
        world_t_camera = image.cam_from_world.inverse().matrix()
        # rotation = world_t_camera.rotation.matrix()
        # translation = world_t_camera.translation
        c2w[image.name] = pad_poses(world_t_camera)

    c2w_recentered = {}
    for i, c_id in enumerate(c2w.keys()):
        # m = pad_poses(poses[i].copy())
        m = c2w[c_id].copy()
        # print(m.shape)
        m[:3, -1] *= scale_factor
        c2w_recentered[c_id] = m

    # Apply transformation.
    transform_lights_to_cm = (
        c2w_recentered["theta_90.0_phi_0.0_all_on.png"] @ light_to_colmap_relative
    )

    # Load light positions.
    with open(light_stage_calibration_path) as f:
        data_list = f.readlines()
        numpy_coords = np.array(
            [
                (
                    float(line.split()[1]),  # -0.4, #offset to center
                    float(line.split()[2]),  # -0.25,
                    float(line.split()[3]),  # -1.0
                )
                for line in data_list
            ]
        )
        CM = np.average(numpy_coords, axis=0)
        coords = [
            (
                line.split("_")[0],
                float(line.split()[1]) - CM[0],
                float(line.split()[2]) - CM[1],
                float(line.split()[3]) - CM[2],
            )
            for line in data_list
        ]

    light_coords = np.asarray([c[1:] for c in coords])
    light_label2coords = {c[0]: i for i, c in enumerate(coords)}
    lights_transformed = transform(light_coords, transform_lights_to_cm)

    if recenter:
        c2w_values = np.asarray([c for c in c2w_recentered.values()])
        poses, recenter_transform = recenter_poses(c2w_values)

        print("recenter transform", recenter_transform)
        # Apply to both lights and camera poses.
        for i, c_id in enumerate(c2w_recentered.keys()):
            # m = pad_poses(poses[i].copy())
            m = recenter_transform @ c2w_recentered[c_id].copy()
            # print(m.shape)
            c2w_recentered[c_id] = m

        lights_transformed = transform(lights_transformed, recenter_transform)

    lights_transformed = lights_transformed[..., :3]

    if global_scale != 1:
        lights_transformed *= global_scale
        lights_transformed[:, 2] += shift_scene_z
        lights_transformed[:, 1] += shift_scene_y
        for k, v in c2w_recentered.items():
            v[:3, -1] *= global_scale
            v[2, -1] += shift_scene_z
            v[1, -1] += shift_scene_y

    # Create control image.
    try:
        import plotly.graph_objects as go
        import plotly.express as px

        cam_pos = np.asarray([v[:3, -1] for v in c2w_recentered.values()])

        fig = px.scatter_3d(
            x=lights_transformed[:, 0],
            y=lights_transformed[:, 1],
            z=lights_transformed[:, 2],
        )
        fig.add_trace(
            go.Scatter3d(
                x=cam_pos[:, 0], y=cam_pos[:, 1], z=cam_pos[:, 2], mode="markers"
            )
        )

        fig.write_image(os.path.join(out_path, "cam_lights_aligned.png"))
    except:
        print("Could not create a plot of the aligned lights. Plotly not available?")

    # Turntable rotation. Add virtual light positions.

    light_positions = {
        "theta_90.0_phi_0.0": lights_transformed,
    }

    ref_id = "theta_90.0_phi_0.0_all_on.png"

    for cam_id, c2w_cam in c2w_recentered.items():
        if cam_id == ref_id:
            continue
        if "theta_90.0" in cam_id:
            relative_transform = get_relative_transform(c2w_recentered[ref_id], c2w_cam)

            light_positions[cam_id.replace("_all_on.png", "")] = transform(
                lights_transformed, relative_transform
            )
        else:
            phi_0_id = f"{('_').join(cam_id.split('_')[:3])}_0.0_all_on.png"

            if phi_0_id == cam_id:
                light_positions[cam_id.replace("_all_on.png", "")] = lights_transformed
                continue

            phi0_c2w = c2w_recentered[phi_0_id]

            theta_compensation = get_relative_transform(
                phi0_c2w, c2w_recentered[ref_id]
            )
            relative_transform = get_relative_transform(c2w_recentered[ref_id], c2w_cam)
            relative_transform = relative_transform @ theta_compensation

            lights_transformed_new = transform(lights_transformed, relative_transform)
            light_positions[cam_id.replace("_all_on.png", "")] = lights_transformed_new
        # Inverse rotation...
        # if "theta_90.0" in cam_id:
        #     relative_transform = get_relative_transform(c2w_recentered[ref_id], c2w_cam)

        #     light_positions[cam_id.replace("_all_on.png", "")] = transform(
        #         lights_transformed, relative_transform
        #     )
        # else:
        #     phi_0_id = f"{('_').join(cam_id.split('_')[:3])}_0.0_all_on.png"

        #     if phi_0_id == cam_id:
        #         light_positions[cam_id.replace("_all_on.png", "")] = lights_transformed
        #         continue

        #     phi0_c2w = c2w_recentered[phi_0_id]

        #     theta_compensation = get_relative_transform(
        #         c2w_recentered[ref_id],
        #         phi0_c2w,
        #     )
        #     relative_transform = get_relative_transform(
        #         c2w_cam, theta_compensation @ c2w_recentered[ref_id]
        #     )
        #     # relative_transform = relative_transform @ theta_compensation

        #     lights_transformed_new = transform(lights_transformed, relative_transform)
        #     light_positions[cam_id.replace("_all_on.png", "")] = lights_transformed_new

    # Update / generate json file.
    image_dir = os.path.join(colmap_dir, "images")

    image_files = sorted(glob.glob(os.path.join(image_dir, "theta_*.png")))

    if all_on_dataset:
        image_files_filtered = [
            ip for ip in image_files if "all_on" in os.path.basename(ip)
        ]
    else:
        image_files_filtered = [
            ip for ip in image_files if not "all_on" in os.path.basename(ip)
        ]

    if resize_to > 0:
        # Resize images, assuming constant image size for now.
        print(f"Resizing images to {resize_to} pixels on the longest side...")
        image_files_filtered_resized = []
        img_resized_dir = os.path.join(
            os.path.dirname(os.path.dirname(image_files_filtered[0])), "resized"
        )
        if os.path.isdir(img_resized_dir):
            print("WARN: Resized image directory already exists. Skipping.")
            image_shape = imageio.imread(image_files_filtered[0]).shape
            scaling_factor = resize_to / max(image_shape[:2])
        else:
            os.makedirs(img_resized_dir, exist_ok=True)

            image_files_filtered_resized = []
            pool = multiprocessing.Pool()
            results = []
            for img_path in image_files_filtered:
                result = pool.apply_async(
                    resize_img, (img_path, resize_to, img_resized_dir)
                )
                results.append(result)
            pool.close()
            pool.join()
            for result in results:
                scaling_factor, img_resized = result.get()
                image_files_filtered_resized.append(img_resized)

            image_files_filtered = image_files_filtered_resized
    else:
        scaling_factor = 1

    # Split into train/test.
    if len(image_files_filtered) > 10000:
        stride = 2
    else:
        stride = 4
    test_image_files = image_files_filtered[::stride]
    train_image_files = [
        le for le in image_files_filtered if le not in test_image_files
    ]

    # Compose json files.
    train_metadata = {}
    test_metadata = {}

    images = reconstruction.images
    name2id = {image.name: i for i, image in images.items()}

    def gather_metadata(image_files, metadata_dict):
        camera_angle_x = 0
        for img_path in image_files:
            basename = os.path.basename(img_path)
            # Replace board part with all_on
            view_name = ("_").join(basename.split("_")[:4])
            image_id = name2id[f"{view_name}_all_on.png"]
            if not "all_on" in basename:
                board_id = int(os.path.splitext(basename)[0].split("_")[-1])
            else:
                board_id = 0

            w, h, fx, fy, cx, cy = get_colmap_camera(
                reconstruction, images[image_id].camera_id
            )
            if resize_to > 0:
                h, w = np.array([h, w]) * scaling_factor
                fy, fx = np.array([fy, fx]) * scaling_factor
                cy, cx = np.array([cy, cx]) * scaling_factor

            curr_camera_angle_x = 2 * np.arctan(w / (2 * fx))

            if camera_angle_x != curr_camera_angle_x:
                print(
                    "WARN: Camera angle x changed from",
                    camera_angle_x,
                    "to",
                    curr_camera_angle_x,
                )
                camera_angle_x = curr_camera_angle_x

            selected_c2w = c2w_recentered[images[image_id].name].copy()

            # selected_w2c = stable_invert_c2w(selected_c2w)
            # To blender coordinate system. Flip y and z axis.
            selected_c2w[:3, 1:3] *= -1

            # flip_z = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            # selected_c2w = selected_c2w @ flip_z

            # print("light positions", [k for k in light_positions])
            # cam_light_pos = light_positions[
            #     f"theta_90.0_{('_').join(os.path.splitext(basename)[0].split('_')[2:4])}"
            # ].tolist()

            # Also convert light positions
            cam_light_pos = light_positions[view_name].copy()
            # Fixme: Why
            # cam_light_pos[:, 1:3] *= -1
            # cam_light_pos = transform(cam_light_pos, flip_z)

            cam_light_pos = cam_light_pos[board_id].tolist()

            if metadata_dict.get(view_name) is None:
                metadata_dict[view_name] = {}
                metadata_dict[view_name]["file_paths"] = []
                metadata_dict[view_name]["light_positions"] = []

            metadata_dict[view_name]["file_paths"].append(
                os.path.splitext(os.path.relpath(img_path, start=out_path))[0]
                if resize_to == -1
                else os.path.splitext(
                    os.path.relpath(img_path, start=out_path).replace(
                        "images", "./resized"
                    )
                )[0]
            )
            metadata_dict[view_name]["light_positions"].append(cam_light_pos)
            metadata_dict[view_name]["rotation"] = float(view_name.split("_")[-1]) * (
                np.pi / 180
            )
            metadata_dict[view_name]["transform_matrix"] = selected_c2w.tolist()
            metadata_dict[view_name]["width"] = int(w)
            metadata_dict[view_name]["height"] = int(h)
            metadata_dict[view_name]["cx"] = cx
            metadata_dict[view_name]["cy"] = cy

        #  Convert view names to frame lists.
        frames = [f for k, f in metadata_dict.items()]
        print(f"Created split with {len(frames)} images")
        metadata_dict = {
            "camera_angle_x": camera_angle_x,
            "frames": frames,
        }
        return metadata_dict

    train_metadata = gather_metadata(train_image_files, train_metadata)
    test_metadata = gather_metadata(test_image_files, test_metadata)

    with open(os.path.join(out_path, "transforms_train.json"), "w") as f:
        json.dump(train_metadata, f, indent=4)

    with open(os.path.join(out_path, "transforms_test.json"), "w") as f:
        json.dump(test_metadata, f, indent=4)


def check_flare(img_path, mask_path=None, threshold=0.2):
    basename = os.path.basename(img_path)
    if "dark" in basename or "all_on" in basename:
        return False

    img = cv2.imread(img_path) / 255.0
    check = (np.sum(img == 1.0) / (img.shape[0] * img.shape[1])) > threshold

    if mask_path:
        img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) / 255.0
        check = check or (np.sum(img[..., -1]) < 0.06 * (img.shape[0] * img.shape[1]))

    return check


def create_exclude_list(dataset_dir, images_dir, threshold=0.2):
    """Create a list of images to exclude based on the flare intensity."""
    out_file = os.path.join(dataset_dir, "exclude_list.txt")
    if os.path.isfile(out_file):
        print("Exclude list already exists. Skipping.")
        return

    print("INFO: Creating exclude list initialization.")

    exclude_list = []
    img_paths = sorted(glob.glob(os.path.join(images_dir, "theta*")))
    img_paths = [ip for ip in img_paths if "all_on" not in os.path.basename(ip)]
    mask_paths = sorted(glob.glob(os.path.join(dataset_dir, "resized", "theta*")))
    assert len(img_paths) == len(mask_paths)

    with multiprocessing.Pool() as pool:
        # exclude_list = pool.map(check_flare, glob.glob(os.path.join(dataset_dir, "images", "theta*")))
        exclude_list = pool.starmap(
            check_flare,
            [
                (img_path, mask_path, threshold)
                for img_path, mask_path in zip(img_paths, mask_paths)
            ],
        )

    exclude_files = np.array(img_paths)[exclude_list]

    with open(out_file, "w") as f:
        for img_path in exclude_files:
            f.write(os.path.splitext(os.path.basename(img_path))[0] + "\n")


def main():
    # Todo: Add preview for matting.
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.colmap_path is None:
        args.colmap_path = shutil.which("colmap")

    if args.demosaic:
        # Demosaic images.
        demosaic_dir = demosaic_images(args.input_dir, args.output_dir)

        args.input_dir = demosaic_dir
    else:
        if os.path.isdir(os.path.join(args.output_dir, "demosaiced")):
            args.input_dir = os.path.join(args.output_dir, "demosaiced")

    # 1. Copy all_on images to the output directory.
    images_dir = copy_all_on_images(args.input_dir, args.output_dir)

    # 2. colmap reconstruction.
    # colmap_dir = colmap_reconstruction(images_dir, args.output_dir, args.colmap_path)
    colmap_dir = manual_colmap_pipeline(
        images_dir, args.output_dir, args.colmap_path, args.high_quality
    )
    # colmap_dir = colmap_reconstruction(images_dir, args.output_dir, args.colmap_path)

    # undistort_images(args.input_dir, args.output_dir, colmap_dir, args.colmap_path)

    # 2. matting.
    matte_dir = generate_mattes(
        images_dir,
        args.output_dir,
        args.matte_anything_path,
        args.prompt,
        args.downscale_factor,
        args.invert_matte,
    )
    matte_dir_alpha = os.path.join(matte_dir, "alpha")
    matte_dir_comp = os.path.join(matte_dir, "comp")

    # 3. Copy mattes for all light directions.
    copy_matte_to_all_light_dirs(
        matte_dir_alpha,
        os.path.join(args.output_dir, "mask"),
        args.input_dir,
        comp_dir=matte_dir_comp,
        skip_all_on=False,
    )

    # copy_matte_to_all_light_dirs(
    #     matte_dir_comp, os.path.join(args.output_dir, "masked"), args.input_dir
    # )

    undistorted_dir = undistort_images(
        os.path.join(args.output_dir, "masked"),
        args.output_dir,
        colmap_dir,
        args.colmap_path,
    )

    # 4. Convert colmap reconstruction to 3DGS format.
    # 3DGS expects the following data structure.
    # <location>
    # ├── images
    # │   ├── 0.png
    # │   ├── 1.png
    # │   └── ...
    # ├── sparse
    # │   ├── 0
    # │   │   ├── cameras.bin
    # │   │   ├── images.bin
    # │   │   ├── points3D.bin
    # We just link the images and the sparse data from colmap.
    dataset_dir = convert_to_3dgs_format(
        undistorted_dir,
        args.output_dir,
        image_dir=os.path.join(os.path.basename(undistorted_dir), "images"),
    )

    # Todo: potentially add downsizing option / use convert.py from 3DGS.

    # 5. Look up light positions
    get_light_positions(dataset_dir)

    if not args.stage_2:
        print("Check for scaling factor")
        return

    # Align light positions and resize images.
    # This outputs the .json transforms for the dataset splits.
    align_light_positions(
        light_stage_calibration_path="/graphics/projects2/data/light_stage/calibration/led_positions_white_25.10.2017",
        colmap_dir=dataset_dir,
        out_path=dataset_dir,
        auto_scale=False,
        recenter=True,
        reconstructed_distance=args.light_stage_reconstructed_distance,
        resize_to=args.resize_to,
        global_scale=args.global_scale,
        shift_scene_z=args.shift_scene_z,
        shift_scene_y=args.shift_scene_y,
        all_on_dataset=args.all_on_dataset,
    )

    # Initialize exclude list with flare images.
    create_exclude_list(dataset_dir, args.input_dir, threshold=args.flare_threshold)

    # Set permissions.
    print("Update permissions.")
    subprocess.run(f"chmod -R g+w {args.output_dir}", shell=True, check=True, text=True)

if __name__ == "__main__":
    main()
    # in_path = "/ceph/datasets/sss_light_stage/preprocessed/red_car_2/dataset/images/"
    # out_path = "/ceph/datasets/sss_light_stage/preprocessed/red_car_2/dataset/resized/"
    # image_paths = glob.glob(os.path.join(in_path, "*all_on.png"))

    # for p in image_paths:
    #     resize_img(p, 800, out_path)

    # out_path = "/graphics/scratch2/staff/engelhar/datasets/sss_light_stage/red_car_2_demosaiced/"
    # demosaic_images(in_path, out_path)
