# SSS GS


<p align="left">
  <strong>
    Subsurface Scattering for Gaussian Splatting
  </strong>
</p>


https://github.com/user-attachments/assets/1a30660f-ff7b-48ad-a54b-fd293295bd41


<p align="center">
    <span> 🌐  <a href="https://sss.jdihlmann.com/"> Project Page </a> </span>&nbsp;&nbsp;&nbsp;
    <span> 📄  <a href="https://arxiv.org/abs/2408.12282"> Paper (Arxiv) </a> </span>&nbsp;&nbsp;&nbsp;
    <span> 💿  <a href="https://huggingface.co/datasets/CGTuebingen/SSS-GS"> Dataset </a> </span>&nbsp;&nbsp;&nbsp;
  <span>  📦  <a href="https://drive.google.com/drive/folders/1znN_KllBKllIY_1PLZUHbnfHsB6KNifR?usp=sharing"> Materials </a> </span>&nbsp;&nbsp;&nbsp;
  <span>  ✍🏻
     <a href="https://github.com/cgtuebingen/SSS-GS?tab=readme-ov-file#citation"> Citation </a> </span>&nbsp;&nbsp;&nbsp;
</p>


# About
SSS GS got accepted to [NeurIPS 2024](https://neurips.cc/) - [Poster Information](https://neurips.cc/virtual/2024/poster/96787)

SSS GS allows for real-time rendering, relighting and material editing of translucent objects utilizing [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) as scene representation.

> We propose photorealistic real-time relighting and novel view synthesis of subsurface scattering objects. We learn to reconstruct the shape and translucent appearance of an object within the 3D Gaussian Splatting framework. Our method decomposes the object into its material properties in a PBR like fashion, with an additional neural subsurface residual component. We achieve high-quality rendering results with our deferred shading approach and allow for detailed material editing capabilities.


# Installation
#### Clone this repo
```shell
git clone https://github.com/cgtuebingen/SSS-GS
```
#### Install dependencies
```shell
# install environment
conda env create --file environment.yml
conda activate sss-gs

# install pytorch=1.12.1
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

# install torch_scatter==2.1.1
pip install torch_scatter==2.1.1

# install kornia==0.6.12
pip install kornia==0.6.12

# install nvdiffrast=0.3.1
git clone https://github.com/NVlabs/nvdiffrast
pip install ./nvdiffrast
```


#### Install the pytorch extensions
We recommend that users compile the extension with CUDA 11.8 to avoid the potential problems mentioned in [3D Guassian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

```shell
# install knn-cuda
pip install ./submodules/simple-knn

# install bvh
pip install ./bvh

# install gaussian splatting sss rasterization
pip install ./gs_sss_rasterization
```


# Dataset
<img width="1480" alt="dataset" src="https://github.com/user-attachments/assets/8766b6f4-0442-4e4e-8a0a-c3dec063cc27" />

We have released the dataset at [CGTübingen/SSS‑GS on Hugging Face](https://huggingface.co/datasets/CGTuebingen/SSS-GS). It contains over 37,000 OLAT images of 25 translucent objects (20 real‑world, 5 synthetic), each captured under 100+ camera views and 100+ light positions, with calibrated transforms compatible with NeRF and Gaussian Splatting pipelines. The data is processed to 800 px images with alpha masks (raw 16 MP captures available upon request). Please check the licensing before utilizing the dataset. 

Find details on the dataset [here](https://huggingface.co/datasets/CGTuebingen/SSS-GS). Move the dataset to the root folder titled `dataset` and unpack the tar files.


# Usage Guide

## Hardware Requirements
- **GPU**: NVIDIA GPU with 24GB+ VRAM (tested on RTX 3090)
- **Memory**: 32GB+ system RAM recommended
- **Storage**: 100GB+ free space for dataset and outputs
- **CUDA**: Version 11.6+ required

## Training

### Basic Training Command
```bash
python train.py -s path_to_dataset_object_directory --eval --wandb
```

### Training Parameters

#### Essential Parameters
- `-s, --source_path`: Path to the dataset object directory (required)
- `--model_path`: Output directory for the trained model (auto-generated if not specified)
- `--eval`: Enable evaluation during training (optional)
- `--wandb`: Enable Weights & Biases logging (optional)

#### Training Configuration
- `--iterations`: Total training iterations (default: 60,000)
- `--batch_size`: Batch size for training (default: 100)
- `--render_iterations`: Frequency of debug renders (default: 1,000)
- `--test_iterations`: Iterations for evaluation (default: [1000, 5000, 7000, 10000, 12000, 15000, 20000, 30000, 40000, 50000, 60000, 90000, 120000])

#### Learning Rates
- `--position_lr_init`: Initial position learning rate (default: 0.00016)
- `--feature_lr`: Feature learning rate (default: 0.0025)
- `--opacity_lr`: Opacity learning rate (default: 0.05)
- `--color_lr`: Color learning rate (default: 0.0025)
- `--scaling_lr`: Scaling learning rate (default: 0.005)
- `--rotation_lr`: Rotation learning rate (default: 0.001)
- `--normals_lr`: Normals learning rate (default: 0.01)

#### Material Learning Rates
- `--base_color_lr`: Base color learning rate (default: 0.01)
- `--roughness_lr`: Roughness learning rate (default: 0.01)
- `--metallic_lr`: Metallic learning rate (default: 0.01)
- `--subsurfaceness_lr`: Subsurfaceness learning rate (default: 0.01)
- `--light_lr`: Light learning rate (default: 0.001)
- `--visibility_lr`: Visibility learning rate (default: 0.0025)

#### Loss Weights
- `--lambda_dssim`: DSSIM loss weight (default: 0.2)
- `--lambda_lpips`: LPIPS loss weight (default: 0.2)
- `--lambda_normal`: Normal loss weight (default: 0.02)
- `--lambda_visibility`: Visibility loss weight (default: 0.01)
- `--lambda_incident_light`: Incident light loss weight (default: 0.02)
- `--lambda_mask_entropy`: Mask entropy loss weight (default: 0.1)
- `--lambda_base_color`: Base color loss weight (default: 0.005)
- `--lambda_base_color_smooth`: Base color smoothness loss weight (default: 0.006)

#### Densification Parameters
- `--densify_until_iter`: Stop densification at this iteration (default: 15,000)
- `--densify_from_iter`: Start densification at this iteration (default: 500)
- `--densification_interval`: Densification frequency (default: 100)
- `--densify_grad_threshold`: Gradient threshold for densification (default: 0.0002)
- `--opacity_reset_interval`: Opacity reset frequency (default: 3,000)



## Evaluation

### Basic Evaluation Command
```bash
python evaluate.py -m output_path --mode test --speedtest
```

### Evaluation Parameters
- `-m, --model_path`: Path to the trained model (required)
- `--mode`: Evaluation mode - `train`, `test`, or `both` (required)
- `--iteration`: Model iteration to evaluate (default: -1, uses latest)
- `--speedtest`: Only measure rendering speed, skip metrics (optional)

### Evaluation Metrics
The evaluation computes:
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity

### Output
Evaluation results are saved as JSON files:
- `evaluation_{mode}.json`: Overall metrics
- `per_view_evaluation_{mode}.json`: Per-view metrics

## Rendering

### Basic Rendering Command
```bash
python render.py -m path_to_model
```

### Rendering Parameters

#### Essential Parameters
- `-m, --model_path`: Path to the trained model (required)
- `--mode`: Rendering mode - `train`, `test`, `both`, or `custom` (default: train)
- `--iteration`: Model iteration to render (default: -1, uses latest)
- `--name`: Custom name for output directory (default: custom)

#### Custom Rendering
- `--custom_transforms_file`: Path to custom camera transforms JSON (required for custom mode)
- `--alternative_transforms_path`: Alternative path for transforms
- `--no_gt_images`: Skip ground truth image rendering (optional)

#### Output Maps
- `--maps`: Comma-separated list of maps to render (default: none)
  - `all`: Render all available maps
  - `none`: Only render final image
  - Individual maps: `normals`, `depth`, `base_color`, `roughness`, `metalness`, `pbr`, `residual`, `visibility`, `color_diffuse`, `color_specular`, `opacity`, `incident_light`, `subsurfaceness`

#### Output Formats
- `--exr`: Save images in EXR format (default: PNG)
- `--skip_videos`: Skip video generation (optional)

### Available Material Maps
- **normals**: Surface normal vectors
- **depth**: Depth information
- **base_color**: Albedo/base color
- **roughness**: Surface roughness
- **metalness**: Metallic properties
- **pbr**: Physically-based rendering component
- **residual**: Neural subsurface scattering residual
- **visibility**: Light visibility
- **color_diffuse**: Diffuse color component
- **color_specular**: Specular color component
- **opacity**: Alpha/opacity values
- **incident_light**: Incident light intensity
- **subsurfaceness**: Subsurface scattering strength

### Material Editing
The renderer supports runtime material editing through override parameters:
```python
override = {
    "residual_color": [0.5, 0.2, 0.2],  # Subsurface color
    "base_color": [1.0, 0.9, 0.9],      # Albedo
    "roughness": 0.9,                   # Surface roughness
    "metallness": 0.0,                  # Metallic properties
    "subsurfaceness": 0.8,              # Subsurface strength
    "opacity": 0.12,                    # Transparency
    "transition": 0.5                   # Animation parameter
}
```

# Known Issues
After the rework of the code for the realeas with fixes in tonemapping etc. we found that the residual sometimes overtakes the PBR term for real-world data sometimes. Leading to a residual only rendering without clear PBR decomposition and nice specular highlights. We recommend playing around with the incident light scheduling in the `scene/sss_model.py` file. We are searching for a generalizable solution again (also looking forward to help from the community). But we rather release this version with the current issues then again postpone the release. If you need comparison against the original version from the paper, feel free to contact us.

# Citation
You can find our paper on [arXiv](https://arxiv.org/abs/2408.12282), please consider citing, if you find this work useful:
```
@inproceeding{Dihlmann2024SSSGS,
 author = {Dihlmann, Jan-Niklas and Majumdar, Arjun and Engelhardt, Andreas and Braun, Raphael and Lensch, Hendrik P.A.},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {121765--121789},
 publisher = {Curran Associates, Inc.},
 title = {Subsurface Scattering for Gaussian Splatting},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/dc72529d604962a86b7730806b6113fa-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}
```

# Related Projects
We thank the authors of the following projects for releasing their code and data, that we used for inspiration and in parts build on top of:

- [Relightable 3D Gaussian Splatting](https://github.com/NJU-3DV/Relightable3DGaussian)
- [Viewing Direction Gaussian Splatting](https://github.com/gmum/ViewingDirectionGaussianSplatting)
- [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)