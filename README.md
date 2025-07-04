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
> ‼️ UPDATE ‼️: Dataset is available now find it at [CGTübingen/SSS‑GS on Hugging Face](https://huggingface.co/datasets/CGTuebingen/SSS-GS)

We propose photorealistic real-time relighting and novel view synthesis of subsurface scattering objects. We learn to reconstruct the shape and translucent appearance of an object within the 3D Gaussian Splatting framework. Our method decomposes the object into its material properties in a PBR like fashion, with an additional neural subsurface residual component. We achieve high-quality rendering results with our deferred shading approach and allow for detailed material editing capabilities.

# Code
We are cuttently finishing the code and it will be hopefully available soon. Sorry for the long waiting time. 

# Dataset
<img width="1480" alt="dataset" src="https://github.com/user-attachments/assets/8766b6f4-0442-4e4e-8a0a-c3dec063cc27" />

We have released the dataset at [CGTübingen/SSS‑GS on Hugging Face](https://huggingface.co/datasets/CGTuebingen/SSS-GS). It contains over 37,000 OLAT images of 25 translucent objects (20 real‑world, 5 synthetic), each captured under 100+ camera views and 100+ light positions, with calibrated transforms compatible with NeRF and Gaussian Splatting pipelines. The data is processed to 800 px images with alpha masks (raw 16 MP captures available upon request). Please check the licensing before utilizing the dataset. 


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
