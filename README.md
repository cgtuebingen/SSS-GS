# SSS GS


<p align="left">
  <strong>
    Subsurface Scattering for Gaussian Splatting
  </strong>
</p>


https://github.com/user-attachments/assets/1a30660f-ff7b-48ad-a54b-fd293295bd41


<p align="center">
    <span> 🌐  <a href="https://sss.jdihlmann.com/"> Project Page </a> </span>&nbsp;&nbsp;&nbsp;
    <span> 📄  <a href="https://arxiv.org/abs/2401.01647"> Paper (Arxiv) </a> </span>&nbsp;&nbsp;&nbsp;
    <span> 💿  <a href="https://github.com/cgtuebingen/SSS-GS?tab=readme-ov-file#dataset"> Dataset (Soon) </a> </span>&nbsp;&nbsp;&nbsp;
  <span>  📦  <a href="https://drive.google.com/drive/folders/1znN_KllBKllIY_1PLZUHbnfHsB6KNifR?usp=sharing"> Materials </a> </span>&nbsp;&nbsp;&nbsp;
  <span>  ✍🏻
     <a href="https://github.com/cgtuebingen/SSS-GS?tab=readme-ov-file#citation"> Citation </a> </span>&nbsp;&nbsp;&nbsp;
</p>

# About
We propose photorealistic real-time relighting and novel view synthesis of subsurface scattering objects. We learn to reconstruct the shape and translucent appearance of an object within the 3D Gaussian Splatting framework. Our method decomposes the object into its material properties in a PBR like fashion, with an additional neural subsurface residual component. We achieve high-quality rendering results with our deferred shading approach and allow for detailed material editing capabilities.

# Code
Paper is currently under review, we will realse the code shortly (~ end of october 2024) in a cleaned version. Up until then if you have any questions regarding the project or need material to compare against fast, feel free to contact us. 

# Dataset
Paper is currently under review, we will realse the dataset shortly (~ end of october 2024) in a cleaned version. Up until then if you have any questions regarding the project or need material to compare against fast, feel free to contact us. The dataset will consist of 20 translucent object with 5 synthetic and 15 real-world captures. Each object is an OLAT light stage capture with > 100 camera poses and > 1000 light positions.


# Citation
You can find our paper on [arXiv](https://arxiv.org/abs/2408.12282), please consider citing, if you find this work useful:
```
@inproceeding{sss_gs,
  author ={Dihlmann, Jan-Niklas and Majumdar, Arjun and Engelhardt, Andreas and Braun, Raphael and Lensch, Hendrik P.A.},
  title ={Subsurface Scattering for Gaussian Splatting},
  booktitle ={arXiv preprint arXiv:2408.12282},
  year ={2024}
}
