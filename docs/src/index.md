```@raw html
---
layout: home

hero:
  name: "MedEye3d.jl"
  text: "Annotating and Visualizing segmentations in medical images"
  tagline: Julia library for visualization and annotation medical images, specialized particularly for rapid development segmentation of 3 dimensional images like CT or PET/CT scans. Has full support of nuclear medicine Data.
  image:
    src: /logo.png
    alt: MedEye3d.jl Graphic
  actions:
    - theme: brand
      text: View on JuliaHealth 
      link: https://JuliaHealth.org
    - theme: alt
      text: View on Github
      link: https://github.com/JuliaHealth/MedEye3d.jl

features:
  - icon: 🔬
    title: Image Registration and I/O
    details: Image Registration and I/O wrappers via Insight Toolkit's ITK library backend for NIFTI and DICOM
  - icon: ⚛️
    title: Image Transformations 
    details: Creating transformations (rotation, cropping, resampling and modification) over loaded image factories 
  - icon: 🤖
    title: Automated Testing
    details: Automated testing suite, available via  
  - icon: 💻
    title: Seamless Integration with other Julia Packages
    details: Downstream integration with other medical-ecosystem pkgs
---
```

````@raw html
<div class="vp-doc" style="width:80%; margin:auto">

<p style="margin-bottom:2cm"></p>

<h1> What is MedEye3d.jl? </h1>

MedImages is a package for standardizing data handling of 3D and 4D medical images, with additional support for analysis and modification.

It is meant to be used in conjunction with the `MedEye3d` visualization package, which further will be integrated within a pipeline `MedPipe3D`!

<h2> Basic usage </h2>


1. Import the package MedImages within a Julia REPL,
2. Pass an `image file path` to `load_image` to get a MedImage object:

```julia
using MedImages
image_path = "/path/to/your/image"
medimage_container = MedImages.load_image(image_path)

help(medimage_container)
```
and enjoy the fruits of your labour!

<div style="text-align: center; margin-top: 4rem; padding: 2rem 0; border-top: 1px solid #eaecef; color: #4e6e8e;">
© 2025 Divyansh Goyal | Last updated: May 21, 2025
</div>

</div>
````
