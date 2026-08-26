```@raw html
---
layout: home

hero:
  name: "MedEye3d.jl"
  text: "High-Performance 3D Medical Image Visualization, Annotation & AI Segmentation in Julia"
  tagline: "Specialized for nuclear medicine, longitudinal therapies (PET/CT/SPECT), GPU-accelerated post-processing, and interactive deep learning segmentation."
  image:
    src: /logo.png
    alt: MedEye3d.jl Graphic
  actions:
    - theme: brand
      text: Get Started
      link: /manual/get_started
    - theme: alt
      text: Code Examples
      link: /manual/code_example
    - theme: alt
      text: View on GitHub
      link: https://github.com/JuliaHealth/MedEye3d.jl

features:
  - icon: 🖥️
    title: ModernGL QuadView Layout
    details: Synchronized 4-panel viewport with cross-plane 3D slice jumping, double-click maximization, and data-level zoom/panning.
  - icon: ☢️
    title: Longitudinal Nuclear Medicine
    details: Multi-cycle radionuclide therapy evaluation across PET/CT (TP0–TP3) and SPECT/CT (TP0–TP4) with quantitative SUV windowing.
  - icon: ⚡
    title: GPU KernelAbstractions
    details: Portable CUDA & CPU kernels for 1.38ms 3D Connected Component Labeling and 12.5µs continuous swept-capsule stroke rasterization.
  - icon: 🤖
    title: Real-Time AI Segmentation
    details: Containerized deep learning worker running HELPNet 3D CNN and MIC-DKFZ NNInteractive with sub-0.3s GPU turnaround.
  - icon: 🦴
    title: Skeletal Subsegmentation
    details: Automated separation of cortical bone surface vs trabecular bone marrow for targeted bone metastasis dosimetry.
  - icon: 🎛️
    title: GLMakie Clinical Dashboard
    details: Dual-handle IntervalSliders, preset windowing, mask layer visibility toggles, and metadata reporting.
  - icon: 📋
    title: Clinical Annotation & PROMISE/RECIP
    details: Structured metadata panel with searchable dropdowns, automatic PROMISE scoring, lesion volume tracking, cross-TP match analysis with RECIP 1.0 classification, and anatomy auto-fill.
---
```

`@raw html
<div class="vp-doc" style="width:85%; margin:auto">

<h2 style="text-align: center; margin-top: 2rem;">Why MedEye3d.jl?</h2>

<p>
Medical imaging reading and segmentation often require juggling bulky viewers, slow slice re-slicing, and disconnected Python AI models. 
<strong>MedEye3d.jl</strong> unifies raw OpenGL rendering performance with Julia's scientific ecosystem and containerized deep learning workers into a cohesive, responsive clinical and research tool.
</p>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin: 2rem 0;">

<div style="border: 1px solid var(--vp-c-divider); border-radius: 8px; padding: 1.2rem; background: var(--vp-c-bg-soft);">
<h3 style="margin-top: 0;">🚀 Sub-Millisecond GPU Post-Processing</h3>
<p>Eliminate discrete brush gaps during manual painting with continuous swept-capsule rasterization (12.5 µs) and filter false positives with GPU multi-pass Connected Component Labeling (1.38 ms on RTX 3090).</p>
</div>

<div style="border: 1px solid var(--vp-c-divider); border-radius: 8px; padding: 1.2rem; background: var(--vp-c-bg-soft);">
<h3 style="margin-top: 0;">🎯 Real-Time Interactive AI</h3>
<p>Incorporate foundation models into your annotation workflow. In-memory session caching and focal inference enable real-time 0.28s interactive scribble segmentation via MIC-DKFZ NNInteractive.</p>
</div>

<div style="border: 1px solid var(--vp-c-divider); border-radius: 8px; padding: 1.2rem; background: var(--vp-c-bg-soft);">
<h3 style="margin-top: 0;">🔬 Longitudinal Therapy Comparison</h3>
<p>Effortlessly track tumor response across time points (PET/CT and SPECT) with synchronized camera coordinates, side-by-side comparison overlays, and automated lesion group tracking.</p>
</div>

</div>

<div style="text-align: center; margin-top: 3rem; padding: 1.5rem 0; border-top: 1px solid var(--vp-c-divider); color: var(--vp-c-text-2);">
MedEye3d.jl is a proud member of the <a href="https://juliahealth.org" target="_blank">JuliaHealth</a> organization.
</div>

</div>
```
