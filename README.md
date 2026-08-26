<div align="center">
  <img src="./docs/src/assets/logo.png" alt="MedEye3d.jl Logo" width="200" align="left" style="margin-right: 20px"/>
  <h1>MedEye3d.jl</h1>
  <p><em>High-performance 3D medical image visualization, longitudinal nuclear medicine reading, and GPU-accelerated annotation in Julia</em></p>

[![JuliaHealth](https://img.shields.io/badge/Julia-Health-purple.svg)](https://juliahealth.org)
[![Documentation](https://img.shields.io/badge/docs-DocumenterVitepress-blue.svg)](https://juliahealth.org/MedEye3d.jl)
[![GPU Powered](https://img.shields.io/badge/GPU-KernelAbstractions.jl-green.svg)](https://github.com/JuliaGPU/KernelAbstractions.jl)
[![Docker AI](https://img.shields.io/badge/AI-HELPNet%20%7C%20nnInteractive-orange.svg)](https://github.com/MIC-DKFZ/nnInteractive)

</div>

<br clear="all"/>

---

## 🌟 Overview

**MedEye3d.jl** is a high-performance Julia framework for real-time visualization, annotation, and AI-assisted segmentation of volumetric medical imaging data (CT, PET/CT, SPECT/CT, and MRI). Built on raw OpenGL (`ModernGL.jl` and `GLFW.jl`) combined with a rich [GLMakie](https://makie.juliaplots.org/) clinical dashboard, MedEye3d is specifically engineered for:

- **QuadView 4-Panel Diagnostic Layout**: Synchronized axial fused PET/CT, axial pure PET, sagittal, and coronal resliced views with cross-plane 3D navigation.
- **Longitudinal Nuclear Medicine Reading**: Seamless comparison of multi-cycle radionuclide therapies ($^{177}\text{Lu}$-PSMA, $^{225}\text{Ac}$, $^{90}\text{Y}$) across PET/CT (TP0–TP3) and SPECT/CT (TP0–TP4).
- **GPU-Accelerated Post-Processing**: Custom `KernelAbstractions.jl` kernels for sub-millisecond 3D Connected Component Labeling (CCL/LCC) and swept-capsule continuous thick-line stroke rasterization.
- **Asynchronous Deep Learning Semiauto Segmentation**: Integrated containerized worker supporting **HELPNet** (3D lesion CNN) and **MIC-DKFZ NNInteractive** (prompt-based interactive foundation model) with sub-0.3s GPU turnaround.
- **Anatomical Bone Compartmentalization**: Automated separation of cortical bone surface vs trabecular bone marrow for skeletal metastasis dosimetry.
- **Clinical Annotation & PROMISE/RECIP Scoring**: Structured metadata panel with searchable dropdowns, automatic PROMISE score computation (SUV vs liver/parotid/blood pool), lesion volume tracking, cross-timepoint match analysis with RECIP 1.0 response classification (CR/PR/SD/PD), and "No CT Correlate" toggle.

---

## 🚀 Installation & Requirements

### Julia 1.9+ Interactive Thread Requirement

OpenGL context management in MedEye3d requires Julia's interactive threadpool. Start Julia with at least **1 interactive thread**:

```bash
# Set environment variable (recommended)
export JULIA_NUM_THREADS=3,1

# Or launch Julia with --threads
julia --threads 3,1
```

### Install the Package

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaHealth/MedEye3d.jl.git")
```

---

## ⚡ Quick Start

### 1. Basic Single / Multi-Modal Viewer

```julia
using MedEye3d

# Single CT scan
ct_data = ("path/to/ct.nii.gz", "CT")
MedEye3d.SegmentationDisplay.displayImage(ct_data)

# Dual-modal fused PET/CT
pet_data = ("path/to/pet.nii.gz", "PET")
MedEye3d.SegmentationDisplay.displayImage((ct_data, pet_data))
```

### 2. Full Longitudinal Nuclear Medicine Application

To launch the complete application with 3D Slicer MRB scene loading, QuadView, Makie metadata dashboard, and AI semiauto segmentation:

```bash
# Start the AI Docker Worker (optional, for AI inference)
./scripts/ai/start_docker_worker.sh

# Launch the interactive application
julia --threads 3,1 scripts/app/run_interactive_mrb.jl data/pat_6_files
```

---

## 🎮 Interactive Controls & Shortcuts

### Mouse Controls
| Action | Description |
| :--- | :--- |
| **Scroll Wheel** | Navigate through slices (Z-axis scrolling). |
| **Shift + Scroll** | Data-level zoom in / zoom out (up to 20x). |
| **Right-Click (Click)** | **Cross-Plane 3D Slice Jump**: Immediately moves orthogonal Sagittal and Coronal views to intersecting $(X, Y, Z)$ slice. |
| **Right-Click (Drag)** | **Pan**: Moves zoomed field of view across the viewport. |
| **Double-Click** | **Maximize / Restore**: Maximizes clicked quad panel to full window; double-click again restores 4-panel grid. |
| **Left-Click (Drag)** | **Continuous Paint / Erase**: Draws smooth continuous thick strokes into active segmentation mask. |

### Keyboard Shortcuts
| Key | Action |
| :--- | :--- |
| `Space` | Reset slice position to volume center. |
| `F3` / `F4` | Decrease / increase slice slab thickness (Maximum Intensity Projection - MIP). |
| `1` – `9` | Toggle individual texture/mask layer visibility. |
| `Ctrl + Z` | Undo last manual mask paint/erase stroke. |

---

## 🏗️ Core Architectural Highlights

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MedEye3d.jl Engine                                │
├──────────────────────────────┬──────────────────────────────┬───────────────┤
│    ModernGL / GLFW Engine    │   GLMakie Clinical Panel     │ AI Worker     │
│  - Single-threaded Consumer  │  - IntervalSliders Windowing │ - Docker 5005 │
│  - Global Lock (Thread-Safe) │  - Timepoint Navigation      │ - HELPNet 3D  │
│  - QuadView & Compare Modes  │  - Bone & Organ Overlays     │ - nnInter-    │
│  - Swept-Capsule Rasterizer  │  - Non-blocking Event Queue  │   active      │
├──────────────────────────────┼──────────────────────────────┤               │
│  Lesion Metadata & Tracking  │  Volume & Match Analysis     │               │
│  - Searchable Dropdowns      │  - PROMISE Score (0-3)       │               │
│  - Anatomy Auto-Fill (JSON)  │  - RECIP 1.0 Classification  │               │
│  - No CT Correlate Toggle    │  - Cross-TP Volume Deltas    │               │
│  - 20-Field Clinical Schema  │  - SUV Comparison vs Liver   │               │
└──────────────────────────────┴──────────────────────────────┴───────────────┘
```

1. **Thread-Safe Single-Consumer Queue**: Producer threads (GLFW callbacks, Makie GUI buttons) write typed event structs into a `Channel{Any}(1000)`. A dedicated consumer task owns the OpenGL context, serialized with `GLOBAL_OPENGL_LOCK`.
2. **GPU KernelAbstractions Engine**:
   - `ConnectedComponents.jl`: 1.38 ms on RTX 3090 (42.5x faster than SimpleITK C++).
   - `StrokeRasterization.jl`: 12.5 $\mu\text{s}$ latency for gapless brush strokes.
3. **Containerized AI Server**: Isolated Python process communicates over high-speed TCP JSON protocol (port 5005) with in-memory session caching for real-time ~0.28s interactive segmentation.

---

## 📖 Documentation

Full documentation, architecture manuals, and code examples are available in the [Official MedEye3d Documentation](https://juliahealth.org/MedEye3d.jl):

- [Getting Started & Installation](docs/src/manual/get_started.md)
- [Code Examples & Tutorials](docs/src/manual/code_example.md)
- [AI Inference Pipeline & Docker Architecture](docs/src/manual/ai_inference_pipeline.md)
- [GPU Kernels & Post-Processing](docs/src/manual/gpu_kernels_and_postprocessing.md)
- [QuadView 4-Panel Layout & 3D Navigation](docs/src/manual/quad_view_and_navigation.md)
- [Bone Subsegmentation & MRB Integration](docs/src/manual/bone_subsegmentation_and_mrb.md)
- [Makie GUI Controls & Windowing](docs/src/manual/gui_controls_and_windowing.md)
- [Lesion Metadata, PROMISE & RECIP Tracking](docs/src/manual/lesion_metadata_and_tracking.md)
- [Developer Playbook & Lock Architecture](docs/src/devs/playbook.md)
- [Complete API Reference](docs/src/api.md)

---

## 📝 Citation & Acknowledgments

If you use MedEye3d.jl in your research or clinical studies, please cite:

```bibtex
@article{medeye3d2026,
  title={MedEye3d.jl: High-Performance 3D Medical Image Visualization, Annotation and AI-Assisted Semiautomated Segmentation in Julia},
  author={Mitura, Jakub and Chrapko, Beata E. and Goyal, Divyansh},
  journal={JuliaHealth},
  year={2026}
}
```

Part of the **[JuliaHealth](https://juliahealth.org)** ecosystem.
