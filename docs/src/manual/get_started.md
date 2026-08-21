# Getting Started with MedEye3d.jl

This guide walks you through setting up MedEye3d.jl, configuring your Julia multi-threading environment, starting the containerized AI worker, and loading your first medical image datasets.

---

## 1. System Requirements & Prerequisites

- **Julia Version**: Julia **1.9 or later** is strictly required (due to the interactive threadpool).
- **GPU & OpenGL**: OpenGL 3.3+ compatible GPU driver (`NVIDIA`, `AMD`, or `Intel`).
- **NVIDIA CUDA (Optional, Recommended)**: For GPU-accelerated KernelAbstractions and deep learning inference.
- **Docker**: For running the containerized AI inference server (`medeye3d-ai`).

---

## 2. The Interactive Thread Requirement

OpenGL context creation and GLFW window loops must run on a dedicated thread to avoid context migration across OS threads. In Julia 1.9+, this is handled via the **interactive threadpool**.

### Configuring the Interactive Thread
Launch Julia with at least 1 interactive thread (the number after the comma):

```bash
# Terminal export (recommended)
export JULIA_NUM_THREADS=3,1

# Launching Julia directly
julia --threads 3,1
```

> **Note:** If MedEye3d is loaded in a Julia session without an interactive thread, it will throw an informative initialization error directing you to configure `JULIA_NUM_THREADS`.

---

## 3. Installation

Install MedEye3d.jl via the Julia package manager:

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaHealth/MedEye3d.jl.git")
```

---

## 4. Starting the AI Inference Worker (Optional)

If you plan to use AI semiauto segmentation (**HELPNet** or **NNInteractive**), start the persistent Docker inference container:

```bash
# From the repository root
./scripts/ai/start_docker_worker.sh
```

The script builds (if not already cached) and launches the `medeye3d-ai` container in the background, mounting the shared `tmp_inference/` directory and exposing TCP port `5005`.

---

## 5. Loading Your First Medical Image

### Single Modality CT Display
To visualize a single 3D NIfTI or DICOM volume:

```julia
using MedEye3d

# Tuple format: (filepath, label_name)
ct_spec = ("path/to/ct_scan.nii.gz", "CT")

# Launches the interactive GLFW OpenGL viewer
viewer = MedEye3d.SegmentationDisplay.displayImage(ct_spec)
```

### Multi-Modal Fused PET/CT Display
To visualize fused multi-modal volumes with alpha blending:

```julia
using MedEye3d

ct_spec  = ("path/to/ct_scan.nii.gz", "CT")
pet_spec = ("path/to/suv_pet.nii.gz", "PET")

# Provide a tuple of image specifications
viewer = MedEye3d.SegmentationDisplay.displayImage((ct_spec, pet_spec))
```

---

## 6. Understanding Medical Coordinate Systems

Medical image volumes have physical voxel spacings $(\Delta_x, \Delta_y, \Delta_z)$ in millimeters (e.g. $0.976\,\text{mm} \times 0.976\,\text{mm} \times 3.0\,\text{mm}$).

MedEye3d automatically computes the **true physical aspect ratio**:
$$\text{Physical Ratio} = \frac{\Delta_{\text{vertical}}}{\Delta_{\text{horizontal}}} \times \frac{H_{\text{pixels}}}{W_{\text{pixels}}}$$

This guarantees:
- Slices extracted along any axis (Axial, Sagittal, Coronal) maintain undistorted physical proportions.
- Circular lesions and anatomical structures remain spherical and true to physical reality.
- Zero black padding wasted at the top and bottom of wide viewports.

---

## 7. Next Steps

- Explore [Code Examples & Tutorials](code_example.md) for programmatic scripting and advanced workflows.
- Learn about the [QuadView 4-Panel Layout](quad_view_and_navigation.md).
- Discover the [AI Inference Pipeline](ai_inference_pipeline.md) and [GPU Post-Processing](gpu_kernels_and_postprocessing.md).
