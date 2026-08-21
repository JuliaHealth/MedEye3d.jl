# Code Examples & Tutorials

This page provides practical, copy-pasteable code examples demonstrating the full feature set of MedEye3d.jl.

---

## Example 1: Basic 3D CT Volume Viewer

Load and display a single volumetric CT image:

```julia
using MedEye3d

# Define the image path and modality tag
ct_path = "data/pat_6_files/Fixed_CT_Volume_0.nii.gz"
ct_spec = (ct_path, "CT")

# Launch the OpenGL viewport
main_handle = MedEye3d.SegmentationDisplay.displayImage(ct_spec)
```

---

## Example 2: Multi-Modal Fused PET/CT with Mask Annotation

Load a registered PET/CT pair with an editable segmentation mask layer:

```julia
using MedEye3d
using MedEye3d.ForDisplayStructs

# Define primary background (CT) and overlay (PET)
ct_spec  = ("data/pat_6_files/Fixed_CT_Volume_0.nii.gz", "CT")
pet_spec = ("data/pat_6_files/SUV_PET_Image_0.nii.gz", "PET")
seg_spec = ("data/pat_6_files/PET_Lesions_0.nii.gz", "manualModif")

# Launch fused multi-layer viewer
main_handle = MedEye3d.SegmentationDisplay.displayImage((ct_spec, pet_spec, seg_spec))
```

---

## Example 3: Programmatic QuadView Orthogonal Display

Configure the 4-panel viewport showing Axial Fused, Axial Pure PET, Sagittal, and Coronal views:

```julia
using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs

# Prepare TextureSpecs for multi-panel rendering
specs_panel1 = [
    TextureSpec(name="CT", isMainImage=true, min_val=-1000.0f0, max_val=1000.0f0),
    TextureSpec(name="PET", isMainImage=false, min_val=0.0f0, max_val=15.0f0),
    TextureSpec(name="manualModif", isMainImage=false)
]

specs_panel2 = [
    TextureSpec(name="PET", isMainImage=true, min_val=0.0f0, max_val=15.0f0)
]

# Launch QuadImage mode
# (See scripts/app/run_interactive_quad.jl for full pipeline integration)
```

---

## Example 4: Longitudinal Therapy Comparison (Time Points)

Navigate across longitudinal imaging sessions (Baseline TP0 vs Follow-up TP1):

```julia
using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.MakieEvents

# Emitting a ChangeTimePointEvent into the core channel:
# channel = main_handle.channel

# Navigate to Next Time Point (+1)
put!(channel, ChangeTimePointEvent(1))

# Navigate to Previous Time Point (-1)
put!(channel, ChangeTimePointEvent(-1))

# Toggle Side-by-Side Comparison Overlay Mode
put!(channel, CompareTimePointsEvent())
```

---

## Example 5: Running AI Semiauto Segmentation

Trigger deep learning inference (HELPNet or NNInteractive) programmatically from Julia:

```julia
using MedEye3d
using MedEye3d.InferenceClient

# Ensure the Docker worker is active:
# ./scripts/ai/start_docker_worker.sh

# 1. HELPNet (3D Lesion Segmentation from Single Click)
click_x, click_y, click_z = 256, 256, 150
pred_mask_helpnet = run_helpnet_inference(ct_vol, pet_vol, click_x, click_y, click_z)

println("HELPNet segmented $(count(pred_mask_helpnet .> 0)) voxels.")

# 2. NNInteractive (Foundation Model Interactive Scribble Segmentation)
# scribble_vol is a binary 3D array containing user strokes
pred_mask_nninteractive = run_nninteractive(ct_vol, pet_vol, scribble_vol, click_x, click_y, click_z)

println("NNInteractive segmented $(count(pred_mask_nninteractive .> 0)) voxels in ~0.28s.")
```

---

## Example 6: GPU-Accelerated CCL Post-Processing

Extract strictly the largest connected component (LCC) from a noisy segmentation mask using `KernelAbstractions.jl`:

```julia
using MedEye3d.ConnectedComponents
using CUDA

# Create a noisy 3D mask with two separate lesions
raw_mask = zeros(UInt8, 128, 128, 128)
raw_mask[20:30, 20:30, 20:30] .= 1   # Small noise cluster (10^3 voxels)
raw_mask[50:80, 50:80, 50:80] .= 1   # Large target lesion (30^3 voxels)

# GPU execution if CUDA is available, CPU fallback otherwise:
use_gpu = CUDA.functional()
gpu_or_cpu_mask = use_gpu ? CuArray(raw_mask) : raw_mask

# In-place LCC extraction
cleaned_mask = extract_largest_connected_component(gpu_or_cpu_mask; connectivity=26)

# Verify that only the largest component was retained
println("Voxel count before: ", count(raw_mask .> 0))
println("Voxel count after:  ", count(Array(cleaned_mask) .> 0))
```
