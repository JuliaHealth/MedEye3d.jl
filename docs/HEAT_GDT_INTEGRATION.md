# MedEye3d Heat-GDT Integration

## Overview

Heat-GDT is a physics-informed interactive segmentation method that uses inhomogeneous heat diffusion over a learned edge field for real-time lesion segmentation in PET/CT.

## Architecture

- **Two-phase design**: Offline edge precomputation + Real-time heat diffusion query
- **Edge network**: HED3D (14.14M parameters) runs once per volume load
- **Heat diffusion PDE**: Runs on Vulkan compute shaders for sub-millisecond response
- **Diagram**: 
  Volume Load &rarr; HED3D Edge Network &rarr; Diffusivity Field $D(x)$ &rarr; [stored on GPU] &rarr; User Click &rarr; Vulkan Heat Diffusion (40 steps) &rarr; Mask &rarr; Display

## Vulkan Compute Pipeline

- **4 compute shaders**: `diffusivity.comp`, `diffuse_step.comp`, `extract_mask.comp`, `seed_init.comp`
- **Ping-pong buffer strategy** for iterative diffusion
- **Direct write** to label texture (already on Vulkan GPU)
- **Pipeline barriers** between compute and render passes

## Mouse Interaction

- **G key**: Toggle Heat-GDT mode on/off (also syncs algorithm dropdown in GUI)
- **E key**: Must enter edit/paint mode before hold-to-grow works
- **Click**: Place seed point, run heat diffusion with `K_base` steps
- **Hold**: Duration increases `K` dynamically (segmentation grows while holding)
  - Formula: `K_total = K_base + round(hold_duration_s × 30)`, capped at 200
- **Slider**: `theta` controls threshold distance, `K_base` controls starting reach
- **Release**: Finalize mask, async copy to CPU for saving

> **Important**: Heat-GDT hold-to-grow only works when:
> 1. Heat-GDT mode is active (press **G** or select "Heat-GDT" in dropdown)
> 2. Edit/paint mode is active (press **E**)
> 3. Mouse held > 100ms (to distinguish from regular clicks)
> 4. Pre-painted scribbles exist for the current lesion

## Preprocessing

- Generate edge/diffusivity volume from trained HED3D model
- Store as additional channel in HDF5 alongside CT, PET, TotalSegmentator
- **Script**: `scripts/ai/generate_edge_image.jl`

## Integration Points

- `src/ai/AIInference.jl`: `run_heatgdt_inference` function
- `src/display/InferenceClient.jl`: `run_heatgdt` client function
- `src/display/Vulkan/VulkanHeatDiffusion.jl`: GPU compute module
- `src/display/LesionMetadataWindow.jl`: GUI sliders and method selector

## Performance Targets

- **Edge precomputation**: ~90ms (once per volume)
- **Interactive click**: <0.5ms (Vulkan compute)
- **Mean Dice**: 0.4287 (46.5% better than HELPNet's 0.2925)
