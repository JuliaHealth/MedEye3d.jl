# API Reference

This document provides a comprehensive reference for all exported modules, data types, structs, enums, and functions in MedEye3d.jl.

---

## 1. Module `MedEye3d.SegmentationDisplay`

The primary OpenGL display engine.

### Functions

```julia
displayImage(imageSpecs; window_width=1000, window_height=1000) -> MainMedEye3d
```
Initializes GLFW, sets up ModernGL shaders, creates textures from the input image specifications, and launches the rendering consumer loop.
- **`imageSpecs`**: Tuple or Vector of `(filepath, modality_name)` or `TextureSpec` configurations.
- **Returns**: `MainMedEye3d` container holding the event channel and window references.

```julia
coordinateDisplay(calcDimsStruct, mainStates, mainChannel)
```
Configures the interactive viewport coordinates, sets mouse/keyboard callbacks, and binds the consumer channel.

---

## 2. Module `MedEye3d.InferenceClient`

Client bridge for containerized deep learning inference.

### Functions

```julia
start_python_worker(; port=5005) -> Bool
```
Checks if the `medeye3d-ai` Docker container is running and responding to TCP ping requests on `port`. If not running, launches `./scripts/ai/start_docker_worker.sh`.

```julia
run_helpnet_inference(ct_vol, pet_vol, click_x, click_y, click_z; port=5005) -> Array{UInt8, 3}
```
Extracts a $64 \times 64 \times 64$ patch centered at $[click\_x, click\_y, click\_z]$, normalizes CT and PET intensities, dispatches a TCP JSON request to the Docker worker, extracts the largest connected component on GPU, and returns the binary 3D prediction mask.

```julia
run_nninteractive(ct_vol, pet_vol, points_vol, click_x, click_y, click_z; port=5005) -> Array{UInt8, 3}
```
Runs prompt-based interactive CT segmentation using MIC-DKFZ NNInteractive foundation model with direct JSON coordinate transfer and in-memory session caching.

```julia
insert_patch!(full_volume, patch, start_x, start_y, start_z)
```
Inserts a 3D segmented sub-volume patch back into the full parent volume at the specified starting indices.

---

## 3. Module `MedEye3d.ConnectedComponents`

High-performance GPU-accelerated 3D Connected Component Labeling.

### Functions

```julia
extract_largest_connected_component(mask::AbstractArray{T, 3}; connectivity::Int=26, use_gpu::Bool=false) -> AbstractArray{T, 3}
```
Finds all connected foreground components in `mask` using `KernelAbstractions.jl` multi-pass relaxation kernels and returns a binary mask containing strictly the largest connected component (LCC).

```julia
label_connected_components(mask::AbstractArray{T, 3}; connectivity::Int=26, use_gpu::Bool=false) -> Tuple{AbstractArray{Int32, 3}, Int}
```
Labels every distinct connected component with an integer ID $1, 2, \dots, N$.

---

## 4. Module `MedEye3d.StrokeRasterization`

GPU and CPU continuous swept-capsule thick-line rasterizer for smooth manual painting and erasing.

### Functions

```julia
rasterize_thick_line!(mask::AbstractMatrix{T}, p1::Tuple{Int,Int}, p2::Tuple{Int,Int}, radius::Int, val::T; use_gpu::Bool=false) -> mask
```
Rasterizes a continuous thick line segment from `p1` to `p2` with brush radius `radius` directly into 2D matrix or SubArray `mask`.

```julia
rasterize_polyline!(mask::AbstractMatrix{T}, points::Vector{Tuple{Int,Int}}, radius::Int, val::T; use_gpu::Bool=false) -> mask
```
Rasterizes a continuous polyline connecting all consecutive coordinates in `points` with swept capsules of thickness `radius`.

---

## 5. Module `MedEye3d.BoneSubsegmentation`

Anatomical bone surface and marrow compartmentalization.

### Functions

```julia
extract_bone_subsegments(bone_mask::AbstractArray{T, 3}) -> Tuple{Array{UInt8, 3}, Array{UInt8, 3}}
```
Separates `bone_mask` into `(cortical_surface_mask, trabecular_marrow_mask)` using 3D morphological operators.

```julia
classify_lesion_bone_compartment(lesion_mask, bone_surface_mask, bone_marrow_mask) -> String
```
Classifies the anatomical localization of a lesion into `"PURE_SURFACE"`, `"PURE_MARROW"`, `"MIXED_TRANSMURAL"`, or `"SOFT_TISSUE_NON_BONE"`.

---

## 6. Core Data Structures & Events

### Struct `TextureSpec`
```julia
@with_kw struct TextureSpec
    name::String = "CT"
    isMainImage::Bool = true
    min_val::Float32 = -1000.0f0
    max_val::Float32 = 1000.0f0
    strokeWidth::Int = 3
    is_visible::Bool = true
    color::RGB = RGB(1.0, 1.0, 1.0)
end
```

### Enum `DisplayMode`
```julia
@enum DisplayMode begin
    SingleImage = 1
    MultiImage  = 2
    QuadImage   = 3
end
```

### Event Structs (`MedEye3d.MakieEvents`)
- `ChangeTimePointEvent(delta::Int)`: Cycles active longitudinal time point.
- `CompareTimePointsEvent()`: Toggles 2-panel comparative layout.
- `ScrollZoomEvent(zoom_delta::Float64)`: Adjusts zoom factor.
- `ShowMaskLayerEvent(layer_name::String, is_visible::Bool)`: Updates OpenGL texture visibility.
- `WindowingEvent(texture_name::String, min_val::Float32, max_val::Float32)`: Adjusts intensity thresholds.
- `DoubleClickEvent(x::Float64, y::Float64, panel_index::Int)`: Toggles panel maximization.
- `SyncLesionEvent(lesion_id::Int)`: Jumps slice cameras to the centroid of the specified lesion.
