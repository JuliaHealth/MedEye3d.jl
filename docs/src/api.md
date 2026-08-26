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

---

## 7. Module `MedEye3d.LesionMetadataWindow`

Clinical annotation panel with structured metadata, automated quantitative analysis, and longitudinal tracking.

### Functions

```julia
create_metadata_window(active_lesion_id, lesion_ids, channel; save_path, ui_hooks) -> Figure
```
Creates the complete GLMakie metadata window with all annotation fields, searchable dropdowns, type buttons, and automated analysis labels.

```julia
compute_lesion_volume(lid::Int, tp_idx::Int) -> Dict{String, Float64}
```
Computes lesion volume from the cached segmentation mask at a specific time point. Returns `Dict("volume_mm3", "volume_cc", "voxel_count", "diameter_mm")`. Results are cached for performance.

```julia
compute_match_analysis(lid::Int, tp_idx::Int) -> Union{MatchAnalysisResult, Nothing}
```
Finds matched lesions across time points, computes volume and SUVmax at both current and baseline, calculates deltas, and classifies RECIP 1.0 response. Returns `nothing` if the lesion is not in any match group.

```julia
format_match_analysis(result::MatchAnalysisResult) -> String
```
Formats match analysis result as a human-readable string:
`"Vol: 1.24cc (13.3mm⌀) | ΔVol: +35.2% (+0.32cc) | ΔSUV: +2.1 | Grp 5 [3 TPs] RECIP-PD"`

```julia
compute_promise_score(suv_max::Float32, bg::Dict{String, Float32}) -> Int
```
Computes PROMISE molecular imaging score (0–3) from lesion SUVmax relative to blood pool, liver, and parotid reference SUVs.

```julia
parse_suv_fields(suv_str::String) -> Dict{String, Float32}
```
Parses `"Max: 12.3 ; Parotid: 8.1 ; Liver: 5.2 ; Blood: 2.8"` into a typed dictionary.

### Struct `MatchAnalysisResult`
```julia
struct MatchAnalysisResult
    group_id::Int
    current_volume_mm3::Float64
    current_volume_cc::Float64
    current_diameter_mm::Float64
    current_suv_max::Float32
    baseline_volume_mm3::Float64
    baseline_volume_cc::Float64
    baseline_suv_max::Float32
    baseline_node::String
    baseline_lid::Int
    volume_delta_pct::Float64     # (current - baseline) / baseline * 100
    volume_delta_abs_cc::Float64  # current - baseline in cc
    suv_delta_abs::Float32        # current - baseline SUVmax
    suv_delta_pct::Float64        # (current - baseline) / baseline * 100
    recip_category::String        # "RECIP-CR", "RECIP-PR", "RECIP-SD", "RECIP-PD"
    n_timepoints::Int
end
```

---

## 8. Module `MedEye3d.LesionAssociation`

Cross-timepoint lesion matching and organ mapping.

### Functions

```julia
load_matches_from_h5(h5_path::String) -> Dict{Int, Vector{Tuple{String,Int,String}}}
```
Loads match groups from the `_meta_/matches.json` HDF5 dataset. Returns global `MATCH_GROUPS` dict.

```julia
get_match_groups() -> Dict{Int, Vector{Tuple{String,Int,String}}}
```
Returns the in-memory match groups dictionary. Each group maps `group_id → Vector{(node_name, segment_int, display_name)}`.

```julia
find_cross_tp_lesion(src_node::String, src_id::Int, dst_node::String) -> Vector{Int}
```
Finds matching lesion segment IDs in the destination time point.

```julia
update_match_group!(src_node, src_id, dst_node, dst_id, h5_path)
```
Links two lesions across time points and persists to HDF5.

```julia
remove_from_match_group!(node, seg_int, h5_path)
```
Unlinks a lesion from its match group and persists to HDF5.

```julia
map_lesions_to_organs(lesion_mask, ts_atlas, ts_names) -> Dict{Int, String}
```
Maps each lesion segment integer to its nearest TotalSegmentator organ name via centroid lookup.

```julia
classify_organ_to_lesion_type(organ_name::String) -> String
```
Classifies a TotalSegmentator organ name into `"Prostate"`, `"Bone Meta"`, `"Lymph Node Meta"`, `"Muscle"`, or `"Organ Meta"`.

---

## 9. Module `MedEye3d.LesionTracker`

Batch longitudinal tracking and reporting.

### Functions

```julia
track_lesions(data_dir; matches_json, output_path) -> Vector{LesionTrackingEntry}
```
Reads match groups, loads NIfTI volumes, computes per-lesion volume and SUV statistics (mean, max, std), and writes a structured JSON report.

### Struct `LesionTrackingEntry`
```julia
struct LesionTrackingEntry
    group_id::Int
    node_name::String
    lesion_name::String
    raw_lesion::String
    segment_int::Int
    volume_mm3::Float64
    match_type::String       # "parent", "IoU", "PROXIMITY", "ARTIFICIAL", "manual"
    iou::Float64
    mean_intensity::Float64
    max_intensity::Float64
    std_intensity::Float64
    has_intensity::Bool
end
```

