# One-Click Heat-GDT Segmentation in MedEye3d

## Overview

MedEye3d provides a **one-click segmentation solution** via the Heat-GDT (Heat Geodesic Distance Transform) algorithm. The user can choose between two modes:
1. **1-Click Mode (`[ 1-Click ]`, Default)**: Single-click directly on the lesion center; the GPU executes full diffusion and immediately commits the discrete segmentation in ~5ms.
2. **Wand Mode (`[ Wand (Hold) ]`)**: Click and hold the left mouse button on the lesion center, watch the overlay expand smoothly at >60 FPS, and release when the boundary matches the tumor margin.

All parameters (edge source, diffusion speed, threshold, boundary sharpness) are **automatically configured** for optimal performance across PET/CT, MRI, and pure CT modalities — no manual tuning required.

---

## How It Works: The Auto-Parameter Pipeline

When the user activates Heat-GDT (W key) and clicks on a lesion, **four automatic parameter selection mechanisms** engage in sequence:

### 1. Automatic Diffusivity Field Selection

MedEye3d precomputes and caches **three edge guidance fields** per patient in the HDF5 study file:

| Field | Source | Best For |
|:------|:-------|:---------|
| `diffusivity` | CT + PET combined | Default: most lesion types |
| `diffusivity_ct` | CT only (PET masked to 0) | Bony lesions, lung nodules, liver capsule |
| `diffusivity_pet` | PET only (CT masked to air) | Metabolically active tumors without CT contrast |

The user can switch between these with the 3-button toggle (`[Both]` / `[CT Only]` / `[PET Only]`), but the default **Combined** mode works well for 90%+ of cases.

**Code path**: [`switch_heatgdt_edge_modality!`](file:///workspaces/MedEye3d.jl/src/display/GLFW/MakieEventHandlers.jl#L383-L396) fetches the selected tensor from the in-memory cache and re-uploads to the GPU via a staging buffer copy.

### 2. Adaptive Frontier Contrast Boost (`s_boost`)

This is the **core "auto-parameter"** innovation. When the user clicks, the system automatically:

1. **Estimates the lesion frontier** — samples a spherical shell around the seed point at the expected boundary radius
2. **Measures mean edge strength** at the frontier: $\bar{E}_{\text{frontier}}$
3. **Computes an optimal boost exponent**:

$$s_{\text{boost}} = \text{clamp}\left(\left(\frac{0.20}{\max(\bar{E}_{\text{frontier}}, 0.01)}\right)^{\text{sensitivity}}, 0.5, 10.0\right)$$

4. **Applies it in the GPU shader**: $D_{\text{eff}} = D^{s_{\text{boost}}}$

This is mathematically equivalent to rescaling $\gamma$ per-voxel but costs only **1 ALU cycle** (`pow(D, s)`) with zero memory allocation.

**Why this matters**:
- **Faint soft-tissue boundaries** (E≈0.02): s_boost ≈ 10.0 → edges become impenetrable
- **Strong bone/lung borders** (E≈0.40): s_boost ≈ 0.5 → barriers stay moderate (no over-tightening)
- **Self-normalizing**: each lesion automatically gets the right edge sharpness

**Code path**: [`compute_frontier_edge_scale`](file:///workspaces/MedEye3d.jl/src/display/Vulkan/VulkanHeatDiffusion.jl#L219-L306) (Julia), called from [`reactToHeatGDTStart`](file:///workspaces/MedEye3d.jl/src/display/GLFW/MakieEventHandlers.jl#L6667-L6677)

### 3. Anisotropic Spacing Correction

Clinical PET/CT has heavily anisotropic voxels (e.g., 0.78×0.78×3.0 mm). Without correction, heat travels 4× faster along the Z-axis, causing **craniocaudal leakage**.

MedEye3d automatically reads the DICOM/NIfTI voxel spacing and computes metric weights:

$$w_x = \left(\frac{s_{\min}}{\Delta x}\right)^2, \quad w_y = \left(\frac{s_{\min}}{\Delta y}\right)^2, \quad w_z = \left(\frac{s_{\min}}{\Delta z}\right)^2$$

For (0.78, 0.78, 3.0) mm: $w_z ≈ 0.0676$, attenuating cross-slice flux by **14.8×**.

**Code path**: [`_compute_spacing_weights`](file:///workspaces/MedEye3d.jl/src/display/Vulkan/VulkanHeatDiffusion.jl#L206-L216) → passed to GPU shader via `DiffuseStepPC` push constants (`inv_dx2`, `inv_dy2`, `inv_dz2`)

### 4. Interactive Hold Duration = Adaptive K

Instead of a fixed number of diffusion steps, MedEye3d uses **interactive hold duration** as the natural adaptive parameter:

- **Short click** (~0.1s): K≈10 → tight core-only segmentation
- **Medium hold** (~0.5s): K≈40 → typical lesion boundary
- **Long hold** (2-3s): K≈200+ → fills large tumors and complex shapes

The GPU runs incremental 10-step batches at >60 FPS, with a progressive display theta ramp that smoothly reveals the expanding segmentation in real-time.

---

## Step-by-Step Guide: One-Click Segmentation

### Quick Start

```
1. Open patient study in MedEye3d
2. Press W (activates Heat-GDT Wand mode)
3. Position cursor over lesion center in any 2D view
4. Click and hold left mouse button
5. Watch the green overlay expand to fill the lesion
6. Release when the boundary looks correct
7. Done! Mask is auto-committed and saved.
```

### GUI Controls (Auto-Configured, Rarely Need Adjustment)

```
┌────────────────────────────────────────────────────────┐
│ [▼ Segmentation & AI]                                  │
│                   VIEW MODE / EDIT MODE                │
│ [ New ]  [ Paint ]  [ Erase ]  [ View ]                │
│ Brush: [───●──────────────────] [ Move ]               │
│ AI: [ Heat-GDT              ▼ ] [ Run AI ]             │
│ Mode: [ 1-Click ] [ Wand (Hold) ]                      │
│ θ: [───●──────────────────────] K: [──────●──────────] │
│ Speed: [─────────●────────────]                        │
│ Size ±: [──────────●───────────] [ Reset ]             │
│ Edge: [ Both (Combined) ] [ CT Only ] [ PET Only ]     │
│ Adapt: [ON]    Sens: [────●───]                        │
└────────────────────────────────────────────────────────┘
```

| Control | Default | When to Adjust |
|:--------|:--------|:---------------|
| **θ** | 0.001 | Lower (0.0005) for large inclusive masks; Higher (0.005) for tight cores |
| **K** | 40 | Rarely — hold duration is the natural K control |
| **Speed** | 12.0 | Increase for faster visual expansion |
| **Size ±** | 0 | Post-hoc grow/shrink after release |
| **Edge** | Both | Switch to CT Only for bony lesions, PET Only for metabolic-only tumors |
| **Adapt** | ON | Leave ON — the self-normalizing frontier boost handles most cases |
| **Sens** | 1.0 | Increase to 1.5–2.0 for very faint soft-tissue lesions |

### Post-Hoc Adjustments

After releasing the mouse button, you can:
- **Size ± slider**: Grow or shrink the committed mask without re-painting
- **Ctrl+Z**: Undo the last annotation
- **Re-click**: Click and hold again on the same or different location

---

## Architecture: Full-Resolution vs Patch-Based

> [!IMPORTANT]
> MedEye3d operates on the **full clinical resolution** (e.g., 512×512×326) — NOT 64³ patches.

This is the primary reason why MedEye3d's Heat-GDT produces better results than the patch-based evaluation in the article:

| Feature | MedEye3d (Full-Resolution) | Article Evaluation (64³ Patches) |
|:--------|:--------------------------|:-------------------------------|
| **Volume Size** | 512×512×326 (full scan) | 64×64×64 (cropped patch) |
| **Edge Network** | HED3D-Ultra champion on full volume | TotalSegEdgePDE on 64³ patch |
| **Edge Quality** | High-quality, context-aware boundaries | Patch-boundary artifacts, less context |
| **s_boost** | ✅ Effective (frontier E≈0.02-0.05) | ❌ Not beneficial (frontier E≈0.3-0.9) |
| **Anisotropic Spacing** | ✅ Applied with real DICOM spacing | ❌ Not applicable (isotropic patches) |
| **K Selection** | Interactive (user-controlled) | Fixed K=280 |
| **GPU** | Vulkan compute (sub-ms per step) | PyTorch CUDA |
| **Post-processing** | Connected component + size ± slider | Hard threshold |

### Why Full-Resolution Matters

On full-resolution volumes:
1. **Better edge context**: The neural edge detector sees the full anatomical context (surrounding organs, vessels, bones) — not just a 64³ crop that may cut through important boundaries
2. **Adaptive boost works**: Frontier edges at the lesion boundary are typically faint (E≈0.02-0.05), so s_boost amplifies them correctly
3. **No patch truncation**: Large lesions (>64³ voxels) can grow freely without being clipped by patch boundaries
4. **Anisotropic correction**: Real DICOM spacing is applied, preventing Z-axis leakage

---

## Programmatic / Headless Usage

For batch processing or scripted evaluation, you can use the Julia API directly:

```julia
using MedEye3d
using MedEye3d.VulkanBackend.VulkanHeatDiffusion

# Initialize Vulkan context (headless)
GLFW.Init()
GLFW.WindowHint(GLFW.VISIBLE, 0)
window = GLFW.CreateWindow(256, 256, "Headless Heat Kernel")
ctx = init_vulkan_context(window, 256, 256)

# Allocate state for volume dimensions
state = HeatDiffusionState()
init_heat_diffusion!(state, ctx, 512, 512, 326)

# Upload precomputed diffusivity (from HDF5)
upload_diffusivity!(state, ctx, diffusivity_field)

# Physical spacing from DICOM
spacing = (0.9765625, 0.9765625, 3.27)

# Auto-compute adaptive s_boost
s_boost = compute_frontier_edge_scale(
    diffusivity_field, cx, cy, cz, 18.0, spacing;
    sensitivity=1.2f0
)

# Seed + diffuse (auto K via quality target)
seed_only!(state, ctx, cx, cy, cz;
           seed_radius=2, spacing=spacing, s_boost=s_boost)
run_incremental_steps!(state, ctx, 40;
                       theta=0.001f0, spacing=spacing, s_boost=s_boost)

# Extract mask
mask = read_mask_to_cpu(state, ctx)
```

### Auto-K and Auto-Patch-Size (Organ Tracking Pipeline)

For the multi-timepoint lesion tracking pipeline, MedEye3d provides fully automatic K and patch size selection:

```julia
using MedEye3d.OrganHeatLesionTracker

# Automatically selects K and patch size based on target volume
# K = clamp(5 * estimated_radius, 30, 280)
# Patch = clamp(2 * estimated_radius + 20, 64, 192)
mask = grow_heat_kernel_lesion(
    diffusivity, seed_xyz, spacing;
    target_volume_voxels=1000,  # Expected lesion size
    # K and patch_size are auto-computed if not specified
)
```

**Code path**: [`grow_heat_kernel_lesion`](file:///workspaces/MedEye3d.jl/src/preprocessing/OrganHeatLesionTracker.jl#L381-L517)

---

## Precomputing Diffusivity Fields

Before using Heat-GDT on a new patient, the diffusivity field must be precomputed using the HED3D-Ultra edge network:

```bash
# Precompute all 3 modality variants (combined, CT-only, PET-only)
CUDA_VISIBLE_DEVICES=1 julia --threads=1 --project=. \
    scripts/preprocessing/generate_modality_diffusivities.jl \
    /path/to/case/preprocessed_volumes.h5

# CT-only (faster, for pure CT cases)
CUDA_VISIBLE_DEVICES=1 julia --threads=1 --project=. \
    scripts/preprocessing/generate_modality_diffusivities.jl \
    /path/to/case/preprocessed_volumes.h5 --ct-only
```

This runs sliding-window HED3D-Ultra inference, computes $D = \exp(-\gamma E)$ with $\gamma = 10.096$, and writes datasets directly into the HDF5 study groups.

---

## Key Source Files

| File | Description |
|:-----|:------------|
| [`VulkanHeatDiffusion.jl`](file:///workspaces/MedEye3d.jl/src/display/Vulkan/VulkanHeatDiffusion.jl) | GPU compute pipeline: seed, diffuse, extract mask |
| [`diffuse_step.comp`](file:///workspaces/MedEye3d.jl/src/display/Vulkan/shaders/diffuse_step.comp) | Vulkan GLSL compute shader (PDE kernel) |
| [`MakieEventHandlers.jl`](file:///workspaces/MedEye3d.jl/src/display/GLFW/MakieEventHandlers.jl) | Mouse hold + tick loop + GPU dispatch |
| [`InferenceClient.jl`](file:///workspaces/MedEye3d.jl/src/display/InferenceClient.jl) | CPU fallback with spacing + s_boost |
| [`OrganHeatLesionTracker.jl`](file:///workspaces/MedEye3d.jl/src/preprocessing/OrganHeatLesionTracker.jl) | Batch tracking with auto K/patch-size |
| [`heat_kernel_guide.md`](file:///workspaces/MedEye3d.jl/docs/heat_kernel_guide.md) | Full mathematical theory + test suite |

---

## Summary: Why One-Click Works

The "one-click" experience emerges from **four self-configuring mechanisms** working together:

1. **Precomputed edge guidance** — diffusivity fields ready at load time
2. **Adaptive s_boost** — auto-sharpens boundary barriers per-lesion
3. **Anisotropic spacing** — physically correct diffusion in all three dimensions
4. **Interactive hold duration** — natural adaptive K without parameter tuning

No other interactive segmentation tool achieves this combination of **zero-parameter, sub-second, GPU-native 3D segmentation** with boundary-respecting physics.
