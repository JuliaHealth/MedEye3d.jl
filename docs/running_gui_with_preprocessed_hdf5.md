# Running the MedImages / MedEye3d GUI with `preprocessed_volumes.h5`

This guide explains how to launch and operate the interactive medical visualization GUI (**MedEye3d.jl**) with the preprocessed 4-timepoint PSMA cohort dataset (`preprocessed_volumes.h5`).

---

## 1. Quick Launch Commands

### Option A: Standard Interactive Viewer (Recommended)
From the root of the `MedEye3d.jl` repository:

```bash
# Set display and thread configuration
export DISPLAY=:0
export HDF5_USE_FILE_LOCKING=FALSE

# Launch interactive 4-pane viewer
julia --project=. scripts/run_interactive_mrb.jl data/cases/psma_patient_all_tp
```

Or using the helper shell script:
```bash
./scripts/app/run_interactive_mrb.sh data/cases/psma_patient_all_tp
```

### Option B: Direct HDF5 Module Launcher (`AppMain.jl`)
If you want to invoke the compiled `AppMain` pipeline directly:

```bash
julia --project=. -e '
using MedEye3d.AppMain
MedEye3d.AppMain.launch_from_h5("data/cases/psma_patient_all_tp/preprocessed_volumes.h5")
'
```

### Option C: File Selector / Launcher Menu
```bash
julia --project=. -e '
using MedEye3d.AppMain
MedEye3d.AppMain.main(["data/cases/psma_patient_all_tp/preprocessed_volumes.h5"])
'
```

---

## 2. Environment Prerequisites

Ensure the following environment variables are set before starting the GUI:

1. **X11 Display**:
   - If running locally on Linux desktop: `export DISPLAY=:0`
   - If running over SSH: connect with `ssh -X` or `ssh -Y` to forward the X11 display.
2. **HDF5 File Locking**:
   - Prevent file-locking conflicts when multiple processes interact with HDF5:
     ```bash
     export HDF5_USE_FILE_LOCKING=FALSE
     ```
3. **Multi-Threading**:
   - Set multi-threading with a dedicated interactive thread for the Makie UI:
     ```bash
     export JULIA_NUM_THREADS=4,1
     ```
4. **Hardware Acceleration (OpenGL / Vulkan)**:
   - For NVIDIA GPU hardware acceleration:
     ```bash
     export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
     export VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
     ```
   - For Mesa OpenGL fallback (software rendering):
     ```bash
     export MESA_GL_VERSION_OVERRIDE=4.3
     export MESA_GLSL_VERSION_OVERRIDE=430
     ```

---

## 3. What Gets Loaded from `preprocessed_volumes.h5`

When you provide `data/cases/psma_patient_all_tp/preprocessed_volumes.h5`, MedEye3d bypasses slow raw NIfTI parsing and loads the dataset instantly:

1. **All 4 Resampled Timepoints**:
   - **TP 0 (Baseline)**: Whole-body PET/CT (`Fixed_CT_Volume_0`, `SUV_PET_Image_0`, `PET_Lesions_0`)
   - **TP 1 (Follow-up 1)**: Late PET/CT (`Fixed_CT_Volume_1`, `SUV_PET_Image_1`, `PET_Lesions_1`)
   - **TP 2 (Follow-up 2)**: Late PET/CT (`Fixed_CT_Volume_2`, `SUV_PET_Image_2`, `PET_Lesions_2`)
   - **TP 3 (Follow-up 3)**: mpMRI T2W + ADC/DWI (`Fixed_CT_Volume_3`, `SUV_PET_Image_3`, `PET_Lesions_3`)
2. **Registration Transforms (`_meta_/transforms/`)**:
   - PyTorch-optimized surface registration transforms (`Transform_FollowUp_to_Baseline_1..3.tfm`) are pre-applied to register all volumes onto the baseline grid.
3. **Anatomical Atlases (`ATLAS/`)**:
   - 117-class TotalSegmentator whole-body atlas (`ATLAS/max_anatomy`)
   - Cortical and trabecular skeleton atlas (`ATLAS/skellytour`)
   - Binary bone mask (`ATLAS/bone_atlas`)
   - 8-zone MRI prostate segmentation (`anatomy_out_fixed_ct_3/max_anatomy`)
4. **Bone Marrow & Cortical Subsegments (`Bone_Subsegments_0.h5`)**:
   - Pre-computed cortical bone surface (Green) and marrow (Red) overlays for skeletal metastases (`F08`, `F07`, `PT02`).
5. **Lesion Centroids & Clinical Names (`medeye3d_lesion_annotations.json`)**:
   - 27 clinical lesion targets (`PT01`, `PT02`, `P01`, `VOI2`, `F08`, `LN01`–`LN10`) with precomputed 3D bounding box coordinates and SUV metrics.

---

## 4. GUI Layout & Navigation

The viewer opens two synchronized windows:

### Window 1: 4-Pane Multi-Planar Orthogonal Viewer (QuadView)
- **Panel 1 (Top-Left)**: Transverse / Axial view (CT/MRI grayscale background + fused PET colormap overlay + segmentation mask).
- **Panel 2 (Top-Right)**: Sagittal view.
- **Panel 3 (Bottom-Left)**: Coronal view.
- **Panel 4 (Bottom-Right)**: Pure Nuclear Medicine viewer or 3D Maximum Intensity Projection (MIP).

### Window 2: Makie Clinical Control & Metadata Panel
- **Timepoint Navigation**: Slider to switch instantly between TP 0, TP 1, TP 2, and TP 3.
- **Lesion Selector Dropdown**: Jump directly to any of the 27 lesions; crosshairs in all 3 planes immediately center on the lesion centroid.
- **Bone Subsegment Toggles**: Toggle cortical surface and marrow subsegment visualization.
- **Anatomy Overlay Toggle**: Show/hide whole-body or prostate zone anatomical outlines.
- **SUV Metrics Inspector**: Live readout of $SUV_{\text{max}}$, $SUV_{\text{mean}}$, $SUV_{\text{peak}}$, and Metabolic Tumor Volume (MTV).

---

## 5. Mouse & Keyboard Controls

| Action | Control |
| :--- | :--- |
| **Scroll Slices** | Mouse Wheel (scrolls through the active plane) |
| **Crosshair Jump** | **Right-Click** on any point (centers Axial, Sagittal, Coronal planes to that 3D voxel) |
| **Maximize Panel** | **Double-Click** on any panel (expands to full screen / restores 4-pane) |
| **Window / Level** | **Middle-Click Drag** or **Shift + Left-Click Drag** (adjusts brightness/contrast) |
| **Paint / Edit Mask**| **Left-Click Drag** (draws segmentation voxels on active mask layer) |
| **Erase Mask** | **Ctrl + Left-Click Drag** (erases mask voxels) |
| **Next Lesion** | Press `N` or click next in Makie lesion dropdown |
| **Previous Lesion** | Press `P` or click previous in Makie lesion dropdown |
| **Cycle Timepoint**| Alt + Scroll or use Timepoint slider |
| **Exit Viewer** | Close window or press `Esc` |

---

## 6. Saving Edits

Any voxel painting or mask edits performed in the GUI are automatically saved back in-place to `preprocessed_volumes.h5` and metadata updates are recorded in `medeye3d_lesion_annotations.json` and `.h5`.
