# Lutetium Workflows: Automated Dataset Preprocessing

The Lutetium therapy and diagnostic workflows (involving `Lu-177` SPECT/CT and `Ga-68` PET/CT scans) generate massive amounts of multi-modal, longitudinal imaging data. Manually parsing NIfTI files, matching spatial transforms, and registering baseline geometries at runtime is too slow for clinical use.

MedEye3d solves this via a robust, highly optimized automated preprocessing pipeline that aligns, crops, AI-segments, and compresses your longitudinal data into ultra-fast `HDF5` structures.

---

## 1. Directory Structure Requirements

Before running the preprocessor, ensure each patient dataset is stored in an independent folder inside a root `data/` directory (e.g., `data/pat_6_files/`). 

Each patient folder must contain a `scene_hierarchy.json` file exported from 3D Slicer. This file defines the rigid alignment transformations (using ITK `.tfm` matrices) from floating timepoint images (PET, SPECT) to the baseline CT.

**Example Folder**:
```
data/pat_6_files/
├── scene_hierarchy.json
├── Fixed_CT_Volume_0.nii.gz         # Baseline CT
├── SUV_PET_Image_0.nii.gz           # Baseline PET
├── PET_Lesions_0.seg.nrrd           # Baseline Lesion Masks
├── Fixed_CT_Volume_1.nii.gz         # TP1 CT
├── SUV_PET_Image_1.nii.gz           # TP1 PET
└── Transform_FollowUp_to_Baseline_1.tfm # ITK Matrix linking TP1 -> TP0
```

---

## 2. Running the Pipeline

To execute the pipeline across **all patients** in your directory, run the bash orchestrator from the project root:

```bash
./scripts/preprocess_all_cases.sh data
```

### What happens under the hood?

The pipeline launches `scripts/preprocess_dataset.jl` for each folder, which executes the following four phases:

#### Phase I: Scene Graph Parsing
The pipeline reads `scene_hierarchy.json` and builds a chronological graph of your data acquisitions. It automatically identifies root baseline nodes and correctly associates `.tfm` alignment files to downstream PET and SPECT nodes.

#### Phase II: ITK Spatial Resampling
All floating volumes (follow-up CTs, PETs, SPECTs, and Lesion Masks) are dynamically resampled onto the exact voxel grid (origin, spacing, direction) of the baseline `Fixed_CT_Volume_0.nii.gz`. 
* Images are resampled using continuous `Linear` interpolation.
* Discrete labels (Lesion Masks) are resampled using `Nearest Neighbor` interpolation.

#### Phase III: HDF5 Serialization
Instead of rewriting NIfTI files, the perfectly aligned float arrays and spatial headers are compressed natively into a monolithic HDF5 database: `preprocessed_volumes.h5`. 
Loading slices directly from byte offsets in this file allows the Interactive App to boot up in <2 seconds.

#### Phase IV: AI Bone Subsegmentation Precomputation
For osteoblastic and osteolytic lesions common in Lutetium therapies, distinguishing cortical bone boundaries from bone marrow is critical.
1. The pipeline automatically runs the `skellytour --subseg` neural network over your baseline CT.
2. It scans all longitudinal masks in your HDF5 file for unique lesion IDs.
3. For each identified lesion that intersects with the skeleton, it executes `bone_subsegmentation.py`.
4. The exact voxel coordinates of the AI-generated bone surface (`surf`) and bone marrow (`marr`) are baked into `Bone_Subsegments_0.h5`.

---

## 3. Using the Preprocessed Data

Once the pipeline completes, launch the MedEye3d Makie viewer:

```bash
julia --project=. scripts/run_interactive_mrb.jl data/pat_6_files
```

The viewer will detect `preprocessed_volumes.h5` and bypass traditional data loading. You will immediately have access to the **Compare Timepoints** mode and the **Skellytour Subsegments** toggles in the GUI.
