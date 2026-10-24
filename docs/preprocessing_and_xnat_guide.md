# MedEye3D: Preprocessing Pipelines, AI Models, Docker & XNAT Integration Guide

This guide provides an end-to-end operational manual for assembling, preprocessing, and deploying multi-modal, longitudinal patient cases into **MedEye3D**, with automated storage and discovery in **XNAT**.

---

## 1. Overview & Architectural Goals

Clinical imaging in oncology involves complex, multi-modal, longitudinal datasets:
- **Lutetium-177 Therapy Cases (`Lu177_PSMA_PIPELINE`)**: Screening whole-body $^{68}\text{Ga}$-PSMA PET/CT paired with sequential post-therapy $^{177}\text{Lu}$ SPECT/CT acquisitions (e.g. 4h, 24h, 72h post-cycle).
- **Bimodal PSMA PET/CT + mpMRI Cases (`Prostata_bimodal_PIPELINE`)**: Co-registered whole-body PSMA PET/CT paired with high-resolution multi-parametric prostate MRI (T2w, ADC, DWI b=2000, DCE).

Loading dozens of raw 3D DICOM/NIfTI series and aligning them on the fly at runtime is too slow for clinical review. **MedEye3D solves this via automated offline or on-demand preprocessing**:
1. All scans across all timepoints are spatially resampled into the unified voxel coordinate grid of the baseline CT.
2. AI segmentation models (TotalSegmentator, Skellytour, nnInteractive/HELPNet, Lymph Node Rules) precompute anatomical atlases, bone cortical/marrow compartments, and lesion boundaries.
3. Everything is packed into a monolithic, chunked, compressed **HDF5 container** (`preprocessed_volumes.h5`).
4. The preprocessed HDF5 is uploaded to XNAT under the `MEDEYE3D` resource, making it instantly discoverable and streamable by any MedEye3D workstation.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 XNAT Cloud Archive                                     │
│  - Raw DICOM Scans & Slicer Bundles (scans/, slicer_3.6/, slicer_final/)               │
│  - Preprocessed Archives (resources/MEDEYE3D/files/preprocessed_volumes.h5)            │
│  - Audit Status (resources/ANNOTATIONS/files/status.json)                              │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Standalone Preprocessing Pipeline                               │
│  1. Download: xnat_preproc_worker.jl download                                          │
│  2. AI Inference: TotalSegmentator (117 organs) + Skellytour (bone marrow/cortex)      │
│  3. Rules & Landmarks: Landmarks.jl + PrimaryHDF5.jl (Lymph node station mapping)      │
│  4. Spatial Alignment: ITK Affine Resampling (CT-CT, SPECT-CT, or MRI-CT)              │
│  5. Packaging: preprocess_dataset.jl -> unified HDF5                                   │
│  6. Upload: XnatClient.upload_preprocessed_hdf5 -> XNAT MEDEYE3D                      │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        MedEye3D Clinical Workstation                                  │
│  - Instant boot (<2s per case) directly from local cache or streaming XNAT            │
│  - QuadView: Synchronized Axial, Coronal, Sagittal + Secondary Compare Window (M2)    │
│  - Full case navigation across all 420+ Lu177 cases & 130+ Prostate bimodal cases       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Docker & AI Environment Setup

Deep learning inference (TotalSegmentator, Skellytour, nnInteractive, nnU-Net) requires PyTorch with NVIDIA CUDA GPU acceleration.

### 2.1. Docker Container: `medeye3d-ai`

The worker image is defined in [`scripts/ai/Dockerfile`](file:///workspaces/MedEye3d.jl/scripts/ai/Dockerfile):
- **Base Image**: `pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime`
- **System Utilities**: `dcm2niix`, `git`, `unzip`, `curl`, `wget`
- **Python Libraries**: `TotalSegmentator`, `Skellytour`, `nnunetv2`, `monai`, `SimpleITK`, `h5py`, `requests`
- **nnInteractive**: MIC-DKFZ interactive 3D segmentation engine

#### Building the Docker Image
```bash
cd /workspaces/MedEye3d.jl
docker build -t medeye3d-ai:latest -f scripts/ai/Dockerfile scripts/ai/
```

#### Running the Container with GPU Passthrough
```bash
docker run -d --restart=unless-stopped \
  --name medeye3d-ai \
  --gpus all \
  --shm-size=16g \
  -p 5005:5005 \
  -v "${PWD}/tmp_inference:/tmp/medeye3d_inference" \
  -v "${PWD}/scripts/ai:/app" \
  medeye3d-ai:latest
```

* **Windows PowerShell**: Use [`scripts/ai/start_docker_worker.ps1`](file:///workspaces/MedEye3d.jl/scripts/ai/start_docker_worker.ps1) which automatically detects Docker Desktop and starts the worker with GPU passthrough.
* **Linux Shell**: Use [`scripts/ai/start_docker_worker.sh`](file:///workspaces/MedEye3d.jl/scripts/ai/start_docker_worker.sh).

---

## 3. Preprocessing Code & Standalone Scripts

| Script | Language | Purpose |
|---|---|---|
| [`scripts/xnat/xnat_preprocess_patient.sh`](file:///workspaces/MedEye3d.jl/scripts/xnat/xnat_preprocess_patient.sh) | Bash | Master orchestrator: downloads raw scans, runs Skellytour & rules, packages HDF5, and uploads to XNAT. |
| [`scripts/xnat/xnat_preproc_worker.jl`](file:///workspaces/MedEye3d.jl/scripts/xnat/xnat_preproc_worker.jl) | Julia | CLI worker for single actions: `download`, `process`, `upload`, `full`. |
| [`scripts/preprocessing/preprocess_dataset.jl`](file:///workspaces/MedEye3d.jl/scripts/preprocessing/preprocess_dataset.jl) | Julia | Multi-timepoint spatial resampling, scene graph parsing, and HDF5 packager. |
| [`scripts/preprocessing/recompute_bones.jl`](file:///workspaces/MedEye3d.jl/scripts/preprocessing/recompute_bones.jl) | Julia | Computes cortical bone surface (`surf`) vs bone marrow (`marr`) per skeletal lesion. |
| [`scripts/preprocessing/lymph_node_rules/.../PrimaryHDF5.jl`](file:///workspaces/MedEye3d.jl/scripts/preprocessing/lymph_node_rules/src/julia_gpu/preprocessing/PrimaryHDF5.jl) | Julia | Anatomical landmark extraction and multi-mask HDF5 aggregation. |
| [`scripts/registration/rerun_mri_registration.py`](file:///workspaces/MedEye3d.jl/scripts/registration/rerun_mri_registration.py) | Python | Surface-based 11-organ registration of mpMRI to baseline CT. |
| [`scripts/preprocessing/prepare_psma_case_for_medeye3d.py`](file:///workspaces/MedEye3d.jl/scripts/preprocessing/prepare_psma_case_for_medeye3d.py) | Python | Assembles multi-timepoint PSMA PET/CT + mpMRI directory structure. |
| [`scripts/preprocessing/reprocess_mri_tps.jl`](file:///workspaces/MedEye3d.jl/scripts/preprocessing/reprocess_mri_tps.jl) | Julia | Resamples MRI timepoints (TP 3-6) into baseline frame using registration transforms. |
| [`scripts/preprocessing/embed_prostate_anatomy.jl`](file:///workspaces/MedEye3d.jl/scripts/preprocessing/embed_prostate_anatomy.jl) | Julia | Embeds prostate zonal anatomy (PZ, TZ, SV) into the HDF5 volume. |

---

## 4. Reusing Pre-Existing TotalSegmentator & Lymph Node Masks

Running TotalSegmentator from scratch on every patient requires 10–20 minutes of GPU compute. If you already have segmentations ready (e.g. from prior pipeline runs, XNAT `slicer_*` resources, or batch cluster jobs), you can skip re-computation and feed them directly into the preprocessor.

### 4.1. Directory Structure & File Naming

Place the segmentation masks in `<case_dir>/segmentations/` (or `<case_dir>/segmentations_fast_backup/`):

```
<case_dir>/
├── Fixed_CT_Volume_0.nii.gz         # (or Fixed_CT_Volume.nii.gz)
├── SUV_PET_Image_0.nii.gz           # Baseline PET
├── PET_Lesions_0.nii.gz             # Lesion mask
└── segmentations/                   # Ready-made organ & lymph node masks
    ├── liver.nii.gz
    ├── spleen.nii.gz
    ├── kidney_left.nii.gz
    ├── kidney_right.nii.gz
    ├── urinary_bladder.nii.gz
    ├── prostate.nii.gz
    ├── aorta.nii.gz
    ├── inferior_vena_cava.nii.gz
    ├── iliac_artery_left.nii.gz
    ├── iliac_artery_right.nii.gz
    ├── iliac_vena_left.nii.gz
    ├── iliac_vena_right.nii.gz
    ├── trachea.nii.gz
    ├── esophagus.nii.gz
    ├── heart.nii.gz
    ├── stomach.nii.gz
    ├── pancreas.nii.gz
    ├── colon.nii.gz
    ├── duodenum.nii.gz
    ├── small_bowel.nii.gz
    ├── gluteus_maximus_left.nii.gz
    ├── gluteus_maximus_right.nii.gz
    ├── iliopsoas_left.nii.gz
    ├── iliopsoas_right.nii.gz
    ├── vertebrae_*.nii.gz
    ├── rib_left_*.nii.gz
    ├── rib_right_*.nii.gz
    ├── hip_left.nii.gz
    ├── hip_right.nii.gz
    ├── femur_left.nii.gz
    ├── femur_right.nii.gz
    ├── sacrum.nii.gz
    └── details/                     # Optional: specialized substructures
```

### 4.2. Supported Formats & Suffix Normalization

The preprocessor automatically strips pipeline suffixes to match standard ontology keys:
* `liver_primary_totalsegmentator.nii.gz` $\to$ `liver`
* `vertebrae_L1_primary_totalsegmentator.nii.gz` $\to$ `vertebrae_L1`
* `mandible_fallback_2_slicerdentalsegmentator.nii.gz` $\to$ `mandible`
* Supported extensions: `.nii.gz`, `.nii`, and native `.h5` (read via `voxel_data`).

### 4.3. Landmark Calculation from Ready Masks

[`Landmarks.jl`](file:///workspaces/MedEye3d.jl/scripts/preprocessing/lymph_node_rules/src/julia_gpu/preprocessing/Landmarks.jl) extracts anatomical coordinate reference points directly from the masks:
- **Carina Landmark**: Extracted from the bifurcation of `trachea` into main bronchi (used to separate superior from inferior mediastinal lymph nodes).
- **Aortic Arch Apex**: Extracted from the highest curvature of `aorta` (used for station 4R/4L/5).
- **Aortic Bifurcation**: Extracted where `aorta` splits into `iliac_artery_left` and `iliac_artery_right` (separates retroperitoneal/para-aortic from common iliac lymph nodes).
- **Internal Iliac Origin**: Separates internal iliac, external iliac, and obturator stations.
- **Digastric Muscle / Mandible**: Separates level I submental/submandibular from level II upper jugular.

### 4.4. Packaging Ready Masks into HDF5

Run `PrimaryHDF5.build_primary_hdf5`:
```bash
julia --project=scripts/preprocessing/lymph_node_rules/src/julia_gpu \
  scripts/preprocessing/lymph_node_rules/src/julia_gpu/preprocessing/preprocess_single.jl \
  <case_dir> --output <case_dir>/primary_masks.h5
```
This loads the CT and all NIfTI masks, validates their alignment, extracts the anatomical landmarks, and serializes everything into `primary_masks.h5` with chunks `(64, 64, 32)` and compression level 1.

---

## 5. Lutetium Pipeline Walkthrough (`Lu177_PSMA_PIPELINE`)

Metastatic prostate cancer patients treated with $^{177}\text{Lu}$-PSMA undergo a baseline $^{68}\text{Ga}$-PSMA PET/CT followed by serial $^{177}\text{Lu}$ SPECT/CT scans.

### 5.1. Input Components
1. **Baseline Whole-Body PSMA PET/CT (TP 0)**:
   - `Fixed_CT_Volume_0.nii.gz`: Primary spatial coordinate reference.
   - `SUV_PET_Image_0.nii.gz`: Quantitative SUV calibrated PET.
   - `PET_Lesions_0.nii.gz`: Expert or AI lesion masks.
   - `Skellytour_0.nii.gz`: Skeletal cortical vs marrow segmentation.
2. **Post-Therapy $^{177}\text{Lu}$ SPECT/CT (TP 1..N)**:
   - `Fixed_CT_Volume_1.nii.gz`: Follow-up CT scan.
   - `SUV_PET_Image_1.nii.gz`: Raw or normalized $^{177}\text{Lu}$ SPECT volume.
   - `Transform_FollowUp_to_Baseline_1.tfm`: ITK Affine transform matrix aligning TP 1 CT to Baseline CT.
3. **`scene_hierarchy.json`**:
   Specifies the chronological hierarchy and transform graph connecting follow-up scans to baseline.

### 5.2. SPECT Intensity Normalization
Unlike calibrated PET SUV, $^{177}\text{Lu}$ SPECT scans contain raw non-SUV counts. The preprocessor computes the 99th percentile ($P_{99}$) of positive voxel intensities and scales values dynamically:
$$\text{Scale Factor} = \frac{8.0}{\max(P_{99}, 1.0)}, \quad \text{SPECT}_{\text{scaled}} = \max\left(0, \text{SPECT} \times \text{Scale Factor}\right)$$
This aligns SPECT dynamic range with standard PET SUV display scales ($0\text{–}15$), enabling meaningful visual comparison in the QuadView and M2 viewer.

### 5.3. Execution Command
```bash
# Preprocess a single Lutetium patient
./scripts/xnat/xnat_preprocess_patient.sh \
  --project FDM_DPI-2024-7-KRN_Lu177_PSMA_PIPELINE \
  --subject pat6 \
  --overwrite
```

---

## 6. PET PSMA - MRI Bimodal Pipeline Walkthrough (`Prostata_bimodal_PIPELINE`)

Bimodal prostate cancer cases pair whole-body PSMA PET/CT with multiparametric prostate MRI (mpMRI).

### 6.1. Input Sequences
- **Whole-Body PET/CT**: Staging PET/CT (Early and/or Late timepoints).
- **Prostate mpMRI Sequences**:
  - `t2_tra_tse_0001.nii.gz`: T2W Axial TSE (primary anatomical reference).
  - `t2_sag_tse_0003.nii.gz`: T2W Sagittal.
  - `t2_cor_tse_0002.nii.gz`: T2W Coronal.
  - `zoomit_diff_tra_adc.nii.gz`: Apparent Diffusion Coefficient (ADC) map.
  - `zoomit_diff_tra_calc_bval.nii.gz`: High b-value DWI ($b=2000$).
  - `km_t1_tra_vibe_dixon_w.nii.gz`: Contrast-enhanced T1W Dixon.
  - `multilabel_mask.nii.gz` & `labels_index.json`: Multi-reader prostate and intra-prostatic lesion segmentations.

### 6.2. Dedicated MRI Preprocessing & Registration Scripts

#### Step 1: 11-Organ Surface Registration (`rerun_mri_registration.py`)
Rigid CT-to-CT registration fails between whole-body CT and dedicated prostate MRI due to different patient positioning, pelvic tilt, and organ deformations.
[`scripts/registration/rerun_mri_registration.py`](file:///workspaces/MedEye3d.jl/scripts/registration/rerun_mri_registration.py) performs non-rigid / surface-guided alignment using **11 anatomical structures**:
- `hip_left`, `hip_right`
- `prostate`
- `gluteus_maximus_left`, `gluteus_maximus_right`
- `iliopsoas_left`, `iliopsoas_right`
- `iliac_vena_left`, `iliac_vena_right`
- `iliac_artery_left`, `iliac_artery_right`

> [!IMPORTANT]
> `urinary_bladder` is deliberately **excluded** from the registration organ set because variable bladder filling between scan sessions causes artificial tissue deformations.

```bash
python scripts/registration/rerun_mri_registration.py
```
Outputs the optimized spatial alignment matrix: `Transform_FollowUp_to_Baseline_3.tfm`.

#### Step 2: Multi-Timepoint Assembly (`prepare_psma_case_for_medeye3d.py`)
[`scripts/preprocessing/prepare_psma_case_for_medeye3d.py`](file:///workspaces/MedEye3d.jl/scripts/preprocessing/prepare_psma_case_for_medeye3d.py) structures the longitudinal dataset:
- **TP 0**: Patient 2 Early Whole-Body PET/CT
- **TP 1**: Patient 2 Late Whole-Body PET/CT
- **TP 2**: Patient 1 Follow-up PET/CT
- **TP 3**: mpMRI T2W Axial + multi-reader segmentations
- **TP 4**: mpMRI ADC Map
- **TP 5**: mpMRI DWI b2000
- **TP 6**: mpMRI T1W CE Dixon
- Builds `scene_hierarchy.json`, `metadata.json`, and `matches.json`.

```bash
python scripts/preprocessing/prepare_psma_case_for_medeye3d.py
```

#### Step 3: High-Resolution Resampling (`reprocess_mri_tps.jl`)
[`scripts/preprocessing/reprocess_mri_tps.jl`](file:///workspaces/MedEye3d.jl/scripts/preprocessing/reprocess_mri_tps.jl) resamples the MRI sequences into the baseline frame using GPU-accelerated interpolation:
```bash
julia --project=. scripts/preprocessing/reprocess_mri_tps.jl
```

#### Step 4: Prostate Zonal Anatomy Embedding (`embed_prostate_anatomy.jl`)
[`scripts/preprocessing/embed_prostate_anatomy.jl`](file:///workspaces/MedEye3d.jl/scripts/preprocessing/embed_prostate_anatomy.jl) embeds the fine prostate zonal masks (Peripheral Zone, Transition Zone, Seminal Vesicles) directly into `/BASELINE/` and `/TP_3/`:
```bash
julia --project=. scripts/preprocessing/embed_prostate_anatomy.jl
```

---

## 7. XNAT Storage, Resource Naming & Discoverability

To ensure MedEye3D automatically discovers, badges, and loads preprocessed cases, files must adhere to the XNAT REST resource conventions.

### 7.1. Resource Structure

For any subject in a project:
```
/data/projects/{PROJECT_ID}/subjects/{SUBJECT_ID}/
  ├── resources/
  │   ├── MEDEYE3D/
  │   │   └── files/
  │   │       └── {SUBJECT_ID}_preprocessed.h5   # Monolithic multi-modal HDF5
  │   ├── ANNOTATIONS/
  │   │   └── files/
  │   │       ├── status.json                     # Case status & audit metadata
  │   │       └── expert_annotations.json         # Clinician validated lesion DB
  │   ├── slicer_3.6/                             # (Fallback) 3D Slicer MRB packages
  │   └── scans_SUV_CT_PET/                       # (Fallback) Raw DICOM series
```

### 7.2. Status JSON Schema (`ANNOTATIONS/status.json`)

MedEye3D writes and reads `status.json` to determine patient progress:
```json
{
  "project_id": "FDM_DPI-2024-7-KRN_Lu177_PSMA_PIPELINE",
  "subject_id": "pat6",
  "status": "PREPROCESSED",
  "assigned_user": "jm",
  "last_modified": "2026-10-24T12:00:00",
  "software_version": "0.5.11",
  "summary": {
    "preprocessed_file": "pat6_preprocessed.h5",
    "file_size_bytes": 1450284000,
    "num_lesions": 64,
    "modalities": ["CT", "PET", "SPECT"],
    "num_timepoints": 4
  }
}
```

### 7.3. How MedEye3D Discovers & Badges Cases at Startup

1. **Project Query**: Dispatches `XnatClient.list_projects` on startup, putting primary pipeline projects (`Lu177_PSMA_PIPELINE`, `Prostata_bimodal_PIPELINE`) at top.
2. **Subject Discovery**: Dispatches `XnatClient.list_subjects` for the active project (retrieving all 420+ Lu177 subjects and 130+ Prostate bimodal subjects).
3. **Instant Cache-Backed Badging**:
   - `[PRE]`: Preprocessed HDF5 is present in local cache (`~/.medeye3d/xnat_cache/{proj}/{subj}/`) or on remote XNAT `MEDEYE3D`.
   - `[RAW]`: Raw scans are present on XNAT; requires preprocessing.
   - `[OK]`: Clinician reviewed and validated (`status.json` has `PROCESSED`).
4. **On-Demand Preprocessing**:
   - When a user selects a `[RAW]` patient, MedEye3D does **not** ignore it.
   - It checks local cache $\to$ checks XNAT for existing HDF5 $\to$ if neither exists, automatically triggers `_run_preprocess_and_upload`.
   - The raw scans are downloaded, preprocessed, cached locally, uploaded to XNAT `MEDEYE3D`, badged `[PRE]`, and seamlessly displayed in the viewer.
5. **Forced Reprocessing**:
   - If a patient is already preprocessed (`[PRE]`) but new scans or model updates are available, clicking the **Preprocess** button (`btn_xnat_preproc`) invokes `_run_preprocess_and_upload(..., force=true)`, rerunning the full pipeline with `--overwrite`.

---

## 8. Verification & Test Suite

The pipeline and XNAT integration are verified by automated test suites:
- [`test/test_xnat_case_selection_and_preprocessing.jl`](file:///workspaces/MedEye3d.jl/test/test_xnat_case_selection_and_preprocessing.jl): Verifies query of all 420+ Lu177 cases and 130+ Prostate bimodal cases, search filtering, on-demand preprocessing dispatch, and subject resource endpoints (31/31 passed).
- [`test/test_xnat_integration.jl`](file:///workspaces/MedEye3d.jl/test/test_xnat_integration.jl): Verifies token authentication, session keepalive, HDF5 download/upload, and `status.json` audit tracking (64/64 passed).
- [`test/test_gui_locking_and_options.jl`](file:///workspaces/MedEye3d.jl/test/test_gui_locking_and_options.jl): Verifies GUI dropdown interaction, modal locking, and companion login window (81/81 passed).
