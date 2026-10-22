# MedEye3d XNAT Integration: Architecture & User Guide

## 1. Overview & System Goals

MedEye3d integrates natively with **XNAT (Extensible Neuroimaging Archive Toolkit)** to provide seamless access to longitudinal, multi-modal imaging archives. Clinicians and researchers can browse repositories, stream pre-aligned and calibrated multi-timepoint HDF5 datasets, review and validate lesions and anatomical lymph node areas, edit segmentations, and save validated annotations back to XNAT with full audit tracking.

### Key Capabilities:
- **24-Hour Rotating Alias/Secret Token Authentication**: Automated "verify-first" token lifecycle management, offline resilience, and interactive renewal dialogs.
- **Support for Multi-Timepoint Regimens**:
  - `FDM_DPI-2024-7-KRN_Prostata_bimodal_PIPELINE`: Co-registered $^{68}\text{Ga}$/$^{18}\text{F}$-PSMA PET/CT and multiparametric Prostate MRI (T2w, ADC, DWI) across multiple clinical timepoints.
  - `FDM_DPI-2024-7-KRN_Lu177_PSMA_PIPELINE`: Pre-therapeutic screening PSMA PET/CT paired with sequential post-therapy $^{177}\text{Lu}$ SPECT/CT scans (e.g., `pat6`).
- **Standardized Preprocessing Pipeline**: Standalone bash/Julia orchestrator (`scripts/xnat/xnat_preprocess_patient.sh`) executing full reruns of Skellytour cortical/trabecular bone extraction and the Lymph Node Rules Engine before packaging into ultra-fast HDF5 volumes.
- **Fast Status Badging & Audit Tracking**: Immediate visual indicators (`RAW` 🔴, `PREPROCESSED` 🟡, `IN_PROGRESS` 🔵, `PROCESSED` 🟢) backed by `status.json`.
- **Zero-Latency Background Preloader**: Automatic background caching of patient $N+1$ ensures instantaneous case switching when clicking **Next Case**.

---

## 2. Authentication Architecture (`src/xnat/XnatAuth.jl`)

XNAT clusters frequently enforce temporary 24-hour Alias and Secret tokens generated via `/data/services/tokens/issue`.

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │                            Startup Check                               │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
      [Cached Token Exists]                     [No Cached Token]
                   │                                     │
                   ▼                                     │
        [Verify Token on XNAT]                           │
         GET /data/JSESSION                              │
                   │                                     │
         ┌─────────┴─────────┐                           │
         ▼                   ▼                           │
    [HTTP 200 OK]       [401 / Expired]                  │
         │                   │                           │
         ▼                   └───────────┬───────────────┘
  Connect Silently                       ▼
  🟢 XNAT: Connected           Prompt Login / Token Modal
                               Enter Alias & Secret
```

### 2.1. The "Verify-First" Policy
1. On startup or when opening the application, MedEye3d inspects `xnat_credentials.json` (stored securely in `%APPDATA%/MedEye3D/` on Windows or `~/.medeye3d/` on Linux).
2. It validates the local expiration timestamp ($T_{\text{now}} < T_{\text{expires}} - 15\text{min}$).
3. It dispatches a silent verification ping to the server (`GET /data/JSESSION`).
4. If valid, the user is authenticated silently without interrupting their workflow.
5. If expired or missing, the login overlay opens, displaying fields for **XNAT URL**, **24h Alias Token**, **24h Secret Token**, or **Username / Password**.

### 2.2. Interactive Token Renewal
Clinicians can click the **Connect 🔑** / **Refresh 🔄** button in the XNAT Case Navigator header at any time to inspect their remaining token validity or paste newly generated 24-hour tokens.

---

## 3. Project Architectures & Modality Normalization

### 3.1. Project: `FDM_DPI-2024-7-KRN_Prostata_bimodal_PIPELINE`
Contains paired baseline and follow-up studies combining PSMA PET/CT and high-resolution multiparametric Prostate MRI:
- **Timepoint 0 (`BASELINE`)**: Initial Staging PSMA PET/CT (CT HU + SUV PET + Whole-Body Anatomy Atlas).
- **Timepoint 1 (`TFM_MRI_0`)**: Baseline Prostate MRI (T2w Axial/Coronal, ADC Map, High b-value DWI) rigidly registered to Baseline CT. Prostate zonal anatomy labels (Peripheral Zone, Transition Zone, Seminal Vesicles) are embedded.
- **Timepoint 2 (`TFM_PET_1`)**: Post-treatment or follow-up PSMA PET/CT.
- **Timepoint 3 (`TFM_MRI_1`)**: Follow-up Prostate MRI.
- **Multi-Monitor Display**: View baseline PET/CT in Window 1 while concurrently displaying registered MRI T2w/ADC in Secondary Window 2 (M2).

### 3.2. Project: `FDM_DPI-2024-7-KRN_Lu177_PSMA_PIPELINE` (e.g., `pat6`)
Follows metastatic prostate cancer patients undergoing radioligand therapy with $^{177}\text{Lu}$-PSMA:
- **Timepoint 0 (`BASELINE`)**: Diagnostic $^{68}\text{Ga}$-PSMA PET/CT with standard uptake value (SUV) quantification.
- **Timepoint 1..N (`TFM_SPECT_1..N`)**: Sequential post-infusion $^{177}\text{Lu}$ SPECT/CT scans (e.g. 24h, 48h, 72h post-cycle 1, 2, 3).
- **Count Normalization**:
  Unlike quantitative PET SUV, $^{177}\text{Lu}$ SPECT values represent non-SUV raw counts. MedEye3d computes the 99th percentile of positive voxel intensities ($P_{99}$) and applies dynamic scaling:
  $$\text{Scale Factor} = \frac{8.0}{\max(P_{99}, 1.0)}, \quad \text{SPECT}_{\text{scaled}} = \max\left(0, \text{SPECT} \times \text{Scale Factor}\right)$$

---

## 4. Standalone Preprocessing Pipeline (`scripts/xnat/`)

A standalone bash orchestrator and Julia worker automate downloading raw DICOM/NIfTI scans from XNAT, running AI inference and landmark rule DAGs, and uploading preprocessed HDF5 archives back to XNAT.

### 4.1. CLI Usage
```bash
# Process a single patient
./scripts/xnat/xnat_preprocess_patient.sh \
    --project FDM_DPI-2024-7-KRN_Lu177_PSMA_PIPELINE \
    --subject pat6

# Process all subjects in a project
./scripts/xnat/xnat_preprocess_patient.sh \
    --project FDM_DPI-2024-7-KRN_Prostata_bimodal_PIPELINE \
    --all

# Overwrite existing preprocessed files and skip re-running Skellytour
./scripts/xnat/xnat_preprocess_patient.sh \
    --project FDM_DPI-2024-7-KRN_Lu177_PSMA_PIPELINE \
    --subject pat6 \
    --overwrite \
    --skip-skellytour
```

### 4.2. Pipeline Execution Steps
1. **Catalog Query**: Lists imaging experiments and scans for the subject on XNAT.
2. **Streaming Download**: Pulls DICOM series or NIfTI volumes into a scratch workspace (`/tmp/xnat_work/{proj}/{subj}`).
3. **Spatial Normalization**: Resamples all secondary timepoints onto the voxel grid of the baseline CT.
4. **Full Rerun of Skellytour**: Executes `AIInference.run_skellytour_segmentation` to generate trabecular and cortical bone subsegmentations.
5. **Full Rerun of Lymph Node Rules DAG**: Evaluates landmark-guided rule engines (`scripts/preprocessing/lymph_node_rules/`) across cervical, thoracic, abdominal, and pelvic stations to generate anatomical boundaries in `max_anatomy.nii.gz`.
6. **Bone Subsegmentation**: Synthesizes cortical bone surface and trabecular marrow layers (`recompute_bones.jl`).
7. **HDF5 Packaging**: Packs all aligned volumes, textures, labels, and metadata into `{subject}_preprocessed.h5`.
8. **Upload to XNAT**:
   - Creates resource `PUT /data/projects/{proj}/subjects/{subj}/resources/MEDEYE3D`
   - Uploads HDF5: `PUT .../resources/MEDEYE3D/files/{subj}_preprocessed.h5?overwrite=true`
   - Initializes audit status: `status.json` with `"status": "PREPROCESSED"` in `resources/ANNOTATIONS`.
9. **Scratch Cleanup**: Deletes temporary raw files to conserve local disk storage.

---

## 5. Case Processing Status & Audit Tracking

MedEye3d tracks case lifecycle states using lightweight audit markers in `resources/ANNOTATIONS/files/status.json`:

| Status Code | Badge | Description |
| :--- | :---: | :--- |
| `RAW` | 🔴 | Only raw DICOM scans exist on XNAT; no preprocessed HDF5 found. |
| `PREPROCESSED` | 🟡 | Preprocessed HDF5 exists; ready for clinical review and validation. |
| `IN_PROGRESS` | 🔵 | A user has opened the case and saved intermediate draft modifications. |
| `PROCESSED` | 🟢 | Clinician has completed lesion/lymph node validation and saved to XNAT. |

### Example `status.json`:
```json
{
  "project_id": "FDM_DPI-2024-7-KRN_Lu177_PSMA_PIPELINE",
  "subject_id": "pat6",
  "status": "PROCESSED",
  "assigned_user": "dr_mitura",
  "last_modified": "2026-10-22T07:15:00",
  "software_version": "0.5.11",
  "summary": {
    "num_lesions": 6,
    "has_expert_mask_edits": true,
    "lymph_nodes_validated": true
  }
}
```

---

## 6. GUI Controls in `LesionMetadataWindow.jl`

The **XNAT Case Navigator** panel is positioned at the top of the Makie control panel:

```
┌────────────────────────────────────────────────────────────────────────┐
│ ▼ XNAT Case Navigator                                                  │
├────────────────────────────────────────────────────────────────────────┤
│ 🟢 XNAT: dr_mitura (23h 48m)                              [Refresh 🔄] │
│ Project: [ FDM_DPI-2024-7-KRN_Lu177_PSMA_PIPELINE  ▼ ]  Filter: [All ▼]│
│ Patient: [ 🟢 pat6 (PROCESSED)                     ▼ ]  Status: PROCESSED│
│                                                                        │
│  [ ◀ Prev ]     [ Save XNAT 💾 ]     [ Next ▶ ]     [ Preproc ⚙ ]      │
└────────────────────────────────────────────────────────────────────────┘
```

### Controls & Actions:
- **Server Status & Auth Button (`Connect 🔑` / `Refresh 🔄`)**:
  Displays active username and countdown to token expiration. Clicking opens the modal token entry dialog.
- **Project Dropdown (`menu_xnat_proj`)**:
  Lists available XNAT projects. Changing project dynamically populates patient list and statuses.
- **Filter Dropdown (`menu_xnat_filter`)**:
  Filters patient dropdown to show *All Cases*, *Unreviewed (🟡)*, *Validated (🟢)*, or *Raw Scans (🔴)*.
- **Patient Dropdown (`menu_xnat_subj`)**:
  Displays patients with status badges. Selecting a patient triggers background preloading for subsequent cases and switches the active dataset.
- **Action Buttons**:
  - `[ ◀ Prev ]`: Moves to previous patient in the queue.
  - `[ Save XNAT 💾 ]`: Flushes all modified mask layers (`flush_all_dirty_masks!`), measurements (`save_measurements_to_h5`), and annotations (`persist_to_active_dataset_h5!`), then uploads `status.json` and the updated HDF5 to XNAT.
  - `[ Next ▶ ]`: Auto-saves current patient edits in the background and advances immediately to the preloaded patient $N+1$.
  - `[ Preproc ⚙ ]`: Launches the preprocessing script for the current patient if only raw scans are present.

---

## 7. Zero-Latency Background Preloader (`src/xnat/XnatPreloader.jl`)

To ensure smooth 60+ FPS navigation without stuttering:
1. When Patient $N$ (`pat6`) is selected, `XnatPreloader.notify_patient_selected` determines Patient $N+1$ (`pat7`).
2. An asynchronous background task checks the local cache directory (`%APPDATA%/MedEye3D/xnat_cache/{proj}/{subj}/` or `~/.medeye3d/xnat_cache/`).
3. If not cached, it streams `pat7_preprocessed.h5` from XNAT.
4. When the user clicks **Next ▶**, the application immediately reads from local cache (<100ms switch time).

---

## 8. Docker & DevContainer Integration

Docker assets are fully updated for containerized deployment:
- **`scripts/ai/Dockerfile`**:
  Includes `dcm2niix`, `unzip`, `curl`, `wget`, `h5py`, `requests`, `monai`, and Skellytour (`git+https://github.com/cpwardell/Skellytour.git`).
- **`.devcontainer/Dockerfile`**:
  Includes `dcm2niix`, `h5py`, and `requests` for local development.
- **`docker-compose.yml`**:
  Defines portable volume mounts (`.:/workspaces/MedEye3d.jl`) and provides the `medeye3d-xnat-preproc` service for server-side cluster processing.
