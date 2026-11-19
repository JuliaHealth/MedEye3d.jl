# Prostata Bimodal Preprocessing Pipeline (`Prostata_bimodal_PIPELINE`)

Comprehensive reference and verification summary for building, repairing, auditing, and syncing the **`Prostata_bimodal_PIPELINE`** cohort (`Pat1`–`Pat133`) into MedEye3D HDF5 bundles (`preprocessed_volumes.h5`).

---

## 1. End-to-End Pipeline Stages

### Stage 1: Inventory & Raw Data Acquisition
- Census file: `data/xnat_bimodal_census.json` (maps `Pat*` IDs to XNAT subject IDs, experiment IDs, `slicer_final` artifacts, and modality flags).
- Dual-GPU orchestrator: [`scripts/preprocessing/batch_process_cohort_dual_gpu.py`](../../scripts/preprocessing/batch_process_cohort_dual_gpu.py) (plus standalone XNAT wrapper [`scripts/xnat/xnat_preprocess_patient.sh`](../../scripts/xnat/xnat_preprocess_patient.sh) and [`scripts/xnat/xnat_preproc_worker.jl`](../../scripts/xnat/xnat_preproc_worker.jl)).
- PET SUV calibration converts AC PET DICOMs into quantitative SUV (`BASELINE/SUV_PET_Image_0.nii.gz`).

### Stage 2: CT–MRI Registration & Multimodal Resampling
- Registration engine: [`docs/register_ct_mri.py`](../register_ct_mri.py) and cohort repair [`scripts/preprocessing/repair_cohort_mri_registration.py`](../../scripts/preprocessing/repair_cohort_mri_registration.py).
- Method:
  1. Whole-CT medial bone landmarks (4 most medial points of `femur_left`, `femur_right`, `hip_left`, `hip_right`) + Procrustes rigid SVD alignment.
  2. Differentiable **Sliced Wasserstein Distance (SWD)** prostate surface refinement on GPU (`Transform_Landmark_SWD.h5` / `Transform_FollowUp_to_Baseline_1.tfm`).
  3. MRI sequence selection prioritizes axial `T2_TSE_RL` over sagittal `T2_TSE_FH` (`select_ranked_mri_sequences()`).
     - `classify_mri_modality()` and [`scripts/preprocessing/repair_cohort_adc_and_dwi.py`](../../scripts/preprocessing/repair_cohort_adc_and_dwi.py) recover missing quantitative `ADC` (`dDWI`, `dWIP`, `IsoADC`, `ddADC`, truncated `s)_*`/`s):*`, and generic `DWI` series with DICOM `ImageType = ADC`, deprioritizing exponential `eADC`) from XNAT MRI experiments (`*_MRI_*`).
     - `unpack_stacked_mri_volume()` de-interleaves 2-way alternating `b=0`/`b_high` DWIs (even/odd mean ratio `>= 1.5`, doubling `sp_z` and retaining high `b`-value slices) and block-stacked multi-b-value DWIs.
  4. Native MRI prostate zones (`1..8`) and prostate tumor (`Label 9`) segmentation is produced by `Dataset501_ProstateZonesLesions` (`models/Dataset501_ProstateZonesLesions`, `T2W + ADC + unpacked HBV/DWI`) via [`scripts/preprocessing/run_prostate_seg_missing.py`](../../scripts/preprocessing/run_prostate_seg_missing.py) and uploaded to `PROSTATE_SEGMENTATION/prostate_segmentation.nii.gz`. `NNUNET_OR_Processed_Lesions_MR.seg.nrrd` is **never** used as an MRI lesion source (it holds PSMA PET nnUNet predictions).
  5. Resamples moving MRI modalities (`T2`, `ADC`, `DWI`, `T1`, `DCE`, skipping multi-component RGB perfusion overlays), 8-zone prostate anatomy (`TFM_*/max_anatomy.nii.gz`), and native MRI lesion masks (`Label 9` -> `TFM_*/PET_Lesions_<tp>.nii.gz`) into the baseline CT grid — whole-body CT `max_anatomy` is never written into `TFM_*`.

### Stage 3: Whole-Body Anatomy + Authoritative Lymph Nodes + CADS (`max_anatomy`)
- Core module: [`scripts/preprocessing/cads_anatomy_wiring.py`](../../scripts/preprocessing/cads_anatomy_wiring.py)
- Cohort rebuilder: [`scripts/preprocessing/repair_max_anatomy_all.py`](../../scripts/preprocessing/repair_max_anatomy_all.py)
- Label layering order into `BASELINE/max_anatomy.nii.gz`, `ATLAS/max_anatomy`, and `_meta_/max_anatomy_labels.json`:
  1. **Primary Whole-Body Anatomy**: Loaded from `raw/packed_anatomical_tensor.h5` + `packed_anatomical_tensor_registry.json` (or embedded HDF5 `registry` dataset).
  2. **Sanitization (`is_excluded_non_organ_mask`)**: Purges non-organ / helper masks (`body`, `body_trunc`, `body_extremities`, `torso_fat`, `subcutaneous_fat`, `skin`, `skeletal_muscle`, `abdominal_cavity`, `thoracic_cavity`, `thorax_wall`, `nasal_cavity_*`, `lung_airways_wall`, `oblique_hyoid_plane`, `digastric_plane_*`, etc.).
  3. **Authoritative Lymph Node Stations (`iter_authoritative_lymph_node_masks`, external `lymph_node_rules` repo)**:
     - Generated externally by the separate `lymph_node_rules` repository (`LN_RULES_DIR` / XNAT `LYMPH_NODE_RESULTS`) and consumed read-only here (no local LN worker or vendored `lymph_node_rules/` ruleset exists in `MedEye3d.jl`).
     - Prioritizes `<case_dir>/lymph_nodes/` (`all_lymph_nodes.nii.gz` + `lymph_node_labels.json`, plus standalone `<station>.nii.gz`),
     - Falls back to `raw/final_results_gpu.h5` (downloaded from XNAT `LYMPH_NODE_RESULTS` or reused from external cache).
     - Old LN stations inside `packed_anatomical_tensor.h5` are skipped whenever authoritative LN masks exist.
     - **Lung-only override rule**: Lymph node voxels may only be overridden by lung labels (`is_lung_label_name`) or CADS structures (`531..539`).
  4. **CADS Structures (`531..539`)** (`merge_cads_structures`):
     - `531`: `seminal_vesicle` (Task 556 idx 8)
     - `532`: `sigmoid_colon` (Task 556 idx 5 — replaces only overlapping colon or empty voxels)
     - `533/534`: `lacrimal_gland_left` / `lacrimal_gland_right` (Task 558 idx 16/17)
     - `535/536/537`: `optic_chiasm` / `optic_nerve_left` / `optic_nerve_right` (Task 558 idx 24/25/26)
     - `538`: `spinal_canal` (Task 556 idx 1)
     - `539`: `pituitary_gland` (Task 558 idx 29)
  5. **MedEye3D Display Range Requirement**: Cases using dynamic label IDs (e.g. `Pat1` with LNs in `428..496` and CADS in `531..539`) require `"Anatomy"` `minAndMaxValue` >= `[1, 10000]` in the viewer ([`AppMain.jl`](../../src/packaging/AppMain.jl), [`MakieEventHandlers.jl`](../../src/display/GLFW/MakieEventHandlers.jl)).

### Stage 4: Neural Edge Maps & Heat-GDT Diffusivities (3 Modalities)
Each processed `BASELINE` contains all three diffusivity models on the `(Z, Y, X)` CT grid:
- **`BASELINE/diffusivity`** (Both CT+PET) & per-MRI **`TFM_*/diffusivity`**: `TotalSegEdgePDE` (`CHAMPION_GAMMA = 10.095886`, `CHAMPION_EPS_D = 1e-5`) via [`batch_process_cohort_dual_gpu.py`](../../scripts/preprocessing/batch_process_cohort_dual_gpu.py) / [`generate_modality_diffusivities.jl`](../../scripts/preprocessing/generate_modality_diffusivities.jl).
- **`BASELINE/edge_ct`, `BASELINE/diffusivity_ct`** (CT-only default): `TotalSegEdgeNet v1` via [`scripts/preprocessing/apply_totalseg_edge_v1.py`](../../scripts/preprocessing/apply_totalseg_edge_v1.py) and [`totalseg_edge_v1_model.py`](../../scripts/preprocessing/totalseg_edge_v1_model.py).
- **`BASELINE/edge_pet`, `BASELINE/diffusivity_pet`** (PET-only): `TotalSegEdgePDE_PET` (`exp(-9.4069 * edge_pet)`) via [`scripts/preprocessing/add_pet_diffusivity.py`](../../scripts/preprocessing/add_pet_diffusivity.py) and [`totalseg_edge_pde_pet_model.py`](../../scripts/preprocessing/totalseg_edge_pde_pet_model.py).

### Stage 5: Lesions, Edge-Slice Trim, Semantic Filtering & Radiological Reports
- **Edge-slice Z trim**: [`scripts/preprocessing/apply_edge_slices_and_semantic_filter.py`](../../scripts/preprocessing/apply_edge_slices_and_semantic_filter.py) removes lesion voxels in `z in {0, 1, Z-2, Z-1}` across TP 0 and all `TFM_*` groups and re-syncs `CENTROIDS`, `organ_mapping`, `segment_names.json`, `matches.json`, and `scene_hierarchy.json`.
- **Lacrimal lesion filtering**: `filter_lacrimal_lesions()` removes physiological lacrimal gland (`533/534`) uptake lesions from `BASELINE/PET_Lesions_0.nii.gz` and metadata.
- **Clinical lesion classification & bone subsegmentation**: CT lesions are repaired/verified by [`scripts/preprocessing/repair_cohort_lesions.py`](../../scripts/preprocessing/repair_cohort_lesions.py) and [`recompute_bones.jl`](../../scripts/preprocessing/recompute_bones.jl).
- **Radiological reports & semantic filter**:
  - [`scripts/preprocessing/populate_radiological_reports.py`](../../scripts/preprocessing/populate_radiological_reports.py) loads clinical dictations from Excel into `_meta_/metadata.json` (`Description` + `EnglishDescription`).
  - `compute_and_store_semantic_filter()` populates `_meta_/semantic_filter.json` (`mentioned_max_anatomy`, `allowed_lesion_ids_by_tp`).

---

## 2. Monolithic HDF5 Schema (`preprocessed_volumes.h5`)

Verified layout (observed on `Pat1/preprocessed_volumes.h5`):

| Group / Dataset | Description |
|---|---|
| `BASELINE/Fixed_CT_Volume_0.nii.gz` | Baseline CT volume (pre-flipped along axis 1) |
| `BASELINE/SUV_PET_Image_0.nii.gz` | Calibrated PET SUV volume |
| `BASELINE/PET_Lesions_0.nii.gz` | Baseline PET/CT lesion label mask (`Int16`) |
| `BASELINE/max_anatomy.nii.gz` | Whole-body CT organs + LN stations + CADS `531..539` (`UInt16`) |
| `BASELINE/skellytour.nii.gz` | Skellytour bone mask |
| `BASELINE/diffusivity`, `diffusivity_ct`, `diffusivity_pet` | Precomputed Heat-GDT diffusivity fields |
| `BASELINE/edge_ct`, `edge_pet` | 3D neural boundary probability maps |
| `ATLAS/max_anatomy`, `ATLAS/skellytour`, `ATLAS/bone_atlas` | Canonical atlas volumes |
| `TFM_Transform_FollowUp_to_Baseline_<k>.tfm/` | Registered MRI volumes (`Fixed_CT_Volume_<k>.nii.gz`, `PET_Lesions_<k>.nii.gz`, `SUV_PET_Image_<k>.nii.gz`, `diffusivity`, 8-zone `max_anatomy.nii.gz`) |
| `CENTROIDS/` | 1-based lesion (`tp<t>_lid<id>`) and lymph node (`tp<t>_ln<id>`) centroids |
| `_meta_/preflipped` | `1` (axis-1 pre-flipped; zero runtime flip needed) |
| `_meta_/max_anatomy_labels.json` | Integer ID → structure name map for `max_anatomy` |
| `_meta_/prostate_anatomy_labels.json` | 8-zone MRI prostate label map (`UR`, `AFS`, `PG`, `SV_L`, `TZ`, `SV_R`, `CZ`, `PZ`) |
| `_meta_/organ_mapping`, `segment_names.json`, `matches.json` | Automated lesion-to-organ mapping, segment display names, and cross-TP links |
| `_meta_/metadata.json`, `semantic_filter.json` | Translated radiological reports and LLM-extracted lesion/organ filter |
| `_meta_/scene_hierarchy.json` | 3D Slicer hierarchy parsed by `AppMain.jl` |
| `_meta_/modification_status.json`, `custom_options.json` | Clean baseline flag (`manually_modified: false`) and dropdown presets |

---

## 3. Verification Commands & Cohort Audit Status

### Automated Verification Commands
```bash
# 1. Python preprocessing invariant unit tests
/mnt/big/project_ssd/project_ssd/lymph_node_rules/for_joanna/.local/conda/envs/seg/bin/python \
  /workspaces/MedEye3d.jl/test/test_preprocessing_invariants.py

# 2. Full cohort HDF5 invariant audit (+ optional --check-xnat)
python3 /workspaces/MedEye3d.jl/scripts/preprocessing/audit_full_cohort.py --no-dice
python3 /workspaces/MedEye3d.jl/scripts/preprocessing/audit_full_cohort.py --no-dice --check-xnat

# 3. Rebuild max_anatomy from authoritative lymph_nodes/ + CADS and upload to XNAT
python3 /workspaces/MedEye3d.jl/scripts/preprocessing/repair_max_anatomy_all.py --upload-xnat --skip-synced
```

### Current Cohort Verification Summary (`Pat1`–`Pat133`)
- **Total patient directories in `data/cases/Prostata_bimodal/`**: **133** (`Pat1`–`Pat133`).
- **Processed cases (`preprocessed_volumes.h5` present)**: **126 / 133**.
  - **Multimodal MRI & `Dataset501` Coverage**: **124 / 126** processed cases have aligned quantitative `ADC`, de-interleaved/unpacked `DWI`, and native `Dataset501_ProstateZonesLesions` (`T2W + ADC + DWI`) 8-zone prostate + `Label 9` tumor segmentations (`Pat46` & `Pat101` lack DWI/ADC acquisitions on XNAT and retain `T2 + T1` only).
  - **PSMA PET/CT vs. Native MRI Lesion Coverage**:
    - **126 / 126** processed cases contain baseline PSMA PET/CT lesions in `BASELINE/PET_Lesions_0.nii.gz` (`0` empty; `4`–`35` lesions per case).
    - **109 / 126** processed cases have MRI tumor segmentations (`Dataset501_ProstateZonesLesions` `Label 9` -> `TFM_*/PET_Lesions_*`); **`17 / 126`** have `0` MRI lesion voxels (e.g., `Pat1`, confirmed by its `20230504_MRI` radiology report stating *"no tumour mass is discernible in the prostate"*).
  - **Local HDF5 + CT–MRI Registration Dice Audit (`audit_full_cohort.py`, 2026-11-19)**: `0` low-Dice failures cohort-wide (`124/126` clean with Dice prior to syncing `Pat1` viewer cache; all 7 re-registered `Dataset501` cases pass: `Pat1 = 0.7612`, `Pat24 = 0.7747`, `Pat25 = 0.7719`, `Pat29 = 0.6961`, `Pat66 = 0.8660`, `Pat110 = 0.8470`, `Pat128 = 0.8686`). Under `--no-dice`, `124/126` pass locally and in `xnat_cache` (`Pat1` and `Pat51` caches reflect live interactive viewer autosaves with `manually_modified: true`).
  - **Unit Tests**: [`test/test_preprocessing_invariants.py`](../../test/test_preprocessing_invariants.py) passes **9/9** tests, [`test/test_lesion_navigation_fix.jl`](../../test/test_lesion_navigation_fix.jl) passes **16/16** testsets (including Quad-MRI `KEY_Y` fallback & `Pat1` `T2/ADC/DWI/T1`), and [`test/test_lesion_edits_recorder.jl`](../../test/test_lesion_edits_recorder.jl) passes **4/4** testsets (`76` assertions).
- **Unprocessed / Incomplete Raw Cases (7 / 133)**:
  - **`Pat8`, `Pat21`**: Census `exp_id` on XNAT contains MRI scans only (no PET/CT or `slicer_final` segmentations; `Pat8` has only a `4,108`-byte stub HDF5 in `Prostata_bimodal_PIPELINE`).
  - **`Pat2`, `Pat13`, `Pat61`**: PET/CT-only (`has_anat = true`, `has_ln = true`, `has_reg = false`, `has_prostate_seg = false`); `slicer_final` lacks MRI registrations and `NNUNET_OR_Processed_Lesions_CT.seg.nrrd` (older ~1 GB HDF5s exist on XNAT).
  - **`Pat102`, `Pat104`**: Census points `exp_id` to `MRI_0` (`has_anat = false`, `has_ln = false`), though local directories contain `Fixed_CT_Volume.nii.gz`, `NNUNET_OR_Processed_Lesions_CT.seg.nrrd`, `packed_anatomical_tensor.h5`, `cads/`, `lymph_nodes/`, and `slicer_final` registered MRIs (`subj_MEDEYE3D = []` on XNAT).

> **XNAT Sync Status**: `--no-dice --check-xnat` confirms **40 / 126** cases (including **`Pat1`** at `2,193,409,065` bytes with clean `PREPROCESSED` `status.json`, `PROSTATE_SEGMENTATION/prostate_segmentation.nii.gz`, and `CT_MRI_REGISTRATION/Transform_Landmark_SWD.h5`) are byte-exact across local storage, `xnat_cache`, and live XNAT. All **85** remaining unmodified local cases pass local + `xnat_cache` checks cleanly and await bulk HDF5 upload via `repair_max_anatomy_all.py --upload-only --skip-synced` (`~3.2 MB/s` aggregate link rate to XNAT). **`Pat51`** intentionally differs due to active expert edits.
