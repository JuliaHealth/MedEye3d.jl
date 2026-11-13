# MedEye3d.jl — Session & Task Summary (2026-11-12 – 2026-11-13)

## 1. Current Active Task

### Objective
1. **Protect Lymph Node Areas in `max_anatomy`**:
   - Ensure no CADS-injected structure (`seminal_vesicle`, `sigmoid_colon`, `lacrimal_gland_left/right`, `optic_chiasm`, `optic_nerve_left/right`, `spinal_canal`, `pituitary_gland`) overwrites any valid lymph node area voxel.
   - Ensure `sigmoid_colon` replaces only overlapping plain colon voxels (`colon`, `nvseg_colon`) and empty voxels, keeping the rest of the colon intact.
   - Exclude non-lymph structural/helper labels that share station prefixes (`thoracic_sternum_exclusion_zone`, `thoracic_chest_wall`, `*_cavity`, `*_helper`, `*_bridge`, `*_base`, `*_dilated`) from lymph node area classification in both Python ([cads_anatomy_wiring.py](file:///workspaces/MedEye3d.jl/scripts/preprocessing/cads_anatomy_wiring.py)) and Julia ([LesionAssociation.jl](file:///workspaces/MedEye3d.jl/src/display/LesionAssociation.jl)).
   - Prevent CADS label IDs (`531..539`) from colliding with existing lymph node station IDs in older cases (e.g., `Pat1`, `Pat6`, `Pat19`, `Pat20`) via dynamic collision-free ID resolution (`resolve_cads_label_ids`).
2. **Separate CT and MRI `max_anatomy`**:
   - Stop displaying CT `max_anatomy` over MRI timepoints (`TFM_*` groups). MRI timepoints own their dedicated 8-zone prostate `max_anatomy` (or remain empty if no prostate segmentation exists).
3. **Cohort-Wide Repair**:
   - Rebuild clean CT `max_anatomy` and MRI `max_anatomy` from `raw/` data across all `Prostata_bimodal` cohort cases using [repair_cads_max_anatomy.py](file:///workspaces/MedEye3d.jl/scripts/preprocessing/repair_cads_max_anatomy.py), validate against the stored atlas (`match >= 0.999`), and re-merge CADS with the corrected rules.

---

## 2. Detailed Chronological Work Done (Last 24 Hours)

### A. XNAT Loading, Download Stability & Cache Fixes
- **Problem**: Loading `Pat9` from XNAT in the MedEye3d GUI opened a single 288 MB MRI prostate crop (`(23, 176, 176)`) with zeroed PET instead of the 1.13 GB whole-body CT + AC PET SUV + 31 CT lesions + CADS anatomy.
- **Root Causes & Fixes**:
  1. **Julia `Downloads.download` 20s Low-Speed Timeout** ([XnatClient.jl](file:///workspaces/MedEye3d.jl/src/xnat/XnatClient.jl)):
     - XNAT Tomcat takes ~16–20s to stage 1.1+ GB files, triggering `CURLOPT_LOW_SPEED_TIME = 20` aborts.
     - Configured `Downloads.Downloader` with `easy_hook` setting `CURLOPT_LOW_SPEED_TIME = 0` and `CURLOPT_LOW_SPEED_LIMIT = 0`, added atomic `.tmp` downloads, added `curl` fallback, and removed the erroneous `.tfm` guard that rejected single-timepoint cases.
  2. **24-Hour Token Wipe** ([XnatAuth.jl](file:///workspaces/MedEye3d.jl/src/xnat/XnatAuth.jl)):
     - Removed the hardcoded `(now_sec - issued_at) >= 86400.0` check so token validity respects `expires_at` instead of forcing offline mode after 24 hours.
  3. **Cache Directory Fragmentation** ([XnatPreloader.jl](file:///workspaces/MedEye3d.jl/src/xnat/XnatPreloader.jl)):
     - Updated `get_local_cache_dir` to normalize project IDs via `XnatClient.resolve_project_id(project_id)` so `FDM_DPI-2024-7-KRN_Prostata_bimodal_PIPELINE` and `Prostata_bimodal_PIPELINE` share the same cache path.
  4. **Local Cohort Fallback** ([LesionMetadataWindow.jl](file:///workspaces/MedEye3d.jl/src/display/LesionMetadataWindow.jl), [xnat_preprocess_patient.sh](file:///workspaces/MedEye3d.jl/scripts/xnat/xnat_preprocess_patient.sh)):
     - Added general resolution for `data/cases/Prostata_bimodal/<PatX>/preprocessed_volumes.h5` in offline mode and preprocessing fallback seeding.
  5. **Verified `Pat9` Cache**: Re-downloaded the genuine 1,130,612,001-byte HDF5 from XNAT (`CT (284, 512, 512)`, `SUV_PET_Image_0` max SUV `132.26`, `31` CT lesions).

### B. CADS Whole-Body CT Segmentation Integration
- **Installed & Wired Standalone CADS** (`third_party/CADS`):
  - **Task 556 (RT Risk Organs)**: `spinal_canal` (`538`), `sigmoid_colon` (`532`), `seminal_vesicle` (`531`).
  - **Task 558 (Head & Neck)**: `lacrimal_gland_left` (`533`), `lacrimal_gland_right` (`534`), `optic_chiasm` (`535`), `optic_nerve_left` (`536`), `optic_nerve_right` (`537`), `pituitary_gland` (`539`).
- **Ontology & Display Formatting**:
  - Added ontology entries in [max_anatomy_to_ontology.json](file:///workspaces/MedEye3d.jl/assets/max_anatomy_to_ontology.json) (marking `lacrimal_gland_left/right` and `pituitary_gland` as benign/physiological PSMA uptake).
  - Added display name mappings in [LesionAssociation.jl](file:///workspaces/MedEye3d.jl/src/display/LesionAssociation.jl).
- **Batch Processing & Sync Scripts**:
  - [batch_cads_cohort.py](file:///workspaces/MedEye3d.jl/scripts/preprocessing/batch_cads_cohort.py): Ran CADS Tasks 556 & 558 across `117` cases in `data/cases/Prostata_bimodal/`.
  - [sync_cads_to_xnat.py](file:///workspaces/MedEye3d.jl/scripts/preprocessing/sync_cads_to_xnat.py): Script to upload updated `preprocessed_volumes.h5` files to XNAT `MEDEYE3D` resources.

### C. `max_anatomy` Protection & MRI Separation Fixes (Today)
- **Fixed [cads_anatomy_wiring.py](file:///workspaces/MedEye3d.jl/scripts/preprocessing/cads_anatomy_wiring.py)**:
  - Protected all valid lymph node area voxels from being overwritten by CADS structures.
  - Restricted `sigmoid_colon` to overwrite only plain `colon` / `nvseg_colon` and empty voxels.
  - Added `resolve_cads_label_ids` to avoid ID collisions when cases already use IDs `>= 531`.
  - Stopped `patch_case_h5_with_cads` from writing CT `max_anatomy` into `TFM_*` (MRI) groups.
- **Fixed [batch_process_cohort_dual_gpu.py](file:///workspaces/MedEye3d.jl/scripts/preprocessing/batch_process_cohort_dual_gpu.py)**:
  - Removed the fallback that copied `max_anat_f` (CT atlas) into `mri_anat_f` when `mri_anatomy_arr` was empty.
- **Fixed [LesionAssociation.jl](file:///workspaces/MedEye3d.jl/src/display/LesionAssociation.jl) & [cads_anatomy_wiring.py](file:///workspaces/MedEye3d.jl/scripts/preprocessing/cads_anatomy_wiring.py)**:
  - Excluded `thoracic_sternum_exclusion_zone`, `thoracic_chest_wall`, `*_cavity`, `*_helper`, `*_bridge`, `*_base`, and `*_dilated` from lymph node area classification.
- **Verified on `Pat9`**:
  - Lymph voxels restored from `3,675,306` to `3,683,478` (100% of raw lymph union preserved).
  - `sigmoid_colon`: `13,259` voxels (carved from `colon` without reducing total `117,451` colon voxels).
  - `seminal_vesicle`: `2,336` voxels.
  - `TFM_*` MRI groups restored to 8-zone prostate anatomy (`24,921` voxels, distinct from CT atlas).
- **Cohort Repair Running**:
  - Two background workers (`repair_cads_max_anatomy.py --slice 0:2` and `1:2`) are actively repairing all cohort cases (`66` cases completed so far).
  - Skipped collision cases (`Pat1`, `Pat6`, `Pat19`, `Pat20`, etc.) will be re-run with the updated collision-free ID logic once the current pass finishes.
