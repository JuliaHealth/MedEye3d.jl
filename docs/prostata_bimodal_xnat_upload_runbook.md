# Prostata Bimodal XNAT Upload Runbook (2026-11-13)

## Status Summary
- **Total cases in cohort**: 133 (Pat1–Pat133)
- **Cases with `preprocessed_volumes.h5`**: 117
- **Cases already fixed (lymph protected, MRI separated)**: 117/117 (100%)
- **Cases missing preprocessing**: 16 (Pat2, Pat8, Pat13, Pat17, Pat21, Pat61, Pat80, Pat87, Pat89, Pat93–Pat95, Pat102, Pat104, Pat110, Pat128)

## Fixes Applied
1. **Lymph node area protection**: No CADS structure overwrites valid lymph voxels.
2. **Sigmoid colon rule**: `sigmoid_colon` replaces only overlapping plain colon (`colon`/`nvseg_colon`) and empty voxels; rest of colon remains intact.
3. **MRI separation**: CT `max_anatomy` no longer written into `TFM_*` (MRI) groups; MRI timepoints retain their 8-zone prostate anatomy.
4. **ID collision handling**: CADS label IDs (531–539) are dynamically resolved to avoid collisions with existing lymph station IDs in older cases.
5. **Helper/exclusion labels excluded**: Labels like `thoracic_sternum_exclusion_zone`, `*_cavity`, `*_helper`, `*_bridge`, `*_base`, `*_dilated` are excluded from lymph node area classification.

## Step-by-Step Workflow

### Step 1: Verify Local Fix Status (Optional)
Run a dry-run to confirm all cases are already fixed:
```bash
cd /workspaces/MedEye3d.jl
python3 scripts/preprocessing/repair_cads_max_anatomy.py --dry-run | tee /tmp/repair_dryrun_latest.log
```
Expected: All existing cases report `DRY-RUN ok match>=0.999 lymph voxels to restore=0`.

### Step 2: (If Needed) Repair Any Remaining Cases
If any case reports `lymph voxels to restore > 0` or `SKIP validation failed`, run the repair:
```bash
# Repair all cases (two parallel slices for speed)
python3 scripts/preprocessing/repair_cads_max_anatomy.py --slice 0:2 &
python3 scripts/preprocessing/repair_cads_max_anatomy.py --slice 1:2 &
wait
```
Or repair specific cases:
```bash
python3 scripts/preprocessing/repair_cads_max_anatomy.py --cases Pat9 Pat44 Pat117
```

### Step 3: Upload Fixed Cases to XNAT
Upload all cases where local `preprocessed_volumes.h5` differs from remote:
```bash
cd /workspaces/MedEye3d.jl
python3 scripts/preprocessing/sync_cads_to_xnat.py
```
This script:
- Reads the census from `data/xnat_bimodal_census.json`.
- For each case with a local HDF5, checks the remote `MEDEYE3D` resource file size.
- Uploads only if the size difference exceeds 1 KB (avoids redundant uploads).
- Uses overwrite=true to replace the remote file.

**Note**: Ensure you have push access to the XNAT project `Prostata_bimodal_PIPELINE`. The script uses your active `gh` authentication session.

### Step 4: Verify Upload Completion
After the sync finishes, spot-check a few cases on XNAT via the GUI or:
```bash
# Example: check Pat9 remote file metadata
gh api \
  -X GET \
  "/projects/Prostata_bimodal_PIPELINE/subjects/Pat9/resources/MEDEYE3D/files?format=json"
```
Confirm the `Size` matches the local file size (typically ~1.1 GB for whole-body CT + AC PET + CADS).

## Cases Requiring Separate Preprocessing
The following 16 cases lack `preprocessed_volumes.h5` and must be preprocessed before upload:
- Pat2, Pat8, Pat13, Pat17, Pat21, Pat61, Pat80, Pat87, Pat89, Pat93, Pat94, Pat95, Pat102, Pat104, Pat110, Pat128

To preprocess them, ensure their raw data exists in `data/cases/Prostata_bimodal/<PatX>/raw/`, then run the cohort pipeline or a targeted single-case script (not included here).

## Validation Metrics (Post-Repair)
For any repaired case, expect:
- **Atlas match**: ≥ 0.999 outside CADS voxels.
- **Lymph voxels restored**: 0 (all preserved).
- **MRI anatomy**: 8-zone prostate labels only in `TFM_*` groups.
- **CADS structures present**: `seminal_vesicle` (531), `sigmoid_colon` (532), `lacrimal_gland_left/right` (533/534), `optic_chiasm` (535), `optic_nerve_left/right` (536/537), `spinal_canal` (538), `pituitary_gland` (539).

## Rollback (If Needed)
If a case needs to be reverted to pre-CADS state:
1. Restore `max_anatomy` from the rebuilt base (available in `repair_cads_max_anatomy.py` logic).
2. Re-run `patch_case_h5_with_cads` with corrected parameters.
3. Re-upload via `sync_cads_to_xnat.py`.

---
Generated: 2026-11-13  
Author: GitHub Copilot (based on session summary 20261113)
