# MedEye3D Preprocessing Documentation

This folder documents the offline and XNAT-backed preprocessing pipelines that build monolithic `preprocessed_volumes.h5` containers for **MedEye3D**, with focus on the **Prostata Bimodal (`Prostata_bimodal_PIPELINE`)** cohort.

## Contents

- **[Prostata Bimodal Pipeline & Verification Status](prostata_bimodal_pipeline.md)** — Stage-by-stage breakdown of the `Prostata_bimodal_PIPELINE` workflow, authoritative `lymph_nodes/` integration, CADS merging, CT–MRI registration, neural edge/diffusivity generation, HDF5 layout, and cohort audit status.
- Related guides in `docs/`:
  - [`preprocessing_and_xnat_guide.md`](../preprocessing_and_xnat_guide.md) — General Docker worker & XNAT integration manual (`Lu177_PSMA_PIPELINE` + bimodal overview)
  - [`anatomy_segmentation_pipeline.md`](../anatomy_segmentation_pipeline.md) — Whole-body anatomy segmentation notes
  - [`prostata_bimodal_xnat_upload_runbook.md`](../prostata_bimodal_xnat_upload_runbook.md) — Historical 2026-11-13 CADS repair & upload runbook

---

## Architecture Overview

To keep clinical startup and slice scrolling sub-second, MedEye3D precomputes all spatial resampling, anatomical atlases, lymph node stations, bone sub-compartments, 3D edge/diffusivity fields, and report metadata offline into one HDF5 container per patient (`preprocessed_volumes.h5`):

1. **Grid & Orientation Convention (`_meta_/preflipped = 1`)**
   - Baseline CT (`BASELINE/Fixed_CT_Volume_0.nii.gz`) defines the canonical reference voxel grid `(Z, Y, X)` in NumPy/HDF5 (viewed as `(X, Y, Z)` in Julia/Makie/Vulkan).
   - Transverse axis-1 (`dims=2` in Julia, `axis=1` in NumPy) is pre-flipped at preprocessing time (`_meta_/preflipped = 1`) so MedEye3D loads volumes without runtime array reversal.
   - Follow-up / multimodal MRI volumes (`T2`, `ADC`, `DWI`, `T1`, `DCE`, etc.) and 8-zone prostate segmentations are resampled into baseline CT space under `TFM_*` groups without overwriting `BASELINE/max_anatomy.nii.gz`.

2. **Whole-Body Anatomy + CADS + Authoritative Lymph Node Stations (External `lymph_node_rules` Repo)**
   - **Repository boundary**: Whole-body CT segmentation (`raw/packed_anatomical_tensor.h5`) and Julia GPU lymph node station generation (`lymph_nodes/` / `raw/final_results_gpu.h5`) are produced in the **separate `lymph_node_rules` repository** (`LN_RULES_DIR`, uploaded to XNAT `ANATOMICAL_TENSOR` / `LYMPH_NODE_RESULTS`). `MedEye3d.jl` consumes those precomputed masks read-only — no local `lymph_node_rules` copy or LN worker daemon is maintained here.
   - **Primary organs & vessels**: Unpacked from `raw/packed_anatomical_tensor.h5` (with `raw/packed_anatomical_tensor_registry.json` or embedded HDF5 `registry`). Excluded non-organ helper masks (`body`, `torso_fat`, `subcutaneous_fat`, `skin`, `skeletal_muscle`, `abdominal_cavity`, `thoracic_cavity`, `thorax_wall`, `oblique_hyoid_plane`, `digastric_plane_*`, etc.) are purged via `is_excluded_non_organ_mask()` in [`cads_anatomy_wiring.py`](../../scripts/preprocessing/cads_anatomy_wiring.py).
   - **Authoritative Lymph Nodes**: [`iter_authoritative_lymph_node_masks()`](../../scripts/preprocessing/cads_anatomy_wiring.py) gives strict precedence to `<case_dir>/lymph_nodes/` (`all_lymph_nodes.nii.gz` + `lymph_node_labels.json`, plus any standalone `<station>.nii.gz`), falling back to `raw/final_results_gpu.h5` from the external `lymph_node_rules` pipeline / XNAT `LYMPH_NODE_RESULTS`. Old lymph node labels from `packed_anatomical_tensor.h5` are skipped when authoritative LN masks exist.
   - **Lung-only LN protection**: Only lung labels (`is_lung_label_name()`) and CADS high-priority structures (`531..539`) may override lymph node voxels; no unrelated organ overwrites LN stations.
   - **CADS Task 556 / 558 (`531..539`)**: `seminal_vesicle` (`531`), `sigmoid_colon` (`532`, replacing only overlapping colon or empty voxels), `lacrimal_gland_left/right` (`533/534`), `optic_chiasm` (`535`), `optic_nerve_left/right` (`536/537`), `spinal_canal` (`538`), `pituitary_gland` (`539`).

3. **3D Neural Edge & Heat-GDT Diffusivity Precomputation**
   - **`BASELINE/diffusivity`** (Both CT+PET) & per-MRI **`TFM_*/diffusivity`** via champion `TotalSegEdgePDE` (`gamma = 10.095886`, `epsilon_D = 1e-5`).
   - **`BASELINE/edge_ct` & `BASELINE/diffusivity_ct`** via `TotalSegEdgeNet v1` ([`apply_totalseg_edge_v1.py`](../../scripts/preprocessing/apply_totalseg_edge_v1.py), [`totalseg_edge_v1_model.py`](../../scripts/preprocessing/totalseg_edge_v1_model.py)).
   - **`BASELINE/edge_pet` & `BASELINE/diffusivity_pet`** via `TotalSegEdgePDE_PET` ([`add_pet_diffusivity.py`](../../scripts/preprocessing/add_pet_diffusivity.py), [`totalseg_edge_pde_pet_model.py`](../../scripts/preprocessing/totalseg_edge_pde_pet_model.py)).

4. **Lesion Cleanup, Organ Mapping, Semantic Filter & Radiological Reports**
   - Unconditional edge-slice lesion trim (`z in {0, 1, Z-2, Z-1}`) via `remove_edge_slice_lesions()` ([`apply_edge_slices_and_semantic_filter.py`](../../scripts/preprocessing/apply_edge_slices_and_semantic_filter.py)).
   - Physiological lacrimal gland (`533/534`) false-positive lesions are filtered out via `filter_lacrimal_lesions()`.
   - Seminal vesicle / organ mapping, 1-based `CENTROIDS/*_lid*` and `*_ln*`, `_meta_/segment_names.json`, `_meta_/matches.json`, and `_meta_/scene_hierarchy.json` are synchronized cleanly, with zero `*_expert` datasets and `_meta_/modification_status.json` (`manually_modified = false`).
   - [`populate_radiological_reports.py`](../../scripts/preprocessing/populate_radiological_reports.py) embeds German + `EnglishDescription` reports into `_meta_/metadata.json`, and `compute_and_store_semantic_filter()` populates `_meta_/semantic_filter.json`.

5. **Verification & Unit Tests**
   - Cohort invariant verifier: [`scripts/preprocessing/audit_full_cohort.py`](../../scripts/preprocessing/audit_full_cohort.py)
   - Python invariant tests: [`test/test_preprocessing_invariants.py`](../../test/test_preprocessing_invariants.py)
   - Julia navigation/XNAT/display tests: [`test/test_lesion_navigation_fix.jl`](../../test/test_lesion_navigation_fix.jl)

---

## Clean Scripts Inventory (`scripts/preprocessing/`)

### `Prostata_bimodal_PIPELINE` Orchestration & Repair
| Script | Role |
|---|---|
| [`batch_process_cohort_dual_gpu.py`](../../scripts/preprocessing/batch_process_cohort_dual_gpu.py) | End-to-end dual-GPU HDF5 builder, XNAT client, registration, MRI ranking, and neural diffusivity pipeline |
| [`cads_anatomy_wiring.py`](../../scripts/preprocessing/cads_anatomy_wiring.py) | Authoritative LN station precedence, lung-only LN protection, CADS `531..539` merging, anatomy sanitation |
| [`repair_max_anatomy_all.py`](../../scripts/preprocessing/repair_max_anatomy_all.py) | Rebuilds `BASELINE/max_anatomy.nii.gz` + `ATLAS/max_anatomy`, syncs `xnat_cache`, and streams HDF5s to XNAT |
| [`repair_cohort_mri_registration.py`](../../scripts/preprocessing/repair_cohort_mri_registration.py) | Recomputes Arun SVD + Prostate SWD registrations (`Transform_Landmark_SWD.h5`) and resamples MRIs/anatomy |
| [`repair_cohort_adc_and_dwi.py`](../../scripts/preprocessing/repair_cohort_adc_and_dwi.py) | Recovers quantitative `ADC` (`dDWI`/`IsoADC`/`s)_*`), skips RGB perfusion overlays, de-interleaves stacked DWI |
| [`repair_cohort_lesions.py`](../../scripts/preprocessing/repair_cohort_lesions.py) | Repairs baseline PSMA PET/CT lesion labels (`PET_Lesions_0.nii.gz`) and lesion metadata |
| [`run_prostate_seg_missing.py`](../../scripts/preprocessing/run_prostate_seg_missing.py) | Runs `Dataset501_ProstateZonesLesions` (`T2W + ADC + DWI`) for 8 prostate zones + `Label 9` lesions |
| [`apply_edge_slices_and_semantic_filter.py`](../../scripts/preprocessing/apply_edge_slices_and_semantic_filter.py) | Unconditional edge-slice lesion trim (`z in {0,1,Z-2,Z-1}`), lacrimal filter, and LLM semantic filter |
| [`populate_radiological_reports.py`](../../scripts/preprocessing/populate_radiological_reports.py) | Embeds Excel PSMA PET + MRI reports (`EnglishDescription`) into `_meta_/metadata.json` |
| [`audit_full_cohort.py`](../../scripts/preprocessing/audit_full_cohort.py) | Full-cohort HDF5 invariant, CT–MRI prostate Dice, `xnat_cache`, and live XNAT audit CLI |

### Neural Edge & Heat-GDT Diffusivity Models
| Script | Role |
|---|---|
| [`apply_totalseg_edge_v1.py`](../../scripts/preprocessing/apply_totalseg_edge_v1.py) | Computes CT-only `BASELINE/edge_ct` and `BASELINE/diffusivity_ct` (`TotalSegEdgeNet v1`) |
| [`totalseg_edge_v1_model.py`](../../scripts/preprocessing/totalseg_edge_v1_model.py) | `TotalSegEdgeNet v1` PyTorch architecture and diffusivity helper |
| [`add_pet_diffusivity.py`](../../scripts/preprocessing/add_pet_diffusivity.py) | Computes PET-only `BASELINE/edge_pet` and `BASELINE/diffusivity_pet` (`TotalSegEdgePDE_PET`) |
| [`totalseg_edge_pde_model.py`](../../scripts/preprocessing/totalseg_edge_pde_model.py) | Both CT+PET `TotalSegEdgePDE` neural edge + diffusivity backbone |
| [`totalseg_edge_pde_pet_model.py`](../../scripts/preprocessing/totalseg_edge_pde_pet_model.py) | PET-only fusion variant (`TotalSegEdgePDE_PET`) |
| [`generate_modality_diffusivities.jl`](../../scripts/preprocessing/generate_modality_diffusivities.jl) | Julia HED3D-Ultra CLI generating CT-only / PET-only diffusivities |
| [`compute_diffusivity.jl`](../../scripts/preprocessing/compute_diffusivity.jl) | Core Julia HED3D-Ultra sliding-window edge & diffusivity computation |

### Lu-177, Single-Case XNAT & Shared Utilities
| Script | Role |
|---|---|
| [`preprocess_dataset.jl`](../../scripts/preprocessing/preprocess_dataset.jl) | Multi-timepoint Julia resampling, scene hierarchy builder, and HDF5 packager (`xnat_preproc_worker.jl`) |
| [`preprocess_all_cases.sh`](../../scripts/preprocessing/preprocess_all_cases.sh) | Batch Lu-177 shell runner calling `preprocess_dataset.jl` + `prefill_anatomy.jl` |
| [`prefill_anatomy.jl`](../../scripts/preprocessing/prefill_anatomy.jl) | Automated lesion geometry, bone subsegmentation, and clinical rule prefill |
| [`recompute_bones.jl`](../../scripts/preprocessing/recompute_bones.jl) | Cortical surface vs trabecular marrow classification for skeletal lesions |
| [`run_bone_subsegmentation.jl`](../../scripts/preprocessing/run_bone_subsegmentation.jl) | KernelAbstractions morphological bone subsegmentation CLI |
| [`pack_h5_gzip.py`](../../scripts/preprocessing/pack_h5_gzip.py) / [`pack_h5_meta.py`](../../scripts/preprocessing/pack_h5_meta.py) | Primary NIfTI-to-HDF5 packing helpers used by `preprocess_dataset.jl` |
| [`translate_reports.jl`](../../scripts/preprocessing/translate_reports.jl) | German-to-English radiology report translation helper (`TranslateReports`) |
| [`prepare_psma_case_for_medeye3d.py`](../../scripts/preprocessing/prepare_psma_case_for_medeye3d.py) | Longitudinal PSMA + mpMRI single-case assembler |
| [`reprocess_mri_tps.jl`](../../scripts/preprocessing/reprocess_mri_tps.jl) / [`embed_prostate_anatomy.jl`](../../scripts/preprocessing/embed_prostate_anatomy.jl) | Standalone Julia MRI resampling and prostate zone embedding utilities |
| [`create_petct_student_dataset.py`](../../scripts/preprocessing/create_petct_student_dataset.py) | Builds the `Pat23_PETCT_student` training dataset (`BASELINE` blank vs expert TP1) |
