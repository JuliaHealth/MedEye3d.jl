# Changelog

All notable changes to MedEye3d.jl are documented here.

---

## [2026-09-01] — Anatomy Mapping: Label Mismatch Fix + Bone Priority

### Fixed

- **Incompatible label numbering caused completely wrong anatomy names**: The HDF5 atlas volume (`ATLAS/max_anatomy`) was stored from a follow-up TP that used a **319-class TotalSegmentator** run with one integer-to-name mapping, but the loaded label dictionary (`_meta_/max_anatomy_labels.json`) was from the baseline scan that used a **201-class TotalSegmentator** with a **completely different numbering scheme**. Result: 196 out of 201 shared integer IDs mapped to different structures (e.g., ID 171 = `sartorius_right` in baseline vs `hip_left` in the atlas). This caused most lesion names to be wrong or "Unknown". **Fix**: Load per-TP labels (`anatomy_labels_tp_N.json`) as the authoritative source since they match the atlas numbering. Only fall back to baseline labels when no per-TP labels exist. (`run_interactive_mrb.jl` L79–115, `preprocess_dataset.jl` L374–400).

- **Centroid-only organ lookup missed bone structures**: The `map_lesions_to_organs` function used a single centroid point to classify lesions, which could land on muscle/background even when the lesion significantly overlapped bone. Now replaced with a **volume-based scan** that counts ALL overlapping atlas labels per lesion, with **bone > organ > lymph > vessel > muscle** priority. A lesion partially in bone and partially in muscle is always classified as bone. (`LesionAssociation.jl` L502–740).

- **38 new ontology entries for 319-class TotalSegmentator labels**: Added anatomy-to-UBERON ontology mappings for head/neck structures (`mandible`, `hyoid`, `parotid_gland`, `eye`, `thyroid_cartilage`, `cricoid_cartilage`, etc.), heart subcomponents, airways/cavities (`nasopharynx`, `oropharynx`, `hypopharynx`, `larynx_air`), and internal vessels (`internal_carotid_artery`, `internal_jugular_vein`). Total ontology entries: 201 → 239. (`max_anatomy_to_ontology.json`).

- **Runtime fallback now uses volume scan**: The `apply_state()` fallback for new/unknown lesions now first tries a volume-based scan via `LesionAssociation.classify_and_pick_best_organ()` using the mask from `tp_data_cache`, only falling back to centroid lookup if the mask is unavailable. (`LesionMetadataWindow.jl` L3248–3297).

- **Muscle/bone classification fix**: `levator_scapulae`, `subscapularis`, `infraspinatus` etc. were incorrectly classified as bone because they contain bone keywords (`scapula`). Added muscle exclusion list and changed `scapula` keyword to `scapula_` (with trailing underscore) to match only the bone structure `scapula_left`/`scapula_right`. Also fixed `classify_organ_to_lesion_type()` to check muscles before bones. (`LesionAssociation.jl` L512–525, L723–732).

---

## [2026-09-01] — Metadata Dropdowns, Anatomy Auto-Fill Fix, Async SUV Recomputation

### Fixed

- **New lesion name not based on anatomy**: New lesions were created with the generic name `"N - New Lesion"` regardless of anatomical location. Now they are auto-named based on the TotalSegmentator anatomy atlas at the current viewer slice position (e.g., `"5: Left Femur"`). When a lesion's anatomy is later determined (e.g., after painting), the dropdown entry is automatically renamed from `"New Lesion"` to the detected anatomy. (`LesionMetadataWindow.jl` L2460–2565, L3325–3355).

- **Selecting lesion from dropdown didn't jump to its location**: New lesions used the format `"N - New Lesion"` (dash separator) but all numeric ID parsers expected `"N: name"` (colon separator). This caused `parse_lesion_id()` to return `nothing`, so `SyncLesionEvent` was never dispatched and the viewer didn't scroll. Introduced `parse_lesion_id()` utility that handles both `:` and ` - ` formats, and updated all 10+ ID parsing locations to use it. (`LesionMetadataWindow.jl` L400–420 + multiple call sites).

- **Map Lesions section showed raw organ IDs instead of full names**: The compare mode "Map Lesions" section showed raw TotalSegmentator names (e.g., `"femur_left"`) instead of full display names from the dropdown. Now shows the same names as the main lesion dropdown, with anatomy lookup fallback for lesions from other timepoints. (`LesionMetadataWindow.jl` L2800–2856, L2730–2768).

- **Map Lesions section overlapping in single TP view**: The "Map Lesions (Compare Mode)" section's dynamic `GridLayout` content could render over other sections even when the section was hidden (`Fixed(0)` rows). Now the section starts with `default_open=false`, and `_build_match_display!()` checks `cv_active[]` and skips building content when not in compare mode. The section is force-opened when entering compare mode. (`LesionMetadataWindow.jl` L2551, L2580–2585, L2424).

- **Anatomy auto-fill bypass for "Unknown" organ mapping**: When `map_lesions_to_organs` stored `"Unknown"` for a lesion (centroid outside any TotalSegmentator atlas region), the auto-fill code skipped the centroid-based atlas fallback. Now `"Unknown"` is treated as empty, enabling the fallback to find the correct anatomy from the atlas. Affected fields: `BaseAnatomy`, `Anatomic Location`, `Anatomical Sublocation`. Fixed in three locations in `LesionMetadataWindow.jl` (L2993, L3126, L3150).

- **Stale SUV/volume values after segmentation changes**: After AI segmentation or manual painting modified a lesion's mask, the cached SUV and volume values were stale. Now `apply_state()` always reads from cache (which is invalidated on mask change) instead of using previously-persisted values. (`LesionMetadataWindow.jl` L3289–3308).

- **Volume cache not invalidated after mask changes**: `invalidate_suv_for_lesion()` only cleared `_lesion_suv_cache` but not `_volume_cache`. Now clears both. (`MakieEventHandlers.jl` L1060–1074).

### Added

- **`invalidate_and_recompute_lesion_metrics_async!(lid, tp_idx, mask_vol)`**: Unified function that invalidates SUV/volume/centroid caches, recomputes centroids from the current mask, and asynchronously (`Threads.@spawn`) recomputes volume and SUV string. Called automatically from:
  - `reactToAIInferenceResult()` — after AI segmentation
  - `react_to_draw()` — after manual painting mouse release
  - `reactToGenManual()` — after manual bone subsegmentation
  (`MakieEventHandlers.jl` L1083–1145)

- **59 new dropdown entries** in `def.json` (derived from [`docs/main_note_med.md`](docs/main_note_med.md)):
  - **Alternative Hypothesis**: +29 entries (Aneurysmal Bone Cyst, Simple Bone Cyst, Brown Tumor, Modic Type 1, Bone Marrow Activation, Inflammatory Spondylitis, Osteoid Osteoma, Desmoid Tumor, Nodular Fasciitis, Lipoma/Angiolipoma/Hibernoma, Intramuscular Myxoma, Gynecomastia/PASH, Elastofibroma Dorsi, Thyroid/Parathyroid Adenoma, Thymoma/Thymic Hyperplasia, Cerebral Infarction, Synchronous Colorectal Cancer, Primary Lung Cancer, Malignant Melanoma Metastasis, Lymphoma, Tuberculosis, Diverticulitis, Cholelithiasis, Dermatofibroma, Active Vascular Calcification, Esophagitis, Bronchogenic Cyst, Pheochromocytoma)
  - **Anatomic Location**: +6 entries (Head/Skull Base/Intracranial, Neck/Thyroid Region, Chest/Mediastinum/Thorax, Seminal Vesicles, Adrenal Gland, Breast/Chest Wall)
  - **Anatomical Sublocation**: +8 entries (Intertrochanteric Region, Seminal Vesicle Lumen, Neural Foramen/Nerve Root, Splenic Hilum, Subscapular/Infrascapular, Subareolar/Breast Fat Pad, Anterior Mediastinum, Gallbladder Fossa)
  - **SUV Metrics**: +4 entries (PROMISE liver/parotid/cold thresholds, LBR > 2.5)
  - **Clinical Context**: +6 entries (PSA Persistence, Post-BCG, Prior Pelvic Radiation, Bisphosphonate Use, Corticosteroid Use, Sickle Cell Disease)
  - **Other Structural Changes**: +5 entries (Shepherd's Crook Deformity, Picture Frame Vertebra, Eggshell Calcification, Bamboo Spine, Popcorn Calcification)

### Documentation

- New: [`docs/lesion_metadata_autofill.md`](docs/lesion_metadata_autofill.md) — comprehensive doc for the metadata auto-fill pipeline, dropdown schema, SUV computation, PROMISE scoring, volume/RECIP calculation
- Updated: [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) — added section 12 (Async SUV/Volume Recomputation)
- Updated: [`docs/architecture_makie_buttons.md`](docs/architecture_makie_buttons.md) — added 6 missing event types to the dispatch table

### Files Changed

| File | Lines Changed | Description |
|---|---|---|
| `src/display/LesionMetadataWindow.jl` | ~25 | Auto-fill "Unknown" bypass fix + always-recompute SUV |
| `src/display/GLFW/MakieEventHandlers.jl` | ~90 | `invalidate_and_recompute_lesion_metrics_async!`, AI result hook, enhanced invalidation |
| `src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl` | ~5 | Paint release → unified recompute |
| `extension/data/def.json` | ~65 | 59 new dropdown entries |
| `docs/lesion_metadata_autofill.md` | new (180 lines) | New documentation |
| `docs/PERFORMANCE.md` | +35 | Section 12 |
| `docs/architecture_makie_buttons.md` | +6 | Event types table |

### Test Results

| Test Suite | Result |
|---|---|
| Visual Regression | 29/29 passed ✓ |
| Windowing Isolation | 17/17 passed ✓ |
| JSON Validation | 20 questions, all valid ✓ |

---

## [2026-09-01] — Anatomy Visibility Fix

### Fixed

- **Anatomy overlay not visible when toggled ON**: Anatomy texture was filtered out by `reactToShowSingleLesion` (set `minAndMaxValue = [1,1]`), `reactToPaintVal`, and `reactToToggleLesion` conditions. Now all three exclude textures named `"Anatomy"`. The anatomy toggle ON handler enforces `minAndMaxValue = [0, 400]` and clears `allowedIDs`. Click-to-select also excludes Anatomy to prevent accidental lesion ID assignment.

### Files Changed

| File | Lines Changed | Description |
|---|---|---|
| `src/display/GLFW/MakieEventHandlers.jl` | L432, L523, L1263, L1794–1808 | Exclude Anatomy from single-lesion, paint, toggle conditions; enforce full range on toggle ON |
| `src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl` | L379, L389 | Exclude Anatomy from click-to-select |
