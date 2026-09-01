# Lesion Metadata: Auto-Fill Pipeline & Dropdown Schema

This document describes the automatic metadata population system for lesion annotations, including anatomy auto-fill, SUV computation, PROMISE scoring, volume calculation, and the extensible dropdown schema.

---

## Dropdown Schema (`def.json`)

All metadata dropdowns are defined in [`extension/data/def.json`](../extension/data/def.json). Each entry has:

```json
{
  "full": "Human-readable description of the field",
  "allowed_answer": ["Option 1", "Option 2", "..."],
  "short q": "Field Name (used as dict key)",
  "category": ["Category Group"],
  "meta_or_prostate": "both|bone_meta|prostate",
  "default_answer": "",
  "comments": { "Option 1": "Detailed explanation..." }
}
```

### Current Fields (20 questions)

| # | Field | Options | Type | Auto-Computed? |
|---|---|---|---|---|
| 1 | Radioligand Type | 4 | Dropdown | No |
| 2 | Lesion tracking name? | — | Free text | **Yes** (auto-generated from organ + lid + TP) |
| 3 | Anatomic Location | 20 | Dropdown | **Yes** (from anatomy atlas) |
| 4 | Anatomical Sublocation | 45 | Dropdown | **Yes** (from anatomy atlas) |
| 5 | Inner Texture / Density / Attenuation | 18 | Dropdown | No |
| 6 | Border and Margin | 11 | Dropdown | No |
| 7 | Lesion Shape | 11 | Dropdown | No |
| 8 | Lesion Orientation | 7 | Dropdown | No |
| 9 | Macroscopic Pattern | 7 | Dropdown | No |
| 10 | Relation to Bone Marrow | 2 | Dropdown (bone only) | No |
| 11 | Periosteal Reaction | 8 | Dropdown (bone only) | No |
| 12 | Other Structural & Soft Tissue Changes | 39 | Multiselect | No |
| 13 | SUV Quantitative Metrics & References | 11 | Multiselect | No |
| 14 | Clinical Context & Staging Variables | 25 | Multiselect | No |
| 15 | SUV max | — | Free text | **Yes** (auto-computed from PET volume) |
| 16 | PRIMARY score pattern? | — | Free text | **Yes** (auto-computed) |
| 17 | PSMA-RADS 2.0 | — | Free text | **Yes** (auto-computed) |
| 18 | Alternative Hypothesis (False Positive) | 85 | Dropdown | No |
| 19 | Certainty | slider 0-10 | Slider | No |
| 20 | Comment | — | Free text | No |

### Adding Custom Options

Users can add custom options at runtime via the "Add option..." textbox. Custom options are persisted in `extension/data/custom_options.json` via `load_custom_options()` / `save_custom_options()`.

### Extending Dropdowns

To add new options permanently, edit `def.json` and append entries to the `"allowed_answer"` array. The JSON must remain valid — validate with:

```bash
python3 -c "import json; json.load(open('extension/data/def.json')); print('OK')"
```

### Source Reference

All dropdown options for Alternative Hypothesis, Anatomic Location, Sublocation, Structural Changes, SUV Metrics, and Clinical Context are derived from the medical reference document [`docs/main_note_med.md`](main_note_med.md), covering:

- PROMISE scoring (0–3) and RECIP 1.0 response criteria
- Appendicular & axial skeleton lesion mimics (enchondroma, fracture, ABC, SBC, bone island, bone infarct, fibrous dysplasia, Paget's, etc.)
- Prostate-specific findings (PSA kinetics, miTNM staging, T3a/T3b/T4 criteria)
- Pelvic & abdominal mimics (LN criteria, ganglia, bowel, splenules, RCC, etc.)
- Head, neck, chest findings (thyroid, thymoma, meningioma, sarcoidosis, etc.)
- Imaging artifacts (halo, motion, post-surgical, hot-clot, radiolysis, flare)

---

## Anatomy Auto-Fill Pipeline

When a lesion is selected, the system automatically fills `BaseAnatomy`, `Anatomic Location`, and `Anatomical Sublocation` from the TotalSegmentator anatomy atlas.

### Data Flow

```
Lesion selected (lid=N)
        │
        ▼
apply_state(get_lesion_state(db, id))
        │
        ├── 1. Resolve raw_organ name:
        │       ├── global_organ_mapping[lid] (from HDF5 _meta_/organ_mapping)
        │       └── If empty or "Unknown" → volume-based scan with bone priority
        │               ├── Loads mask from tp_data_cache[tp_idx].mask
        │               ├── classify_and_pick_best_organ(mask, atlas, ts_names, lid)
        │               │   Scans ALL lesion voxels, counts overlapping atlas labels,
        │               │   picks best using priority: bone > organ > lymph > vessel > muscle
        │               └── Falls back to centroid-based atlas lookup if mask unavailable
        │
        ├── 2. lookup_anatomy(raw_organ) → max_anatomy_to_ontology.json entry
        │       Returns: { "detailed", "side", "anatomic_location", "anatomical_sublocation" }
        │
        ├── 3. Auto-fill BaseAnatomy (ONLY if BaseAnatomy is empty in saved data)
        │       └── Sets t_base from entry["detailed"], t_side from entry["side"]
        │
        ├── 4. Auto-fill Anatomic Location & Sublocation (independent of BaseAnatomy)
        │       ├── Runs when these fields are empty in saved data, even if BaseAnatomy is saved
        │       ├── Injects values into data dict (prevents generic restore loop from overwriting)
        │       └── Persists via db_updates (saved for future sessions)
        │
        ├── 5. Generic schema restore loop (L3410): restores ALL fields from data dict
        │       └── Auto-filled values are already in data, so they survive this step
        │
        └── 6. If lookup_anatomy returns nothing → map_ts_to_anatomy() static fallback
                Returns: (anatomic_location, anatomical_sublocation) from TS_TO_ANATOMY dict
```

### Label Loading: Per-TP Labels Are Authoritative

> [!WARNING]
> The baseline `_meta_/max_anatomy_labels.json` and the per-TP `_meta_/anatomy_labels_tp_N.json` use **completely different integer-to-name mappings** because they come from different TotalSegmentator versions (201-class vs 319-class). The `ATLAS/max_anatomy` volume uses the per-TP numbering. Loading baseline labels would assign wrong names to 196 out of 201 label IDs.

At load time, `run_interactive_mrb.jl` loads **per-TP labels first** as the authoritative source (they match the atlas numbering), filtering out junk intermediate class names (`*_class_*`). Only if no per-TP labels exist does it fall back to baseline labels. This yields ~234 real structure labels covering all atlas IDs.

### Volume-Based Organ Mapping with Bone Priority

The `map_lesions_to_organs()` function uses a **volume-based scan** instead of single-centroid lookup:

1. For each lesion, iterates ALL voxels where `mask == lid`
2. Looks up each voxel in the atlas, counting occurrences per label
3. Filters to KNOWN labels only (present in `ts_names`)
4. Picks the best label using tissue priority:
   - **Priority 1 (Bone)**: vertebrae, ribs, femur, skull, sternum, mandible, hyoid, etc.
   - **Priority 2 (Organ)**: liver, kidney, lung, spleen, heart, brain, etc.
   - **Priority 3 (Lymph)**: lymph nodes
   - **Priority 4 (Vessel)**: aorta, arteries, veins
   - **Priority 5 (Muscle)**: all soft tissue / muscles
5. Within the same priority class, picks the label with the most voxels

**Bone priority rule**: A lesion partially in bone and partially in muscle is always classified as bone, even if the bone overlap is smaller. This ensures bone metastases are correctly identified.

### Key Files

| File | Purpose |
|---|---|
| `data/max_anatomy_to_ontology.json` | 239 entries mapping TotalSegmentator organ names to ontology fields (anatomic_location, sublocation, side, detailed name, lesion_type) |
| `LesionAssociation.jl:L502–740` | `classify_tissue_priority()`, `count_atlas_overlap()`, `pick_best_organ()`, `classify_and_pick_best_organ()`, `map_lesions_to_organs()` — volume-based organ mapping with bone priority |
| `LesionMetadataWindow.jl:L258–302` | `load_anatomy_mapping()`, `lookup_anatomy()` — JSON-based lookup with lowercase-first then exact-case matching |
| `LesionMetadataWindow.jl:L304–351` | `TS_TO_ANATOMY` const dict — static fallback mapping for when JSON lookup fails |
| `LesionMetadataWindow.jl:L3248–3297` | `apply_state()` auto-fill section — orchestrates the full pipeline with volume scan fallback |
| `run_interactive_mrb.jl:L79–107` | Label merging: loads baseline labels then merges per-TP labels |

### Organ Mapping Sources

1. **`_meta_/organ_mapping`** in the preprocessed HDF5: Dictionary `{lesion_id: organ_name}` computed by `map_lesions_to_organs()` during preprocessing. Uses volume-based scanning with bone priority over ALL lesion voxels.

2. **Volume-based atlas fallback**: If the organ mapping returns `""` or `"Unknown"`, the runtime `apply_state()` loads the mask from `tp_data_cache` and runs `classify_and_pick_best_organ()` to scan all lesion voxels against the atlas with bone priority.

3. **Centroid-based fallback**: Last resort if the mask volume is unavailable. Looks up `lesion_centroids_cache[(tp, lid)]`, then queries the atlas volume at those coordinates.

4. **Static dict fallback**: If `lookup_anatomy()` returns `nothing`, the `map_ts_to_anatomy()` function uses a hardcoded `TS_TO_ANATOMY` dict to map common organ names to locations.

### "Unknown" Organ Handling

Some lesions have their organ mapping set to `"Unknown"` when their centroid falls outside any TotalSegmentator atlas region (e.g., in soft tissue not covered by the atlas). Previously, `"Unknown"` bypassed the centroid-based fallback because the code only triggered the fallback for **empty** strings. Now, `"Unknown"` is treated as equivalent to empty:

```julia
# Before (buggy):
if isempty(raw_organ) && _MEH.global_ts_atlas[] !== nothing
# After (fixed):
if (isempty(raw_organ) || raw_organ == "Unknown") && _MEH.global_ts_atlas[] !== nothing
```

### Independent Location/Sublocation Auto-Fill

The Anatomic Location and Sublocation fields are auto-filled **independently** of BaseAnatomy. This means:
- Even if BaseAnatomy is already saved (e.g., from a previous session), the Location and Sublocation will still be auto-filled if empty
- Auto-filled values are injected into the `data` dict AND persisted via `db_updates`, preventing the generic restore loop (L3410) from overwriting them
- This solves the race condition where the auto-fill set `w.i_selected[]` but the later restore loop reset it to "- select -"

---

## SUV Auto-Computation Pipeline

SUV values are automatically computed when a lesion is selected, and asynchronously recomputed when segmentation changes.

### Components

| Function | Location | Purpose |
|---|---|---|
| `compute_suv_max_at_centroid(pet_vol, centroid)` | L561 | SUVmax in 3×3×3 neighborhood around centroid |
| `compute_background_suvs(pet_vol, ts_atlas, ts_names)` | L581 | Mean SUV in liver (30mm sphere), parotid (bilateral centers), blood pool (aorta center 1/3) |
| `compute_lesion_suv_string(lid, tp_idx)` | L725 | Formats: `"Max: X.X ; Parotid: X.X ; Liver: X.X ; Blood: X.X"` |
| `compute_promise_score(suv_max, bg)` | L778 | PROMISE 0–3 scoring: 0=<blood, 1=blood≤x<liver, 2=liver≤x<parotid, 3=≥parotid |
| `compute_suv_comparison_string(suv_max, bg)` | L795 | Human-readable comparison: `"Liver: ≥ (1.5×) ; Parotid: < (0.8×) | PROMISE 2"` |

### Caching

| Cache | Type | Key | Invalidated By |
|---|---|---|---|
| `_bg_suv_cache` | `Dict{Int, Dict{String, Float32}}` | tp_idx | Never (background organs don't change) |
| `_lesion_suv_cache` | `Dict{Tuple{Int,Int}, String}` | (tp_idx, lid) | `invalidate_suv_for_lesion()` |
| `_volume_cache` | `Dict{Tuple{Int,Int}, Dict{String, Float64}}` | (tp_idx, lid) | `invalidate_suv_for_lesion()` |
| `lesion_centroids_cache` | `Dict{Any, Vector{Int}}` | (tp_idx, lid) or lid | `invalidate_suv_for_lesion()` |

### Async Recomputation

After any mask modification, `invalidate_and_recompute_lesion_metrics_async!(lid, tp_idx, mask_vol)` is called. This function:

1. **Synchronously** invalidates all caches for the lesion
2. **Synchronously** recomputes the centroid from the current mask (fast scan)
3. **Asynchronously** (`Threads.@spawn`) recomputes volume and SUV string, populating caches

This ensures the next `apply_state()` call finds fresh cached values (O(1) lookup), or computes from scratch on cache miss.

**Trigger points**:
- `reactToAIInferenceResult()` in `MakieEventHandlers.jl` — after AI writes segmentation mask
- `react_to_draw()` in `ReactOnMouseClickAndDrag.jl` — after manual painting mouse release
- `reactToGenManual()` in `MakieEventHandlers.jl` — after manual bone subsegmentation

---

## Volume & RECIP Auto-Computation

### Volume Computation

`compute_lesion_volume(lid, tp_idx)` counts voxels with the target label in the mask and multiplies by voxel spacing (from `Main.first_spacing / Main.HIRES_FACTOR`). Returns:
- `volume_mm3`: total volume in cubic millimeters
- `volume_cc`: total volume in cubic centimeters (mL)
- `voxel_count`: raw voxel count
- `diameter_mm`: equivalent sphere diameter

**Precomputed at startup**: `precompute_all_volumes!(mask_vol, tp_idx)` does a single O(N) pass over the mask, counting all unique labels and caching their volumes. This is called alongside `precompute_mask_centroids!()` during TP loading.

### RECIP Classification

For lesions tracked across time points (via match groups in `LesionAssociation`), `compute_match_analysis()` computes:
- Volume delta (current vs. baseline): absolute cc and percentage
- SUV delta: absolute and percentage
- RECIP classification: CR (disappeared), PR (>30% decrease), SD (stable), PD (>20% increase)

### Match Analysis Display

The metadata panel shows a one-line summary:
```
Vol: 2.34cc (16.5mm⌀) | ΔVol: +15.2% (+0.31cc) | ΔSUV: +2.1 (+18.3%) | Grp 3 [2 TPs] RECIP-PD
```

---

## Lesion Type Auto-Detection

When `BaseAnatomy` is determined, the system also auto-detects the lesion type (bone_meta, prostate, soft_tissue, lymph_node, etc.) using the JSON ontology entry's `"lesion_type"` field. This controls which category-specific dropdowns are shown (e.g., "Relation to Bone Marrow" only appears for bone_meta lesions).

If the organ mapping returns `"Unknown"`, the same centroid-based atlas fallback is used for type detection (since the same fix was applied at L2993).
