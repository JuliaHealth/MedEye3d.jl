# Lesion Metadata Panel, Longitudinal Tracking & Clinical Scoring

MedEye3d includes a comprehensive **Lesion Metadata Panel** (`src/display/LesionMetadataWindow.jl`) that provides structured clinical annotation, automated quantitative analysis, and longitudinal lesion tracking — all integrated with the OpenGL viewer through the event channel.

---

## 1. Panel Architecture

The metadata panel is a dedicated GLMakie window divided into collapsible sections:

```
┌─────────────────────────────────────────────────────────────────┐
│  [< Prev]   Lesion: 18: femur_left_L18 [Grp 5, 3 TPs]  [Next >]  │
├─────────────────────────────────────────────────────────────────┤
│  ▼ Lesion Navigation                                            │
│    - Lesion dropdown selector                                   │
│    - Sync / Jump / Delete buttons                               │
│    - Previous / Next lesion navigation                          │
├─────────────────────────────────────────────────────────────────┤
│  ▼ Lesion Metadata                                              │
│    ┌──────────────────────────────────────────────────────────┐ │
│    │ [Prostate] [Bone Meta] [Organ Meta] [Lymph Node]         │ │
│    │ Radioligand Type:  [68Ga-PSMA-11 ▼]                      │ │
│    │ Tracking Name:     [femur_left_L18_PET_TP0_PAT6]         │ │
│    │ Anatomic Location: [Axial Skeleton ▼]  (auto-filled)     │ │
│    │ Sublocation:       [Medullary Cavity ▼] (auto-filled)    │ │
│    │ Anatomy:           [Femur Left ▼]       Side: [Left ▼]   │ │
│    │ Anatomical Details: [+ Row]                               │ │
│    │   [Inside / Contained In ▼] [Femur Shaft ▼] [-]          │ │
│    │ ☐ No CT Correlate                                         │ │
│    │ Liver: ≥ (2.34×) ; Parotid: < (0.87×) | PROMISE 2       │ │
│    │ Vol: 1.24cc (13.3mm⌀) | ΔVol: +35% | RECIP-PD           │ │
│    ├──────────────────────────────────────────────────────────┤ │
│    │ Inner Texture:     [Sclerotic / Blastic ▼]               │ │
│    │ Border and Margin: [Well-Defined ▼]                      │ │
│    │ Lesion Shape:      [Round ▼]                              │ │
│    │ ...                                                       │ │
│    ├──────────────────────────────────────────────────────────┤ │
│    │ SUV max: Max: 12.3 ; Parotid: 8.1 ; Liver: 5.2 ; ...   │ │
│    │ Clinical Context:  [Gleason Score > 6 ▼]                 │ │
│    │ PRIMARY score:     [3b - focal intense ▼]                │ │
│    │ PSMA-RADS 2.0:     [4 - Highly suspicious ▼]            │ │
│    │ Alternative Hyp:   [None / Malignant Suspected ▼]        │ │
│    │ Certainty:         [5 ▼]                                  │ │
│    │ Comment:           [Free-text clinical notes...]          │ │
│    └──────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  ▼ Windowing Controls                                           │
│  ▼ Display Layers                                               │
│  ▼ Map Lesions (Cross-TP Matching)                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Searchable Dropdowns (Type-to-Filter)

All dropdown menus in the metadata panel support **type-to-filter** search:

1. **Click** any dropdown to open it — all options are shown
2. **Start typing** — the options list filters in real-time to show only matching entries
3. **Escape** — resets the filter and shows all options again
4. **Backspace** — removes the last typed character from the filter

This is implemented via a centralized `SearchableMenuState` system with a single figure-level keyboard handler, avoiding the performance issues of per-menu event listeners.

!!! tip "Searchable Fields"
    All dropdowns are searchable, including:
    - Anatomic Location (14 options)
    - Anatomical Sublocation (35 options)
    - Base Anatomy (~200 anatomical structures from max_anatomy ontology)
    - Anatomical Details relation + structure menus
    - All morphology, clinical context, and assessment fields

---

## 3. Lesion Type Classification & Dynamic Visibility

Four type buttons at the top of the metadata section classify the active lesion:

| Button | Description | Auto-Fills |
| :--- | :--- | :--- |
| **Prostate** | Primary prostate lesion | Location→Prostate Gland, Sublocation→PZ |
| **Bone Meta** | Skeletal metastasis | Location→Axial Skeleton, Texture→Sclerotic |
| **Organ Meta** | Visceral/soft tissue metastasis | Location→Solid Organ |
| **Lymph Node** | Lymph node metastasis | Location→Lymph Node Chain |

### Dynamic Field Visibility

Fields automatically show/hide based on the selected lesion type:

| Field | Prostate | Bone Meta | Organ Meta | LN Meta |
| :--- | :---: | :---: | :---: | :---: |
| PRIMARY Score | ✅ | ❌ | ❌ | ❌ |
| PSMA-RADS 2.0 | ❌ | ✅ | ✅ | ✅ |
| Bone Marrow Relation | ❌ | ✅ | ❌ | ❌ |
| Periosteal Reaction | ❌ | ✅ | ❌ | ❌ |
| Alternative Hypothesis | ❌ | ✅ | ✅ | ✅ |

---

## 4. Anatomical Auto-Fill from JSON Mapping

When a lesion is loaded, its segmentation mask overlap with the TotalSegmentator anatomy atlas is used to **automatically fill**:

- **Base Anatomy** dropdown (e.g., "Femur Left", "Liver", "Vertebrae T5")
- **Anatomic Location** (e.g., "Axial Skeleton", "Solid Organ")
- **Anatomical Sublocation** (e.g., "Medullary Cavity", "Prostate Peripheral Zone")
- **Side** (Left / Right / NA)
- **Lesion Type** (Prostate / Bone Meta / Organ Meta / Lymph Node)

This mapping is defined in `data/max_anatomy_to_ontology.json` (201 entries), which maps each TotalSegmentator organ name to structured annotation fields.

---

## 5. "No CT Correlate" Toggle

A toggle switch labeled **"No CT Correlate"** hides all CT-specific morphology fields when activated. This is used for PET-only lesions where no corresponding CT abnormality is visible.

### Fields Hidden When Checked
- Inner Texture / Density / Attenuation
- Border and Margin
- Lesion Shape
- Lesion Orientation
- Relation to Bone Marrow
- Periosteal Reaction
- Other Structural & Soft Tissue Changes

### Fields Always Visible
- Anatomic Location, Sublocation, Base Anatomy
- SUV metrics, PROMISE score, SUV comparison
- Macroscopic Pattern, Clinical Context
- PSMA-RADS, Alternative Hypothesis, Certainty, Comment

The toggle state is persisted as `NoCTCorrelate=true/false` in the lesion database.

---

## 6. Automated SUV Computation & PROMISE Scoring

### SUV Auto-Fill

When a lesion is selected, MedEye3d automatically computes:
- **SUVmax** from the PET volume at the lesion centroid (3×3×3 neighborhood)
- **Background reference SUVs** from TotalSegmentator-segmented organs:
  - **Liver** mean SUV (via liver label)
  - **Parotid** mean SUV (via parotid labels)
  - **Blood Pool** mean SUV (via vena cava / aorta labels)

The result is displayed as:
```
Max: 12.3 ; Parotid: 8.1 ; Liver: 5.2 ; Blood: 2.8
```

### PROMISE Score

The PROMISE molecular imaging score is automatically computed based on lesion SUVmax relative to reference organs:

| Score | Criteria | Clinical Meaning |
| :---: | :--- | :--- |
| **0** | SUVmax < Blood Pool mean | No significant uptake |
| **1** | Blood Pool ≤ SUVmax < Liver mean | Low uptake, likely benign |
| **2** | Liver ≤ SUVmax < Parotid mean | Moderate uptake, suspicious |
| **3** | SUVmax ≥ Parotid mean | High uptake, likely malignant |

### SUV Comparison Label

A green label displays the quantitative comparison:
```
Liver: ≥ (2.34×) ; Parotid: < (0.87×) ; Blood: ≥ (4.12×) | PROMISE 2
```

Each comparison shows:
- `≥` or `<` relative to the reference organ
- The **ratio** (e.g., `2.34×` means SUVmax is 2.34 times the liver mean)

---

## 7. Volume Computation & Match Analysis

### Automatic Volume Measurement

For every lesion, MedEye3d automatically computes:

| Metric | Description |
| :--- | :--- |
| **Volume (mm³)** | Voxel count × display-space voxel volume |
| **Volume (cc)** | Volume in cubic centimeters (= mm³ / 1000) |
| **Equivalent Diameter** | Sphere diameter with the same volume: $d = 2 \cdot \sqrt[3]{\frac{3V}{4\pi}}$ |

Volume computation uses the display-space voxel spacing:

$$V_{\text{voxel}} = \frac{\Delta x}{\text{HIRES}} \times \frac{\Delta y}{\text{HIRES}} \times \Delta z$$

where $\Delta x, \Delta y, \Delta z$ are the native CT spacing and HIRES is typically 2.0.

### Cross-Timepoint Match Analysis

When a lesion belongs to a **match group** (linked across time points), MedEye3d automatically computes **longitudinal changes** vs the baseline (earliest time point in the group):

| Metric | Formula |
| :--- | :--- |
| **Volume Delta (%)** | $\frac{V_{\text{current}} - V_{\text{baseline}}}{V_{\text{baseline}}} \times 100$ |
| **Volume Delta (cc)** | $V_{\text{current}} - V_{\text{baseline}}$ |
| **SUV Delta** | $\text{SUVmax}_{\text{current}} - \text{SUVmax}_{\text{baseline}}$ |
| **SUV Delta (%)** | $\frac{\text{SUVmax}_{\text{current}} - \text{SUVmax}_{\text{baseline}}}{\text{SUVmax}_{\text{baseline}}} \times 100$ |

### RECIP Response Classification

Based on the RECIP 1.0 criteria for PSMA-directed therapy response assessment:

| Category | Criteria | Description |
| :--- | :--- | :--- |
| **RECIP-CR** | Lesion disappeared (baseline volume > 0, current ≈ 0) | Complete Response |
| **RECIP-PR** | Volume decreased > 30% | Partial Response |
| **RECIP-SD** | Volume change between -30% and +20% | Stable Disease |
| **RECIP-PD** | Volume increased > 20% | Progressive Disease |

The result is displayed as a light blue label:
```
Vol: 1.24cc (13.3mm⌀) | ΔVol: +35.2% (+0.32cc) | ΔSUV: +2.1 (+18.3%) | Grp 5 [3 TPs] RECIP-PD
```

---

## 8. Cross-Timepoint Lesion Matching

### Match Groups

Lesions across different time points are linked into **match groups**. Each group has a unique ID and contains entries from multiple time points:

```
Group 5:
  TP0 (Baseline): PET_Lesions_0, Segment_18 → "femur_left_L18_PET_0"
  TP1 (Follow-Up): PET_Lesions_1, Segment_23 → "femur_left_IoU_PET_1"
  TP2 (Follow-Up): PET_Lesions_2, Segment_22 → "femur_left_PROXIMITY_PET_2"
```

Match groups are loaded from `_meta_/matches.json` in the preprocessed HDF5 file.

### Map Lesions Panel

The **Map Lesions** section at the bottom of the metadata panel allows manual cross-TP linking:

1. Click **Refresh** to load all lesion IDs from left and right time points
2. Check the corresponding lesions in the left and right columns
3. Click **Link** to add them to the same match group
4. Click **Unlink** to remove a lesion from its match group

All changes are persisted back to the HDF5 file.

---

## 9. Metadata Persistence

### Auto-Save

Every change to metadata fields triggers automatic saving:
- **JSON file**: `~/medeye3d_lesion_annotations.json`
- **HDF5 dataset**: Embedded in the preprocessed volumes file

### Stored Fields

User-entered fields are stored with their schema-defined names (e.g., `"Anatomic Location"`, `"Certainty"`). Auto-computed fields are prefixed with `_`:

| Auto-Computed Key | Description |
| :--- | :--- |
| `_Volume_mm3` | Lesion volume in mm³ |
| `_Volume_cc` | Lesion volume in cc |
| `_Diameter_mm` | Equivalent sphere diameter |
| `_MatchGroup` | Cross-TP match group ID |
| `_RECIP` | RECIP response classification |
| `_VolDelta_pct` | Volume change vs baseline (%) |
| `_VolDelta_cc` | Volume change vs baseline (cc) |
| `_SUVDelta` | SUVmax change vs baseline |
| `_SUVDelta_pct` | SUVmax change vs baseline (%) |

### Internal Fields

| Key | Description |
| :--- | :--- |
| `LesionType` | Prostate / Bone Meta / Organ Meta / Lymph Node Meta |
| `BaseAnatomy` | Auto-detected anatomy from TotalSegmentator |
| `BaseAnatomySide` | Left / Right / NA |
| `NoCTCorrelate` | true / false |
| `Anatomical Details` | Serialized relation:structure pairs |

---

## 10. Schema Definition

The metadata annotation schema is defined in `extension/data/def.json` with 20 structured questions:

| Field | Type | Options |
| :--- | :--- | :--- |
| Radioligand Type | Dropdown | 68Ga-PSMA-11, 18F-PSMA-1007, 18F-DCFPyL, Other |
| Lesion tracking name | Free text | Auto-generated |
| Anatomic Location | Dropdown | 14 standardized locations |
| Anatomical Sublocation | Dropdown | 35 specific sub-regions |
| Inner Texture / Density | Dropdown | 18 CT density descriptors |
| Border and Margin | Dropdown | 11 border characteristics |
| Lesion Shape | Dropdown | 11 shape descriptors |
| Lesion Orientation | Dropdown | 7 orientation options |
| Macroscopic Pattern | Dropdown | 7 distribution patterns |
| Bone Marrow Relation | Dropdown | 2 categories |
| Periosteal Reaction | Dropdown | 8 reaction types |
| Other Structural Changes | Dropdown | 34 associated findings |
| SUV Quantitative Metrics | Dropdown | 7 reference thresholds |
| Clinical Context | Dropdown | 19 staging variables |
| SUV max | Free text | Auto-computed |
| PRIMARY score pattern | Free text | Auto-suggested |
| PSMA-RADS 2.0 | Free text | Scoring guidance |
| Alternative Hypothesis | Dropdown | 56 false-positive categories |
| Certainty | Dropdown | 0 (uncertain) – 5 (definite) |
| Comment | Free text | Clinical notes |
