# Makie GUI Controls, Windowing & Layer Visibility

MedEye3d provides a GLMakie control panel window (`src/display/LesionMetadataWindow.jl`) synchronized via the core event channel.

---

## 1. Windowing & Intensity Controls

Interactive windowing controls allow real-time adjustment of contrast and display thresholds:

### Dual-Handle IntervalSliders
- **CT Window Range**: Minimum and maximum Hounsfield Units (HU).
- **PET Window Range**: Minimum and maximum SUV thresholds.
- Synchronized two-way with numerical text boxes and +/- step buttons.

### Preset Buttons
- **CT Presets**:
  - `Bone`: [-1000, 1500] HU
  - `Soft Tissue`: [-150, 250] HU
  - `Lung`: [-1000, -200] HU
- **PET Presets**:
  - `SUV 0-5`
  - `SUV 0-10`
  - `SUV 0-20`

---

## 2. Mask Layer Visibility Toggles

Individual buttons allow granular toggling of mask layers without re-uploading textures:
- **`Show Bone Surface`**: Toggles the cortical bone surface mesh.
- **`Show Bone Marrow`**: Toggles the trabecular bone marrow volume.
- **`Show Mask`**: Toggles AI and manual lesion segmentations.
- **`Show Lesion`**: Toggles the active target lesion highlight.

Under the hood, these buttons emit `ShowMaskLayerEvent` which updates OpenGL shader uniforms (`Uniforms.setTextureVisibility`) directly on the GPU in $\mathcal{O}(1)$ time.

---

## 3. Manual Painting & AI Semiauto Controls

- **Brush Radius Slider**: Adjusts stroke radius from 1 to 20 pixels.
- **Paint Value Slider**: Controls mask label ID (e.g. `0` for erase, `1` for lesion, `2` for organ).
- **AI Algorithm Menu**: Selects between `HELPNet (AI)` and `NNInteractive`.
- **Run Semiauto AI**: Dispatches background inference request for the active lesion scribbles.

---

## 4. Lesion Metadata Panel & Clinical Annotation

The metadata panel provides comprehensive clinical annotation with:

- **Searchable Dropdowns**: Type-to-filter in any dropdown for rapid annotation
- **Lesion Type Buttons**: Prostate / Bone Meta / Organ Meta / Lymph Node with dynamic field visibility
- **Anatomical Auto-Fill**: Automatic anatomy detection from TotalSegmentator atlas
- **No CT Correlate Toggle**: Hides CT morphology fields for PET-only lesions
- **PROMISE Score**: Auto-computed SUV comparison vs liver, parotid, and blood pool
- **Volume & Match Analysis**: Automatic volume computation, cross-TP delta tracking, and RECIP response classification

For full details, see [Lesion Metadata & Tracking](lesion_metadata_and_tracking.md).

