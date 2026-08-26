# QuadView 4-Panel Layout & 3D Spatial Navigation

MedEye3d provides a synchronized multi-panel diagnostic interface designed for nuclear medicine and oncology reading.

---

## 1. Quad View Layout Structure

The main OpenGL display renders four synchronized panels simultaneously:

```
┌───────────────────────────────────┬───────────────────────────────────┐
│  Panel 1: Axial Fused CT + PET    │  Panel 2: Axial Pure PET          │
│  - Primary CT background          │  - Invertible colormap            │
│  - Alpha-blended PET overlay      │  - Quantitative SUV evaluation    │
│  - Segmentation & manual masks    │  - Independent windowing          │
├───────────────────────────────────┼───────────────────────────────────┤
│  Panel 3: Sagittal Resampled      │  Panel 4: Coronal Resampled       │
│  - Orthogonal Sagittal slice      │  - Orthogonal Coronal slice       │
│  - Isotropic voxel spacing        │  - Isotropic voxel spacing        │
│  - Crosshair slice intersection   │  - Crosshair slice intersection   │
└───────────────────────────────────┴───────────────────────────────────┘
```

---

## 2. Interactive Navigation Features

### Cross-Plane 3D Slice Jumping
- **Right-Click**: Clicking on any point in any panel computes the 3D voxel coordinate $(X, Y, Z)$ and immediately jumps the orthogonal panels (Sagittal and Coronal) to the intersecting anatomical slice.

### Double-Click Panel Maximization
- **Double-Click**: Double-clicking any quad panel maximizes it to fill the entire window for detailed inspection.
- Double-clicking again restores the 4-panel split layout.

### Data-Level Zoom & Panning
- **Shift + Scroll**: Continuously zooms into the active panel up to 20x.
- **Right-Click Drag**: Pans across the zoomed field of view with full coordinate translation.

---

## 3. Side-by-Side Comparison Mode (Panel 5)

Clicking the **Compare Volumes** button in the Makie GUI activates a 2-panel comparative layout:
- **Left Panel (Panel 1)**: Baseline / Reference time point.
- **Right Panel (Panel 5)**: Follow-up / Response evaluation time point.
- Slice scrolling is automatically synchronized across both volumes.

### Synchronized Lesion Navigation

When navigating to a lesion in Compare Mode:
1. The left panel scrolls to the lesion's centroid at the current time point
2. The right panel automatically finds the **matched lesion** in the comparison time point via `LesionAssociation.find_cross_tp_lesion()`
3. Both panels isolate and highlight the corresponding lesions from the same **match group**

### Quantitative Tracking

The metadata panel automatically displays longitudinal changes when a matched lesion is selected:
- **Volume delta**: Percentage and absolute change in lesion volume (cc)
- **SUVmax delta**: Change in maximum SUV compared to baseline
- **RECIP classification**: Complete Response (CR), Partial Response (PR), Stable Disease (SD), or Progressive Disease (PD)

See [Lesion Metadata & Tracking](lesion_metadata_and_tracking.md) for full details on match analysis.

