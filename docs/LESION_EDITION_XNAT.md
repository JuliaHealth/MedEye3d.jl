# Lesion-Editing HDF5 & XNAT Persistence (`lesion_edits.h5`)

## Overview

MedEye3d captures rich, per-lesion interactive segmentation history (Heat-GDT kernel growth, connected-component elongation across multiple clicks, post-hoc threshold/dilation/smoothing adjustments, and manual brush painting/erasing) in an **independent HDF5 file** alongside the main `preprocessed_volumes.h5`.

Keeping this data out of `preprocessed_volumes.h5` prevents bloating the primary viewer dataset and allows fine-tuning datasets to be stored and synced independently.

---

## Local File Naming & Memory Design

- When loading `preprocessed_volumes.h5`, the recorder writes to `<case_dir>/lesion_edits.h5`.
- When loading `<Subject>_preprocessed.h5` (e.g. from `~/.medeye3d/xnat_cache/`), it writes to `<case_dir>/<Subject>_preprocessed_lesion_edits.h5`.
- **Memory-cautious buffering**:
  - In Julia RAM, only `64×64×64` `UInt8` patches (`256 KiB` each) are held for active/dirty lesions.
  - Full `/CT/volume` and `/PET/volume` datasets are written **lazily once** on the first flush using `chunk=(64, 64, 64)` and `deflate=4` compression, without duplicating full 3D arrays in RAM.
  - Flushed event buffers are cleared from RAM immediately after writing to disk.

---

## HDF5 Schema (`lesion_edits.h5`)

```text
lesion_edits.h5
├── /_meta_/
│   └── attrs: case_id, source_h5, spacing_mm, origin_mm, ct_size, pet_size
├── /CT/
│   └── volume                      (full CT volume, chunk=(64,64,64), deflate=4)
├── /PET/
│   └── volume                      (full PET volume, chunk=(64,64,64), deflate=4)
└── /lesions/
    └── lesion_<id>_tp<tp>_c<x>_<y>_<z>_<seq>/
        ├── attrs:
        │   ├── lesion_id, tp_index
        │   ├── seed_vox            (initial Heat-GDT click voxel)
        │   ├── last_seed_vox       (latest connected Heat-GDT click voxel)
        │   ├── chunk_origin_vox    (1-based [x0, y0, z0] of patch corner in full CT)
        │   ├── chunk_size          ([64, 64, 64])
        │   ├── chunk_center_vox    (center voxel [cx, cy, cz] in full CT)
        │   └── chunk_center_mm     (physical center [wx, wy, wz] in CT world mm)
        ├── initial_heat_mask       (64×64×64 UInt8: initial Heat-GDT output)
        ├── final_mask              (64×64×64 UInt8: latest mask after all edits)
        ├── manual_painted_mask     (64×64×64 UInt8: cumulative manually painted voxels)
        ├── manual_erased_mask      (64×64×64 UInt8: cumulative manually erased voxels)
        ├── subpatches/             (created when connected elongation exceeds 64×64×64)
        │   └── subpatch_<k>/
        │       ├── attrs: chunk_origin_vox, chunk_center_vox, chunk_center_mm
        │       ├── final_mask, manual_painted_mask, manual_erased_mask
        └── events/                 (ordered edit history for fine-tuning)
            ├── 0001_heat_gdt       (attrs: event_type, timestamp, seed_vox, theta, K, dilation, smoothing, edge_modality, subpatch_idx)
            ├── 0002_heat_gdt       (connected continuation click on elongated lesion)
            ├── 0003_manual_paint   (attrs: brush_radius, subpatch_idx; 64×64×64 delta)
            ├── 0004_manual_erase   (attrs: brush_radius, subpatch_idx; 64×64×64 delta)
            └── 0005_posthoc_size   (attrs: theta, dilation, smoothing)
```

---

## Connected-Component Lesion Grouping

When a user triggers Heat-GDT on an existing `lesion_id`, `LesionEditsRecorder.are_voxels_connected_26` checks 26-connectivity in a bounded local bounding box around the previous and new seeds:
- **Same connected component** (e.g. an elongated lesion grown across multiple nearby clicks): edits append to the **same** lesion group, automatically adding an overflow `64×64×64` `subpatch_<k>` if the new click falls outside the primary `64×64×64` chunk.
- **Disconnected component**: a new separate lesion group is created.

---

## XNAT Upload & Download Integration

When saving back to XNAT via **`Save XNAT`** (`_do_save_to_xnat` in `LesionMetadataWindow.jl`):
1. Pending masks and `LesionEditsRecorder` patches are flushed locally (`lesion_edits.h5`).
2. `XnatClient.save_annotations_to_xnat` uploads `status.json` (and annotations) to `resources/ANNOTATIONS`.
3. `XnatClient.upload_preprocessed_hdf5` uploads the main HDF5 to `resources/MEDEYE3D/files/{Subject}_preprocessed.h5` (with experiment-level `MEDEYE3D` fallback).
4. `XnatClient.upload_lesion_edits_hdf5` uploads the lesion-editing HDF5 (when present) to `resources/MEDEYE3D/files/{Subject}_lesion_edits.h5` (with experiment-level `MEDEYE3D` fallback).

When downloading a patient from XNAT:
- `XnatClient.download_preprocessed_hdf5` explicitly filters **out** `*lesion_edits*.h5` so it only fetches the main preprocessed volume.
- `XnatClient.download_lesion_edits_hdf5` fetches `*lesion_edits*.h5` from `MEDEYE3D` into the sibling local path alongside the cached main HDF5 so subsequent edits continue appending to the existing history.
