# Anatomic Segmentation Pipeline

## Overview

The anatomic segmentation pipeline transforms a raw CT scan into a **max-anatomy integer mask** containing up to ~201 distinct anatomical structures. It combines multiple AI models, post-processes their outputs, and composites them into a single NIfTI volume with a JSON label dictionary.

## Architecture

```
CT Volume (.nii.gz)
        │
        ├──→ TotalSegmentator (8 tasks)     ──→ ~150 binary NIfTI masks
        ├──→ SlicerDentalSegmentator         ──→ mandible.nii.gz
        ├──→ NV-Segment-CTMR (Vista3D)      ──→ rectum, celiac_trunk, pulmonary_artery
        └──→ Skellytour                       ──→ bone subsegments
                │
                ▼
        postprocess_anatomy.py
        ├── Dilate mandible → subtract from skull
        └── Split bilateral muscles → _left/_right
                │
                ▼
        build_max_anatomy.py
        ├── Coarse-to-fine layering (6 passes)
        ├── Deduplication
        └── Output: max_anatomy.nii.gz + max_anatomy_labels.json
```

## Prerequisites

### Docker Container
All AI models run inside the `medeye3d-ai` Docker container on **GPU 1**:
```bash
# Start the Docker worker
bash /mnt/big/project_ssd/project_ssd/MedEye3d.jl/scripts/ai/start_docker_worker.sh
```

### Environment Variables
Stored in `/mnt/big/project_ssd/project_ssd/MedEye3d.jl/.env`:
```
TOTALSEG_LICENSE_NUMBER=aca_XHEO7L1IH2U7G7
CUDA_VISIBLE_DEVICES=1
```

### Docker Configuration
- **GPU**: `--gpus '"device=1"'` (GPU 1 only, GPU 0 reserved for GUI)
- **Shared Memory**: `--shm-size=64g`
- **Port**: 5005 (for Julia IPC)
- **Volume Mounts**: Project dir, home dir, inference temp dir

## Pipeline Steps

### Step 1: TotalSegmentator (8 Tasks)

Runs sequentially inside Docker. Each task produces multiple binary NIfTI masks:

| Task | Structures Produced | ~Count |
|------|-------------------|--------|
| `total` | All major organs, vessels, bones | ~100 |
| `thigh_shoulder_muscles` | Deltoid, subscapularis, supraspinatus, infraspinatus, teres_major, coracobrachial, triceps_brachii, sartorius, quadriceps, thigh compartments | ~14 |
| `abdominal_muscles` | Pectoralis major, latissimus dorsi, rectus abdominis, obliques, erector spinae, psoas, quadratus lumborum, trapezius, serratus anterior (all bilateral) | ~22 |
| `headneck_muscles` | Sternocleidomastoid, platysma, levator scapulae, scalenes, pharyngeal constrictors, prevertebral, sterno-thyroid, thyrohyoid | ~20 |
| `head_muscles` | Masseter, temporalis, pterygoids, tongue, digastric | ~10 |
| `headneck_bones_vessels` | Hyoid, thyroid cartilage, cricoid, carotid arteries, jugular veins | ~10 |
| `head_glands_cavities` | Parotid, submandibular, sublingual glands, nasal/oral cavities | ~8 |
| `heartchambers_highres` | Atrial appendage, ventricle parts | ~5 |

**License**: TotalSegmentator requires a license key, auto-registered before each task via the wrapper.

**Marker Files**: Each task writes a `.ts_task_{name}_{mode}_completed` marker to avoid re-running.

### Step 2: SlicerDentalSegmentator

Extracts the mandible using nnU-Net weights:
1. Crops the CT to the skull bounding box (from TotalSegmentator's skull mask)
2. Runs nnU-Net model 111 (3d_fullres)
3. Extracts label 2 (mandible) and pads back to original geometry

### Step 3: NV-Segment-CTMR (Vista3D)

NVIDIA's foundation model for structures not in TotalSegmentator:
- `rectum` (sub-organ of colon)
- `celiac_trunk` (arterial branch)
- `pulmonary_artery`

**Important**: Requires `transformers==4.44.2` (newer versions crash).

### Step 4: Skellytour

Bone subsegmentation into cortical vs. trabecular regions.

### Step 5: Post-Processing (`postprocess_anatomy.py`)

Located at: `scripts/ai/postprocess_anatomy.py`

1. **Mandible-Skull Separation**: Morphologically dilates mandible by 3 voxels (~3mm ball structuring element), then subtracts dilated region from skull. Original non-dilated mandible is preserved as its own class.

2. **Bilateral Muscle Splitting**: For 10 muscles that TotalSegmentator outputs as single unsided masks, splits into `_left`/`_right` using connected component analysis + sagittal midline centroid classification:
   - deltoid, subscapularis, supraspinatus, infraspinatus, teres_major
   - coracobrachial, triceps_brachii, serratus_anterior, trapezius, pectoralis_minor

### Step 6: Max Anatomy Compositor (`build_max_anatomy.py`)

Composites all individual binary masks into a single integer-labeled volume using **coarse-to-fine layering** (later passes overwrite earlier ones):

| Pass | Category | Priority | Example Structures |
|------|----------|----------|-------------------|
| 1a | Coarse parent organs | Lowest | heart, colon, liver, kidneys |
| 1b | Regular organs/vessels | ↓ | stomach, spleen, aorta, trachea |
| 1c | Fine sub-organs | ↑ (overwrites parents) | rectum→colon, lung lobes, kidney cysts, celiac_trunk |
| 2 | Bones & vertebrae | ↓ | ribs, vertebrae, skull, scapula |
| 3 | Mandible | ↓ | mandible (overwrites dilated skull gap) |
| 4 | Muscles | ↓ | All 90 muscle structures |
| 5 | Remaining | ↓ | Anything not categorized |
| 6 | Skellytour subsegments | Highest | Bone internal subsegments |

**Excluded from max_anatomy** (generic envelopes):
- `body`, `body_trunc`, `body_extremities`, `skin`
- `tissue_fat`, `tissue_muscle`, `fat`, `muscle`

**Deduplication**: Structures appearing from multiple models (e.g., `tongue` from both `total` and `head_muscles` tasks) are included only once.

## Output Files

For each input CT, the pipeline produces:

```
anatomy_out/
├── max_anatomy.nii.gz          # Integer mask (uint16), ~201 classes
├── max_anatomy_labels.json     # {id: "structure_name"} mapping
├── aorta.nii.gz                # Individual binary masks...
├── brain.nii.gz
├── colon.nii.gz
├── deltoid_left.nii.gz
├── deltoid_right.nii.gz
├── heart.nii.gz
├── ... (~200 more .nii.gz files)
├── .ts_task_total_*_completed  # Marker files (prevent re-runs)
├── .nv_segment_task_completed
└── .skellytour_task_completed
```

## How to Run

### Single CT Volume (inside Docker)
```bash
docker exec medeye3d-ai python3 \
    /mnt/big/project_ssd/project_ssd/lymph_node_rules/src/anatomy_segmentation/run_segmentation.py \
    <INPUT_CT.nii.gz> \
    <OUTPUT_DIR> \
    --task all
```

### Batch: All Time Points
```bash
bash /mnt/big/project_ssd/project_ssd/MedEye3d.jl/scripts/ai/run_all_timepoints.sh
```

This iterates over all `Fixed_CT_Volume_*.nii.gz` and `SPECT_CT_Volume_*.nii.gz` files and runs the full pipeline for each.

### Manual Step-by-Step (for debugging)
```bash
CT=/mnt/big/project_ssd/project_ssd/MedEye3d.jl/data/pat_6_files/Fixed_CT_Volume_0.nii.gz
OUT=/mnt/big/project_ssd/project_ssd/MedEye3d.jl/data/pat_6_files/anatomy_out

# 1. Run all TotalSegmentator tasks
docker exec medeye3d-ai python3 -c "
from src.anatomy_segmentation.wrappers.totalsegmentator import run_totalsegmentator
for task in ['total', 'thigh_shoulder_muscles', 'abdominal_muscles', 'headneck_muscles', 'head_muscles', 'headneck_bones_vessels', 'head_glands_cavities', 'heartchambers_highres']:
    run_totalsegmentator('$CT', '$OUT', task=task)
"

# 2. Run dental segmentation
docker exec medeye3d-ai python3 -c "
from src.anatomy_segmentation.wrappers.slicer_dental import get_mandible_slicer
get_mandible_slicer('$CT', '$OUT')
"

# 3. Run NV-Segment
docker exec medeye3d-ai python3 -c "
from src.anatomy_segmentation.wrappers.nv_segment import run_nv_segmentator
run_nv_segmentator('$CT', '$OUT')
"

# 4. Run Skellytour
docker exec medeye3d-ai python3 -c "
from src.anatomy_segmentation.wrappers.skellytour import run_skellytour
run_skellytour('$CT', '$OUT')
"

# 5. Post-process (mandible dilation + muscle splitting)
docker exec medeye3d-ai python3 scripts/ai/postprocess_anatomy.py $OUT

# 6. Build max anatomy
docker exec medeye3d-ai python3 \
    /mnt/big/project_ssd/project_ssd/lymph_node_rules/src/anatomy_segmentation/build_max_anatomy.py \
    $OUT $OUT/max_anatomy.nii.gz
```

### Viewing Results
Load in 3D Slicer:
1. Open CT: `Fixed_CT_Volume_0.nii.gz`
2. Overlay: `anatomy_out/max_anatomy.nii.gz`
3. Import labels from: `anatomy_out/max_anatomy_labels.json`

## Key Files

| File | Location | Purpose |
|------|----------|---------|
| `run_segmentation.py` | `lymph_node_rules/src/anatomy_segmentation/` | Pipeline orchestrator |
| `build_max_anatomy.py` | `lymph_node_rules/src/anatomy_segmentation/` | Max anatomy compositor |
| `postprocess_anatomy.py` | `MedEye3d.jl/scripts/ai/` | Mandible dilation + muscle splitting |
| `totalsegmentator.py` | `lymph_node_rules/.../wrappers/` | TotalSegmentator wrapper |
| `nv_segment.py` | `lymph_node_rules/.../wrappers/` | Vista3D/NV-Segment wrapper |
| `slicer_dental.py` | `lymph_node_rules/.../wrappers/` | Mandible extraction wrapper |
| `skellytour.py` | `lymph_node_rules/.../wrappers/` | Bone subsegmentation wrapper |
| `start_docker_worker.sh` | `MedEye3d.jl/scripts/ai/` | Docker container launcher |
| `.env` | `MedEye3d.jl/` | License key + GPU config |

## Timing Estimates

On a single NVIDIA GPU (per CT volume):
- TotalSegmentator (8 tasks): ~15–25 minutes
- SlicerDentalSegmentator: ~3–5 minutes
- NV-Segment-CTMR: ~5–10 minutes
- Skellytour: ~2–5 minutes
- Post-processing + build: ~2 minutes
- **Total per volume: ~30–45 minutes**

## Troubleshooting

### TotalSegmentator License Error
```bash
docker exec medeye3d-ai totalseg_set_license -l aca_XHEO7L1IH2U7G7 -sv
```

### NV-Segment Crash (`all_tied_weights_keys`)
Downgrade transformers:
```bash
docker exec medeye3d-ai pip install transformers==4.44.2
```

### Out of Memory
- `oculomotor_muscles` task is excluded by default (causes OOM even with 64GB shm)
- Increase `--shm-size` in `start_docker_worker.sh` if needed

### Re-running a Failed Task
Delete the corresponding marker file:
```bash
rm anatomy_out/.ts_task_<taskname>_*_completed
```
Then re-run the pipeline — it will only re-do the missing task.
