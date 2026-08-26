#!/bin/bash
# run_all_timepoints.sh
# Runs the full anatomic segmentation pipeline on ALL CT time points.
# Calls each tool directly (not via run_segmentation.py) to avoid import issues.
#
# Usage: bash scripts/ai/run_all_timepoints.sh [patient_dir]

set -uo pipefail

PATIENT_DIR="${1:-/mnt/big/project_ssd/project_ssd/MedEye3d.jl/data/pat_6_files}"
CONTAINER="medeye3d-ai"
POSTPROCESS_SCRIPT="/mnt/big/project_ssd/project_ssd/MedEye3d.jl/scripts/ai/postprocess_anatomy.py"
BUILD_SCRIPT="/mnt/big/project_ssd/project_ssd/lymph_node_rules/src/anatomy_segmentation/build_max_anatomy.py"
LYMPH_DIR="/mnt/big/project_ssd/project_ssd/lymph_node_rules"

# TotalSegmentator tasks to run
TS_TASKS=(
    "total"
    "thigh_shoulder_muscles"
    "abdominal_muscles"
    "headneck_muscles"
    "head_muscles"
    "headneck_bones_vessels"
    "head_glands_cavities"
    "heartchambers_highres"
)

echo "========================================"
echo "  Anatomy Segmentation Batch Runner"
echo "========================================"
echo "Patient directory: $PATIENT_DIR"
echo ""

# Check Docker container is running
if ! docker ps -q -f name="$CONTAINER" | grep -q .; then
    echo "ERROR: Docker container '$CONTAINER' is not running."
    echo "Start it with: bash scripts/ai/start_docker_worker.sh"
    exit 1
fi

# Collect all CT volumes to process
CT_FILES=()
for f in "$PATIENT_DIR"/Fixed_CT_Volume_*.nii.gz; do
    [ -f "$f" ] && CT_FILES+=("$f")
done
for f in "$PATIENT_DIR"/SPECT_CT_Volume_*.nii.gz; do
    [ -f "$f" ] && CT_FILES+=("$f")
done

echo "Found ${#CT_FILES[@]} CT volumes to process:"
for f in "${CT_FILES[@]}"; do
    echo "  - $(basename $f)"
done
echo ""

TOTAL=${#CT_FILES[@]}
CURRENT=0
FAILED=0
SKIPPED=0

for CT_FILE in "${CT_FILES[@]}"; do
    CURRENT=$((CURRENT + 1))
    BASENAME=$(basename "$CT_FILE" .nii.gz)
    
    # Determine output directory suffix
    if [[ "$BASENAME" == Fixed_CT_Volume_* ]]; then
        IDX="${BASENAME#Fixed_CT_Volume_}"
        OUT_SUFFIX="fixed_ct_${IDX}"
    elif [[ "$BASENAME" == SPECT_CT_Volume_* ]]; then
        IDX="${BASENAME#SPECT_CT_Volume_}"
        OUT_SUFFIX="spect_ct_${IDX}"
    else
        OUT_SUFFIX=$(echo "$BASENAME" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
    fi
    
    OUT_DIR="$PATIENT_DIR/anatomy_out_${OUT_SUFFIX}"
    MAX_ANAT="$OUT_DIR/max_anatomy.nii.gz"
    LOG_FILE="$OUT_DIR/segmentation.log"
    
    echo "========================================"
    echo "[$CURRENT/$TOTAL] Processing: $BASENAME"
    echo "  Input:  $CT_FILE"
    echo "  Output: $OUT_DIR"
    echo "========================================"
    
    # Skip if max_anatomy already exists
    if [ -f "$MAX_ANAT" ]; then
        echo "  ✅ max_anatomy.nii.gz already exists, skipping."
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    
    mkdir -p "$OUT_DIR"
    echo "$(date -Iseconds) Starting segmentation for $BASENAME" > "$LOG_FILE"
    
    # === Step 1: TotalSegmentator (8 tasks) ===
    echo "  [1/5] TotalSegmentator..."
    for TASK in "${TS_TASKS[@]}"; do
        MARKER="$OUT_DIR/.ts_task_${TASK}_default_completed"
        if [ -f "$MARKER" ]; then
            echo "    TS task '$TASK' already done, skipping."
            continue
        fi
        echo "    Running TS task: $TASK..."
        if docker exec "$CONTAINER" TotalSegmentator \
            -i "$CT_FILE" -o "$OUT_DIR" --task "$TASK" 2>&1 | tee -a "$LOG_FILE"; then
            echo "completed" > "$MARKER"
            echo "    ✅ $TASK done."
        else
            echo "    ⚠️  $TASK had issues (check log)."
        fi
    done
    
    # === Step 2: SlicerDentalSegmentator ===
    echo "  [2/5] SlicerDentalSegmentator..."
    if [ -f "$OUT_DIR/mandible.nii.gz" ]; then
        echo "    mandible.nii.gz already exists, skipping."
    else
        docker exec "$CONTAINER" python3 -c "
import sys
sys.path.insert(0, '$LYMPH_DIR')
from src.anatomy_segmentation.wrappers.slicer_dental import get_mandible_slicer
get_mandible_slicer('$CT_FILE', '$OUT_DIR')
" 2>&1 | tee -a "$LOG_FILE" || echo "    ⚠️  Dental segmentation had issues."
    fi
    
    # === Step 3: NV-Segment-CTMR ===
    echo "  [3/5] NV-Segment-CTMR..."
    if [ -f "$OUT_DIR/.nv_segment_task_completed" ]; then
        echo "    NV-Segment already done, skipping."
    else
        docker exec "$CONTAINER" python3 -c "
import sys
sys.path.insert(0, '$LYMPH_DIR')
from src.anatomy_segmentation.wrappers.nv_segment import run_nv_segmentator
run_nv_segmentator('$CT_FILE', '$OUT_DIR')
" 2>&1 | tee -a "$LOG_FILE" || echo "    ⚠️  NV-Segment had issues."
    fi
    
    # === Step 4: Skellytour ===
    echo "  [4/5] Skellytour..."
    if [ -f "$OUT_DIR/.skellytour_task_completed" ]; then
        echo "    Skellytour already done, skipping."
    else
        docker exec "$CONTAINER" python3 -c "
import sys
sys.path.insert(0, '$LYMPH_DIR')
from src.anatomy_segmentation.wrappers.skellytour import run_skellytour
run_skellytour('$CT_FILE', '$OUT_DIR')
" 2>&1 | tee -a "$LOG_FILE" || echo "    ⚠️  Skellytour had issues."
    fi
    
    # === Step 5: Post-process + Build Max Anatomy ===
    echo "  [5/5] Post-processing + building max_anatomy..."
    docker exec "$CONTAINER" python3 "$POSTPROCESS_SCRIPT" "$OUT_DIR" 2>&1 | tee -a "$LOG_FILE"
    docker exec "$CONTAINER" python3 "$BUILD_SCRIPT" "$OUT_DIR" "$MAX_ANAT" 2>&1 | tee -a "$LOG_FILE"
    
    # Verify
    if [ -f "$MAX_ANAT" ]; then
        NIFTI_COUNT=$(ls "$OUT_DIR"/*.nii.gz 2>/dev/null | wc -l)
        echo "  ✅ Done: $NIFTI_COUNT NIfTI files, max_anatomy built."
    else
        echo "  ❌ max_anatomy.nii.gz was NOT created!"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

echo "========================================"
echo "  Batch Complete!"
echo "  Total: $TOTAL | Processed: $((TOTAL - SKIPPED)) | Skipped: $SKIPPED | Failed: $FAILED"
echo "========================================"
