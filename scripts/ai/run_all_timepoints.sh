#!/bin/bash
# run_all_timepoints.sh — Full anatomic segmentation pipeline for ALL CT time points
# Runs inside the MedEye3d devcontainer (sharp_ramanujan).
# Produces complete NIfTI segmentations for lymph node rules pipeline and max_anatomy.
#
# Usage: bash scripts/ai/run_all_timepoints.sh [patient_dir]
#
# Prerequisites (inside container):
#   - TotalSegmentator v2.18+, MuscleMap, Skellytour, nnUNetv2, transformers
#   - /mnt/big/project_ssd/project_ssd mounted (see .devcontainer/devcontainer.json)
#
# For cases in lymph_node_rules/data/processed_cases_restored/:
#   bash scripts/ai/run_all_timepoints.sh /mnt/big/project_ssd/project_ssd/lymph_node_rules/data/processed_cases_restored/FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat6

set -uo pipefail

PATIENT_DIR="${1:-/mnt/big/project_ssd/project_ssd/MedEye3d.jl/data/pat_6_files}"
CONTAINER="sharp_ramanujan"
LYMPH_DIR="/mnt/big/project_ssd/project_ssd/lymph_node_rules"
POSTPROCESS_SCRIPT="/mnt/big/project_ssd/project_ssd/MedEye3d.jl/scripts/ai/postprocess_anatomy.py"
BUILD_SCRIPT="$LYMPH_DIR/src/anatomy_segmentation/build_max_anatomy.py"
SYNTHESIZE_SCRIPT="$LYMPH_DIR/src/julia_gpu/preprocessing/synthesize_missing_segmentations.py"

# All 14 TotalSegmentator tasks for complete anatomy coverage
# IMPORTANT: 'total' MUST be LAST because heartchambers_highres also
# outputs aorta.nii.gz (truncated, heart-only) which would overwrite
# the full aorta from 'total' if 'total' ran first.
TS_TASKS=(
    "thigh_shoulder_muscles"
    "abdominal_muscles"
    "headneck_muscles"
    "head_muscles"
    "headneck_bones_vessels"
    "head_glands_cavities"
    "heartchambers_highres"
    "lung_vessels"
    "renal_arteries"
    "tissue_types"
    "vertebrae_body"
    "trunk_cavities"
    "body"
    "liver_segments"
    "liver_vessels"
    "total"  # MUST be last — overwrites heartchambers_highres aorta
)

echo "=========================================="
echo "  Anatomy Segmentation — Full Pipeline"
echo "  Container: $CONTAINER"
echo "  $(date -Iseconds)"
echo "=========================================="
echo "Patient directory: $PATIENT_DIR"
echo ""

# Check Docker container is running
if ! docker ps -q -f name="$CONTAINER" | grep -q .; then
    echo "ERROR: Docker container '$CONTAINER' is not running."
    echo "Open MedEye3d.jl in VS Code to start the devcontainer."
    exit 1
fi

# Check if /mnt/big is accessible in the container
if ! docker exec "$CONTAINER" test -d "/mnt/big/project_ssd/project_ssd" 2>/dev/null; then
    echo "WARNING: /mnt/big/project_ssd/project_ssd not mounted in container."
    echo "Rebuild devcontainer or add the mount manually."
    echo "Falling back to host execution..."
    USE_DOCKER=false
else
    USE_DOCKER=true
fi

# Helper function to run commands (in docker or on host)
run_cmd() {
    if [ "$USE_DOCKER" = true ]; then
        docker exec "$CONTAINER" "$@"
    else
        "$@"
    fi
}

# Collect all CT volumes to process
CT_FILES=()
for f in "$PATIENT_DIR"/Fixed_CT_Volume_*.nii.gz; do
    [ -f "$f" ] && CT_FILES+=("$f")
done
for f in "$PATIENT_DIR"/SPECT_CT_Volume_*.nii.gz; do
    [ -f "$f" ] && CT_FILES+=("$f")
done
# Also check for bare Fixed_CT_Volume.nii.gz (lymph_node_rules processed cases)
if [ -f "$PATIENT_DIR/Fixed_CT_Volume.nii.gz" ] && [ ${#CT_FILES[@]} -eq 0 ]; then
    CT_FILES+=("$PATIENT_DIR/Fixed_CT_Volume.nii.gz")
fi

echo "Found ${#CT_FILES[@]} CT volumes to process:"
for f in "${CT_FILES[@]}"; do echo "  - $(basename "$f")"; done
echo ""

TOTAL=${#CT_FILES[@]}
CURRENT=0
FAILED=0
SKIPPED=0

for CT_FILE in "${CT_FILES[@]}"; do
    CURRENT=$((CURRENT + 1))
    BASENAME=$(basename "$CT_FILE" .nii.gz)

    # Determine output directory
    if [[ "$BASENAME" == "Fixed_CT_Volume" ]]; then
        OUT_DIR="$PATIENT_DIR/segmentations"
    elif [[ "$BASENAME" == Fixed_CT_Volume_* ]]; then
        IDX="${BASENAME#Fixed_CT_Volume_}"
        OUT_DIR="$PATIENT_DIR/anatomy_out_fixed_ct_${IDX}"
    elif [[ "$BASENAME" == SPECT_CT_Volume_* ]]; then
        IDX="${BASENAME#SPECT_CT_Volume_}"
        OUT_DIR="$PATIENT_DIR/anatomy_out_spect_ct_${IDX}"
    else
        OUT_SUFFIX=$(echo "$BASENAME" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
        OUT_DIR="$PATIENT_DIR/anatomy_out_${OUT_SUFFIX}"
    fi

    LOG_FILE="$OUT_DIR/segmentation.log"

    echo "=========================================="
    echo "[$CURRENT/$TOTAL] Processing: $BASENAME"
    echo "  Input:  $CT_FILE"
    echo "  Output: $OUT_DIR"
    echo "=========================================="

    mkdir -p "$OUT_DIR"
    echo "$(date -Iseconds) Starting segmentation for $BASENAME" > "$LOG_FILE"

    # ===============================================================
    # Step 1: TotalSegmentator (14 tasks — primary for ~82 structures)
    # ===============================================================
    echo "  [1/7] TotalSegmentator (${#TS_TASKS[@]} tasks)..."
    for TASK in "${TS_TASKS[@]}"; do
        MARKER="$OUT_DIR/.ts_task_${TASK}_default_completed"
        if [ -f "$MARKER" ]; then
            echo "    TS '$TASK' ✓ (cached)"
            continue
        fi
        echo "    Running TS task: $TASK..."
        if run_cmd TotalSegmentator -i "$CT_FILE" -o "$OUT_DIR" --task "$TASK" 2>&1 | tee -a "$LOG_FILE"; then
            echo "completed" > "$MARKER"
            echo "    ✅ $TASK done."
        else
            echo "    ⚠️  $TASK had issues." | tee -a "$LOG_FILE"
        fi
    done

    # ===============================================================
    # Step 2: SlicerDentalSegmentator (mandible — critical for neck LN)
    # ===============================================================
    echo "  [2/7] SlicerDentalSegmentator..."
    if [ -f "$OUT_DIR/mandible.nii.gz" ]; then
        echo "    mandible.nii.gz ✓ (cached)"
    else
        run_cmd python3 -c "
import sys; sys.path.insert(0, '$LYMPH_DIR')
from src.anatomy_segmentation.wrappers.slicer_dental import get_mandible_slicer
get_mandible_slicer('$CT_FILE', '$OUT_DIR')
" 2>&1 | tee -a "$LOG_FILE" || echo "    ⚠️  Dental had issues." | tee -a "$LOG_FILE"
    fi

    # ===============================================================
    # Step 3: MuscleMap (58 muscles — critical for neck/pelvis LN)
    # ===============================================================
    echo "  [3/7] MuscleMap..."
    if [ -f "$OUT_DIR/.musclemap_task_completed" ]; then
        echo "    MuscleMap ✓ (cached)"
    else
        if run_cmd mm_segment -i "$CT_FILE" -o "$OUT_DIR" 2>&1 | tee -a "$LOG_FILE"; then
            echo "completed" > "$OUT_DIR/.musclemap_task_completed"
            echo "    ✅ MuscleMap done."
        else
            echo "    ⚠️  MuscleMap had issues." | tee -a "$LOG_FILE"
        fi
    fi

    # ===============================================================
    # Step 4: NV-Segment-CTMR (140 structures — 2nd tier model)
    # ===============================================================
    echo "  [4/7] NV-Segment-CTMR..."
    if [ -f "$OUT_DIR/.nv_segment_task_completed" ]; then
        echo "    NV-Segment ✓ (cached)"
    else
        run_cmd python3 -c "
import sys; sys.path.insert(0, '$LYMPH_DIR')
from src.anatomy_segmentation.wrappers.nv_segment import run_nv_segmentator
run_nv_segmentator('$CT_FILE', '$OUT_DIR')
" 2>&1 | tee -a "$LOG_FILE" || echo "    ⚠️  NV-Segment had issues." | tee -a "$LOG_FILE"
    fi

    # ===============================================================
    # Step 5: Skellytour (bone subsegmentation)
    # ===============================================================
    echo "  [5/7] Skellytour..."
    if [ -f "$OUT_DIR/.skellytour_task_completed" ]; then
        echo "    Skellytour ✓ (cached)"
    else
        run_cmd python3 -c "
import sys; sys.path.insert(0, '$LYMPH_DIR')
from src.anatomy_segmentation.wrappers.skellytour import run_skellytour
run_skellytour('$CT_FILE', '$OUT_DIR')
" 2>&1 | tee -a "$LOG_FILE" || echo "    ⚠️  Skellytour had issues." | tee -a "$LOG_FILE"
    fi

    # ===============================================================
    # Step 6: Post-processing (mandible/skull cleanup, bilateral splits)
    # ===============================================================
    echo "  [6/7] Post-processing..."
    run_cmd python3 "$POSTPROCESS_SCRIPT" "$OUT_DIR" 2>&1 | tee -a "$LOG_FILE"

    # ===============================================================
    # Step 7: Synthesize missing segmentations (iliac split, etc.)
    # ===============================================================
    echo "  [7/7] Synthesize missing segmentations..."
    if [ -f "$SYNTHESIZE_SCRIPT" ]; then
        # For lymph_node_rules cases, segmentations are in a subdir
        if [ -d "$OUT_DIR/../segmentations" ] && [ "$OUT_DIR" != "$PATIENT_DIR/segmentations" ]; then
            # MedEye3d anatomy_out_* case: segmentations are in the OUT_DIR directly
            CASE_LIKE_DIR=$(mktemp -d)
            ln -sf "$OUT_DIR" "$CASE_LIKE_DIR/segmentations"
            ln -sf "$CT_FILE" "$CASE_LIKE_DIR/Fixed_CT_Volume.nii.gz"
            run_cmd python3 "$SYNTHESIZE_SCRIPT" "$CASE_LIKE_DIR" 2>&1 | tee -a "$LOG_FILE"
            rm -rf "$CASE_LIKE_DIR"
        else
            run_cmd python3 "$SYNTHESIZE_SCRIPT" "$PATIENT_DIR" 2>&1 | tee -a "$LOG_FILE"
        fi
    fi

    # Build max anatomy
    echo "  Building max_anatomy..."
    MAX_ANAT="$OUT_DIR/max_anatomy.nii.gz"
    if [ -f "$BUILD_SCRIPT" ]; then
        run_cmd python3 "$BUILD_SCRIPT" "$OUT_DIR" "$MAX_ANAT" 2>&1 | tee -a "$LOG_FILE"
    fi

    # Verify
    NIFTI_COUNT=$(ls "$OUT_DIR"/*.nii.gz 2>/dev/null | wc -l)
    echo "  📊 $NIFTI_COUNT NIfTI files generated."
    if [ -f "$MAX_ANAT" ]; then
        echo "  ✅ max_anatomy.nii.gz created."
    else
        echo "  ⚠️  max_anatomy.nii.gz not created (build_max_anatomy may need config)."
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

echo "=========================================="
echo "  Batch Complete! $(date -Iseconds)"
echo "  Total: $TOTAL | Processed: $((TOTAL - SKIPPED)) | Skipped: $SKIPPED | Failed: $FAILED"
echo "=========================================="
