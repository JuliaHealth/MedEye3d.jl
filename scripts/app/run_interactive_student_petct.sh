#!/bin/bash
# ==========================================================
#  MedEye3d Interactive PET/CT Student Dataset Viewer
# ==========================================================
# Opens the student practice dataset:
#   TP0 (BASELINE): PET/CT with BLANK mask (for annotation practice)
#   TP1 (Follow-up): PET/CT with EXPERT mask (reference)
#
# Usage:
#   bash scripts/app/run_interactive_student_petct.sh
# ==========================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Student dataset path
DATASET="${PROJECT_ROOT}/data/cases/Prostata_bimodal/Pat23_PETCT_student/preprocessed_volumes.h5"
PATIENT_ID="Pat23_PETCT_student"

# GPU selection (default: GPU 0)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Display
export DISPLAY="${DISPLAY:-:0}"

# Log file
LOG_DIR="${PROJECT_ROOT}/data/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/viewer_${PATIENT_ID}_${TIMESTAMP}.log"
LATEST_LOG="${LOG_DIR}/viewer_latest.log"

echo ""
echo "=========================================================="
echo "  MedEye3d Student PET/CT Practice Dataset"
echo "=========================================================="
echo "  Dataset:  ${DATASET}"
echo "  GPU:      CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  Display:  ${DISPLAY}"
echo "  Log File: ${LOG_FILE}"
echo "  Latest:   ${LATEST_LOG}"
echo ""
echo "  TP0 (BASELINE): BLANK mask - annotate lesions here"
echo "  TP1 (Reference): EXPERT annotations - for self-check"
echo ""
echo "  Controls & Shortcuts:"
echo "    Mouse wheel       - scroll through slices"
echo "    Right-click       - jump all orthogonal planes to clicked point"
echo "    Double-click      - zoom panel / restore 4-pane"
echo "    Left-drag         - paint on mask (in edit mode)"
echo "    C key             - toggle Baseline vs Follow-up compare mode"
echo "    Z key or 0        - zoom to fit"
echo "    M key             - toggle manual edit mode"
echo "    E key             - create new lesion (enter edit mode)"
echo "    G key             - grow lesion via heat-GDT"
echo "    Ctrl+Del          - delete current lesion (Ctrl+Z to undo)"
echo "    Ctrl+Z            - undo last action"
echo "    Spacebar          - toggle mask overlay visibility"
echo "    Close window      - exit"
echo "=========================================================="
echo ""

cd "${PROJECT_ROOT}"

# Use the same Julia entry point as run_interactive_mrb.sh
julia --project=. --threads=1 scripts/app/run_interactive_mrb.jl "${DATASET}" 2>&1 | tee -a "${LOG_FILE}"

# Update latest log symlink
ln -sf "${LOG_FILE}" "${LATEST_LOG}"
