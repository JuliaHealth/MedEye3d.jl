#!/usr/bin/env bash
# ==============================================================================
# run_gpu_all.sh
# Batch executes the Julia GPU pipeline across all cases in a folder.
#
# Usage:
#   ./run_gpu_all.sh <cases_base_dir> [--clear-cache]
# ==============================================================================
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <cases_base_dir> [--clear-cache]"
    echo "Example: $0 data/processed_cases_restored --clear-cache"
    exit 1
fi

CASES_BASE_DIR="$1"
shift 1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================================================="
echo "BATCH JULIA GPU PIPELINE EXECUTION"
echo "Cases Base Dir: $CASES_BASE_DIR"
echo "=============================================================================="

CASE_DIRS=$(find "$CASES_BASE_DIR" -maxdepth 1 -mindepth 1 -type d | sort)

COUNT=0
for case_dir in $CASE_DIRS; do
    if [ ! -f "$case_dir/Fixed_CT_Volume.nii.gz" ] && [ ! -d "$case_dir/segmentations" ]; then
        continue
    fi
    COUNT=$((COUNT + 1))
    echo "------------------------------------------------------------------------------"
    echo "[$COUNT] Processing GPU Case: $(basename "$case_dir")"
    echo "------------------------------------------------------------------------------"
    "$SCRIPT_DIR/run_gpu_single.sh" "$case_dir" "$@"
done

echo "=============================================================================="
echo "Batch GPU processing complete. Total cases processed: $COUNT"
echo "=============================================================================="
