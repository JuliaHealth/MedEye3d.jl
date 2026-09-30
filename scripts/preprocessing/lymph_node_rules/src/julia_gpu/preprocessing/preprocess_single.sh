#!/usr/bin/env bash
# ==============================================================================
# preprocess_single.sh
# Converts a single patient's folder of NIfTI segmentations and CT scan
# into native MedImages HDF5 format with computed landmarks.
#
# Usage:
#   ./preprocess_single.sh <case_dir> [--clear-cache]
# ==============================================================================
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <case_dir> [--clear-cache] [--output <path>]"
    echo "Example: $0 data/processed_cases_restored/FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat44 --clear-cache"
    exit 1
fi

CASE_DIR="$1"
shift 1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JULIA_PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=============================================================================="
echo "RUNNING MEDIMAGES PREPROCESSING (Single Patient)"
echo "Case Dir:      $CASE_DIR"
echo "Julia Project: $JULIA_PROJECT_DIR"
echo "=============================================================================="

julia --project="$JULIA_PROJECT_DIR" "$SCRIPT_DIR/preprocess_single.jl" "$CASE_DIR" "$@"
