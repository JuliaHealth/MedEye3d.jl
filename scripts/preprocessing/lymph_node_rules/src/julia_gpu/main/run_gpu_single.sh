#!/usr/bin/env bash
# ==============================================================================
# run_gpu_single.sh
# Runs GPU DAG rule execution on a single patient case to produce lymph node masks and MRB.
#
# Usage:
#   ./run_gpu_single.sh <case_dir> [--clear-cache] [--output_dir <path>]
# ==============================================================================
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <case_dir> [--clear-cache] [--output_dir <path>]"
    echo "Example: $0 data/processed_cases_restored/FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat44 --clear-cache"
    exit 1
fi

CASE_DIR="$1"
shift 1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JULIA_PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=============================================================================="
echo "RUNNING JULIA GPU PIPELINE (Single Patient)"
echo "Case Dir:      $CASE_DIR"
echo "Julia Project: $JULIA_PROJECT_DIR"
echo "=============================================================================="

julia --project="$JULIA_PROJECT_DIR" "$SCRIPT_DIR/run_gpu_single.jl" "$CASE_DIR" "$@"
