#!/bin/bash
# =============================================================================
# MedEye3d Interactive 4-Pane Medical Viewer - Köln PSMA Dataset
# Launches the interactive viewer using the preprocessed HDF5 database.
# =============================================================================
set -e
export DEBIAN_FRONTEND=noninteractive

# Ensure clipboard support for Makie textboxes (Ctrl+C/V)
if ! command -v xclip &>/dev/null && ! command -v xsel &>/dev/null; then
    echo "Installing xclip for clipboard support..."
    sudo apt-get update -qq && sudo apt-get install -y -qq xclip 2>/dev/null || true
fi

# Graphics & rendering settings
export MESA_GL_VERSION_OVERRIDE=4.3
export MESA_GLSL_VERSION_OVERRIDE=430
if [ -f "/etc/vulkan/icd.d/nvidia_icd.json" ]; then
    export VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json"
    export VK_DRIVER_FILES="/etc/vulkan/icd.d/nvidia_icd.json"
fi

# Performance & multi-threading
export JULIA_NUM_THREADS=4,1
export HDF5_USE_FILE_LOCKING=FALSE

# Ensure we run from repository root
cd "$(dirname "$0")/../.."

# Resolve dataset path (accepts directory or .h5 file, defaults to psma_patient_all_tp)
TARGET="${1:-data/cases/psma_patient_all_tp/preprocessed_volumes.h5}"
if [ -d "$TARGET" ]; then
    H5_PATH="$TARGET/preprocessed_volumes.h5"
else
    H5_PATH="$TARGET"
fi

if [ ! -f "$H5_PATH" ]; then
    echo "ERROR: Preprocessed dataset not found: $H5_PATH"
    echo "Please ensure the dataset exists or run preprocessing first."
    exit 1
fi

H5_PATH="$(realpath "$H5_PATH")"
mkdir -p data
LOG_FILE="data/app_interactive.log"

echo ""
echo "============================================================"
echo "  MedEye3d Interactive 4-Pane Viewer (Köln PSMA Dataset)"
echo "============================================================"
echo ""
echo "  Display:    $DISPLAY"
echo "  Dataset:    $H5_PATH"
echo "  Logging to: $(pwd)/$LOG_FILE"
echo ""
echo "  Controls:"
echo "    Mouse wheel       - Scroll through slices"
echo "    Right-click       - Crosshair jump (aligns all 3 planes)"
echo "    Double-click      - Zoom panel / restore 4-pane QuadView"
echo "    Left-drag         - Paint on active mask layer"
echo "    Ctrl + Left-drag  - Erase mask voxels"
echo "    Middle-drag       - Adjust Window / Level (contrast)"
echo "    Timepoint Slider  - Switch between TP 0, 1, 2, 3"
echo "    Lesion Dropdown   - Jump camera to lesion centroid"
echo "    Close window      - Exit viewer"
echo ""
echo "============================================================"
echo ""

julia --project=. -e '
    using MedEye3d
    using MedEye3d.AppMain
    AppMain.launch_from_h5(ARGS[1])
' "$H5_PATH" 2>&1 | tee "$LOG_FILE"
