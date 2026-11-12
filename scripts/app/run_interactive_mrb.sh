#!/bin/bash
# Interactive multi-modality medical image visualization
# Run this inside the Docker container to start the viewer
set -e
export DEBIAN_FRONTEND=noninteractive

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

show_help() {
    cat << 'EOF'
Usage: ./scripts/app/run_interactive_mrb.sh [PATIENT_ID | PATH_TO_H5 | PATH_TO_DIR]

Convenient launcher for MedEye3d interactive multi-modality viewer.

Arguments:
  PATIENT_ID       Patient number or folder name (e.g., 23, Pat23, Pat44).
                   Searches in data/cases/Prostata_bimodal/
  PATH_TO_H5       Direct path to preprocessed_volumes.h5
  PATH_TO_DIR      Directory containing preprocessed_volumes.h5 (e.g., data/pat_6_files)

Defaults:
  - If no argument is provided, defaults to Pat23 (data/cases/Prostata_bimodal/Pat23/preprocessed_volumes.h5)
    or data/pat_6_files/preprocessed_volumes.h5 if Pat23 is missing.
  - Interactive viewer runs on GPU 0 (CUDA_VISIBLE_DEVICES=0).

Logging:
  All session logs are saved to:
    data/logs/viewer_<Patient>_<Timestamp>.log
  The latest log is always accessible at:
    data/logs/viewer_latest.log (and data/app_interactive.log)
EOF
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
    exit 0
fi

# Ensure X11 display authorization
[ -f "$SCRIPT_DIR/ensure_x11.sh" ] && source "$SCRIPT_DIR/ensure_x11.sh"

# Ensure clipboard support for Makie textboxes (Ctrl+C/V)
if ! command -v xclip &>/dev/null && ! command -v xsel &>/dev/null; then
    echo "Installing xclip for clipboard support..."
    sudo apt-get update -qq && sudo apt-get install -y -qq xclip 2>/dev/null || true
fi

# GPU isolation: GPU 0 for interactive viewer / local testing; GPU 1 reserved for background jobs
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HDF5_USE_FILE_LOCKING="FALSE"

# Vulkan / OpenGL drivers
if [ -f "/etc/vulkan/icd.d/nvidia_icd.json" ]; then
    export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
    export VK_DRIVER_FILES="${VK_DRIVER_FILES:-/etc/vulkan/icd.d/nvidia_icd.json}"
fi
export MESA_GL_VERSION_OVERRIDE=4.3
export MESA_GLSL_VERSION_OVERRIDE=430

# Ensure MPICH is installed (needed by HDF5_jll mpi+mpich artifact variant)
if ! dpkg -l libmpich-dev &>/dev/null 2>&1; then
    echo "Installing MPICH for HDF5_jll..."
    sudo apt-get update -qq && sudo apt-get install -y -qq libmpich-dev 2>/dev/null || true
fi

# Fix HDF5_jll RTLD_DEEPBIND issue if present
HDF5_WRAPPER=$(find /data/packages/HDF5_jll -name "x86_64-linux-gnu-*mpi+mpich.jl" 2>/dev/null | head -1)
if [ -n "$HDF5_WRAPPER" ] && ! grep -q "RTLD_GLOBAL" "$HDF5_WRAPPER" 2>/dev/null; then
    echo "  [HDF5] Patching RTLD_DEEPBIND → RTLD_DEEPBIND | RTLD_GLOBAL..."
    python3 -c "
with open('$HDF5_WRAPPER', 'r') as f:
    c = f.read()
c = c.replace('RTLD_LAZY | RTLD_DEEPBIND,', 'RTLD_LAZY | RTLD_DEEPBIND | RTLD_GLOBAL,')
with open('$HDF5_WRAPPER', 'w') as f:
    f.write(c)
" 2>/dev/null || true
    find /data/compiled -name "*HDF5*" -exec rm -rf {} + 2>/dev/null || true
    find /data/compiled -name "*MedImages*" -exec rm -rf {} + 2>/dev/null || true
    find /data/compiled -name "*MedEye3d*" -exec rm -rf {} + 2>/dev/null || true
fi

# Target dataset resolution
TARGET_ARG="$1"

if [ -z "$TARGET_ARG" ]; then
    if [ -f "data/cases/Prostata_bimodal/Pat51/preprocessed_volumes.h5" ]; then
        TARGET_ARG="Pat51"
    elif [ -f "data/cases/Prostata_bimodal/Pat23/preprocessed_volumes.h5" ]; then
        TARGET_ARG="Pat23"
    elif [ -f "data/pat_6_files/preprocessed_volumes.h5" ]; then
        TARGET_ARG="data/pat_6_files"
    fi
fi

RESOLVED_H5=""

# Case A: Plain integer like 23 -> Pat23
if [[ "$TARGET_ARG" =~ ^[0-9]+$ ]]; then
    TARGET_ARG="Pat${TARGET_ARG}"
fi

# Case B: Direct file
if [ -f "$TARGET_ARG" ]; then
    RESOLVED_H5="$TARGET_ARG"
# Case C: Directory with preprocessed_volumes.h5
elif [ -d "$TARGET_ARG" ] && [ -f "$TARGET_ARG/preprocessed_volumes.h5" ]; then
    RESOLVED_H5="$TARGET_ARG/preprocessed_volumes.h5"
# Case D: In data/cases/Prostata_bimodal/<TARGET>
elif [ -f "data/cases/Prostata_bimodal/$TARGET_ARG/preprocessed_volumes.h5" ]; then
    RESOLVED_H5="data/cases/Prostata_bimodal/$TARGET_ARG/preprocessed_volumes.h5"
elif [ -f "$REPO_ROOT/data/cases/Prostata_bimodal/$TARGET_ARG/preprocessed_volumes.h5" ]; then
    RESOLVED_H5="$REPO_ROOT/data/cases/Prostata_bimodal/$TARGET_ARG/preprocessed_volumes.h5"
# Case E: In data/<TARGET>
elif [ -f "data/$TARGET_ARG/preprocessed_volumes.h5" ]; then
    RESOLVED_H5="data/$TARGET_ARG/preprocessed_volumes.h5"
elif [ -f "$REPO_ROOT/data/$TARGET_ARG/preprocessed_volumes.h5" ]; then
    RESOLVED_H5="$REPO_ROOT/data/$TARGET_ARG/preprocessed_volumes.h5"
fi

if [ -z "$RESOLVED_H5" ] || [ ! -f "$RESOLVED_H5" ]; then
    echo "ERROR: Could not find preprocessed_volumes.h5 for input '$TARGET_ARG'."
    echo "Checked locations:"
    echo "  - $TARGET_ARG"
    echo "  - $TARGET_ARG/preprocessed_volumes.h5"
    echo "  - data/cases/Prostata_bimodal/$TARGET_ARG/preprocessed_volumes.h5"
    echo "  - data/$TARGET_ARG/preprocessed_volumes.h5"
    exit 1
fi

# Setup structured logging
LOG_DIR="$REPO_ROOT/data/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
CASE_DIR=$(dirname "$RESOLVED_H5")
CASE_NAME=$(basename "$CASE_DIR")
if [ "$CASE_NAME" = "." ] || [ "$CASE_NAME" = "/" ]; then
    CASE_NAME=$(basename "$RESOLVED_H5" .h5)
fi

SESSION_LOG="$LOG_DIR/viewer_${CASE_NAME}_${TIMESTAMP}.log"
LATEST_LOG="$LOG_DIR/viewer_latest.log"
LEGACY_LOG="$REPO_ROOT/data/app_interactive.log"

# Initialize session log with header
{
    echo "=========================================================="
    echo "  MedEye3d Viewer Session Log"
    echo "=========================================================="
    echo "  Session Start:   $(date -u +'%Y-%m-%d %H:%M:%S UTC') [$(date +'%Y-%m-%d %H:%M:%S %Z')]"
    echo "  Target Case:     $CASE_NAME"
    echo "  Dataset Path:    $RESOLVED_H5"
    echo "  GPU Device:      CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "  Display:         $DISPLAY"
    echo "  Git Commit:      $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
    echo "  Julia Version:   $(julia -e 'print(VERSION)' 2>/dev/null || echo 'unknown')"
    if command -v nvidia-smi &>/dev/null; then
        echo "  GPU Details:"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/    GPU /' || true
    fi
    echo "=========================================================="
    echo ""
} > "$SESSION_LOG"

# Symlink latest log and update legacy log
ln -sfn "$SESSION_LOG" "$LATEST_LOG"
cp -f "$SESSION_LOG" "$LEGACY_LOG"

echo ""
echo "=========================================================="
echo "  MedEye3d Interactive Multi-Modality Viewer"
echo "=========================================================="
echo "  Dataset:  $RESOLVED_H5"
echo "  GPU:      CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  Display:  $DISPLAY"
echo "  Log File: $SESSION_LOG"
echo "  Latest:   $LATEST_LOG"
echo ""
echo "  Controls & Shortcuts:"
echo "    Mouse wheel       - scroll through slices"
echo "    Right-click       - jump all orthogonal planes to clicked point"
echo "    Double-click      - zoom panel / restore 4-pane"
echo "    Left-drag         - paint on mask (in edit mode)"
echo "    Y key             - toggle Quad MRI view (T2 / ADC / DWI / T1)"
echo "    Shift+Y / Shift+C - toggle MRI comparison window"
echo "    C key             - toggle Baseline vs Follow-up compare mode"
echo "    Z key or 0        - zoom to fit"
echo "    M key             - toggle manual edit mode"
echo "    V key             - toggle move lesion mode"
echo "    Ctrl+Del          - delete current lesion (Ctrl+Z to undo)"
echo "    Ctrl+Z            - undo last action (restore deleted lesion)"
echo "    Spacebar          - toggle mask overlay visibility"
echo "    Close window      - exit"
echo "=========================================================="
echo ""

# Auto-resolve Manifest.toml if generated for a different Julia version.
CURRENT_JULIA_VER=$(julia -e 'print(VERSION)')
MANIFEST_VER=$(grep -m1 'julia_version' Manifest.toml 2>/dev/null | sed 's/.*"\(.*\)"/\1/')
if [ -n "$MANIFEST_VER" ] && [ "$CURRENT_JULIA_VER" != "$MANIFEST_VER" ]; then
    echo "  [MANIFEST] Version mismatch: Manifest=$MANIFEST_VER, Julia=$CURRENT_JULIA_VER" | tee -a "$SESSION_LOG" "$LEGACY_LOG"
    echo "  [MANIFEST] Auto-resolving (rm + instantiate)..." | tee -a "$SESSION_LOG" "$LEGACY_LOG"
    rm -f Manifest.toml
    julia --project=. -e '
        using Pkg
        devpkgs = PackageSpec[]
        for p in ["/workspaces/MedImages.jl", "/mnt/big/project_ssd/project_ssd/MedImages.jl"]
            if isdir(p)
                push!(devpkgs, PackageSpec(path=p))
                break
            end
        end
        for p in ["/workspaces/ITKIOWrapper.jl", "/mnt/big/project_ssd/project_ssd/ITKIOWrapper.jl"]
            if isdir(p)
                push!(devpkgs, PackageSpec(path=p))
                break
            end
        end
        if !isempty(devpkgs)
            Pkg.develop(devpkgs)
        end
        Pkg.instantiate()
    ' >> "$SESSION_LOG" 2>&1
    echo "  [MANIFEST] Resolved for Julia $CURRENT_JULIA_VER" | tee -a "$SESSION_LOG" "$LEGACY_LOG"
fi

# Run Julia and log all stdout and stderr
set +e
julia --project=. --threads=1 scripts/app/run_interactive_mrb.jl "$RESOLVED_H5" 2>&1 | tee -a "$SESSION_LOG" "$LEGACY_LOG"
JULIA_EXIT_CODE=${PIPESTATUS[0]}
set -e

{
    echo ""
    echo "=========================================================="
    echo "  Session Ended:   $(date -u +'%Y-%m-%d %H:%M:%S UTC') [$(date +'%Y-%m-%d %H:%M:%S %Z')]"
    echo "  Exit Status:     $JULIA_EXIT_CODE"
    echo "  Log Saved To:    $SESSION_LOG"
    echo "=========================================================="
} | tee -a "$SESSION_LOG" "$LEGACY_LOG"

exit $JULIA_EXIT_CODE
