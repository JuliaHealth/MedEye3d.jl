#!/bin/bash
# Interactive 4-pane medical image visualization from MRB
# Run this inside the Docker container to start the viewer
set -e
export DEBIAN_FRONTEND=noninteractive

# Ensure clipboard support for Makie textboxes (Ctrl+C/V)
if ! command -v xclip &>/dev/null && ! command -v xsel &>/dev/null; then
    echo "Installing xclip for clipboard support..."
    sudo apt-get update -qq && sudo apt-get install -y -qq xclip 2>/dev/null || true
fi

# Use Mesa software renderer if no GPU driver available
export MESA_GL_VERSION_OVERRIDE=4.3
export MESA_GLSL_VERSION_OVERRIDE=430



echo ""
echo "============================================"
echo "  MedEye3d Interactive 4-Pane Viewer (MRB)"
echo "============================================"
echo ""
echo "  Display: $DISPLAY"
echo "  Controls:"
echo "    Mouse wheel  - scroll through slices"
echo "    Right-click  - jump all planes to clicked point"
echo "    Double-click - zoom panel / restore 4-pane"
echo "    Left-drag    - paint on mask"
echo "    Close window - exit"
echo ""
echo "============================================"
echo ""

cd "$(dirname "$0")/../.."
LOG_FILE="data/app_interactive.log"
echo "  Logging to: $(pwd)/$LOG_FILE"
echo ""

# Auto-resolve Manifest.toml if generated for a different Julia version.
# This handles shared bind-mount between host and Docker with different Julia patch versions.
CURRENT_JULIA_VER=$(julia -e 'print(VERSION)')
MANIFEST_VER=$(grep -m1 'julia_version' Manifest.toml 2>/dev/null | sed 's/.*"\(.*\)"/\1/')
if [ -n "$MANIFEST_VER" ] && [ "$CURRENT_JULIA_VER" != "$MANIFEST_VER" ]; then
    echo "  [MANIFEST] Version mismatch: Manifest=$MANIFEST_VER, Julia=$CURRENT_JULIA_VER"
    echo "  [MANIFEST] Auto-resolving (rm + instantiate)..."
    rm -f Manifest.toml
    julia --project=. -e '
        using Pkg
        # Re-add local dev dependency if MedImages.jl is available
        for p in ["/workspaces/MedImages.jl", "/mnt/big/project_ssd/project_ssd/MedImages.jl"]
            if isdir(p)
                Pkg.develop(path=p)
                break
            end
        end
        Pkg.instantiate()
    '
    echo "  [MANIFEST] Resolved for Julia $CURRENT_JULIA_VER"
fi

JULIA_NUM_THREADS=3,1 julia --project=. scripts/app/run_interactive_mrb.jl "$@" 2>&1 | tee "$LOG_FILE"
