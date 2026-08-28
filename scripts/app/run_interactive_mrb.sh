#!/bin/bash
# Interactive 4-pane medical image visualization from MRB
# Run this inside the Docker container to start the viewer
set -e
export DEBIAN_FRONTEND=noninteractive

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
JULIA_NUM_THREADS=3,1 julia --project=. scripts/run_interactive_mrb.jl 2>&1 | tee app_execution.log
