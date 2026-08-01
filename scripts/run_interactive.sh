#!/bin/bash
# Interactive 4-pane medical image visualization
# Run this inside the Docker container to start the viewer
set -e
export DEBIAN_FRONTEND=noninteractive

# Use Mesa software renderer if no GPU driver available
export MESA_GL_VERSION_OVERRIDE=4.3
export MESA_GLSL_VERSION_OVERRIDE=430

echo ""
echo "============================================"
echo "  MedEye3d Interactive 4-Pane Viewer"
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

cd /workspaces/MedEye3d.jl
julia --project=. scripts/run_interactive_quad.jl

