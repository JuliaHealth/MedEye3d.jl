#!/bin/bash
export DEBIAN_FRONTEND=noninteractive
# Start Xvfb
Xvfb :106 -screen 0 1920x1080x24 -ac &
export DISPLAY=:106
export MESA_GL_VERSION_OVERRIDE=4.3
export MESA_GLSL_VERSION_OVERRIDE=430

sleep 2

# Run the julia test script
julia --project=. scripts/test_gui_mrb.jl &
JULIA_PID=$!

sleep 90
scrot -z /workspaces/MedEye3d.jl/mrb_test_gui.png

kill $JULIA_PID
