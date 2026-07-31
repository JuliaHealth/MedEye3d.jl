#!/bin/bash
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y scrot xvfb x11-utils xdotool

# Start Xvfb
Xvfb :106 -screen 0 1920x1080x24 -ac &
export DISPLAY=:106
export MESA_GL_VERSION_OVERRIDE=4.3
export MESA_GLSL_VERSION_OVERRIDE=430

# Give Xvfb time to start
sleep 2

# Run the julia test script
julia --project=. scripts/run_screenshot_v3.jl
