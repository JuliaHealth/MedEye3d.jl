#!/bin/bash
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y scrot xvfb x11-utils xdotool

# Start Xvfb
Xvfb :105 -screen 0 1920x1080x24 -ac &
export DISPLAY=:105
export MESA_GL_VERSION_OVERRIDE=4.3
export MESA_GLSL_VERSION_OVERRIDE=430

# Give Xvfb time to start
sleep 2

# Run the julia test script
julia --project=. /home/jm/.gemini/antigravity-cli/brain/14756b4d-1a16-43a6-8ef5-9276c73dcac6/scratch/run_screenshot_v3.jl
