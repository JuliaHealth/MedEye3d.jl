#!/bin/bash
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y scrot xvfb x11-utils xdotool imagemagick

# Kill any existing Xvfb on :106
kill $(cat /tmp/.X106-lock 2>/dev/null) 2>/dev/null || true
rm -f /tmp/.X106-lock /tmp/.X11-unix/X106 2>/dev/null || true
sleep 1

# Start Xvfb
Xvfb :106 -screen 0 1920x1920x24 -ac &
export DISPLAY=:106
export MESA_GL_VERSION_OVERRIDE=4.3
export MESA_GLSL_VERSION_OVERRIDE=430

# Give Xvfb time to start
sleep 2

# Run the julia test script
julia --project=. scripts/run_goal_screenshot.jl
