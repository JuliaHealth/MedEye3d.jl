#!/bin/bash
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y scrot xvfb x11-utils xdotool

# Start Xvfb
Xvfb :103 -screen 0 1920x1080x24 -ac &
export DISPLAY=:103
export MESA_GL_VERSION_OVERRIDE=4.3
export MESA_GLSL_VERSION_OVERRIDE=430

# Give Xvfb time to start
sleep 2

# Update the environment with MedImages from GitHub
julia --project=. -e 'using Pkg; Pkg.add(url="https://github.com/JuliaHealth/MedImages.jl"); Pkg.instantiate()'

# Run the julia test script in the background
julia --project=. scripts/test_quad_image.jl &
JULIA_PID=$!

# Wait for the ready file
echo "Waiting for the display to be ready..."
# Remove any old ready file
rm -f data/ready.txt
while [ ! -f data/ready.txt ]; do
    sleep 1
done
# Sleep an extra 2 seconds to let window draw
sleep 2

# Capture the whole screen
echo "Taking screenshot..."
scrot data/screenshot_medeye.png

# Clean up
rm -f data/ready.txt

# Bring julia to foreground
wait $JULIA_PID
echo "Done."
