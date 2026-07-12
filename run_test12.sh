#!/bin/bash
export DISPLAY=:99
Xvfb :99 -screen 0 1000x1000x24 &
sleep 2
echo "Running test..."
julia --project=. scripts/test_quad_image.jl
