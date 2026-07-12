#!/bin/bash
export DISPLAY=:99
Xvfb :99 -screen 0 1000x1000x24 &
sleep 2

echo "Precompiling..."
julia --project=. -e 'using Pkg; Pkg.precompile();'

echo "Generating data..."
julia --project=. scripts/generate_synthetic_nifti.jl

echo "Running test..."
julia --project=. scripts/test_quad_image.jl > debug9.log 2>&1 &
JULIA_PID=$!

echo "Waiting for test script to reach screenshot phase..."
for i in {1..120}; do
    if grep -q "Taking screenshot..." debug9.log 2>/dev/null; then
        echo "Found screenshot trigger!"
        break
    fi
    sleep 1
done

sleep 5
echo "Taking screenshot..."
import -window root screenshot_medeye_021.png
kill $JULIA_PID
