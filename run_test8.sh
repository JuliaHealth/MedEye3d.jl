#!/bin/bash
export DISPLAY=:99
Xvfb :99 -screen 0 1000x1000x24 &
sleep 2
julia --project=. scripts/test_quad_image.jl > debug8.log 2>&1 &
JULIA_PID=$!
sleep 180
import -window root screenshot_medeye_020.png
kill $JULIA_PID
