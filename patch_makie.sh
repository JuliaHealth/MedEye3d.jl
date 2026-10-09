#!/bin/bash
awk '
/function reactToHeatGDTTick/ {
    in_func = 1
    while ((getline < "scratch_tmp.jl") > 0) {
        print
    }
    next
}
in_func && /^end/ {
    in_func = 0
    next
}
!in_func {
    print
}' src/display/GLFW/MakieEventHandlers.jl > src/display/GLFW/MakieEventHandlers_new.jl
mv src/display/GLFW/MakieEventHandlers_new.jl src/display/GLFW/MakieEventHandlers.jl
