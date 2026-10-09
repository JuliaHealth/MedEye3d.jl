#!/bin/bash
xvfb-run -a julia --project=/mnt/big/project_ssd/project_ssd/MedEye3d.jl -e '
using Pkg;
Pkg.instantiate();
try
    println("Loading MedEye3d...")
    using MedEye3d
    println("SUCCESS: MedEye3d loaded without IntervalArithmetic errors.")
catch e
    println("ERROR: Failed to load MedEye3d!")
    Base.showerror(stdout, e)
    exit(1)
end
'
