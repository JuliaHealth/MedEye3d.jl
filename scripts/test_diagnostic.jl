#!/usr/bin/env julia
# Diagnostic test for MedEye3d

# Replicate the full app setup without blocking readline()
println("=== LOADING APP ===")

using MedEye3d
using MedEye3d.SegmentationDisplay
using MedEye3d.LesionMetadataWindow  
using MedEye3d.ForDisplayStructs
using MedEye3d.MakieEvents
using Statistics
using Observables

# Run app/run_interactive_mrb.jl lines 1-456 by evaluating all setup code
# We need all the same setup but skip the readline() at the end

# Include all the setup code from the real script
data_dir_pat6 = ENV["MEDEYE_DATA_DIR"]
include(joinpath(@__DIR__, "app", "run_interactive_mrb.jl"))
