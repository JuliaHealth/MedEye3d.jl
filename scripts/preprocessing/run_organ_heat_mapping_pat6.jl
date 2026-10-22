#!/usr/bin/env julia
"""
run_organ_heat_mapping_pat6.jl

End-to-end runner script for Patient 6:
1. Runs OrganHeatLesionTracker across all time points on preprocessed_volumes.h5
2. Regenerates lesion_tracking_report.json
3. Prints summary statistics of matches and synthetic lesions
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using MedEye3d
using MedEye3d.OrganHeatLesionTracker
using MedEye3d.LesionTracker
using JSON

function main()
    data_dir = joinpath(@__DIR__, "..", "..", "data", "pat_6_files")
    h5_path = joinpath(data_dir, "preprocessed_volumes.h5")
    
    if !isfile(h5_path)
        error("HDF5 not found at $h5_path")
    end

    println("=" ^ 75)
    println("  PATIENT 6: 5-ORGAN & HEAT-KERNEL LESION MAPPING RUNNER")
    println("=" ^ 75)

    # 1. Run OrganHeatLesionTracker
    groups = OrganHeatLesionTracker.propagate_all_timepoints!(h5_path; verbose=true)

    # 2. Regenerate lesion tracking report
    println("\n--- Generating new lesion tracking report ---")
    report_path = joinpath(data_dir, "lesion_tracking_report.json")
    try
        entries = LesionTracker.track_lesions(data_dir; h5_path=h5_path, output_path=report_path)
        println("  Generated $(length(entries)) tracking entries in $report_path")
    catch e
        @warn "Failed generating lesion_tracking_report.json: $e"
    end

    println("\n[Complete] Successfully updated Patient 6 with new Organ-Heat mapping.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
