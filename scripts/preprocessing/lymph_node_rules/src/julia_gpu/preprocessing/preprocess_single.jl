#!/usr/bin/env julia
# ==============================================================================
# preprocess_single.jl
# Converts a single patient's folder of NIfTI segmentations and CT scan
# into the native MedImages HDF5 format (primary_masks.h5) with computed landmarks.
# ==============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Printf
using HDF5

include("MedImagesIO.jl")
include("Landmarks.jl")
include("PrimaryHDF5.jl")
include("coordinator.jl")

using .MedImagesIO
using .Landmarks
using .PrimaryHDF5
using .Coordinator

function print_usage()
    println("""
Usage:
  julia --project=src/julia_gpu src/julia_gpu/preprocessing/preprocess_single.jl <case_dir> [options]

Arguments:
  case_dir               Path to patient case directory containing Fixed_CT_Volume.nii.gz and segmentations/

Options:
  --output, -o <path>    Output HDF5 path (default: <case_dir>/primary_masks.h5)
  --clear-cache, -f      Force re-computation even if primary_masks.h5 exists
  --help, -h             Show this help message
""")
end

function main(args)
    if isempty(args) || "--help" in args || "-h" in args
        print_usage()
        exit(0)
    end

    case_dir = args[1]
    if !isdir(case_dir)
        @error "Case directory not found: $case_dir"
        exit(1)
    end

    force_clear = "--clear-cache" in args || "-f" in args || "--force" in args
    
    out_h5 = joinpath(case_dir, "primary_masks.h5")
    for i in 1:length(args)-1
        if (args[i] == "--output" || args[i] == "-o") && i+1 <= length(args)
            out_h5 = args[i+1]
        end
    end

    println("==============================================================================")
    println("MEDIMAGES PREPROCESSING (Single Patient)")
    println("Case Directory: $case_dir")
    println("Target HDF5:    $out_h5")
    println("Clear Cache:    $force_clear")
    println("==============================================================================")

    if force_clear && isfile(out_h5)
        println("  [Cache] Removing existing HDF5 file: $out_h5")
        rm(out_h5, force=true)
    end

    if !force_clear && isfile(out_h5)
        println("  [Cache] Verified existing HDF5 container: $out_h5. Use --clear-cache to force rebuild.")
        exit(0)
    end

    try
        PrimaryHDF5.build_primary_hdf5(case_dir, out_h5; force=force_clear, verbose=true)
        println("==============================================================================")
        println("Successfully generated MedImages container: $out_h5")
        println("==============================================================================")
    catch e
        @error "MedImages preprocessing failed: $e"
        Base.show_backtrace(stderr, catch_backtrace())
        exit(1)
    end
end

main(ARGS)
