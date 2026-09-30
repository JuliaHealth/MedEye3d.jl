#!/usr/bin/env julia
# ==============================================================================
# run_gpu_single.jl
# Executes pure Julia KernelAbstractions GPU rule engine on preprocessed case data,
# generating all support structures and lymph node areas, and packaging a per-patient MRB.
# ==============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Printf
using JSON
using CUDA
using KernelAbstractions
using HDF5
using NIfTI
using Statistics

include("../preprocessing/MedImagesIO.jl")
include("../preprocessing/Landmarks.jl")
include("../preprocessing/PrimaryHDF5.jl")
include("../preprocessing/coordinator.jl")
include("Evaluator.jl")

# GPU Kernel modules
include("gpu_engine_mega/jfa.jl")
include("gpu_engine_mega/hull_planes.jl")
include("gpu_engine_mega/patch_internal_iliac.jl")
include("gpu_engine_mega/patch_external_iliac.jl")
include("gpu_engine_mega/patch_common_iliac.jl")
include("gpu_engine_mega/patch_iliac_bifurcation.jl")
include("gpu_engine_mega/patch_station1.jl")
include("gpu_engine_mega/patch_presacral_anterior.jl")
include("gpu_engine_mega/patch_pleural_space.jl")
include("gpu_engine_mega/mega_fused_dag_kernel_v2.jl")
include("tensor_manager.jl")
include("engine_ka.jl")

include("MaskPacker.jl")
include("StaticArena.jl")
include("GpuCCL.jl")

include("RuleExecutors.jl")
using .RuleExecutors
include("kernels_custom_rules.jl")
using .CustomRules
include("batch_executor.jl")
include("DagVm.jl")
using .DagVm

using .MedImagesIO
using .Landmarks
using .PrimaryHDF5
using .Evaluator
using .Coordinator

function print_usage()
    println("""
Usage:
  julia --project=src/julia_gpu src/julia_gpu/main/run_gpu_single.jl <case_dir> [options]

Arguments:
  case_dir               Path to patient case directory containing Fixed_CT_Volume.nii.gz and primary_masks.h5

Options:
  --output_dir, -o <dir> Output directory for generated segmentations and MRB
  --clear-cache, -f      Clear cached intermediate files and force full recomputation
  --help, -h             Show this help message
""")
end

function run_dag_execution(case_dir::String, h5_path::String; output_dir::String="")
    println("="^85)
    println("Pure Julia KernelAbstractions GPU Execution")
    println("Case: $(basename(case_dir))")
    println("Primary HDF5: $h5_path")
    println("="^85)
    
    # 1. Compile DAG Rules
    project_root = normpath(joinpath(@__DIR__, "../../.."))
    json_dir = joinpath(project_root, "jsons")
    all_rules = Dict{String, Any}()
    for f in readdir(json_dir)
        full_p = joinpath(json_dir, f)
        if isfile(full_p) && endswith(f, ".json")
            d = JSON.parsefile(full_p)
            merge!(all_rules, d)
        end
    end
    
    # 2. Execute GPU DAG
    ct_path = joinpath(case_dir, "Fixed_CT_Volume.nii.gz")
    out_dir = isempty(output_dir) ? joinpath(case_dir, "lymph_node_outputs") : output_dir
    mkpath(out_dir)
    
    println("Executing GPU DAG across $(length(all_rules)) rules...")
    gen_masks = DagVm.run_pipeline(h5_path; resolve_overlaps=true)
    
    # 3. Save NIfTI outputs
    
    h5_out = joinpath(dirname(h5_path), "..", "..", "final_results_gpu.h5") # or just "final_results_gpu.h5" in current dir
    h5_out_local = "final_results_gpu.h5"
    println("\nSaving $(length(gen_masks)) raw masks to $h5_out_local...")
    h5open(h5_out_local, "w") do f
        for (k, v) in gen_masks
            try
                arr = collect(v)
                if eltype(arr) == Bool; arr = UInt8.(arr)
                elseif eltype(arr) != UInt8; arr = UInt8.(arr .> 0); end
                if ndims(arr) == 3
                    f[k] = arr
                end
            catch e
                println("ERROR saving $k: $e")
            end
        end
    end

    println("\nSaving $(length(gen_masks)) generated segmentations to $out_dir...")
    for (mname, arr) in gen_masks
        out_p = joinpath(out_dir, "$(mname).nii.gz")
        arr_u8 = eltype(arr) == Bool ? UInt8.(arr) : UInt8.(arr .> 0); try MedImagesIO.save_nifti_mask(arr_u8, ct_path, out_p); catch e; println("NIfTI save failed: ", e); end
    end
    
    println("==============================================================================")
    println("GPU Execution Completed Successfully! Outputs saved to: $out_dir")
    println("==============================================================================")
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
    out_dir = ""
    for i in 1:length(args)-1
        if (args[i] == "--output_dir" || args[i] == "-o") && i+1 <= length(args)
            out_dir = args[i+1]
        end
    end

    Coordinator.coordinate_case_pipeline(case_dir; 
                                         force_step1=force_clear, 
                                         force_step2=force_clear, 
                                         run_dag_fn=(cdir, h5) -> run_dag_execution(cdir, h5; output_dir=out_dir))
end

main(ARGS)
