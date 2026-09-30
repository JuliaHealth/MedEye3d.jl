module Coordinator

using Printf
using JSON

include(joinpath(@__DIR__, "Landmarks.jl"))
include(joinpath(@__DIR__, "PrimaryHDF5.jl"))
using .Landmarks
using .PrimaryHDF5

export coordinate_case_pipeline, run_step1_inference, run_step2_packaging, is_step1_complete

const ESSENTIAL_MASKS = [
    "body", "trachea", "aorta", "esophagus", "mandible", "hyoid", 
    "sternum", "clavicula_left", "clavicula_right", "scapula_left", 
    "scapula_right", "iliac_artery_left", "iliac_artery_right", 
    "psoas_major_left", "psoas_major_right"
]

"""
    is_step1_complete(case_dir::String) -> Bool
Checks if the essential base segmentation masks exist in the case's segmentations directory or primary_masks.h5.
"""
function is_step1_complete(case_dir::String)::Bool
    if isfile(joinpath(case_dir, "primary_masks.h5"))
        return true
    end
    for sdir in ["segmentations_fast_backup", "segmentations"]
        seg_dir = joinpath(case_dir, sdir)
        if isdir(seg_dir)
            all_found = true
            for mask in ESSENTIAL_MASKS
                f1 = joinpath(seg_dir, "$mask.nii.gz")
                f2 = joinpath(seg_dir, "$(mask)_primary_totalsegmentator.nii.gz")
                f3 = joinpath(seg_dir, "$(mask)_fallback_2_slicerdentalsegmentator.nii.gz")
                if !isfile(f1) && !isfile(f2) && !isfile(f3)
                    all_found = false
                    break
                end
            end
            if all_found
                return true
            end
        end
    end
    return false
end

"""
    run_step1_inference(case_dir::String; force::Bool=false)
Runs multi-model inference (TotalSegmentator v2, MuscleMap, NV-Segment, SlicerDentalSegmentator)
if masks are missing or if force=true.
"""
function run_step1_inference(case_dir::String; force::Bool=false)
    case_id = basename(case_dir)
    println("="^80)
    println("Step 1 Preprocessing: Multi-Model Inference Cache")
    println("Case: $case_id")
    println("="^80)

    if !force && is_step1_complete(case_dir)
        println("  [Step 1 Cache] Verified. Essential masks already generated. Skipping inference.")
        return true
    end

    println("  [Step 1 Cache] Invalidation triggered (force=$force). Executing inference models...")
    python_script = joinpath(@__DIR__, "../../../old/src_v2/trigger_segmentation.py")
    cmd = `python3 $python_script $case_id`
    
    try
        run(cmd)
        println("  [Step 1 Cache] Inference successfully completed.")
        return true
    catch e
        @error "Failed to execute Step 1 inference script: $e"
        return false
    end
end

"""
    run_step2_packaging(case_dir::String; force::Bool=false) -> String
Builds the primary HDF5 container (`primary_masks.h5`) containing all base masks and computed landmarks.
"""
function run_step2_packaging(case_dir::String; force::Bool=false)::String
    h5_path = joinpath(case_dir, "primary_masks.h5")
    println("="^80)
    println("Step 2 Preprocessing: Primary HDF5 Packaging & Landmarks")
    println("Target: $h5_path")
    println("="^80)

    if !force && isfile(h5_path)
        println("  [Step 2 Cache] Verified primary HDF5 container exists. Skipping re-assembly.")
        return h5_path
    end

    println("  [Step 2 Cache] Assembling primary HDF5 and computing landmarks in Julia...")
    @time PrimaryHDF5.build_primary_hdf5(case_dir, h5_path; force=force, verbose=true)
    return h5_path
end

"""
    coordinate_case_pipeline(case_dir::String; 
                             force_step1::Bool=false, 
                             force_step2::Bool=false, 
                             run_dag_fn::Union{Nothing, Function}=nothing)

Main coordination entry point managing Step 1 inference, Step 2 HDF5 assembly, and Step 3 DAG execution.
"""
function coordinate_case_pipeline(case_dir::String; 
                                  force_step1::Bool=false, 
                                  force_step2::Bool=false, 
                                  run_dag_fn::Union{Nothing, Function}=nothing)
    println("\n" * "="^85)
    println("COORDINATING PREPROCESSING & INFERENCE PIPELINE")
    println("Case Directory: $case_dir")
    println("Flags: force_step1=$force_step1, force_step2=$force_step2")
    println("="^85)

    # 1. Step 1 Preprocessing
    run_step1_inference(case_dir; force=force_step1)

    # 2. Step 2 Preprocessing
    # Note: If Step 1 was forced to refresh, Step 2 should also refresh
    rebuild_step2 = force_step2 || force_step1
    h5_path = run_step2_packaging(case_dir; force=rebuild_step2)

    # 3. Step 3 Execution
    if run_dag_fn !== nothing
        println("\n" * "="^85)
        println("Step 3: Executing Downstream Pure Julia GPU DAG...")
        println("="^85)
        run_dag_fn(case_dir, h5_path)
    end

    return h5_path
end

end # module Coordinator
