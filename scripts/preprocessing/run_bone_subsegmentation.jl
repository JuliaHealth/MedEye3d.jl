#!/usr/bin/env julia
# Script to run KernelAbstractions-based morphological bone surface and marrow subsegmentation

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using MedEye3d
using MedImages
using MedEye3d.BoneSubsegmentation
using NIfTI

function process_patient_dataset(data_dir::String, out_dir::String)
    mkpath(out_dir)
    
    ct_path = joinpath(data_dir, "Fixed_CT_Volume_0.nii.gz")
    pet_path = joinpath(data_dir, "PET_Lesions_0.nii.gz")
    ts_path = joinpath(data_dir, "TS_all_Segmentation_0.seg.nrrd")
    
    if !isfile(ct_path) || !isfile(pet_path)
        println("Dataset files not found in $data_dir")
        return
    end
    
    println("Loading CT and Lesion mask...")
    ct = MedImages.load_image(ct_path, "CT")
    mask = MedImages.load_image(pet_path, "CT")
    
    mask_vol = Float32.(mask.voxel_data)
    bone_vol = zeros(Float32, size(mask_vol))
    
    # Check if TS segmentation is available
    if isfile(ts_path)
        println("Loading TotalSegmentator atlas...")
        ts_atlas, _ = MedEye3d.LesionAssociation.load_nrrd_labelmap(ts_path)
        if ts_atlas !== nothing
            # 26: bone/rib/vertebra labels
            bone_vol = Float32.(ts_atlas .> 0)
        end
    end
    
    spacing = Tuple(Float64.(ct.spacing))
    unique_lesions = filter(x -> x > 0, sort(unique(mask_vol)))
    
    println("Found $(length(unique_lesions)) lesions to process.")
    
    for lid in unique_lesions
        lid_int = Int(lid)
        println("Processing Lesion $lid_int with KernelAbstractions...")
        surf, marr = generate_bone_subsegments(mask_vol, bone_vol, spacing, lid_int)
        
        surf_count = count(surf)
        marr_count = count(marr)
        println("  -> Lesion $lid_int: Bone Surface = $surf_count voxels, Bone Marrow = $marr_count voxels")
    end
    println("Preprocessing finished successfully!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    data_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "..", "data", "pat_6_files")
    out_dir = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "..", "..", "data", "pat_6_files", "subseg_out")
    process_patient_dataset(data_dir, out_dir)
end
