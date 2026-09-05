#!/usr/bin/env julia
"""
embed_prostate_anatomy.jl

Embeds the nnU-Net prostate zone segmentation into the HDF5 file as per-TP anatomy
for MRI timepoints (TP 3-6), using the same transform+resample pipeline as preprocess_dataset.jl.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using MedImages, HDF5, JSON, NIfTI, LinearAlgebra

# Load shared SceneHierarchy for parse_tfm + apply_transform_to_medimage
include(joinpath(@__DIR__, "..", "lib", "SceneHierarchy.jl"))
using .SceneHierarchy

const DATA_DIR = joinpath(@__DIR__, "..", "..", "data", "cases", "psma_patient_all_tp")
const H5_PATH = joinpath(DATA_DIR, "preprocessed_volumes.h5")
const ANAT_PATH = joinpath(DATA_DIR, "anatomy_out_fixed_ct_3", "max_anatomy.nii.gz")
const LABELS_PATH = joinpath(DATA_DIR, "anatomy_out_fixed_ct_3", "max_anatomy_labels.json")
const LESION_PATH = joinpath(DATA_DIR, "anatomy_out_fixed_ct_3", "prostate_dl_lesion.nii.gz")

function gpu_resample_to_image(im_fixed, im_moving, interp)
    try
        gpu_fixed = MedImages.to_gpu(im_fixed)
        gpu_moving = MedImages.to_gpu(im_moving)
        gpu_result = MedImages.resample_to_image(gpu_fixed, gpu_moving, interp)
        return MedImages.to_cpu(gpu_result)
    catch e
        println("    GPU resample failed ($e), using CPU...")
        return MedImages.resample_to_image(im_fixed, im_moving, interp)
    end
end

function main()
    println("=== Embedding Prostate Zone Anatomy into HDF5 ===")
    
    if !isfile(ANAT_PATH)
        error("Prostate anatomy not found: $ANAT_PATH")
    end
    
    # Load prostate anatomy (native MRI space, 512x512x28)
    anat_img = MedImages.load_image(ANAT_PATH, "CT")
    println("  Loaded anatomy: $(size(anat_img.voxel_data)), spacing: $(anat_img.spacing)")
    
    # Load baseline CT as reference
    baseline_nii_path = joinpath(DATA_DIR, "Fixed_CT_Volume_0.nii.gz")
    if !isfile(baseline_nii_path)
        error("Baseline NIfTI not found: $baseline_nii_path")
    end
    baseline_ct = MedImages.load_image(baseline_nii_path, "CT")
    println("  Baseline grid: $(size(baseline_ct.voxel_data)), spacing: $(baseline_ct.spacing)")
    
    # Load the transform for TP 3 (MRI → Baseline)
    tfm_path = joinpath(DATA_DIR, "Transform_FollowUp_to_Baseline_3.tfm")
    T_ITK = SceneHierarchy.parse_tfm(tfm_path)
    println("  Transform loaded from $(tfm_path)")
    
    # Apply transform to anatomy (updates origin, direction, spacing)
    anat_tfm = SceneHierarchy.apply_transform_to_medimage(anat_img, T_ITK)
    println("  Transformed origin: $(anat_tfm.origin), spacing: $(anat_tfm.spacing)")
    
    # Resample to baseline grid
    println("  Resampling anatomy $(size(anat_tfm.voxel_data)) → $(size(baseline_ct.voxel_data))...")
    anat_resampled = gpu_resample_to_image(baseline_ct, anat_tfm, MedImages.Nearest_neighbour_en)
    
    # Convert to UInt16 and pre-flip
    anat_vox = UInt16.(round.(max.(0.0f0, Float32.(anat_resampled.voxel_data))))
    anat_flipped = reverse(anat_vox, dims=2)
    
    unique_labels = sort(unique(anat_flipped))
    nz = count(x -> x > 0, anat_flipped)
    println("  Resampled anatomy: $(size(anat_flipped)), unique: $unique_labels, non-zero: $nz")
    
    if nz == 0
        error("Resampled anatomy is all zeros! Check transform and spatial alignment.")
    end
    
    h5open(H5_PATH, "r+") do f
        # Write to MRI TP groups (TP 3-6)
        for tp in 3:6
            grp = "TFM_Transform_FollowUp_to_Baseline_$(tp).tfm"
            ds_name = "$grp/max_anatomy.nii.gz"
            
            if haskey(f, ds_name)
                delete_object(f, ds_name)
            end
            
            write(f, ds_name, anat_flipped)
            println("  ✅ Written $ds_name ($(size(anat_flipped)), $nz non-zero)")
        end
        
        # Embed prostate anatomy labels
        if isfile(LABELS_PATH)
            labels_json = read(LABELS_PATH, String)
            ds_meta = "_meta_/prostate_anatomy_labels.json"
            if haskey(f, ds_meta)
                delete_object(f, ds_meta)
            end
            write(f, ds_meta, labels_json)
            println("  ✅ Embedded prostate anatomy labels")
        end
        
        # Process DL lesion mask with same transform
        if isfile(LESION_PATH)
            println("\n  Processing DL lesion mask...")
            lesion_img = MedImages.load_image(LESION_PATH, "CT")
            lesion_tfm = SceneHierarchy.apply_transform_to_medimage(lesion_img, T_ITK)
            lesion_resampled = gpu_resample_to_image(baseline_ct, lesion_tfm, MedImages.Nearest_neighbour_en)
            lesion_vox = Int16.(round.(max.(0.0f0, Float32.(lesion_resampled.voxel_data))))
            lesion_flipped = reverse(lesion_vox, dims=2)
            nz_les = count(x -> x > 0, lesion_flipped)
            
            for tp in 3:6
                grp = "TFM_Transform_FollowUp_to_Baseline_$(tp).tfm"
                ds_name = "$grp/prostate_dl_lesion.nii.gz"
                if haskey(f, ds_name)
                    delete_object(f, ds_name)
                end
                write(f, ds_name, lesion_flipped)
                println("  ✅ Written $ds_name ($nz_les non-zero voxels)")
            end
        end
    end
    
    println("\n[SUCCESS] Prostate anatomy embedded into HDF5 for MRI TPs 3-6!")
end

main()
