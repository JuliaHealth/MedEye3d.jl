#!/usr/bin/env julia
"""
reprocess_mri_tps.jl

Re-processes MRI timepoints (TP 3-6) in the HDF5 file after MRI registration
transform has been updated. Re-resamples CT, PET, and mask volumes using the
new transform.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using MedImages, HDF5, JSON, NIfTI, LinearAlgebra

# Load shared SceneHierarchy
include(joinpath(@__DIR__, "..", "lib", "SceneHierarchy.jl"))
using .SceneHierarchy

const DATA_DIR = joinpath(@__DIR__, "..", "..", "data", "cases", "psma_patient_all_tp")
const H5_PATH = joinpath(DATA_DIR, "preprocessed_volumes.h5")

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

function process_and_write!(h5_file, baseline_ct, nii_path, tfm_path, group, ds_name, is_mask)
    if !isfile(nii_path)
        println("  ⚠️  File not found: $nii_path")
        return
    end
    
    img = MedImages.load_image(nii_path, "CT")
    T_ITK = SceneHierarchy.parse_tfm(tfm_path)
    img_tfm = SceneHierarchy.apply_transform_to_medimage(img, T_ITK)
    
    interpolator = is_mask ? MedImages.Nearest_neighbour_en : MedImages.Linear_en
    
    if T_ITK != Matrix{Float64}(I, 4, 4) || img.spacing != baseline_ct.spacing || img.origin != baseline_ct.origin
        img_res = gpu_resample_to_image(baseline_ct, img_tfm, interpolator)
    else
        img_res = img_tfm
    end
    
    # Pre-flip (matching preprocess_dataset.jl)
    if is_mask
        vox = Float32.(img_res.voxel_data)
        vox = max.(0.0f0, vox)
        vox_flipped = reverse(Int16.(round.(vox)), dims=2)
        nz = count(v -> v > 0, vox_flipped)
        
        full_path = "$group/$ds_name"
        if haskey(h5_file, full_path)
            delete_object(h5_file, full_path)
        end
        write(h5_file, full_path, vox_flipped)
        println("  ✅ $full_path: $(size(vox_flipped)), $nz non-zero, unique=$(sort(unique(vox_flipped[vox_flipped .> 0])))")
    else
        vox_flipped = reverse(Float32.(img_res.voxel_data), dims=2)
        
        full_path = "$group/$ds_name"
        if haskey(h5_file, full_path)
            delete_object(h5_file, full_path)
        end
        write(h5_file, full_path, vox_flipped)
        println("  ✅ $full_path: $(size(vox_flipped)), range=$(minimum(vox_flipped))-$(maximum(vox_flipped))")
    end
end

function main()
    println("=== Re-processing MRI TPs with New Registration Transform ===")
    
    # Load baseline CT as reference
    baseline_nii_path = joinpath(DATA_DIR, "Fixed_CT_Volume_0.nii.gz")
    baseline_ct = MedImages.load_image(baseline_nii_path, "CT")
    println("  Baseline grid: $(size(baseline_ct.voxel_data)), spacing: $(baseline_ct.spacing)")
    
    # Transform for MRI TPs
    tfm_path = joinpath(DATA_DIR, "Transform_FollowUp_to_Baseline_3.tfm")
    println("  Transform: $tfm_path")
    
    # MRI TP files to reprocess
    # CT (T2) volume, PET (actually duplicate T2), and mask
    mri_files = [
        ("Fixed_CT_Volume_3.nii.gz", false),
        ("SUV_PET_Image_3.nii.gz", false),
        ("PET_Lesions_3.nii.gz", true),
    ]
    
    h5open(H5_PATH, "r+") do f
        group = "TFM_Transform_FollowUp_to_Baseline_3.tfm"
        
        for (fname, is_mask) in mri_files
            nii_path = joinpath(DATA_DIR, fname)
            println("\n  Processing $fname (mask=$is_mask)...")
            process_and_write!(f, baseline_ct, nii_path, tfm_path, group, fname, is_mask)
        end
        
        # Copy same resampled data for TPs 4-6 (they share the same MRI data + transform)
        for tp in 4:6
            target_group = "TFM_Transform_FollowUp_to_Baseline_$(tp).tfm"
            for (fname, _) in mri_files
                src_ds = "$group/$fname"
                tgt_ds = "$target_group/$(replace(fname, "_3." => "_$(tp)."))"
                if haskey(f, src_ds)
                    data = read(f[src_ds])
                    if haskey(f, tgt_ds)
                        delete_object(f, tgt_ds)
                    end
                    write(f, tgt_ds, data)
                    println("  ✅ Copied to $tgt_ds")
                end
            end
        end
        
        # Store the new transform in _meta_
        if haskey(f, "_meta_/transforms/Transform_FollowUp_to_Baseline_3.tfm")
            delete_object(f, "_meta_/transforms/Transform_FollowUp_to_Baseline_3.tfm")
        end
        tfm_content = read(tfm_path, String)
        write(f, "_meta_/transforms/Transform_FollowUp_to_Baseline_3.tfm", tfm_content)
        println("\n  ✅ Updated transform in _meta_")
    end
    
    println("\n[SUCCESS] MRI TPs re-processed with new registration transform!")
end

main()
