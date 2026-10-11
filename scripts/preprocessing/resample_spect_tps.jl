#!/usr/bin/env julia
"""
resample_spect_tps.jl

Resamples the 4 SPECT timepoints (TP 0, 1, 2, 4) onto the baseline CT coordinate space
using their pre-saved ITK affine transforms and saves them into the repaired HDF5 file.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using MedImages
using HDF5
using LinearAlgebra

include(joinpath(@__DIR__, "..", "lib", "SceneHierarchy.jl"))
using .SceneHierarchy

const DATA_DIR = joinpath(@__DIR__, "..", "..", "data", "pat_6_files")
const H5_PATH = joinpath(DATA_DIR, "preprocessed_volumes_repaired.h5")

function main()
    println("=== Resampling SPECT Timepoints to Baseline Coordinate Grid ===")
    
    baseline_path = joinpath(DATA_DIR, "Fixed_CT_Volume_0.nii.gz")
    println("Loading Baseline CT: $baseline_path")
    baseline_ct = MedImages.load_image(baseline_path, "CT")
    println("Baseline grid: size=$(size(baseline_ct.voxel_data)), spacing=$(baseline_ct.spacing)")
    
    spect_tps = [0, 1, 2, 4]
    
    h5_file = h5open(H5_PATH, "r+")
    try
        for tp in spect_tps
            tfm_name = "Transform_SPECT_to_Baseline_$(tp).tfm"
            tfm_path = joinpath(DATA_DIR, tfm_name)
            group_name = "TFM_" * tfm_name
            println("\n------------------------------------------------------------")
            println("Processing SPECT TP $tp -> Group: $group_name")
            println("Transform: $tfm_path")
            
            T_ITK = SceneHierarchy.parse_tfm(tfm_path)
            println("ITK Transform Matrix:\n", T_ITK)
            
            # 1. SPECT CT
            ct_fname = "SPECT_CT_Volume_$(tp).nii.gz"
            ct_path = joinpath(DATA_DIR, ct_fname)
            if isfile(ct_path)
                println("  [1/4] Resampling $ct_fname...")
                t_start = time()
                img = MedImages.load_image(ct_path, "CT")
                img_tfm = SceneHierarchy.apply_transform_to_medimage(img, T_ITK)
                img_res = MedImages.resample_to_image(baseline_ct, img_tfm, MedImages.Linear_en)
                img_flip = MedImages.update_voxel_data(img_res, reverse(Float32.(img_res.voxel_data), dims=2))
                MedImages.save_med_image(h5_file, group_name, ct_fname, img_flip; compress=3)
                println("    Saved $group_name/$ct_fname in $(round(time()-t_start, digits=1))s")
            else
                @warn "SPECT CT not found: $ct_path"
            end
            
            # 2. SPECT NM
            nm_fname = "SPECT_NM_Vendor_Volume_$(tp).nii.gz"
            nm_path = joinpath(DATA_DIR, nm_fname)
            if isfile(nm_path)
                println("  [2/4] Resampling $nm_fname...")
                t_start = time()
                img = MedImages.load_image(nm_path, "NM")
                img_tfm = SceneHierarchy.apply_transform_to_medimage(img, T_ITK)
                img_res = MedImages.resample_to_image(baseline_ct, img_tfm, MedImages.Linear_en)
                # Scale values if needed, Float32
                vox = Float32.(img_res.voxel_data)
                img_flip = MedImages.update_voxel_data(img_res, reverse(vox, dims=2))
                MedImages.save_med_image(h5_file, group_name, nm_fname, img_flip; compress=3)
                println("    Saved $group_name/$nm_fname in $(round(time()-t_start, digits=1))s")
            else
                @warn "SPECT NM not found: $nm_path"
            end
            
            # 3. SPECT Lesions
            mask_fname = "SPECT_Lesions_$(tp).nii.gz"
            mask_path = joinpath(DATA_DIR, mask_fname)
            if isfile(mask_path)
                println("  [3/4] Resampling $mask_fname...")
                t_start = time()
                img = MedImages.load_image(mask_path, "CT")
                img_tfm = SceneHierarchy.apply_transform_to_medimage(img, T_ITK)
                img_res = MedImages.resample_to_image(baseline_ct, img_tfm, MedImages.Nearest_neighbour_en)
                vox = Float32.(img_res.voxel_data)
                vox = max.(0.0f0, vox)
                vox_i16 = Int16.(round.(vox))
                nz = count(v -> v > 0, vox_i16)
                img_flip = MedImages.update_voxel_data(img_res, reverse(vox_i16, dims=2))
                MedImages.save_med_image(h5_file, group_name, mask_fname, img_flip; compress=3)
                println("    Saved $group_name/$mask_fname ($nz non-zero voxels) in $(round(time()-t_start, digits=1))s")
            else
                @warn "SPECT Lesions not found: $mask_path"
            end
            
            # 4. max_anatomy
            anat_path = joinpath(DATA_DIR, "anatomy_out_spect_ct_$(tp)", "max_anatomy.nii.gz")
            if isfile(anat_path)
                println("  [4/4] Resampling max_anatomy.nii.gz...")
                t_start = time()
                img = MedImages.load_image(anat_path, "CT")
                img_tfm = SceneHierarchy.apply_transform_to_medimage(img, T_ITK)
                img_res = MedImages.resample_to_image(baseline_ct, img_tfm, MedImages.Nearest_neighbour_en)
                vox = Float32.(img_res.voxel_data)
                vox = max.(0.0f0, vox)
                vox_u16 = UInt16.(round.(vox))
                img_flip = MedImages.update_voxel_data(img_res, reverse(vox_u16, dims=2))
                MedImages.save_med_image(h5_file, group_name, "max_anatomy.nii.gz", img_flip; compress=3)
                println("    Saved $group_name/max_anatomy.nii.gz in $(round(time()-t_start, digits=1))s")
            else
                @warn "max_anatomy not found: $anat_path"
            end
        end
    finally
        close(h5_file)
    end
    
    println("\n[SUCCESS] All SPECT timepoints resampled and saved into $H5_PATH!")
end

main()
