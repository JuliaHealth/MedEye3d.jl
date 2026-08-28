using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using MedImages
using NIfTI
using HDF5
using LinearAlgebra
using Statistics
using JSON
using CUDA

include(joinpath(@__DIR__, "..", "..", "src", "ai", "AIInference.jl"))
using .AIInference
include(joinpath(@__DIR__, "..", "lib", "SceneHierarchy.jl"))
using .SceneHierarchy

"""
GPU-accelerated resample_to_image: moves voxel data to GPU, resamples, moves back to CPU.
"""
function gpu_resample_to_image(im_fixed::MedImages.MedImage, im_moving::MedImages.MedImage, interp::MedImages.Interpolator_enum)
    # Move both images' voxel data to GPU
    gpu_fixed = MedImages.update_voxel_data(im_fixed, CuArray(Float32.(im_fixed.voxel_data)))
    gpu_moving = MedImages.update_voxel_data(im_moving, CuArray(Float32.(im_moving.voxel_data)))
    # Resample on GPU
    gpu_result = MedImages.resample_to_image(gpu_fixed, gpu_moving, interp)
    # Move result back to CPU
    cpu_result = MedImages.update_voxel_data(gpu_result, Array(gpu_result.voxel_data))
    return cpu_result
end

function main()
    data_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "..", "data", "pat_6_files")
    studies = parse_studies_from_hierarchy(data_dir)
    
    baseline_ct_path = joinpath(data_dir, studies[1][4])
    baseline_ct = MedImages.load_image(baseline_ct_path, "CT")
    
    h5_path = joinpath(data_dir, "preprocessed_volumes.h5")
    
    # 4. Bone Subsegmentation Precomputation (per-timepoint Skellytour)
    println("Starting Bone Subsegmentation Precomputation...")
    bone_h5_path = joinpath(data_dir, "Bone_Subsegments_0.h5")
    bone_h5 = h5open(bone_h5_path, "w")
    
    h5_read = h5open(h5_path, "r")
    
    cis = CartesianIndices(size(baseline_ct.voxel_data))
    temp_lesion = joinpath(data_dir, "temp_lesion.nii.gz")
    temp_surf = joinpath(data_dir, "temp_surf.nii.gz")
    temp_marr = joinpath(data_dir, "temp_marr.nii.gz")
    
    for (s_idx, study) in enumerate(studies)
        modality = study[1]
        orig_tp = study[2]
        date_str = study[3]
        ct_fname = study[4]
        pet_fname = study[5]
        mask_fname = study[6]
        node_name = study[7]
        tfm_fname = study[8]
        skellytour_source = study[12]  # per-timepoint Skellytour from hierarchy
        max_anatomy_source = study[10]  # per-timepoint max_anatomy
        max_anatomy_labels_source = study[11]  # per-timepoint max_anatomy labels
        
        # Load per-timepoint Skellytour (no sharing across time points!)
        if isempty(skellytour_source)
            error("No Skellytour path in scene_hierarchy.json for $(ct_fname). Run: julia scripts/preprocessing/update_scene_hierarchy.jl")
        end
        skelly_path = joinpath(data_dir, skellytour_source)
        if !isfile(skelly_path)
            error("Skellytour file not found: $skelly_path. Run anatomy segmentation for $(ct_fname) first.")
        end
        println("  Loading Skellytour for $(ct_fname): $(basename(skellytour_source))")
        skelly_img = MedImages.load_image(skelly_path, "CT")
        
        # Apply transform if this is a non-baseline timepoint (e.g., SPECT)
        tfm_path = joinpath(data_dir, tfm_fname)
        if tfm_fname != "" && isfile(tfm_path)
            T_ITK = parse_tfm(tfm_path)
            skelly_img = apply_transform_to_medimage(skelly_img, T_ITK)
            println("    Applied transform $(tfm_fname) to Skellytour")
        end
        
        # Resample Skellytour to baseline CT grid if dimensions differ
        # IMPORTANT: save resampled version as temp file so Python gets the correct grid
        skelly_path_for_python = skelly_path
        if size(skelly_img.voxel_data) != size(baseline_ct.voxel_data) || skelly_img.spacing != baseline_ct.spacing || skelly_img.origin != baseline_ct.origin
            println("    Resampling Skellytour $(size(skelly_img.voxel_data)) → $(size(baseline_ct.voxel_data))")
            skelly_resampled = gpu_resample_to_image(baseline_ct, skelly_img, MedImages.Nearest_neighbour_en)
            skelly_vox = Float32.(skelly_resampled.voxel_data)
            # Save resampled Skellytour for Python to use
            skelly_path_for_python = joinpath(data_dir, "temp_skelly_resampled.nii.gz")
            MedImages.create_nii_from_medimage(skelly_resampled, skelly_path_for_python)
        else
            skelly_vox = Float32.(skelly_img.voxel_data)
        end
        
        # Load max_anatomy path and bone label IDs for surface computation
        max_anat_path_for_python = ""
        bone_labels_str = ""
        if !isempty(max_anatomy_source) && !isempty(max_anatomy_labels_source)
            max_anat_path = joinpath(data_dir, max_anatomy_source)
            labels_path = joinpath(data_dir, max_anatomy_labels_source)
            if isfile(max_anat_path) && isfile(labels_path)
                max_labels = JSON.parsefile(labels_path)
                bone_kws = ["femur", "hip", "vertebra", "rib", "sacrum", "clavicula", "humerus",
                            "scapula_", "sternum", "skull", "palate", "bone", "spine", "mandible", "costal"]
                bone_ids = [k for (k, v) in max_labels if any(kw -> occursin(kw, lowercase(v)), bone_kws)]
                bone_labels_str = join(bone_ids, ",")
                println("    max_anatomy bone labels: $(length(bone_ids)) IDs")
                
                # Check if max_anatomy needs resampling to baseline grid
                max_anat_img = MedImages.load_image(max_anat_path, "CT")
                
                # Apply transform if this is a non-baseline timepoint
                if tfm_fname != "" && isfile(tfm_path)
                    T_ITK = parse_tfm(tfm_path)
                    max_anat_img = apply_transform_to_medimage(max_anat_img, T_ITK)
                    println("    Applied transform $(tfm_fname) to max_anatomy")
                end
                
                if size(max_anat_img.voxel_data) != size(baseline_ct.voxel_data) || max_anat_img.spacing != baseline_ct.spacing || max_anat_img.origin != baseline_ct.origin
                    println("    Resampling max_anatomy $(size(max_anat_img.voxel_data)) → $(size(baseline_ct.voxel_data))")
                    max_anat_resampled = gpu_resample_to_image(baseline_ct, max_anat_img, MedImages.Nearest_neighbour_en)
                    max_anat_path_for_python = joinpath(data_dir, "temp_max_anatomy_resampled.nii.gz")
                    MedImages.create_nii_from_medimage(max_anat_resampled, max_anat_path_for_python)
                else
                    max_anat_path_for_python = max_anat_path
                end
            end
        end
        
        group = tfm_fname == "" ? "BASELINE" : "TFM_" * tfm_fname
        if !haskey(h5_read, group) || !haskey(h5_read[group], mask_fname)
            continue
        end
        println("  Extracting bone subsegments for Study $s_idx ($node_name in $group/$mask_fname)...")
        mask_vol = read(h5_read["$group/$mask_fname"])
        ct_vol = read(h5_read["$group/$ct_fname"])
        
        temp_ct = joinpath(data_dir, "temp_ct.nii.gz")
        ct_to_save = MedImages.update_voxel_data(baseline_ct, Float32.(ct_vol))
        MedImages.create_nii_from_medimage(ct_to_save, temp_ct)
        
        lesion_ids = unique(mask_vol)
        filter!(x -> x > 0, lesion_ids)
        
        for lid_float in lesion_ids
            lid = Int(lid_float)
            println("    Processing Study $s_idx Lesion ID: $lid")
            
            bin_mask = (mask_vol .== lid_float)
            
            # Require minimum overlap with Skellytour (at least 5% of lesion or 5 voxels)
            skelly_overlap = count(bin_mask .& (skelly_vox .> 0))
            lesion_voxels = count(bin_mask)
            min_overlap = max(5, round(Int, 0.05 * lesion_voxels))
            if skelly_overlap < min_overlap
                println("      Skipping: insufficient bone overlap ($skelly_overlap < $min_overlap)")
                continue
            end
            
            img_to_save = MedImages.update_voxel_data(baseline_ct, Float32.(bin_mask))
            MedImages.create_nii_from_medimage(img_to_save, temp_lesion)
            
            try
                AIInference.run_bone_subsegmentation(temp_lesion, skelly_path_for_python, temp_surf, temp_marr; 
                    ct_path=temp_ct, max_anatomy_path=max_anat_path_for_python, bone_label_ids=bone_labels_str)
                
                surf_nii = NIfTI.niread(temp_surf)
                marr_nii = NIfTI.niread(temp_marr)
                
                surf_aligned = reverse(Float32.(surf_nii.raw), dims=2)
                marr_aligned = reverse(Float32.(marr_nii.raw), dims=2)
                
                surf_pts = findall(surf_aligned .> 0)
                marr_pts = findall(marr_aligned .> 0)
                
                lin_surf = map(idx -> LinearIndices(cis)[idx], surf_pts)
                lin_marr = map(idx -> LinearIndices(cis)[idx], marr_pts)
                
                bone_h5["$(node_name)_lesion_$(lid)_surf"] = lin_surf
                bone_h5["$(node_name)_lesion_$(lid)_marr"] = lin_marr
                bone_h5["tp_$(s_idx-1)_lesion_$(lid)_surf"] = lin_surf
                bone_h5["tp_$(s_idx-1)_lesion_$(lid)_marr"] = lin_marr
                
                if s_idx == 1
                    bone_h5["lesion_$(lid)_surf"] = lin_surf
                    bone_h5["lesion_$(lid)_marr"] = lin_marr
                end
                
            catch e
                println("      Error processing lesion $lid: ", e)
            end
        end
    end
    
    close(h5_read)
    close(bone_h5)
    rm(temp_lesion, force=true)
    rm(temp_surf, force=true)
    rm(temp_marr, force=true)
    rm(joinpath(data_dir, "temp_ct.nii.gz"), force=true)
    rm(joinpath(data_dir, "temp_skelly_resampled.nii.gz"), force=true)
    rm(joinpath(data_dir, "temp_max_anatomy_resampled.nii.gz"), force=true)
    
    println("Pre-processing complete.")
end
main()
