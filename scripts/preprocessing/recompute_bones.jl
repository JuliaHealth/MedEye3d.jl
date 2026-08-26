using MedImages
using NIfTI
using HDF5
using LinearAlgebra
using Statistics

include(joinpath(@__DIR__, "..", "..", "src", "ai", "AIInference.jl"))
using .AIInference
include(joinpath(@__DIR__, "..", "lib", "SceneHierarchy.jl"))
using .SceneHierarchy

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
        # Resample Skellytour to baseline CT grid if dimensions differ
        if size(skelly_img.voxel_data) != size(baseline_ct.voxel_data)
            println("    Resampling Skellytour $(size(skelly_img.voxel_data)) → $(size(baseline_ct.voxel_data))")
            skelly_resampled = MedImages.resample_to_image(baseline_ct, skelly_img, MedImages.Nearest_neighbour_en)
            skelly_vox = Float32.(skelly_resampled.voxel_data)
        else
            skelly_vox = Float32.(skelly_img.voxel_data)
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
            
            if !any(bin_mask .& (skelly_vox .> 0))
                println("      Skipping: Not a bone lesion.")
                continue
            end
            
            img_to_save = MedImages.update_voxel_data(baseline_ct, Float32.(bin_mask))
            MedImages.create_nii_from_medimage(img_to_save, temp_lesion)
            
            try
                AIInference.run_bone_subsegmentation(temp_lesion, skelly_path, temp_surf, temp_marr; ct_path=temp_ct)
                
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
    
    println("Pre-processing complete.")
end
main()
