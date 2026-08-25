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
    
    skelly_path = joinpath(data_dir, "Skellytour_0.nii.gz")
    h5_path = joinpath(data_dir, "preprocessed_volumes.h5")
    
    # 4. Bone Subsegmentation Precomputation
    println("Starting Bone Subsegmentation Precomputation...")
    bone_h5_path = joinpath(data_dir, "Bone_Subsegments_0.h5")
    bone_h5 = h5open(bone_h5_path, "w")
    
    h5_read = h5open(h5_path, "r")
    skelly_nii = NIfTI.niread(skelly_path)
    skelly_vox = Float32.(skelly_nii.raw)
    skelly_aligned = reverse(skelly_vox, dims=2)
    
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
        
        group = tfm_fname == "" ? "BASELINE" : "TFM_" * tfm_fname
        if !haskey(h5_read, group) || !haskey(h5_read[group], mask_fname)
            continue
        end
        println("  Extracting bone subsegments for Study $s_idx ($node_name in $group/$mask_fname)...")
        mask_vol = read(h5_read["$group/$mask_fname"])
        
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
                AIInference.run_bone_subsegmentation(temp_lesion, skelly_path, temp_surf, temp_marr)
                
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
    
    println("Pre-processing complete.")
end
main()
