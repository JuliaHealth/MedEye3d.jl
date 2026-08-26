using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

using MedImages
using JSON
using HDF5
using LinearAlgebra
using NIfTI

# Load MedEye3d logic needed for AI inference
include(joinpath(@__DIR__, "..", "..", "src", "ai", "AIInference.jl"))
using .AIInference

# Load shared SceneHierarchy module
include(joinpath(@__DIR__, "..", "lib", "SceneHierarchy.jl"))
using .SceneHierarchy

using MedEye3d

const HIRES_FACTOR = 2.0

function main()
    if isempty(ARGS)
        error("Usage: julia scripts/preprocess_dataset.jl <data_dir>")
    end
    data_dir = abspath(ARGS[1])
    
    scene_json = joinpath(data_dir, "scene_hierarchy.json")
    if !isfile(scene_json)
        error("scene_hierarchy.json not found in $data_dir")
    end
    
    hierarchy = JSON.parse(read(scene_json, String))
    
    baseline_ct_path = ""
    baseline_ct = nothing
    
    studies = parse_studies_from_hierarchy(data_dir)
    if isempty(studies)
        error("No studies parsed from hierarchy in $data_dir")
    end
    
    # 1. Identify baseline CT (first study in chronological hierarchy)
    baseline_ct_path = joinpath(data_dir, studies[1][4])
    if !isfile(baseline_ct_path)
        error("Baseline CT not found at $baseline_ct_path!")
    end
    
    println("Baseline CT: ", baseline_ct_path)
    baseline_ct = MedImages.load_image(baseline_ct_path, "CT")
    
    # 2. Skellytour Execution
    skelly_path = joinpath(data_dir, "Skellytour_0.nii.gz")
    if !isfile(skelly_path)
        println("Generating Skellytour bone mask...")
        skelly_out_dir = joinpath(data_dir, "skelly_0")
        mkpath(skelly_out_dir)
        AIInference.run_skellytour_segmentation(baseline_ct_path, skelly_out_dir)
        
        out_files = filter(x -> endswith(x, ".nii.gz"), readdir(skelly_out_dir))
        if !isempty(out_files)
            mv(joinpath(skelly_out_dir, out_files[1]), skelly_path, force=true)
            println("Moved Skellytour output to $skelly_path")
        end
    end
    
    # 3. Process All Nodes
    h5_path = joinpath(data_dir, "preprocessed_volumes.h5")
    println("Resampling and saving to $h5_path")
    h5_file = h5open(h5_path, "w")
    
    # Embed JSON metadata as HDF5 datasets (HDF5 = single source of truth)
    # Using datasets in _meta_ group instead of attributes due to 64KB attribute size limit
    if !haskey(h5_file, "_meta_")
        create_group(h5_file, "_meta_")
    end
    for json_name in ["scene_hierarchy.json", "metadata.json", "matches.json"]
        json_path = joinpath(data_dir, json_name)
        if isfile(json_path)
            h5_file["_meta_/$json_name"] = read(json_path, String)
            println("  Embedded $json_name as HDF5 dataset ($(filesize(json_path)) bytes)")
        end
    end
    
    function process_file(name, tfm_path, group_name, is_mask=false)
        if isempty(name)
            return
        end
        nii_path = joinpath(data_dir, name)
        if !isfile(nii_path)
            println("  Warning: Could not find file for $name")
            return
        end
        
        println("  Processing $name...")
        img_type = "CT" # default
        if occursin("PET", name) || occursin("SPECT", name) || occursin("SUV", name) || occursin("NM", name)
            img_type = occursin("SPECT", name) || occursin("NM", name) ? "NM" : "PET"
        end
        
        img = if endswith(name, ".seg.nrrd")
            nii_alt = replace(nii_path, ".seg.nrrd" => ".nii.gz")
            if isfile(nii_alt)
                MedImages.load_image(nii_alt, "CT")
            else
                println("  Skipping raw .seg.nrrd $name (NRRD header parsed directly at runtime)")
                return
            end
        else
            MedImages.load_image(nii_path, img_type)
        end
        
        T_ITK = parse_tfm(tfm_path)
        img_tfm = apply_transform_to_medimage(img, T_ITK)
        
        # Resample
        interpolator = is_mask ? MedImages.Nearest_neighbour_en : MedImages.Linear_en
        
        if T_ITK != Matrix{Float64}(I, 4, 4) || img.spacing != baseline_ct.spacing || img.origin != baseline_ct.origin
            img_res = MedImages.resample_to_image(baseline_ct, img_tfm, interpolator)
        else
            img_res = img
        end
        
        MedImages.save_med_image(h5_file, group_name, name, img_res; compress=3)

        # Also save at display resolution (2× in-plane upsampling)
        display_sp = (img_res.spacing[1] / HIRES_FACTOR, img_res.spacing[2] / HIRES_FACTOR, img_res.spacing[3])
        interpolator_display = is_mask ? MedImages.Nearest_neighbour_en : MedImages.Linear_en
        img_display = MedImages.resample_to_spacing(img_res, display_sp, interpolator_display)
        display_group = group_name * "_DISPLAY"
        MedImages.save_med_image(h5_file, display_group, name, img_display; compress=3)
        println("    Saved display-resolution ($(size(img_display.voxel_data))) to $display_group/$name")
    end
    
    for study in studies
        modality = study[1]
        orig_tp = study[2]
        date_str = study[3]
        ct_fname = study[4]
        pet_fname = study[5]
        mask_fname = study[6]
        node_name = study[7]
        tfm_fname = study[8]
        ts_fname = length(study) >= 9 ? study[9] : ""
        
        ct_fname_full = joinpath(data_dir, ct_fname)
        ts_dir = joinpath(data_dir, "anatomy_out")
        max_anatomy_file = joinpath(ts_dir, "max_anatomy.nii.gz")
        
        if !isfile(max_anatomy_file)
            println("    max_anatomy.nii.gz missing. Triggering Comprehensive Anatomy Segmentation via AI TCP worker...")
            try
                dict_req = Dict(
                    "command" => "run_anatomy",
                    "ct_path" => ct_fname_full,
                    "out_dir" => ts_dir,
                    "mode" => "--fast"
                )
                worker_script = joinpath(dirname(@__DIR__), "ai", "python_worker.py")
                MedEye3d.InferenceClient.start_python_worker(worker_script)
                
                resp = MedEye3d.InferenceClient.send_json_request(dict_req)
                if get(resp, "status", "") != "success"
                    println("    Warning: Anatomy segmentation reported error: ", get(resp, "message", "Unknown error"))
                else
                    println("    Anatomy segmentation finished successfully.")
                end
            catch e
                println("    Warning: Anatomy segmentation failed: ", e)
            end
        end
        
        tfm_path = tfm_fname != "" ? joinpath(data_dir, tfm_fname) : ""
        group = tfm_fname == "" ? "BASELINE" : "TFM_" * tfm_fname
        
        process_file(ct_fname, tfm_path, group, false)
        process_file(pet_fname, tfm_path, group, false)
        process_file(mask_fname, tfm_path, group, true)
        if !isempty(ts_fname)
            process_file(ts_fname, tfm_path, group, true)
        end
    end
    
    close(h5_file)
    GC.gc()
    println("Saved preprocessed volumes.")
    
    # 4. Bone Subsegmentation Precomputation (per-timepoint Skellytour)
    println("Starting Bone Subsegmentation Precomputation...")
    bone_h5_path = joinpath(data_dir, "Bone_Subsegments_0.h5")
    bone_h5 = h5open(bone_h5_path, "w")
    
    # Iterate over all masks saved in preprocessed_volumes.h5
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
        local_skelly_path = joinpath(data_dir, skellytour_source)
        if !isfile(local_skelly_path)
            error("Skellytour file not found: $local_skelly_path. Run anatomy segmentation for $(ct_fname) first.")
        end
        println("  Loading Skellytour for $(ct_fname): $(basename(skellytour_source))")
        skelly_img = MedImages.load_image(local_skelly_path, "CT")
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
        
        lesion_ids = unique(mask_vol)
        filter!(x -> x > 0, lesion_ids)
        
        for lid_float in lesion_ids
            lid = Int(lid_float)
            println("    Processing Study $s_idx Lesion ID: $lid")
            
            # Create binary mask for this lesion
            bin_mask = (mask_vol .== lid_float)
            
            # Check if it overlaps with bone (skellytour > 0)
            if !any(bin_mask .& (skelly_vox .> 0))
                println("      Skipping: Not a bone lesion.")
                continue
            end
            
            # Save temporary NIfTI
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
                
                # Convert CartesianIndex to linear index
                lin_surf = map(idx -> LinearIndices(cis)[idx], surf_pts)
                lin_marr = map(idx -> LinearIndices(cis)[idx], marr_pts)
                
                # Store by study node name and TP index
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
    
    # Clean up temp files
    rm(temp_lesion, force=true)
    rm(temp_surf, force=true)
    rm(temp_marr, force=true)
    
    println("Pre-processing complete.")
end

main()
