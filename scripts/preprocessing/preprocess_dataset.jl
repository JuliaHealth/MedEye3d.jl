using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

using MedImages
using JSON
using HDF5
using LinearAlgebra
using NIfTI
using CUDA

# Load MedEye3d logic needed for AI inference
include(joinpath(@__DIR__, "..", "..", "src", "ai", "AIInference.jl"))
using .AIInference

# Load shared SceneHierarchy module
include(joinpath(@__DIR__, "..", "lib", "SceneHierarchy.jl"))
using .SceneHierarchy

using MedEye3d

const HIRES_FACTOR = 2.0

"""
GPU-accelerated resample_to_image: moves voxel data to GPU, resamples, moves back to CPU.
"""
function gpu_resample_to_image(im_fixed::MedImages.MedImage, im_moving::MedImages.MedImage, interp::MedImages.Interpolator_enum)
    gpu_fixed = MedImages.update_voxel_data(im_fixed, CuArray(Float32.(im_fixed.voxel_data)))
    gpu_moving = MedImages.update_voxel_data(im_moving, CuArray(Float32.(im_moving.voxel_data)))
    gpu_result = MedImages.resample_to_image(gpu_fixed, gpu_moving, interp)
    cpu_result = MedImages.update_voxel_data(gpu_result, Array(gpu_result.voxel_data))
    return cpu_result
end

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
    
    # Translate German descriptions to English via DIZ LLM before embedding metadata
    println("Translating radiological descriptions if needed...")
    try
        include(joinpath(@__DIR__, "translate_reports.jl"))
        TranslateReports.translate_descriptions!(data_dir)
    catch e
        @warn "Translation step failed or skipped: $e"
    end

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
            img_res = gpu_resample_to_image(baseline_ct, img_tfm, interpolator)
        else
            img_res = img
        end
        
        # Compact masks to Int8/Int16 (saves 75% storage + eliminates runtime conversion)
        if is_mask
            vox = Float32.(img_res.voxel_data)
            vox = max.(0.0f0, vox)
            max_id = round(Int, maximum(vox))
            T = max_id + 5 <= 127 ? Int8 : Int16
            img_res = MedImages.update_voxel_data(img_res, T.(round.(vox)))
            println("    Compacted mask to $T (max_id=$max_id)")
        end
        
        # Native resolution: saved WITHOUT flip (loading code applies reverse if needed)
        MedImages.save_med_image(h5_file, group_name, name, img_res; compress=3)

        # Also save at display resolution (2× in-plane upsampling)
        display_sp = (img_res.spacing[1] / HIRES_FACTOR, img_res.spacing[2] / HIRES_FACTOR, img_res.spacing[3])
        interpolator_display = is_mask ? MedImages.Nearest_neighbour_en : MedImages.Linear_en
        img_display = MedImages.resample_to_spacing(img_res, display_sp, interpolator_display)
        
        # Pre-flip DISPLAY resolution only (single flip — replaces runtime reverse())
        img_display = MedImages.update_voxel_data(img_display, reverse(img_display.voxel_data, dims=2))
        
        # Compact display mask
        if is_mask
            vox = Float32.(img_display.voxel_data)
            vox = max.(0.0f0, vox)
            max_id = round(Int, maximum(vox))
            T = max_id + 5 <= 127 ? Int8 : Int16
            img_display = MedImages.update_voxel_data(img_display, T.(round.(vox)))
        end
        
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
        max_anat_src = length(study) >= 10 ? study[10] : ""
        
        ct_fname_full = joinpath(data_dir, ct_fname)
        
        # Check per-TP max_anatomy — trigger inference if missing
        if !isempty(max_anat_src)
            max_anat_full = joinpath(data_dir, max_anat_src)
            tp_anat_dir = dirname(max_anat_full)
            if !isfile(max_anat_full)
                println("    Per-TP max_anatomy missing: $max_anat_src")
                println("    Triggering anatomy segmentation for $ct_fname...")
                try
                    dict_req = Dict(
                        "command" => "run_anatomy",
                        "ct_path" => ct_fname_full,
                        "out_dir" => tp_anat_dir,
                        "mode" => "--fast"
                    )
                    worker_script = joinpath(dirname(@__DIR__), "ai", "python_worker.py")
                    MedEye3d.InferenceClient.start_python_worker(worker_script)
                    resp = MedEye3d.InferenceClient.send_json_request(dict_req)
                    if get(resp, "status", "") == "success"
                        println("    ✅ Anatomy segmentation completed for $ct_fname")
                    else
                        println("    ⚠️  Anatomy segmentation error: ", get(resp, "message", "Unknown"))
                    end
                catch e
                    println("    ⚠️  Anatomy inference failed: $e")
                    println("    Run manually: bash scripts/ai/run_all_timepoints.sh")
                end
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
        
        # Process per-TP max_anatomy into HDF5 (resampled, pre-flipped, UInt16)
        if !isempty(max_anat_src)
            max_anat_path = joinpath(data_dir, max_anat_src)
            if isfile(max_anat_path)
                anat_name = "max_anatomy.nii.gz"
                println("  Processing per-TP max_anatomy: $max_anat_src")
                
                img = MedImages.load_image(max_anat_path, "CT")
                T_ITK = parse_tfm(tfm_path)
                img_tfm = apply_transform_to_medimage(img, T_ITK)
                
                if T_ITK != Matrix{Float64}(I, 4, 4) || img.spacing != baseline_ct.spacing || img.origin != baseline_ct.origin
                    img_res = gpu_resample_to_image(baseline_ct, img_tfm, MedImages.Nearest_neighbour_en)
                else
                    img_res = img_tfm
                end
                
                # Save native res WITHOUT flip
                MedImages.save_med_image(h5_file, group, anat_name, img_res; compress=3)
                
                # Display resolution: flip ONCE + UInt16 compact
                display_sp = (img_res.spacing[1]/HIRES_FACTOR, img_res.spacing[2]/HIRES_FACTOR, img_res.spacing[3])
                img_disp = MedImages.resample_to_spacing(img_res, display_sp, MedImages.Nearest_neighbour_en)
                img_disp = MedImages.update_voxel_data(img_disp, reverse(img_disp.voxel_data, dims=2))
                vox = UInt16.(round.(max.(0.0f0, Float32.(img_disp.voxel_data))))
                img_disp = MedImages.update_voxel_data(img_disp, vox)
                display_group = group * "_DISPLAY"
                MedImages.save_med_image(h5_file, display_group, anat_name, img_disp; compress=3)
                println("    Saved $(size(vox)) to $display_group/$anat_name (UInt16)")
            else
                println("  ⚠️  Per-TP max_anatomy not found: $max_anat_path")
            end
        end
    end
    
    # Mark HDF5 as pre-flipped so loading code skips reverse()
    h5_file["_meta_/preflipped"] = 1
    
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
        max_anatomy_source = study[10]  # per-timepoint max_anatomy
        max_anatomy_labels_source = study[11]  # per-timepoint max_anatomy labels
        
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
        # IMPORTANT: save resampled version as temp file so Python gets the correct grid
        skelly_path_for_python = local_skelly_path
        if size(skelly_img.voxel_data) != size(baseline_ct.voxel_data)
            println("    Resampling Skellytour $(size(skelly_img.voxel_data)) → $(size(baseline_ct.voxel_data))")
            skelly_resampled = gpu_resample_to_image(baseline_ct, skelly_img, MedImages.Nearest_neighbour_en)
            skelly_vox = Float32.(skelly_resampled.voxel_data)
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
                if size(max_anat_img.voxel_data) != size(baseline_ct.voxel_data)
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
        
        lesion_ids = unique(mask_vol)
        filter!(x -> x > 0, lesion_ids)
        
        for lid_float in lesion_ids
            lid = Int(lid_float)
            println("    Processing Study $s_idx Lesion ID: $lid")
            
            # Create binary mask for this lesion
            bin_mask = (mask_vol .== lid_float)
            
            # Require minimum overlap with Skellytour (at least 5% of lesion or 5 voxels)
            skelly_overlap = count(bin_mask .& (skelly_vox .> 0))
            lesion_voxels = count(bin_mask)
            min_overlap = max(5, round(Int, 0.05 * lesion_voxels))
            if skelly_overlap < min_overlap
                println("      Skipping: insufficient bone overlap ($skelly_overlap < $min_overlap)")
                continue
            end
            
            # Save temporary NIfTI
            img_to_save = MedImages.update_voxel_data(baseline_ct, Float32.(bin_mask))
            MedImages.create_nii_from_medimage(img_to_save, temp_lesion)
            
            try
                AIInference.run_bone_subsegmentation(temp_lesion, skelly_path_for_python, temp_surf, temp_marr;
                    ct_path=joinpath(data_dir, "temp_ct.nii.gz"), max_anatomy_path=max_anat_path_for_python, bone_label_ids=bone_labels_str)
                
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
