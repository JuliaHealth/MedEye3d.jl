using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

using MedImages
using JSON
using HDF5
using LinearAlgebra
using Statistics
using NIfTI
using CUDA

# Load MedEye3d logic needed for AI inference
include(joinpath(@__DIR__, "..", "..", "src", "ai", "AIInference.jl"))
using .AIInference

# Load shared SceneHierarchy module
include(joinpath(@__DIR__, "..", "lib", "SceneHierarchy.jl"))
using .SceneHierarchy

const HIRES_FACTOR = 1.0

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
        
        # Always use Int16 for label masks — matches TextureSpec{Int16} and R16_SINT format
        # Int16 range [-32768, 32767] supports any segmentation label count
        if is_mask
            vox = Float32.(img_res.voxel_data)
            vox = max.(0.0f0, vox)
            max_id = round(Int, maximum(vox))
            img_res = MedImages.update_voxel_data(img_res, Int16.(round.(vox)))
            println("    Compacted mask to Int16 (max_id=$max_id)")
        end
        
        # Pre-flip native resolution (single flip during preprocessing eliminates runtime reverse)
        img_flipped = MedImages.update_voxel_data(img_res, reverse(img_res.voxel_data, dims=2))
        MedImages.save_med_image(h5_file, group_name, name, img_flipped; compress=3)
        println("    Saved native pre-flipped ($(size(img_flipped.voxel_data))) to $group_name/$name")
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
                
                # Native resolution: pre-flipped + UInt16 compact
                img_disp = MedImages.update_voxel_data(img_res, reverse(img_res.voxel_data, dims=2))
                vox = UInt16.(round.(max.(0.0f0, Float32.(img_disp.voxel_data))))
                img_disp = MedImages.update_voxel_data(img_disp, vox)
                MedImages.save_med_image(h5_file, group, anat_name, img_disp; compress=3)
                println("    Saved $(size(vox)) to $group/$anat_name (UInt16 pre-flipped)")
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
    
    # ═══════════════════════════════════════════════════════════════════════
    # Phase 1.5: Store registration transforms (.tfm) into HDF5 _meta_
    # ═══════════════════════════════════════════════════════════════════════
    
    println("\n=== Phase 1.5: Storing registration transforms ==")
    h5_file = h5open(h5_path, "r+")
    if !haskey(h5_file, "_meta_/transforms")
        create_group(h5_file, "_meta_/transforms")
    end
    tfm_count = 0
    for (s_idx, study) in enumerate(studies)
        tfm_name = study[8]  # e.g. "Transform_FollowUp_to_Baseline_1.tfm"
        if !isempty(tfm_name)
            tfm_path = joinpath(data_dir, tfm_name)
            if isfile(tfm_path)
                tfm_content = read(tfm_path, String)
                ds_name = "_meta_/transforms/$tfm_name"
                if haskey(h5_file, ds_name)
                    delete_object(h5_file, ds_name)
                end
                h5_file[ds_name] = tfm_content
                tfm_count += 1
                println("  Stored transform: $tfm_name ($(length(tfm_content)) bytes)")
            else
                println("  ⚠️  Transform file not found: $tfm_path")
            end
        end
    end
    println("  Stored $tfm_count transforms total")
    close(h5_file)
    
    # ═══════════════════════════════════════════════════════════════════════
    # Phase 2: Store atlas, skellytour, bone_atlas, labels, organ mapping,
    #          centroids, and bone subsegments into the SAME HDF5.
    #          This makes HDF5 the SINGLE source of truth for startup.
    # ═══════════════════════════════════════════════════════════════════════
    
    println("\n=== Phase 2: Embedding startup data into HDF5 ===")
    h5_file = h5open(h5_path, "r+")  # Reopen for appending
    
    # --- 2a. Global max_anatomy atlas (baseline) ---
    baseline_study = studies[1]
    max_anatomy_source = length(baseline_study) >= 10 ? baseline_study[10] : ""
    max_anatomy_labels_file = length(baseline_study) >= 11 ? baseline_study[11] : ""
    skellytour_source = length(baseline_study) >= 12 ? baseline_study[12] : ""
    
    if !haskey(h5_file, "ATLAS")
        create_group(h5_file, "ATLAS")
    end
    
    # Store max_anatomy atlas (pre-flipped UInt16)
    ts_atlas_aligned = nothing
    ts_names = Dict{Int,String}()
    if !isempty(max_anatomy_source)
        max_anat_path = joinpath(data_dir, max_anatomy_source)
        if isfile(max_anat_path)
            println("  Storing global max_anatomy atlas...")
            anat_img = MedImages.load_image(max_anat_path, "CT")
            anat_raw = UInt16.(round.(max.(0.0f0, Float32.(anat_img.voxel_data))))
            ts_atlas_aligned = reverse(anat_raw, dims=2)  # Pre-flip
            h5_file["ATLAS/max_anatomy"] = ts_atlas_aligned
            println("    Saved ATLAS/max_anatomy ($(size(ts_atlas_aligned)))")
        end
    end
    
    # Store max_anatomy labels JSON
    if !isempty(max_anatomy_labels_file)
        labels_path = joinpath(data_dir, max_anatomy_labels_file)
        # Prefer real names if available
        real_labels_path = joinpath(data_dir, "anatomy_out", "max_anatomy_labels.json")
        if isfile(real_labels_path)
            labels_path = real_labels_path
        end
        if isfile(labels_path)
            println("  Storing max_anatomy labels...")
            h5_file["_meta_/max_anatomy_labels.json"] = read(labels_path, String)
            raw_labels = JSON.parsefile(labels_path)
            ts_names = Dict{Int,String}(parse(Int, k) => v for (k, v) in raw_labels)
            println("    Saved _meta_/max_anatomy_labels.json ($(length(ts_names)) classes)")
        end
    end
    
    # Store per-TP anatomy labels
    for (s_idx, study) in enumerate(studies)
        tp_i = s_idx - 1
        tp_labels_file = length(study) >= 11 ? study[11] : ""
        if !isempty(tp_labels_file)
            tp_labels_path = joinpath(data_dir, tp_labels_file)
            if isfile(tp_labels_path)
                h5_file["_meta_/anatomy_labels_tp_$(tp_i).json"] = read(tp_labels_path, String)
            end
        end
    end
    
    # Store Skellytour (pre-flipped Float32)
    if !isempty(skellytour_source)
        skelly_path = joinpath(data_dir, skellytour_source)
        if isfile(skelly_path)
            println("  Storing Skellytour bone atlas...")
            skelly_img = MedImages.load_image(skelly_path, "CT")
            skelly_aligned = reverse(Float32.(skelly_img.voxel_data), dims=2)
            h5_file["ATLAS/skellytour"] = skelly_aligned
            println("    Saved ATLAS/skellytour ($(size(skelly_aligned)))")
        end
    end
    
    # Compute and store bone_atlas (binary mask of bone-related label IDs)
    if ts_atlas_aligned !== nothing && !isempty(ts_names)
        println("  Computing bone_atlas binary mask...")
        bone_keywords = ["femur", "hip", "vertebra", "rib", "sacrum", "clavicula", "humerus",
                          "scapula", "sternum", "skull", "palate", "bone", "spine", "mandible",
                          "costal"]
        bone_label_ids = Set{Int}()
        for (k, v) in ts_names
            if any(kw -> occursin(kw, lowercase(v)), bone_keywords)
                push!(bone_label_ids, k)
            end
        end
        bone_atlas = Float32.(in.(ts_atlas_aligned, Ref(bone_label_ids)))
        h5_file["ATLAS/bone_atlas"] = bone_atlas
        h5_file["_meta_/bone_label_ids"] = collect(bone_label_ids)
        println("    Saved ATLAS/bone_atlas ($(count(bone_atlas .> 0)) bone voxels, $(length(bone_label_ids)) label IDs)")
    end
    
    # Compute and store organ_mapping for baseline mask
    if ts_atlas_aligned !== nothing && !isempty(ts_names)
        println("  Computing organ_mapping for baseline mask...")
        # Read baseline mask from HDF5
        base_mask_fname = studies[1][6]
        base_group = studies[1][8] == "" ? "BASELINE" : "TFM_" * studies[1][8]
        if haskey(h5_file, "$base_group/$base_mask_fname")
            mask_raw = read(h5_file["$base_group/$base_mask_fname"])
            mask_f32 = Float32.(mask_raw)
            
            # Map each lesion to the most common overlapping organ
            organ_mapping = Dict{Int, String}()
            unique_ids = filter(x -> x > 0, unique(mask_f32))
            for lid_f in unique_ids
                lid = Int(lid_f)
                lesion_mask = mask_f32 .== lid_f
                # Find most common atlas label overlapping this lesion
                atlas_vals = ts_atlas_aligned[lesion_mask]
                nonzero = filter(x -> x > 0, atlas_vals)
                if !isempty(nonzero)
                    # Count occurrences
                    counts = Dict{UInt16, Int}()
                    for v in nonzero
                        counts[v] = get(counts, v, 0) + 1
                    end
                    best_label = first(sort(collect(counts), by=x->x[2], rev=true))[1]
                    organ_mapping[lid] = get(ts_names, Int(best_label), "Unknown")
                end
            end
            
            # Store as JSON
            organ_json = JSON.json(Dict(string(k) => v for (k, v) in organ_mapping))
            h5_file["_meta_/organ_mapping"] = organ_json
            println("    Saved _meta_/organ_mapping ($(length(organ_mapping)) lesions mapped)")
        end
    end
    
    # --- 2b. Compute and store centroids for ALL TPs ---
    println("  Computing lesion centroids for all TPs...")
    if !haskey(h5_file, "CENTROIDS")
        create_group(h5_file, "CENTROIDS")
    end
    
    centroid_count = 0
    for (s_idx, study) in enumerate(studies)
        tp_idx = s_idx - 1
        node_name = study[7]
        mask_fname = study[6]
        tfm_fname = study[8]
        group = tfm_fname == "" ? "BASELINE" : "TFM_" * tfm_fname
        
        if haskey(h5_file, group) && haskey(h5_file[group], mask_fname)
            mask_vol = Float32.(read(h5_file["$group/$mask_fname"]))
            u_lids = filter(x -> x > 0, unique(mask_vol))
            for lid_f in u_lids
                lid = Int(lid_f)
                # Find centroid (mean of voxel coordinates)
                indices = findall(mask_vol .== lid_f)
                if !isempty(indices)
                    cx = round(Int, mean(idx -> idx[1], indices))
                    cy = round(Int, mean(idx -> idx[2], indices))
                    cz = round(Int, mean(idx -> idx[3], indices))
                    h5_file["CENTROIDS/tp$(tp_idx)_lid$(lid)"] = Int32[cx, cy, cz]
                    centroid_count += 1
                end
            end
        end
    end
    println("    Saved $centroid_count centroids across $(length(studies)) TPs")
    
    # --- 2c. Bone subsegmentation (moved into main HDF5) ---
    println("\n=== Phase 3: Bone Subsegmentation ===")
    if !haskey(h5_file, "BONE_SUBSEG")
        create_group(h5_file, "BONE_SUBSEG")
    end
    
    cis = CartesianIndices(size(baseline_ct.voxel_data))
    temp_lesion = joinpath(data_dir, "temp_lesion.nii.gz")
    temp_surf = joinpath(data_dir, "temp_surf.nii.gz")
    temp_marr = joinpath(data_dir, "temp_marr.nii.gz")
    
    for (s_idx, study) in enumerate(studies)
        modality = study[1]
        ct_fname = study[4]
        mask_fname = study[6]
        node_name = study[7]
        tfm_fname = study[8]
        skellytour_src = study[12]
        max_anatomy_src = length(study) >= 10 ? study[10] : ""
        max_anatomy_lbl = length(study) >= 11 ? study[11] : ""
        
        # Load per-timepoint Skellytour
        if isempty(skellytour_src)
            @warn "No Skellytour path for $(ct_fname), skipping bone subseg"
            continue
        end
        local_skelly_path = joinpath(data_dir, skellytour_src)
        if !isfile(local_skelly_path)
            @warn "Skellytour not found: $local_skelly_path, skipping"
            continue
        end
        println("  Loading Skellytour for $(ct_fname): $(basename(skellytour_src))")
        skelly_img = MedImages.load_image(local_skelly_path, "CT")
        tfm_path = joinpath(data_dir, tfm_fname)
        if tfm_fname != "" && isfile(tfm_path)
            T_ITK = parse_tfm(tfm_path)
            skelly_img = apply_transform_to_medimage(skelly_img, T_ITK)
        end
        
        skelly_path_for_python = local_skelly_path
        if size(skelly_img.voxel_data) != size(baseline_ct.voxel_data) || skelly_img.spacing != baseline_ct.spacing || skelly_img.origin != baseline_ct.origin
            skelly_resampled = gpu_resample_to_image(baseline_ct, skelly_img, MedImages.Nearest_neighbour_en)
            skelly_vox = Float32.(skelly_resampled.voxel_data)
            skelly_path_for_python = joinpath(data_dir, "temp_skelly_resampled.nii.gz")
            MedImages.create_nii_from_medimage(skelly_resampled, skelly_path_for_python)
        else
            skelly_vox = Float32.(skelly_img.voxel_data)
        end
        
        # Load max_anatomy for surface computation
        max_anat_path_for_python = ""
        bone_labels_str = ""
        if !isempty(max_anatomy_src) && !isempty(max_anatomy_lbl)
            max_anat_path = joinpath(data_dir, max_anatomy_src)
            labels_path = joinpath(data_dir, max_anatomy_lbl)
            if isfile(max_anat_path) && isfile(labels_path)
                max_labels = JSON.parsefile(labels_path)
                bone_kws = ["femur", "hip", "vertebra", "rib", "sacrum", "clavicula", "humerus",
                            "scapula_", "sternum", "skull", "palate", "bone", "spine", "mandible", "costal"]
                bone_ids = [k for (k, v) in max_labels if any(kw -> occursin(kw, lowercase(v)), bone_kws)]
                bone_labels_str = join(bone_ids, ",")
                
                max_anat_img = MedImages.load_image(max_anat_path, "CT")
                if tfm_fname != "" && isfile(tfm_path)
                    T_ITK = parse_tfm(tfm_path)
                    max_anat_img = apply_transform_to_medimage(max_anat_img, T_ITK)
                end
                if size(max_anat_img.voxel_data) != size(baseline_ct.voxel_data) || max_anat_img.spacing != baseline_ct.spacing || max_anat_img.origin != baseline_ct.origin
                    max_anat_resampled = gpu_resample_to_image(baseline_ct, max_anat_img, MedImages.Nearest_neighbour_en)
                    max_anat_path_for_python = joinpath(data_dir, "temp_max_anatomy_resampled.nii.gz")
                    MedImages.create_nii_from_medimage(max_anat_resampled, max_anat_path_for_python)
                else
                    max_anat_path_for_python = max_anat_path
                end
            end
        end
        
        group = tfm_fname == "" ? "BASELINE" : "TFM_" * tfm_fname
        if !haskey(h5_file, group) || !haskey(h5_file[group], mask_fname)
            continue
        end
        println("  Extracting bone subsegments for Study $s_idx ($node_name)...")
        mask_vol = read(h5_file["$group/$mask_fname"])
        ct_vol = read(h5_file["$group/$ct_fname"])
        
        temp_ct = joinpath(data_dir, "temp_ct.nii.gz")
        ct_unflipped = reverse(Float32.(ct_vol), dims=2)
        ct_to_save = MedImages.update_voxel_data(baseline_ct, ct_unflipped)
        MedImages.create_nii_from_medimage(ct_to_save, temp_ct)
        
        mask_unflipped = reverse(Float32.(mask_vol), dims=2)
        lesion_ids = unique(mask_unflipped)
        filter!(x -> x > 0, lesion_ids)
        
        for lid_float in lesion_ids
            lid = Int(lid_float)
            println("    Processing Study $s_idx Lesion ID: $lid")
            
            bin_mask = (mask_unflipped .== lid_float)
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
                
                # Store in BONE_SUBSEG group of main HDF5
                h5_file["BONE_SUBSEG/$(node_name)_lesion_$(lid)_surf"] = lin_surf
                h5_file["BONE_SUBSEG/$(node_name)_lesion_$(lid)_marr"] = lin_marr
                h5_file["BONE_SUBSEG/tp_$(s_idx-1)_lesion_$(lid)_surf"] = lin_surf
                h5_file["BONE_SUBSEG/tp_$(s_idx-1)_lesion_$(lid)_marr"] = lin_marr
                
                if s_idx == 1
                    h5_file["BONE_SUBSEG/lesion_$(lid)_surf"] = lin_surf
                    h5_file["BONE_SUBSEG/lesion_$(lid)_marr"] = lin_marr
                end
            catch e
                println("      Error processing lesion $lid: ", e)
            end
        end
    end
    
    close(h5_file)
    
    # Clean up temp files
    rm(temp_lesion, force=true)
    rm(temp_surf, force=true)
    rm(temp_marr, force=true)
    rm(joinpath(data_dir, "temp_ct.nii.gz"), force=true)
    rm(joinpath(data_dir, "temp_skelly_resampled.nii.gz"), force=true)
    rm(joinpath(data_dir, "temp_max_anatomy_resampled.nii.gz"), force=true)
    
    println("\n✅ Pre-processing complete. HDF5 is now the single source of truth.")
    println("   All atlas, centroids, organ mapping, and bone subsegments stored in: $h5_path")
end

main()
