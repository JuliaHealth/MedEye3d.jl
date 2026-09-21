#!/usr/bin/env julia
# Fix corrupted atlas in HDF5 by reimporting from source NIfTI
# and recomputing organ mapping for all TPs

using HDF5, JSON, NIfTI

data_dir = "/workspaces/MedEye3d.jl/data/pat_6_files"
h5_path = joinpath(data_dir, "preprocessed_volumes.h5")
src_path = joinpath(data_dir, "anatomy_out", "max_anatomy.nii.gz")
labels_path = joinpath(data_dir, "anatomy_out", "max_anatomy_labels.json")

println("=== Step 1: Load source NIfTI (raw, no rescaling) ===")
nii = niread(src_path)
raw_data = nii.raw  # Raw integer data without rescaling
println("  Source type: $(eltype(raw_data)), size: $(size(raw_data))")
raw_uint16 = UInt16.(max.(0, Int.(raw_data)))
unique_src = sort(unique(raw_uint16[raw_uint16 .> 0]))
println("  Source labels: $(length(unique_src)) unique, range $(minimum(unique_src))-$(maximum(unique_src))")

println("\n=== Step 2: Pre-flip Y axis (same as preprocessing) ===")
atlas_fixed = reverse(raw_uint16, dims=2)
println("  Atlas fixed size: $(size(atlas_fixed))")

println("\n=== Step 3: Load label mapping ===")
ts_names = Dict{Int,String}(parse(Int, k) => v for (k, v) in JSON.parsefile(labels_path))
println("  Label mapping: $(length(ts_names)) entries")

println("\n=== Step 4: Verify anatomy makes sense ===")
# Check sternum location
sternum_id = findfirst(v -> v == "sternum", ts_names)
if sternum_id !== nothing
    sternum_locs = findall(x -> x == sternum_id, atlas_fixed)
    if !isempty(sternum_locs)
        sz = [l[3] for l in sternum_locs]
        println("  Sternum (label $sternum_id): z=$(minimum(sz))-$(maximum(sz)), $(length(sternum_locs)) voxels")
    end
end

# Check skull
skull_id = findfirst(v -> v == "skull", ts_names)
if skull_id !== nothing
    skull_locs = findall(x -> x == skull_id, atlas_fixed)
    if !isempty(skull_locs)
        sz = [l[3] for l in skull_locs]
        println("  Skull (label $skull_id): z=$(minimum(sz))-$(maximum(sz)), $(length(skull_locs)) voxels")
    end
end

# Check liver
liver_id = findfirst(v -> v == "liver", ts_names)
if liver_id !== nothing
    liver_locs = findall(x -> x == liver_id, atlas_fixed)
    if !isempty(liver_locs)
        sz = [l[3] for l in liver_locs]
        println("  Liver (label $liver_id): z=$(minimum(sz))-$(maximum(sz)), $(length(liver_locs)) voxels")
    end
end

println("\n=== Step 5: Write fixed atlas to HDF5 ===")
# Need to add LesionAssociation
push!(LOAD_PATH, "/workspaces/MedEye3d.jl/src")
using MedEye3d
using MedEye3d.LesionAssociation

h5open(h5_path, "r+") do h5
    # Overwrite ATLAS/max_anatomy
    if haskey(h5, "ATLAS/max_anatomy")
        delete_object(h5, "ATLAS/max_anatomy")
    end
    h5["ATLAS/max_anatomy"] = atlas_fixed
    println("  Wrote ATLAS/max_anatomy ($(size(atlas_fixed)))")
    
    # Overwrite BASELINE/max_anatomy.nii.gz  
    if haskey(h5, "BASELINE/max_anatomy.nii.gz")
        delete_object(h5, "BASELINE/max_anatomy.nii.gz")
    end
    h5["BASELINE/max_anatomy.nii.gz"] = atlas_fixed
    println("  Wrote BASELINE/max_anatomy.nii.gz ($(size(atlas_fixed)))")
    
    # Also recompute bone_atlas from the fixed atlas
    bone_keywords = ["femur", "hip", "sacrum", "skull", "sternum", "scapula", 
        "vertebrae", "rib", "clavicula", "humerus", "mandible", "patella",
        "radius", "tibia", "ulna", "costal", "ilium", "ischium", "pubis"]
    bone_label_ids = Set{UInt16}()
    for (k, v) in ts_names
        if any(bw -> occursin(bw, lowercase(v)), bone_keywords)
            push!(bone_label_ids, UInt16(k))
        end
    end
    bone_atlas = Float32.(in.(atlas_fixed, Ref(bone_label_ids)))
    if haskey(h5, "ATLAS/bone_atlas")
        delete_object(h5, "ATLAS/bone_atlas")
    end
    h5["ATLAS/bone_atlas"] = bone_atlas
    println("  Wrote ATLAS/bone_atlas ($(Int(sum(bone_atlas))) bone voxels)")
    
    println("\n=== Step 6: Recompute organ_mapping for all TPs ===")
    # Map TP index to (group, mask_key)
    tp_masks = Dict{Int, Tuple{String,String}}()
    
    # TP 0 = BASELINE
    for k in keys(h5["BASELINE"])
        if occursin("Lesion", k) && occursin("expert", k)
            tp_masks[0] = ("BASELINE", k)
        elseif occursin("Lesion", k) && !occursin("expert", k) && !haskey(tp_masks, 0)
            tp_masks[0] = ("BASELINE", k)
        end
    end
    
    # Other TPs from TFM groups - use segment_names to know which TP index
    sn = JSON.parse(read(h5["_meta_/segment_names.json"]))
    
    for grp_name in keys(h5)
        if !startswith(grp_name, "TFM_"); continue; end
        for k in keys(h5[grp_name])
            if occursin("Lesion", k)
                # Find TP index from segment_names
                for (tp_str, tp_sn) in sn
                    tp_idx = parse(Int, tp_str)
                    tp_idx == 0 && continue  # already handled
                    # Check if this mask matches the TP
                    sn_values = values(tp_sn)
                    for sv in sn_values
                        if occursin(k[1:min(10,length(k))], sv) || occursin(split(k, ".")[1], sv)
                            tp_masks[tp_idx] = (grp_name, k)
                            break
                        end
                    end
                end
                # Fallback: extract TP from mask name pattern like PET_Lesions_1.nii.gz
                m = match(r"(\w+)_Lesions_(\d+)", k)
                if m !== nothing
                    # Map based on study config order
                    tp_masks_keys = sort(collect(keys(tp_masks)))
                    if !any(v -> v[2] == k, values(tp_masks))
                        # Assign to next available TP
                    end
                end
            end
        end
    end
    
    # Simpler approach: just iterate ALL groups and compute organ mapping for each mask
    all_organ_mappings = Dict{String, Dict{String, String}}()
    
    # TP 0
    mask_key = haskey(h5, "BASELINE/PET_Lesions_0.nii.gz_expert") ? "BASELINE/PET_Lesions_0.nii.gz_expert" : "BASELINE/PET_Lesions_0.nii.gz"
    if haskey(h5, mask_key)
        mask = Float32.(read(h5[mask_key]))
        organ_map = LesionAssociation.map_lesions_to_organs(mask, atlas_fixed, ts_names)
        tp_map = Dict{String,String}(string(k) => v for (k,v) in organ_map)
        all_organ_mappings["0"] = tp_map
        println("  TP 0: $(length(organ_map)) lesions")
        for (lid, name) in sort(collect(organ_map), by=x->x[1])
            println("    Lesion $lid: $name")
        end
    end
    
    # All TFM TPs (1-7)
    tfm_groups = sort([k for k in keys(h5) if startswith(k, "TFM_")])
    for (idx, grp) in enumerate(tfm_groups)
        tp_idx = idx  # TPs 1-7
        for k in keys(h5[grp])
            if occursin("Lesion", k)
                mask = Float32.(read(h5["$grp/$k"]))
                organ_map = LesionAssociation.map_lesions_to_organs(mask, atlas_fixed, ts_names)
                tp_map = Dict{String,String}(string(k2) => v for (k2,v) in organ_map)
                all_organ_mappings[string(tp_idx)] = tp_map
                println("  TP $tp_idx ($k): $(length(organ_map)) lesions")
            end
        end
    end
    
    # Save
    if haskey(h5, "_meta_/organ_mapping")
        delete_object(h5, "_meta_/organ_mapping")
    end
    h5["_meta_/organ_mapping"] = JSON.json(all_organ_mappings)
    println("\n  Saved organ_mapping for $(length(all_organ_mappings)) TPs")
end

println("\n=== DONE ===")
