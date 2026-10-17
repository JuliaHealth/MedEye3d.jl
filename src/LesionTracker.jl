module LesionTracker

using JSON
using Statistics
using HDF5
using MedImages
using ..LesionAssociation

export track_lesions, LesionTrackingEntry

"""
One tracked observation: a specific lesion at a specific time point.
"""
struct LesionTrackingEntry
    group_id::Int
    node_name::String          # e.g. "PET_Lesions_1"
    lesion_name::String        # display name from matches.json
    raw_lesion::String         # "Segment_18"
    segment_int::Int           # 18
    volume_mm3::Float64
    match_type::String         # "parent","IoU","PROXIMITY","ARTIFICIAL"
    iou::Float64               # 0.0 for parent entries
    mean_intensity::Float64    # mean SUV/NM in masked region
    max_intensity::Float64
    std_intensity::Float64
    has_intensity::Bool        # false when functional image not found
end

# Derive functional image path from node name
function _functional_image_path(data_dir, node_name)
    m = match(r"^(PET|SPECT)_Lesions_(\d+)$", node_name)
    if m === nothing
        return nothing
    end
    modality, tp = m.captures[1], m.captures[2]
    if modality == "PET"
        return joinpath(data_dir, "SUV_PET_Image_$tp.nii.gz")
    elseif modality == "SPECT"
        return joinpath(data_dir, "SPECT_NM_Vendor_Volume_$tp.nii.gz")
    end
    return nothing
end

# Derive mask and CT paths
function _mask_and_ct_paths(data_dir, node_name)
    m = match(r"^(PET|SPECT)_Lesions_(\d+)$", node_name)
    if m === nothing
        return nothing, nothing
    end
    modality, tp = m.captures[1], m.captures[2]
    mask_path = joinpath(data_dir, "$(modality)_Lesions_$tp.nii.gz")
    
    ct_path = if modality == "PET"
        joinpath(data_dir, "Fixed_CT_Volume_$tp.nii.gz")
    else
        joinpath(data_dir, "SPECT_CT_Volume_$tp.nii.gz")
    end
    
    return mask_path, ct_path
end

# Load and cache volumes (avoid reloading same file twice)
function _load_cached(cache, path, is_mask=false)
    haskey(cache, path) && return cache[path]
    if !isfile(path)
        return nothing
    end
    
    img = MedImages.load_image(path)
    cache[path] = img
    return img
end

# Load functional and mask 3D arrays from HDF5 for a given node_name (preferring _expert mask)
function _load_h5_cached_arrays(cache, h5_path::String, node_name::String)
    cache_key = "h5::$(h5_path)::$(node_name)"
    haskey(cache, cache_key) && return cache[cache_key]
    m = match(r"^(PET|SPECT)_Lesions_(\d+)$", node_name)
    modality = m !== nothing ? String(m.captures[1]) : "PET"
    tp = m !== nothing ? String(m.captures[2]) : "0"
    func_name = modality == "PET" ? "SUV_PET_Image_$tp" : "SPECT_NM_Vendor_Volume_$tp"
    mask_name = m !== nothing ? "$(modality)_Lesions_$tp" : node_name
    expert_name = "$(mask_name)_expert"
    try
        res = HDF5.h5open(h5_path, "r") do f
            target_grp = nothing
            for cand in ("timepoint_$tp", tp == "0" ? "BASELINE" : "TFM_$tp", "BASELINE")
                if haskey(f, cand) && (haskey(f[cand], expert_name) || haskey(f[cand], mask_name))
                    target_grp = f[cand]
                    break
                end
            end
            if target_grp === nothing
                for k in keys(f)
                    k in ("_meta_", "ATLAS", "CENTROIDS", "BONE_SUBSEG") && continue
                    g = f[k]
                    if g isa HDF5.Group && (haskey(g, expert_name) || haskey(g, mask_name))
                        target_grp = g
                        break
                    end
                end
            end
            target_grp === nothing && return (nothing, nothing)
            func_arr = if haskey(target_grp, func_name)
                read(target_grp[func_name])
            else
                found_f = nothing
                for dk in keys(target_grp)
                    if startswith(dk, "SUV_PET_Image") || startswith(dk, "SPECT_NM_Vendor_Volume")
                        found_f = read(target_grp[dk])
                        break
                    end
                end
                found_f
            end
            mask_arr = if haskey(target_grp, expert_name)
                read(target_grp[expert_name])
            elseif haskey(target_grp, mask_name)
                read(target_grp[mask_name])
            else
                nothing
            end
            return (func_arr, mask_arr)
        end
        cache[cache_key] = res
        return res
    catch
        return (nothing, nothing)
    end
end

function _compute_stats_arrays(func_data::AbstractArray, mask_data::AbstractArray, label_int::Int)
    size(func_data) == size(mask_data) || return (0.0, 0.0, 0.0, false)
    mask_idx = findall(x -> round(Int, x) == label_int, mask_data)
    isempty(mask_idx) && return (0.0, 0.0, 0.0, false)
    values = Float64[Float64(func_data[i]) for i in mask_idx if !isnan(func_data[i])]
    isempty(values) && return (0.0, 0.0, 0.0, false)
    return mean(values), maximum(values), (length(values) > 1 ? std(values) : 0.0), true
end

# Compute intensity stats for a given lesion label within a volume
function _compute_stats(functional_vol, mask_vol, label_int, ct_vol)
    # resample functional → mask grid if needed
    if size(functional_vol.voxel_data) != size(mask_vol.voxel_data)
        functional_resampled = MedImages.resample_to_image(mask_vol, functional_vol, MedImages.Linear_en)
    else
        functional_resampled = functional_vol
    end
    
    mask_data = mask_vol.voxel_data
    func_data = functional_resampled.voxel_data
    
    # Build mask
    mask_idx = findall(x -> round(Int, x) == label_int, mask_data)
    
    if isempty(mask_idx)
        return 0.0, 0.0, 0.0, false
    end
    
    values = func_data[mask_idx]
    values = filter(x -> !isnan(x), values)
    
    if isempty(values)
        return 0.0, 0.0, 0.0, false
    end
    
    return mean(values), maximum(values), (length(values) > 1 ? std(values) : 0.0), true
end

"""
    track_lesions(data_dir; matches_json, output_path, h5_path)

Main entry point. Reads matches from HDF5 (`_meta_/matches.json`) or `matches.json`,
loads volumes from HDF5 (preferring `_expert` masks) or NIfTI fallback, computes per-lesion stats,
and writes the report into HDF5 (`_meta_/lesion_tracking_report.json`) and `output_path`.
Returns Vector{LesionTrackingEntry}.
"""
function track_lesions(
    data_dir::String;
    matches_json::String = isdir(data_dir) ? joinpath(data_dir, "matches.json") : "",
    output_path::String  = isdir(data_dir) ? joinpath(data_dir, "lesion_tracking_report.json") : "",
    h5_path::String      = (endswith(lowercase(data_dir), ".h5") || endswith(lowercase(data_dir), ".hdf5")) ? data_dir : joinpath(data_dir, "preprocessed_volumes.h5")
)::Vector{LesionTrackingEntry}
    
    data = nothing
    if !isempty(h5_path) && isfile(h5_path)
        try
            HDF5.h5open(h5_path, "r") do f
                if haskey(f, "_meta_") && haskey(f["_meta_"], "matches.json")
                    data = JSON.parse(String(read(f["_meta_/matches.json"])))
                end
            end
        catch e
            @warn "Failed to read _meta_/matches.json from $h5_path: $e"
        end
    end
    if data === nothing
        if !isempty(matches_json) && isfile(matches_json)
            data = JSON.parse(read(matches_json, String))
        else
            error("matches.json not found in HDF5 ($h5_path) or at $matches_json")
        end
    end
    
    cache = Dict{String, Any}()
    entries = LesionTrackingEntry[]
    
    # data is an array of group roots
    for group_root in data
        group_id = group_root["group_id"]
        
        # Flatten all nodes in this group
        nodes = [group_root]
        if haskey(group_root, "children")
            append!(nodes, group_root["children"])
        end
        
        for node in nodes
            node_name = node["name"]
            raw_lesion = node["raw_lesion"]
            
            segment_int = parse(Int, replace(raw_lesion, "Segment_" => ""))
            lesion_name = get(node, "lesion", raw_lesion)
            volume_mm3 = Float64(get(node, "volume_mm3", 0.0))
            
            match_type = get(node, "match_type", "parent")
            iou = Float64(get(node, "IoU", 0.0))
            
            # 1. Prefer reading directly from HDF5 (including _expert mask edits)
            if !isempty(h5_path) && isfile(h5_path)
                func_arr, mask_arr = _load_h5_cached_arrays(cache, h5_path, node_name)
                if func_arr !== nothing && mask_arr !== nothing
                    mean_int, max_int, std_int, has_int = _compute_stats_arrays(func_arr, mask_arr, segment_int)
                    push!(entries, LesionTrackingEntry(
                        group_id, node_name, lesion_name, raw_lesion, segment_int, volume_mm3,
                        match_type, iou, mean_int, max_int, std_int, has_int
                    ))
                    continue
                end
            end

            # 2. Fallback to NIfTI files in data_dir
            func_path = _functional_image_path(data_dir, node_name)
            mask_path, ct_path = _mask_and_ct_paths(data_dir, node_name)
            
            if func_path === nothing || mask_path === nothing
                push!(entries, LesionTrackingEntry(
                    group_id, node_name, lesion_name, raw_lesion, segment_int, volume_mm3,
                    match_type, iou, 0.0, 0.0, 0.0, false
                ))
                continue
            end
            
            func_vol = _load_cached(cache, func_path)
            mask_vol = _load_cached(cache, mask_path, true)
            ct_vol = _load_cached(cache, ct_path)
            
            if func_vol === nothing || mask_vol === nothing || ct_vol === nothing
                push!(entries, LesionTrackingEntry(
                    group_id, node_name, lesion_name, raw_lesion, segment_int, volume_mm3,
                    match_type, iou, 0.0, 0.0, 0.0, false
                ))
                continue
            end
            
            mean_int, max_int, std_int, has_int = _compute_stats(func_vol, mask_vol, segment_int, ct_vol)
            
            push!(entries, LesionTrackingEntry(
                group_id, node_name, lesion_name, raw_lesion, segment_int, volume_mm3,
                match_type, iou, mean_int, max_int, std_int, has_int
            ))
        end
    end
    
    # Sort
    sort!(entries, by = x -> (x.group_id, x.node_name))
    
    # Write JSON
    json_entries = map(e -> Dict(
        "group_id" => e.group_id,
        "node_name" => e.node_name,
        "lesion_name" => e.lesion_name,
        "raw_lesion" => e.raw_lesion,
        "segment_int" => e.segment_int,
        "volume_mm3" => e.volume_mm3,
        "match_type" => e.match_type,
        "iou" => e.iou,
        "mean_intensity" => e.mean_intensity,
        "max_intensity" => e.max_intensity,
        "std_intensity" => e.std_intensity,
        "has_intensity" => e.has_intensity
    ), entries)
    
    if !isempty(h5_path) && isfile(h5_path)
        try
            HDF5.h5open(h5_path, "r+") do f
                meta_grp = haskey(f, "_meta_") ? f["_meta_"] : create_group(f, "_meta_")
                if haskey(meta_grp, "lesion_tracking_report.json")
                    delete_object(meta_grp, "lesion_tracking_report.json")
                end
                meta_grp["lesion_tracking_report.json"] = JSON.json(json_entries, 2)
            end
        catch e
            @warn "Failed to write lesion_tracking_report.json to HDF5 $h5_path: $e"
        end
    end

    if !isempty(output_path)
        try
            open(output_path, "w") do f
                JSON.print(f, json_entries, 2)
            end
        catch; end
    end
    
    return entries
end

end
