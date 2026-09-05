module LesionTracker

using JSON
using Statistics
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
    
    return mean(values), maximum(values), std(values), true
end

"""
    track_lesions(data_dir; matches_json, output_path)

Main entry point. Reads matches.json, loads volumes, computes per-lesion stats,
writes JSON report. Returns Vector{LesionTrackingEntry}.
"""
function track_lesions(
    data_dir::String;
    matches_json::String = joinpath(data_dir, "matches.json"),
    output_path::String  = joinpath(data_dir, "lesion_tracking_report.json")
)::Vector{LesionTrackingEntry}
    
    if !isfile(matches_json)
        error("matches.json not found at $matches_json")
    end
    
    data = JSON.parse(read(matches_json, String))
    
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
            
            # Load images
            func_path = _functional_image_path(data_dir, node_name)
            mask_path, ct_path = _mask_and_ct_paths(data_dir, node_name)
            
            if func_path === nothing || mask_path === nothing
                # Can't process
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
    
    open(output_path, "w") do f
        JSON.print(f, json_entries, 2)
    end
    
    return entries
end

end
