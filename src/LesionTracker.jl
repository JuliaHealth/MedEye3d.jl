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
    match_type::String         # "parent", "IoU", "PROXIMITY", "ARTIFICIAL"
    iou::Float64               # 0.0 for parent entries
    mean_intensity::Float64    # mean SUV/NM in masked region
    max_intensity::Float64
    std_intensity::Float64
    has_intensity::Bool        # false when functional image not found
end

# JSON serialization support for custom struct
JSON.lower(e::LesionTrackingEntry) = Dict(
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
)

"""
    track_lesions(data_dir; matches_json, output_path, modality_hint)

Main entry point. Reads matches.json, loads volumes, computes per-lesion stats,
writes JSON report. Returns Vector{LesionTrackingEntry}.

# Arguments
- `data_dir::String`       — directory containing all .nii.gz files
- `matches_json::String`   — path to matches.json (default: data_dir/matches.json)
- `output_path::String`    — output JSON path (default: data_dir/lesion_tracking_report.json)
"""
function track_lesions(
    data_dir::String;
    matches_json::String = joinpath(data_dir, "matches.json"),
    output_path::String  = joinpath(data_dir, "lesion_tracking_report.json")
)::Vector{LesionTrackingEntry}

    entries = LesionTrackingEntry[]
    
    if !isfile(matches_json)
        @warn "matches.json not found at $matches_json"
        return entries
    end
    
    match_groups = JSON.parse(read(matches_json, String))
    @info "Loaded $(length(match_groups)) match groups from $matches_json"
    
    cache = Dict{String, MedImage}()
    
    for group in match_groups
        gid = get(group, "group_id", 0)
        
        # Collect all lesions in this group (parent + children)
        lesions_in_group = [group]
        append!(lesions_in_group, get(group, "children", []))
        
        for lesion_data in lesions_in_group
            node_name = get(lesion_data, "name", "")
            raw_lesion = get(lesion_data, "raw_lesion", "")
            lesion_name = get(lesion_data, "lesion", "")
            vol_mm3 = get(lesion_data, "volume_mm3", 0.0)
            
            # Parent node doesn't have match_type and IoU in matches.json
            match_type = get(lesion_data, "match_type", "parent")
            iou = get(lesion_data, "IoU", 0.0)
            
            if isempty(node_name) || isempty(raw_lesion)
                continue
            end
            
            # Parse segment_int
            m = match(r"Segment_(\d+)", raw_lesion)
            segment_int = m !== nothing ? parse(Int, m.captures[1]) : 0
            
            # Determine functional and mask paths
            mask_path, func_path, ct_path = _get_paths(data_dir, node_name)
            
            mean_intensity = 0.0
            max_intensity = 0.0
            std_intensity = 0.0
            has_intensity = false
            
            if isfile(mask_path) && isfile(func_path) && isfile(ct_path)
                try
                    # We use the ct_path as the reference space. We load it as-is.
                    ct_img = _load_cached(cache, ct_path, "CT", nothing, nothing)
                    
                    # We load mask and func, but request them to be resampled to the CT image space
                    mask_img = _load_cached(cache, mask_path, "MASK", ct_img, MedImages.Nearest_neighbour_en)
                    func_img = _load_cached(cache, func_path, "FUNC", ct_img, MedImages.Linear_en)
                    
                    # Compute stats (they are already resampled)
                    mean_i, max_i, std_i = _compute_stats(func_img, mask_img, segment_int)
                    
                    mean_intensity = mean_i
                    max_intensity = max_i
                    std_intensity = std_i
                    has_intensity = true
                catch e
                    @warn "Failed to compute stats for $node_name $raw_lesion: $e"
                end
            else
                @warn "Missing files for $node_name: mask=$mask_path, func=$func_path, ct=$ct_path"
            end
            
            push!(entries, LesionTrackingEntry(
                gid,
                node_name,
                lesion_name,
                raw_lesion,
                segment_int,
                vol_mm3,
                match_type,
                iou,
                mean_intensity,
                max_intensity,
                std_intensity,
                has_intensity
            ))
        end
    end
    
    # Sort entries by group_id, then node_name
    sort!(entries, by = x -> (x.group_id, x.node_name))
    
    # Save to JSON
    open(output_path, "w") do io
        JSON.print(io, entries, 2)
    end
    @info "Report saved to $output_path"
    
    return entries
end

function _get_paths(data_dir, node_name)
    m = match(r"(PET|SPECT)_Lesions_(\d+)", node_name)
    if m !== nothing
        modality = m.captures[1]
        idx = m.captures[2]
        
        mask_path = joinpath(data_dir, "$(node_name).nii.gz")
        
        if modality == "PET"
            func_path = joinpath(data_dir, "SUV_PET_Image_$(idx).nii.gz")
            ct_path = joinpath(data_dir, "Fixed_CT_Volume_$(idx).nii.gz")
        else # SPECT
            func_path = joinpath(data_dir, "SPECT_NM_Vendor_Volume_$(idx).nii.gz")
            ct_path = joinpath(data_dir, "SPECT_CT_Volume_$(idx).nii.gz")
        end
        
        return mask_path, func_path, ct_path
    end
    
    return "", "", ""
end

function _load_cached(cache::Dict{String, MedImage}, path::String, modality::String, ref_img::Union{MedImage, Nothing}, interp_method)
    cache_key = path * (ref_img !== nothing ? "_resampled" : "")
    if haskey(cache, cache_key)
        return cache[cache_key]
    end
    img = MedImages.load_image(path, modality)
    
    if ref_img !== nothing
        @info "Resampling $path to reference CT..."
        img = MedImages.resample_to_image(ref_img, img, interp_method)
    end
    
    cache[cache_key] = img
    return img
end

function _compute_stats(resampled_func::MedImage, resampled_mask::MedImage, label_int::Int)
    mask_data = resampled_mask.voxel_data
    func_data = resampled_func.voxel_data
    
    target_val = convert(eltype(mask_data), label_int)
    
    # Fast vectorized extraction
    intensities = Float64.(func_data[mask_data .== target_val])
    
    if isempty(intensities)
        return 0.0, 0.0, 0.0
    end
    
    return sum(intensities)/length(intensities), maximum(intensities), std(intensities)
end

end # module
