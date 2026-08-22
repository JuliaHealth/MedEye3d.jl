module LesionAssociation

using JSON
using Statistics
using CodecZlib

const ASSOC_JSON_PATH = joinpath(homedir(), "medeye3d_lesion_associations.json")
const OVERLAP_MAPPING = Dict{Tuple{String, String, String}, Vector{String}}()

# Cross-TP match groups loaded from matches.json
# group_id → Vector{(node_name, segment_int_value, lesion_display_name)}
const MATCH_GROUPS = Dict{Int, Vector{Tuple{String, Int, String}}}()

export load_associations, save_associations, get_children, map_link
export parse_nrrd_segment_names, load_matches_json, get_match_groups
export find_cross_tp_lesion, get_group_id_for_lesion

"""
Parse NRRD seg.nrrd header to extract LabelValue → Name mapping.
Returns Dict{Int, String} where key=label integer, value=segment name.
"""
function parse_nrrd_segment_names(nrrd_path::String)::Dict{Int, String}
    label_values = Dict{Int, Int}()   # seg_idx → label_value
    names = Dict{Int, String}()       # seg_idx → name
    
    if !isfile(nrrd_path)
        @warn "NRRD file not found: $nrrd_path"
        return Dict{Int, String}()
    end
    
    open(nrrd_path, "r") do f
        for line in eachline(f)
            # NRRD header ends at first blank line
            isempty(strip(line)) && break
            
            # Match SegmentN_LabelValue:=M
            m_lv = match(r"Segment(\d+)_LabelValue:=(\d+)", line)
            if m_lv !== nothing
                seg_idx = parse(Int, m_lv.captures[1])
                label_val = parse(Int, m_lv.captures[2])
                label_values[seg_idx] = label_val
            end
            
            # Match SegmentN_Name:=<name>
            m_name = match(r"Segment(\d+)_Name:=(.+)", line)
            if m_name !== nothing
                seg_idx = parse(Int, m_name.captures[1])
                names[seg_idx] = strip(m_name.captures[2])
            end
        end
    end
    
    # Merge: build LabelValue → Name
    result = Dict{Int, String}()
    for (seg_idx, label_val) in label_values
        if haskey(names, seg_idx)
            result[label_val] = names[seg_idx]
        end
    end
    @info "Parsed $(length(result)) segment names from $(basename(nrrd_path))"
    return result
end

"""
Load matches.json and build cross-TP match groups.
Each group has a group_id and contains all matched lesions across time points.

Returns the populated MATCH_GROUPS dict.
Format: group_id → Vector{(node_name, segment_int_value, lesion_display_name)}
"""
function load_matches_json(json_path::String)
    empty!(MATCH_GROUPS)
    
    if !isfile(json_path)
        @warn "matches.json not found: $json_path"
        return MATCH_GROUPS
    end
    
    try
        data = JSON.parse(read(json_path, String))
        
        for root in data
            # group_id might be at root or inside the first child
            gid = get(root, "group_id", 0)
            if gid == 0 && haskey(root, "children") && !isempty(root["children"])
                gid = get(root["children"][1], "group_id", 0)
            end
            
            if gid == 0
                continue
            end
            
            if !haskey(MATCH_GROUPS, gid)
                MATCH_GROUPS[gid] = Tuple{String, Int, String}[]
            end
            
            # Parse the root/parent entry
            raw = get(root, "raw_lesion", "")
            if !isempty(raw)
                seg_int = parse(Int, replace(raw, "Segment_" => ""))
                name = get(root, "lesion", raw)
                node = get(root, "name", "")
                entry = (node, seg_int, name)
                if !(entry in MATCH_GROUPS[gid])
                    push!(MATCH_GROUPS[gid], entry)
                end
            end
            
            # Parse children
            for child in get(root, "children", [])
                c_raw = get(child, "raw_lesion", "")
                if isempty(c_raw)
                    continue
                end
                c_int = parse(Int, replace(c_raw, "Segment_" => ""))
                c_name = get(child, "lesion", c_raw)
                c_node = get(child, "name", "")
                c_entry = (c_node, c_int, c_name)
                if !(c_entry in MATCH_GROUPS[gid])
                    push!(MATCH_GROUPS[gid], c_entry)
                end
            end
        end
        
        @info "Loaded $(length(MATCH_GROUPS)) match groups from $(basename(json_path))"
    catch e
        @warn "Failed to load matches.json: $e"
    end
    
    return MATCH_GROUPS
end

"""
Get loaded match groups.
"""
get_match_groups() = MATCH_GROUPS

"""
Find the corresponding lesion in a target time point given a source lesion.
src_node_name: e.g. "PET_Lesions_0"
src_segment_int: e.g. 18 (the label value in the mask)
dst_node_name: e.g. "PET_Lesions_1"

Returns Vector{Int} of matching segment integers in the target TP, or empty.
"""
function find_cross_tp_lesion(src_node_name::String, src_segment_int::Int, dst_node_name::String)::Vector{Int}
    results = Int[]
    for (gid, members) in MATCH_GROUPS
        # Check if source is in this group
        src_match = any(m -> m[1] == src_node_name && m[2] == src_segment_int, members)
        if src_match
            # Find all matching entries in the destination node
            for (node, seg_int, name) in members
                if node == dst_node_name
                    push!(results, seg_int)
                end
            end
        end
    end
    return results
end

# ── Existing association functions (for manual overrides) ────────────────────

function load_associations()
    empty!(OVERLAP_MAPPING)
    if !isfile(ASSOC_JSON_PATH)
        return
    end
    try
        data = JSON.parse(read(ASSOC_JSON_PATH, String))
        for entry in data
            p_name = get(entry, "parent_name", "")
            p_sid = get(entry, "parent_lesion", "")
            c_name = get(entry, "child_name", "")
            children = get(entry, "children", [])
            
            if isempty(p_name) || isempty(p_sid) || isempty(c_name) || isempty(children)
                continue
            end
            
            c_ids = String[]
            for child in children
                cid = get(child, "child_lesion", "")
                if !isempty(cid)
                    push!(c_ids, cid)
                end
            end
            
            if !isempty(c_ids)
                fwd_key = (p_name, c_name, p_sid)
                OVERLAP_MAPPING[fwd_key] = c_ids
                
                for cid in c_ids
                    rev_key = (c_name, p_name, cid)
                    if !haskey(OVERLAP_MAPPING, rev_key)
                        OVERLAP_MAPPING[rev_key] = String[]
                    end
                    if !(p_sid in OVERLAP_MAPPING[rev_key])
                        push!(OVERLAP_MAPPING[rev_key], p_sid)
                    end
                end
            end
        end
    catch e
        @warn "Failed to load lesion associations: $e"
    end
end

function save_associations(p_name::String, c_name::String, p_sid::String, c_sids::Vector{String})
    data = []
    if isfile(ASSOC_JSON_PATH)
        try
            data = JSON.parse(read(ASSOC_JSON_PATH, String))
        catch
            data = []
        end
    end
    
    # Remove old map
    filter!(d -> !(d["parent_name"] == p_name && d["child_name"] == c_name && d["parent_lesion"] == p_sid), data)
    
    if !isempty(c_sids)
        children_arr = [Dict("child_lesion" => cid, "IoU" => 1.0, "manual" => true) for cid in c_sids]
        push!(data, Dict(
            "parent_name" => p_name,
            "parent_lesion" => p_sid,
            "child_name" => c_name,
            "children" => children_arr
        ))
    end
    
    open(ASSOC_JSON_PATH, "w") do f
        write(f, JSON.json(data, 4))
    end
    
    load_associations()
end

function get_children(p_name::String, c_name::String, p_sid::String)
    if isempty(OVERLAP_MAPPING)
        load_associations()
    end
    return get(OVERLAP_MAPPING, (p_name, c_name, p_sid), String[])
end

function map_link(tp1_name::String, tp2_name::String, lesion_id::String)
    # Manual linkage overriding any spatial IoU: 
    # Just link "lesion_id" to "lesion_id" across TPs
    save_associations(tp1_name, tp2_name, lesion_id, [lesion_id])
    @info "Explicitly mapped $lesion_id from $tp1_name to $tp2_name"
end

"""
Get the cross-TP match group ID for a given lesion name (e.g. "unknown_18_PET_0").
Returns the group_id as Int or nothing if not found.
"""
function get_group_id_for_lesion(lesion_name::String)::Union{Int, Nothing}
    for (gid, members) in MATCH_GROUPS
        for (node, seg_int, name) in members
            if name == lesion_name
                return gid
            end
        end
    end
    return nothing
end

"""
Load a .seg.nrrd file as a 3D UInt8 labelmap array.
Parses the NRRD header for sizes and encoding, then reads the gzip-compressed data.
Returns (volume::Array{UInt8,3}, sizes::Tuple{Int,Int,Int}) or (nothing, nothing) on failure.
"""
function load_nrrd_labelmap(nrrd_path::String)
    if !isfile(nrrd_path)
        @warn "NRRD file not found: $nrrd_path"
        return nothing, nothing
    end
    
    sizes = nothing
    encoding = ""
    dtype = ""
    header_end_offset = 0
    
    # Parse header
    open(nrrd_path, "r") do f
        for line in eachline(f)
            header_end_offset = position(f)
            stripped = strip(line)
            isempty(stripped) && break
            
            if startswith(stripped, "sizes:")
                parts = split(strip(split(stripped, ":"; limit=2)[2]))
                sizes = Tuple(parse.(Int, parts))
            elseif startswith(stripped, "encoding:")
                encoding = strip(split(stripped, ":"; limit=2)[2])
            elseif startswith(stripped, "type:")
                dtype = strip(split(stripped, ":"; limit=2)[2])
            end
        end
    end
    
    if sizes === nothing || length(sizes) < 3
        @warn "Could not parse NRRD sizes from $nrrd_path"
        return nothing, nothing
    end
    
    if encoding != "gzip"
        @warn "Only gzip encoding supported for NRRD, got: $encoding"
        return nothing, nothing
    end
    
    # Read raw data after header (after the blank line)
    try
        raw_bytes = open(nrrd_path, "r") do f
            # Skip header by reading until blank line
            for line in eachline(f)
                isempty(strip(line)) && break
            end
            # Read remaining bytes (gzip-compressed data)
            read(f)
        end
        
        # Decompress
        decompressed = CodecZlib.transcode(CodecZlib.GzipDecompressor, raw_bytes)
        
        expected_size = prod(sizes)
        if length(decompressed) < expected_size
            @warn "NRRD decompressed data too small: $(length(decompressed)) < $expected_size"
            return nothing, nothing
        end
        
        # Reshape to 3D array (Fortran order like Julia's column-major)
        volume = reshape(reinterpret(UInt8, decompressed[1:expected_size]), sizes)
        @info "Loaded NRRD labelmap: $(sizes) from $(basename(nrrd_path))"
        return volume, sizes
    catch e
        @error "Failed to load NRRD: $e"
        return nothing, nothing
    end
end

"""
Map each lesion in `lesion_mask` to its nearest organ in the TS atlas.
- `lesion_mask::Array{Float32,3}` - the lesion segmentation (from MedImages)
- `ts_atlas::Array{UInt8,3}` - the TotalSegmentator labelmap (from load_nrrd_labelmap)
- `ts_names::Dict{Int,String}` - label value → organ name (from parse_nrrd_segment_names)

Returns Dict{Int, String} mapping lesion segment int → organ name.
Both arrays must be in the SAME coordinate space (same resampling grid).
"""
function map_lesions_to_organs(lesion_mask::AbstractArray, ts_atlas::AbstractArray, ts_names::Dict{Int, String})
    result = Dict{Int, String}()
    
    unique_lesions = sort(unique(lesion_mask))
    lesion_ints = filter(x -> x > 0, unique_lesions)
    
    for lesion_val in lesion_ints
        seg_int = Int(lesion_val)
        
        # Find centroid of this lesion
        indices = findall(x -> x == lesion_val, lesion_mask)
        if isempty(indices)
            continue
        end
        
        # Compute centroid
        cx_raw = mean(i[1] for i in indices)
        cy_raw = mean(i[2] for i in indices)
        cz_raw = mean(i[3] for i in indices)
        
        # Scale to atlas dimensions
        scale_x = size(ts_atlas, 1) / size(lesion_mask, 1)
        scale_y = size(ts_atlas, 2) / size(lesion_mask, 2)
        scale_z = size(ts_atlas, 3) / size(lesion_mask, 3)
        
        cx = round(Int, cx_raw * scale_x)
        cy = round(Int, cy_raw * scale_y)
        cz = round(Int, cz_raw * scale_z)
        
        # Clamp to atlas bounds
        cx = clamp(cx, 1, size(ts_atlas, 1))
        cy = clamp(cy, 1, size(ts_atlas, 2))
        cz = clamp(cz, 1, size(ts_atlas, 3))
        
        # Look up TS atlas at centroid (search expanding neighborhood)
        organ_name = ""
        for radius in [0, 1, 2, 4, 8]
            for dx in -radius:radius
                for dy in -radius:radius
                    for dz in -radius:radius
                        if dx*dx + dy*dy + dz*dz > radius*radius
                            continue
                        end
                        nx = clamp(cx + dx, 1, size(ts_atlas, 1))
                        ny = clamp(cy + dy, 1, size(ts_atlas, 2))
                        nz = clamp(cz + dz, 1, size(ts_atlas, 3))
                        ts_val = Int(ts_atlas[nx, ny, nz])
                        if ts_val > 0 && haskey(ts_names, ts_val)
                            organ_name = ts_names[ts_val]
                            break
                        end
                    end
                    !isempty(organ_name) && break
                end
                !isempty(organ_name) && break
            end
            !isempty(organ_name) && break
        end
        
        if !isempty(organ_name)
            result[seg_int] = organ_name
        end
    end
    
    @info "Mapped $(length(result))/$(length(lesion_ints)) lesions to organs"
    return result
end

"""
    classify_organ_to_lesion_type(organ_name::String) → String

Classify a TotalSegmentator organ name into a lesion type category.
Returns one of: "Prostate", "Bone Meta", "Lymph Node Meta", "Organ Meta".

Mirrors the Slicer extension's categorization logic (LesionMetadata.py L4258-4266):
- Prostate → "Prostate"
- Bone/vertebra/rib/femur/... (excluding vascular) → "Bone Meta"  
- Lymph node → "Lymph Node Meta"
- Everything else → "Organ Meta"
"""
function classify_organ_to_lesion_type(organ_name::String)::String
    org = lowercase(organ_name)
    
    # Bone keywords — TotalSegmentator segment names that indicate skeletal structures
    bone_kws = ["femur", "hip", "vertebra", "rib", "sacrum", "clavicula", "clavicle",
                "humerus", "scapula", "sternum", "skull", "palate", "bone", "spine",
                "ilium", "ischium", "pubis", "tibia", "radius", "carpal", "tarsal",
                "costal_cartilage"]
    
    # Vascular exclusions — some TS names share bone keywords (e.g. "iliac_artery")
    vascular_exclusions = ["vena", "artery", "vein", "vessel", "trunk"]
    
    if occursin("prostate", org)
        return "Prostate"
    elseif any(kw -> occursin(kw, org), bone_kws) && !any(v -> occursin(v, org), vascular_exclusions)
        return "Bone Meta"
    elseif occursin("lymph", org) || occursin("node", org)
        return "Lymph Node Meta"
    else
        return "Organ Meta"
    end
end

export load_nrrd_labelmap, map_lesions_to_organs, classify_organ_to_lesion_type

end # module
