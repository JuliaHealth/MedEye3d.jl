module LesionAssociation

using JSON
using Statistics
using CodecZlib
using HDF5

const ASSOC_JSON_PATH = joinpath(homedir(), "medeye3d_lesion_associations.json")
const OVERLAP_MAPPING = Dict{Tuple{String, String, String}, Vector{String}}()

# Cross-TP match groups loaded from matches.json
# group_id → Vector{(node_name, segment_int_value, lesion_display_name)}
const MATCH_GROUPS = Dict{Int, Vector{Tuple{String, Int, String}}}()

export load_associations, save_associations, get_children, map_link
export parse_nrrd_segment_names, load_matches_json, load_matches_from_h5, save_matches_to_h5, get_match_groups
export find_cross_tp_lesion, get_group_id_for_lesion
export update_match_group!, remove_from_match_group!

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
        _parse_matches_data(data)
        @info "Loaded $(length(MATCH_GROUPS)) match groups from $(basename(json_path))"
    catch e
        @warn "Failed to load matches.json: $e"
    end
    
    return MATCH_GROUPS
end

"""
Load match groups from the 'matches.json' HDF5 string attribute.
HDF5 is the single source of truth after preprocessing.
"""
function load_matches_from_h5(h5_path::String)
    empty!(MATCH_GROUPS)
    if !isfile(h5_path)
        @warn "HDF5 file not found: $h5_path"
        return MATCH_GROUPS
    end
    try
        HDF5.h5open(h5_path, "r") do f
            if haskey(f, "_meta_") && haskey(f["_meta_"], "matches.json")
                json_str = read(f["_meta_/matches.json"])
                _parse_matches_data(JSON.parse(json_str))
                @info "Loaded $(length(MATCH_GROUPS)) match groups from HDF5 _meta_ dataset"
            else
                @warn "No _meta_/matches.json dataset in $h5_path"
            end
        end
    catch e
        @warn "Failed to load matches from HDF5: $e"
    end
    return MATCH_GROUPS
end

"""
Rebuild matches.json from MATCH_GROUPS and save as HDF5 dataset in _meta_ group.
"""
function save_matches_to_h5(h5_path::String)
    if !isfile(h5_path)
        @warn "HDF5 file not found for saving: $h5_path"
        return
    end
    output = _match_groups_to_json()
    json_str = JSON.json(output, 4)
    try
        HDF5.h5open(h5_path, "r+") do f
            if !haskey(f, "_meta_")
                create_group(f, "_meta_")
            end
            if haskey(f["_meta_"], "matches.json")
                delete_object(f["_meta_"], "matches.json")
            end
            f["_meta_/matches.json"] = json_str
        end
        @info "Saved $(length(MATCH_GROUPS)) match groups to HDF5 _meta_ dataset"
    catch e
        @warn "Failed to save matches to HDF5: $e"
    end
end

"""
Link src and dst lesions in a match group and persist to HDF5.
"""
function update_match_group!(src_node::String, src_seg_int::Int, dst_node::String, dst_seg_int::Int, h5_path::String)
    # Find all groups containing EITHER src OR dst
    matching_gids = Int[]
    for (gid, members) in MATCH_GROUPS
        if any(m -> (m[1] == src_node && m[2] == src_seg_int) || (m[1] == dst_node && m[2] == dst_seg_int), members)
            push!(matching_gids, gid)
        end
    end
    
    if isempty(matching_gids)
        new_gid = isempty(MATCH_GROUPS) ? 1 : maximum(keys(MATCH_GROUPS)) + 1
        MATCH_GROUPS[new_gid] = [
            (src_node, src_seg_int, "manual_$(src_seg_int)_$(src_node)"),
            (dst_node, dst_seg_int, "manual_$(dst_seg_int)_$(dst_node)")
        ]
    else
        # Merge all matching groups into the first one
        primary_gid = matching_gids[1]
        for i in 2:length(matching_gids)
            gid = matching_gids[i]
            for m in MATCH_GROUPS[gid]
                if !(m in MATCH_GROUPS[primary_gid])
                    push!(MATCH_GROUPS[primary_gid], m)
                end
            end
            delete!(MATCH_GROUPS, gid)
        end
        
        # Ensure src and dst are in the primary group
        src_entry = (src_node, src_seg_int, "manual_$(src_seg_int)_$(src_node)")
        dst_entry = (dst_node, dst_seg_int, "manual_$(dst_seg_int)_$(dst_node)")
        if !(src_entry in MATCH_GROUPS[primary_gid])
            push!(MATCH_GROUPS[primary_gid], src_entry)
        end
        if !(dst_entry in MATCH_GROUPS[primary_gid])
            push!(MATCH_GROUPS[primary_gid], dst_entry)
        end
    end
    
    save_matches_to_h5(h5_path)
end

"""
Remove a lesion from its match group and persist to HDF5.
"""
function remove_from_match_group!(node::String, seg_int::Int, h5_path::String)
    for (gid, members) in MATCH_GROUPS
        idx = findfirst(m -> m[1] == node && m[2] == seg_int, members)
        if idx !== nothing
            deleteat!(members, idx)
            if length(members) <= 1
                delete!(MATCH_GROUPS, gid)
            end
            break
        end
    end
    save_matches_to_h5(h5_path)
end

"""
Internal: Parse matches JSON data array into MATCH_GROUPS.
"""
function _parse_matches_data(data)
    for root in data
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
end

"""
Internal: Convert MATCH_GROUPS to JSON-serializable array.
"""
function _match_groups_to_json()
    output = []
    for (gid, members) in sort(collect(MATCH_GROUPS))
        if isempty(members); continue; end
        parent_idx = argmin([_tp_index_from_node(m[1]) for m in members])
        parent = members[parent_idx]
        children = [members[i] for i in 1:length(members) if i != parent_idx]
        
        root = Dict(
            "name" => parent[1],
            "raw_lesion" => "Segment_$(parent[2])",
            "node" => "parent",
            "lesion" => parent[3],
            "volume_mm3" => 0.0,
            "group_id" => gid,
            "children" => [
                Dict(
                    "name" => c[1],
                    "raw_lesion" => "Segment_$(c[2])",
                    "lesion" => c[3],
                    "volume_mm3" => 0.0,
                    "group_id" => gid
                ) for c in children
            ]
        )
        push!(output, root)
    end
    return output
end

function _tp_index_from_node(node::String)::Int
    parts = split(node, "_")
    tp = tryparse(Int, parts[end])
    return tp !== nothing ? tp : 999
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
    is_lymph_node_structure(name::String)::Bool

Determine whether a structure name represents a clinical lymph node area/station.
Checks known clinical station prefixes, substrings (lymph, node, knoten), and regional definitions.
"""
function is_lymph_node_structure(name::String)::Bool
    isempty(name) && return false
    ln = lowercase(name)
    
    # Substring checks
    (occursin("lymph", ln) || occursin("knoten", ln) || occursin("node", ln)) && return true
    
    # Clinical station prefixes and patterns in max_anatomy
    prefixes = ["neck_", "thoracic_station_", "axillary_", "abdominal_station_",
                "abdominal_mesenteric", "abdominal_obturator", "abdominal_presacral",
                "abdominal_pararectal", "abdominal_renal_hilar", "abdominal_common_iliac",
                "abdominal_external_iliac", "abdominal_internal_iliac", "abdominal_inguinal",
                "deep_inguinal", "superficial_inguinal", "inguinal_"]
    for p in prefixes
        startswith(ln, p) && return true
    end
    
    return false
end

"""
    is_pelvic_lymph_node(name::String)::Bool

Determine whether a lymph node station is a regional pelvic lymph node (miN1).
Includes internal iliac, external iliac, obturator, presacral, and pararectal stations.
"""
function is_pelvic_lymph_node(name::String)::Bool
    ln = lowercase(name)
    # Check regional pelvic nodal stations
    is_pelv = occursin("obturator", ln) || occursin("presacral", ln) || occursin("pararectal", ln) ||
              occursin("perirectal", ln) || occursin("internal_iliac", ln) || occursin("external_iliac", ln) ||
              (occursin("pelvic", ln) && occursin("lymph", ln))
    # Common iliac is extra-pelvic / distant (M1a) in prostate cancer staging
    if occursin("common_iliac", ln)
        return false
    end
    return is_pelv
end

"""
    classify_lymph_node_location(organ_name::String) -> Tuple{String, String}

Return (anatomic_location, anatomical_sublocation) corresponding to UI dropdown options.
- Regional pelvic nodes -> ("Pelvic Lymph Node", sublocation)
- Extra-pelvic distant nodes -> ("Distant Lymph Node (Common Iliac, Retroperitoneal, Inguinal, Supraclavicular, Axillary)", sublocation)
"""
function classify_lymph_node_location(organ_name::String)::Tuple{String, String}
    ln = lowercase(organ_name)
    if is_pelvic_lymph_node(organ_name)
        subloc = if occursin("obturator", ln)
            "Obturator"
        elseif occursin("external_iliac", ln)
            "External Iliac"
        elseif occursin("internal_iliac", ln)
            "Internal Iliac"
        elseif occursin("presacral", ln)
            "Presacral"
        elseif occursin("pararectal", ln) || occursin("perirectal", ln)
            "Perirectal"
        else
            "Intranodal Cortex (Asymmetric Gradient)"
        end
        return ("Pelvic Lymph Node", subloc)
    else
        subloc = if occursin("common_iliac", ln)
            "Common Iliac"
        elseif occursin("paraort", ln) || occursin("para-aort", ln) || occursin("aortic_hiatus", ln) || occursin("station_16", ln)
            "Retroperitoneal / Para-Aortic"
        elseif occursin("inguin", ln)
            "Inguinal"
        elseif occursin("supraclav", ln)
            "Supraclavicular"
        elseif occursin("axill", ln) || occursin("rotter", ln)
            "Axillary"
        elseif occursin("thoracic", ln) || occursin("prevascular", ln) || occursin("subcarin", ln) || occursin("subaortic", ln) || occursin("hilar", ln) || occursin("paraoesophageal", ln)
            "Anterior Mediastinum"
        elseif occursin("neck", ln) || occursin("cervical", ln) || occursin("jugular", ln)
            "Cervical / Neck"
        elseif occursin("mesenteric", ln) || occursin("celiac", ln) || occursin("pancreatic", ln) || occursin("gastric", ln) || occursin("hepatic", ln) || occursin("splenic", ln)
            "Abdominal / Mesenteric"
        else
            "Testicular Lymph Node Route (Para-aortic / Renal Hilum)"
        end
        return ("Distant Lymph Node (Common Iliac, Retroperitoneal, Inguinal, Supraclavicular, Axillary)", subloc)
    end
end

"""
    classify_tissue_priority(organ_name::String) → Int

Classify a structure name into a tissue priority class.
Lower number = higher priority for lesion naming.
  1 = Prostate, 2 = Lymph Node, 3 = Bone, 4 = Solid Organ, 5 = Vessel, 6 = Muscle/Soft Tissue
"""
function classify_tissue_priority(name::String)::Int
    ln = lowercase(name)
    
    # 1. Prostate gland primary site
    occursin("prostate", ln) && return 1
    
    # 2. Lymph node stations (takes priority over organs, vessels, and muscles)
    is_lymph_node_structure(name) && return 2
    
    # 3. Bone keywords — skeletal structures
    bone_kw = ["vertebra", "rib_", "femur", "humerus", "scapula_", "hip_",
               "sacrum", "skull", "sternum", "clavicle", "costal", "hyoid",
               "mandible", "styloid", "zygomatic", "cricoid", "thyroid_cartilage",
               "ilium", "ischium", "pubis", "tibia", "bone", "spine"]
    vessel_excl = ["artery", "vein", "vena", "vessel"]
    muscle_excl = ["levator", "subscapularis", "infraspinatus", "supraspinatus",
                   "teres", "coracobrachial"]
    
    if any(k -> occursin(k, ln), bone_kw) && !any(v -> occursin(v, ln), vessel_excl) && !any(m -> occursin(m, ln), muscle_excl)
        return 3  # Bone
    end
    
    # 4. Solid organs
    organ_kw = ["liver", "kidney", "lung", "spleen", "pancreas", "heart",
                "thyroid_gland", "adrenal", "stomach", "colon", "rectum",
                "esophagus", "gallbladder", "bladder", "bowel",
                "duodenum", "trachea", "brain", "spinal_cord", "eye",
                "parotid", "submandibular", "optic_nerve"]
    any(k -> occursin(k, ln), organ_kw) && return 4  # Solid organ
    
    # 5. Vessels
    vessel_kw = ["aorta", "artery", "vein", "vena", "trunk", "carotid", "jugular"]
    any(k -> occursin(k, ln), vessel_kw) && return 5  # Vessel
    
    return 6  # Muscle / soft tissue
end

"""
    count_atlas_overlap(mask, atlas, lid, ts_names) → Dict{Int,Int}

Count how many voxels of lesion `lid` in `mask` overlap each KNOWN atlas label.
Uses broadcasting — works on both CPU Arrays and GPU CuArrays.
Handles mismatched mask/atlas sizes via coordinate scaling.
"""
function count_atlas_overlap(mask::AbstractArray{<:Real,3}, atlas::AbstractArray{<:Real,3},
                             lid::Integer, ts_names::Dict{Int,String})::Dict{Int,Int}
    # Get lesion voxel indices (CPU-side, since we need coordinates)
    indices = findall(x -> x == eltype(mask)(lid), mask)
    isempty(indices) && return Dict{Int,Int}()
    
    # Scale factors from mask to atlas coordinates
    sx = size(atlas, 1) / size(mask, 1)
    sy = size(atlas, 2) / size(mask, 2)
    sz = size(atlas, 3) / size(mask, 3)
    
    counts = Dict{Int,Int}()
    for idx in indices
        ax = clamp(round(Int, idx[1] * sx), 1, size(atlas, 1))
        ay = clamp(round(Int, idx[2] * sy), 1, size(atlas, 2))
        az = clamp(round(Int, idx[3] * sz), 1, size(atlas, 3))
        v = Int(atlas[ax, ay, az])
        # Only count KNOWN labels (present in ts_names)
        if v > 0 && haskey(ts_names, v)
            counts[v] = get(counts, v, 0) + 1
        end
    end
    return counts
end

"""
    pick_best_organ(counts, ts_names) → String

Given atlas label counts, pick the best organ using clinical priority rules:
1. Prostate Primary: Any non-zero overlap in prostate gland.
2. Lymph Node Rule: Any non-zero overlap in a lymph node area takes absolute priority
   over adjacent organs, muscles, and vessels (even if abutting or partially in muscle/organ).
   Ties between lymph node stations are broken by voxel count.
3. Fallback: Standard priority (Bone > Solid Organ > Vessel > Muscle).
"""
function pick_best_organ(counts::Dict{Int,Int}, ts_names::Dict{Int,String})::String
    isempty(counts) && return ""
    
    # Rule 1: Prostate-wins rule for primary tumor inside prostate
    for (label_id, cnt) in counts
        cnt <= 0 && continue
        name = get(ts_names, label_id, "")
        if occursin("prostate", lowercase(name))
            return name
        end
    end
    
    # Rule 2: Lymph node rule: ANY overlap in a lymph node station takes priority
    # over adjacent organ, muscle, or vessel.
    ln_candidates = Tuple{String, Int}[]
    for (label_id, cnt) in counts
        cnt <= 0 && continue
        name = get(ts_names, label_id, "")
        isempty(name) && continue
        if is_lymph_node_structure(name)
            push!(ln_candidates, (name, cnt))
        end
    end
    
    if !isempty(ln_candidates)
        # Pick the lymph node station with the largest overlapping voxel count
        sort!(ln_candidates, by = x -> x[2], rev = true)
        return ln_candidates[1][1]
    end
    
    # Rule 3: General tissue priority fallback (Bone > Solid Organ > Vessel > Muscle)
    best_name = ""
    best_priority = 7
    best_count = 0
    
    for (label_id, cnt) in counts
        cnt <= 0 && continue
        name = get(ts_names, label_id, "")
        isempty(name) && continue
        
        priority = classify_tissue_priority(name)
        if priority < best_priority || (priority == best_priority && cnt > best_count)
            best_name = name
            best_priority = priority
            best_count = cnt
        end
    end
    return best_name
end

"""
    classify_and_pick_best_organ(mask, atlas, ts_names, lid) → String

Single-lesion volume-based organ lookup with bone priority.
Scans ALL voxels of lesion `lid`, counts overlapping known atlas labels,
and picks the best one using tissue priority (bone > organ > lymph > vessel > muscle).
"""
function classify_and_pick_best_organ(mask::AbstractArray{<:Real,3},
                                      atlas::AbstractArray{<:Real,3},
                                      ts_names::Dict{Int,String},
                                      lid::Integer)::String
    counts = count_atlas_overlap(mask, atlas, lid, ts_names)
    return pick_best_organ(counts, ts_names)
end

"""
Map each lesion in `lesion_mask` to its anatomical organ in the TS atlas.

Uses volume-based scanning with tissue priority (bone > organ > lymph > vessel > muscle):
1. Scans ALL voxels of each lesion for overlapping atlas labels
2. Filters to known labels only (present in `ts_names`)  
3. Picks the best label using tissue priority, breaking ties by voxel count
4. Falls back to expanding-sphere centroid search for zero-overlap lesions

Returns Dict{Int, String} mapping lesion ID → organ name.
"""
function map_lesions_to_organs(lesion_mask::AbstractArray, ts_atlas::AbstractArray, ts_names::Dict{Int, String})
    result = Dict{Int, String}()
    
    unique_lesions = sort(unique(lesion_mask))
    lesion_ints = filter(x -> x > 0, unique_lesions)
    
    for lesion_val in lesion_ints
        seg_int = Int(lesion_val)
        
        # Volume-based scan with bone priority
        organ_name = classify_and_pick_best_organ(lesion_mask, ts_atlas, ts_names, seg_int)
        
        if isempty(organ_name)
            # Fallback: expanding sphere from centroid for zero-overlap lesions
            indices = findall(x -> x == lesion_val, lesion_mask)
            if !isempty(indices)
                cx_raw = mean(i[1] for i in indices)
                cy_raw = mean(i[2] for i in indices)
                cz_raw = mean(i[3] for i in indices)
                
                scale_x = size(ts_atlas, 1) / size(lesion_mask, 1)
                scale_y = size(ts_atlas, 2) / size(lesion_mask, 2)
                scale_z = size(ts_atlas, 3) / size(lesion_mask, 3)
                
                cx = clamp(round(Int, cx_raw * scale_x), 1, size(ts_atlas, 1))
                cy = clamp(round(Int, cy_raw * scale_y), 1, size(ts_atlas, 2))
                cz = clamp(round(Int, cz_raw * scale_z), 1, size(ts_atlas, 3))
                
                for radius in [0, 1, 2, 4, 8, 16, 32]
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
            end
        end
        
        if !isempty(organ_name)
            result[seg_int] = organ_name
        end
    end
    
    @info "Mapped $(length(result))/$(length(lesion_ints)) lesions to organs (volume-based with bone priority)"
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
                "mandible", "hyoid", "styloid", "zygomatic"]
    
    # Muscle keywords — max_anatomy has 90 muscles
    muscle_kws = ["gluteus", "autochthon", "iliopsoas", "pectoralis", "subscapularis",
                  "supraspinatus", "infraspinatus", "latissimus", "rectus_abdominis",
                  "oblique", "erector", "trapezius", "deltoid", "sartorius", "quadriceps",
                  "scalene", "platysma", "masseter", "temporalis", "pterygoid",
                  "coracobrachial", "serratus", "teres_major", "triceps", "psoas",
                  "quadratus", "sternocleidomastoid", "pharyngeal", "prevertebral",
                  "tongue", "digastric", "thigh_medial", "thigh_posterior",
                  "levator_scapulae", "sterno_thyroid", "thyrohyoid", "transversospinalis"]
    
    # Vascular exclusions — some TS names share bone keywords (e.g. "iliac_artery")
    vascular_exclusions = ["vena", "artery", "vein", "vessel", "trunk"]
    
    is_muscle = any(kw -> occursin(kw, org), muscle_kws)
    is_bone = !is_muscle && any(kw -> occursin(kw, org), bone_kws) && !any(v -> occursin(v, org), vascular_exclusions)

    if occursin("prostate", org)
        return "Prostate"
    elseif is_lymph_node_structure(organ_name)
        return "Lymph Node Meta"
    elseif is_muscle
        return "Technical Artifact"
    elseif is_bone
        return "Bone Meta"
    else
        return "Organ Meta"
    end
end

"""
    format_clinical_station_name(raw_name::String) → String

Convert a raw max_anatomy lymph node station name (e.g. `"Abdominal_Obturator_Left"`,
`"Thoracic_Station_7_Subcarinial"`, `"Neck_Level_IIa_Upper_Jugular_Left"`)
into a clinician-friendly display name (e.g. `"Obturator Lymph Node (Left)"`,
`"Station 7 Subcarinal Lymph Node"`, `"Neck Level IIa Upper Jugular Lymph Node (Left)"`).
"""
function format_clinical_station_name(raw_name::String)::String
    isempty(raw_name) && return ""
    ln = lowercase(raw_name)

    # Determine side suffix
    side = if occursin("_left", ln) || endswith(ln, "left")
        " (Left)"
    elseif occursin("_right", ln) || endswith(ln, "right")
        " (Right)"
    else
        ""
    end

    # ── Pelvic / Abdominal non-station nodes ──
    occursin("obturator", ln) && return "Obturator Lymph Node$side"
    occursin("internal_iliac", ln) && return "Internal Iliac Lymph Node$side"
    occursin("external_iliac", ln) && return "External Iliac Lymph Node$side"
    occursin("common_iliac", ln) && return "Common Iliac Lymph Node$side"
    occursin("presacral", ln) && return "Presacral Lymph Node"
    (occursin("pararectal", ln) || occursin("perirectal", ln)) && return "Pararectal Lymph Node"
    occursin("mesenteric", ln) && return "Mesenteric Interenteric Lymph Node"
    occursin("renal_hilar", ln) && return "Renal Hilar Lymph Node$side"
    occursin("paraaortic", ln) && !occursin("station_", ln) && return "Para-Aortic Lymph Node"

    # ── Japanese Gastric Cancer Association (JGCA) abdominal stations ──
    occursin("station_10", ln) && return "Station 10 Splenic Hilum Lymph Node"
    occursin("station_11", ln) && return "Station 11 Splenic Artery Lymph Node"
    occursin("station_13", ln) && return "Station 13 Posterior Pancreaticoduodenal Lymph Node"
    occursin("station_14", ln) && return "Station 14 Superior Mesenteric Artery (SMA) Lymph Node"
    occursin("station_16a1", ln) && return "Station 16a1 Aortic Hiatus Lymph Node"
    occursin("station_16a2", ln) && return "Station 16a2 Upper Middle Paraaortic Lymph Node"
    occursin("station_16b1", ln) && return "Station 16b1 Lower Middle Paraaortic Lymph Node"
    occursin("station_16b2", ln) && return "Station 16b2 Caudal Paraaortic Lymph Node"
    occursin("station_17", ln) && return "Station 17 Anterior Pancreaticoduodenal Lymph Node"
    (occursin("station_1_2", ln) || occursin("station_1+2", ln)) && return "Station 1/2 Paracardial Lymph Node"
    occursin("station_1_right", ln) && return "Station 1 Right Paracardial Lymph Node"
    occursin("station_2_left", ln) && return "Station 2 Left Paracardial Lymph Node"
    occursin("station_3_lesser", ln) && return "Station 3 Lesser Curvature Lymph Node"
    occursin("station_4_greater", ln) && return "Station 4 Greater Curvature Lymph Node"
    occursin("station_5_6", ln) && return "Station 5/6 Pyloric Lymph Node"
    occursin("station_5_supra", ln) && return "Station 5 Suprapyloric Lymph Node"
    occursin("station_6_infra", ln) && return "Station 6 Infrapyloric Lymph Node"
    occursin("station_7_left_gastric", ln) && return "Station 7 Left Gastric Lymph Node"
    occursin("station_8_common_hepatic", ln) && return "Station 8 Common Hepatic Lymph Node"
    occursin("station_9_celiac", ln) && return "Station 9 Celiac Lymph Node"
    occursin("inferior_pancreatic", ln) && return "Inferior Pancreatic Lymph Node"

    # ── Inguinal nodes ──
    occursin("deep_inguinal", ln) && return "Deep Inguinal Lymph Node$side"
    occursin("superficial_inguinal", ln) && return "Superficial Inguinal Lymph Node$side"
    occursin("inguinal", ln) && return "Inguinal Lymph Node$side"

    # ── Axillary nodes ──
    (occursin("axillary_level_i_", ln) || endswith(ln, "axillary_level_i")) && return "Axillary Level I Lymph Node$side"
    occursin("axillary_level_ii_", ln) && return "Axillary Level II Lymph Node$side"
    occursin("axillary_level_iii", ln) && return "Axillary Level III Lymph Node$side"
    occursin("rotter", ln) && return "Rotter Interpectoral Lymph Node$side"
    occursin("axillary", ln) && return "Axillary Lymph Node$side"

    # ── Thoracic mediastinal stations (IASLC) ──
    occursin("mammary", ln) && return "Internal Mammary Lymph Node$side"
    if occursin("upperparatracheal", ln) || (occursin("station_2_", ln) && !occursin("left_paracardial", ln))
        st = occursin("left", ln) ? "2L" : (occursin("right", ln) ? "2R" : "2")
        return "Station $st Upper Paratracheal Lymph Node"
    end
    occursin("prevascular", ln) && return "Station 3A Prevascular Lymph Node$side"
    occursin("retrotracheal", ln) && return "Station 3P Retrotracheal Lymph Node$side"
    if occursin("lowerparatracheal", ln) || (occursin("station_4_", ln) && occursin("paratracheal", ln))
        st = occursin("left", ln) ? "4L" : (occursin("right", ln) ? "4R" : "4")
        return "Station $st Lower Paratracheal Lymph Node"
    end
    occursin("subaortic", ln) && return "Station 5 Subaortic (AP Window) Lymph Node"
    (occursin("station_6_paraaortic", ln) || (occursin("station_6", ln) && occursin("paraaortic", ln))) && return "Station 6 Paraaortic (Ascending Aorta) Lymph Node"
    (occursin("station_7", ln) || occursin("subcarin", ln)) && return "Station 7 Subcarinal Lymph Node"
    (occursin("station_8", ln) || occursin("paraoesophageal", ln)) && return "Station 8 Paraesophageal Lymph Node$side"
    (occursin("hilar", ln) || occursin("interlobar", ln)) && return "Station 10/11 Hilar / Interlobar Lymph Node$side"
    occursin("prepericardial", ln) && return "Prepericardial Lymph Node$side"
    occursin("supraclavicular", ln) && return "Supraclavicular Lymph Node$side"
    occursin("chest_wall", ln) && return "Chest Wall Lymph Node$side"

    # ── Neck levels ──
    (occursin("level_ia", ln) || occursin("submental", ln)) && return "Neck Level Ia Submental Lymph Node"
    (occursin("level_ib", ln) || occursin("submandibular", ln)) && return "Neck Level Ib Submandibular Lymph Node$side"
    occursin("level_iia", ln) && return "Neck Level IIa Upper Jugular Lymph Node$side"
    occursin("level_iib", ln) && return "Neck Level IIb Upper Jugular Lymph Node$side"
    (occursin("level_iii", ln) || occursin("middle_jugular", ln)) && return "Neck Level III Middle Jugular Lymph Node$side"
    (occursin("level_iv", ln) || occursin("lower_jugular", ln)) && return "Neck Level IV Lower Jugular Lymph Node$side"
    (occursin("level_va", ln) || occursin("posterior_triangle", ln)) && return "Neck Level Va Upper Posterior Triangle Lymph Node$side"
    (occursin("level_vi", ln) || occursin("anterior_cervical", ln)) && return "Neck Level VI Anterior Cervical Lymph Node"
    (occursin("level_xb", ln) || occursin("occipital", ln)) && return "Neck Level Xb Occipital Lymph Node$side"
    occursin("parotid", ln) && return "Parotid Lymph Node$side"
    occursin("retropharyngeal", ln) && return "Retropharyngeal Lymph Node"

    # ── Fallback: titlecase with underscores → spaces ──
    return titlecase(replace(strip(raw_name), "_" => " "))
end

"""
    format_adjacent_structure_name(raw_name::String) → String

Convert a raw TotalSegmentator / max_anatomy structure name into a clinician-friendly
name suitable for the "Adjacent To" rows of Anatomical Details.
E.g. `"iliopsoas_left"` → `"Iliopsoas Muscle (Left)"`.
"""
function format_adjacent_structure_name(raw_name::String)::String
    isempty(raw_name) && return ""
    ln = lowercase(raw_name)

    # Determine side
    side = if endswith(ln, "_left")
        " (Left)"
    elseif endswith(ln, "_right")
        " (Right)"
    else
        ""
    end

    # Strip side suffix for matching
    base = replace(replace(ln, r"_left$" => ""), r"_right$" => "")

    # Skip known lymph node structures — they are already formatted elsewhere
    is_lymph_node_structure(raw_name) && return format_clinical_station_name(raw_name)

    # Named muscle mapping
    muscle_map = Dict(
        "iliopsoas" => "Iliopsoas Muscle",
        "obturator_internus" => "Obturator Internus Muscle",
        "piriformis" => "Piriformis Muscle",
        "gluteus_maximus" => "Gluteus Maximus Muscle",
        "gluteus_medius" => "Gluteus Medius Muscle",
        "gluteus_minimus" => "Gluteus Minimus Muscle",
        "pectoralis_major" => "Pectoralis Major Muscle",
        "pectoralis_minor" => "Pectoralis Minor Muscle",
        "psoas_major" => "Psoas Major Muscle",
        "rectus_abdominis" => "Rectus Abdominis Muscle",
        "sternocleidomastoid" => "Sternocleidomastoid Muscle",
        "subscapularis" => "Subscapularis Muscle",
        "latissimus_dorsi" => "Latissimus Dorsi Muscle",
        "serratus_anterior" => "Serratus Anterior Muscle",
        "deltoid" => "Deltoid Muscle",
        "trapezius" => "Trapezius Muscle",
        "sartorius" => "Sartorius Muscle",
        "quadriceps_femoris" => "Quadriceps Femoris Muscle",
        "scalene" => "Scalene Muscle",
        "digastric" => "Digastric Muscle",
        "platysma" => "Platysma Muscle",
        "masseter" => "Masseter Muscle",
        "levator_scapulae" => "Levator Scapulae Muscle",
    )

    if haskey(muscle_map, base)
        return "$(muscle_map[base])$side"
    end

    # Named organ mapping
    organ_map = Dict(
        "prostate" => "Prostate Gland",
        "urinary_bladder" => "Urinary Bladder",
        "rectum" => "Rectum",
        "esophagus" => "Esophagus",
        "trachea" => "Trachea",
        "stomach" => "Stomach",
        "liver" => "Liver",
        "spleen" => "Spleen",
        "pancreas" => "Pancreas",
        "heart" => "Heart",
        "aorta" => "Aorta",
        "kidney" => "Kidney",
        "adrenal_gland" => "Adrenal Gland",
        "gallbladder" => "Gallbladder",
        "duodenum" => "Duodenum",
        "colon" => "Colon",
        "thyroid_gland" => "Thyroid Gland",
        "spinal_cord" => "Spinal Cord",
        "lung" => "Lung",
        "uterus" => "Uterus",
    )

    if haskey(organ_map, base)
        return "$(organ_map[base])$side"
    end

    # Vessel patterns
    for (pat, label) in [("iliac_artery" => "Iliac Artery"), ("iliac_vein" => "Iliac Vein"),
                         ("femoral_artery" => "Femoral Artery"), ("femoral_vein" => "Femoral Vein"),
                         ("carotid" => "Carotid Artery"), ("jugular" => "Jugular Vein"),
                         ("subclavian_artery" => "Subclavian Artery"), ("subclavian_vein" => "Subclavian Vein"),
                         ("pulmonary_artery" => "Pulmonary Artery"), ("pulmonary_vein" => "Pulmonary Vein"),
                         ("portal_vein" => "Portal Vein"), ("splenic_vein" => "Splenic Vein"),
                         ("hepatic_artery" => "Hepatic Artery"), ("celiac_trunk" => "Celiac Trunk"),
                         ("superior_mesenteric_artery" => "Superior Mesenteric Artery"),
                         ("inferior_vena_cava" => "Inferior Vena Cava"),
                         ("superior_vena_cava" => "Superior Vena Cava"),
                         ("brachiocephalic_vein" => "Brachiocephalic Vein"),
                         ("brachiocephalic_trunk" => "Brachiocephalic Trunk")]
        if occursin(pat, ln)
            return "$label$side"
        end
    end

    # Bone patterns
    for (pat, label) in [("vertebra" => "Vertebra"), ("femur" => "Femur"), ("hip" => "Hip Bone"),
                         ("sacrum" => "Sacrum"), ("sternum" => "Sternum"), ("rib" => "Rib"),
                         ("scapula" => "Scapula"), ("clavicle" => "Clavicle"), ("humerus" => "Humerus"),
                         ("mandible" => "Mandible"), ("hyoid" => "Hyoid Bone")]
        if occursin(pat, ln)
            return "$label$side"
        end
    end

    # Generic muscle detection
    if any(kw -> occursin(kw, ln), ["muscle", "gluteus", "psoas", "autochthon", "erector",
                                     "oblique", "transversospinalis", "rectus_abdominis"])
        name_clean = titlecase(replace(base, "_" => " "))
        if !occursin("Muscle", name_clean)
            name_clean *= " Muscle"
        end
        return "$name_clean$side"
    end

    # Fallback: titlecase
    return titlecase(replace(strip(raw_name), "_" => " "))
end

"""
    generate_detailed_anatomy_rows(counts::Dict{Int,Int}, ts_names::Dict{Int,String}; max_rows::Int=4) → String

Generate a pipe-separated Anatomical Details string from atlas overlap counts.
- Row 1: `"Inside / Contained In:Station Name"` (the primary lymph node station with highest overlap)
- Rows 2+: `"Adjacent To:Structure Name"` for secondary overlapping structures (non-LN organs, vessels, muscles)
  filtered to ≥ 3 voxels and sorted descending by count.

Returns "" if no lymph node overlap is found.

The output format matches MedEye3d's serialization: `"Rel1:Struct1 | Rel2:Struct2 | ..."`.
"""
function generate_detailed_anatomy_rows(counts::Dict{Int,Int}, ts_names::Dict{Int,String}; max_rows::Int=4)::String
    isempty(counts) && return ""

    # Separate lymph node stations from other structures
    ln_entries = Tuple{String, Int}[]
    other_entries = Tuple{String, Int}[]

    for (label_id, cnt) in counts
        cnt <= 0 && continue
        name = get(ts_names, label_id, "")
        isempty(name) && continue
        if is_lymph_node_structure(name)
            push!(ln_entries, (name, cnt))
        else
            push!(other_entries, (name, cnt))
        end
    end

    # No lymph node overlap → no detailed rows
    isempty(ln_entries) && return ""

    # Primary: lymph node station with largest overlap
    sort!(ln_entries, by = x -> x[2], rev = true)
    primary_raw = ln_entries[1][1]
    primary_formatted = format_clinical_station_name(primary_raw)

    parts = String[]
    push!(parts, "Inside / Contained In:$primary_formatted")

    # Secondary LN stations (if lesion overlaps multiple)
    for i in 2:min(length(ln_entries), max_rows)
        sec_raw = ln_entries[i][1]
        sec_cnt = ln_entries[i][2]
        sec_cnt < 3 && continue  # skip negligible overlaps
        sec_formatted = format_clinical_station_name(sec_raw)
        push!(parts, "Adjacent To:$sec_formatted")
    end

    # Adjacent non-LN structures (organs, muscles, vessels)
    sort!(other_entries, by = x -> x[2], rev = true)
    for (name, cnt) in other_entries
        length(parts) >= max_rows && break
        cnt < 3 && continue  # skip tiny overlaps
        adj_formatted = format_adjacent_structure_name(name)
        push!(parts, "Adjacent To:$adj_formatted")
    end

    return join(parts, " | ")
end

"""
    lookup_anatomy(raw_organ::String) -> String

Format a raw organ name into a human-readable anatomy string.
"""
function lookup_anatomy(raw_organ::String)
    isempty(raw_organ) && return ""
    return titlecase(replace(strip(raw_organ), "_" => " "))
end

export load_nrrd_labelmap, map_lesions_to_organs, classify_organ_to_lesion_type
export classify_tissue_priority, classify_and_pick_best_organ, count_atlas_overlap, pick_best_organ, lookup_anatomy
export is_lymph_node_structure, is_pelvic_lymph_node, classify_lymph_node_location
export format_clinical_station_name, format_adjacent_structure_name, generate_detailed_anatomy_rows

end # module
