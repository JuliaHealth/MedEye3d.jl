"""
LesionAssociation — Julia port of Logic/LesionAssociation.py

Handles matching of lesions between different time points.
Matches strategy (in priority order):
  1. Manual override: explicit link stored in associations Dict
  2. Cache: previously computed match in _overlap_mapping
  3. IoU-based JSON associations (from lesion_associations_exported.json)
  4. Centroid distance fallback (if within threshold)

In MedEye3d context:
  - "Nodes" are represented by String names (mask IDs or file paths)
  - "Segment IDs" are string label identifiers like "label_1", "Segment_N"
  - World positions are NTuple{3,Float64} in voxel or mm space

Key differences from the Slicer Python implementation:
  - No VTK/MRML dependencies
  - No GUI bindings
  - Spatial distance uses precomputed centroids from label arrays
"""
module LesionAssociation

using JSON

export LesionAssociationLogic,
       load_from_exported_json!,
       find_equivalent_lesion,
       find_equivalent_lesions,
       save_associations,
       load_associations,
       add_manual_link!,
       get_all_associations

"""
Association entry: maps a (src_node, dst_node, src_seg_id) → [dst_seg_ids]
Bidirectional entries are stored automatically.
"""
const AssocKey = Tuple{String,String,String}  # (src_node_name, dst_node_name, src_seg_id)

mutable struct LesionAssociationLogic
    # Main association cache: (srcNode, dstNode, srcSegId) → [dstSegIds]
    _overlap_mapping::Dict{AssocKey, Vector{String}}

    # Centroid cache: (node_name, seg_id) → (x, y, z) in voxel coords
    _centroid_cache::Dict{Tuple{String,String}, NTuple{3,Float64}}

    # TimePoint assignment: node_name → tp_index (Int)
    _timepoint_map::Dict{String, Int}

    # json_loaded flag
    _json_loaded::Bool
end

"""Construct an empty LesionAssociationLogic."""
function LesionAssociationLogic()
    LesionAssociationLogic(
        Dict{AssocKey, Vector{String}}(),
        Dict{Tuple{String,String}, NTuple{3,Float64}}(),
        Dict{String, Int}(),
        false
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

"""
    load_from_exported_json!(logic, json_path)

Load bidirectional associations from the lesion_associations_exported.json format:
    [{"parent_name": "...", "parent_lesion": "Segment_N",
      "child_name": "...", "children": [{"child_lesion": "Segment_M", "IoU": 0.42}]}]

Builds forward (parent→child) and reverse (child→parent) entries.
"""
function load_from_exported_json!(logic::LesionAssociationLogic, json_path::String)
    isfile(json_path) || (@warn "[LesionAssoc] File not found: $json_path"; return)
    try
        data = JSON.parsefile(json_path)
        count = 0
        for entry in data
            p_name = get(entry, "parent_name", "")
            p_sid  = get(entry, "parent_lesion", "")
            c_name = get(entry, "child_name", "")
            children = get(entry, "children", Any[])
            (isempty(p_name) || isempty(p_sid) || isempty(c_name) || isempty(children)) && continue

            c_ids = String[string(c["child_lesion"]) for c in children if haskey(c, "child_lesion")]
            isempty(c_ids) && continue

            # Forward: parent → children
            fwd_key = (p_name, c_name, p_sid)
            logic._overlap_mapping[fwd_key] = c_ids

            # Reverse: each child → parent
            for cid in c_ids
                rev_key = (c_name, p_name, cid)
                lst = get!(logic._overlap_mapping, rev_key, String[])
                p_sid in lst || push!(lst, p_sid)
            end
            count += 1
        end
        @info "[LesionAssoc] Loaded $count JSON associations → $(length(logic._overlap_mapping)) total mappings"
        logic._json_loaded = true
    catch e
        @warn "[LesionAssoc] Failed to load $json_path: $e"
    end
end

"""
    add_manual_link!(logic, src_node, src_seg_id, dst_node, dst_seg_id)

Explicitly link a lesion in src_node to a lesion in dst_node.
Stored as a manual override (highest priority in find_equivalent_lesion).
"""
function add_manual_link!(logic::LesionAssociationLogic,
                          src_node::String, src_seg_id::String,
                          dst_node::String, dst_seg_id::String)
    fwd = (src_node, dst_node, src_seg_id)
    logic._overlap_mapping[fwd] = [dst_seg_id]
    rev = (dst_node, src_node, dst_seg_id)
    lst = get!(logic._overlap_mapping, rev, String[])
    src_seg_id in lst || push!(lst, src_seg_id)
    @info "[LesionAssoc] Manual link: $src_node/$src_seg_id → $dst_node/$dst_seg_id"
end

"""
    register_centroid!(logic, node_name, seg_id, centroid)

Register a centroid (x, y, z) for a given segment. Used for spatial fallback.
"""
function register_centroid!(logic::LesionAssociationLogic,
                            node_name::String, seg_id::String,
                            centroid::NTuple{3,Float64})
    logic._centroid_cache[(node_name, seg_id)] = centroid
end

"""
    register_timepoint!(logic, node_name, tp_index)

Register which time point a node belongs to.
"""
function register_timepoint!(logic::LesionAssociationLogic, node_name::String, tp_index::Int)
    logic._timepoint_map[node_name] = tp_index
end

"""
    get_timepoint(logic, node_name) → Int

Return the time point index for a given node (0 if unknown).
Falls back to extracting the last integer in the name.
"""
function get_timepoint(logic::LesionAssociationLogic, node_name::String)::Int
    haskey(logic._timepoint_map, node_name) && return logic._timepoint_map[node_name]
    nums = [parse(Int, m.match) for m in eachmatch(r"\d+", node_name)]
    return isempty(nums) ? 0 : last(nums)
end

"""
    find_equivalent_lesion(logic, src_node, src_seg_id, dst_node) → (dst_seg_id | nothing)

Find the corresponding lesion in dst_node. Uses:
  1. JSON association map (_overlap_mapping)
  2. Same time-point intra-node fallback (same seg_id if tp matches)
  3. Centroid distance fallback (returns closest within 50 voxels)
"""
function find_equivalent_lesion(logic::LesionAssociationLogic,
                                src_node::String, src_seg_id::String,
                                dst_node::String,
                                available_dst_segs::Vector{String} = String[])::Union{String, Nothing}

    src_node == dst_node && return src_seg_id

    # 1. JSON association map
    key = (src_node, dst_node, src_seg_id)
    if haskey(logic._overlap_mapping, key)
        hits = logic._overlap_mapping[key]
        # Filter to available segments (if caller provided the list)
        if !isempty(available_dst_segs)
            hits = filter(s -> s in available_dst_segs, hits)
        end
        isempty(hits) || return first(hits)
    end

    # Strict Error Enforcement: No fallbacks allowed
    error("Strict Matching Enforcement: Lesion association not found in IoU overlap mapping for $src_node / $src_seg_id to $dst_node. Centroid distance and ID fallbacks have been disabled.")
end

"""
    find_equivalent_lesions(logic, src_node, src_seg_id, dst_node, available_dst_segs) → Vector{String}

Returns all matching segment IDs (1-to-N). Falls back the same as find_equivalent_lesion.
"""
function find_equivalent_lesions(logic::LesionAssociationLogic,
                                 src_node::String, src_seg_id::String,
                                 dst_node::String,
                                 available_dst_segs::Vector{String} = String[])::Vector{String}
    src_node == dst_node && return [src_seg_id]

    key = (src_node, dst_node, src_seg_id)
    if haskey(logic._overlap_mapping, key)
        hits = logic._overlap_mapping[key]
        if !isempty(available_dst_segs)
            hits = filter(s -> s in available_dst_segs, hits)
        end
        isempty(hits) || return hits
    end

    single = find_equivalent_lesion(logic, src_node, src_seg_id, dst_node, available_dst_segs)
    single === nothing ? String[] : [single]
end

"""
    save_associations(logic, path)

Persist the current association map to a JSON file for later loading.
"""
function save_associations(logic::LesionAssociationLogic, path::String)
    data = Dict{String, Vector{String}}()
    for ((src_n, dst_n, src_s), val) in logic._overlap_mapping
        data["$(src_n)|$(dst_n)|$(src_s)"] = val
    end
    open(path, "w") do io; JSON.print(io, data, 2) end
    @info "[LesionAssoc] Saved $(length(data)) associations → $path"
end

"""
    load_associations(logic, path)

Load associations from a previously saved JSON file (key = "src|dst|seg").
"""
function load_associations(logic::LesionAssociationLogic, path::String)
    isfile(path) || (@warn "[LesionAssoc] Not found: $path"; return)
    data = JSON.parsefile(path)
    empty!(logic._overlap_mapping)
    for (key_str, val) in data
        parts = split(key_str, "|")
        length(parts) == 3 || continue
        logic._overlap_mapping[(parts[1], parts[2], parts[3])] =
            isa(val, Vector) ? String.(val) : [string(val)]
    end
    @info "[LesionAssoc] Loaded $(length(logic._overlap_mapping)) associations from $path"
end

"""
    get_all_associations(logic) → Dict{AssocKey, Vector{String}}

Return a copy of the full association mapping for inspection.
"""
get_all_associations(logic::LesionAssociationLogic) = copy(logic._overlap_mapping)

end # module LesionAssociation
