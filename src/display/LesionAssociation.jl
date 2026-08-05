module LesionAssociation

using JSON

const ASSOC_JSON_PATH = joinpath(homedir(), "medeye3d_lesion_associations.json")
const OVERLAP_MAPPING = Dict{Tuple{String, String, String}, Vector{String}}()

export load_associations, save_associations, get_children, map_link

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

end # module
