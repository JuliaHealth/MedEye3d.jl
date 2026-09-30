module DagOrchestrator

using JSON

export execute_graph, resolve_global_overlaps, DagState

mutable struct DagState
    node_to_channel::Dict{String, Int}
    next_channel::Int
    dims::Tuple{Int, Int, Int}
    spacing::Tuple{Float64, Float64, Float64}
    origin::Tuple{Float64, Float64, Float64}
    direction::Tuple{Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64}
    known_masks::Set{String}
    computed_landmarks::Dict{String, Any}
end

DagState() = DagState(Dict{String, Int}(), 1, (512, 512, 512), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), Set{String}(), Dict{String, Any}())

function get_channel!(state::DagState, node_name::String)
    if !haskey(state.node_to_channel, node_name)
        state.node_to_channel[node_name] = state.next_channel
        state.next_channel += 1
    end
    return state.node_to_channel[node_name]
end

function get_execution_levels(deps::Dict)
    levels = Vector{Vector{String}}()
    resolved = Set{String}()
    remaining = Dict(String(node) => Set{String}(String.(deps_list)) for (node, deps_list) in deps)
    
    while !isempty(remaining)
        current_level = String[]
        for (node, deps_list) in remaining
            if issubset(deps_list, resolved)
                push!(current_level, node)
            end
        end
        
        if isempty(current_level)
            error("Cycle detected in graph! Remaining: $remaining")
        end
        
        push!(levels, current_level)
        union!(resolved, current_level)
        
        for node in current_level
            delete!(remaining, node)
        end
    end
    
    return levels
end

function resolve_channel_sided(state::DagState, name::String, side::Union{String, Nothing}=nothing)
    if side !== nothing
        # 1. If name contains $side, substitute it
        sided_name = replace(name, "\$side" => side)
        if sided_name != name
            return get_channel!(state, sided_name)
        end
        
        # 2. Check rib patterns like rib_3 -> rib_left_3 / rib_3_left
        side_l = lowercase(side)
        rib_m = match(r"^rib_(\d+)$", lowercase(name))
        if rib_m !== nothing
            num = rib_m.captures[1]
            for cand in ["rib_$(side_l)_$num", "rib_$(num)_$(side_l)", "rib_$(side_l)_$(num)_cleaned"]
                if cand in state.known_masks || haskey(state.node_to_channel, cand)
                    return get_channel!(state, cand)
                end
            end
        end
        
        # 3. Check if a sided variant exists in known masks or already assigned
        for cand in [name * "_" * side_l, name * "_" * side, name * "_" * uppercase(side_l[1:1]) * side_l[2:end], "$(name)_$(side_l)"]
            if cand in state.known_masks || haskey(state.node_to_channel, cand)
                return get_channel!(state, cand)
            end
        end
        
        # Also check case-insensitive match for sided name
        for km in state.known_masks
            if lowercase(km) == lowercase(name * "_" * side_l) || lowercase(km) == lowercase(name * "_" * side)
                return get_channel!(state, km)
            end
        end
    end
    
    # 4. Check direct match
    if name in state.known_masks || haskey(state.node_to_channel, name)
        return get_channel!(state, name)
    end
    
    # 5. Check case-insensitive match
    for km in state.known_masks
        if lowercase(km) == lowercase(name)
            return get_channel!(state, km)
        end
    end
    
    # 5b. If side is nothing, check if sided variant was generated
    if side === nothing
        for cand in [name * "_left", name * "_right", name * "_Left", name * "_Right"]
            if cand in state.known_masks || haskey(state.node_to_channel, cand)
                return get_channel!(state, cand)
            end
        end
    end
    
    # 6. Check alias fallbacks
    aliases = Dict(
        "clavicle" => "clavicula",
        "clavicle_left" => "clavicula_left",
        "clavicle_right" => "clavicula_right",
        "cricoid" => "cricoid_cartilage",
        "thyroid" => "thyroid_gland",
        "scalene" => "anterior_scalene",
        "scalene_muscle" => "anterior_scalene",
        "manubrium" => "sternum",
        "airway" => "trachea"
    )
    if haskey(aliases, lowercase(name))
        alias = aliases[lowercase(name)]
        if alias in state.known_masks || haskey(state.node_to_channel, alias)
            return get_channel!(state, alias)
        end
    end
    
    if (name in state.known_masks || haskey(state.node_to_channel, name))
        return get_channel!(state, name)
    end
    
    return nothing
end

function resolve_input_channel_sided(state::DagState, params::Dict, deps_graph::Dict, node::String, side::Union{String, Nothing}=nothing)
    rule_type = get(params, "rule", "")

    for k in ["input", "input_mask", "input_image", "inputs", "targets", "target_landmarks", "base_landmark", "mask_name", "input_landmarks", "target", "components", "landmark", "landmarks", "input_a"]
        if haskey(params, k)
            v = params[k]
            if v isa String
                ch = resolve_channel_sided(state, v, side)
                return ch !== nothing ? ch : get_channel!(state, v)
            elseif v isa Vector && !isempty(v) && v[1] isa String
                ch = resolve_channel_sided(state, v[1], side)
                return ch !== nothing ? ch : get_channel!(state, v[1])
            elseif k == "base_landmark" && v isa Dict
                # Dict-based base_landmark: use first key as the primary input
                first_key = first(keys(v))
                ch = resolve_channel_sided(state, first_key, side)
                return ch !== nothing ? ch : get_channel!(state, first_key)
            end
        end
    end
    
    if rule_type in ["AxillaryRTOG", "AxillaryRTOGRelaxed"]
        lvl = lowercase(get(params, "level", "i"))
        s = side !== nothing ? lowercase(side) : (occursin("right", lowercase(node)) ? "right" : "left")
        if lvl == "i"
            return resolve_channel_sided(state, "helper_rtog_I_z_$s", s)
        elseif lvl == "ii"
            return resolve_channel_sided(state, "helper_rtog_II_z_$s", s)
        elseif lvl == "iii"
            return resolve_channel_sided(state, "helper_rtog_III_z_$s", s)
        elseif lvl == "rotter"
            return resolve_channel_sided(state, "helper_rotter_pm_ext_$s", s)
        end
    end
    
    deps_list = get(deps_graph, node, [])
    if !isempty(deps_list)
        return resolve_channel_sided(state, String(deps_list[1]), side)
    end
    return 1
end

function resolve_constraint_for_julia(state::DagState, constraint_config::Dict, rule_params::Dict, side::Union{String, Nothing}=nothing)
    c_type = get(constraint_config, "constraint_type", "")
    c_type = get(constraint_config, "constraint_type", "")
    landmark = get(constraint_config, "landmark", nothing)
    
    if c_type == "BetweenLandmarksXLimit"
        # Convert to two X constraints: LeftOf landmark_2 and RightOf landmark_1
        # This restricts the mask to be between the two landmarks in X
        lm1 = get(constraint_config, "landmark_1", nothing)
        lm2 = get(constraint_config, "landmark_2", nothing)
        constraints_out = []
        
        if lm1 !== nothing
            ch1 = resolve_channel_sided(state, lm1, side)
            if ch1 === nothing && lm1 in state.known_masks
                ch1 = get_channel!(state, lm1)
            end
            if ch1 !== nothing
                push!(constraints_out, Dict{String, Any}(
                    "type" => "RightOf", "axis" => 1, "boundary_part" => "max",
                    "offset_mm" => 0.0, "slice_wise" => true,
                    "landmark_channels" => [ch1]
                ))
            end
        end
        
        if lm2 !== nothing
            ch2 = resolve_channel_sided(state, lm2, side)
            if ch2 === nothing && lm2 in state.known_masks
                ch2 = get_channel!(state, lm2)
            end
            if ch2 !== nothing
                push!(constraints_out, Dict{String, Any}(
                    "type" => "LeftOf", "axis" => 1, "boundary_part" => "min",
                    "offset_mm" => 0.0, "slice_wise" => true,
                    "landmark_channels" => [ch2]
                ))
            end
        end
        
        # Return multiple constraints as a list
        return isempty(constraints_out) ? nothing : constraints_out
    end
    
    if c_type == "PlaneLimit"
        side_to_keep = get(constraint_config, "side_to_keep", "negative")
        lm_channels = Int[]
        if landmark === nothing
            landmark = get(constraint_config, "landmarks", get(constraint_config, "reference", nothing))
        end
        if landmark !== nothing
            lm_list = landmark isa Vector ? landmark : [landmark]
            for lm in lm_list
                lm_str = String(lm)
                if side !== nothing
                    if side == "left" && (occursin(r"(?i)_right$", lm_str) || occursin(r"(?i)_r$", lm_str))
                        continue
                    elseif side == "right" && (occursin(r"(?i)_left$", lm_str) || occursin(r"(?i)_l$", lm_str))
                        continue
                    end
                elseif get(rule_params, "is_bilateral", false) == true
                    if occursin(r"(?i)_left$", lm_str) || occursin(r"(?i)_right$", lm_str)
                        continue
                    end
                end
                
                # Check if plane is in state.computed_landmarks
                candidates = [lm_str]
                if side !== nothing
                    push!(candidates, lm_str * "_" * side)
                    push!(candidates, lm_str * "_" * uppercasefirst(side))
                    push!(candidates, replace(lm_str, "\$side" => side))
                else
                    push!(candidates, lm_str * "_left")
                    push!(candidates, lm_str * "_right")
                    push!(candidates, lm_str * "_Left")
                    push!(candidates, lm_str * "_Right")
                end
                for cand in candidates
                    if haskey(state.computed_landmarks, cand)
                        c_val = state.computed_landmarks[cand]
                        if c_val isa Vector && length(c_val) == 2 && c_val[1] isa Vector && c_val[2] isa Vector
                            return Dict{String, Any}(
                                "type" => "PlaneLimit",
                                "side_to_keep" => side_to_keep,
                                "point" => Float64.(c_val[1]),
                                "normal" => Float64.(c_val[2])
                            )
                        end
                    end
                end
                
                lm_ch = side !== nothing ? resolve_channel_sided(state, lm_str, side) : get_channel!(state, lm_str)
                if lm_ch !== nothing
                    push!(lm_channels, lm_ch)
                end
            end
        end
        if isempty(lm_channels)
            return nothing
        end
        result = Dict{String, Any}(
            "type" => "PlaneLimit",
            "side_to_keep" => side_to_keep,
            "landmark_channels" => lm_channels
        )
        if haskey(constraint_config, "boundary_part")
            result["boundary_part"] = constraint_config["boundary_part"]
        end
        if haskey(constraint_config, "offset_mm")
            result["offset_mm"] = Float64(constraint_config["offset_mm"])
        end
        if haskey(constraint_config, "mode")
            result["mode"] = constraint_config["mode"]
        end
        return result
    end
    b_part = get(constraint_config, "boundary_part", "")
    offset_mm = get(constraint_config, "offset_mm", 0.0)
    slice_wise = get(constraint_config, "slice_wise", false)
    
    axis_map = Dict(
        "SuperiorTo" => 3, "InferiorTo" => 3,
        "AnteriorTo" => 2, "PosteriorTo" => 2,
        "LeftOf" => 1, "RightOf" => 1,
        "SideLimit" => 1, "MedialTo" => 1, "LateralTo" => 1,
        "SuperiorToLowestOf" => 3, "InferiorToLowestOf" => 3,
        "SuperiorToHighestOf" => 3, "InferiorToHighestOf" => 3
    )
    
    effective_type = c_type
    side_val = get(constraint_config, "side_for_limit", get(constraint_config, "side", get(rule_params, "side", side)))
    if side_val !== nothing
        side = String(side_val)
    end
    
    if side === nothing && side !== nothing
        side = replace(side, "_" => "")
    end
    
    explicit_limit_idx = nothing
    if c_type == "SideLimit"
        effective_type = (side !== nothing && lowercase(side) == "left") ? "LeftOf" : "RightOf"
        explicit_limit_idx = Int(round(state.dims[1] / 2))
    elseif c_type == "MedialTo"
        effective_type = (side !== nothing && lowercase(side) == "left") ? "RightOf" : "LeftOf"
    elseif c_type == "LateralTo"
        effective_type = (side !== nothing && lowercase(side) == "left") ? "LeftOf" : "RightOf"
    elseif c_type == "InferiorToLowestOf"
        effective_type = "InferiorTo"
        b_part = "min"
    elseif c_type == "SuperiorToLowestOf"
        effective_type = "SuperiorTo"
        b_part = "min"
    elseif c_type == "SuperiorToHighestOf"
        effective_type = "SuperiorTo"
        b_part = "max"
    elseif c_type == "InferiorToHighestOf"
        effective_type = "InferiorTo"
        b_part = "max"
    end
    
    axis = get(axis_map, effective_type, 3)
    
    if isempty(b_part)
        bp_defaults = Dict(
            "SuperiorTo" => "max", "InferiorTo" => "min",
            "AnteriorTo" => "min", "PosteriorTo" => "max",
            "LeftOf" => "max", "RightOf" => "min"
        )
        b_part = get(bp_defaults, effective_type, "min")
    # DO NOT TRANSLATE MEDIAL/LATERAL/ANTERIOR/POSTERIOR HERE!
    # The DAG VM must interpret them based on the runtime NIfTI direction (RAS vs LPS)!
    # elseif b_part == "medial"
    #     b_part = (side !== nothing && lowercase(side) == "left") ? "min" : "max"
    # elseif b_part == "lateral"
    #     b_part = (side !== nothing && lowercase(side) == "left") ? "max" : "min"
    # elseif b_part == "anterior"
    #     b_part = "min"
    # elseif b_part == "posterior"
    #     b_part = "max"
    # elseif b_part == "superior"
    #     b_part = "max"
    # elseif b_part == "inferior"
    #     b_part = "min"
    # end
    
    result = Dict{String, Any}(
        "type" => effective_type,
        "axis" => axis,
        "boundary_part" => b_part,
        "offset_mm" => Float64(offset_mm),
        "slice_wise" => slice_wise
    )
    if explicit_limit_idx !== nothing
        result["limit_idx"] = explicit_limit_idx
    end
    
    if landmark === nothing
        landmark = get(constraint_config, "landmarks", get(constraint_config, "reference", get(constraint_config, "target", get(constraint_config, "input", get(constraint_config, "fallback_landmark", nothing)))))
    end
    
    if landmark !== nothing
        lm_list = landmark isa Vector ? landmark : [landmark]
        lm_channels = Int[]
        for lm in lm_list
            if lm isa Dict
                # Extract landmark string from dict (e.g., {landmark: "heart", part: "min"})
                lm = get(lm, "landmark", get(lm, "organ", get(lm, "name", nothing)))
                if lm === nothing
                    continue
                end
            end
            lm_str = String(lm)
            
            # Check contralateral filtering
            if side !== nothing
                if side == "left" && (occursin(r"(?i)_right$", lm_str) || occursin(r"(?i)_r$", lm_str))
                    continue
                elseif side == "right" && (occursin(r"(?i)_left$", lm_str) || occursin(r"(?i)_l$", lm_str))
                    continue
                end
            elseif get(rule_params, "is_bilateral", false) == true
                if occursin(r"(?i)_left$", lm_str) || occursin(r"(?i)_right$", lm_str)
                    continue
                end
            end

            # 1. First try to resolve as a 3D mask channel
            lm_ch = side !== nothing ? resolve_channel_sided(state, lm_str, side) : 
                    (lm_str in state.known_masks || haskey(state.node_to_channel, lm_str) ? get_channel!(state, lm_str) : nothing)
            
            if lm_ch === nothing && haskey(constraint_config, "fallback_landmark")
                fb = String(constraint_config["fallback_landmark"])
                lm_ch = side !== nothing ? resolve_channel_sided(state, fb, side) : 
                        (fb in state.known_masks || haskey(state.node_to_channel, fb) ? get_channel!(state, fb) : nothing)
            end
            
            if lm_ch !== nothing
                push!(lm_channels, lm_ch)
            else
                # 2. If no mask channel exists, check if landmark is in computed_landmarks
                candidates = [lm_str]
                if side !== nothing
                    push!(candidates, lm_str * "_" * side)
                    push!(candidates, lm_str * "_" * uppercasefirst(side))
                    push!(candidates, replace(lm_str, "\$side" => side))
                else
                    push!(candidates, lm_str * "_left")
                    push!(candidates, lm_str * "_right")
                    push!(candidates, lm_str * "_Left")
                    push!(candidates, lm_str * "_Right")
                end
                num_found = false
                for cand in candidates
                    if haskey(state.computed_landmarks, cand)
                        c_val = state.computed_landmarks[cand]
                        is_vox = occursin(r"(?i)(_z|_z_left|_z_right|_vox|_index|_slice)$", cand) || cand in ["cricoid", "low_cricoid_plane", "diaphragm_level", "aortic_bifurcation"]
                        if c_val isa Number
                            val = Float64(c_val)
                            if is_vox || (val >= 1.0 && val <= Float64(state.dims[axis]) && val == floor(val) && abs(val - state.origin[axis]) > 100.0)
                                limit_idx = round(Int, val)
                            else
                                limit_idx = round(Int, (val - state.origin[axis]) / state.spacing[axis]) + 1
                            end
                            result["limit_idx"] = clamp(limit_idx, 1, state.dims[axis])
                            num_found = true
                            break
                        elseif c_val isa Vector && length(c_val) == 3 && all(x -> x isa Number, c_val)
                            val = Float64(c_val[axis])
                            limit_idx = round(Int, (val - state.origin[axis]) / state.spacing[axis]) + 1
                            result["limit_idx"] = clamp(limit_idx, 1, state.dims[axis])
                            num_found = true
                            break
                        end
                    end
                end
                if num_found
                    return result
                end
            end
        end
        if isempty(lm_channels) && !haskey(result, "limit_idx")
            return nothing
        end
        if !isempty(lm_channels)
            result["landmark_channels"] = lm_channels
        end
    end
    
    return result
end

function add_boolean_channels_sided!(op::Dict, params::Dict, rule_type::String, state::DagState, side::Union{String, Nothing}=nothing)
    is_intersect_rule = rule_type in ["BitwiseAnd", "Intersect"] || get(params, "operation", "") == "Intersect"
    is_subtract_rule = rule_type in ["Subtract", "Exclude"] || get(params, "operation", "") == "Subtract"
    
    is_union_rule = rule_type in ["BitwiseOr", "Union", "Combine", "ConvexHull2D", "ConvexHull"] || get(params, "operation", "") == "Union"
    
    if is_union_rule || is_intersect_rule || is_subtract_rule
        if haskey(params, "inputs") && params["inputs"] isa Vector && length(params["inputs"]) > 1
            extra_inputs = params["inputs"][2:end]
            channels = [resolve_channel_sided(state, String(n), side) for n in extra_inputs if n isa String]
            filter!(x -> x !== nothing && x > 0, channels)
            if is_subtract_rule
                op["subtract_channels"] = vcat(get(op, "subtract_channels", []), channels)
            elseif is_intersect_rule
                op["intersect_channels"] = vcat(get(op, "intersect_channels", []), channels)
            else
                op["union_channels"] = vcat(get(op, "union_channels", []), channels)
            end
        end
        
        if is_union_rule && haskey(params, "components") && params["components"] isa Vector && length(params["components"]) > 1
            extra_comps = params["components"][2:end]
            channels = [resolve_channel_sided(state, String(n), side) for n in extra_comps if n isa String]
            filter!(x -> x !== nothing && x > 0, channels)
            op["union_channels"] = vcat(get(op, "union_channels", []), channels)
        end
        
        for key in ["input_landmarks", "base_landmark", "landmarks", "targets", "target_landmarks", "input_b"]
            if haskey(params, key)
                val = params[key]
                val_list = val isa Vector ? val : [val]
                channels = [resolve_channel_sided(state, String(n), side) for n in val_list if n isa String]
                filter!(x -> x !== nothing && x > 0, channels)
                if rule_type in ["BitwiseOr", "Combine", "Union", "DilatedMask", "DistanceExpansion", "AnisotropicMargin", "PrimaryVector"]
                    op["union_channels"] = vcat(get(op, "union_channels", []), channels)
                elseif rule_type in ["Intersect", "BitwiseAnd"]
                    op["intersect_channels"] = vcat(get(op, "intersect_channels", []), channels)
                elseif is_subtract_rule
                    op["subtract_channels"] = vcat(get(op, "subtract_channels", []), channels)
                elseif val isa Vector && length(val) > 1
                    op["union_channels"] = vcat(get(op, "union_channels", []), channels)
                end
            end
        end
    end
    
    if haskey(params, "union_with")
        uw = params["union_with"]
        uw_list = uw isa String ? [uw] : uw
        channels = [resolve_channel_sided(state, String(n), side) for n in uw_list if n isa String]
        filter!(x -> x !== nothing && x > 0, channels)
        op["union_channels"] = vcat(get(op, "union_channels", []), channels)
    end
    
    intersect_fat = get(params, "intersect_fat", false) || get(params, "intersect_with_fat", false)
    if intersect_fat
        if "tissue_fat" in state.known_masks
            fat_ch = resolve_channel_sided(state, "tissue_fat", side)
            if fat_ch !== nothing
                op["intersect_channels"] = vcat(get(op, "intersect_channels", []), [fat_ch])
            end
        end
    end
    
    if haskey(params, "exclude")
        ex = params["exclude"]
        ex_list = ex isa String ? [ex] : ex
        channels = Int[]
        for n in ex_list
            if n isa String
                ch = resolve_channel_sided(state, String(n), side)
                if ch !== nothing && ch > 0 push!(channels, ch) end
            elseif n isa Dict
                lm = get(n, "landmark", get(n, "organ", nothing))
                if lm !== nothing
                    ch = resolve_channel_sided(state, String(lm), side)
                    if ch !== nothing && ch > 0 push!(channels, ch) end
                end
            end
        end
        op["subtract_channels"] = vcat(get(op, "subtract_channels", []), channels)
    end
    
    if haskey(params, "intersect_with")
        iw = params["intersect_with"]
        iw_list = iw isa String ? [iw] : iw
        channels = [resolve_channel_sided(state, String(n), side) for n in iw_list if n isa String]
        filter!(x -> x !== nothing && x > 0, channels)
        op["intersect_channels"] = vcat(get(op, "intersect_channels", []), channels)
    end
    
    anc_dict = Dict{String, Int}()
    for (k, v) in params
        if v isa String
            ch = resolve_channel_sided(state, v, side)
            if ch !== nothing && ch > 0
                anc_dict[v] = ch
            end
        elseif v isa Vector
            for n in v
                if n isa String
                    ch = resolve_channel_sided(state, n, side)
                    if ch !== nothing && ch > 0
                        anc_dict[n] = ch
                    end
                end
            end
        end
    end
    
    rule_type = get(params, "rule", "")
    if rule_type in ["AxillaryRTOG", "AxillaryRTOGRelaxed"]
        s = side !== nothing ? lowercase(side) : (occursin("right", lowercase(get(op, "target", ""))) ? "right" : "left")
        pec_ch = resolve_channel_sided(state, "pectoralis_major_$s", s)
        if pec_ch !== nothing && pec_ch > 0; anc_dict["pectoralis_major_$s"] = pec_ch; end
        sub_ch = resolve_channel_sided(state, "helper_subscapularis_$s", s)
        if sub_ch !== nothing && sub_ch > 0; anc_dict["helper_subscapularis_$s"] = sub_ch; end
        art_ch = resolve_channel_sided(state, "subclavian_artery_$s", s)
        if art_ch !== nothing && art_ch > 0; anc_dict["subclavian_artery_$s"] = art_ch; end
        vein_ch = resolve_channel_sided(state, "subclavian_vein_$s", s)
        if vein_ch !== nothing && vein_ch > 0; anc_dict["subclavian_vein_$s"] = vein_ch; end
        hum_ch = resolve_channel_sided(state, "humerus_$s", s)
        if hum_ch !== nothing && hum_ch > 0; anc_dict["humerus_$s"] = hum_ch; end
        lat_ch = resolve_channel_sided(state, "latissimus_dorsi_$s", s)
        if lat_ch !== nothing && lat_ch > 0; anc_dict["latissimus_dorsi_$s"] = lat_ch; end
        clav_ch = resolve_channel_sided(state, "clavicula_$s", s)
        if clav_ch !== nothing && clav_ch > 0; anc_dict["clavicula_$s"] = clav_ch; end
    elseif rule_type == "IliacBifurcationCustom"
        art_l = resolve_channel_sided(state, "iliac_artery_left", "left")
        if art_l !== nothing && art_l > 0; anc_dict["iliac_artery_left"] = art_l; end
        art_r = resolve_channel_sided(state, "iliac_artery_right", "right")
        if art_r !== nothing && art_r > 0; anc_dict["iliac_artery_right"] = art_r; end
        ven_l = resolve_channel_sided(state, "iliac_vena_left", "left")
        if ven_l !== nothing && ven_l > 0; anc_dict["iliac_vena_left"] = ven_l; end
        ven_r = resolve_channel_sided(state, "iliac_vena_right", "right")
        if ven_r !== nothing && ven_r > 0; anc_dict["iliac_vena_right"] = ven_r; end
    elseif rule_type == "CommonIliacCustom"
        aorta_ch = resolve_channel_sided(state, "aorta", nothing)
        if aorta_ch !== nothing && aorta_ch > 0; anc_dict["aorta"] = aorta_ch; end
        # Common iliac arteries and veins
        for suffix in ["left", "right"]
            for vessel in ["iliac_artery_common", "iliac_vena_common", "iliac_artery", "iliac_vena"]
                name = "$(vessel)_$(suffix)"
                ch = resolve_channel_sided(state, name, suffix)
                if ch !== nothing && ch > 0; anc_dict[name] = ch; end
            end
        end
    elseif rule_type == "ExternalIliacCustom"
        for suffix in ["left", "right"]
            for vessel in ["iliac_artery", "iliac_vena"]
                name = "$(vessel)_$(suffix)"
                ch = resolve_channel_sided(state, name, suffix)
                if ch !== nothing && ch > 0; anc_dict[name] = ch; end
            end
        end
        femur_ch = resolve_channel_sided(state, "femur", side !== nothing ? lowercase(side) : nothing)
        if femur_ch !== nothing && femur_ch > 0; anc_dict["femur"] = femur_ch; end
    end
    
if !isempty(anc_dict)
        op["ancillary_channels_dict"] = anc_dict
    end
end

function create_operation(state::DagState, node::String, rule::Dict, deps_graph::Dict, side::Union{String, Nothing}=nothing)
    rule_type = get(rule, "rule", "Unknown")
    params = copy(rule)
    
    if rule_type == "GeometricConstraint"
        rule_type = "Base"
        c = Dict{String, Any}()
        if haskey(params, "constraint_type")
            c["constraint_type"] = params["constraint_type"]
        end
        if haskey(params, "landmark")
            c["landmark"] = params["landmark"]
        end
        if haskey(params, "boundary_part")
            c["boundary_part"] = params["boundary_part"]
        end
        if haskey(params, "offset_mm")
            c["offset_mm"] = params["offset_mm"]
        end
        if haskey(params, "mode")
            c["mode"] = params["mode"]
        end
        if haskey(c, "constraint_type")
            params["constraints"] = vcat(get(params, "constraints", []), [c])
        end
    end
    
    if rule_type == "MarginAroundLandmark"
        # MarginAroundLandmark is AnisotropicMargin with uniform margin around the landmark
        rule_type = "AnisotropicMargin"
        mm = get(params, "margin_mm", 10)
        params["base_landmark"] = get(params, "landmark", get(params, "base_landmark", nothing))
        params["margins_mm"] = Dict("all" => mm)
        params["rule"] = "AnisotropicMargin"
        # If include_landmark is false, we'd need to subtract, but for now include is default
    end
    
    op = Dict{String, Any}("type" => rule_type, "target" => side !== nothing ? node * "_" * lowercase(side) : node)
    params["spacing_x"] = state.spacing[1]
    params["spacing_y"] = state.spacing[2]
    params["spacing_z"] = state.spacing[3]
    params["origin_x"] = state.origin[1]
    params["origin_y"] = state.origin[2]
    params["origin_z"] = state.origin[3]
    params["direction_00"] = state.direction[1]
    params["direction_11"] = length(state.direction) >= 5 ? state.direction[5] : 1.0
    params["direction_22"] = length(state.direction) >= 9 ? state.direction[9] : 1.0
    if side !== nothing
        params["side"] = side
    elseif !haskey(params, "side")
        if occursin(r"(?i)(_left$|_left\b|left)", node)
            params["side"] = "left"
        elseif occursin(r"(?i)(_right$|_right\b|right)", node)
            params["side"] = "right"
        end
    end
    if rule_type == "DilatedMask"
        dil_mm = get(params, "dilation_mm", 0)
        if get(params, "slice_wise", false)
            rule_type = "AnisotropicMargin"
            params["margin_left"] = dil_mm
            params["margin_right"] = dil_mm
            params["margin_anterior"] = dil_mm
            params["margin_posterior"] = dil_mm
            params["margin_superior"] = 0.0
            params["margin_inferior"] = 0.0
        else
            rule_type = "DistanceExpansion"
            params["distance_mm"] = dil_mm
        end
    elseif rule_type == "Margin" || rule_type == "AnisotropicMargin"
        if rule_type == "Margin"
            rule_type = "AnisotropicMargin"
        end
        m = get(params, "margin_mm", 0)
        
        margins_dict = get(params, "margins_mm", Dict())
        if margins_dict isa Dict
            for (k, v) in margins_dict
                if k == "medial"
                    params[(side == "left") ? "margin_right" : "margin_left"] = v
                elseif k == "lateral"
                    params[(side == "left") ? "margin_left" : "margin_right"] = v
                else
                    params["margin_\$k"] = v
                end
            end
        end
        
        for k in ["left", "right", "anterior", "posterior", "superior", "inferior"]
            mk = "margin_\$k"
            if !haskey(params, mk)
                params[mk] = m
            end
        end
        
        if haskey(params, "margin_medial")
            params[(side == "left") ? "margin_right" : "margin_left"] = params["margin_medial"]
        end
        if haskey(params, "margin_lateral")
            params[(side == "left") ? "margin_left" : "margin_right"] = params["margin_lateral"]
        end
    elseif rule_type == "Crop"
        rule_type = "DistanceExpansion"
        params["distance_mm"] = 0.0
    end
    
    target_name = node
    if side !== nothing
        target_name = replace(node, "\$side" => side)
        if target_name == node
            target_name = node * "_" * side
        end
    end
    
    out_ch = get_channel!(state, target_name)
    in_ch = resolve_input_channel_sided(state, params, deps_graph, node, side)
    
    op = Dict{String, Any}(
        "rule" => rule_type,
        "target" => target_name,
        "input_channel" => in_ch,
        "out_channel" => out_ch,
        "params" => params
    )
    
    add_boolean_channels_sided!(op, params, rule_type, state, side)
    
    if rule_type == "VolumetricBoundary2D"
        boundaries = get(params, "boundaries", Dict())
        op["growth_mode"] = get(params, "growth_mode", "convex_hull")
        op["ap_margin_only"] = get(params, "ap_margin_only", false)
        op["per_layer_margin"] = get(params, "per_layer_margin", false)
        op["posterior_offset_mm"] = Float64(get(params, "posterior_offset_mm", 5.0))
        op["min_depth_mm"] = Float64(get(params, "min_depth_mm", 0.0))
        op["erode_iterations"] = Int(get(params, "erode_iterations", 0))
        
        if haskey(boundaries, "structures")
            op["structure_channels"] = [side !== nothing ? resolve_channel_sided(state, String(s), side) : get_channel!(state, String(s)) for s in boundaries["structures"]]
        else
            op["structure_channels"] = [side !== nothing ? resolve_channel_sided(state, String(s), side) : get_channel!(state, String(s)) for (k, v) in boundaries for s in (v isa Vector ? v : [v])]
        end
        filter!(x -> x !== nothing && x > 0, op["structure_channels"])
        
        for d in ["medial", "lateral", "posterior", "anterior"]
            chs = Int[]
            for (bk, bv) in boundaries
                bk_lower = lowercase(bk)
                if bk_lower == d || startswith(bk_lower, d * "_")
                    b_list = bv isa Vector ? bv : [bv]
                    for s in b_list
                        c = side !== nothing ? resolve_channel_sided(state, String(s), side) : get_channel!(state, String(s))
                        if c !== nothing && c > 0
                            push!(chs, c)
                        end
                    end
                end
            end
            op["$(d)_channels"] = unique(chs)
        end
    elseif rule_type == "VolumetricConvexHull3D"
        if haskey(params, "target")
            tgt = String(params["target"])
            ch = side !== nothing ? resolve_channel_sided(state, tgt, side) : get_channel!(state, tgt)
            if ch !== nothing && ch > 0; op["target_channel"] = ch; end
        end
    elseif rule_type in ["ConvexHullBridge", "ConvexHullBridging"]
        lm1 = get(params, "landmark_1", "")
        lm2 = get(params, "landmark_2", "")
        if (isempty(lm1) || isempty(lm2)) && haskey(params, "landmarks") && length(params["landmarks"]) >= 2
            lm1 = params["landmarks"][1]
            lm2 = params["landmarks"][2]
        end
        if (isempty(lm1) || isempty(lm2)) && haskey(params, "bridge_landmarks")
            bl = params["bridge_landmarks"]
            if length(bl) >= 2
                lm1 = bl[1]
                lm2 = bl[2]
            elseif length(bl) == 1
                lm1 = bl[1]
                lm2 = bl[1]
            end
        end
        if isempty(lm1) && haskey(params, "input")
            lm1 = params["input"]
            lm2 = isempty(lm2) ? params["input"] : lm2
        end
        ch1 = side !== nothing ? resolve_channel_sided(state, String(lm1), side) : get_channel!(state, String(lm1))
        ch2 = side !== nothing ? resolve_channel_sided(state, String(lm2), side) : get_channel!(state, String(lm2))
        if ch1 !== nothing && ch1 > 0; op["landmark_1_channel"] = ch1; end
        if ch2 !== nothing && ch2 > 0; op["landmark_2_channel"] = ch2; end
    elseif rule_type == "GeometricPrimitive"
        op["primitive"] = get(params, "primitive", "Cylinder")
        op["radius_mm"] = get(params, "radius_mm", 20.0)
        for p_key in ["p1", "p2"]
            if haskey(params, p_key)
                p_def = params[p_key]
                lm = p_def isa Dict ? String(get(p_def, "landmark", "")) : String(p_def)
                ch = side !== nothing ? resolve_channel_sided(state, lm, side) : get_channel!(state, lm)
                if ch !== nothing && ch > 0
                    op["$(p_key)_channel"] = ch
                    if p_def isa Dict
                        op["$(p_key)_split"] = get(p_def, "split", "")
                        op["$(p_key)_direction"] = get(p_def, "direction", "")
                    end
                end
            end
        end
    elseif rule_type == "VolumetricBoundary2DLateralGrowth"
        if haskey(params, "obstacle")
            obs = String(params["obstacle"])
            ch = side !== nothing ? resolve_channel_sided(state, obs, side) : get_channel!(state, obs)
            if ch !== nothing && ch > 0; op["obstacle_channel"] = ch; end
        end
    elseif rule_type == "LimitZByLandmark"
        lm_str = String(get(params, "landmark", get(params, "reference", get(params, "input_landmarks", ""))))
        if !isempty(lm_str)
            ch = side !== nothing ? resolve_channel_sided(state, lm_str, side) : get_channel!(state, lm_str)
            if ch !== nothing && ch > 0; op["landmark_channel"] = ch; end
        end
    elseif rule_type == "SplitMask"
        input_name = String(get(params, "input", get(params, "landmark", "")))
        if !isempty(input_name)
            in_ch = side !== nothing ? resolve_channel_sided(state, input_name, side) : get_channel!(state, input_name)
            if in_ch !== nothing && in_ch > 0; op["input_channel"] = in_ch; end
        end
        op["axis"] = get(params, "axis", "x")
        op["split_method"] = get(params, "split_method", "center")
        op["keep"] = get(params, "keep", get(params, "side_to_keep", "min"))
    elseif rule_type == "GenericMorphologyDAG"
        inputs_list = get(params, "inputs", [])
        input_chs = Int[]
        for inp in inputs_list
            inp_str = inp isa Dict ? String(get(inp, "landmark", "")) : String(inp)
            if !isempty(inp_str)
                ch = side !== nothing ? resolve_channel_sided(state, inp_str, side) : get_channel!(state, inp_str)
                if ch !== nothing && ch > 0
                    push!(input_chs, ch)
                end
            end
        end
        op["input_channels"] = input_chs
        op["operations"] = get(params, "operations", [])
    end
    
    constraints = copy(get(params, "constraints", []))
    if haskey(params, "roi") && params["roi"] isa Dict
        roi = params["roi"]
        roi_lm = get(roi, "landmark", nothing)
        pad = get(roi, "padding_mm", [50.0, 50.0, 50.0])
        pad_x = Float64(pad[1])
        pad_y = Float64(length(pad) > 1 ? pad[2] : pad[1])
        pad_z = Float64(length(pad) > 2 ? pad[3] : pad[1])
        if roi_lm !== nothing
            push!(constraints, Dict{String, Any}("constraint_type" => "SuperiorTo", "landmark" => roi_lm, "boundary_part" => "min", "offset_mm" => -pad_z))
            push!(constraints, Dict{String, Any}("constraint_type" => "InferiorTo", "landmark" => roi_lm, "boundary_part" => "max", "offset_mm" => pad_z))
            push!(constraints, Dict{String, Any}("constraint_type" => "PosteriorTo", "landmark" => roi_lm, "boundary_part" => "min", "offset_mm" => -pad_y))
            push!(constraints, Dict{String, Any}("constraint_type" => "AnteriorTo", "landmark" => roi_lm, "boundary_part" => "max", "offset_mm" => pad_y))
            push!(constraints, Dict{String, Any}("constraint_type" => "LeftOf", "landmark" => roi_lm, "boundary_part" => "min", "offset_mm" => -pad_x))
            push!(constraints, Dict{String, Any}("constraint_type" => "RightOf", "landmark" => roi_lm, "boundary_part" => "max", "offset_mm" => pad_x))
        end
    end
    if haskey(params, "z_plane_restriction")
        zr = params["z_plane_restriction"]
        # Python defaults: superior_part="max", inferior_part="min"
        zr_bp_defaults = Dict("inferior" => "min", "superior" => "max")
        for (bound_key, constraint_dir) in [("inferior", "SuperiorTo"), ("superior", "InferiorTo")]
            if haskey(zr, bound_key)
                val = zr[bound_key]
                val_list = val isa Vector ? val : [val]
                for v in val_list
                    c = Dict{String, Any}("constraint_type" => constraint_dir, "fallback_landmark" => "sternum")
                    # Set default boundary_part matching Python z_plane_restriction defaults
                    c["boundary_part"] = zr_bp_defaults[bound_key]
                    if v isa Dict
                        c["landmark"] = get(v, "landmark", get(v, "organ", nothing))
                        if haskey(v, "part") c["boundary_part"] = v["part"] end
                        if haskey(v, "offset_mm") c["offset_mm"] = v["offset_mm"] end
                    else
                        c["landmark"] = v
                    end
                    # Explicit overrides from top-level zr keys
                    if haskey(zr, "$(bound_key)_part") c["boundary_part"] = zr["$(bound_key)_part"] end
                    if haskey(zr, "$(bound_key)_offset_mm") c["offset_mm"] = zr["$(bound_key)_offset_mm"] end
                    push!(constraints, c)
                end
            end
        end
    end

    if !isempty(constraints)
        julia_constraints = []
        for c in constraints
            jc = resolve_constraint_for_julia(state, c, params, side)
            if jc !== nothing
                if jc isa Vector
                    append!(julia_constraints, jc)
                else
                    push!(julia_constraints, jc)
                end
            else
                c_type = get(c, "constraint_type", "")
                c_lm = get(c, "landmark", get(c, "landmark_1", ""))
                println("    [DEBUG] Constraint $c_type with landmark '$c_lm' resolved to nothing for $(op["target"])")
            end
        end
        if !isempty(julia_constraints)
            op["constraints"] = julia_constraints
        end
    end
    
    if haskey(params, "layer_wise_propagation")
        lwp = copy(params["layer_wise_propagation"])
        if haskey(lwp, "target_landmark")
            t_names = lwp["target_landmark"]
            t_list = t_names isa Vector ? t_names : [t_names]
            t_channels = Int[]
            for t in t_list
                if t isa String
                    ch = resolve_channel_sided(state, t, side)
                    if ch !== nothing && ch > 0
                        push!(t_channels, ch)
                    end
                end
            end
            if !isempty(t_channels)
                lwp["target_channels"] = t_channels
                op["layer_wise_propagation"] = lwp
            end
        end
    end
    
    if haskey(params, "base_landmark") && params["base_landmark"] isa Dict
        bld = copy(params["base_landmark"])
        bld_parsed = Dict{String, Any}()
        for (k, v) in bld
            ch = resolve_channel_sided(state, k, side)
            if ch !== nothing
                bld_parsed[string(ch)] = v
            end
        end
        op["base_landmark_dict"] = bld_parsed
    end
    
    if rule_type == "PrimaryVector" && haskey(params, "target_landmark")
        v = params["target_landmark"]
        ch = resolve_channel_sided(state, v isa Vector ? v[1] : String(v), side)
        if ch !== nothing
            op["target_channel"] = ch
        end
    end
    
    return op
end

function execute_graph_level(rules::Dict, deps::Dict, level::Vector{String}, masks::Dict, dims::Tuple, spacing::Tuple, origin::Tuple, direction::Tuple, gen_masks::Dict=Dict(), computed_landmarks::Dict=Dict())
    state = DagState()
    state.dims = (Int(dims[1]), Int(dims[2]), Int(dims[3]))
    state.spacing = (Float64(spacing[1]), Float64(spacing[2]), Float64(spacing[3]))
    state.origin = (Float64(origin[1]), Float64(origin[2]), Float64(origin[3]))
    state.direction = direction
    state.computed_landmarks = computed_landmarks
    
    # Populate known masks from raw masks, generated masks, and rule names
    for k in keys(masks); push!(state.known_masks, k); end
    for k in keys(gen_masks); push!(state.known_masks, k); end
    for k in keys(rules); push!(state.known_masks, k); end
    
    all_operations = []
    all_bilateral_splits = []
    
    for node in level
        rule = rules[node]
        is_bilateral = get(rule, "is_bilateral", false)
        
        # If there are explicitly defined split rules in the JSONs, do NOT implicitly expand.
        if is_bilateral
            has_explicit_split = false
            for k in keys(rules)
                if lowercase(k) == lowercase(node * "_left") || lowercase(k) == lowercase(node * "_right")
                    has_explicit_split = true
                    break
                end
            end
            if has_explicit_split
                is_bilateral = false
            end
        end
        
        if is_bilateral
            for side in ["left", "right"]
                op = create_operation(state, node, rule, deps, side)
                split_c = Dict{String, Any}(
                    "type" => side == "left" ? "LeftOf" : "RightOf",
                    "axis" => 1,
                    "boundary_part" => "center",
                    "offset_mm" => 0.0,
                    "slice_wise" => false,
                    "limit_idx" => Int(round(dims[1] / 2))
                )
                op["constraints"] = vcat(get(op, "constraints", []), [split_c])
                push!(all_operations, op)
            end
        else
            op = create_operation(state, node, rule, deps, nothing)
            push!(all_operations, op)
        end
    end
    
    payload = Dict(
        "operations" => all_operations
    )
    if !isempty(all_bilateral_splits)
        payload["bilateral_splits"] = all_bilateral_splits
    end
    
    return JSON.json(payload), state.node_to_channel
end

function stratify_dag(overlap_dag::Vector{Dict}, mask_names::Vector{String})
    name_to_idx = Dict{String, Int}(name => i for (i, name) in enumerate(mask_names))
    deps = Dict{Int, Set{Int}}(i => Set{Int}() for i in 1:length(mask_names))
    
    for d in overlap_dag
        t_idx = name_to_idx[d["target"]]
        s_idx = name_to_idx[d["source"]]
        push!(deps[t_idx], s_idx)
    end
    
    levels = Vector{Vector{Dict}}()
    resolved = Set{Int}()
    
    while true
        current_level_nodes = Int[]
        for i in 1:length(mask_names)
            if !(i in resolved) && issubset(deps[i], resolved)
                push!(current_level_nodes, i)
            end
        end
        
        if isempty(current_level_nodes)
            if length(resolved) < length(mask_names)
                println("WARNING: Cycle detected in overlap_precedence DAG!")
            end
            break
        end
        
        current_level_ops = Dict[]
        for d in overlap_dag
            t_idx = name_to_idx[d["target"]]
            if t_idx in current_level_nodes
                s_idx = name_to_idx[d["source"]]
                push!(current_level_ops, Dict(
                    "type" => 0,
                    "target_idx" => t_idx,
                    "arg_idx" => s_idx
                ))
            end
        end
        
        push!(levels, current_level_ops)
        union!(resolved, current_level_nodes)
    end
    
    return levels, name_to_idx
end

function resolve_global_overlaps(rules::Dict, masks_dict::Dict)
    overlap_dag = Dict[]
    masks_to_load = Set{String}()
    
    for (rule_name, params) in rules
        precedence_list = get(params, "overlap_precedence", [])
        if !isempty(precedence_list)
            for prec_name in precedence_list
                push!(overlap_dag, Dict(
                    "type" => "subtract",
                    "target" => rule_name,
                    "source" => prec_name
                ))
                push!(masks_to_load, rule_name)
                push!(masks_to_load, prec_name)
            end
        end
    end
    
    if isempty(overlap_dag)
        return nothing
    end
    
    mask_names = collect(keys(masks_dict))
    
    new_rules = Dict[]
    for name in collect(masks_to_load)
        if !haskey(masks_dict, name)
            left_name = name * "_left"
            right_name = name * "_right"
            
            left_exists = haskey(masks_dict, left_name) || haskey(masks_dict, name * "_Left")
            right_exists = haskey(masks_dict, right_name) || haskey(masks_dict, name * "_Right")
            
            if left_exists || right_exists
                for d in copy(overlap_dag)
                    if d["target"] == name || d["source"] == name
                        filter!(x -> x != d, overlap_dag)
                        if left_exists
                            rule_l = copy(d)
                            if rule_l["target"] == name; rule_l["target"] = left_name; end
                            if rule_l["source"] == name; rule_l["source"] = left_name; end
                            push!(new_rules, rule_l)
                        end
                        if right_exists
                            rule_r = copy(d)
                            if rule_r["target"] == name; rule_r["target"] = right_name; end
                            if rule_r["source"] == name; rule_r["source"] = right_name; end
                            push!(new_rules, rule_r)
                        end
                    end
                end
            end
        end
    end
    append!(overlap_dag, new_rules)
    
    valid_dag = Dict[]
    for d in overlap_dag
        if haskey(masks_dict, d["target"]) && haskey(masks_dict, d["source"])
            push!(valid_dag, d)
        end
    end
    
    if isempty(valid_dag)
        return nothing
    end
    
    ops_levels, name_to_idx = stratify_dag(valid_dag, mask_names)
    
    payload = Dict(
        "rule" => "MegaFused",
        "ops_levels" => ops_levels,
        "inputs_map" => name_to_idx
    )
    
    return JSON.json(payload)
end

end
