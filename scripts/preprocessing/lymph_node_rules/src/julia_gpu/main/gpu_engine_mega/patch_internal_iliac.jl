using JSON
using CUDA
using LinearAlgebra

function _parse_point_or_bbox(val, sp)
    if isa(val, Vector) || isa(val, Tuple)
        if length(val) == 3 && all(x -> isa(x, Number), val)
            return [Float64(val[1]), Float64(val[2]), Float64(val[3])]
        elseif length(val) == 3 && all(x -> (isa(x, Vector) || isa(x, Tuple)), val)
            p_idx = [(val[1][1] + val[1][2])/2.0, (val[2][1] + val[2][2])/2.0, (val[3][1] + val[3][2])/2.0]
            return [(p_idx[1] - 1.0) * sp[1], (p_idx[2] - 1.0) * sp[2], (p_idx[3] - 1.0) * sp[3]]
        end
    end
    return nothing
end

function execute_mega2_internal_iliac(batch_json_str)
    batch_dict = JSON.parse(batch_json_str)
    out_ch = Int32(batch_dict["out_ch"])
    props = batch_dict["props"]
    mapping = batch_dict["mapping"]
    side = get(batch_dict, "side", get(props, "side", ""))
    
    dims = TM_MEGA2.dims
    sp = TM_MEGA2.sp
    
    out_arr = falses(dims)
    
    p1 = _parse_point_or_bbox(get(props, "internal_iliac_p1", nothing), sp)
    p2 = _parse_point_or_bbox(get(props, "internal_iliac_p2", nothing), sp)
    
    if p1 !== nothing && p2 !== nothing
        direction = p2 .- p1
        norm_dir = norm(direction)
        if norm_dir > 0
            dir_vec = direction ./ norm_dir
            radius_mm = get(props, "radius_mm", 7.0)
            length_mm = get(props, "length_mm", 135.0)
            
            # Bound the search area
            for z in 1:dims[3], y in 1:dims[2], x in 1:dims[1]
                phys_pt = [(x - 1.0) * sp[1], (y - 1.0) * sp[2], (z - 1.0) * sp[3]]
                v = phys_pt .- p1
                proj = dot(v, dir_vec)
                
                if proj >= 0.0 && proj <= length_mm
                    dist2 = sum(v.^2) - proj^2
                    if dist2 <= radius_mm^2
                        out_arr[x, y, z] = true
                    end
                end
            end
        end
    end
    
    # Dynamic midline split using iliac_artery
    il_art_l_m = get(mapping, "iliac_artery_left", get(mapping, "iliac_artery_common_left", get(mapping, "iliac_artery_external_left", nothing)))
    il_art_r_m = get(mapping, "iliac_artery_right", get(mapping, "iliac_artery_common_right", get(mapping, "iliac_artery_external_right", nothing)))
    
    function safe_load(m)
        if m !== nothing
            ch = m["ch"]
            if ch > 0
                arr = adapt(CPU(), TM_MEGA2.tensor_in[ch, :, :, :])
            else
                arr = adapt(CPU(), TM_MEGA2.tensor_out[-ch, :, :, :])
            end
            return arr .== m["id"]
        end
        return falses(dims)
    end
    
    # Fallback to loading p1/p2 masks from mapping if not in props
    p1_pt = nothing
    p2_pt = nothing
    if haskey(props, "internal_iliac_p1") && haskey(props, "internal_iliac_p2")
        bb1 = props["internal_iliac_p1"]
        bb2 = props["internal_iliac_p2"]
        p1_idx = [(bb1[1][1] + bb1[1][2])/2.0, (bb1[2][1] + bb1[2][2])/2.0, (bb1[3][1] + bb1[3][2])/2.0]
        p2_idx = [(bb2[1][1] + bb2[1][2])/2.0, (bb2[2][1] + bb2[2][2])/2.0, (bb2[3][1] + bb2[3][2])/2.0]
        p1_pt = [p1_idx[1] * sp[1], p1_idx[2] * sp[2], p1_idx[3] * sp[3]]
        p2_pt = [p2_idx[1] * sp[1], p2_idx[2] * sp[2], p2_idx[3] * sp[3]]
    end

    
    il_art_l = safe_load(il_art_l_m)
    il_art_r = safe_load(il_art_r_m)
    
    for z in 1:dims[3]
        slice_mask = view(out_arr, :, :, z)
        if !any(slice_mask)
            continue
        end
        
        x_l_idx = [x for y in 1:dims[2] for x in 1:dims[1] if il_art_l[x, y, z]]
        x_r_idx = [x for y in 1:dims[2] for x in 1:dims[1] if il_art_r[x, y, z]]
        
        if !isempty(x_l_idx) && !isempty(x_r_idx)
            com_x_l = sum(x_l_idx) / length(x_l_idx)
            com_x_r = sum(x_r_idx) / length(x_r_idx)
            mid_x = round(Int, (com_x_l + com_x_r) / 2.0)
            
            left_is_high = com_x_l > com_x_r
            
            for y in 1:dims[2], x in 1:dims[1]
                if slice_mask[x, y]
                    if left_is_high
                        valid = lowercase(side) == "left" ? (x >= mid_x) : (x < mid_x)
                    else
                        valid = lowercase(side) == "left" ? (x < mid_x) : (x >= mid_x)
                    end
                    if !valid
                        out_arr[x, y, z] = false
                    end
                end
            end
        end
    end
    
    # Exclusions
    if haskey(props, "exclude")
        for exc in props["exclude"]
            if haskey(mapping, exc)
                exc_m = mapping[exc]
                exc_arr = safe_load(exc_m)
                out_arr .&= .!exc_arr
            end
        end
    end
    
    out_uint8 = UInt8.(out_arr)
    insert_mega2_mask!(adapt(TM_MEGA2.backend, out_uint8), out_ch)
    return "Executed MegaV2 InternalIliac"
end
