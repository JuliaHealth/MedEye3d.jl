using JSON
using CUDA

function get_mask_cpu(ch, dims)
    if ch > 0
        arr = adapt(CPU(), TM_MEGA2.tensor_in[ch, :, :, :])
    else
        arr = adapt(CPU(), TM_MEGA2.tensor_out[-ch, :, :, :])
    end
    return arr
end

function execute_mega2_axillary_helper_b(batch_json_str)
    batch_dict = JSON.parse(batch_json_str)
    out_ch = Int32(batch_dict["out_ch"])
    props = batch_dict["props"]
    mapping = batch_dict["mapping"]
    side = get(batch_dict, "side", get(props, "side", ""))
    
    dims = TM_MEGA2.dims
    sp = TM_MEGA2.sp
    
    pec_name = get(props, "pectoralis_major", "")
    sub_name = get(props, "subscapularis", "")
    pec_m = get(mapping, pec_name, nothing)
    sub_m = get(mapping, sub_name, nothing)
    
    function safe_load(m)
        if m !== nothing
            if m["id"] == 0
                return get_mask_cpu(m["ch"], dims) .> 0
            else
                return get_mask_cpu(m["ch"], dims) .== m["id"]
            end
        end
        return falses(dims)
    end
    
    pec_arr = safe_load(pec_m)
    sub_arr = safe_load(sub_m)
    
    exc_arr = falses(dims)
    if haskey(props, "exclude")
        for exc_name in props["exclude"]
            if haskey(mapping, exc_name)
                em = mapping[exc_name]
                if em["id"] == 0
                    exc_arr .|= (get_mask_cpu(em["ch"], dims) .> 0)
                else
                    exc_arr .|= (get_mask_cpu(em["ch"], dims) .== em["id"])
                end
            end
        end
    end
    
    x_spacing = sp[1]
    medial_vox = round(Int, 40.0 / x_spacing)
    lateral_vox = round(Int, 50.0 / x_spacing)
    
    out_arr = falses(dims)
    
    for z in 1:dims[3]
        pec_sl = view(pec_arr, :, :, z)
        sub_sl = view(sub_arr, :, :, z)
        
        if !any(pec_sl) || !any(sub_sl)
            continue
        end
        
        pec_ys = [y for x in 1:dims[1] for y in 1:dims[2] if pec_sl[x, y]]
        sub_ys = [y for x in 1:dims[1] for y in 1:dims[2] if sub_sl[x, y]]
        
        # larger Y = more posterior. 
        # In python: pec_y_post = int(np.max(np.where(pec_sl)[0])) -> max Y
        pec_y_post = maximum(pec_ys)
        sub_y_ant = minimum(sub_ys)
        
        if sub_y_ant <= pec_y_post
            continue
        end
        
        pec_xs = [x for x in 1:dims[1] for y in 1:dims[2] if pec_sl[x, y]]
        sub_xs = [x for x in 1:dims[1] for y in 1:dims[2] if sub_sl[x, y]]
        
        pec_x_centroid = round(Int, sum(pec_xs) / length(pec_xs))
        sub_x_centroid = round(Int, sum(sub_xs) / length(sub_xs))
        x_ref = div(pec_x_centroid + sub_x_centroid, 2)
        
        if lowercase(side) == "left"
            x_medial = x_ref - medial_vox
            x_lateral = x_ref + lateral_vox
        else
            x_medial = x_ref + medial_vox
            x_lateral = x_ref - lateral_vox
            x_medial, x_lateral = min(x_medial, x_lateral), max(x_medial, x_lateral)
        end
        
        x_lo = max(1, min(x_medial, x_lateral))
        x_hi = min(dims[1], max(x_medial, x_lateral))
        
        for y in pec_y_post:sub_y_ant
            for x in x_lo:x_hi
                if !exc_arr[x, y, z]
                    out_arr[x, y, z] = true
                end
            end
        end
    end
    
    out_uint8 = UInt8.(out_arr)
    insert_mega2_mask!(adapt(TM_MEGA2.backend, out_uint8), out_ch)
    return "Executed MegaV2 AxillaryHelperB"
end
