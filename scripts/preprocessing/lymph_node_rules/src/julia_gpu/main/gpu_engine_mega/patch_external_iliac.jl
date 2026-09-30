using JSON
using CUDA
using Images
using ImageMorphology

function get_mask_cpu(ch, dims)
    if ch > 0
        arr = adapt(CPU(), TM_MEGA2.tensor_in[ch, :, :, :])
    else
        arr = adapt(CPU(), TM_MEGA2.tensor_out[-ch, :, :, :])
    end
    return arr
end

function get_z_max(arr)
    dims = size(arr)
    for z in dims[3]:-1:1
        if any(view(arr, :, :, z))
            return z
        end
    end
    return dims[3]
end

function get_z_from_point(arr)
    # The 'point' is just a mask with a single voxel, or a small blob.
    # We find the mean z.
    dims = size(arr)
    zs = [z for z in 1:dims[3] if any(view(arr, :, :, z))]
    if isempty(zs)
        return 1
    end
    return round(Int, sum(zs) / length(zs))
end

function execute_mega2_external_iliac(batch_json_str)
    batch_dict = JSON.parse(batch_json_str)
    out_ch = Int32(batch_dict["out_ch"])
    props = batch_dict["props"]
    mapping = batch_dict["mapping"]
    side = get(batch_dict, "side", get(props, "side", ""))
    
    dims = TM_MEGA2.dims
    sp = TM_MEGA2.sp
    
    # Load masks
    femur_m = get(mapping, "femur", nothing)
    femur_l_m = get(mapping, "femur_left", nothing)
    femur_r_m = get(mapping, "femur_right", nothing)
    
    il_art_l_m = get(mapping, "iliac_artery_external_left", get(mapping, "iliac_artery_left", nothing))
    il_art_r_m = get(mapping, "iliac_artery_external_right", get(mapping, "iliac_artery_right", nothing))
    il_ven_l_m = get(mapping, "iliac_vena_external_left", get(mapping, "iliac_vena_left", nothing))
    il_ven_r_m = get(mapping, "iliac_vena_external_right", get(mapping, "iliac_vena_right", nothing))
    
    function safe_load(m)
        if m !== nothing
            return get_mask_cpu(m["ch"], dims) .== m["id"]
        end
        return falses(dims)
    end
    
    femur_arr = safe_load(femur_m)
    if !any(femur_arr)
        femur_arr = lowercase(side) == "left" ? safe_load(femur_l_m) : safe_load(femur_r_m)
        if !any(femur_arr)
            femur_arr = safe_load(femur_l_m) .| safe_load(femur_r_m)
        end
    end
    il_art_l = safe_load(il_art_l_m)
    il_art_r = safe_load(il_art_r_m)
    il_ven_l = safe_load(il_ven_l_m)
    il_ven_r = safe_load(il_ven_r_m)
    
    art_arr = lowercase(side) == "left" ? il_art_l : il_art_r
    ven_arr = lowercase(side) == "left" ? il_ven_l : il_ven_r
    
    # 2. Z-plane bounds
    z_femur = dims[3]
    if any(femur_arr)
        z_femur = get_z_max(femur_arr)
    end
    
    z_bif = 1
    if haskey(props, "internal_iliac_p1_z")
        z_bif = Int(props["internal_iliac_p1_z"])
    end
    
    z_min = min(z_femur, z_bif)
    z_max = max(z_femur, z_bif)
    
    # 3. Base dilation
    combined = art_arr .| ven_arr
    padding_mm = get(props, "padding_mm", 7.0)
    
    out_arr = falses(dims)
    
    if any(combined)
        # Compute distance transform
        bg = .!combined
        ft = feature_transform(combined)
        
        for z in 1:dims[3]
            if z < z_min || z > z_max
                continue
            end
            
            slice_com = view(combined, :, :, z)
            if !any(slice_com)
                continue
            end
            
            # Midline split
            x_l_idx = [x for y in 1:dims[2] for x in 1:dims[1] if il_art_l[x, y, z]]
            x_r_idx = [x for y in 1:dims[2] for x in 1:dims[1] if il_art_r[x, y, z]]
            
            mid_x = 0
            if !isempty(x_l_idx) && !isempty(x_r_idx)
                com_x_l = sum(x_l_idx) / length(x_l_idx)
                com_x_r = sum(x_r_idx) / length(x_r_idx)
                mid_x = round(Int, (com_x_l + com_x_r) / 2.0)
                
                # left side (in python, com_x_l > com_x_r means LPS coords? actually X is right-to-left)
                # wait, x_l_idx > x_r_idx means left is higher index.
                left_is_high = com_x_l > com_x_r
            else
                mid_x = -1
                left_is_high = true
            end
            
            for y in 1:dims[2], x in 1:dims[1]
                if !bg[x, y, z] || begin
                        p = ft[x, y, z]
                        dx = (x - p[1]) * sp[1]
                        dy = (y - p[2]) * sp[2]
                        dz = (z - p[3]) * sp[3]
                        sqrt(dx^2 + dy^2 + dz^2) <= padding_mm
                    end
                    
                    if mid_x != -1
                        if left_is_high
                            valid = lowercase(side) == "left" ? (x >= mid_x) : (x < mid_x)
                        else
                            valid = lowercase(side) == "left" ? (x < mid_x) : (x >= mid_x)
                        end
                        if valid
                            out_arr[x, y, z] = true
                        end
                    else
                        out_arr[x, y, z] = true
                    end
                end
            end
        end
    end
    
    # 5. Global exclusions
    if haskey(props, "exclude")
        for exc in props["exclude"]
            if haskey(mapping, exc)
                exc_m = mapping[exc]
                exc_arr = get_mask_cpu(exc_m["ch"], dims) .== exc_m["id"]
                out_arr .&= .!exc_arr
            end
        end
    end
    
    out_uint8 = UInt8.(out_arr)
    insert_mega2_mask!(adapt(TM_MEGA2.backend, out_uint8), out_ch)
    return "Executed MegaV2 ExternalIliac"
end
