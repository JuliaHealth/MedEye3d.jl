using JSON
using CUDA
using Images
using ImageMorphology

function execute_mega2_iliac_bifurcation(batch_json_str)
    batch_dict = JSON.parse(batch_json_str)
    out_ch = Int32(batch_dict["out_ch"])
    props = batch_dict["props"]
    mapping = batch_dict["mapping"]
    side = get(batch_dict, "side", get(props, "side", ""))
    
    dims = TM_MEGA2.dims
    sp = TM_MEGA2.sp
    
    # Load masks
    # Load masks
    il_art_l_m = get(mapping, "iliac_artery_left", get(mapping, "iliac_artery_common_left", get(mapping, "iliac_artery_external_left", nothing)))
    il_art_r_m = get(mapping, "iliac_artery_right", get(mapping, "iliac_artery_common_right", get(mapping, "iliac_artery_external_right", nothing)))
    il_ven_l_m = get(mapping, "iliac_vena_left", get(mapping, "iliac_vena_common_left", get(mapping, "iliac_vena_external_left", nothing)))
    il_ven_r_m = get(mapping, "iliac_vena_right", get(mapping, "iliac_vena_common_right", get(mapping, "iliac_vena_external_right", nothing)))
    
    function safe_load(m)
        if m !== nothing
            return get_mask_cpu(m["ch"], dims) .== m["id"]
        end
        return falses(dims)
    end
    
    il_art_l = safe_load(il_art_l_m)
    il_art_r = safe_load(il_art_r_m)
    il_ven_l = safe_load(il_ven_l_m)
    il_ven_r = safe_load(il_ven_r_m)
    
    art_arr = lowercase(side) == "left" ? il_art_l : il_art_r
    ven_arr = lowercase(side) == "left" ? il_ven_l : il_ven_r
    
    # Z-bounds +/- 10mm around internal_iliac_p1
    z_bif = 1
    if haskey(props, "internal_iliac_p1_z")
        z_bif = Int(props["internal_iliac_p1_z"])
    elseif haskey(props, "internal_iliac_p1")
        bb = props["internal_iliac_p1"]
        z_bif = round(Int, (bb[3][1] + bb[3][2]) / 2.0)
    end
    
    z_offset_idx = max(1, round(Int, 10.0 / sp[3]))
    z_min = max(1, z_bif - z_offset_idx)
    z_max = min(dims[3], z_bif + z_offset_idx)
    
    # Base dilation
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
    
    # Exclusions
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
    return "Executed MegaV2 IliacBifurcation"
end
