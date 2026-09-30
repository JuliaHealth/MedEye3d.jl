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

function execute_mega2_station1(batch_json_str)
    batch_dict = JSON.parse(batch_json_str)
    out_ch = Int32(batch_dict["out_ch"])
    props = batch_dict["props"]
    mapping = batch_dict["mapping"]
    side = batch_dict["side"]
    
    dims = TM_MEGA2.dims
    
    # 1. Base input
    base_comp = get(props, "input", get(props, "base_landmark", ""))
    if isa(base_comp, Dict)
        base_comp = get(base_comp, "landmark", "")
    end
    base_m = mapping[base_comp]
    base_arr = get_mask_cpu(base_m["ch"], dims) .== base_m["id"]
    
    # 2. Get dependencies
    cricoid_m = get(mapping, "cricoid", get(mapping, "cricoid_cartilage", nothing))
    trachea_m = get(mapping, "trachea", nothing)
    manubrium_m = get(mapping, "manubrium", get(mapping, "sternum", nothing))
    clavicle_m = get(mapping, "clavicle", get(mapping, "clavicula", nothing))
    clav_l_m = get(mapping, "clavicula_left", get(mapping, "clavicle_left", nothing))
    clav_r_m = get(mapping, "clavicula_right", get(mapping, "clavicle_right", nothing))
    esophagus_m = get(mapping, "esophagus", nothing)
    thyroid_m = get(mapping, "thyroid", get(mapping, "thyroid_gland", nothing))
    lung_l_m = get(mapping, "lung_left", get(mapping, "lung_l", nothing))
    lung_r_m = get(mapping, "lung_right", get(mapping, "lung_r", nothing))
    scm_m = get(mapping, "sternocleidomastoid", get(mapping, "scm", nothing))
    scalene_m = get(mapping, "scalene_muscle", get(mapping, "scalene", nothing))
    
    function safe_load(m)
        if m !== nothing
            return get_mask_cpu(m["ch"], dims) .== m["id"]
        end
        return falses(dims)
    end
    
    cricoid_arr = safe_load(cricoid_m)
    trachea_arr = safe_load(trachea_m)
    manubrium_arr = safe_load(manubrium_m)
    clavicle_arr = safe_load(clavicle_m) .| safe_load(clav_l_m) .| safe_load(clav_r_m)
    
    esophagus_arr = safe_load(esophagus_m)
    thyroid_arr = safe_load(thyroid_m)
    lung_l_arr = safe_load(lung_l_m)
    lung_r_arr = safe_load(lung_r_m)
    scm_arr = safe_load(scm_m)
    scalene_arr = safe_load(scalene_m)
    
    # 1. Superior Margin (Cricoid Min Z)
    cricoid_z_min = dims[3]
    if any(cricoid_arr)
        for z in 1:dims[3]
            if any(view(cricoid_arr, :, :, z))
                cricoid_z_min = z
                break
            end
        end
    elseif any(trachea_arr)
        # If cricoid missing, top 20% of trachea
        zs_tr = [z for z in 1:dims[3] if any(view(trachea_arr, :, :, z))]
        if !isempty(zs_tr)
            cricoid_z_min = maximum(zs_tr)
        end
    end
    
    mask_superior = falses(dims)
    mask_superior[:, :, 1:cricoid_z_min] .= true
    
    # 2. Midline
    mask_side = falses(dims)
    global_mid_x = div(dims[1], 2)
    
    if any(trachea_arr)
        # find mean x of trachea
        xs = [x for z in 1:dims[3] for y in 1:dims[2] for x in 1:dims[1] if trachea_arr[x, y, z]]
        if !isempty(xs)
            global_mid_x = round(Int, sum(xs) / length(xs))
        end
    end
    
    for z in 1:dims[3]
        slice_tr = view(trachea_arr, :, :, z)
        if any(slice_tr)
            xs = [x for y in 1:dims[2] for x in 1:dims[1] if slice_tr[x, y]]
            mid_x_z = round(Int, sum(xs) / length(xs))
        else
            mid_x_z = global_mid_x
        end
        
        if lowercase(side) == "left"
            mask_side[mid_x_z:end, :, z] .= true
        elseif lowercase(side) == "right"
            mask_side[1:mid_x_z, :, z] .= true
        else
            mask_side[:, :, z] .= true
        end
    end
    
    # 3. Floor
    thoracic_bones = manubrium_arr .| clavicle_arr
    floor_z_full = zeros(Int, dims[1], dims[2])
    
    if any(thoracic_bones)
        any_true = falses(dims[1], dims[2])
        for x in 1:dims[1], y in 1:dims[2]
            # search from top (Z=dims[3]) downwards
            for z in dims[3]:-1:1
                if thoracic_bones[x, y, z]
                    any_true[x, y] = true
                    floor_z_full[x, y] = z
                    break
                end
            end
        end
        
        if !all(any_true)
            ft = feature_transform(any_true)
            floor_z_full = floor_z_full[ft]
        end
        
        floor_z_smooth = round.(Int, imfilter(Float64.(floor_z_full), Kernel.gaussian(2.0)))
    else
        floor_z_smooth = zeros(Int, dims[1], dims[2])
    end
    
    mask_inferior = falses(dims)
    for z in 1:dims[3], y in 1:dims[2], x in 1:dims[1]
        if z > floor_z_smooth[x, y]
            mask_inferior[x, y, z] = true
        end
    end
    
    # Combine
    out_arr = mask_superior .& mask_inferior .& mask_side .& base_arr
    
    out_arr .&= .!trachea_arr
    out_arr .&= .!esophagus_arr
    out_arr .&= .!thyroid_arr
    out_arr .&= .!lung_l_arr
    out_arr .&= .!lung_r_arr
    out_arr .&= .!scm_arr
    out_arr .&= .!scalene_arr
    
    out_uint8 = UInt8.(out_arr)
    insert_mega2_mask!(adapt(TM_MEGA2.backend, out_uint8), out_ch)
    return "Executed MegaV2 Station1"
end
