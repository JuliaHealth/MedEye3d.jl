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

function execute_mega2_pleural_space(batch_json_str)
    batch_dict = JSON.parse(batch_json_str)
    out_ch = Int32(batch_dict["out_ch"])
    props = batch_dict["props"]
    mapping = batch_dict["mapping"]
    side = get(batch_dict, "side", get(props, "side", ""))
    
    lung_name = get(props, "lung_landmark", "")
    if lung_name == ""
        eff_side = side == "" ? "left" : lowercase(side)
        lung_name = "lung_" * eff_side
    end
    
    distance_lateral_mm = get(props, "distance_lateral_mm", get(props, "distance_mm", 20.0))
    distance_medial_mm = get(props, "distance_medial_mm", 10.0)
    
    eff_side = side != "" ? lowercase(side) : (occursin("left", lung_name) ? "left" : "right")
    
    dims = TM_MEGA2.dims
    sp = TM_MEGA2.sp
    
    lung_m = get(mapping, lung_name, nothing)
    if lung_m !== nothing
        lung_arr = get_mask_cpu(lung_m["ch"], dims) .== lung_m["id"]
    else
        lung_arr = falses(dims)
    end
    
    shift_lateral_voxels = round(Int, distance_lateral_mm / sp[1])
    shift_medial_voxels = round(Int, distance_medial_mm / sp[1])
    
    out_arr = falses(dims)
    
    for z in 1:dims[3]
        slice_lung = view(lung_arr, :, :, z)
        if !any(slice_lung)
            continue
        end
        
        y_indices = [y for x in 1:dims[1] for y in 1:dims[2] if slice_lung[x, y]]
        unique_ys = unique(y_indices)
        
        for y in unique_ys
            row_xs = [x for x in 1:dims[1] if slice_lung[x, y]]
            if !isempty(row_xs)
                if eff_side == "left"
                    x_lateral = maximum(row_xs)
                    x_moved_lateral = min(dims[1], x_lateral + shift_lateral_voxels)
                    x_moved_medial = max(1, x_lateral - shift_medial_voxels)
                    out_arr[x_moved_medial:x_moved_lateral, y, z] .= true
                else
                    x_lateral = minimum(row_xs)
                    x_moved_lateral = max(1, x_lateral - shift_lateral_voxels)
                    x_moved_medial = min(dims[1], x_lateral + shift_medial_voxels)
                    out_arr[x_moved_lateral:x_moved_medial, y, z] .= true
                end
            end
        end
    end
    
    out_uint8 = UInt8.(out_arr)
    insert_mega2_mask!(adapt(TM_MEGA2.backend, out_uint8), out_ch)
    return "Executed MegaV2 PleuralSpace"
end
