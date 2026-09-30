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

function execute_mega2_presacral_anterior(batch_json_str)
    batch_dict = JSON.parse(batch_json_str)
    out_ch = Int32(batch_dict["out_ch"])
    props = batch_dict["props"]
    mapping = batch_dict["mapping"]
    
    sacrum_name = get(props, "landmark", "sacrum")
    distance_mm = get(props, "distance_mm", 10.0)
    output_type = get(props, "output_type", "zone")
    
    dims = TM_MEGA2.dims
    sp = TM_MEGA2.sp
    
    sacrum_m = get(mapping, sacrum_name, get(mapping, "helper_sacrum_fused", get(mapping, "sacrum", nothing)))
    if sacrum_m !== nothing
        sacrum_arr = get_mask_cpu(sacrum_m["ch"], dims) .== sacrum_m["id"]
    else
        sacrum_arr = falses(dims)
    end
    
    dist_voxels = round(Int, distance_mm / sp[2])
    out_arr = falses(dims)
    
    for z in 1:dims[3]
        for x in 1:dims[1]
            ys = [y for y in 1:dims[2] if sacrum_arr[x, y, z]]
            if !isempty(ys)
                min_y = minimum(ys)
                start_y = max(1, min_y - dist_voxels)
                
                if output_type == "front_line"
                    out_arr[x, min_y, z] = true
                elseif output_type == "moved_line"
                    out_arr[x, start_y, z] = true
                else
                    if start_y + 1 <= min_y - 1
                        out_arr[x, start_y+1:min_y-1, z] .= true
                    end
                end
            end
        end
    end
    
    out_uint8 = UInt8.(out_arr)
    insert_mega2_mask!(adapt(TM_MEGA2.backend, out_uint8), out_ch)
    return "Executed MegaV2 PresacralAnterior"
end
