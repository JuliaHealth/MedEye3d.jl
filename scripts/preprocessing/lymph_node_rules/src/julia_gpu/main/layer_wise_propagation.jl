using ImageMorphology

function run_layer_wise_propagation_cpu!(arr::Array{UInt8, 3}, target_arr::Array{UInt8, 3}, 
                                         direction::String, step_dilation_px::Int, max_slices::Int)
    Z, Y, X = size(arr, 3), size(arr, 2), size(arr, 1)
    
    # Find seed edge (z index)
    has_seed = vec(any(arr .> 0, dims=(1, 2)))
    if !any(has_seed) || !any(target_arr .> 0)
        return
    end
    
    idxs = findall(has_seed)
    
    # We assume z increases towards superior.
    # So 'inferior' means decreasing Z.
    # Wait, in LPS, Z=1 is inferior. Z=end is superior.
    # So 'inferior' means we want to propagate from the lowest seed Z down to Z=1.
    if direction == "inferior"
        curr_z = minimum(idxs)
        step = -1
    else
        curr_z = maximum(idxs)
        step = 1
    end
    
    # Target Dilation (5px for robust hit detection, matching Python)
    target_dil = copy(target_arr)
    for _ in 1:5
        tmp = copy(target_dil)
        for k in 1:Z, j in 1:Y, i in 1:X
            if target_dil[i, j, k] > 0
                if i > 1 tmp[i-1, j, k] = 1 end
                if i < X tmp[i+1, j, k] = 1 end
                if j > 1 tmp[i, j-1, k] = 1 end
                if j < Y tmp[i, j+1, k] = 1 end
            end
        end
        target_dil = tmp
    end
    
    slices_propagated = 0
    curr_mask_arr = arr[:, :, curr_z]
    
    while (1 <= curr_z + step <= Z) && (slices_propagated < max_slices)
        curr_z += step
        slices_propagated += 1
        
        next_mask_arr = copy(curr_mask_arr)
        
        # Dilation
        if step_dilation_px > 0
            for _ in 1:step_dilation_px
                tmp = copy(next_mask_arr)
                for j in 1:Y, i in 1:X
                    if next_mask_arr[i, j] > 0
                        if i > 1 tmp[i-1, j] = UInt8(1) end
                        if i < X tmp[i+1, j] = UInt8(1) end
                        if j > 1 tmp[i, j-1] = UInt8(1) end
                        if j < Y tmp[i, j+1] = UInt8(1) end
                    end
                end
                next_mask_arr = tmp
            end
        end
        
        # Check target hit (against dilated target)
        target_slice = target_dil[:, :, curr_z]
        if any((next_mask_arr .> 0) .& (target_slice .> 0))
            arr[:, :, curr_z] .= next_mask_arr
            break
        end
        
        arr[:, :, curr_z] .= next_mask_arr
        curr_mask_arr = next_mask_arr
    end
end
