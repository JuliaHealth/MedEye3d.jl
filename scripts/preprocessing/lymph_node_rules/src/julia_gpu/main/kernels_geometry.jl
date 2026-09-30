using KernelAbstractions
using KernelAbstractions: @index, @localmem, @synchronize

@kernel function z_plane_clip_kernel!(output, input, z_min, z_max)
    I, J, K = @index(Global, NTuple)
    
    if I <= size(output, 1) && J <= size(output, 2) && K <= size(output, 3)
        idx = I + (J-1)*size(output, 1) + (K-1)*size(output, 1)*size(output, 2)
        if K >= z_min && K <= z_max
            output[idx] = input[idx]
        else
            output[idx] = 0
        end
    end
end

function run_z_plane_clip!(backend, output, input, z_min, z_max)
    kernel! = z_plane_clip_kernel!(backend)
    kernel!(output, input, z_min, z_max, ndrange=size(output))
    KernelAbstractions.synchronize(backend)
end

@kernel function lateral_bridge_kernel!(output, base_mask, target_mask, dims, max_pix, x_center, y_search_min, y_search_max, slice_has_base)
    # 2D Grid over Y and Z
    J, K = @index(Global, NTuple)
    
    if J <= dims[2] && K <= dims[3]
        s_min = dims[1] + 1
        s_max = 0
        has_base = false
        has_target = false
        
        for i in 1:dims[1]
            idx = i + (J-1)*dims[1] + (K-1)*dims[1]*dims[2]
            output[idx] = base_mask[idx]
            
            if base_mask[idx] > 0
                s_min = min(s_min, i)
                s_max = max(s_max, i)
                has_base = true
            end
            if target_mask[idx] > 0
                has_target = true
            end
        end
        
        if slice_has_base[K] == 1
            if has_base
                # Bridge to target
                t_left = 0
                for i in s_min-1:-1:1
                    idx = i + (J-1)*dims[1] + (K-1)*dims[1]*dims[2]
                    if target_mask[idx] > 0
                        t_left = i
                        break
                    end
                end
                
                if t_left > 0 && (s_min - t_left) <= max_pix
                    for i in t_left+1:s_min-1
                        idx = i + (J-1)*dims[1] + (K-1)*dims[1]*dims[2]
                        output[idx] = 1
                    end
                end
                
                t_right = dims[1] + 1
                for i in s_max+1:dims[1]
                    idx = i + (J-1)*dims[1] + (K-1)*dims[1]*dims[2]
                    if target_mask[idx] > 0
                        t_right = i
                        break
                    end
                end
                
                if t_right <= dims[1] && (t_right - s_max) <= max_pix
                    for i in s_max+1:t_right-1
                        idx = i + (J-1)*dims[1] + (K-1)*dims[1]*dims[2]
                        output[idx] = 1
                    end
                end
            end
        else
            if has_target && J >= y_search_min && J <= y_search_max
                # Gap closure across midline
                t_left = 0
                for i in x_center:-1:1
                    idx = i + (J-1)*dims[1] + (K-1)*dims[1]*dims[2]
                    if target_mask[idx] > 0
                        t_left = i
                        break
                    end
                end
                
                t_right = dims[1] + 1
                for i in x_center+1:dims[1]
                    idx = i + (J-1)*dims[1] + (K-1)*dims[1]*dims[2]
                    if target_mask[idx] > 0
                        t_right = i
                        break
                    end
                end
                
                if t_left > 0 && t_right <= dims[1]
                    dist = t_right - t_left
                    if dist <= max_pix * 1.5
                        for i in t_left:t_right-1
                            idx = i + (J-1)*dims[1] + (K-1)*dims[1]*dims[2]
                            output[idx] = 1
                        end
                    end
                end
            end
        end
    end
end

function run_lateral_bridge!(backend, output, base_mask_host, target_mask_device, sp_x, max_mm)
    dims = size(base_mask_host)
    max_pix = Int(round(max_mm / sp_x))
    x_center = dims[1] ÷ 2
    
    # Calculate slice_has_base and y_search_min/max on host
    slice_has_base_host = zeros(UInt8, dims[3])
    y_min_base = dims[2] + 1
    y_max_base = 0
    
    for K in 1:dims[3]
        for J in 1:dims[2]
            for I in 1:dims[1]
                if base_mask_host[I, J, K] > 0
                    slice_has_base_host[K] = 1
                    y_min_base = min(y_min_base, J)
                    y_max_base = max(y_max_base, J)
                end
            end
        end
    end
    
    buffer = 30
    y_search_min = max(1, y_min_base - buffer)
    y_search_max = min(dims[2], y_max_base + buffer)
    
    slice_has_base_dev = adapt(backend, slice_has_base_host)
    base_mask_dev = adapt(backend, base_mask_host)
    
    kernel! = lateral_bridge_kernel!(backend)
    # Launch 2D grid over Y, Z
    kernel!(output, base_mask_dev, target_mask_device, dims, max_pix, x_center, y_search_min, y_search_max, slice_has_base_dev, ndrange=(dims[2], dims[3]))
    KernelAbstractions.synchronize(backend)
end

@kernel function primary_vector_kernel!(output, primary_mask, dims, sp_x, sp_y, sp_z, base_margin, exp_margin, cx, cy, cz, dx, dy, dz)
    I, J, K = @index(Global, NTuple)
    
    if I <= dims[1] && J <= dims[2] && K <= dims[3]
        idx = I + (J-1)*dims[1] + (K-1)*dims[1]*dims[2]
        if primary_mask[idx] > 0
            output[idx] = 1
        else
            val = 0
            
            is_half_space = (I - cx)*dx + (J - cy)*dy + (K - cz)*dz > 0
            r_max = is_half_space ? max(base_margin, exp_margin) : base_margin
            
            if r_max > 0
                r_pix_x = Int(ceil(r_max / sp_x))
                r_pix_y = Int(ceil(r_max / sp_y))
                r_pix_z = Int(ceil(r_max / sp_z))
                
                min_x = max(1, I - r_pix_x)
                max_x = min(dims[1], I + r_pix_x)
                min_y = max(1, J - r_pix_y)
                max_y = min(dims[2], J + r_pix_y)
                min_z = max(1, K - r_pix_z)
                max_z = min(dims[3], K + r_pix_z)
                
                for k_off in min_z:max_z
                    for j_off in min_y:max_y
                        for i_off in min_x:max_x
                            n_idx = i_off + (j_off-1)*dims[1] + (k_off-1)*dims[1]*dims[2]
                            if primary_mask[n_idx] > 0
                                dist = sqrt( ((i_off - I)*sp_x)^2 + ((j_off - J)*sp_y)^2 + ((k_off - K)*sp_z)^2 )
                                if dist <= r_max
                                    val = 1
                                    break
                                end
                            end
                        end
                        if val == 1 break end
                    end
                    if val == 1 break end
                end
            end
            
            output[idx] = val
        end
    end
end

function run_primary_vector!(backend, output, primary_host, vector_host, sp_x, sp_y, sp_z, base_margin, exp_margin)
    dims = size(primary_host)
    
    com_p = [0.0, 0.0, 0.0]
    count_p = 0
    com_v = [0.0, 0.0, 0.0]
    count_v = 0
    
    for K in 1:dims[3], J in 1:dims[2], I in 1:dims[1]
        if primary_host[I,J,K] > 0
            com_p[1] += I
            com_p[2] += J
            com_p[3] += K
            count_p += 1
        end
        if vector_host[I,J,K] > 0
            com_v[1] += I
            com_v[2] += J
            com_v[3] += K
            count_v += 1
        end
    end
    
    if count_p > 0
        com_p ./= count_p
    end
    if count_v > 0
        com_v ./= count_v
    end
    
    dir_v = com_v .- com_p
    norm_v = sqrt(sum(dir_v.^2))
    if norm_v > 0
        dir_v ./= norm_v
    end
    
    primary_dev = adapt(backend, primary_host)
    
    kernel! = primary_vector_kernel!(backend)
    kernel!(output, primary_dev, dims, 
            Float32(sp_x), Float32(sp_y), Float32(sp_z),
            Float32(base_margin), Float32(exp_margin),
            Float32(com_p[1]), Float32(com_p[2]), Float32(com_p[3]),
            Float32(dir_v[1]), Float32(dir_v[2]), Float32(dir_v[3]),
            ndrange=dims)
    KernelAbstractions.synchronize(backend)
end

function resolve_extreme_point(arr, split_mode, direction, is_left_side, sp_x, sp_y, sp_z, dir_00)
    dims = size(arr)
    
    # Calculate median X for medial/lateral split
    median_x = 0
    if split_mode == "medial" || split_mode == "lateral"
        xs = Int[]
        for K in 1:dims[3], J in 1:dims[2], I in 1:dims[1]
            if arr[I,J,K] > 0
                push!(xs, I)
            end
        end
        if !isempty(xs)
            sort!(xs)
            median_x = xs[div(length(xs), 2) + 1]
        end
    end
    
    # direction corresponds to ITK LPS. In our grid:
    # X: Right to Left (if dir_00 = 1.0)
    # Y: Anterior to Posterior
    # Z: Inferior to Superior
    
    extreme_val = (direction == "anterior" || direction == "inferior" || direction == "right") ? typemax(Int) : typemin(Int)
    extreme_pt = nothing
    
    is_ras = (dir_00 < 0) # Just an approximation for the split logic used in python
    
    for K in 1:dims[3], J in 1:dims[2], I in 1:dims[1]
        if arr[I,J,K] > 0
            # Apply split condition
            if split_mode == "medial" || split_mode == "lateral"
                is_left_of_cut = I < median_x
                
                lat_cond = is_left_side ? (is_ras ? is_left_of_cut : !is_left_of_cut) : (is_ras ? !is_left_of_cut : is_left_of_cut)
                med_cond = !lat_cond
                
                if split_mode == "medial" && !med_cond
                    continue
                end
                if split_mode == "lateral" && !lat_cond
                    continue
                end
            end
            
            # Check extremity
            val = (direction == "anterior" || direction == "posterior") ? J :
                  (direction == "inferior" || direction == "superior") ? K : I
                  
            if (direction == "anterior" || direction == "inferior" || direction == "right")
                if val < extreme_val
                    extreme_val = val
                    extreme_pt = (I, J, K)
                end
            else
                if val > extreme_val
                    extreme_val = val
                    extreme_pt = (I, J, K)
                end
            end
        end
    end
    
    return extreme_pt
end

@kernel function cylinder_primitive_kernel!(output, p1_x, p1_y, p1_z, p2_x, p2_y, p2_z, r_x, r_y, r_z)
    I, J, K = @index(Global, NTuple)
    
    if I <= size(output, 1) && J <= size(output, 2) && K <= size(output, 3)
        Lx = p2_x - p1_x
        Ly = p2_y - p1_y
        Lz = p2_z - p1_z
        L2 = Lx*Lx + Ly*Ly + Lz*Lz
        
        if L2 > 0
            Vx = Float32(I - p1_x)
            Vy = Float32(J - p1_y)
            Vz = Float32(K - p1_z)
            t = (Vx*Lx + Vy*Ly + Vz*Lz) / L2
            t = max(0.0f0, min(1.0f0, t))
            
            Cx = p1_x + t*Lx
            Cy = p1_y + t*Ly
            Cz = p1_z + t*Lz
            
            dx = (Float32(I) - Cx) / r_x
            dy = (Float32(J) - Cy) / r_y
            dz = (Float32(K) - Cz) / r_z
            
            dist2 = dx*dx + dy*dy + dz*dz
            if dist2 <= 1.0f0
                output[I, J, K] = 1
            end
        end
    end
end
