using KernelAbstractions
using KernelAbstractions: @index, @localmem, @synchronize

@kernel function anisotropic_dilation_kernel!(output, input, dims, sp_x, sp_y, sp_z, m_pos_x, m_neg_x, m_pos_y, m_neg_y, m_pos_z, m_neg_z, max_rad_x, max_rad_y, max_rad_z, offset_x, offset_y, offset_z)
    I_local, J_local, K_local = @index(Global, NTuple)
    I = I_local + offset_x - 1
    J = J_local + offset_y - 1
    K = K_local + offset_z - 1
    
    if I <= dims[1] && J <= dims[2] && K <= dims[3]
        if input[I, J, K] > 0
            output[I, J, K] = UInt8(1)
        else
            val = UInt8(0)
            
            x_min = max(1, I - max_rad_x)
            x_max = min(dims[1], I + max_rad_x)
            y_min = max(1, J - max_rad_y)
            y_max = min(dims[2], J + max_rad_y)
            z_min = max(1, K - max_rad_z)
            z_max = min(dims[3], K + max_rad_z)
            
            for k_off in z_min:z_max
                for j_off in y_min:y_max
                    for i_off in x_min:x_max
                        if input[i_off, j_off, k_off] > 0
                            dx = Float32((I - i_off) * sp_x)
                            dy = Float32((J - j_off) * sp_y)
                            dz = Float32((K - k_off) * sp_z)
                            
                            m_x = dx > 0 ? m_pos_x : m_neg_x
                            m_y = dy > 0 ? m_pos_y : m_neg_y
                            m_z = dz > 0 ? m_pos_z : m_neg_z
                            
                            nx = m_x > 0 ? (dx / m_x) : (dx == 0 ? 0.0f0 : 1000.0f0)
                            ny = m_y > 0 ? (dy / m_y) : (dy == 0 ? 0.0f0 : 1000.0f0)
                            nz = m_z > 0 ? (dz / m_z) : (dz == 0 ? 0.0f0 : 1000.0f0)
                            dist_sq = nx*nx + ny*ny + nz*nz
                            
                            if dist_sq <= 1.0f0
                                val = UInt8(1)
                                break
                            end
                        end
                    end
                    if val == UInt8(1) break end
                end
                if val == UInt8(1) break end
            end
            
            output[I, J, K] = val
        end
    end
end

using Adapt

function run_anisotropic_dilation!(backend, output, input, sp_x, sp_y, sp_z, margins)
    all_default = Float32(get(margins, "all", 0.0))
    m_pos_x = Float32(get(margins, "left", all_default))
    m_neg_x = Float32(get(margins, "right", all_default))
    m_pos_y = Float32(get(margins, "posterior", all_default))
    m_neg_y = Float32(get(margins, "anterior", all_default))
    m_pos_z = Float32(get(margins, "superior", all_default))
    m_neg_z = Float32(get(margins, "inferior", all_default))
    
    max_rx = max(m_pos_x, m_neg_x)
    max_ry = max(m_pos_y, m_neg_y)
    max_rz = max(m_pos_z, m_neg_z)
    
    max_rad_x = ceil(Int, max_rx / max(0.001f0, sp_x))
    max_rad_y = ceil(Int, max_ry / max(0.001f0, sp_y))
    max_rad_z = ceil(Int, max_rz / max(0.001f0, sp_z))
    
    dims = size(output)
    
    # GPU-native bounding box via 1D projections (~6 KB transfer instead of 131 MB)
    if input isa Array
        # CPU path
        in_host = input
        idx = findall(in_host .> 0)
        if isempty(idx)
            fill!(output, 0)
            return
        end
        min_x = maximum([1, minimum(i[1] for i in idx) - max_rad_x])
        max_x = minimum([dims[1], maximum(i[1] for i in idx) + max_rad_x])
        min_y = maximum([1, minimum(i[2] for i in idx) - max_rad_y])
        max_y = minimum([dims[2], maximum(i[2] for i in idx) + max_rad_y])
        min_z = maximum([1, minimum(i[3] for i in idx) - max_rad_z])
        max_z = minimum([dims[3], maximum(i[3] for i in idx) + max_rad_z])
    else
        # GPU: project along each axis to find bounds without full download
        x_proj = Array(dropdims(any(input .> UInt8(0), dims=(2,3)), dims=(2,3)))  # 1D, ~2 KB
        y_proj = Array(dropdims(any(input .> UInt8(0), dims=(1,3)), dims=(1,3)))  # 1D, ~2 KB
        z_proj = Array(dropdims(any(input .> UInt8(0), dims=(1,2)), dims=(1,2)))  # 1D, ~2 KB
        
        x_inds = findall(x_proj)
        if isempty(x_inds)
            fill!(output, 0)
            return
        end
        y_inds = findall(y_proj)
        z_inds = findall(z_proj)
        
        min_x = max(1, minimum(x_inds) - max_rad_x)
        max_x = min(dims[1], maximum(x_inds) + max_rad_x)
        min_y = max(1, minimum(y_inds) - max_rad_y)
        max_y = min(dims[2], maximum(y_inds) + max_rad_y)
        min_z = max(1, minimum(z_inds) - max_rad_z)
        max_z = min(dims[3], maximum(z_inds) + max_rad_z)
    end
    
    ndrange = (max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1)
    kernel! = anisotropic_dilation_kernel!(backend)
    kernel!(output, input, dims, Float32(sp_x), Float32(sp_y), Float32(sp_z), m_pos_x, m_neg_x, m_pos_y, m_neg_y, m_pos_z, m_neg_z, max_rad_x, max_rad_y, max_rad_z, min_x, min_y, min_z, ndrange=ndrange)
    KernelAbstractions.synchronize(backend)
end

@kernel function anisotropic_erosion_kernel!(output, input, dims, sp_x, sp_y, sp_z, m_pos_x, m_neg_x, m_pos_y, m_neg_y, m_pos_z, m_neg_z, max_rad_x, max_rad_y, max_rad_z, offset_x, offset_y, offset_z)
    I_local, J_local, K_local = @index(Global, NTuple)
    I = I_local + offset_x - 1
    J = J_local + offset_y - 1
    K = K_local + offset_z - 1
    
    if I <= dims[1] && J <= dims[2] && K <= dims[3]
        if input[I, J, K] == 0
            output[I, J, K] = UInt8(0)
        else
            val = UInt8(1)
            x_min = max(1, I - max_rad_x)
            x_max = min(dims[1], I + max_rad_x)
            y_min = max(1, J - max_rad_y)
            y_max = min(dims[2], J + max_rad_y)
            z_min = max(1, K - max_rad_z)
            z_max = min(dims[3], K + max_rad_z)
            
            for k_off in z_min:z_max
                for j_off in y_min:y_max
                    for i_off in x_min:x_max
                        if input[i_off, j_off, k_off] == 0
                            dx = Float32((I - i_off) * sp_x)
                            dy = Float32((J - j_off) * sp_y)
                            dz = Float32((K - k_off) * sp_z)
                            
                            m_x = dx > 0 ? m_pos_x : m_neg_x
                            m_y = dy > 0 ? m_pos_y : m_neg_y
                            m_z = dz > 0 ? m_pos_z : m_neg_z
                            
                            nx = m_x > 0 ? (dx / m_x) : (dx == 0 ? 0.0f0 : 1000.0f0)
                            ny = m_y > 0 ? (dy / m_y) : (dy == 0 ? 0.0f0 : 1000.0f0)
                            nz = m_z > 0 ? (dz / m_z) : (dz == 0 ? 0.0f0 : 1000.0f0)
                            dist_sq = nx*nx + ny*ny + nz*nz
                            
                            if dist_sq <= 1.0f0
                                val = UInt8(0)
                                break
                            end
                        end
                    end
                    if val == UInt8(0) break end
                end
                if val == UInt8(0) break end
            end
            output[I, J, K] = val
        end
    end
end

function run_anisotropic_erosion!(backend, output, input, sp_x, sp_y, sp_z, margins)
    all_default = Float32(get(margins, "all", 0.0))
    m_pos_x = Float32(get(margins, "left", all_default))
    m_neg_x = Float32(get(margins, "right", all_default))
    m_pos_y = Float32(get(margins, "posterior", all_default))
    m_neg_y = Float32(get(margins, "anterior", all_default))
    m_pos_z = Float32(get(margins, "superior", all_default))
    m_neg_z = Float32(get(margins, "inferior", all_default))
    
    max_rx = max(m_pos_x, m_neg_x)
    max_ry = max(m_pos_y, m_neg_y)
    max_rz = max(m_pos_z, m_neg_z)
    
    max_rad_x = ceil(Int, max_rx / max(0.001f0, sp_x))
    max_rad_y = ceil(Int, max_ry / max(0.001f0, sp_y))
    max_rad_z = ceil(Int, max_rz / max(0.001f0, sp_z))
    
    dims = size(output)
    
    # For erosion, we need bbox of foreground (what to erode from)
    # Use GPU projection for bbox instead of full volume download
    if input isa Array
        in_host = input
        idx = findall(in_host .> 0)
        if isempty(idx)
            fill!(output, 0)
            return
        end
        min_x = max(1, minimum(i[1] for i in idx) - max_rad_x)
        max_x = min(dims[1], maximum(i[1] for i in idx) + max_rad_x)
        min_y = max(1, minimum(i[2] for i in idx) - max_rad_y)
        max_y = min(dims[2], maximum(i[2] for i in idx) + max_rad_y)
        min_z = max(1, minimum(i[3] for i in idx) - max_rad_z)
        max_z = min(dims[3], maximum(i[3] for i in idx) + max_rad_z)
    else
        # GPU: project along each axis for foreground bbox
        x_proj = Array(dropdims(any(input .> UInt8(0), dims=(2,3)), dims=(2,3)))
        x_inds = findall(x_proj)
        if isempty(x_inds)
            fill!(output, 0)
            return
        end
        y_proj = Array(dropdims(any(input .> UInt8(0), dims=(1,3)), dims=(1,3)))
        z_proj = Array(dropdims(any(input .> UInt8(0), dims=(1,2)), dims=(1,2)))
        y_inds = findall(y_proj)
        z_inds = findall(z_proj)
        
        # For erosion, expand bbox slightly since we're checking neighborhood
        min_x = max(1, minimum(x_inds) - max_rad_x)
        max_x = min(dims[1], maximum(x_inds) + max_rad_x)
        min_y = max(1, minimum(y_inds) - max_rad_y)
        max_y = min(dims[2], maximum(y_inds) + max_rad_y)
        min_z = max(1, minimum(z_inds) - max_rad_z)
        max_z = min(dims[3], maximum(z_inds) + max_rad_z)
    end
    
    ndrange = (max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1)
    
    # Pre-fill output with copy of input
    copyto!(output, input)
    
    kernel! = anisotropic_erosion_kernel!(backend)
    kernel!(output, input, dims, Float32(sp_x), Float32(sp_y), Float32(sp_z), m_pos_x, m_neg_x, m_pos_y, m_neg_y, m_pos_z, m_neg_z, max_rad_x, max_rad_y, max_rad_z, min_x, min_y, min_z, ndrange=ndrange)
    KernelAbstractions.synchronize(backend)
end

@kernel function coronal_lateral_growth_kernel!(output, input, dims, rad_x)
    I, J, K = @index(Global, NTuple)
    
    if I <= dims[1] && J <= dims[2] && K <= dims[3]
        if input[I, J, K] > 0
            output[I, J, K] = UInt8(1)
        else
            val = UInt8(0)
            x_min = max(1, I - rad_x)
            x_max = min(dims[1], I + rad_x)
            
            for i_off in x_min:x_max
                if input[i_off, J, K] > 0
                    val = UInt8(1)
                    break
                end
            end
            output[I, J, K] = val
        end
    end
end

function run_coronal_lateral_growth!(backend, output, input, sp_x, margin_mm)
    dims = size(output)
    rad_x = Int(round(margin_mm / sp_x))
    
    kernel! = coronal_lateral_growth_kernel!(backend)
    kernel!(output, input, dims, rad_x, ndrange=dims)
    KernelAbstractions.synchronize(backend)
end
