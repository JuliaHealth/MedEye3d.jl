module VmKernel

using KernelAbstractions
using Adapt
using Main.VmTypes

export execute_vm_kernel!, execute_vm_kernel_with_planes!

@kernel function vm_mega_kernel!(
    tensor::AbstractArray{UInt32, 4}, # (X, Y, Z, C)
    instructions::AbstractArray{Instruction, 1},
    x_dim::Int32, y_dim::Int32, z_dim::Int32
)
    i, j, k = @index(Global, NTuple)

    if i <= x_dim && j <= y_dim && k <= z_dim
        # Execute instructions sequentially for this voxel
        for inst in instructions
            
            # Read input 1
            val1 = false
            if inst.in1_id >= 0
                val1 = (tensor[i, j, k, inst.in1_channel] & (UInt32(1) << inst.in1_id)) != 0
            end
            
            # Read input 2
            val2 = false
            if inst.in2_id >= 0
                val2 = (tensor[i, j, k, inst.in2_channel] & (UInt32(1) << inst.in2_id)) != 0
            end
            
            result = false
            
            if inst.opcode == OP_MASK
                result = val1 && val2
            elseif inst.opcode == OP_EXCLUDE
                result = val1 && !val2
            elseif inst.opcode == OP_COMBINE
                result = val1 || val2
            elseif inst.opcode == OP_LIMIT_PLANE
                dist = inst.farg1 * Float32(i) + inst.farg2 * Float32(j) + inst.farg3 * Float32(k) + inst.farg4
                result = val1 && (dist >= 0f0)
            elseif inst.opcode == OP_DRAW_SPHERE
                dx = Float32(i) - inst.farg1
                dy = Float32(j) - inst.farg2
                dz = Float32(k) - inst.farg3
                r = inst.farg4
                result = (dx*dx + dy*dy + dz*dz) <= (r*r)
            elseif inst.opcode == OP_SPLIT_MASK
                coord = (inst.farg1 == 1f0) ? Float32(i) : ((inst.farg1 == 2f0) ? Float32(j) : Float32(k))
                if inst.farg3 == 1f0
                    result = val1 && (coord < inst.farg2)
                else
                    result = val1 && (coord > inst.farg2)
                end
            elseif inst.opcode == OP_CONVEX_HULL_APPROX
                result = val1
            elseif inst.opcode == OP_EDGE_EXTRACT
                # Extract surface voxels using 6-connectivity
                # val1 = the mask to extract edges from
                if val1
                    is_edge = false
                    # Check 6-connected neighbors: if any neighbor is OFF, this is an edge
                    if i > 1
                        n = (tensor[i-1, j, k, inst.in1_channel] & (UInt32(1) << inst.in1_id)) != 0
                        is_edge = is_edge || !n
                    else
                        is_edge = true  # boundary is edge
                    end
                    if !is_edge && i < x_dim
                        n = (tensor[i+1, j, k, inst.in1_channel] & (UInt32(1) << inst.in1_id)) != 0
                        is_edge = is_edge || !n
                    end
                    if !is_edge && j > 1
                        n = (tensor[i, j-1, k, inst.in1_channel] & (UInt32(1) << inst.in1_id)) != 0
                        is_edge = is_edge || !n
                    end
                    if !is_edge && j < y_dim
                        n = (tensor[i, j+1, k, inst.in1_channel] & (UInt32(1) << inst.in1_id)) != 0
                        is_edge = is_edge || !n
                    end
                    if !is_edge && k > 1
                        n = (tensor[i, j, k-1, inst.in1_channel] & (UInt32(1) << inst.in1_id)) != 0
                        is_edge = is_edge || !n
                    end
                    if !is_edge && k < z_dim
                        n = (tensor[i, j, k+1, inst.in1_channel] & (UInt32(1) << inst.in1_id)) != 0
                        is_edge = is_edge || !n
                    end
                    result = is_edge
                end
            elseif inst.opcode == OP_DILATE
                # Neighborhood scan for anisotropic dilation
                m_px = round(Int, inst.farg1)
                m_nx = round(Int, inst.farg2)
                m_py = round(Int, inst.farg3)
                m_ny = round(Int, inst.farg4)
                m_pz = round(Int, inst.farg5)
                m_nz = round(Int, inst.farg6)
                
                found = false
                x_min = max(1, i - m_nx)
                x_max = min(x_dim, i + m_px)
                y_min = max(1, j - m_ny)
                y_max = min(y_dim, j + m_py)
                z_min = max(1, k - m_nz)
                z_max = min(z_dim, k + m_pz)
                
                for nx in x_min:x_max
                    dx = Float32(nx - i)
                    m_x = dx > 0 ? inst.farg1 : inst.farg2
                    norm_dx = m_x > 0 ? dx / m_x : (dx == 0 ? 0f0 : 1f0)
                    
                    for ny in y_min:y_max
                        dy = Float32(ny - j)
                        m_y = dy > 0 ? inst.farg3 : inst.farg4
                        norm_dy = m_y > 0 ? dy / m_y : (dy == 0 ? 0f0 : 1f0)
                        
                        for nz in z_min:z_max
                            dz = Float32(nz - k)
                            m_z = dz > 0 ? inst.farg5 : inst.farg6
                            norm_dz = m_z > 0 ? dz / m_z : (dz == 0 ? 0f0 : 1f0)
                            
                            dist2 = norm_dx*norm_dx + norm_dy*norm_dy + norm_dz*norm_dz
                            if dist2 <= 1f0
                                if (tensor[nx, ny, nz, inst.in1_channel] & (UInt32(1) << inst.in1_id)) != 0
                                    found = true
                                    break
                                end
                            end
                        end
                        if found; break; end
                    end
                    if found; break; end
                end
                result = found
            end
            
            # Write result
            if result
                tensor[i, j, k, inst.out_channel] |= (UInt32(1) << inst.out_id)
            end
        end
    end
end

# Extended kernel with per-slice half-space planes table
@kernel function vm_mega_kernel_with_planes!(
    tensor::AbstractArray{UInt32, 4},
    instructions::AbstractArray{Instruction, 1},
    planes_table::AbstractArray{Float32, 3},  # (max_planes, 3, Z) -> A, B, D per plane per slice
    num_planes::AbstractArray{Int32, 1},       # (Z,) -> number of active planes per slice
    x_dim::Int32, y_dim::Int32, z_dim::Int32
)
    i, j, k = @index(Global, NTuple)

    if i <= x_dim && j <= y_dim && k <= z_dim
        for inst in instructions
            val1 = false
            if inst.in1_id >= 0
                val1 = (tensor[i, j, k, inst.in1_channel] & (UInt32(1) << inst.in1_id)) != 0
            end
            val2 = false
            if inst.in2_id >= 0
                val2 = (tensor[i, j, k, inst.in2_channel] & (UInt32(1) << inst.in2_id)) != 0
            end
            
            result = false
            
            if inst.opcode == OP_MASK
                result = val1 && val2
            elseif inst.opcode == OP_EXCLUDE
                result = val1 && !val2
            elseif inst.opcode == OP_COMBINE
                result = val1 || val2
            elseif inst.opcode == OP_LIMIT_PLANE
                dist = inst.farg1 * Float32(i) + inst.farg2 * Float32(j) + inst.farg3 * Float32(k) + inst.farg4
                result = val1 && (dist >= 0f0)
            elseif inst.opcode == OP_DRAW_SPHERE
                dx = Float32(i) - inst.farg1
                dy = Float32(j) - inst.farg2
                dz = Float32(k) - inst.farg3
                r = inst.farg4
                result = (dx*dx + dy*dy + dz*dz) <= (r*r)
            elseif inst.opcode == OP_SPLIT_MASK
                coord = (inst.farg1 == 1f0) ? Float32(i) : ((inst.farg1 == 2f0) ? Float32(j) : Float32(k))
                if inst.farg3 == 1f0
                    result = val1 && (coord < inst.farg2)
                else
                    result = val1 && (coord > inst.farg2)
                end
            elseif inst.opcode == OP_CONVEX_HULL_APPROX
                result = val1
            elseif inst.opcode == OP_EDGE_EXTRACT
                if val1
                    is_edge = false
                    if i > 1
                        n = (tensor[i-1, j, k, inst.in1_channel] & (UInt32(1) << inst.in1_id)) != 0
                        is_edge = is_edge || !n
                    else
                        is_edge = true
                    end
                    if !is_edge && i < x_dim
                        n = (tensor[i+1, j, k, inst.in1_channel] & (UInt32(1) << inst.in1_id)) != 0
                        is_edge = is_edge || !n
                    end
                    if !is_edge && j > 1
                        n = (tensor[i, j-1, k, inst.in1_channel] & (UInt32(1) << inst.in1_id)) != 0
                        is_edge = is_edge || !n
                    end
                    if !is_edge && j < y_dim
                        n = (tensor[i, j+1, k, inst.in1_channel] & (UInt32(1) << inst.in1_id)) != 0
                        is_edge = is_edge || !n
                    end
                    if !is_edge && k > 1
                        n = (tensor[i, j, k-1, inst.in1_channel] & (UInt32(1) << inst.in1_id)) != 0
                        is_edge = is_edge || !n
                    end
                    if !is_edge && k < z_dim
                        n = (tensor[i, j, k+1, inst.in1_channel] & (UInt32(1) << inst.in1_id)) != 0
                        is_edge = is_edge || !n
                    end
                    result = is_edge
                end
            elseif inst.opcode == OP_HALF_SPACE
                # Per-slice 2D convex hull test via half-space intersection
                # For this voxel at slice k, test against all plane equations for slice k
                # planes_table[p, 1, k] = A, [p, 2, k] = B, [p, 3, k] = D
                # Point inside hull if: A*i + B*j + D <= epsilon for ALL planes
                np = num_planes[k]
                if np > 0
                    inside = true
                    for p in 1:np
                        A = planes_table[p, 1, k]
                        B = planes_table[p, 2, k]
                        D = planes_table[p, 3, k]
                        if A * Float32(i) + B * Float32(j) + D > 1f-5
                            inside = false
                            break
                        end
                    end
                    # Subtract obstacles: inside hull AND NOT obstacle (val2)
                    result = inside && !val2
                end
            elseif inst.opcode == OP_DILATE
                m_px = round(Int, inst.farg1)
                m_nx = round(Int, inst.farg2)
                m_py = round(Int, inst.farg3)
                m_ny = round(Int, inst.farg4)
                m_pz = round(Int, inst.farg5)
                m_nz = round(Int, inst.farg6)
                
                found = false
                x_min = max(1, i - m_nx)
                x_max = min(x_dim, i + m_px)
                y_min = max(1, j - m_ny)
                y_max = min(y_dim, j + m_py)
                z_min = max(1, k - m_nz)
                z_max = min(z_dim, k + m_pz)
                
                for nx in x_min:x_max
                    dx = Float32(nx - i)
                    m_x = dx > 0 ? inst.farg1 : inst.farg2
                    norm_dx = m_x > 0 ? dx / m_x : (dx == 0 ? 0f0 : 1f0)
                    
                    for ny in y_min:y_max
                        dy = Float32(ny - j)
                        m_y = dy > 0 ? inst.farg3 : inst.farg4
                        norm_dy = m_y > 0 ? dy / m_y : (dy == 0 ? 0f0 : 1f0)
                        
                        for nz in z_min:z_max
                            dz = Float32(nz - k)
                            m_z = dz > 0 ? inst.farg5 : inst.farg6
                            norm_dz = m_z > 0 ? dz / m_z : (dz == 0 ? 0f0 : 1f0)
                            
                            dist2 = norm_dx*norm_dx + norm_dy*norm_dy + norm_dz*norm_dz
                            if dist2 <= 1f0
                                if (tensor[nx, ny, nz, inst.in1_channel] & (UInt32(1) << inst.in1_id)) != 0
                                    found = true
                                    break
                                end
                            end
                        end
                        if found; break; end
                    end
                    if found; break; end
                end
                result = found
            end
            
            if result
                tensor[i, j, k, inst.out_channel] |= (UInt32(1) << inst.out_id)
            end
        end
    end
end

function execute_vm_kernel!(backend, tensor, instructions; x_dim, y_dim, z_dim)
    kernel! = vm_mega_kernel!(backend)
    d_instructions = adapt(backend, instructions)
    kernel!(tensor, d_instructions, Int32(x_dim), Int32(y_dim), Int32(z_dim), ndrange=(x_dim, y_dim, z_dim))
    KernelAbstractions.synchronize(backend)
end

function execute_vm_kernel_with_planes!(backend, tensor, instructions, planes_table, num_planes; x_dim, y_dim, z_dim)
    kernel! = vm_mega_kernel_with_planes!(backend)
    d_instructions = adapt(backend, instructions)
    d_planes = adapt(backend, planes_table)
    d_nplanes = adapt(backend, num_planes)
    kernel!(tensor, d_instructions, d_planes, d_nplanes, Int32(x_dim), Int32(y_dim), Int32(z_dim), ndrange=(x_dim, y_dim, z_dim))
    KernelAbstractions.synchronize(backend)
end

end
