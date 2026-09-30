module MegaKernelMod
using CUDA, KernelAbstractions, Adapt
const OP_COPY = Int32(1); const OP_EXCLUDE = Int32(2); const OP_INTERSECT = Int32(3)
const OP_Z_RESTRICT_MIN = Int32(4); const OP_Z_RESTRICT_MAX = Int32(5)
const OP_GEOM_CONSTRAINT = Int32(6); const OP_ANISOTROPIC_MARGIN = Int32(7)

struct Instruction
    opcode::Int32
    roi_min_x::Int32; roi_max_x::Int32
    roi_min_y::Int32; roi_max_y::Int32
    roi_min_z::Int32; roi_max_z::Int32
    arg1::Float32; arg2::Float32; arg3::Float32; arg4::Float32; arg5::Float32; arg6::Float32
    in1::Int32; in2::Int32; out::Int32
end

@kernel function mega_kernel!(volume, @Const(instructions), num_instructions::Int32, sp_x::Float32, sp_y::Float32, sp_z::Float32, dims_x::Int32, dims_y::Int32, dims_z::Int32)
    i, j, k = @index(Global, NTuple)
    for ins_idx in 1:num_instructions
        ins = instructions[ins_idx]
        in_roi = true
        if ins.roi_max_x > Int32(0)
            if i < ins.roi_min_x || i > ins.roi_max_x || j < ins.roi_min_y || j > ins.roi_max_y || k < ins.roi_min_z || k > ins.roi_max_z
                in_roi = false
            end
        end
        if in_roi
            op = ins.opcode
            if op == OP_ANISOTROPIC_MARGIN
                found = false
                rx_f = max(ins.arg1, ins.arg2) / sp_x
                ry_f = max(ins.arg3, ins.arg4) / sp_y
                rz_f = max(ins.arg5, ins.arg6) / sp_z
                rad_x = Int32(ceil(rx_f)) + Int32(2)
                rad_y = Int32(ceil(ry_f)) + Int32(2)
                rad_z = Int32(ceil(rz_f)) + Int32(2)
                
                for dz in -rad_z:rad_z
                    for dy in -rad_y:rad_y
                        for dx in -rad_x:rad_x
                            ni = i + dx; nj = j + dy; nk = k + dz
                            if ni >= Int32(1) && ni <= dims_x && nj >= Int32(1) && nj <= dims_y && nk >= Int32(1) && nk <= dims_z
                                if volume[ni, nj, nk, ins.in1] > UInt8(0)
                                    abs_dx = abs(Float32(dx) * sp_x)
                                    abs_dy = abs(Float32(dy) * sp_y)
                                    abs_dz = abs(Float32(dz) * sp_z)
                                    total = abs_dx + abs_dy + abs_dz + 1f-8
                                    
                                    m_x = dx > 0 ? ins.arg1 : ins.arg2
                                    m_y = dy > 0 ? ins.arg3 : ins.arg4
                                    m_z = dz > 0 ? ins.arg5 : ins.arg6
                                    
                                    threshold = (abs_dx / total) * m_x + (abs_dy / total) * m_y + (abs_dz / total) * m_z
                                    
                                    dist_sq = (dx*sp_x)^2 + (dy*sp_y)^2 + (dz*sp_z)^2
                                    if dist_sq <= threshold^2 && threshold > 0.0f0
                                        found = true
                                        break
                                    end
                                end
                            end
                        end
                        if found; break; end
                    end
                    if found; break; end
                end
                volume[i, j, k, ins.out] = found ? UInt8(1) : UInt8(0)
            end
        end
    end
end

function run_mega_kernel!(backend, volume_gpu, instructions_cpu, spacing)
    num_ins = Int32(length(instructions_cpu))
    ins_gpu = adapt(backend, instructions_cpu)
    dims_x, dims_y, dims_z = Int32(size(volume_gpu, 1)), Int32(size(volume_gpu, 2)), Int32(size(volume_gpu, 3))
    kernel! = mega_kernel!(backend)
    kernel!(volume_gpu, ins_gpu, num_ins, Float32(spacing[1]), Float32(spacing[2]), Float32(spacing[3]), dims_x, dims_y, dims_z, ndrange=(dims_x, dims_y, dims_z))
    KernelAbstractions.synchronize(backend)
    return volume_gpu
end
end # module
