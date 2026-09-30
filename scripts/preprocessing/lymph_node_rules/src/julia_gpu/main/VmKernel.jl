module VmKernel

using KernelAbstractions
using KernelAbstractions: @index
using Adapt
using CUDA
using ..InstructionSet

export execute_vm_step!

@kernel function vm_step_kernel!(
    step_output::AbstractArray{UInt8, 4},       # (X, Y, Z, num_ops)
    packed_tensor::AbstractArray{<:Integer, 4},     # (X, Y, Z, C)
    instructions::AbstractArray{VmInstruction, 1},
    dims_x::Int32, dims_y::Int32, dims_z::Int32,
    sp_x::Float32, sp_y::Float32, sp_z::Float32,
    orig_x::Float32, orig_y::Float32, orig_z::Float32,
    dir_00::Float32, dir_11::Float32, dir_22::Float32
)
    i, j, k = @index(Global, NTuple)

    if i <= dims_x && j <= dims_y && k <= dims_z
        # Compute physical coordinates in LPS
        pos_x = orig_x + (Float32(i) - 1.0f0) * sp_x * dir_00
        pos_y = orig_y + (Float32(j) - 1.0f0) * sp_y * dir_11
        pos_z = orig_z + (Float32(k) - 1.0f0) * sp_z * dir_22

        # Iterate over all operations for this step
        for inst in instructions
            out_ch = inst.out_idx
            if out_ch <= 0
                continue
            end

            # Read input 1 boolean state
            val1 = false
            if inst.in1_ch > 0
                val1 = (packed_tensor[i, j, k, inst.in1_ch] == inst.in1_id)
            end

            # Read input 2 boolean state
            val2 = false
            if inst.in2_ch > 0
                val2 = (packed_tensor[i, j, k, inst.in2_ch] == inst.in2_id)
            end

            res = false
            op = inst.opcode

            if op == OP_COPY
                res = val1
            elseif op == OP_UNION
                res = val1 || val2
            elseif op == OP_INTERSECT
                res = val1 && val2
            elseif op == OP_SUBTRACT
                res = val1 && !val2
            elseif op == OP_Z_PLANE_CUT
                res = val1 && (Float32(k) >= inst.arg1) && (Float32(k) <= inst.arg2)
            elseif op == OP_PLANE_CLIP
                dist = inst.arg1 * pos_x + inst.arg2 * pos_y + inst.arg3 * pos_z + inst.arg4
                res = val1 && (inst.arg5 >= 0.5f0 ? (dist >= 0f0) : (dist <= 0f0))
            elseif op == OP_BOX_CROP
                res = val1 && (Float32(k) >= inst.arg1) && (Float32(k) <= inst.arg2) &&
                              (Float32(j) >= inst.arg3) && (Float32(j) <= inst.arg4) &&
                              (Float32(i) >= inst.arg5) && (Float32(i) <= inst.arg6)
            elseif op == OP_RELATIVE_GEOMETRIC
                in_z = (pos_z > inst.arg1) && (pos_z <= inst.arg2)
                in_y = (pos_y >= inst.arg3) && (pos_y <= inst.arg4)
                in_x = (pos_x >= inst.arg5) && (pos_x <= inst.arg6)
                res = in_z && in_y && in_x
            elseif op == OP_BILATERAL_SPLIT
                is_left = (inst.arg2 >= 0.5f0)
                res = val1 && (is_left ? (Float32(i) >= inst.arg1) : (Float32(i) < inst.arg1))
            elseif op == OP_CONSTRAINTS
                res = val1
                if inst.arg1 > -100000.0f0; res = res && (pos_z >= inst.arg1); end
                if inst.arg2 <  100000.0f0; res = res && (pos_z <= inst.arg2); end
                if inst.arg3 > -100000.0f0; res = res && (pos_y >= inst.arg3); end
                if inst.arg4 <  100000.0f0; res = res && (pos_y <= inst.arg4); end
                if inst.arg5 > -100000.0f0; res = res && (pos_x >= inst.arg5); end
                if inst.arg6 <  100000.0f0; res = res && (pos_x <= inst.arg6); end
            end

            step_output[i, j, k, out_ch] = res ? UInt8(1) : UInt8(0)
        end
    end
end

function execute_vm_step!(
    backend::KernelAbstractions.Backend,
    step_output::AbstractArray{UInt8, 4},
    packed_tensor::AbstractArray{<:Integer, 4},
    instructions::Vector{VmInstruction},
    dims::Tuple{Int, Int, Int},
    spacing::Tuple{Float64, Float64, Float64},
    origin::Tuple{Float64, Float64, Float64},
    direction::Tuple
)
    if isempty(instructions)
        return
    end

    inst_gpu = adapt(typeof(packed_tensor) <: Array ? Array : CuArray, instructions)

    dir_00 = Float32(direction[1])
    dir_11 = Float32(direction[5])
    dir_22 = Float32(direction[9])

    kernel! = vm_step_kernel!(backend)
    kernel!(
        step_output,
        packed_tensor,
        inst_gpu,
        Int32(dims[1]), Int32(dims[2]), Int32(dims[3]),
        Float32(spacing[1]), Float32(spacing[2]), Float32(spacing[3]),
        Float32(origin[1]), Float32(origin[2]), Float32(origin[3]),
        dir_00, dir_11, dir_22,
        ndrange=dims
    )
    KernelAbstractions.synchronize(backend)
end

end # module
