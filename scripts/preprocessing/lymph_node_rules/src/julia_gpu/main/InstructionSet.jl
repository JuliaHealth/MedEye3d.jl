module InstructionSet

export VmOpcode, VmInstruction, OP_COPY, OP_UNION, OP_INTERSECT, OP_SUBTRACT,
       OP_Z_PLANE_CUT, OP_PLANE_CLIP, OP_CONSTRAINTS, OP_RELATIVE_GEOMETRIC,
       OP_BILATERAL_SPLIT, OP_DYNAMIC_SURFACE_SPLIT, OP_BOX_CROP

# Opcodes
const OP_COPY                   = Int32(1)
const OP_UNION                  = Int32(2)
const OP_INTERSECT              = Int32(3)
const OP_SUBTRACT               = Int32(4)
const OP_Z_PLANE_CUT            = Int32(5)
const OP_PLANE_CLIP             = Int32(6)
const OP_CONSTRAINTS            = Int32(7)
const OP_RELATIVE_GEOMETRIC     = Int32(8)
const OP_BILATERAL_SPLIT        = Int32(9)
const OP_DYNAMIC_SURFACE_SPLIT  = Int32(10)
const OP_BOX_CROP               = Int32(11)

"""
    VmInstruction
A compact 64-byte instruction struct passed directly to the GPU KernelAbstractions kernel.
"""
struct VmInstruction
    opcode::Int32       # Operation opcode
    in1_ch::Int32       # Channel index of input 1 in PackedTensor (1-based, 0 if unused)
    in1_id::Int32       # Integer ID of input 1 in PackedTensor
    in2_ch::Int32       # Channel index of input 2 in PackedTensor (0 if unused)
    in2_id::Int32       # Integer ID of input 2 in PackedTensor
    out_idx::Int32      # Output layer index in step output array (1-based)
    
    # Generic float arguments (meaning depends on opcode)
    arg1::Float32       # e.g., z_min / plane_nx / sup_z_max / x_split
    arg2::Float32       # e.g., z_max / plane_ny / z_limit   / is_left (1.0 or 0.0)
    arg3::Float32       # e.g., y_min / plane_nz / y_limit   / custom_flag
    arg4::Float32       # e.g., y_max / plane_d  / y_post
    arg5::Float32       # e.g., x_min / side_val / x_min
    arg6::Float32       # e.g., x_max / pad_val  / x_max
end

end # module
