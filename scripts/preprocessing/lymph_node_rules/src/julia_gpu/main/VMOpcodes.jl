module VMOpcodes

export VMInstruction,
       OP_NOP, OP_LOAD_BASE_MASK, OP_UNION_BASE_MASK, OP_LOAD_OUTPUT_BUFFER,
       OP_DILATE_BOX, OP_ANISOTROPIC_EXPAND,
       OP_Z_BOUNDS, OP_BBOX, OP_PLANE, OP_SLICE_CONSTRAINT,
       OP_EXCLUDE_MASK, OP_EXCLUDE_MARGIN, OP_INTERSECT_MASK, OP_WRITE_OUTPUT

const OP_NOP               = Int32(0)
const OP_LOAD_BASE_MASK     = Int32(1)   # arg1=ch, arg2=id
const OP_UNION_BASE_MASK    = Int32(2)   # arg1=ch, arg2=id
const OP_LOAD_OUTPUT_BUFFER = Int32(3)   # loads active = (level_output[i, j, k, rule_idx] > 0)
const OP_DILATE_BOX         = Int32(10)  # arg1=ch, arg2=id, arg3=rx, arg4=ry, arg5=rz
const OP_ANISOTROPIC_EXPAND = Int32(11)  # arg1=ch, arg2=id, arg3..8=roi(xmin,xmax,ymin,ymax,zmin,zmax), arg9..14=rad(rxp,rxn,ryp,ryn,rzp,rzn), farg1..6=margins(mxp,mxn,myp,myn,mzp,mzn)
const OP_Z_BOUNDS           = Int32(20)  # arg1=k_min, arg2=k_max
const OP_BBOX               = Int32(21)  # arg1=axis (1=x,2=y,3=z), arg2=min_val, arg3=max_val
const OP_PLANE              = Int32(22)  # arg1=side (1=pos, -1=neg), farg1=nx, farg2=ny, farg3=nz, farg4=d
const OP_SLICE_CONSTRAINT   = Int32(23)  # arg1=ch, arg2=id, arg3=direction (0=sup,1=inf,2=ant,3=post,4=right,5=left), arg4=slice_axis (3=z)
const OP_EXCLUDE_MASK       = Int32(30)  # arg1=ch, arg2=id
const OP_EXCLUDE_MARGIN     = Int32(31)  # arg1=ch, arg2=id, arg3=rx, arg4=ry, arg5=rz
const OP_INTERSECT_MASK     = Int32(32)  # arg1=ch, arg2=id — deactivate if mask NOT present (for AND)
const OP_WRITE_OUTPUT       = Int32(100) # rule_idx (which output channel in level_output)

struct VMInstruction
    opcode::Int32
    rule_idx::Int32
    arg1::Int32
    arg2::Int32
    arg3::Int32
    arg4::Int32
    arg5::Int32
    arg6::Int32
    arg7::Int32
    arg8::Int32
    arg9::Int32
    arg10::Int32
    arg11::Int32
    arg12::Int32
    arg13::Int32
    arg14::Int32
    farg1::Float32
    farg2::Float32
    farg3::Float32
    farg4::Float32
    farg5::Float32
    farg6::Float32
end

# Convenience constructor with default zeros
VMInstruction(opcode::Integer, rule_idx::Integer;
              arg1=0, arg2=0, arg3=0, arg4=0, arg5=0, arg6=0,
              arg7=0, arg8=0, arg9=0, arg10=0, arg11=0, arg12=0, arg13=0, arg14=0,
              farg1=0.0f0, farg2=0.0f0, farg3=0.0f0, farg4=0.0f0, farg5=0.0f0, farg6=0.0f0) =
    VMInstruction(Int32(opcode), Int32(rule_idx),
                  Int32(arg1), Int32(arg2), Int32(arg3), Int32(arg4), Int32(arg5), Int32(arg6),
                  Int32(arg7), Int32(arg8), Int32(arg9), Int32(arg10), Int32(arg11), Int32(arg12), Int32(arg13), Int32(arg14),
                  Float32(farg1), Float32(farg2), Float32(farg3), Float32(farg4), Float32(farg5), Float32(farg6))

end # module
