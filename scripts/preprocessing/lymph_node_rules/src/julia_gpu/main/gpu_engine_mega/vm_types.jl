module VmTypes

export Instruction, OP_MASK, OP_EXCLUDE, OP_COMBINE, OP_LIMIT_PLANE, OP_LIMIT_DIR, OP_JFA_SEED, OP_DRAW_SPHERE, OP_DRAW_CYLINDER, OP_SPLIT_MASK, OP_CONVEX_HULL_APPROX, OP_EDGE_EXTRACT, OP_HALF_SPACE, OP_FLOOD_SEED, OP_INTERSECT, OP_DILATE, OP_CC_INIT, OP_CC_PROPAGATE, OP_CC_KEEP_LARGEST

const OP_MASK = UInt32(1)
const OP_EXCLUDE = UInt32(2)
const OP_COMBINE = UInt32(3)
const OP_LIMIT_PLANE = UInt32(4)
const OP_LIMIT_DIR = UInt32(5)
const OP_JFA_SEED = UInt32(6)
const OP_DRAW_SPHERE = UInt32(7)
const OP_DRAW_CYLINDER = UInt32(8)
const OP_SPLIT_MASK = UInt32(9)
const OP_CONVEX_HULL_APPROX = UInt32(10)
const OP_EDGE_EXTRACT = UInt32(11)    # Extract surface voxels (6-connectivity)
const OP_HALF_SPACE = UInt32(12)      # Point-in-convex-hull via per-slice half-space test
const OP_FLOOD_SEED = UInt32(13)      # Seed a point for JFA-based flood fill
const OP_INTERSECT = UInt32(14)
const OP_DILATE = UInt32(15)          # Per-voxel anisotropic dilation via neighborhood scan
const OP_CC_INIT = UInt32(16)         # Initialize labels for connected components
const OP_CC_PROPAGATE = UInt32(17)    # Propagate connected component labels
const OP_CC_KEEP_LARGEST = UInt32(18) # Keep only the largest component

struct Instruction
    opcode::UInt32
    in1_id::Int32      # ID of first input mask
    in1_channel::Int32 # Channel of first input mask
    in2_id::Int32      # ID of second input mask (if any)
    in2_channel::Int32 # Channel of second input mask
    out_id::Int32      # ID to write
    out_channel::Int32 # Channel to write to
    farg1::Float32     # Generic float argument 1
    farg2::Float32
    farg3::Float32
    farg4::Float32
    farg5::Float32
    farg6::Float32
end

end
