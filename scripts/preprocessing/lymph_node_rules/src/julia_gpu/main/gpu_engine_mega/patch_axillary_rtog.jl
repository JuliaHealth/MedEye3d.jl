# patch_axillary_rtog.jl
# AxillaryRTOG is now implemented as OP_AXILLARY_ZONE (opcode 9) in mega_fused_dag_kernel_v2.jl
# The geometry is precomputed on CPU via compute_mega2_axillary_geometry command in julia_server.jl
# This file is kept for backward compatibility but the function is no longer called.
#
# The OP_AXILLARY_ZONE opcode encodes:
#   - Plane normal (nx, ny, nz) from cross-product of (rib3-coracoid) and (rib5-coracoid)
#   - Coracoid point (anterior-most point of scapula)
#   - Lateral boundary line (rib5 -> coracoid, projected at each Z)
#   - Medial boundary line (rib3 -> coracoid, projected at each Z)
#   - Level code: 1=LevelI, 2=LevelII, 3=LevelIII, 4=Rotter's
#   - Side: 1.0=left, 0.0=right

function execute_mega2_axillary_rtog(batch_json_str)
    println("[DEPRECATED] execute_mega2_axillary_rtog called — AxillaryRTOG is now handled via OP_AXILLARY_ZONE GPU kernel. This call should not happen.")
    return "DEPRECATED: AxillaryRTOG now uses OP_AXILLARY_ZONE"
end
