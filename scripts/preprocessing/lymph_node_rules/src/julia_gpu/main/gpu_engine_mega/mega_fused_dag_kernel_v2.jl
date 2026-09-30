using KernelAbstractions
using CUDA

# ─── Opcode constants ────────────────────────────────────────────────────────
# All operations are parameterized — no rule-specific opcodes
const OP_ANISOTROPIC_MARGIN   = 1   # anisotropic distance margin from nearest-border
const OP_EXPAND_LIMIT         = 2   # expand with a limit mask
const OP_BOOLEAN_OR           = 3   # in_val OR limit_val → out
const OP_BOOLEAN_AND          = 4   # in_val AND limit_val → out
const OP_BOOLEAN_SUB          = 5   # in_val AND NOT limit_val → out
const OP_VOLUMETRIC_BOUNDARY  = 6   # volumetric 2D boundary slice
const OP_MASK                 = 7   # copy in_val → out
const OP_DISTANCE_EXPANSION   = 8   # isotropic distance expansion (Euclidean)
const OP_LIMIT_PLANE_SIGNED   = 9
const OP_ANTERIOR_GROWTH      = 10
const OP_PROPAGATE_Z          = 11  # 1D lookahead scan along Y-axis
const OP_DILATE_3D            = 12  # Morphological 3D dilation (arg1 = radius)
const OP_ERODE_3D             = 13  # Morphological 3D erosion (arg1 = radius)
const OP_CONVEX_HULL_2D       = 14  # Volumetric slice-by-slice 2D convex hull
const OP_Z_LIMIT              = 15  # 1D scan for min/max Z limit plane
# Encoding:
#   arg1-3 = unit normal  (nx, ny, nz)
#   arg4-6 = point on plane (px, py, pz) in mm
#   arg7   = signed distance threshold   (dist ≥ threshold → true)
#   arg8   = if 1.0 → also intersect with in_val mask; if 0.0 → standalone spatial mask
# 
# Used for: AP-plane test (posterior/anterior to PM plane), lateral/medial border planes,
#           cranial/caudal Z-plane limits, etc.
const OP_LIMIT_PLANE_SIGNED   = 9

# ─── Instruction struct ───────────────────────────────────────────────────────
# All opcodes share this struct. Float args are opcode-specific.
struct DagInstructionV2
    opcode::Int32

    in_ch::Int32       # >0 → packed tensor_in channel; <0 → abs = tensor_out channel; 0 = unused
    in_id::UInt16      # packed id value to match in tensor_in (when in_ch > 0)

    limit_ch::Int32    # same encoding as in_ch
    limit_id::UInt16

    out_ch::Int32      # tensor_out channel to write result into

    # Generic float arguments — meaning depends on opcode (see constants above)
    arg1::Float32
    arg2::Float32
    arg3::Float32
    arg4::Float32
    arg5::Float32
    arg6::Float32
    arg7::Float32
    arg8::Float32
    arg9::Float32
    arg10::Float32
    arg11::Float32
    arg12::Float32
    arg13::Float32
    arg14::Float32
    arg15::Float32
    arg16::Float32
    arg17::Float32

    # Bounding box (voxel indices, inclusive, 1-indexed)
    bb_min_x::Int32
    bb_max_x::Int32
    bb_min_y::Int32
    bb_max_y::Int32
    bb_min_z::Int32
    bb_max_z::Int32
end

# ─── Main fused DAG kernel ────────────────────────────────────────────────────
@kernel function mega_fused_dag_kernel_v2!(tensor_in, tensor_out, dt_tensor, instructions, num_instructions, dims, sp_x, sp_y, sp_z)
    I, J, K = @index(Global, NTuple)

    if I <= dims[1] && J <= dims[2] && K <= dims[3]

        for i in 1:num_instructions
            inst = instructions[i]

            # Bounding box cull
            if I < inst.bb_min_x || I > inst.bb_max_x ||
               J < inst.bb_min_y || J > inst.bb_max_y ||
               K < inst.bb_min_z || K > inst.bb_max_z
                continue
            end

            # Read primary input
            # in_id=0 means "match any nonzero voxel" (binary mask check)
            in_val = false
            if inst.in_ch > 0 && inst.in_ch <= size(tensor_in, 1)
                raw = tensor_in[inst.in_ch, I, J, K]
                in_val = inst.in_id == 0 ? (raw > 0) : (raw == inst.in_id)
            elseif inst.in_ch < 0 && (-inst.in_ch) <= size(tensor_out, 1)
                in_val = (tensor_out[-inst.in_ch, I, J, K] > 0)
            end

            # Read limit/secondary input
            # limit_id=0 means "match any nonzero voxel" (binary mask check)
            limit_val = false
            if inst.limit_ch > 0 && inst.limit_ch <= size(tensor_in, 1)
                raw = tensor_in[inst.limit_ch, I, J, K]
                limit_val = inst.limit_id == 0 ? (raw > 0) : (raw == inst.limit_id)
            elseif inst.limit_ch < 0 && (-inst.limit_ch) <= size(tensor_out, 1)
                limit_val = (tensor_out[-inst.limit_ch, I, J, K] > 0)
            end

            out_val = false

            # ── Boolean / masking opcodes ──────────────────────────────────
            if inst.opcode == OP_MASK
                out_val = in_val

            elseif inst.opcode == OP_BOOLEAN_OR
                prev_out = (tensor_out[inst.out_ch, I, J, K] > 0)
                out_val = in_val || limit_val || prev_out

            elseif inst.opcode == OP_BOOLEAN_AND
                out_val = in_val && limit_val

            elseif inst.opcode == OP_BOOLEAN_SUB
                out_val = in_val && !limit_val

            # ── Anisotropic margin from nearest surface voxel ──────────────
            elseif inst.opcode == OP_ANISOTROPIC_MARGIN
                dt_ch = Int32(inst.arg1)
                if dt_ch >= 1  # guard: dt_ch=0 means not assigned → skip
                    bx = dt_tensor[1, dt_ch, I, J, K]
                    by = dt_tensor[2, dt_ch, I, J, K]
                    bz = dt_tensor[3, dt_ch, I, J, K]

                    if bx > 0 && by > 0 && bz > 0
                        dx = Float32((I - bx) * sp_x)
                        dy = Float32((J - by) * sp_y)
                        dz = Float32((K - bz) * sp_z)
                        abs_dx = abs(dx); abs_dy = abs(dy); abs_dz = abs(dz)
                        actual_dist = sqrt(dx*dx + dy*dy + dz*dz)
                        total = abs_dx + abs_dy + abs_dz + 1f-8
                        m_x = dx > 0 ? inst.arg2 : inst.arg3
                        m_y = dy > 0 ? inst.arg5 : inst.arg4
                        m_z = dz > 0 ? inst.arg6 : inst.arg7
                        threshold = (abs_dx / total) * m_x + (abs_dy / total) * m_y + (abs_dz / total) * m_z
                        if actual_dist == 0.0f0 || (actual_dist <= threshold && threshold > 0.0f0)
                            out_val = true
                        end
                    end
                end

            # ── Signed plane constraint (general-purpose, used for all planar tests) ──
            # Tests: dot(pos - point_on_plane, normal) >= threshold
            # arg1=nx, arg2=ny, arg3=nz  (unit normal)
            # arg4=px, arg5=py, arg6=pz  (point on plane, in mm)
            # arg7=threshold             (signed dist threshold; typically 0.0 or small margin)
            # arg8=intersect_with_in     (1.0 → AND with in_val; 0.0 → pure spatial test)

            elseif inst.opcode == OP_PROPAGATE_Z
                # Instead of distance transforms, we scan along the Z-axis in the original input mask
                if inst.in_ch != 0
                    ch = inst.in_ch
                    found = false
                    if ch > 0
                        for k in 1:dims[3]
                            if tensor_in[ch, I, J, k] == inst.in_id
                                dz = Float32(K - k) * sp_z
                                m_z = dz > 0 ? inst.arg6 : inst.arg7
                                if abs(dz) <= m_z
                                    found = true
                                    break
                                end
                            end
                        end
                    elseif ch < 0
                        for k in 1:dims[3]
                            if tensor_out[-ch, I, J, k] > 0
                                dz = Float32(K - k) * sp_z
                                m_z = dz > 0 ? inst.arg6 : inst.arg7
                                if abs(dz) <= m_z
                                    found = true
                                    break
                                end
                            end
                        end
                    end
                    if found
                        out_val = true
                    end
                end

            elseif inst.opcode == OP_LIMIT_PLANE_SIGNED
                vx = Float32(I - 1) * sp_x
                vy = Float32(J - 1) * sp_y
                vz = Float32(K - 1) * sp_z
                signed_dist = (vx - inst.arg4) * inst.arg1 +
                              (vy - inst.arg5) * inst.arg2 +
                              (vz - inst.arg6) * inst.arg3
                plane_ok = signed_dist >= inst.arg7
                if inst.arg8 > 0.5f0
                    out_val = plane_ok && in_val
                else
                    out_val = plane_ok
                end
            
            # ── Anterior Growth Mask (1D lookahead scan along Y-axis) ──
            elseif inst.opcode == OP_ANTERIOR_GROWTH
                dist_mm = inst.arg1
                dist_vox = round(Int32, dist_mm / sp_y)
                is_moved_line = inst.arg2 > 0.5f0
                
                is_self_limit = limit_val
                if !is_self_limit
                    first_hit = -1
                    max_scan = min(Int32(dims[2]), Int32(J + dist_vox))
                    for y_scan in Int32(J+1):max_scan
                        scan_limit = false
                        if inst.limit_ch > 0 && inst.limit_ch <= size(tensor_in, 1)
                            raw_s = tensor_in[inst.limit_ch, I, y_scan, K]
                            scan_limit = inst.limit_id == 0 ? (raw_s > 0) : (raw_s == inst.limit_id)
                        elseif inst.limit_ch < 0 && (-inst.limit_ch) <= size(tensor_out, 1)
                            scan_limit = (tensor_out[-inst.limit_ch, I, y_scan, K] > 0)
                        end
                        
                        if scan_limit
                            first_hit = y_scan
                            break
                        end
                    end
                    
                    if first_hit != -1
                        if is_moved_line
                            # It is the moved line if it's exactly at dist_vox, OR if we hit the edge of the image (J == 1)
                            if (first_hit - J) == dist_vox || J == 1
                                out_val = true
                            end
                        else
                            out_val = true
                        end
                    end
                end

            # ── Isotropic distance expansion (Euclidean distance from nearest border) ──
            # Uses the same JFA dt_tensor as anisotropic margin, but with uniform threshold.
            # arg1 = dt_ch (slot index in dt_tensor, 1-based)
            # arg2 = expansion radius in mm
            elseif inst.opcode == OP_DISTANCE_EXPANSION
                dt_ch = Int32(inst.arg1)
                if dt_ch >= 1
                    bx = dt_tensor[1, dt_ch, I, J, K]
                    by = dt_tensor[2, dt_ch, I, J, K]
                    bz = dt_tensor[3, dt_ch, I, J, K]
                    if bx > 0 && by > 0 && bz > 0
                        dx = Float32((I - bx) * sp_x)
                        dy = Float32((J - by) * sp_y)
                        dz = Float32((K - bz) * sp_z)
                        dist = sqrt(dx*dx + dy*dy + dz*dz)
                        if dist <= inst.arg2
                            out_val = true
                        end
                    end
                end

            # ── Morphological Dilation ────────────────────────────────────────────────
            elseif inst.opcode == OP_DILATE_3D
                r = Int32(inst.arg1)
                found = false
                for dz in -r:r, dy in -r:r, dx in -r:r
                    if dx*dx + dy*dy + dz*dz <= r*r
                        nx, ny, nz = I+dx, J+dy, K+dz
                        if nx >= 1 && nx <= dims[1] && ny >= 1 && ny <= dims[2] && nz >= 1 && nz <= dims[3]
                            val = false
                            if inst.in_ch > 0
                                raw = tensor_in[inst.in_ch, nx, ny, nz]
                                val = inst.in_id == 0 ? (raw > 0) : (raw == inst.in_id)
                            elseif inst.in_ch < 0
                                val = (tensor_out[-inst.in_ch, nx, ny, nz] > 0)
                            end
                            if val
                                found = true
                                break
                            end
                        end
                    end
                end
                out_val = found

            # ── Morphological Erosion ─────────────────────────────────────────────────
            elseif inst.opcode == OP_ERODE_3D
                r = Int32(inst.arg1)
                found = true
                for dz in -r:r, dy in -r:r, dx in -r:r
                    if dx*dx + dy*dy + dz*dz <= r*r
                        nx, ny, nz = I+dx, J+dy, K+dz
                        if nx >= 1 && nx <= dims[1] && ny >= 1 && ny <= dims[2] && nz >= 1 && nz <= dims[3]
                            val = false
                            if inst.in_ch > 0
                                raw = tensor_in[inst.in_ch, nx, ny, nz]
                                val = inst.in_id == 0 ? (raw > 0) : (raw == inst.in_id)
                            elseif inst.in_ch < 0
                                val = (tensor_out[-inst.in_ch, nx, ny, nz] > 0)
                            end
                            if !val
                                found = false
                                break
                            end
                        else
                            found = false
                            break
                        end
                    end
                end
                out_val = found

            # ── Volumetric boundary (2D slice-wise) ─────────────────────────
            # This opcode is handled by the hull batch path (execute_mega2_hull), not
            # the main batch kernel. If it appears here, fall through to out_val=false.
            # elseif inst.opcode == OP_VOLUMETRIC_BOUNDARY → handled externally

            end

            # Write result — always write (both 0 and 1) so that AND/SUB ops can clear voxels.
            # Guard: only write if out_ch is within bounds of tensor_out.
            if inst.out_ch > 0 && inst.out_ch <= size(tensor_out, 1)
                tensor_out[inst.out_ch, I, J, K] = out_val ? UInt8(1) : UInt8(0)
            end
        end
    end
end
