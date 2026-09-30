# patch_extremum_marker.jl
# Generic GPU extremum-point marker using KernelAbstractions binary tree reduction.
#
# Given any input mask channel, this finds the voxel that is most extreme along
# a specified axis direction (anterior/posterior/lateral/medial/superior/inferior),
# then stamps a single-Z-slice marker into the output channel.
#
# The result Z index is returned to Python so bbox_cache can be updated for
# downstream LimitZByLandmark rules.
#
# Direction encoding:
#   0 = anterior   → minimize Y
#   1 = posterior  → maximize Y
#   2 = lateral_left  → maximize X
#   3 = lateral_right → minimize X
#   4 = superior   → maximize Z
#   5 = inferior   → minimize Z

using KernelAbstractions
using CUDA
using Adapt
using JSON

# ── Phase 1: per-block reduction ──────────────────────────────────────────────
# Each thread covers one voxel (flat linear index over X*Y*Z).
# Reduces its workgroup of BLOCK_SIZE voxels into one (best_coord, best_z) pair
# written to block_results[block_idx, 1:2].
# BLOCK_SIZE must be a power of 2 (we always launch with 512).
@kernel function extremum_reduce_phase1!(
        tensor_in,
        block_results,
        in_ch::Int32,
        in_id::UInt16,
        direction::Int32,
        dims_x::Int32, dims_y::Int32, dims_z::Int32)

    gi  = @index(Global, Linear)
    li  = @index(Local,  Linear)
    blk = @index(Group,  Linear)

    N = @uniform @groupsize()[1]   # compile-time constant for @localmem

    # Shared memory: one pair (best_axis_coord, best_z) per thread
    sval = @localmem Int32 (N,)
    sz   = @localmem Int32 (N,)

    # Sentinel initial values based on direction (minimize vs maximize)
    INIT_VAL = (direction == Int32(0) || direction == Int32(3) || direction == Int32(5)) ?
               Int32(2147483647) :   # minimize → +INF sentinel
               Int32(-1)             # maximize → -INF sentinel

    @inbounds sval[li] = INIT_VAL
    @inbounds sz[li]   = Int32(0)

    total_voxels = dims_x * dims_y * dims_z
    if gi <= total_voxels
        # Decode flat linear index to 1-based (I, J, K)
        g = gi - 1
        K_idx = Int32(g ÷ (dims_x * dims_y) + 1)
        rem   = Int32(g % (dims_x * dims_y))
        J_idx = Int32(rem ÷ dims_x + 1)
        I_idx = Int32(rem % dims_x + 1)

        raw = tensor_in[in_ch, I_idx, J_idx, K_idx]
        is_active = (in_id == UInt16(0)) ? (raw > UInt16(0)) : (raw == in_id)

        if is_active
            coord = (direction == Int32(0)) ? J_idx :  # anterior  = min Y
                    (direction == Int32(1)) ? J_idx :  # posterior = max Y
                    (direction == Int32(2)) ? I_idx :  # left lat  = max X
                    (direction == Int32(3)) ? I_idx :  # right lat = min X
                    (direction == Int32(4)) ? K_idx :  # superior  = max Z
                                             K_idx     # inferior  = min Z
            @inbounds sval[li] = Int32(coord)
            @inbounds sz[li]   = Int32(K_idx)
        end
    end

    @synchronize

    # ── Binary tree reduction in shared memory ────────────────────────────────
    stride = N ÷ 2
    while stride > 0
        if li <= stride
            @inbounds begin
                a = sval[li]
                b = sval[li + stride]
                bz_b = sz[li + stride]
                # For minimize directions: keep smaller coord; for maximize: keep larger
                keep_b = (direction == Int32(0) || direction == Int32(3) || direction == Int32(5)) ?
                         (b != INIT_VAL && (a == INIT_VAL || b < a)) :
                         (b > a)
                if keep_b
                    sval[li] = b
                    sz[li]   = bz_b
                end
            end
        end
        @synchronize
        stride = stride ÷ 2
    end

    # First thread in block writes to global block_results
    if li == 1
        @inbounds block_results[blk, 1] = sval[1]
        @inbounds block_results[blk, 2] = sz[1]
    end
end

# ── Phase 2: reduce block_results → single global winner ─────────────────────
# n_blocks is at most ~1150 for 512×512×284 / 512. We use up to 1024 threads.
@kernel function extremum_reduce_phase2!(
        block_results,
        global_result,
        n_blocks::Int32,
        direction::Int32)

    li  = @index(Local, Linear)

    N = @uniform @groupsize()[1]

    sval2 = @localmem Int32 (N,)
    sz2   = @localmem Int32 (N,)

    INIT_VAL = (direction == Int32(0) || direction == Int32(3) || direction == Int32(5)) ?
               Int32(2147483647) : Int32(-1)

    if li <= n_blocks
        @inbounds sval2[li] = block_results[li, 1]
        @inbounds sz2[li]   = block_results[li, 2]
    else
        @inbounds sval2[li] = INIT_VAL
        @inbounds sz2[li]   = Int32(0)
    end

    @synchronize

    stride = N ÷ 2
    while stride > 0
        if li <= stride
            @inbounds begin
                a = sval2[li]
                b = sval2[li + stride]
                bz_b = sz2[li + stride]
                keep_b = (direction == Int32(0) || direction == Int32(3) || direction == Int32(5)) ?
                         (b != INIT_VAL && (a == INIT_VAL || b < a)) :
                         (b > a)
                if keep_b
                    sval2[li] = b
                    sz2[li]   = bz_b
                end
            end
        end
        @synchronize
        stride = stride ÷ 2
    end

    if li == 1
        @inbounds global_result[1] = sval2[1]
        @inbounds global_result[2] = sz2[1]
    end
end

# ── Phase 3: fill output channel at best_z with ones (full XY slice) ──────────
@kernel function fill_extremum_marker!(tensor_out, out_ch::Int32, best_z::Int32,
        dims_x::Int32, dims_y::Int32)
    I, J = @index(Global, NTuple)
    if I <= dims_x && J <= dims_y
        @inbounds tensor_out[out_ch, I, J, best_z] = UInt8(1)
    end
end

# ── Main dispatch function ────────────────────────────────────────────────────
function execute_mega2_extremum_marker(batch_json_str)
    batch_dict = JSON.parse(batch_json_str)
    out_ch    = Int32(batch_dict["out_ch"])
    props     = batch_dict["props"]
    mapping   = batch_dict["mapping"]

    lm_name       = get(props, "landmark", "")
    direction_str = get(props, "direction", "anterior")
    side_str      = get(props, "side", "")

    # Encode direction as integer
    direction = if direction_str == "anterior"
        Int32(0)
    elseif direction_str == "posterior"
        Int32(1)
    elseif direction_str == "lateral"
        side_str == "left" ? Int32(2) : Int32(3)
    elseif direction_str == "lateral_left"
        Int32(2)
    elseif direction_str == "lateral_right"
        Int32(3)
    elseif direction_str == "medial"
        # Medial = toward midline: for left side, smaller X; for right side, larger X
        side_str == "left" ? Int32(3) : Int32(2)
    elseif direction_str == "superior"
        Int32(4)
    elseif direction_str == "inferior"
        Int32(5)
    else
        Int32(0)  # default anterior
    end

    # Resolve landmark channel/id from mapping
    m = get(mapping, lm_name, nothing)
    if m === nothing
        println("[ExtremumMarker] WARNING: missing landmark $lm_name in mapping → empty output")
        return "[JSON] " * JSON.json(Dict("best_z" => 0, "direction" => direction_str, "error" => "missing landmark"))
    end
    in_ch = Int32(m["ch"])   # Already 1-based (sent from orchestrator as ch+1)
    in_id = UInt16(m["id"])

    dims    = TM_MEGA2.dims
    backend = TM_MEGA2.backend


    total_voxels = dims[1] * dims[2] * dims[3]
    BLOCK_SIZE   = 512
    n_blocks     = Int32((total_voxels + BLOCK_SIZE - 1) ÷ BLOCK_SIZE)

    # Allocate block result buffer on GPU
    block_results = adapt(backend, zeros(Int32, n_blocks, 2))
    global_result = adapt(backend, zeros(Int32, 2))

    # ── Phase 1: each block finds its local extremum ──────────────────────────
    extremum_reduce_phase1!(backend, 512)(
        TM_MEGA2.tensor_in, block_results,
        in_ch, in_id, direction,
        Int32(dims[1]), Int32(dims[2]), Int32(dims[3]),
        ndrange = (n_blocks * 512,)
    )
    KernelAbstractions.synchronize(backend)

    # ── Phase 2: reduce block_results → global winner on CPU ─────────────────
    # block_results is small (~1MB for typical volume), CPU argmin is fast.
    br_cpu = Array(block_results)  # (n_blocks, 2): col1=extremum_coord, col2=Z_index

    # Find best block (min or max of col1 depending on direction)
    want_min = (direction == Int32(0) || direction == Int32(3) || direction == Int32(5))
    best_idx = 1
    best_val = br_cpu[1, 1]
    for i in 2:n_blocks
        v = br_cpu[i, 1]
        if want_min ? (v != typemax(Int32) && (best_val == typemax(Int32) || v < best_val)) :
                      (v > best_val)
            best_val = v
            best_idx = i
        end
    end
    best_z = Int32(br_cpu[best_idx, 2])

    # Free intermediate GPU buffer
    if backend isa CUDABackend
        CUDA.unsafe_free!(block_results)
    end

    # ── Phase 3: stamp single-Z-slice marker into output channel ─────────────
    if best_z >= Int32(1) && best_z <= Int32(dims[3])
        fill_extremum_marker!(backend, 256)(
            TM_MEGA2.tensor_out, out_ch, best_z,
            Int32(dims[1]), Int32(dims[2]),
            ndrange = (dims[1], dims[2])
        )
        KernelAbstractions.synchronize(backend)
    else
        println("[ExtremumMarker] WARNING: best_z=$best_z out of range [1, $(dims[3])] for $lm_name")
    end

    println("[ExtremumMarker] landmark=$lm_name direction=$direction_str best_z=$best_z out_ch=$out_ch")
    return "[JSON] " * JSON.json(Dict("best_z" => Int(best_z), "direction" => direction_str))
end
