# ============================================================================
# gpu_edt.jl — Exact 3D Euclidean Distance Transform on GPU
# ============================================================================
# Implements the Felzenszwalb-Huttenlocher separable parabola algorithm
# as 3 KernelAbstractions passes (one per axis).
# Matches scipy.ndimage.distance_transform_edt exactly.
#
# Usage:
#   dt_sq = gpu_exact_edt(backend, seed_mask_gpu, spacing)
#   # seed_mask_gpu: UInt8 GPU array, >0 = seed (distance 0)
#   # spacing: (sx, sy, sz) in mm
#   # Returns: Float32 GPU array of SQUARED Euclidean distances
# ============================================================================

using KernelAbstractions, CUDA, Adapt

const EDT_LARGE = 1.0f10  # Sentinel for "no seed" — larger than any real dist²

# ── Initialization kernel ─────────────────────────────────────────────────────
@kernel function edt_init_kernel!(D, @Const(seed_mask), dims_x::Int32, dims_y::Int32, dims_z::Int32)
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        @inbounds D[i, j, k] = seed_mask[i, j, k] > UInt8(0) ? 0.0f0 : EDT_LARGE
    end
end

# ── 1D Felzenszwalb EDT along X axis ─────────────────────────────────────────
# Each thread processes one (j,k) line of length dims_x
@kernel function edt_pass_x_kernel!(
    D_out, @Const(D_in),
    v_buf, z_buf,          # scratch: [max_line_len, n_lines]
    dims_x::Int32, dims_y::Int32, dims_z::Int32,
    sp_sq::Float32         # spacing_x²
)
    line_id = @index(Global)
    j = Int32(((line_id - Int32(1)) % dims_y) + Int32(1))
    k = Int32(((line_id - Int32(1)) ÷ dims_y) + Int32(1))
    if j <= dims_y && k <= dims_z
        n = dims_x

        @inbounds begin
            # --- Phase 1: Build lower envelope of parabolas ---
            kk = Int32(1)
            v_buf[1, line_id] = Int32(1)
            z_buf[1, line_id] = -1.0f18
            z_buf[2, line_id] = 1.0f18

            for q in Int32(2):n
                f_q = D_in[q, j, k]
                p = v_buf[kk, line_id]
                f_p = D_in[p, j, k]
                s = ((f_q + Float32(q) * Float32(q) * sp_sq) -
                     (f_p + Float32(p) * Float32(p) * sp_sq)) /
                    (2.0f0 * sp_sq * Float32(q - p))

                while s <= z_buf[kk, line_id]
                    kk -= Int32(1)
                    if kk < Int32(1)
                        break
                    end
                    p = v_buf[kk, line_id]
                    f_p = D_in[p, j, k]
                    s = ((f_q + Float32(q) * Float32(q) * sp_sq) -
                         (f_p + Float32(p) * Float32(p) * sp_sq)) /
                        (2.0f0 * sp_sq * Float32(q - p))
                end

                if kk < Int32(1)
                    kk = Int32(1)
                    v_buf[1, line_id] = q
                    z_buf[1, line_id] = -1.0f18
                    z_buf[2, line_id] = 1.0f18
                else
                    kk += Int32(1)
                    v_buf[kk, line_id] = q
                    z_buf[kk, line_id] = s
                    z_buf[kk + Int32(1), line_id] = 1.0f18
                end
            end

            # --- Phase 2: Evaluate lower envelope ---
            ll = Int32(1)
            for q in Int32(1):n
                while z_buf[ll + Int32(1), line_id] < Float32(q)
                    ll += Int32(1)
                end
                p = v_buf[ll, line_id]
                D_out[q, j, k] = sp_sq * Float32((q - p) * (q - p)) + D_in[p, j, k]
            end
        end # @inbounds
    end # bounds check
end

# ── 1D Felzenszwalb EDT along Y axis ─────────────────────────────────────────
# Each thread processes one (i,k) line of length dims_y
@kernel function edt_pass_y_kernel!(
    D_out, @Const(D_in),
    v_buf, z_buf,
    dims_x::Int32, dims_y::Int32, dims_z::Int32,
    sp_sq::Float32         # spacing_y²
)
    line_id = @index(Global)
    i = Int32(((line_id - Int32(1)) % dims_x) + Int32(1))
    k = Int32(((line_id - Int32(1)) ÷ dims_x) + Int32(1))
    if i <= dims_x && k <= dims_z
        n = dims_y

        @inbounds begin
            kk = Int32(1)
            v_buf[1, line_id] = Int32(1)
            z_buf[1, line_id] = -1.0f18
            z_buf[2, line_id] = 1.0f18

            for q in Int32(2):n
                f_q = D_in[i, q, k]
                p = v_buf[kk, line_id]
                f_p = D_in[i, p, k]
                s = ((f_q + Float32(q) * Float32(q) * sp_sq) -
                     (f_p + Float32(p) * Float32(p) * sp_sq)) /
                    (2.0f0 * sp_sq * Float32(q - p))

                while s <= z_buf[kk, line_id]
                    kk -= Int32(1)
                    if kk < Int32(1); break; end
                    p = v_buf[kk, line_id]
                    f_p = D_in[i, p, k]
                    s = ((f_q + Float32(q) * Float32(q) * sp_sq) -
                         (f_p + Float32(p) * Float32(p) * sp_sq)) /
                        (2.0f0 * sp_sq * Float32(q - p))
                end

                if kk < Int32(1)
                    kk = Int32(1)
                    v_buf[1, line_id] = q
                    z_buf[1, line_id] = -1.0f18
                    z_buf[2, line_id] = 1.0f18
                else
                    kk += Int32(1)
                    v_buf[kk, line_id] = q
                    z_buf[kk, line_id] = s
                    z_buf[kk + Int32(1), line_id] = 1.0f18
                end
            end

            ll = Int32(1)
            for q in Int32(1):n
                while z_buf[ll + Int32(1), line_id] < Float32(q)
                    ll += Int32(1)
                end
                p = v_buf[ll, line_id]
                D_out[i, q, k] = sp_sq * Float32((q - p) * (q - p)) + D_in[i, p, k]
            end
        end
    end
end

# ── 1D Felzenszwalb EDT along Z axis ─────────────────────────────────────────
# Each thread processes one (i,j) line of length dims_z
@kernel function edt_pass_z_kernel!(
    D_out, @Const(D_in),
    v_buf, z_buf,
    dims_x::Int32, dims_y::Int32, dims_z::Int32,
    sp_sq::Float32         # spacing_z²
)
    line_id = @index(Global)
    i = Int32(((line_id - Int32(1)) % dims_x) + Int32(1))
    j = Int32(((line_id - Int32(1)) ÷ dims_x) + Int32(1))
    if i <= dims_x && j <= dims_y
        n = dims_z

        @inbounds begin
            kk = Int32(1)
            v_buf[1, line_id] = Int32(1)
            z_buf[1, line_id] = -1.0f18
            z_buf[2, line_id] = 1.0f18

            for q in Int32(2):n
                f_q = D_in[i, j, q]
                p = v_buf[kk, line_id]
                f_p = D_in[i, j, p]
                s = ((f_q + Float32(q) * Float32(q) * sp_sq) -
                     (f_p + Float32(p) * Float32(p) * sp_sq)) /
                    (2.0f0 * sp_sq * Float32(q - p))

                while s <= z_buf[kk, line_id]
                    kk -= Int32(1)
                    if kk < Int32(1); break; end
                    p = v_buf[kk, line_id]
                    f_p = D_in[i, j, p]
                    s = ((f_q + Float32(q) * Float32(q) * sp_sq) -
                         (f_p + Float32(p) * Float32(p) * sp_sq)) /
                        (2.0f0 * sp_sq * Float32(q - p))
                end

                if kk < Int32(1)
                    kk = Int32(1)
                    v_buf[1, line_id] = q
                    z_buf[1, line_id] = -1.0f18
                    z_buf[2, line_id] = 1.0f18
                else
                    kk += Int32(1)
                    v_buf[kk, line_id] = q
                    z_buf[kk, line_id] = s
                    z_buf[kk + Int32(1), line_id] = 1.0f18
                end
            end

            ll = Int32(1)
            for q in Int32(1):n
                while z_buf[ll + Int32(1), line_id] < Float32(q)
                    ll += Int32(1)
                end
                p = v_buf[ll, line_id]
                D_out[i, j, q] = sp_sq * Float32((q - p) * (q - p)) + D_in[i, j, p]
            end
        end
    end
end

# ── GPU overlap assignment kernel ─────────────────────────────────────────────
@kernel function edt_assign_overlap_kernel!(
    arr1, arr2,
    @Const(overlap),
    @Const(dt1), @Const(dt2),
    dims_x::Int32, dims_y::Int32, dims_z::Int32
)
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        @inbounds if overlap[i, j, k] > UInt8(0)
            d1 = dt1[i, j, k]
            d2 = dt2[i, j, k]
            if d1 < d2
                # mask1 is closer → mask1 wins → zero mask2
                arr2[i, j, k] = UInt8(0)
            else
                # mask2 wins (ties go to mask2, matching Python: dt2 <= dt1)
                arr1[i, j, k] = UInt8(0)
            end
        end
    end
end

# ── Main entry point ──────────────────────────────────────────────────────────
"""
    gpu_exact_edt(backend, seed_mask_gpu, spacing) → Float32 GPU array

Compute exact 3D Euclidean Distance Transform fully on GPU.
Uses Felzenszwalb-Huttenlocher separable parabola algorithm (3 passes).

- `seed_mask_gpu`: UInt8 GPU array where >0 marks seed voxels (distance = 0)
- `spacing`: (sx, sy, sz) voxel spacing in mm
- Returns: Float32 GPU array of **squared** Euclidean distances

Matches `scipy.ndimage.distance_transform_edt(~seed_mask, sampling=spacing)**2`.
"""
function gpu_exact_edt!(out, backend, seed_mask_gpu, spacing)
    dims = size(seed_mask_gpu)
    dx, dy, dz = Int32(dims[1]), Int32(dims[2]), Int32(dims[3])
    max_dim = max(dims...)

    if !haskey(StaticArena.EDT_ARENA, :D_a)
        StaticArena.init_edt_arena(backend, dims)
    end

    # Fetch pre-allocated buffers from StaticArena
    D_a = view(StaticArena.EDT_ARENA[:D_a], 1:Int(dx), 1:Int(dy), 1:Int(dz))
    D_b = view(StaticArena.EDT_ARENA[:D_b], 1:Int(dx), 1:Int(dy), 1:Int(dz))

    # Step 0: Initialize D_a from seed mask
    edt_init_kernel!(backend)(D_a, seed_mask_gpu, dx, dy, dz, ndrange=dims)
    KernelAbstractions.synchronize(backend)

    # Step 1: Pass along X — D_a → D_b
    n_x = Int(dy) * Int(dz)
    v_x = view(StaticArena.EDT_ARENA[:v_x], 1:Int(dx), 1:n_x)
    z_x = view(StaticArena.EDT_ARENA[:z_x], 1:(Int(dx) + 1), 1:n_x)
    edt_pass_x_kernel!(backend)(D_b, D_a, v_x, z_x, dx, dy, dz,
                                Float32(spacing[1])^2, ndrange=n_x)
    KernelAbstractions.synchronize(backend)

    # Step 2: Pass along Y — D_b → D_a
    n_y = Int(dx) * Int(dz)
    v_y = view(StaticArena.EDT_ARENA[:v_y], 1:Int(dy), 1:n_y)
    z_y = view(StaticArena.EDT_ARENA[:z_y], 1:(Int(dy) + 1), 1:n_y)
    edt_pass_y_kernel!(backend)(D_a, D_b, v_y, z_y, dx, dy, dz,
                                Float32(spacing[2])^2, ndrange=n_y)
    KernelAbstractions.synchronize(backend)

    # Step 3: Pass along Z — D_a → D_b
    n_z = Int(dx) * Int(dy)
    v_z = view(StaticArena.EDT_ARENA[:v_z], 1:Int(dz), 1:n_z)
    z_z = view(StaticArena.EDT_ARENA[:z_z], 1:(Int(dz) + 1), 1:n_z)
    edt_pass_z_kernel!(backend)(D_b, D_a, v_z, z_z, dx, dy, dz,
                                Float32(spacing[3])^2, ndrange=n_z)
    KernelAbstractions.synchronize(backend)

    # We must copy D_b out so it isn't overwritten by subsequent calls.
    # We can fetch a preallocated array from EDT_ARENA!
    # Let's assume the caller passes in an `out` buffer or we just allocate it for now.
    # Actually, allocating just ONE Float32 array of crop_dims is negligible!
    # It's the 8 arrays together that cause OOM.
    copyto!(out, D_b)
    return out
end
