# ============================================================================
# gpu_feature_transform.jl — GPU 3D Nearest Feature Transform
# ============================================================================
# Extends the Felzenszwalb-Huttenlocher separable parabola algorithm to also
# track the nearest feature (seed) coordinates alongside squared distances.
#
# After 3 passes (X, Y, Z), for each voxel (i,j,k):
#   Fx[i,j,k], Fy[i,j,k], Fz[i,j,k] = coordinates of nearest seed voxel
#   D[i,j,k] = squared Euclidean distance to that seed
#
# This is used for anisotropic (directional) margin thresholding where we
# need to know the direction vector to the nearest seed, not just the distance.
# ============================================================================

using KernelAbstractions, CUDA, Adapt

const FT_LARGE = 1.0f10  # Sentinel for "no seed"

# ── Initialization kernel ─────────────────────────────────────────────────────
# Sets D=0 and F=(i,j,k) for seed voxels; D=FT_LARGE and F=(0,0,0) for background
@kernel function ft_init_kernel!(
    D::AbstractArray{Float32, 3},
    Fx::AbstractArray{Int16, 3}, Fy::AbstractArray{Int16, 3}, Fz::AbstractArray{Int16, 3},
    @Const(seed_mask::AbstractArray{T, 3}),
    dims_x::Int32, dims_y::Int32, dims_z::Int32
) where T
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        @inbounds begin
            if seed_mask[i, j, k] > zero(T)
                D[i, j, k] = 0.0f0
                Fx[i, j, k] = Int16(i)
                Fy[i, j, k] = Int16(j)
                Fz[i, j, k] = Int16(k)
            else
                D[i, j, k] = FT_LARGE
                Fx[i, j, k] = Int16(0)
                Fy[i, j, k] = Int16(0)
                Fz[i, j, k] = Int16(0)
            end
        end
    end
end

# ── 1D Feature Transform along X axis ─────────────────────────────────────────
# Each thread processes one (j,k) line. Builds lower envelope of parabolas and
# tracks which seed's X coordinate was nearest at each position.
@kernel function ft_pass_x_kernel!(
    D_out::AbstractArray{Float32, 3}, @Const(D_in::AbstractArray{Float32, 3}),
    Fx_out::AbstractArray{Int16, 3}, @Const(Fx_in::AbstractArray{Int16, 3}),
    v_buf, z_buf,
    dims_x::Int32, dims_y::Int32, dims_z::Int32,
    sp_sq::Float32
)
    line_id = @index(Global)
    j = Int32(((line_id - Int32(1)) % dims_y) + Int32(1))
    k = Int32(((line_id - Int32(1)) ÷ dims_y) + Int32(1))
    if j <= dims_y && k <= dims_z
        n = dims_x
        @inbounds begin
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
                    if kk < Int32(1); break; end
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

            ll = Int32(1)
            for q in Int32(1):n
                while z_buf[ll + Int32(1), line_id] < Float32(q)
                    ll += Int32(1)
                end
                p = v_buf[ll, line_id]
                D_out[q, j, k] = sp_sq * Float32((q - p) * (q - p)) + D_in[p, j, k]
                # Track feature: for X pass, nearest X coord is p itself
                # (input Fx_in is irrelevant in pass 1 — seed index IS the X coord)
                Fx_out[q, j, k] = Int16(p)
            end
        end
    end
end

# ── 1D Feature Transform along Y axis ─────────────────────────────────────────
@kernel function ft_pass_y_kernel!(
    D_out::AbstractArray{Float32, 3}, @Const(D_in::AbstractArray{Float32, 3}),
    Fx_out::AbstractArray{Int16, 3}, @Const(Fx_in::AbstractArray{Int16, 3}),
    Fy_out::AbstractArray{Int16, 3},
    v_buf, z_buf,
    dims_x::Int32, dims_y::Int32, dims_z::Int32,
    sp_sq::Float32
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
                # Track features: propagate Fx from winner, set Fy = p
                Fx_out[i, q, k] = Fx_in[i, p, k]
                Fy_out[i, q, k] = Int16(p)
            end
        end
    end
end

# ── 1D Feature Transform along Z axis ─────────────────────────────────────────
@kernel function ft_pass_z_kernel!(
    D_out::AbstractArray{Float32, 3}, @Const(D_in::AbstractArray{Float32, 3}),
    Fx_out::AbstractArray{Int16, 3}, @Const(Fx_in::AbstractArray{Int16, 3}),
    Fy_out::AbstractArray{Int16, 3}, @Const(Fy_in::AbstractArray{Int16, 3}),
    Fz_out::AbstractArray{Int16, 3},
    v_buf, z_buf,
    dims_x::Int32, dims_y::Int32, dims_z::Int32,
    sp_sq::Float32
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
                # Track features: propagate Fx, Fy from winner, set Fz = p
                Fx_out[i, j, q] = Fx_in[i, j, p]
                Fy_out[i, j, q] = Fy_in[i, j, p]
                Fz_out[i, j, q] = Int16(p)
            end
        end
    end
end

# ── Anisotropic directional threshold kernel ──────────────────────────────────
# For each voxel, computes the direction-weighted margin threshold and tests
# whether the distance to the nearest seed is within that threshold.
@kernel function aniso_threshold_kernel!(
    output::AbstractArray{UInt8, 3},
    @Const(Fx::AbstractArray{Int16, 3}),
    @Const(Fy::AbstractArray{Int16, 3}),
    @Const(Fz::AbstractArray{Int16, 3}),
    @Const(sp_x::Float32), @Const(sp_y::Float32), @Const(sp_z::Float32),
    @Const(mx_pos::Float32), @Const(mx_neg::Float32),
    @Const(my_pos::Float32), @Const(my_neg::Float32),
    @Const(mz_pos::Float32), @Const(mz_neg::Float32),
    @Const(dx_dim::Int32), @Const(dy_dim::Int32), @Const(dz_dim::Int32)
)
    i, j, k = @index(Global, NTuple)
    if i <= dx_dim && j <= dy_dim && k <= dz_dim
        @inbounds begin
            fx = Fx[i, j, k]; fy = Fy[i, j, k]; fz = Fz[i, j, k]
            if fx > Int16(0)  # has a nearest feature
                ddx = Float32(Int32(i) - Int32(fx)) * sp_x
                ddy = Float32(Int32(j) - Int32(fy)) * sp_y
                ddz = Float32(Int32(k) - Int32(fz)) * sp_z
                if ddx == 0f0 && ddy == 0f0 && ddz == 0f0
                    output[i, j, k] = UInt8(1)  # seed voxel — always include
                else
                    abs_dx = abs(ddx); abs_dy = abs(ddy); abs_dz = abs(ddz)
                    dist = sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
                    total = abs_dx + abs_dy + abs_dz + 1f-8
                    mx = ddx > 0f0 ? mx_pos : mx_neg
                    my = ddy > 0f0 ? my_pos : my_neg
                    mz = ddz > 0f0 ? mz_pos : mz_neg
                    threshold = (abs_dx / total) * mx + (abs_dy / total) * my + (abs_dz / total) * mz
                    output[i, j, k] = (dist <= threshold && threshold > 0f0) ? UInt8(1) : UInt8(0)
                end
            else
                output[i, j, k] = UInt8(0)
            end
        end
    end
end

# ── Isotropic threshold kernel (simple: dist² ≤ margin²) ─────────────────────
@kernel function iso_threshold_kernel!(
    output::AbstractArray{UInt8, 3},
    @Const(D_sq::AbstractArray{Float32, 3}),
    @Const(margin_sq::Float32),
    @Const(dx_dim::Int32), @Const(dy_dim::Int32), @Const(dz_dim::Int32)
)
    i, j, k = @index(Global, NTuple)
    if i <= dx_dim && j <= dy_dim && k <= dz_dim
        @inbounds begin
            output[i, j, k] = D_sq[i, j, k] <= margin_sq ? UInt8(1) : UInt8(0)
        end
    end
end

# ============================================================================
# High-level API
# ============================================================================

"""
    gpu_isotropic_expansion!(backend, output, mask, dims, spacing, margin_mm)

GPU-native isotropic expansion: include all voxels within `margin_mm` of mask.
Uses existing gpu_exact_edt! (squared EDT) and a simple threshold kernel.
Zero CPU↔GPU transfers. Returns `output`.
"""
function gpu_isotropic_expansion!(
    backend, output::AbstractArray{UInt8, 3},
    mask_gpu::AbstractArray{T, 3},
    dims::Tuple{Int, Int, Int},
    spacing,
    margin_mm::Float32
) where T
    dx, dy, dz = Int32(dims[1]), Int32(dims[2]), Int32(dims[3])

    # Convert mask to UInt8 seed if needed
    seed = mask_gpu isa CuArray{UInt8, 3} ? mask_gpu : map(x -> x > zero(T) ? UInt8(1) : UInt8(0), mask_gpu)

    # Compute squared EDT using preallocated arena
    edt_out = StaticArena.EDT_ARENA[:D_b]  # result lives in D_b after gpu_exact_edt!
    gpu_exact_edt!(edt_out, backend, seed, Float32.(spacing))
    # edt_out now holds squared distances — no sqrt needed!

    # Threshold: D² ≤ margin²
    margin_sq = margin_mm * margin_mm
    iso_threshold_kernel!(backend, (8, 8, 4))(output, edt_out, margin_sq, dx, dy, dz; ndrange=dims)
    KernelAbstractions.synchronize(backend)

    return output
end

"""
    gpu_anisotropic_expansion!(backend, output, mask, dims, spacing,
                                mx_pos, mx_neg, my_pos, my_neg, mz_pos, mz_neg,
                                Fx, Fy, Fz)

GPU-native anisotropic expansion: uses GPU feature transform to find nearest
seed coordinates, then applies directional margin thresholding.
Zero CPU↔GPU transfers. Returns `output`.
"""
function gpu_anisotropic_expansion!(
    backend, output::AbstractArray{UInt8, 3},
    mask_gpu::AbstractArray{T, 3},
    dims::Tuple{Int, Int, Int},
    spacing,
    mx_pos::Float32, mx_neg::Float32,
    my_pos::Float32, my_neg::Float32,
    mz_pos::Float32, mz_neg::Float32,
    Fx::AbstractArray{Int16, 3}, Fy::AbstractArray{Int16, 3}, Fz::AbstractArray{Int16, 3}
) where T
    dx, dy, dz = Int32(dims[1]), Int32(dims[2]), Int32(dims[3])
    sp_x, sp_y, sp_z = Float32(spacing[1]), Float32(spacing[2]), Float32(spacing[3])

    # Convert mask to UInt8 seed if needed
    seed = mask_gpu isa CuArray{UInt8, 3} ? mask_gpu : map(x -> x > zero(T) ? UInt8(1) : UInt8(0), mask_gpu)

    # Fetch preallocated ping-pong buffers from EDT arena
    D_a = view(StaticArena.EDT_ARENA[:D_a], 1:Int(dx), 1:Int(dy), 1:Int(dz))
    D_b = view(StaticArena.EDT_ARENA[:D_b], 1:Int(dx), 1:Int(dy), 1:Int(dz))

    # We need two sets of Fx buffers for ping-pong: use Fx/Fy/Fz as one set,
    # and create temporary views. Actually, for proper ping-pong with feature
    # propagation, we need: Fx_a/Fx_b for X coord, Fy for Y coord, Fz for Z coord.
    # Since X pass writes to Fx_out only, Y pass reads Fx and writes Fx+Fy,
    # Z pass reads Fx+Fy and writes Fx+Fy+Fz — we can use the same buffers
    # if we're careful about the ping-pong direction.

    # Use FT_ARENA buffers for ping-pong
    Fx_a = StaticArena.FT_ARENA[:Fx_a]
    Fx_b = StaticArena.FT_ARENA[:Fx_b]

    # Step 0: Initialize D_a, Fx_a, Fy, Fz from seed mask
    ft_init_kernel!(backend, (8, 8, 4))(D_a, Fx_a, Fy, Fz, seed, dx, dy, dz; ndrange=dims)
    KernelAbstractions.synchronize(backend)

    # Step 1: Pass along X — D_a → D_b, Fx_a → Fx_b
    n_x = Int(dy) * Int(dz)
    v_x = view(StaticArena.EDT_ARENA[:v_x], 1:Int(dx), 1:n_x)
    z_x = view(StaticArena.EDT_ARENA[:z_x], 1:(Int(dx) + 1), 1:n_x)
    ft_pass_x_kernel!(backend)(D_b, D_a, Fx_b, Fx_a, v_x, z_x, dx, dy, dz,
                                sp_x^2; ndrange=n_x)
    KernelAbstractions.synchronize(backend)

    # Step 2: Pass along Y — D_b → D_a, Fx_b → Fx_a, writes Fy
    n_y = Int(dx) * Int(dz)
    v_y = view(StaticArena.EDT_ARENA[:v_y], 1:Int(dy), 1:n_y)
    z_y = view(StaticArena.EDT_ARENA[:z_y], 1:(Int(dy) + 1), 1:n_y)
    ft_pass_y_kernel!(backend)(D_a, D_b, Fx_a, Fx_b, Fy, v_y, z_y, dx, dy, dz,
                                sp_y^2; ndrange=n_y)
    KernelAbstractions.synchronize(backend)

    # Step 3: Pass along Z — D_a → D_b, Fx_a → Fx_b, Fy → Fy (in-place read), writes Fz
    # Need Fy ping-pong too: Fy_in read, Fy_out write
    Fy_b = StaticArena.FT_ARENA[:Fy_b]
    n_z = Int(dx) * Int(dy)
    v_z = view(StaticArena.EDT_ARENA[:v_z], 1:Int(dz), 1:n_z)
    z_z = view(StaticArena.EDT_ARENA[:z_z], 1:(Int(dz) + 1), 1:n_z)
    ft_pass_z_kernel!(backend)(D_b, D_a, Fx_b, Fx_a, Fy_b, Fy, Fz, v_z, z_z, dx, dy, dz,
                                sp_z^2; ndrange=n_z)
    KernelAbstractions.synchronize(backend)

    # Step 4: Directional threshold using Fx_b, Fy_b, Fz (final feature coords)
    aniso_threshold_kernel!(backend, (8, 8, 4))(
        output, Fx_b, Fy_b, Fz,
        sp_x, sp_y, sp_z,
        mx_pos, mx_neg, my_pos, my_neg, mz_pos, mz_neg,
        dx, dy, dz; ndrange=dims
    )
    KernelAbstractions.synchronize(backend)

    return output
end
