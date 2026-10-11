module HeatDiffusionV2

using CUDA
using ChainRulesCore

export heat_diffusion_v2, extract_mask, varadhan_distance

# ──────────────────────────────────────────────────────────────────────────────
# Directional spatial shift operators with replicate padding for N-D arrays
# ──────────────────────────────────────────────────────────────────────────────
@inline xp(x) = cat(selectdim(x, 1, 2:size(x,1)), selectdim(x, 1, size(x,1):size(x,1)), dims=1)
@inline xm(x) = cat(selectdim(x, 1, 1:1), selectdim(x, 1, 1:size(x,1)-1), dims=1)
@inline yp(x) = cat(selectdim(x, 2, 2:size(x,2)), selectdim(x, 2, size(x,2):size(x,2)), dims=2)
@inline ym(x) = cat(selectdim(x, 2, 1:1), selectdim(x, 2, 1:size(x,2)-1), dims=2)
@inline zp(x) = cat(selectdim(x, 3, 2:size(x,3)), selectdim(x, 3, size(x,3):size(x,3)), dims=3)
@inline zm(x) = cat(selectdim(x, 3, 1:1), selectdim(x, 3, 1:size(x,3)-1), dims=3)

# ──────────────────────────────────────────────────────────────────────────────
# Extract velocity components from (X, Y, Z, 3, B) → (X, Y, Z, B) each
# ──────────────────────────────────────────────────────────────────────────────
@inline vx_of(v) = v[:, :, :, 1, :]
@inline vy_of(v) = v[:, :, :, 2, :]
@inline vz_of(v) = v[:, :, :, 3, :]

"""
    diffuse_step_v2(u, D, beta, v, dt)

Single step of the reaction-diffusion-advection PDE on grid (X, Y, Z, B):

    ∂u/∂t = ∇·(D ∇u) − β u − v·∇u

- `D`    : diffusivity field     (X, Y, Z, B)
- `beta` : absorption/reaction   (X, Y, Z, B), ≥ 0
- `v`    : advection velocity    (X, Y, Z, 3, B)
"""
function diffuse_step_v2(u, D, beta, v, dt)
    # ── Diffusion: face conductivities via arithmetic average ──
    D_xp = 0.5f0 .* (D .+ xp(D))
    D_xm = 0.5f0 .* (D .+ xm(D))
    D_yp = 0.5f0 .* (D .+ yp(D))
    D_ym = 0.5f0 .* (D .+ ym(D))
    D_zp = 0.5f0 .* (D .+ zp(D))
    D_zm = 0.5f0 .* (D .+ zm(D))

    flux_diffusion = D_xp .* (xp(u) .- u) .+
                     D_xm .* (xm(u) .- u) .+
                     D_yp .* (yp(u) .- u) .+
                     D_ym .* (ym(u) .- u) .+
                     D_zp .* (zp(u) .- u) .+
                     D_zm .* (zm(u) .- u)

    # ── Advection: central-difference upwind ──
    vx = vx_of(v)
    vy = vy_of(v)
    vz = vz_of(v)

    advection = vx .* (xp(u) .- xm(u)) ./ 2.0f0 .+
                vy .* (yp(u) .- ym(u)) ./ 2.0f0 .+
                vz .* (zp(u) .- zm(u)) ./ 2.0f0

    # ── Reaction (absorption) ──
    reaction = beta .* u

    return u .+ dt .* (flux_diffusion .- reaction .- advection)
end

"""
    heat_diffusion_v2(u0, D, beta, v, K, dt)

Solves the 3D reaction-diffusion-advection PDE for `K` time steps:

    ∂u/∂t = ∇·(D ∇u) − β u − v·∇u

Differentiable with exact analytic adjoint via `ChainRulesCore.rrule`.

# Arguments
- `u0`   : initial distribution   (X, Y, Z, B)
- `D`    : diffusivity field      (X, Y, Z, B)
- `beta` : absorption/reaction    (X, Y, Z, B), ≥ 0
- `v`    : advection velocity     (X, Y, Z, 3, B)
- `K`    : number of time steps
- `dt`   : time step size
"""
function heat_diffusion_v2(u0, D, beta, v, K::Int, dt::Float32)
    u = u0
    for k in 1:K
        u = diffuse_step_v2(u, D, beta, v, dt)
    end
    return u
end

"""
    heat_diffusion_v2(u0, D, K, dt)

Backward-compatible wrapper: pure diffusion (β = 0, v = 0).
"""
function heat_diffusion_v2(u0, D, K::Int, dt::Float32)
    beta = CUDA.zeros(Float32, size(D)...)
    v    = CUDA.zeros(Float32, size(D, 1), size(D, 2), size(D, 3), 3, size(D, 4))
    return heat_diffusion_v2(u0, D, beta, v, K, dt)
end

function ChainRulesCore.rrule(::typeof(heat_diffusion_v2),
                              u0, D, K::Int, dt::Float32)
    beta = CUDA.zeros(Float32, size(D)...)
    v    = CUDA.zeros(Float32, size(D, 1), size(D, 2), size(D, 3), 3, size(D, 4))
    u_final, pb6 = ChainRulesCore.rrule(heat_diffusion_v2, u0, D, beta, v, K, dt)
    function heat_diffusion_v2_4arg_pullback(Δu_in)
        _, grad_u0, grad_D, _, _, _, _ = pb6(Δu_in)
        return (NoTangent(), grad_u0, grad_D, NoTangent(), NoTangent())
    end
    return u_final, heat_diffusion_v2_4arg_pullback
end

# ──────────────────────────────────────────────────────────────────────────────
# Exact Analytic Adjoint via ChainRulesCore.rrule (6 arguments)
# ──────────────────────────────────────────────────────────────────────────────
function ChainRulesCore.rrule(::typeof(heat_diffusion_v2),
                              u0, D, beta, v, K::Int, dt::Float32)
    # ── Forward pass: store full trajectory ──
    trajectory = Vector{typeof(u0)}(undef, K + 1)
    trajectory[1] = u0
    for k in 1:K
        trajectory[k + 1] = diffuse_step_v2(trajectory[k], D, beta, v, dt)
    end
    u_final = trajectory[end]

    function heat_diffusion_v2_pullback(Δu_in)
        λ = ChainRulesCore.unthunk(Δu_in)

        grad_D    = CUDA.zeros(Float32, size(D)...)
        grad_beta = CUDA.zeros(Float32, size(beta)...)
        grad_v    = CUDA.zeros(Float32, size(v)...)

        vx = vx_of(v)
        vy = vy_of(v)
        vz = vz_of(v)

        for k in K:-1:1
            u_k = trajectory[k]

            # ── grad_D: face-conductivity gradient (same as original) ──
            du_xp = xp(u_k) .- u_k; dλ_xp = xp(λ) .- λ
            du_yp = yp(u_k) .- u_k; dλ_yp = yp(λ) .- λ
            du_zp = zp(u_k) .- u_k; dλ_zp = zp(λ) .- λ

            g_xp = -dt .* du_xp .* dλ_xp
            g_yp = -dt .* du_yp .* dλ_yp
            g_zp = -dt .* du_zp .* dλ_zp

            # Project face gradients onto voxel centres
            g_D = 0.5f0 .* (g_xp .+ xm(g_xp) .+
                             g_yp .+ ym(g_yp) .+
                             g_zp .+ zm(g_zp))
            grad_D .+= g_D

            # ── grad_beta: −dt * u^(k) * λ^(k+1) ──
            grad_beta .+= -dt .* u_k .* λ

            # ── grad_v: −dt * λ^(k+1) * central_grad_u for each direction ──
            du_cd_x = (xp(u_k) .- xm(u_k)) ./ 2.0f0
            du_cd_y = (yp(u_k) .- ym(u_k)) ./ 2.0f0
            du_cd_z = (zp(u_k) .- zm(u_k)) ./ 2.0f0

            grad_v[:, :, :, 1, :] .+= -dt .* λ .* du_cd_x
            grad_v[:, :, :, 2, :] .+= -dt .* λ .* du_cd_y
            grad_v[:, :, :, 3, :] .+= -dt .* λ .* du_cd_z

            # ── Adjoint evolution of λ ──
            # Diffusion adjoint (same operator as forward, self-adjoint)
            L_D_λ = let
                D_xp = 0.5f0 .* (D .+ xp(D))
                D_xm = 0.5f0 .* (D .+ xm(D))
                D_yp = 0.5f0 .* (D .+ yp(D))
                D_ym = 0.5f0 .* (D .+ ym(D))
                D_zp = 0.5f0 .* (D .+ zp(D))
                D_zm = 0.5f0 .* (D .+ zm(D))

                D_xp .* (xp(λ) .- λ) .+
                D_xm .* (xm(λ) .- λ) .+
                D_yp .* (yp(λ) .- λ) .+
                D_ym .* (ym(λ) .- λ) .+
                D_zp .* (zp(λ) .- λ) .+
                D_zm .* (zm(λ) .- λ)
            end

            # Advection adjoint: ∇·(v λ)
            div_vλ = (xp(vx .* λ) .- xm(vx .* λ)) ./ 2.0f0 .+
                     (yp(vy .* λ) .- ym(vy .* λ)) ./ 2.0f0 .+
                     (zp(vz .* λ) .- zm(vz .* λ)) ./ 2.0f0

            # λ^(k) = λ^(k+1) + dt * (L_D[λ] − β λ + ∇·(v λ))
            λ = λ .+ dt .* (L_D_λ .- beta .* λ .+ div_vλ)
        end

        # After the backward loop, λ holds the gradient w.r.t. u0
        return (NoTangent(), λ, grad_D, grad_beta, grad_v, NoTangent(), NoTangent())
    end

    return u_final, heat_diffusion_v2_pullback
end

# ──────────────────────────────────────────────────────────────────────────────
# Utility functions (identical to original)
# ──────────────────────────────────────────────────────────────────────────────

"""
    varadhan_distance(u, dt, K)

Estimates geodesic distance from heat profile using Varadhan's asymptotic relation:
    d_g ≈ sqrt(-2 t log(u / u_max + ε))
"""
function varadhan_distance(u, dt::Float32, K::Int)
    t = dt * Float32(K)
    u_clamped = max.(u, 1f-8)
    return sqrt.(max.(-2.0f0 .* t .* log.(u_clamped), 0.0f0))
end

"""
    extract_mask(u, theta, tau=0.05f0)

Differentiable soft thresholding to extract lesion mask from heat potential:
    M̂ = σ((u − θ) / τ)
"""
function extract_mask(u, theta::Float32, tau::Float32=0.05f0)
    return 1.0f0 ./ (1.0f0 .+ exp.(-(u .- theta) ./ tau))
end

end # module HeatDiffusionV2
