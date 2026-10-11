module HeatDiffusion

using CUDA
using ChainRulesCore

export heat_diffusion, extract_mask, varadhan_distance

# Directional spatial shift operators with zero/replicate padding for N-D arrays
@inline xp(x) = cat(selectdim(x, 1, 2:size(x,1)), selectdim(x, 1, size(x,1):size(x,1)), dims=1)
@inline xm(x) = cat(selectdim(x, 1, 1:1), selectdim(x, 1, 1:size(x,1)-1), dims=1)
@inline yp(x) = cat(selectdim(x, 2, 2:size(x,2)), selectdim(x, 2, size(x,2):size(x,2)), dims=2)
@inline ym(x) = cat(selectdim(x, 2, 1:1), selectdim(x, 2, 1:size(x,2)-1), dims=2)
@inline zp(x) = cat(selectdim(x, 3, 2:size(x,3)), selectdim(x, 3, size(x,3):size(x,3)), dims=3)
@inline zm(x) = cat(selectdim(x, 3, 1:1), selectdim(x, 3, 1:size(x,3)-1), dims=3)

"""
    diffuse_step(u, D, dt)

Executes a single step of inhomogeneous 3D heat diffusion on grid (X, Y, Z, B):
∂u/∂t = ∇ · (D ∇u)
where D is the spatially varying diffusivity/conductance field.
"""
function diffuse_step(u, D, dt)
    # Face conductivities via arithmetic average of adjacent voxels
    D_xp = 0.5f0 .* (D .+ xp(D))
    D_xm = 0.5f0 .* (D .+ xm(D))
    D_yp = 0.5f0 .* (D .+ yp(D))
    D_ym = 0.5f0 .* (D .+ ym(D))
    D_zp = 0.5f0 .* (D .+ zp(D))
    D_zm = 0.5f0 .* (D .+ zm(D))

    flux = D_xp .* (xp(u) .- u) .+
           D_xm .* (xm(u) .- u) .+
           D_yp .* (yp(u) .- u) .+
           D_ym .* (ym(u) .- u) .+
           D_zp .* (zp(u) .- u) .+
           D_zm .* (zm(u) .- u)

    return u .+ dt .* flux
end

"""
    heat_diffusion(u0, D, K, dt)

Solves 3D inhomogeneous heat diffusion for K time steps starting from initial distribution u0.
Differentiable with exact adjoint state method via ChainRulesCore.rrule.
"""
function heat_diffusion(u0, D, K::Int, dt::Float32)
    u = u0
    for k in 1:K
        u = diffuse_step(u, D, dt)
    end
    return u
end

# Exact Analytic Adjoint rrule
function ChainRulesCore.rrule(::typeof(heat_diffusion), u0, D, K::Int, dt::Float32)
    trajectory = Vector{typeof(u0)}(undef, K + 1)
    trajectory[1] = u0
    for k in 1:K
        trajectory[k + 1] = diffuse_step(trajectory[k], D, dt)
    end
    u_final = trajectory[end]

    function heat_diffusion_pullback(Δu_in)
        λ = ChainRulesCore.unthunk(Δu_in)
        grad_D = CUDA.zeros(Float32, size(D)...)

        for k in K:-1:1
            u_k = trajectory[k]
            
            # Gradients across positive coordinate faces
            du_xp = xp(u_k) .- u_k; dλ_xp = xp(λ) .- λ
            du_yp = yp(u_k) .- u_k; dλ_yp = yp(λ) .- λ
            du_zp = zp(u_k) .- u_k; dλ_zp = zp(λ) .- λ

            g_xp = -dt .* du_xp .* dλ_xp
            g_yp = -dt .* du_yp .* dλ_yp
            g_zp = -dt .* du_zp .* dλ_zp

            # Project face gradients onto voxel centers
            g_D = 0.5f0 .* (g_xp .+ xm(g_xp) .+
                            g_yp .+ ym(g_yp) .+
                            g_zp .+ zm(g_zp))

            grad_D .+= g_D
            λ = diffuse_step(λ, D, dt)
        end
        return (NoTangent(), λ, grad_D, NoTangent(), NoTangent())
    end

    return u_final, heat_diffusion_pullback
end

"""
    varadhan_distance(u, dt, K)

Estimates geodesic distance from heat profile using Varadhan's asymptotic relation:
d_g ≈ sqrt(-2 * t * log(u / u_max + eps))
"""
function varadhan_distance(u, dt::Float32, K::Int)
    t = dt * Float32(K)
    u_clamped = max.(u, 1f-8)
    # Varadhan distance metric
    return sqrt.(max.(-2.0f0 .* t .* log.(u_clamped), 0.0f0))
end

"""
    extract_mask(u, theta, tau=0.05f0)

Differentiable soft thresholding to extract lesion mask from heat potential:
M_hat = sigmoid((u - theta) / tau)
"""
function extract_mask(u, theta::Float32, tau::Float32=0.05f0)
    return 1.0f0 ./ (1.0f0 .+ exp.(-(u .- theta) ./ tau))
end

end # module
