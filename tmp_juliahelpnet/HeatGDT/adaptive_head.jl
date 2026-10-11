# adaptive_head.jl — Adaptive diffusion parameter head for Heat-GDT
# Included directly (no module wrapper)

using Lux
using Statistics: mean, std

"""
    AdaptiveParamHead(; feat_dim=8, K_base=20)

Lightweight MLP that takes local seed features and outputs optimal `(K_scale, theta, tau)`.

# Arguments
- `feat_dim`: number of input features (e.g., local SUV mean, CT mean, lesion radius estimate, etc.)
- `K_base`: base number of diffusion steps (stored, not trainable)

# Outputs (named tuple)
- `K_scale ∈ [0.5, 2.0]`: multiplier for base diffusion steps K
- `theta   ∈ [0.005, 0.035]`: optimal threshold
- `tau     ∈ [0.01, 0.05]`: soft threshold temperature
"""
struct AdaptiveParamHead <: Lux.AbstractLuxContainerLayer{(:mlp,)}
    mlp
    K_base::Int
end

function AdaptiveParamHead(; feat_dim::Int=8, K_base::Int=20)
    mlp = Chain(
        Dense(feat_dim, 32, relu),
        Dense(32, 16, relu),
        Dense(16, 3)  # outputs: raw_K_scale, raw_theta, raw_tau
    )
    return AdaptiveParamHead(mlp, K_base)
end

function (m::AdaptiveParamHead)(features, ps, st)
    raw, st_new = m.mlp(features, ps.mlp, st.mlp)

    # Clamp outputs to valid ranges using sigmoid scaling
    K_scale = 0.5f0  .+ 1.5f0  .* sigmoid.(raw[1:1, :])  # [0.5,  2.0  ]
    theta   = 0.005f0 .+ 0.030f0 .* sigmoid.(raw[2:2, :])  # [0.005, 0.035]
    tau     = 0.01f0  .+ 0.04f0  .* sigmoid.(raw[3:3, :])  # [0.01,  0.05 ]

    return (K_scale=K_scale, theta=theta, tau=tau), (mlp=st_new,)
end

"""
    extract_seed_features(x, seed_mask)

Extract local features around the seed point for adaptive parameter conditioning.

# Arguments
- `x`: `(X, Y, Z, C, B)` input volume (expects C ≥ 3: CT, PET, tissue/anatomy)
- `seed_mask`: `(X, Y, Z, B)` binary seed mask

# Returns
`(8, B)` Float32 feature matrix with columns:
1. `mean_ct`          — global CT mean
2. `std_ct`           — global CT standard deviation
3. `mean_pet`         — global PET mean
4. `max_pet`          — global PET maximum
5. `mean_ts`          — global tissue/anatomy mean
6. `seed_radius_est`  — cube-root of seed voxel count (radius proxy)
7. `total_intensity`  — mean PET intensity within seed
8. `lesion_density`   — mean tissue value within seed
"""
function extract_seed_features(x, seed_mask)
    B = size(x, 5)
    feats = zeros(Float32, 8, B)

    for b in 1:B
        ct  = x[:, :, :, 1, b]
        pet = x[:, :, :, 2, b]
        ts  = x[:, :, :, 3, b]
        s   = seed_mask[:, :, :, b]

        # Global statistics
        feats[1, b] = mean(ct)
        feats[2, b] = std(ct)
        feats[3, b] = mean(pet)
        feats[4, b] = maximum(pet)
        feats[5, b] = mean(ts)

        # Seed-based features
        n_seed = sum(s)
        feats[6, b] = cbrt(Float32(n_seed))              # radius estimate
        feats[7, b] = sum(pet .* s) / max(n_seed, 1.0f0) # local intensity
        feats[8, b] = sum(ts .* s)  / max(n_seed, 1.0f0) # anatomical context
    end

    return feats
end
