using Statistics

# Create a small 64x64x64 volume
D_patch = ones(Float32, 64, 64, 64)
seed_mask = zeros(Float32, 64, 64, 64)
# Add a 2D circle in the center slice
for y in 1:64, x in 1:64
    if (x - 32)^2 + (y - 32)^2 <= 16
        seed_mask[x, y, 32] = 1.0f0
    end
end
println("Initial voxels: ", count(seed_mask .> 0))

function _replicate_pad(arr)
    s = size(arr)
    padded = similar(arr, s[1]+2, s[2]+2, s[3]+2, s[4])
    padded[2:end-1, 2:end-1, 2:end-1, :] .= arr
    padded[1, :, :, :] .= padded[2, :, :, :]
    padded[end, :, :, :] .= padded[end-1, :, :, :]
    padded[:, 1, :, :] .= padded[:, 2, :, :]
    padded[:, end, :, :] .= padded[:, end-1, :, :]
    padded[:, :, 1, :] .= padded[:, :, 2, :]
    padded[:, :, end, :] .= padded[:, :, end-1, :]
    return padded
end

D_4d = reshape(D_patch, 64, 64, 64, 1)
seed_4d = reshape(seed_mask, 64, 64, 64, 1)

D_gpu = D_4d
N = 64
D_padded = _replicate_pad(D_gpu)
D_xp_nbr = D_padded[3:N+2, 2:N+1, 2:N+1, :]
D_xm_nbr = D_padded[1:N,   2:N+1, 2:N+1, :]
D_yp_nbr = D_padded[2:N+1, 3:N+2, 2:N+1, :]
D_ym_nbr = D_padded[2:N+1, 1:N,   2:N+1, :]
D_zp_nbr = D_padded[2:N+1, 2:N+1, 3:N+2, :]
D_zm_nbr = D_padded[2:N+1, 2:N+1, 1:N,   :]

D_xp = 0.5f0 .* (D_gpu .+ D_xp_nbr)
D_xm = 0.5f0 .* (D_gpu .+ D_xm_nbr)
D_yp = 0.5f0 .* (D_gpu .+ D_yp_nbr)
D_ym = 0.5f0 .* (D_gpu .+ D_ym_nbr)
D_zp = 0.5f0 .* (D_gpu .+ D_zp_nbr)
D_zm = 0.5f0 .* (D_gpu .+ D_zm_nbr)

u = seed_4d
dt = 0.16f0
K = 200

for k in 1:K
    u_padded = _replicate_pad(u)
    u_xp = u_padded[3:N+2, 2:N+1, 2:N+1, :]
    u_xm = u_padded[1:N,   2:N+1, 2:N+1, :]
    u_yp = u_padded[2:N+1, 3:N+2, 2:N+1, :]
    u_ym = u_padded[2:N+1, 1:N,   2:N+1, :]
    u_zp = u_padded[2:N+1, 2:N+1, 3:N+2, :]
    u_zm = u_padded[2:N+1, 2:N+1, 1:N,   :]
    
    flux = D_xp .* (u_xp .- u) .+
           D_xm .* (u_xm .- u) .+
           D_yp .* (u_yp .- u) .+
           D_ym .* (u_ym .- u) .+
           D_zp .* (u_zp .- u) .+
           D_zm .* (u_zm .- u)
    global u = u .+ dt .* flux
end

theta = 0.001f0
tau = 0.0001f0
mask_field = 1.0f0 ./ (1.0f0 .+ exp.(-(u .- theta) ./ tau))
binary_mask = UInt8.(mask_field[:, :, :, 1] .> 0.5f0)
println("Final voxels (K=200): ", count(binary_mask .> 0))

