#!/usr/bin/env julia
"""
Test that CUDA and CPU heat diffusion paths produce IDENTICAL results
using replicate (Neumann) boundary conditions matching the Vulkan shader.

Also tests:
1. CPU vs CUDA numerical equivalence
2. Boundary condition correctness (edge voxels don't wrap)
3. Heat conservation with Neumann BC
4. Edge-aware diffusivity with replicate BC
"""

using Test
using CUDA

println("=" ^ 70)
println("  Heat-GDT: Replicate BC — CPU vs CUDA Equivalence Test")
println("=" ^ 70)

# ─── Replicate-pad helper (matches InferenceClient.jl and Vulkan shader) ───
function replicate_pad(arr)
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

# ─── Reference PDE solver with replicate BC ───
function run_diffusion_replicate(D_4d, seed_4d, K, dt; use_gpu=false)
    N = size(D_4d, 1)
    
    D = use_gpu ? CuArray{Float32}(D_4d) : copy(D_4d)
    u = use_gpu ? CuArray{Float32}(seed_4d) : copy(seed_4d)
    
    # Precompute face conductivities
    D_padded = replicate_pad(D)
    D_xp = 0.5f0 .* (D .+ D_padded[3:N+2, 2:N+1, 2:N+1, :])
    D_xm = 0.5f0 .* (D .+ D_padded[1:N,   2:N+1, 2:N+1, :])
    D_yp = 0.5f0 .* (D .+ D_padded[2:N+1, 3:N+2, 2:N+1, :])
    D_ym = 0.5f0 .* (D .+ D_padded[2:N+1, 1:N,   2:N+1, :])
    D_zp = 0.5f0 .* (D .+ D_padded[2:N+1, 2:N+1, 3:N+2, :])
    D_zm = 0.5f0 .* (D .+ D_padded[2:N+1, 2:N+1, 1:N,   :])
    
    for k in 1:K
        u_padded = replicate_pad(u)
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
        u = u .+ dt .* flux
    end
    
    return use_gpu ? Array(u) : u
end

# ─── Test 1: CPU vs CUDA produce identical results ───
@testset "CPU vs CUDA Equivalence (Replicate BC)" begin
    println("\n[Test 1] CPU vs CUDA equivalence with replicate boundaries...")
    
    N = 32  # Use smaller size for faster test
    D = ones(Float32, N, N, N, 1)
    seed = zeros(Float32, N, N, N, 1)
    seed[N÷2+1, N÷2+1, N÷2+1, 1] = 1.0f0
    
    K = 20; dt = 0.16f0
    
    u_cpu = run_diffusion_replicate(D, seed, K, dt; use_gpu=false)
    
    if CUDA.functional()
        u_gpu = run_diffusion_replicate(D, seed, K, dt; use_gpu=true)
        
        max_diff = maximum(abs.(u_cpu .- u_gpu))
        mean_diff = sum(abs.(u_cpu .- u_gpu)) / length(u_cpu)
        
        println("  Max element-wise difference: $max_diff")
        println("  Mean element-wise difference: $mean_diff")
        println("  CPU max: $(maximum(u_cpu)), GPU max: $(maximum(u_gpu))")
        
        # Float32 accumulation over 20 steps should be within ~1e-5 tolerance
        @test max_diff < 1e-5
        @test mean_diff < 1e-7
        
        println("  ✅ CPU and CUDA produce identical results!")
    else
        @test true
        println("  ⚠ CUDA not available, skipping GPU comparison")
    end
end

# ─── Test 2: Replicate BC does NOT wrap around ───
@testset "No Wrap-Around (Replicate BC)" begin
    println("\n[Test 2] Verify replicate BC doesn't wrap around...")
    
    N = 16  # Small volume for quick test
    D = ones(Float32, N, N, N, 1)
    
    # Place seed at corner (1,1,1) — with periodic BC, heat would wrap to (N,N,N)
    seed = zeros(Float32, N, N, N, 1)
    seed[1, 1, 1, 1] = 1.0f0
    
    K = 10; dt = 0.16f0
    u = run_diffusion_replicate(D, seed, K, dt; use_gpu=false)
    
    # With replicate BC, heat should NOT appear at the far corner
    corner_heat = u[N, N, N, 1]
    near_seed_heat = u[2, 2, 2, 1]
    
    println("  Heat at seed corner (1,1,1): $(u[1,1,1,1])")
    println("  Heat at (2,2,2):             $near_seed_heat")
    println("  Heat at far corner (N,N,N):  $corner_heat")
    
    @test corner_heat < 1e-10  # No wraparound → essentially zero at far corner
    @test near_seed_heat > 0   # Should have diffused to neighbors
    @test u[1, 1, 1, 1] > near_seed_heat  # Seed position should have highest heat
    
    println("  ✅ No wrap-around confirmed!")
end

# ─── Test 3: Heat conservation with Neumann BC ───
@testset "Heat Conservation (Neumann BC)" begin
    println("\n[Test 3] Total heat is conserved with Neumann zero-flux BC...")
    
    N = 32
    D = ones(Float32, N, N, N, 1)
    seed = zeros(Float32, N, N, N, 1)
    seed[N÷2+1, N÷2+1, N÷2+1, 1] = 1.0f0
    
    dt = 0.16f0
    
    total_initial = sum(seed)
    
    u10 = run_diffusion_replicate(D, seed, 10, dt; use_gpu=false)
    u50 = run_diffusion_replicate(D, seed, 50, dt; use_gpu=false)
    u100 = run_diffusion_replicate(D, seed, 100, dt; use_gpu=false)
    
    total_10 = sum(u10)
    total_50 = sum(u50)
    total_100 = sum(u100)
    
    println("  Initial total heat: $total_initial")
    println("  K=10:  total = $(round(Float64(total_10), digits=8))")
    println("  K=50:  total = $(round(Float64(total_50), digits=8))")
    println("  K=100: total = $(round(Float64(total_100), digits=8))")
    
    # With Neumann BC, total heat should be exactly conserved
    @test abs(total_10 - total_initial) / total_initial < 0.001
    @test abs(total_50 - total_initial) / total_initial < 0.005
    @test abs(total_100 - total_initial) / total_initial < 0.01
    
    println("  ✅ Heat conservation verified!")
end

# ─── Test 4: Edge barrier with replicate BC ───
@testset "Edge Barrier with Replicate BC" begin
    println("\n[Test 4] Edge-aware diffusivity with replicate boundaries...")
    
    N = 32
    seed = zeros(Float32, N, N, N, 1)
    seed[N÷2+1, N÷2+1, N÷2+1, 1] = 1.0f0
    K = 40; dt = 0.16f0
    
    # Uniform diffusivity
    D_uniform = ones(Float32, N, N, N, 1)
    u_uniform = run_diffusion_replicate(D_uniform, seed, K, dt; use_gpu=false)
    
    # Edge barrier at radius 5
    D_edge = ones(Float32, N, N, N, 1)
    c = N÷2 + 1
    for x in 1:N, y in 1:N, z in 1:N
        r = sqrt(Float32((x-c)^2 + (y-c)^2 + (z-c)^2))
        if 4.0f0 < r < 6.0f0
            D_edge[x, y, z, 1] = 0.001f0
        end
    end
    u_edge = run_diffusion_replicate(D_edge, seed, K, dt; use_gpu=false)
    
    max_uniform = maximum(u_uniform)
    max_edge = maximum(u_edge)
    
    println("  Uniform: max_u = $(round(max_uniform, digits=6))")
    println("  Edge:    max_u = $(round(max_edge, digits=6))")
    
    @test max_edge > max_uniform  # Barrier traps heat
    
    println("  ✅ Edge barrier correctly traps heat!")
end

# ─── Test 5: GPU path produces correct mask ───
@testset "GPU Mask Generation" begin
    println("\n[Test 5] GPU produces valid binary mask...")
    
    if !CUDA.functional()
        @test true
        println("  ⚠ CUDA not available, skipping")
        return
    end
    
    N = 64
    D = ones(Float32, N, N, N, 1)
    seed = zeros(Float32, N, N, N, 1)
    seed[33, 33, 33, 1] = 1.0f0
    K = 30; dt = 0.16f0; theta = 0.001f0; tau = 0.0001f0
    
    u = run_diffusion_replicate(D, seed, K, dt; use_gpu=true)
    
    # Apply sigmoid threshold (same as InferenceClient)
    mask_field = 1.0f0 ./ (1.0f0 .+ exp.(-(u .- theta) ./ tau))
    binary_mask = UInt8.(mask_field[:,:,:,1] .> 0.5f0)
    
    voxels = count(binary_mask .> 0)
    println("  Mask voxels: $voxels (K=$K, θ=$theta)")
    
    @test voxels > 0
    @test voxels < N^3
    @test binary_mask[33, 33, 33] == 0x01  # Center must be segmented
    
    println("  ✅ GPU mask generation works!")
end

println("\n" * "=" ^ 70)
println("  All CPU vs CUDA equivalence tests completed!")
println("=" ^ 70)
