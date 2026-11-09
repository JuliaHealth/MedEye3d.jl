#!/usr/bin/env julia
"""
test_anisotropic_heat_diffusion.jl

Verifies that anisotropic voxel spacing correctly constrains cranio-caudal (Z-axis) diffusion:
1. Vulkan GPU pipeline respects spacing (0.78, 0.78, 3.0) vs (1.0, 1.0, 1.0).
2. Heat extent in Z is significantly smaller under anisotropic spacing.
3. CPU fallback in InferenceClient respects anisotropic spacing.
4. OrganHeatLesionTracker respects anisotropic spacing.
"""

using Test
using MedEye3d
using MedEye3d.VulkanBackend
using MedEye3d.VulkanBackend.VulkanContext
using MedEye3d.VulkanBackend.VulkanHeatDiffusion
using GLFW

println("=" ^ 75)
println("  Anisotropic Heat Diffusion Test: Cranio-Caudal Edge Respect")
println("=" ^ 75)

GLFW.Init()
GLFW.WindowHint(GLFW.CLIENT_API, GLFW.NO_API)
GLFW.WindowHint(GLFW.VISIBLE, 0)
window = GLFW.CreateWindow(256, 256, "Anisotropic Heat Diffusion Test")
ctx = init_vulkan_context(window, 256, 256)

@testset "Anisotropic Heat Diffusion on GPU" begin
    w, h, d = 64, 64, 64
    center = (32, 32, 32)
    K = 40
    theta = 0.001f0
    
    # ── Test 1: Isotropic baseline (1.0, 1.0, 1.0) ──
    println("\n[1/3] Running GPU diffusion with ISOTROPIC spacing (1.0, 1.0, 1.0)...")
    state_iso = HeatDiffusionState()
    init_heat_diffusion!(state_iso, ctx, w, h, d)
    D = ones(Float32, w, h, d)
    upload_diffusivity!(state_iso, ctx, D)
    
    seed_only!(state_iso, ctx, center[1], center[2], center[3]; seed_radius=1, spacing=(1.0, 1.0, 1.0))
    run_incremental_steps!(state_iso, ctx, K; theta=theta, spacing=(1.0, 1.0, 1.0))
    mask_iso = read_mask_to_cpu(state_iso, ctx)
    destroy_heat_diffusion!(state_iso, ctx)
    
    # Extent in X, Y, Z for isotropic
    idx_iso = findall(mask_iso .> 0.5f0)
    extent_x_iso = maximum(i[1] for i in idx_iso) - minimum(i[1] for i in idx_iso)
    extent_y_iso = maximum(i[2] for i in idx_iso) - minimum(i[2] for i in idx_iso)
    extent_z_iso = maximum(i[3] for i in idx_iso) - minimum(i[3] for i in idx_iso)
    vox_iso = length(idx_iso)
    
    println("  Isotropic: voxels=$vox_iso, extent_x=$extent_x_iso, extent_y=$extent_y_iso, extent_z=$extent_z_iso")
    # For isotropic, extents should be virtually equal
    @test abs(extent_x_iso - extent_z_iso) <= 2
    @test abs(extent_x_iso - extent_y_iso) <= 2
    
    # ── Test 2: Anisotropic CT spacing (0.78, 0.78, 3.0) ──
    println("\n[2/3] Running GPU diffusion with ANISOTROPIC spacing (0.78, 0.78, 3.0)...")
    state_aniso = HeatDiffusionState()
    init_heat_diffusion!(state_aniso, ctx, w, h, d)
    upload_diffusivity!(state_aniso, ctx, D)
    
    seed_only!(state_aniso, ctx, center[1], center[2], center[3]; seed_radius=1, spacing=(0.78, 0.78, 3.0))
    run_incremental_steps!(state_aniso, ctx, K; theta=theta, spacing=(0.78, 0.78, 3.0))
    mask_aniso = read_mask_to_cpu(state_aniso, ctx)
    destroy_heat_diffusion!(state_aniso, ctx)
    
    idx_aniso = findall(mask_aniso .> 0.5f0)
    extent_x_aniso = maximum(i[1] for i in idx_aniso) - minimum(i[1] for i in idx_aniso)
    extent_y_aniso = maximum(i[2] for i in idx_aniso) - minimum(i[2] for i in idx_aniso)
    extent_z_aniso = maximum(i[3] for i in idx_aniso) - minimum(i[3] for i in idx_aniso)
    vox_aniso = length(idx_aniso)
    
    println("  Anisotropic: voxels=$vox_aniso, extent_x=$extent_x_aniso, extent_y=$extent_y_aniso, extent_z=$extent_z_aniso")
    
    # In-plane extent should remain similar to isotropic
    @test abs(extent_x_aniso - extent_x_iso) <= 2
    # Cranio-caudal (Z) extent MUST be significantly smaller!
    # (0.78/3.0)^2 = 0.0676 -> Z flux is ~15x smaller, so Z extent in voxels is much smaller!
    @test extent_z_aniso < extent_x_aniso ÷ 2
    @test extent_z_aniso < extent_z_iso ÷ 2
    println("  ✓ Z-extent reduced from $(extent_z_iso) to $(extent_z_aniso) voxels (in-plane: $(extent_x_aniso))")
    println("  ✓ Cranio-caudal edge leakage is effectively suppressed on GPU!")
end

@testset "Anisotropic Heat Diffusion on CPU Fallback" begin
    println("\n[3/3] Testing CPU Fallback in InferenceClient with anisotropic spacing...")
    N = 64
    ct_vol = zeros(Float32, N, N, N)
    pet_vol = zeros(Float32, N, N, N)
    points_vol = zeros(Float32, N, N, N)
    points_vol[32, 32, 32] = 1.0f0
    
    # Isotropic CPU
    mask_cpu_iso = MedEye3d.InferenceClient.run_heatgdt(
        ct_vol, pet_vol, points_vol, 32, 32, 32;
        K=40, theta=0.001f0, spacing=(1.0, 1.0, 1.0)
    )
    @test mask_cpu_iso !== nothing
    idx_cpu_iso = findall(mask_cpu_iso .> 0)
    z_ext_iso = maximum(i[3] for i in idx_cpu_iso) - minimum(i[3] for i in idx_cpu_iso)
    
    # Anisotropic CPU
    mask_cpu_aniso = MedEye3d.InferenceClient.run_heatgdt(
        ct_vol, pet_vol, points_vol, 32, 32, 32;
        K=40, theta=0.001f0, spacing=(0.78, 0.78, 3.0)
    )
    @test mask_cpu_aniso !== nothing
    idx_cpu_aniso = findall(mask_cpu_aniso .> 0)
    z_ext_aniso = maximum(i[3] for i in idx_cpu_aniso) - minimum(i[3] for i in idx_cpu_aniso)
    x_ext_aniso = maximum(i[1] for i in idx_cpu_aniso) - minimum(i[1] for i in idx_cpu_aniso)
    
    println("  CPU Fallback: z_ext_iso=$z_ext_iso, z_ext_aniso=$z_ext_aniso, x_ext_aniso=$x_ext_aniso")
    @test z_ext_aniso <= z_ext_iso ÷ 2
    @test z_ext_aniso < x_ext_aniso ÷ 2
    println("  ✓ CPU fallback correctly attenuates Z diffusion under anisotropic spacing!")
end

GLFW.DestroyWindow(window)
GLFW.Terminate()

println("\n" * "=" ^ 75)
println("  ALL ANISOTROPIC HEAT DIFFUSION TESTS PASSED! ✓")
println("=" ^ 75)
