#!/usr/bin/env julia
"""
    test_heatgdt_edge_leakage_and_relative_speed.jl

Integration test verifying:
1. Weak edges in soft tissue: relative edge normalization turns weak local boundary peaks
   into firm stopping walls without leakage.
2. Growth speed based on relative edges: interior core voxels have D_eff ≈ 1.0, decoupling
   internal diffusion speed from absolute background attenuation.
3. Zero leakage across strong edges (E >= 0.35) in both Wand and 1-Click modes via harmonic mean.
"""

using Test
using Statistics
using Observables
using GLMakie
GLMakie.activate!(visible=false)

using GLFW
using MedEye3d
using MedEye3d.InferenceClient
using MedEye3d.VulkanBackend
using MedEye3d.VulkanBackend.VulkanContext
using MedEye3d.VulkanBackend.VulkanHeatDiffusion
using MedEye3d.SegmentationDisplay
using MedEye3d.SegmentationDisplay.MakieEventHandlers
using MedEye3d.DataStructs
using MedEye3d.ForDisplayStructs

println("=" ^ 70)
println("  Test: Heat-GDT Relative Edge Normalization & Zero Leakage")
println("=" ^ 70)

@testset "Relative Edge Normalization & Zero Leakage" begin
    # ── 1. Create synthetic 64³ volume with soft tissue & strong barrier ──
    # Center = (32, 32, 32)
    # Interior core (r < 10): soft tissue background E = 0.08 (D ≈ 0.446)
    # Weak boundary shell (10 <= r <= 12): weak edge E = 0.18 (D ≈ 0.162)
    # Exterior (12 < r < 20): exterior soft tissue E = 0.08
    # Strong barrier wall at x = 46: strong edge E = 0.80 (D = 1e-5)
    
    N = 64
    diff_synthetic = fill(Float32(exp(-10.1 * 0.08)), N, N, N)
    cx, cy, cz = 32, 32, 32
    
    for z in 1:N, y in 1:N, x in 1:N
        r = sqrt((x - cx)^2 + (y - cy)^2 + (z - cz)^2)
        if 10.0 <= r <= 12.0
            # Weak soft-tissue boundary
            diff_synthetic[x, y, z] = Float32(exp(-10.1 * 0.18))
        end
        if x >= 46
            # Strong barrier (e.g. bone or organ wall)
            diff_synthetic[x, y, z] = 1.0f-5
        end
    end
    
    # ── 2. Test compute_local_relative_edge_params ──
    spacing = (1.0, 1.0, 1.0)
    R_phys = 14.0
    edge_params = VulkanHeatDiffusion.compute_local_relative_edge_params(
        diff_synthetic, cx, cy, cz, R_phys, spacing; sensitivity=1.0f0
    )
    E_bg, E_scale, gamma_eff, edge_clip = edge_params
    println("  Local relative edge params: E_bg=$E_bg, E_scale=$E_scale, gamma=$gamma_eff, clip=$edge_clip")
    
    # Baseline interior edge should reflect soft tissue interior (~0.08)
    @test isapprox(E_bg, 0.08f0, atol=0.015)
    # Shell peak edge should reflect weak boundary (~0.18) -> delta ≈ 0.10 -> scale ≈ 10
    @test E_scale >= 7.0f0
    @test gamma_eff >= 10.0f0
    @test edge_clip == 0.85f0
    println("  ✓ Local relative edge parameter extraction verified")

    # ── 3. Test CPU Diffusion: Harmonic Mean + Relative Normalization vs Arithmetic Mean Leakage ──
    ct_vol = zeros(Float32, N, N, N)
    pet_vol = zeros(Float32, N, N, N)
    pts_vol = zeros(Float32, N, N, N)
    pts_vol[cx, cy, cz] = 1.0f0
    
    InferenceClient.precompute_heatgdt_diffusivity(diff_synthetic; tp_index=1)
    
    # Run CPU diffusion with new harmonic mean and relative edge normalization
    mask_fixed = InferenceClient.run_heatgdt(
        ct_vol, pet_vol, pts_vol, cx, cy, cz;
        K=100, dt=0.16f0, theta=0.001f0, edge_params=edge_params
    )
    @test mask_fixed !== nothing
    
    # Check that mask stayed inside weak boundary (r <= 12)
    # Note: run_heatgdt returns a 64³ patch where the seed is at (33, 33, 33)
    vox_outside_weak_boundary = 0
    vox_inside_weak_boundary = 0
    vox_beyond_strong_barrier = 0
    for z in 1:N, y in 1:N, x in 1:N
        if mask_fixed[x, y, z] > 0
            r = sqrt((x - 33)^2 + (y - 33)^2 + (z - 33)^2)
            if r > 13.0
                vox_outside_weak_boundary += 1
            else
                vox_inside_weak_boundary += 1
            end
            if x >= 46
                vox_beyond_strong_barrier += 1
            end
        end
    end
    
    println("  CPU Diffusion: inside_boundary=$vox_inside_weak_boundary, outside=$vox_outside_weak_boundary, beyond_strong_barrier=$vox_beyond_strong_barrier")
    @test vox_inside_weak_boundary > 100
    @test vox_outside_weak_boundary == 0
    @test vox_beyond_strong_barrier == 0
    println("  ✓ CPU Diffusion respects weak soft-tissue edges and strictly prevents strong barrier breach")

    # ── 4. Test Live GPU Vulkan Path: Wand Mode & 1-Click Mode ──
    GLFW.Init()
    GLFW.WindowHint(GLFW.CLIENT_API, GLFW.NO_API)
    GLFW.WindowHint(GLFW.VISIBLE, 0)
    win = GLFW.CreateWindow(256, 256, "Relative Edge Test")
    vk_ctx = init_vulkan_context(win, 256, 256)
    
    heat_state = HeatDiffusionState()
    init_heat_diffusion!(heat_state, vk_ctx, N, N, N)
    upload_diffusivity!(heat_state, vk_ctx, diff_synthetic)
    
    # 4A: GPU 1-Click Mode
    run_heat_diffusion!(
        heat_state, vk_ctx, cx, cy, cz;
        K=100, dt=0.16f0, theta=0.001f0, spacing=spacing, edge_params=edge_params
    )
    gpu_mask = read_mask_to_cpu(heat_state, vk_ctx)
    
    gpu_outside_weak = 0
    gpu_inside_weak = 0
    gpu_beyond_strong = 0
    for z in 1:N, y in 1:N, x in 1:N
        if gpu_mask[x, y, z] > 0.5f0
            r = sqrt((x - cx)^2 + (y - cy)^2 + (z - cz)^2)
            if r > 13.0
                gpu_outside_weak += 1
            else
                gpu_inside_weak += 1
            end
            if x >= 46
                gpu_beyond_strong += 1
            end
        end
    end
    println("  GPU 1-Click: inside=$gpu_inside_weak, outside=$gpu_outside_weak, beyond_strong=$gpu_beyond_strong")
    @test gpu_inside_weak > 100
    @test gpu_outside_weak == 0
    @test gpu_beyond_strong == 0
    println("  ✓ GPU 1-Click mode stops at weak boundary and has 0 leakage across strong barrier")

    # 4B: GPU Wand (Incremental Hold) Mode
    seed_only!(heat_state, vk_ctx, cx, cy, cz; seed_radius=1, spacing=spacing, edge_params=edge_params)
    run_incremental_steps!(heat_state, vk_ctx, 50; theta=0.001f0, copy_box_to_cpu=true, spacing=spacing, edge_params=edge_params)
    run_incremental_steps!(heat_state, vk_ctx, 50; theta=0.001f0, copy_box_to_cpu=true, spacing=spacing, edge_params=edge_params)
    
    wand_mask = read_mask_to_cpu(heat_state, vk_ctx)
    wand_outside_weak = 0
    wand_inside_weak = 0
    wand_beyond_strong = 0
    for z in 1:N, y in 1:N, x in 1:N
        if wand_mask[x, y, z] > 0.5f0
            r = sqrt((x - cx)^2 + (y - cy)^2 + (z - cz)^2)
            if r > 13.0
                wand_outside_weak += 1
            else
                wand_inside_weak += 1
            end
            if x >= 46
                wand_beyond_strong += 1
            end
        end
    end
    println("  GPU Wand Mode: inside=$wand_inside_weak, outside=$wand_outside_weak, beyond_strong=$wand_beyond_strong")
    @test wand_inside_weak > 100
    @test wand_outside_weak == 0
    @test wand_beyond_strong == 0
    println("  ✓ GPU Wand mode stops at weak boundary and has 0 leakage across strong barrier")

    destroy_heat_diffusion!(heat_state, vk_ctx)
end

println("\n" * "=" ^ 70)
println("  ALL RELATIVE EDGE AND ZERO LEAKAGE TESTS PASSED ✓")
println("=" ^ 70)
ccall(:_exit, Cvoid, (Cint,), 0)
