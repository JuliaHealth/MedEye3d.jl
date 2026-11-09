#!/usr/bin/env julia
"""
Test for shape-adaptive local edge contrast enhancement in Heat-GDT.
Verifies:
1. Frontier edge sampling & exponent calculation (`compute_frontier_edge_scale`).
2. Edge sensitivity parameter scaling.
3. Vulkan GPU shader execution with `s_boost`:
   - Weak barrier leakage under standard diffusion (s=1.0).
   - Strong edge confinement under shape-adaptive boost (s=6.0).
"""

using Test
using MedEye3d
using MedEye3d.VulkanBackend
using MedEye3d.VulkanBackend.VulkanContext
using MedEye3d.VulkanBackend.VulkanHeatDiffusion
using GLFW

println("=" ^ 70)
println("  Test: Adaptive Local Edge Heat Kernel")
println("=" ^ 70)

@testset "Frontier Edge Scale Computation" begin
    w, h, d = 64, 64, 64
    spacing = (1.0, 1.0, 1.0)
    
    # 1. Homogeneous low-contrast volume (weak edges, D ≈ 0.90, E ≈ 0.01)
    diff_weak = fill(0.90f0, w, h, d)
    s_weak = compute_frontier_edge_scale(diff_weak, 32, 32, 32, 10.0, spacing; sensitivity=1.0f0)
    println("  Low contrast (weak edge) s_boost: $s_weak (expected > 1.0)")
    @test s_weak > 1.0f0
    @test s_weak <= 10.0f0
    
    # 2. High contrast volume (strong edges, D ≈ 0.05, E ≈ 0.30)
    diff_strong = fill(0.05f0, w, h, d)
    s_strong = compute_frontier_edge_scale(diff_strong, 32, 32, 32, 10.0, spacing; sensitivity=1.0f0)
    println("  High contrast (strong edge) s_boost: $s_strong (expected < 1.0)")
    @test s_strong < 1.0f0
    @test s_strong >= 0.5f0
    
    # 3. Sensitivity variation
    s_weak_sens05 = compute_frontier_edge_scale(diff_weak, 32, 32, 32, 10.0, spacing; sensitivity=0.5f0)
    s_weak_sens15 = compute_frontier_edge_scale(diff_weak, 32, 32, 32, 10.0, spacing; sensitivity=1.5f0)
    println("  Sensitivity test: sens=0.5 -> $s_weak_sens05, sens=1.5 -> $s_weak_sens15")
    @test s_weak_sens15 > s_weak_sens05
end

@testset "Vulkan GPU Diffusion with Adaptive s_boost" begin
    GLFW.Init()
    GLFW.WindowHint(GLFW.CLIENT_API, GLFW.NO_API)
    GLFW.WindowHint(GLFW.VISIBLE, 0)
    window = GLFW.CreateWindow(256, 256, "Vulkan Adaptive Edge Test")
    ctx = init_vulkan_context(window, 256, 256)
    
    w, h, d = 64, 64, 64
    state = HeatDiffusionState()
    init_heat_diffusion!(state, ctx, w, h, d)
    
    # Construct a volume with a weak soft-tissue boundary at x=40
    # Inside (x < 40): D = 1.0 (homogeneous parenchyma)
    # Barrier at x=40..42: D = 0.35 (weak edge, typically leaks)
    # Outside (x > 42): D = 1.0
    D_field = fill(1.0f0, w, h, d)
    D_field[40:42, :, :] .= 0.35f0
    upload_diffusivity!(state, ctx, D_field)
    
    # Seed at (32, 32, 32), 8 voxels away from the barrier at x=40
    seed_x, seed_y, seed_z = 32, 32, 32
    
    # Pass 1: Standard diffusion (s_boost = 1.0f0)
    println("\n  Running Pass 1: Standard diffusion (s_boost = 1.0)...")
    seed_only!(state, ctx, seed_x, seed_y, seed_z; seed_radius=2, s_boost=1.0f0)
    run_incremental_steps!(state, ctx, 60; theta=0.001f0, s_boost=1.0f0)
    mask_std = read_mask_to_cpu(state, ctx)
    vox_std = count(mask_std .> 0.5f0)
    leakage_std = count(mask_std[43:end, :, :] .> 0.5f0)
    println("  Standard: total voxels = $vox_std, leaked past barrier (x > 42) = $leakage_std")
    
    # Pass 2: Boosted adaptive diffusion (s_boost = 6.0f0)
    println("\n  Running Pass 2: Boosted adaptive diffusion (s_boost = 6.0)...")
    seed_only!(state, ctx, seed_x, seed_y, seed_z; seed_radius=2, s_boost=6.0f0)
    run_incremental_steps!(state, ctx, 60; theta=0.001f0, s_boost=6.0f0)
    mask_boost = read_mask_to_cpu(state, ctx)
    vox_boost = count(mask_boost .> 0.5f0)
    leakage_boost = count(mask_boost[43:end, :, :] .> 0.5f0)
    println("  Boosted:  total voxels = $vox_boost, leaked past barrier (x > 42) = $leakage_boost")
    
    # The boosted diffusion should strongly suppress barrier penetration
    @test leakage_boost < leakage_std
    @test vox_boost < vox_std
    println("  ✓ Soft-tissue barrier penetration successfully suppressed by s_boost!")
    
    destroy_heat_diffusion!(state, ctx)
    GLFW.DestroyWindow(window)
end

println("\n" * "=" ^ 70)
println("  Adaptive Local Edge Tests PASSED!")
println("=" ^ 70)
