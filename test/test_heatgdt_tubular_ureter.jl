#!/usr/bin/env julia
"""
    test_heatgdt_tubular_ureter.jl

Integration test verifying Tubular & Elongated Structure Heat Diffusion in Heat-GDT:
1. Automatic Wall Sensing & Tubular Detection:
   Verifies that compute_directional_relative_edge_params accurately detects when
   a seed is placed inside a narrow channel (r_wall <= 3.5 mm) like a ureter,
   automatically sets is_tubular=1.0, adapts R_core, anchors E_bg, and sets dynamic E_barrier.
2. Prevention of Diffusion Stalling in Narrow Channels:
   Verifies that diffusion does NOT stall at the seed, but instead flows freely along
   the longitudinal lumen axis of the ureter.
3. Strict Lateral Confinement:
   Verifies exactly 0 leaked voxels through the circumferential ureter walls into
   surrounding retroperitoneal fat or psoas muscle.
4. Parity across CPU PDE and Vulkan GPU Compute Shader in both 1-Click and Wand modes.
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

println("=" ^ 75)
println("  Test: Heat-GDT Tubular Structure & Ureter Channel Propagation")
println("=" ^ 75)

@testset "Tubular & Elongated Structure Propagation" begin
    # ── 1. Create synthetic 64³ volume modeling a narrow Ureter (diameter 3.5 mm) ──
    # Grid: 64³, spacing = (1.0, 1.0, 1.0) mm
    # Tube runs along Z axis at (cx, cy) = (32, 32) from z = 14 to z = 50 (length = 36 mm)
    # Lumen (r_xy <= 1.8 mm): E = 0.22 (D ≈ 0.108) (notice non-zero gradient due to narrow scale)
    # Circumferential Wall (1.8 < r_xy <= 3.0 mm): E = 0.55 (D ≈ 0.0038) (strong wall)
    # Surrounding Retroperitoneal Fat / Muscle (r_xy > 3.0 mm): E = 0.45 (D ≈ 0.010)
    # Tube Ends (z < 14 or z > 50): capped by solid anatomical barrier E = 0.60
    
    N = 64
    cx, cy, cz = 32, 32, 32
    diff_synthetic = fill(Float32(exp(-10.1 * 0.45)), N, N, N)
    
    for z in 1:N, y in 1:N, x in 1:N
        r_xy = sqrt((x - cx)^2 + (y - cy)^2)
        if 14 <= z <= 50
            if r_xy <= 1.8
                # Ureter lumen
                diff_synthetic[x, y, z] = Float32(exp(-10.1 * 0.22))
            elseif r_xy <= 3.0
                # Ureter wall
                diff_synthetic[x, y, z] = Float32(exp(-10.1 * 0.55))
            else
                # Surrounding retroperitoneal tissue
                diff_synthetic[x, y, z] = Float32(exp(-10.1 * 0.45))
            end
        else
            # Outside tube longitudinal range (solid boundary)
            diff_synthetic[x, y, z] = Float32(exp(-10.1 * 0.60))
        end
    end

    # ── 2. Test Tubular Detection & Parameter Extraction ──
    spacing = (1.0, 1.0, 1.0)
    R_phys = 14.0
    
    dir_params = VulkanHeatDiffusion.compute_directional_relative_edge_params(
        diff_synthetic, cx, cy, cz, R_phys, spacing; sensitivity=1.0f0
    )
    
    p_xp, p_xm, p_yp, p_ym, p_zp, p_zm, E_bg, edge_clip, gamma_eff, is_dir, E_barrier, is_tubular = dir_params
    println("  Tubular Analysis Results:")
    println("    Peaks: +X=$p_xp, -X=$p_xm, +Y=$p_yp, -Y=$p_ym, +Z=$p_zp, -Z=$p_zm")
    println("    E_bg=$E_bg, E_barrier=$E_barrier, is_tubular=$is_tubular, is_dir=$is_dir")
    
    # 1. Automatic tubular detection:
    @test is_tubular == 1.0f0
    @test is_dir == 1.0f0
    
    # 2. Dynamic barrier ceiling must be above lumen edge (E_seed ≈ 0.22) so seed doesn't stall:
    @test E_barrier >= 0.37f0
    
    # 3. Baseline E_bg anchored to lumen center:
    @test isapprox(E_bg, 0.22f0, atol=0.03)
    
    # 4. Transverse walls (±X, ±Y) must be recognized as strong barriers:
    @test p_xp >= 0.40f0
    @test p_xm >= 0.40f0
    @test p_yp >= 0.40f0
    @test p_ym >= 0.40f0
    
    # 5. Longitudinal peaks (±Z) along the open lumen must remain low:
    @test p_zp <= 0.30f0
    @test p_zm <= 0.30f0
    println("  ✓ Automatic tubular detection verified: transverse walls detected, lumen corridor kept open")

    # ── 3. Test CPU PDE: Tubular Flow vs Stalling ──
    ct_vol = zeros(Float32, N, N, N)
    pet_vol = zeros(Float32, N, N, N)
    pts_vol = zeros(Float32, N, N, N)
    pts_vol[cx, cy, cz] = 1.0f0
    
    InferenceClient.precompute_heatgdt_diffusivity(diff_synthetic; tp_index=1)
    
    # Run CPU diffusion with tubular-aware parameters
    mask_cpu = InferenceClient.run_heatgdt(
        ct_vol, pet_vol, pts_vol, cx, cy, cz;
        K=80, dt=0.16f0, theta=0.001f0, edge_params=dir_params
    )
    @test mask_cpu !== nothing
    
    # Analyze mask extension along Z (longitudinal) and X-Y (transverse):
    z_min_found = N
    z_max_found = 1
    leaked_lateral_voxels = 0
    lumen_voxels = 0
    
    for z in 1:N, y in 1:N, x in 1:N
        if mask_cpu[x, y, z] > 0
            r_xy = sqrt((x - 33)^2 + (y - 33)^2)
            if r_xy > 3.0
                leaked_lateral_voxels += 1
            else
                lumen_voxels += 1
                z_min_found = min(z_min_found, z)
                z_max_found = max(z_max_found, z)
            end
        end
    end
    
    longitudinal_span = z_max_found - z_min_found + 1
    println("  CPU Tubular Diffusion: lumen voxels=$lumen_voxels, longitudinal span=$longitudinal_span voxels (z=$z_min_found to $z_max_found), lateral leakage=$leaked_lateral_voxels")
    
    # Must propagate along the tube without stalling:
    @test longitudinal_span >= 15 # Flowed at least 15 mm along the ureter
    @test lumen_voxels >= 50
    # Must have exactly 0 leakage through the circumferential wall:
    @test leaked_lateral_voxels == 0
    println("  ✓ CPU Diffusion flows freely along ureter with strictly ZERO lateral wall leakage")

    # ── 4. Test Live GPU Vulkan Path: 1-Click and Wand Modes ──
    GLFW.Init()
    GLFW.WindowHint(GLFW.CLIENT_API, GLFW.NO_API)
    GLFW.WindowHint(GLFW.VISIBLE, 0)
    win = GLFW.CreateWindow(256, 256, "Tubular Ureter Test")
    vk_ctx = init_vulkan_context(win, 256, 256)
    
    heat_state = HeatDiffusionState()
    init_heat_diffusion!(heat_state, vk_ctx, N, N, N)
    upload_diffusivity!(heat_state, vk_ctx, diff_synthetic)
    
    # 4A: GPU 1-Click Mode on Ureter
    run_heat_diffusion!(
        heat_state, vk_ctx, cx, cy, cz;
        K=80, dt=0.16f0, theta=0.001f0, spacing=spacing, edge_params=dir_params
    )
    gpu_mask = read_mask_to_cpu(heat_state, vk_ctx)
    
    gpu_z_min = N
    gpu_z_max = 1
    gpu_leaked = 0
    gpu_lumen = 0
    for z in 1:N, y in 1:N, x in 1:N
        if gpu_mask[x, y, z] > 0.5f0
            r_xy = sqrt((x - cx)^2 + (y - cy)^2)
            if r_xy > 3.0
                gpu_leaked += 1
            else
                gpu_lumen += 1
                gpu_z_min = min(gpu_z_min, z)
                gpu_z_max = max(gpu_z_max, z)
            end
        end
    end
    gpu_span = gpu_z_max - gpu_z_min + 1
    println("  GPU 1-Click: lumen=$gpu_lumen, span=$gpu_span voxels, lateral leakage=$gpu_leaked")
    @test gpu_span >= 15
    @test gpu_lumen >= 50
    @test gpu_leaked == 0
    println("  ✓ GPU 1-Click Mode: flows along ureter with ZERO lateral leakage")

    # 4B: GPU Wand Mode (Incremental Hold & Drag along Ureter)
    seed_only!(heat_state, vk_ctx, cx, cy, cz; seed_radius=1, spacing=spacing, edge_params=dir_params)
    run_incremental_steps!(heat_state, vk_ctx, 40; theta=0.001f0, copy_box_to_cpu=true, spacing=spacing, edge_params=dir_params)
    run_incremental_steps!(heat_state, vk_ctx, 40; theta=0.001f0, copy_box_to_cpu=true, spacing=spacing, edge_params=dir_params)
    
    wand_mask = read_mask_to_cpu(heat_state, vk_ctx)
    wand_z_min = N
    wand_z_max = 1
    wand_leaked = 0
    wand_lumen = 0
    for z in 1:N, y in 1:N, x in 1:N
        if wand_mask[x, y, z] > 0.5f0
            r_xy = sqrt((x - cx)^2 + (y - cy)^2)
            if r_xy > 3.0
                wand_leaked += 1
            else
                wand_lumen += 1
                wand_z_min = min(wand_z_min, z)
                wand_z_max = max(wand_z_max, z)
            end
        end
    end
    wand_span = wand_z_max - wand_z_min + 1
    println("  GPU Wand Mode: lumen=$wand_lumen, span=$wand_span voxels, lateral leakage=$wand_leaked")
    @test wand_span >= 15
    @test wand_lumen >= 50
    @test wand_leaked == 0
    println("  ✓ GPU Wand Mode: smooth continuous growth along ureter with ZERO lateral leakage")

    destroy_heat_diffusion!(heat_state, vk_ctx)
    destroy_vulkan_context!(vk_ctx)
end

println("\n" * "=" ^ 75)
println("  ALL TUBULAR URETER HEAT DIFFUSION TESTS PASSED ✓")
println("=" ^ 75)
ccall(:_exit, Cvoid, (Cint,), 0)
