#!/usr/bin/env julia
"""
    test_heatgdt_directional_normalization.jl

Integration test verifying Directional (6-Sector Cardinal) Edge Normalization in Heat-GDT:
1. Directional Parameter Extraction:
   Verifies that compute_directional_relative_edge_params accurately isolates
   independent peak edge strengths for cranial (+Z, lung E≈0.85) vs caudal (-Z, liver soft tissue E≈0.16).
2. Elimination of Strong-Edge Shadowing:
   Contrasts isotropic normalization (which leaks across -Z due to +Z lung edge shadowing)
   with directional normalization (which stops cleanly at both +Z and -Z with 0 leakage).
3. Parity across CPU PDE and Vulkan GPU Compute Shader in both 1-Click and Wand modes.
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
println("  Test: Heat-GDT Directional Edge Normalization (Diaphragm / Asymmetric)")
println("=" ^ 75)

@testset "Directional Edge Normalization & Asymmetric Confinement" begin
    # ── 1. Create synthetic 64³ volume modeling a Diaphragmatic / Subdiaphragmatic Mass ──
    # Center = (32, 32, 32)
    # Core (r <= 10): Tumor mass with background edge E = 0.05 (D ≈ 0.6035)
    # Vicinity Boundary Shell (10 < r <= 13):
    #   - Superior (+Z, Cranial): lung pleura barrier E = 0.85 (D = 1e-5)
    #   - Inferior (-Z, Caudal): normal liver parenchyma boundary E = 0.16 (D ≈ 0.1987)
    #   - Lateral (±X, ±Y): retroperitoneal soft-tissue margin E = 0.25 (D ≈ 0.0799)
    # Exterior (r > 13):
    #   - Inferior (z < cz): open liver soft tissue parenchyma E = 0.05 (D ≈ 0.6035)
    #     (If the weak -Z boundary is breached, heat will leak extensively here!)
    #   - Superior (z >= cz): lung cavity E = 0.85 (D = 1e-5)
    
    N = 64
    cx, cy, cz = 32, 32, 32
    diff_synthetic = fill(Float32(exp(-10.1 * 0.05)), N, N, N)
    
    for z in 1:N, y in 1:N, x in 1:N
        r = sqrt((x - cx)^2 + (y - cy)^2 + (z - cz)^2)
        dx = x - cx
        dy = y - cy
        dz = z - cz
        ax = abs(dx)
        ay = abs(dy)
        az = abs(dz)
        
        if 10.0 < r <= 12.0
            # Boundary shell with directionally asymmetric edge strengths
            if az >= ax && az >= ay
                if dz >= 0
                    # Cranial (+Z): Strong Lung Barrier (E = 0.85)
                    diff_synthetic[x, y, z] = 1.0f-5
                else
                    # Caudal (-Z): Weak Liver Boundary (E = 0.16)
                    diff_synthetic[x, y, z] = Float32(exp(-10.1 * 0.16))
                end
            else
                # Lateral (±X, ±Y): Moderate boundary (E = 0.25)
                diff_synthetic[x, y, z] = Float32(exp(-10.1 * 0.25))
            end
        elseif r > 12.0
            if az >= ax && az >= ay && dz >= 0
                # Superior exterior: lung air / pleura
                diff_synthetic[x, y, z] = 1.0f-5
            else
                # Inferior and lateral exterior: soft tissue parenchyma
                diff_synthetic[x, y, z] = Float32(exp(-10.1 * 0.05))
            end
        end
    end

    # ── 2. Test Directional Parameter Extraction ──
    spacing = (1.0, 1.0, 1.0)
    R_phys = 16.0
    
    dir_params = VulkanHeatDiffusion.compute_directional_relative_edge_params(
        diff_synthetic, cx, cy, cz, R_phys, spacing; sensitivity=1.0f0
    )
    iso_params = VulkanHeatDiffusion.compute_local_relative_edge_params(
        diff_synthetic, cx, cy, cz, R_phys, spacing; sensitivity=1.0f0
    )
    
    p_xp, p_xm, p_yp, p_ym, p_zp, p_zm, E_bg, edge_clip, gamma_eff, is_dir = dir_params[1:10]
    println("  Directional Params:")
    println("    Peaks: +X=$p_xp, -X=$p_xm, +Y=$p_yp, -Y=$p_ym, +Z=$p_zp, -Z=$p_zm")
    println("    E_bg=$E_bg, edge_clip=$edge_clip, gamma=$gamma_eff, is_directional=$is_dir")
    println("  Isotropic Params:")
    println("    E_bg=$(iso_params[1]), E_scale=$(iso_params[2]), gamma=$(iso_params[3]), clip=$(iso_params[4])")
    
    # Baseline interior edge should reflect lesion interior (~0.05)
    @test isapprox(E_bg, 0.05f0, atol=0.015)
    # Cranial (+Z) peak should capture strong lung barrier (E >= 0.80)
    @test p_zp >= 0.80f0
    # Caudal (-Z) peak should capture weak liver boundary (E in [0.15, 0.17])
    @test isapprox(p_zm, 0.16f0, atol=0.02)
    # Lateral peaks should capture moderate margin (E in [0.23, 0.27])
    @test isapprox(p_xp, 0.25f0, atol=0.03)
    @test isapprox(p_xm, 0.25f0, atol=0.03)
    @test isapprox(p_yp, 0.25f0, atol=0.03)
    @test isapprox(p_ym, 0.25f0, atol=0.03)
    @test is_dir == 1.0f0
    
    # Isotropic E_scale is dominated by +Z peak (0.85), so E_scale ≈ 1 / (0.85 - 0.05) ≈ 1.25
    @test iso_params[2] < 2.0f0
    println("  ✓ Directional parameter extraction verified: +Z (lung) and -Z (liver) peaks isolated independently")

    # ── 3. Test CPU PDE: Isotropic Shadowing Leakage vs Directional Confinement ──
    ct_vol = zeros(Float32, N, N, N)
    pet_vol = zeros(Float32, N, N, N)
    pts_vol = zeros(Float32, N, N, N)
    pts_vol[cx, cy, cz] = 1.0f0
    
    InferenceClient.precompute_heatgdt_diffusivity(diff_synthetic; tp_index=1)
    
    # 3A: CPU Diffusion with Isotropic Normalization (demonstrates the shadowing flaw)
    mask_iso = InferenceClient.run_heatgdt(
        ct_vol, pet_vol, pts_vol, cx, cy, cz;
        K=150, dt=0.16f0, theta=0.001f0, edge_params=iso_params
    )
    @test mask_iso !== nothing
    
    leaked_iso_caudal = 0
    for z in 1:N, y in 1:N, x in 1:N
        if mask_iso[x, y, z] > 0
            dz = z - 33
            r = sqrt((x - 33)^2 + (y - 33)^2 + dz^2)
            if dz < 0 && r > 12.0
                leaked_iso_caudal += 1
            end
        end
    end
    println("  Isotropic CPU Diffusion: Caudal (-Z) Leaked Voxels past barrier = $leaked_iso_caudal")
    @test leaked_iso_caudal > 10
    println("  ✓ Confirmed: Isotropic normalization leaks across -Z liver margin due to +Z lung shadowing")

    # 3B: CPU Diffusion with Directional Normalization (resolves the shadowing flaw)
    mask_dir = InferenceClient.run_heatgdt(
        ct_vol, pet_vol, pts_vol, cx, cy, cz;
        K=150, dt=0.16f0, theta=0.001f0, edge_params=dir_params
    )
    @test mask_dir !== nothing
    
    leaked_dir_caudal = 0
    leaked_dir_cranial = 0
    inside_dir_core = 0
    for z in 1:N, y in 1:N, x in 1:N
        if mask_dir[x, y, z] > 0
            dz = z - 33
            r = sqrt((x - 33)^2 + (y - 33)^2 + dz^2)
            if dz < 0 && r > 12.0
                leaked_dir_caudal += 1
            elseif dz > 0 && r > 12.0
                leaked_dir_cranial += 1
            elseif r <= 10.0
                inside_dir_core += 1
            end
        end
    end
    println("  Directional CPU Diffusion: core=$inside_dir_core, leaked caudal (-Z)=$leaked_dir_caudal, leaked cranial (+Z)=$leaked_dir_cranial")
    @test inside_dir_core > 200
    @test leaked_dir_caudal == 0
    @test leaked_dir_cranial == 0
    println("  ✓ Directional CPU Diffusion: 0 caudal leakage and 0 cranial leakage! Mass perfectly confined")

    # ── 4. Test Live GPU Vulkan Path: 1-Click and Wand Modes with Directional Normalization ──
    GLFW.Init()
    GLFW.WindowHint(GLFW.CLIENT_API, GLFW.NO_API)
    GLFW.WindowHint(GLFW.VISIBLE, 0)
    win = GLFW.CreateWindow(256, 256, "Directional Normalization Test")
    vk_ctx = init_vulkan_context(win, 256, 256)
    
    heat_state = HeatDiffusionState()
    init_heat_diffusion!(heat_state, vk_ctx, N, N, N)
    upload_diffusivity!(heat_state, vk_ctx, diff_synthetic)
    
    # 4A: GPU 1-Click Mode with Directional Push Constants
    run_heat_diffusion!(
        heat_state, vk_ctx, cx, cy, cz;
        K=150, dt=0.16f0, theta=0.001f0, spacing=spacing, edge_params=dir_params
    )
    gpu_mask = read_mask_to_cpu(heat_state, vk_ctx)
    
    gpu_leaked_caudal = 0
    gpu_leaked_cranial = 0
    gpu_inside_core = 0
    for z in 1:N, y in 1:N, x in 1:N
        if gpu_mask[x, y, z] > 0.5f0
            dz = z - cz
            r = sqrt((x - cx)^2 + (y - cy)^2 + dz^2)
            if dz < 0 && r > 12.0
                gpu_leaked_caudal += 1
            elseif dz > 0 && r > 12.0
                gpu_leaked_cranial += 1
            elseif r <= 10.0
                gpu_inside_core += 1
            end
        end
    end
    println("  GPU 1-Click: core=$gpu_inside_core, leaked caudal (-Z)=$gpu_leaked_caudal, leaked cranial (+Z)=$gpu_leaked_cranial")
    @test gpu_inside_core > 200
    @test gpu_leaked_caudal == 0
    @test gpu_leaked_cranial == 0
    println("  ✓ GPU 1-Click Mode: 0 caudal and 0 cranial leakage with directional push constants")

    # 4B: GPU Wand Mode (Incremental Hold & Drag) with Directional Push Constants
    seed_only!(heat_state, vk_ctx, cx, cy, cz; seed_radius=1, spacing=spacing, edge_params=dir_params)
    run_incremental_steps!(heat_state, vk_ctx, 75; theta=0.001f0, copy_box_to_cpu=true, spacing=spacing, edge_params=dir_params)
    run_incremental_steps!(heat_state, vk_ctx, 75; theta=0.001f0, copy_box_to_cpu=true, spacing=spacing, edge_params=dir_params)
    
    wand_mask = read_mask_to_cpu(heat_state, vk_ctx)
    wand_leaked_caudal = 0
    wand_leaked_cranial = 0
    wand_inside_core = 0
    for z in 1:N, y in 1:N, x in 1:N
        if wand_mask[x, y, z] > 0.5f0
            dz = z - cz
            r = sqrt((x - cx)^2 + (y - cy)^2 + dz^2)
            if dz < 0 && r > 12.0
                wand_leaked_caudal += 1
            elseif dz > 0 && r > 12.0
                wand_leaked_cranial += 1
            elseif r <= 10.0
                wand_inside_core += 1
            end
        end
    end
    println("  GPU Wand Mode: core=$wand_inside_core, leaked caudal (-Z)=$wand_leaked_caudal, leaked cranial (+Z)=$wand_leaked_cranial")
    @test wand_inside_core > 200
    @test wand_leaked_caudal == 0
    @test wand_leaked_cranial == 0
    println("  ✓ GPU Wand Mode: 0 caudal and 0 cranial leakage across incremental ticks")

    destroy_heat_diffusion!(heat_state, vk_ctx)
    destroy_vulkan_context!(vk_ctx)
end

println("\n" * "=" ^ 75)
println("  ALL DIRECTIONAL EDGE NORMALIZATION TESTS PASSED ✓")
println("=" ^ 75)
ccall(:_exit, Cvoid, (Cint,), 0)
