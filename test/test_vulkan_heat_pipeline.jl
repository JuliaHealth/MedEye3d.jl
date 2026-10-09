#!/usr/bin/env julia
"""
End-to-end test for VulkanHeatDiffusion compute pipeline on actual Vulkan GPU.
Verifies:
1. Compute pipeline creation & resource allocation (images, views, descriptors)
2. Diffusivity texture upload (3D R32_SFLOAT)
3. GPU seed impulse initialization (seed_only!)
4. Incremental diffusion stepping (run_incremental_steps!)
5. GPU soft-threshold mask extraction
6. Readback to CPU (read_mask_to_cpu)
7. Diffusion growth verification: K=40 produces >= voxels than K=20
8. Safe resource cleanup (destroy_heat_diffusion!)
"""

using Test
using MedEye3d
using MedEye3d.VulkanBackend
using MedEye3d.VulkanBackend.VulkanContext
using MedEye3d.VulkanBackend.VulkanHeatDiffusion
using GLFW

println("=" ^ 70)
println("  Vulkan Heat Diffusion GPU Pipeline Test")
println("=" ^ 70)

GLFW.Init()
GLFW.WindowHint(GLFW.CLIENT_API, GLFW.NO_API)
GLFW.WindowHint(GLFW.VISIBLE, 0)
window = GLFW.CreateWindow(256, 256, "Vulkan Heat Pipeline Test")
ctx = init_vulkan_context(window, 256, 256)

@testset "Vulkan Heat Diffusion GPU Pipeline" begin
    w, h, d = 64, 64, 64
    println("\n[1/5] Initializing HeatDiffusionState ($w×$h×$d)...")
    state = HeatDiffusionState()
    init_heat_diffusion!(state, ctx, w, h, d)
    
    @test state.is_initialized == true
    @test state.width == w
    @test state.height == h
    @test state.depth == d
    println("  ✓ Pipelines, descriptors, and 3D storage images allocated successfully")
    
    # ── Upload diffusivity ──
    println("\n[2/5] Uploading 3D diffusivity field...")
    D = ones(Float32, w, h, d)
    # Add an edge barrier at x=20
    D[20:22, :, :] .= 0.01f0
    upload_diffusivity!(state, ctx, D)
    @test state.diffusivity_uploaded == true
    println("  ✓ Diffusivity field uploaded to GPU")
    
    # ── Seed point ──
    println("\n[3/5] Seeding heat at center (32, 32, 32)...")
    seed_only!(state, ctx, 32, 32, 32)
    @test state.accumulated_steps == 0
    @test state.current_buffer_is_ping == true
    println("  ✓ Heat field seeded successfully")
    
    # ── Incremental diffusion step 1 (K=20) ──
    println("\n[4/5] Running K=20 diffusion steps...")
    t1 = time_ns()
    run_incremental_steps!(state, ctx, 20; theta=0.0001f0)
    dt1_ms = (time_ns() - t1) / 1e6
    @test state.accumulated_steps == 20
    println("  ✓ 20 steps + extract mask completed in $(round(dt1_ms, digits=2))ms")
    
    mask1 = read_mask_to_cpu(state, ctx)
    vox1 = count(mask1 .> 0.5f0)
    println("  ✓ Mask voxels at K=20: $vox1")
    @test vox1 > 0
    @test mask1[32, 32, 32] > 0.5f0
    
    # ── Incremental diffusion step 2 (K=40) ──
    println("\n[5/5] Running additional 20 diffusion steps (total K=40)...")
    t2 = time_ns()
    run_incremental_steps!(state, ctx, 20; theta=0.0001f0)
    dt2_ms = (time_ns() - t2) / 1e6
    @test state.accumulated_steps == 40
    println("  ✓ 20 incremental steps completed in $(round(dt2_ms, digits=2))ms")
    
    mask2 = read_mask_to_cpu(state, ctx)
    vox2 = count(mask2 .> 0.5f0)
    println("  ✓ Mask voxels at K=40: $vox2")
    @test vox2 >= vox1
    
    # Clean up
    println("\nCleaning up GPU resources...")
    destroy_heat_diffusion!(state, ctx)
    @test state.is_initialized == false
    println("  ✓ Resources destroyed cleanly")
end

GLFW.DestroyWindow(window)
GLFW.Terminate()

println("=" ^ 70)
println("  Vulkan Heat Diffusion GPU Pipeline: ALL TESTS PASSED ✓")
println("=" ^ 70)
