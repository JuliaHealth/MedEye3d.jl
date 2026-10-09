#!/usr/bin/env julia
"""
End-to-end integration test verifying:
1. Diffusivity loading from HDF5 pat_6_files/preprocessed_volumes.h5
2. Automatic GPU pipeline initialization via ensure_heatgdt_gpu!
3. Heat-GDT GPU seeding without "diffusivity not uploaded" error
4. Interactive incremental diffusion stepping on the full volume
5. Timepoint switching re-uploading diffusivity to GPU
"""

using Test
using HDF5
using GLFW
using MedEye3d
using MedEye3d.InferenceClient
using MedEye3d.VulkanBackend
using MedEye3d.VulkanBackend.VulkanContext
using MedEye3d.VulkanBackend.VulkanHeatDiffusion
using MedEye3d.SegmentationDisplay
using MedEye3d.SegmentationDisplay.MakieEventHandlers

println("=" ^ 70)
println("  MedEye3d Full Heat-GDT Integration Test (pat_6)")
println("=" ^ 70)

h5_path = joinpath(@__DIR__, "..", "data", "pat_6_files", "preprocessed_volumes.h5")
@assert isfile(h5_path) "HDF5 file not found: $h5_path"

# Step 1: Preload diffusivity from HDF5 exactly like AppMain.launch_from_h5
println("\n[1/5] Loading BASELINE/diffusivity from HDF5...")
h5open(h5_path, "r") do h5
    diff_vol = Float32.(read(h5["BASELINE/diffusivity"]))
    is_preflipped = haskey(h5, "_meta_/preflipped") && read(h5["_meta_/preflipped"]) == 1
    if !is_preflipped
        diff_vol = reverse(diff_vol, dims=2)
    end
    InferenceClient.precompute_heatgdt_diffusivity(diff_vol; tp_index=0)
    println("  ✓ Loaded diffusivity: $(size(diff_vol))")
end

@test InferenceClient.is_heatgdt_available() == true

# Step 2: Initialize Vulkan context
println("\n[2/5] Initializing GLFW & Vulkan context...")
GLFW.Init()
GLFW.WindowHint(GLFW.CLIENT_API, GLFW.NO_API)
GLFW.WindowHint(GLFW.VISIBLE, 0)
win = GLFW.CreateWindow(256, 256, "Integration Test Window")
vk_ctx = init_vulkan_context(win, 256, 256)
MakieEventHandlers._vk_heat_ctx[] = vk_ctx
MakieEventHandlers.h5_save_path_ref[] = h5_path

# Step 3: Run ensure_heatgdt_gpu!
println("\n[3/5] Running ensure_heatgdt_gpu!...")
ready = SegmentationDisplay.ensure_heatgdt_gpu!(vk_ctx)
@test ready == true
@test SegmentationDisplay._vk_heat_uploaded[] == true

heat_state = MakieEventHandlers._vk_heat_state[]
@test heat_state !== nothing
@test heat_state.is_initialized == true
@test heat_state.diffusivity_uploaded == true
@test (heat_state.width, heat_state.height, heat_state.depth) == (512, 512, 326)
println("  ✓ GPU pipeline initialized & diffusivity uploaded ($(heat_state.width)×$(heat_state.height)×$(heat_state.depth))")

# Step 4: Test seeding (W key press + click simulation)
println("\n[4/5] Testing GPU heat seeding and incremental diffusion...")
cx, cy, cz = 256, 256, 163
VulkanHeatDiffusion.seed_only!(heat_state, vk_ctx, cx, cy, cz)
@test heat_state.accumulated_steps == 0
println("  ✓ seed_only! succeeded without error")

# Run 10 incremental steps
t0 = time_ns()
VulkanHeatDiffusion.run_incremental_steps!(heat_state, vk_ctx, 10; theta=0.001f0)
dt_ms = (time_ns() - t0) / 1e6
@test heat_state.accumulated_steps == 10
println("  ✓ 10 incremental diffusion steps completed in $(round(dt_ms, digits=2))ms")

mask = VulkanHeatDiffusion.read_mask_to_cpu(heat_state, vk_ctx)
vox_count = count(mask .>= 0.001f0)
println("  ✓ Active lesion voxels: $vox_count")
@test vox_count > 0

# Step 5: Test TP switching re-upload
println("\n[5/5] Testing timepoint switching re-upload (TP=1)...")
MakieEventHandlers.reload_heatgdt_diffusivity_for_tp!(1)
@test heat_state.diffusivity_uploaded == true
println("  ✓ TP=1 diffusivity re-uploaded successfully")

# Cleanup
println("\nCleaning up GPU resources...")
VulkanHeatDiffusion.destroy_heat_diffusion!(heat_state, vk_ctx)
destroy_vulkan_context!(vk_ctx)
GLFW.DestroyWindow(win)
GLFW.Terminate()

println("=" ^ 70)
println("  ALL HEAT-GDT INTEGRATION TESTS PASSED ✓")
println("=" ^ 70)
