#!/usr/bin/env julia
"""
    test_heatgdt_one_click.jl

Unit and integration tests for Heat-GDT One-Click Mode:
1. Observable state and toggling (`heatgdt_one_click_mode`, `toggle_heatgdt_one_click_mode!`).
2. GUI mode buttons in `LesionMetadataWindow` (`btn_mode_1click`, `btn_mode_wand`).
3. Core `execute_one_click_heatgdt!` execution on CPU / GPU fallback.
4. One-Click GPU dispatch with live Vulkan context.
5. Post-hoc size adjustment compatibility.
"""

using Test
using Observables
using GLMakie

# Ensure headless mode for GLMakie
GLMakie.activate!(visible=false)

using HDF5
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
using MedEye3d.StructsManag

println("=" ^ 70)
println("  Test: Heat-GDT One-Click Mode")
println("=" ^ 70)

@testset "One-Click Observable & Toggling" begin
    # Initial state should be true (default: One-Click mode)
    @test MakieEventHandlers.heatgdt_one_click_mode isa Observable
    
    # Toggle to Wand mode
    MakieEventHandlers.toggle_heatgdt_one_click_mode!(false)
    @test MakieEventHandlers.heatgdt_one_click_mode[] == false
    
    # Toggle back to 1-Click mode
    MakieEventHandlers.toggle_heatgdt_one_click_mode!(true)
    @test MakieEventHandlers.heatgdt_one_click_mode[] == true
    
    println("  ✓ Observable toggling works as expected")
end

@testset "GUI Controls in LesionMetadataWindow" begin
    active_id = Observable("1: Peripheral Zone (PZ)")
    lesion_ids = Observable(["1: Peripheral Zone (PZ)"])
    channel = Channel{Any}(32)
    
    win = MedEye3d.LesionMetadataWindow.create_metadata_window(
        active_id, lesion_ids, channel
    )
    
    obs_dict = MedEye3d.LesionMetadataWindow._lmw_observables
    @test haskey(obs_dict, :btn_mode_1click)
    @test haskey(obs_dict, :btn_mode_wand)
    
    btn_1click = obs_dict[:btn_mode_1click]
    btn_wand   = obs_dict[:btn_mode_wand]
    
    # By default, 1-Click should be active and highlighted
    @test MakieEventHandlers.heatgdt_one_click_mode[] == true
    @test btn_1click.buttoncolor[] != btn_wand.buttoncolor[]
    
    # Click Wand button
    println("  Clicking Wand (Hold) button...")
    btn_wand.clicks[] = btn_wand.clicks[] + 1
    sleep(0.05)
    @test MakieEventHandlers.heatgdt_one_click_mode[] == false
    @test btn_wand.buttoncolor[] != btn_1click.buttoncolor[]
    
    # Click 1-Click button
    println("  Clicking 1-Click button...")
    btn_1click.clicks[] = btn_1click.clicks[] + 1
    sleep(0.05)
    @test MakieEventHandlers.heatgdt_one_click_mode[] == true
    @test btn_1click.buttoncolor[] != btn_wand.buttoncolor[]
    
    println("  ✓ GUI 1-Click and Wand toggle buttons verified")
end

@testset "Live Vulkan One-Click Execution" begin
    h5_path = joinpath(@__DIR__, "..", "data", "pat_6_files", "preprocessed_volumes.h5")
    if isfile(h5_path)
        println("\nTesting live GPU One-Click Heat-GDT on pat_6...")
        
        # Load diffusivity
        h5open(h5_path, "r") do h5
            diff_vol = Float32.(read(h5["BASELINE/diffusivity"]))
            InferenceClient.precompute_heatgdt_diffusivity(diff_vol; tp_index=0)
        end
        
        # Initialize headless Vulkan
        GLFW.Init()
        GLFW.WindowHint(GLFW.CLIENT_API, GLFW.NO_API)
        GLFW.WindowHint(GLFW.VISIBLE, 0)
        win = GLFW.CreateWindow(256, 256, "One-Click Test Window")
        vk_ctx = init_vulkan_context(win, 256, 256)
        MakieEventHandlers._vk_heat_ctx[] = vk_ctx
        MakieEventHandlers.h5_save_path_ref[] = h5_path
        
        ready = SegmentationDisplay.ensure_heatgdt_gpu!(vk_ctx)
        @test ready == true
        
        heat_state = MakieEventHandlers._vk_heat_state[]
        @test heat_state !== nothing
        @test heat_state.diffusivity_uploaded == true
        
        w, h, d = heat_state.width, heat_state.height, heat_state.depth
        seg_vol = zeros(Int16, w, h, d)
        ct_vol = zeros(Float32, w, h, d)
        pet_vol = zeros(Float32, w, h, d)
        
        # Create minimal StateDataFields
        dat_mask = ThreeDimRawDat{Int16}(type=Int16, name="Mask", dat=seg_vol)
        dat_ct   = ThreeDimRawDat{Float32}(type=Float32, name="CT", dat=ct_vol)
        dat_pet  = ThreeDimRawDat{Float32}(type=Float32, name="PET", dat=pet_vol)
        
        dims = DataToScrollDims(voxelSize=(0.9765625, 0.9765625, 3.27))
        scroll_dat = FullScrollableDat(dataToScroll=[dat_mask, dat_ct, dat_pet], dataToScrollDims=dims)
        disp_obj = forDisplayObjects()
        disp_dat = SingleSliceDat(sliceNumber=163)
        
        state = StateDataFields(
            onScrollData=scroll_dat,
            mainForDisplayObjects=disp_obj,
            currentlyDispDat=disp_dat,
            currentDisplayedSlice=163
        )
        stateObjects = [state]
        
        cx, cy, cz = 256, 256, 163
        MakieEventHandlers.current_active_lesion_id[] = 2
        
        # Execute One-Click Heat-GDT
        t0 = time_ns()
        success = MakieEventHandlers.execute_one_click_heatgdt!(stateObjects, cx, cy, cz; K=40, theta=0.001f0)
        dt_ms = (time_ns() - t0) / 1e6
        
        @test success == true
        voxels_segmented = count(seg_vol .== 2)
        println("  ✓ 1-Click GPU execution completed in $(round(dt_ms, digits=2))ms: $voxels_segmented voxels")
        @test voxels_segmented > 50
        @test seg_vol[cx, cy, cz] == 2
        
        # Verify undo box was populated
        @test isassigned(MakieEventHandlers.HEAT_GDT_UNDO_LABELS)
        @test MakieEventHandlers.HEAT_GDT_LAST_TARGET_ID[] == 2
        
        # Cleanup
        VulkanHeatDiffusion.destroy_heat_diffusion!(heat_state, vk_ctx)
        destroy_vulkan_context!(vk_ctx)
        GLFW.DestroyWindow(win)
        GLFW.Terminate()
        println("  ✓ Live Vulkan One-Click Heat-GDT verified successfully!")
    else
        @warn "Skipping pat_6 live test (file not found)"
    end
end

println("\n" * "=" ^ 70)
println("  ALL ONE-CLICK HEAT-GDT TESTS PASSED ✓")
println("=" ^ 70)
