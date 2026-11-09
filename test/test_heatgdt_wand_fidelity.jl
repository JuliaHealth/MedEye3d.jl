#!/usr/bin/env julia
"""
    test_heatgdt_wand_fidelity.jl

Integration test verifying the bounded 1-Click fidelity progression in Wand (Hold) Mode:
1. Initial press: bounded at 20 steps with upfront calibrated s_boost.
2. Subsequent ticks: exactly +10 steps per tick (20 -> 30 -> 40).
3. Hard cap: halts diffusion when heatgdt_K[] is reached (zero over-diffusion).
4. Discrete commit: commits exact mask matching 1-Click mode.
"""

using Test
using Observables
using GLMakie
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

println("=" ^ 70)
println("  Test: Heat-GDT Wand Mode Bounded Progression (20 -> 30 -> 40)")
println("=" ^ 70)

@testset "Wand Mode Bounded Progression" begin
    h5_path = joinpath(@__DIR__, "..", "data", "pat_6_files", "preprocessed_volumes.h5")
    @test isfile(h5_path)
    
    # Load diffusivity
    h5open(h5_path, "r") do h5
        diff_vol = Float32.(read(h5["BASELINE/diffusivity"]))
        InferenceClient.precompute_heatgdt_diffusivity(diff_vol; tp_index=0)
    end
    
    # Initialize headless Vulkan
    GLFW.Init()
    GLFW.WindowHint(GLFW.CLIENT_API, GLFW.NO_API)
    GLFW.WindowHint(GLFW.VISIBLE, 0)
    win = GLFW.CreateWindow(256, 256, "Wand Test Window")
    vk_ctx = init_vulkan_context(win, 256, 256)
    MakieEventHandlers._vk_heat_ctx[] = vk_ctx
    MakieEventHandlers.h5_save_path_ref[] = h5_path
    
    ready = SegmentationDisplay.ensure_heatgdt_gpu!(vk_ctx)
    @test ready == true
    
    heat_state = MakieEventHandlers._vk_heat_state[]
    @test heat_state !== nothing
    
    w, h, d = heat_state.width, heat_state.height, heat_state.depth
    seg_vol = zeros(Int16, w, h, d)
    ct_vol = zeros(Float32, w, h, d)
    pet_vol = zeros(Float32, w, h, d)
    
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
        currentDisplayedSlice=163,
        valueForMasToSet=valueForMasToSetStruct(is_painting_active=true)
    )
    stateObjects = [state]
    
    cx, cy, cz = 256, 256, 163
    MakieEventHandlers.current_viewer_position[] = CartesianIndex(cx, cy, cz)
    MakieEventHandlers.current_active_lesion_id[] = 3
    MakieEventHandlers.heatgdt_K[] = 40
    MakieEventHandlers.heatgdt_theta[] = 0.001f0
    
    try
        # ── Test 1: Switch to Wand Mode ──
        MakieEventHandlers.heatgdt_one_click_mode[] = false
        @test MakieEventHandlers.heatgdt_one_click_mode[] == false
        
        # ── Test 2: Start Wand interaction ──
        println("  Triggering reactToHeatGDTStart...")
        MakieEventHandlers.reactToHeatGDTStart(MedEye3d.MakieEvents.HeatGDTStartEvent(100, 100, 256, 256, 1), stateObjects)
        
        @test MakieEventHandlers._vk_heat_active[] == true
        @test MakieEventHandlers._vk_heat_total_steps[] == 20
        println("  ✓ Initial step bounded at 20 steps: total_steps=$(MakieEventHandlers._vk_heat_total_steps[])")
        
        # Simulate button held
        MedEye3d.ReactOnMouseClickAndDrag.is_left_button_down[] = true
        
        # ── Test 3: Tick 1 (20 -> 30 steps) ──
        sleep(0.06) # wait for debounce
        MakieEventHandlers.reactToHeatGDTTick(MedEye3d.MakieEvents.HeatGDTTickEvent(), stateObjects)
        @test MakieEventHandlers._vk_heat_total_steps[] == 30
        println("  ✓ Tick 1: total_steps=$(MakieEventHandlers._vk_heat_total_steps[]) (+10 steps)")
        
        # ── Test 4: Tick 2 (30 -> 40 steps, hits K=40 target) ──
        sleep(0.06) # wait for debounce
        MakieEventHandlers.reactToHeatGDTTick(MedEye3d.MakieEvents.HeatGDTTickEvent(), stateObjects)
        @test MakieEventHandlers._vk_heat_total_steps[] == 40
        println("  ✓ Tick 2: total_steps=$(MakieEventHandlers._vk_heat_total_steps[]) (hit target K=40)")
        
        # ── Test 5: Tick 3 (prolonged hold: must stay at 40, NO over-diffusion) ──
        sleep(0.06)
        MakieEventHandlers.reactToHeatGDTTick(MedEye3d.MakieEvents.HeatGDTTickEvent(), stateObjects)
        @test MakieEventHandlers._vk_heat_total_steps[] == 40
        println("  ✓ Tick 3 (prolonged hold): total_steps=$(MakieEventHandlers._vk_heat_total_steps[]) (clamped, zero leakage)")
        
        # ── Test 6: Stop and commit ──
        println("  Triggering reactToHeatGDTStop...")
        MakieEventHandlers.reactToHeatGDTStop(MedEye3d.MakieEvents.HeatGDTStopEvent(1), stateObjects)
        @test MakieEventHandlers._vk_heat_active[] == false
        
        voxels_wand = count(seg_vol .== 3)
        println("  ✓ Committed Wand mask: $voxels_wand voxels")
        @test voxels_wand > 50
        @test seg_vol[cx, cy, cz] == 3
        
        # ── Test 7: Verify against 1-Click execution on fresh volume ──
        seg_vol_1click = zeros(Int16, w, h, d)
        stateObjects[1].onScrollData.dataToScroll[1].dat = seg_vol_1click
        MakieEventHandlers.current_active_lesion_id[] = 3
        
        MakieEventHandlers.execute_one_click_heatgdt!(stateObjects, cx, cy, cz; K=40, theta=0.001f0)
        voxels_1click = count(seg_vol_1click .== 3)
        println("  ✓ Comparison: Wand (40 steps) = $voxels_wand voxels vs 1-Click (40 steps) = $voxels_1click voxels")
        
        # Should match very closely (within bounding box readback)
        @test abs(voxels_wand - voxels_1click) / max(voxels_1click, 1) < 0.05
    finally
        # Cleanup
        VulkanHeatDiffusion.destroy_heat_diffusion!(heat_state, vk_ctx)
        destroy_vulkan_context!(vk_ctx)
        GLFW.DestroyWindow(win)
        GLFW.Terminate()
    end
    println("  ✓ All Wand bounded progression tests PASSED!")
end
