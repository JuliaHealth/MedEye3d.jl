#!/usr/bin/env julia
"""
Test for Heat-GDT edge modality switching (Combined, CT only, PET only).
Verifies:
1. Observables and state initialization.
2. Modality switching logic via `switch_heatgdt_edge_modality!`.
3. HDF5 dataset loading:
   - "diffusivity" for :combined
   - "diffusivity_ct" for :ct_only
   - "diffusivity_pet" for :pet_only
4. Graceful fallback to "diffusivity" when modal dataset is absent.
5. Multi-modal caching in `_tp_diffusivity_cache`.
"""

using Test
using HDF5
using MedEye3d
using MedEye3d.InferenceClient

MEH = MedEye3d.SegmentationDisplay.MakieEventHandlers

println("=" ^ 70)
println("  Test: Heat-GDT Modality Edge Switching")
println("=" ^ 70)

@testset "Edge Modality Switching & Caching" begin
    # Create temporary HDF5 file with distinct arrays for each modality
    tmp_h5 = tempname() * ".h5"
    w, h, d = 16, 16, 16
    
    val_combined = fill(0.70f0, w, h, d)
    val_ct       = fill(0.50f0, w, h, d)
    val_pet      = fill(0.90f0, w, h, d)
    
    h5open(tmp_h5, "w") do h5
        g = create_group(h5, "BASELINE")
        g["diffusivity"] = val_combined
        g["diffusivity_ct"] = val_ct
        g["diffusivity_pet"] = val_pet
        
        # FollowUp with ONLY "diffusivity" to test fallback
        g1 = create_group(h5, "TFM_Transform_FollowUp_to_Baseline_1.tfm")
        g1["diffusivity"] = val_combined
        
        # _meta_/preflipped
        gm = create_group(h5, "_meta_")
        gm["preflipped"] = Int32(1)
    end
    
    # Point MakieEventHandlers to our test HDF5
    MEH.h5_save_path_ref[] = tmp_h5
    empty!(MEH._tp_diffusivity_cache)
    InferenceClient.clear_heatgdt_diffusivity!()
    
    # 1. Test Default / Combined
    println("\n[1/4] Testing Combined modality...")
    MEH.switch_heatgdt_edge_modality!(:combined)
    @test MEH.heatgdt_edge_modality[] == :combined
    @test InferenceClient._heatgdt_diffusivity[] !== nothing
    @test InferenceClient._heatgdt_diffusivity[][1, 1, 1] ≈ 0.70f0
    println("  ✓ Combined diffusivity loaded correctly (val ≈ 0.70)")
    
    # 2. Test CT Only
    println("\n[2/4] Testing CT Only modality...")
    MEH.switch_heatgdt_edge_modality!(:ct_only)
    @test MEH.heatgdt_edge_modality[] == :ct_only
    @test InferenceClient._heatgdt_diffusivity[] !== nothing
    @test InferenceClient._heatgdt_diffusivity[][1, 1, 1] ≈ 0.50f0
    println("  ✓ CT-only diffusivity loaded correctly (val ≈ 0.50)")
    
    # 3. Test PET Only
    println("\n[3/4] Testing PET Only modality...")
    MEH.switch_heatgdt_edge_modality!(:pet_only)
    @test MEH.heatgdt_edge_modality[] == :pet_only
    @test InferenceClient._heatgdt_diffusivity[] !== nothing
    @test InferenceClient._heatgdt_diffusivity[][1, 1, 1] ≈ 0.90f0
    println("  ✓ PET-only diffusivity loaded correctly (val ≈ 0.90)")
    
    # 4. Verify Caching
    println("\n[4/4] Verifying Cache & Fallback...")
    @test haskey(MEH._tp_diffusivity_cache, (0, :combined))
    @test haskey(MEH._tp_diffusivity_cache, (0, :ct_only))
    @test haskey(MEH._tp_diffusivity_cache, (0, :pet_only))
    
    # Test Fallback on FollowUp 1 (which lacks diffusivity_ct)
    MEH.current_tp_index[] = 1
    MEH.switch_heatgdt_edge_modality!(:ct_only)
    @test InferenceClient._heatgdt_diffusivity[][1, 1, 1] ≈ 0.70f0  # Fell back to "diffusivity"
    println("  ✓ Fallback to 'diffusivity' when modality dataset is missing works as expected")
    
    # Clean up
    rm(tmp_h5, force=true)
    MEH.h5_save_path_ref[] = ""
    MEH.current_tp_index[] = 0
    MEH.switch_heatgdt_edge_modality!(:combined)
end

@testset "Path Resolution Fallback (h5_path_ref when h5_save_path_ref is empty)" begin
    tmp_h5 = tempname() * ".h5"
    w, h, d = 8, 8, 8
    val_ct = fill(0.33f0, w, h, d)
    val_pet = fill(0.88f0, w, h, d)
    
    h5open(tmp_h5, "w") do h5
        g = create_group(h5, "BASELINE")
        g["diffusivity"] = fill(0.5f0, w, h, d)
        g["diffusivity_ct"] = val_ct
        g["diffusivity_pet"] = val_pet
        gm = create_group(h5, "_meta_")
        gm["preflipped"] = Int32(1)
    end
    
    # Simulate AppMain.jl launch where h5_save_path_ref is empty and only h5_path_ref is set
    MEH.h5_save_path_ref[] = ""
    MEH.h5_path_ref[] = tmp_h5
    empty!(MEH._tp_diffusivity_cache)
    InferenceClient.clear_heatgdt_diffusivity!()
    MEH.current_tp_index[] = 0
    
    # Test CT Only with empty h5_save_path_ref
    MEH.switch_heatgdt_edge_modality!(:ct_only)
    @test MEH.heatgdt_edge_modality[] == :ct_only
    @test InferenceClient._heatgdt_diffusivity[] !== nothing
    @test InferenceClient._heatgdt_diffusivity[][1, 1, 1] ≈ 0.33f0
    println("  ✓ Fallback from h5_save_path_ref to h5_path_ref loaded CT diffusivity (0.33)")
    
    # Test PET Only with empty h5_save_path_ref
    MEH.switch_heatgdt_edge_modality!(:pet_only)
    @test MEH.heatgdt_edge_modality[] == :pet_only
    @test InferenceClient._heatgdt_diffusivity[][1, 1, 1] ≈ 0.88f0
    println("  ✓ Fallback from h5_save_path_ref to h5_path_ref loaded PET diffusivity (0.88)")
    
    rm(tmp_h5, force=true)
    MEH.h5_path_ref[] = ""
    MEH.switch_heatgdt_edge_modality!(:combined)
end

println("\n" * "=" ^ 70)
println("  Modality Edge Switch Tests PASSED!")
println("=" ^ 70)
