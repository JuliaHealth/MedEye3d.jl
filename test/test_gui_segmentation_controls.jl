using Test
using Observables
using GLMakie
using MedEye3d

# Ensure headless mode for GLMakie
GLMakie.activate!(visible=false)

using MedEye3d.LesionMetadataWindow
import MedEye3d.SegmentationDisplay.MakieEventHandlers as MEH

@testset "GUI Segmentation Controls & Edge Guidance Toggle" begin
    # Create required observables for metadata window
    active_id = Observable("1: Peripheral Zone (PZ)")
    lesion_ids = Observable(["1: Peripheral Zone (PZ)"])
    channel = Channel{Any}(32)
    display_cfg = Dict{String, Any}("heatgdt_ramp_speed" => 12.0)
    
    # Instantiate metadata window
    win = LesionMetadataWindow.create_metadata_window(
        active_id, lesion_ids, channel
    )
    
    obs_dict = LesionMetadataWindow._lmw_observables
    
    # 1. Verify existence of observables & callbacks
    @test haskey(obs_dict, :algo_combo_selection)
    @test haskey(obs_dict, :btn_edge_both)
    @test haskey(obs_dict, :btn_edge_ct)
    @test haskey(obs_dict, :btn_edge_pet)
    @test haskey(obs_dict, :open_sec_seg!)
    @test haskey(obs_dict, :scroll_to_top!)
    
    # 2. Verify default selection is Heat-GDT
    @test obs_dict[:algo_combo_selection][] == "Heat-GDT"
    
    btn_both = obs_dict[:btn_edge_both]
    btn_ct = obs_dict[:btn_edge_ct]
    btn_pet = obs_dict[:btn_edge_pet]
    
    # 3. Verify initial edge toggle state (default is :combined)
    @test MEH.heatgdt_edge_modality[] === :combined
    @test btn_both.buttoncolor[] != btn_ct.buttoncolor[]
    @test btn_ct.buttoncolor[] == btn_pet.buttoncolor[]
    
    # 4. Click CT Only button
    println("Testing CT Only edge toggle click...")
    btn_ct.clicks[] = btn_ct.clicks[] + 1
    sleep(0.05)
    @test MEH.heatgdt_edge_modality[] === :ct_only
    @test btn_ct.buttoncolor[] != btn_both.buttoncolor[]
    @test btn_both.buttoncolor[] == btn_pet.buttoncolor[]
    
    # 5. Click PET Only button
    println("Testing PET Only edge toggle click...")
    btn_pet.clicks[] = btn_pet.clicks[] + 1
    sleep(0.05)
    @test MEH.heatgdt_edge_modality[] === :pet_only
    @test btn_pet.buttoncolor[] != btn_both.buttoncolor[]
    @test btn_both.buttoncolor[] == btn_ct.buttoncolor[]
    
    # 6. Click Both (Combined) button
    println("Testing Both edge toggle click...")
    btn_both.clicks[] = btn_both.clicks[] + 1
    sleep(0.05)
    @test MEH.heatgdt_edge_modality[] === :combined
    @test btn_both.buttoncolor[] != btn_ct.buttoncolor[]
    @test btn_ct.buttoncolor[] == btn_pet.buttoncolor[]

    # 7. Test open_sec_seg! and scroll_to_top!
    @test_nowarn obs_dict[:open_sec_seg!]()
    @test_nowarn obs_dict[:scroll_to_top!]()

    # 8. Test W key wand mode activation
    println("Testing OneShotHeatGDTEvent (W key)...")
    stateObjects = MedEye3d.StructsManag.StateDataFields[]
    # Ensure mode starts inactive
    romcad = MedEye3d.ReactOnMouseClickAndDrag
    romcad._heatgdt_mode_active[] = false
    
    MEH.reactToOneShotHeatGDT(MedEye3d.MakieEvents.OneShotHeatGDTEvent(), stateObjects)
    @test romcad._heatgdt_mode_active[] == true
    @test obs_dict[:algo_combo_selection][] == "Heat-GDT"
    
    println("✅ All GUI segmentation control and edge toggle tests PASSED!")
end
