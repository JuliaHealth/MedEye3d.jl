using Test
using MedEye3d
using MedEye3d.DataStructs
using MedEye3d.ForDisplayStructs
using MedEye3d.MakieEvents
using MedEye3d.SegmentationDisplay.MakieEventHandlers

@testset "PET vs SPECT Windowing Offset Isolation Tests" begin
    # 1. Verify initial windowing defaults
    MakieEventHandlers.current_windowing["CT"] = Float32[-150.0, 250.0]
    MakieEventHandlers.current_windowing["PET"] = Float32[0.0, 10.0]
    MakieEventHandlers.current_windowing["SPECT"] = Float32[0.0, 10.0]
    
    # Configure TP modalities
    MakieEventHandlers.tp_modalities[0] = "PET"
    MakieEventHandlers.tp_modalities[1] = "SPECT"
    MakieEventHandlers.tp_modalities[2] = "PET"
    MakieEventHandlers.current_tp_index[] = 0
    MakieEventHandlers.compare_mode[] = false
    
    # Create mock PET stateObjects (panel 1, 2)
    spec_ct1 = ForDisplayStructs.TextureSpec{Float32}(name="CT", minAndMaxValue=[-150.0f0, 250.0f0])
    spec_pet1 = ForDisplayStructs.TextureSpec{Float32}(name="PET", minAndMaxValue=[0.0f0, 10.0f0])
    
    st1 = ForDisplayStructs.StateDataFields(
        switchIndex=1,
        mainForDisplayObjects=ForDisplayStructs.forDisplayObjects(listOfTextSpecifications=[spec_ct1, spec_pet1]),
        onScrollData=DataStructs.FullScrollableDat(currentTpIndex=0)
    )
    
    spec_pet2 = ForDisplayStructs.TextureSpec{Float32}(name="PET", minAndMaxValue=[0.0f0, 10.0f0])
    st2 = ForDisplayStructs.StateDataFields(
        switchIndex=2,
        mainForDisplayObjects=ForDisplayStructs.forDisplayObjects(listOfTextSpecifications=[spec_pet2]),
        onScrollData=DataStructs.FullScrollableDat(currentTpIndex=0)
    )
    
    stateObjects = [st1, st2]
    
    # 2. Adjust PET Windowing: should update PET texture, NOT touch SPECT
    reactToWindowing(WindowingEvent("PET", 2.0f0, 15.0f0), stateObjects)
    
    @test MakieEventHandlers.current_windowing["PET"] == Float32[2.0, 15.0]
    @test MakieEventHandlers.current_windowing["SPECT"] == Float32[0.0, 10.0]
    @test spec_pet1.minAndMaxValue == Float32[2.0, 15.0]
    @test spec_pet2.minAndMaxValue == Float32[2.0, 15.0]
    @test spec_ct1.minAndMaxValue == Float32[-150.0, 250.0]
    println("✓ PET window update: PET updated to [2, 15], SPECT remained [0, 10]")
    
    # 3. Adjust SPECT Windowing: should update current_windowing["SPECT"], but NOT touch PET textures (since panel TP is PET)
    reactToWindowing(WindowingEvent("SPECT", 0.5f0, 5.0f0), stateObjects)
    
    @test MakieEventHandlers.current_windowing["SPECT"] == Float32[0.5, 5.0]
    @test MakieEventHandlers.current_windowing["PET"] == Float32[2.0, 15.0]
    @test spec_pet1.minAndMaxValue == Float32[2.0, 15.0]
    @test spec_pet2.minAndMaxValue == Float32[2.0, 15.0]
    println("✓ SPECT window update: SPECT updated to [0.5, 5], PET texture remained unchanged at [2, 15]")
    
    # 4. Compare Mode: Left panel is PET (TP 0), Right panel (panel 5) is SPECT (TP 1)
    MakieEventHandlers.compare_mode[] = true
    MakieEventHandlers.current_tp_index[] = 0
    MakieEventHandlers.compare_right_tp[] = 1
    
    spec_pet5 = ForDisplayStructs.TextureSpec{Float32}(name="PET", minAndMaxValue=[0.5f0, 5.0f0])
    st5 = ForDisplayStructs.StateDataFields(
        switchIndex=5,
        mainForDisplayObjects=ForDisplayStructs.forDisplayObjects(listOfTextSpecifications=[spec_pet5]),
        onScrollData=DataStructs.FullScrollableDat(currentTpIndex=1)
    )
    push!(stateObjects, st5)
    
    # Adjust SPECT window in compare mode
    reactToWindowing(WindowingEvent("SPECT", 1.0f0, 8.0f0), stateObjects)
    
    @test MakieEventHandlers.current_windowing["SPECT"] == Float32[1.0, 8.0]
    @test MakieEventHandlers.current_windowing["PET"] == Float32[2.0, 15.0]
    @test spec_pet1.minAndMaxValue == Float32[2.0, 15.0]   # Panel 1 (PET) untouched
    @test spec_pet5.minAndMaxValue == Float32[1.0, 8.0]    # Panel 5 (SPECT) updated
    println("✓ Compare mode: SPECT update only changed panel 5 (SPECT), panel 1 (PET) remained [2, 15]")
    
    # Adjust PET window in compare mode
    reactToWindowing(WindowingEvent("PET", 3.0f0, 18.0f0), stateObjects)
    
    @test MakieEventHandlers.current_windowing["PET"] == Float32[3.0, 18.0]
    @test MakieEventHandlers.current_windowing["SPECT"] == Float32[1.0, 8.0]
    @test spec_pet1.minAndMaxValue == Float32[3.0, 18.0]   # Panel 1 (PET) updated
    @test spec_pet5.minAndMaxValue == Float32[1.0, 8.0]    # Panel 5 (SPECT) untouched
    println("✓ Compare mode: PET update only changed panel 1 (PET), panel 5 (SPECT) remained [1, 8]")
end
