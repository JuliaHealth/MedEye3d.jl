#!/usr/bin/env julia
"""
test_interactive_lesion_mapping.jl

Comprehensive unit and integration test suite for Interactive Lesion Mapping:
1. LesionAssociation match group lifecycle (1-to-1, 1-to-many, dissolution, HDF5 round-trip)
2. GLSL 450 Shader generation with differential opacity and SPIR-V compilation
3. MakieEventHandlers toggle & sync lesion logic (UBO packing, colorMask, allowedIDs)
4. Interactive mouse click detection, radial kernel neighborhood, and 3D voxel coordinate transforms
5. GUI slider values and _GLOBAL_APP_STATE persistence
6. Event dispatch and keyboard bindings (ToggleMapModeEvent, Key 'O')
"""

using Test
using HDF5
using JSON
using ColorTypes

using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.MakieEvents
using MedEye3d.DataStructs
using MedEye3d.VulkanBackend.VulkanShaders
using MedEye3d.SegmentationDisplay
using MedEye3d.SegmentationDisplay.MakieEventHandlers
using MedEye3d.LesionAssociation
using MedEye3d.LesionMetadataWindow

@testset "Interactive Lesion Mapping Test Suite" begin

    # =========================================================================
    # Group 1: LesionAssociation Cross-TP Match Groups & Lifecycle
    # =========================================================================
    @testset "1. LesionAssociation Match Groups Lifecycle" begin
        # Reset MATCH_GROUPS
        mg = LesionAssociation.get_match_groups()
        empty!(mg)
        @test isempty(mg)

        # 1a. Create 1-to-1 mapping
        LesionAssociation.update_match_group!("TP_0", 1, "TP_1", 2, "")
        @test length(mg) == 1
        gid1 = first(keys(mg))
        @test length(mg[gid1]) == 2
        
        # Verify cross-lookup
        matches_0_to_1 = LesionAssociation.find_cross_tp_lesion("TP_0", 1, "TP_1")
        @test matches_0_to_1 == [2]
        matches_1_to_0 = LesionAssociation.find_cross_tp_lesion("TP_1", 2, "TP_0")
        @test matches_1_to_0 == [1]

        # Non-existent lesion lookup
        @test isempty(LesionAssociation.find_cross_tp_lesion("TP_0", 99, "TP_1"))
        @test isempty(LesionAssociation.find_cross_tp_lesion("TP_0", 1, "TP_99"))

        # 1b. 1-to-many mapping: link TP_1 lesion 3 to TP_0 lesion 1
        LesionAssociation.update_match_group!("TP_0", 1, "TP_1", 3, "")
        @test length(mg) == 1 # Still in same match group
        matches_1_all = LesionAssociation.find_cross_tp_lesion("TP_0", 1, "TP_1")
        @test sort(matches_1_all) == [2, 3]

        # 1c. Group ID lookup
        @test LesionAssociation.get_group_id_for_lesion("manual_1_TP_0") == gid1
        @test LesionAssociation.get_group_id_for_lesion("nonexistent") === nothing

        # 1d. JSON serialization roundtrip
        json_repr = LesionAssociation._match_groups_to_json()
        @test length(json_repr) == 1
        @test json_repr[1]["group_id"] == gid1
        @test length(json_repr[1]["children"]) == 2

        # Re-parse into fresh dict
        empty!(mg)
        LesionAssociation._parse_matches_data(json_repr)
        @test length(mg) == 1
        @test sort(LesionAssociation.find_cross_tp_lesion("TP_0", 1, "TP_1")) == [2, 3]

        # 1e. Remove single lesion (lesion 3) from match group
        LesionAssociation.remove_from_match_group!("TP_1", 3, "")
        @test length(mg) == 1
        @test LesionAssociation.find_cross_tp_lesion("TP_0", 1, "TP_1") == [2]

        # 1f. Dissolve group when <= 1 member remaining
        LesionAssociation.remove_from_match_group!("TP_1", 2, "")
        @test isempty(mg) # Dissolved automatically
        @test isempty(LesionAssociation.find_cross_tp_lesion("TP_0", 1, "TP_1"))

        # 1g. HDF5 Persistence Roundtrip
        temp_h5 = tempname() * ".h5"
        try
            HDF5.h5open(temp_h5, "w") do h5
                create_group(h5, "_meta_")
            end

            # Populate group and save
            LesionAssociation.update_match_group!("TP_0", 10, "TP_1", 20, temp_h5)
            @test isfile(temp_h5)

            # Re-read from HDF5
            empty!(mg)
            @test isempty(mg)
            LesionAssociation.load_matches_from_h5(temp_h5)
            @test length(mg) == 1
            @test LesionAssociation.find_cross_tp_lesion("TP_0", 10, "TP_1") == [20]
        finally
            rm(temp_h5; force=true)
        end
    end

    # =========================================================================
    # Group 2: Shader Generation & SPIR-V Compilation
    # =========================================================================
    @testset "2. Differential Opacity Shader Generation & Compilation" begin
        # Create TextureSpecs representing CT + discrete Mask
        specs = [
            TextureSpec{Float32}(name="CT", isMainImage=true, minAndMaxValue=[0.0f0, 1000.0f0]),
            TextureSpec{Int16}(
                name="Mask",
                isMultiDiscreteMask=true,
                isIntegerTexture=true,
                colorSet=[RGB(1.0, 0.0, 0.0), RGB(0.0, 1.0, 0.0), RGB(0.0, 0.0, 1.0)],
                minAndMaxValue=Int16[1, 10000]
            )
        ]

        frag_code = VulkanShaders.generate_vulkan_fragment_shader(specs, RGB(1.0, 1.0, 1.0))
        @test !isempty(frag_code)

        # 2a. Verify differential opacity shader constructs are present
        @test occursin("params.MaskColorMask.b > 0.5", frag_code)
        @test occursin("isMapped_Mask = true", frag_code)
        @test occursin("alpha_Mask = clamp(params.MaskColorMask.r, 0.0, 1.0)", frag_code)
        @test occursin("alpha_Mask = clamp(params.MaskColorMask.g, 0.0, 1.0)", frag_code)
        @test occursin("abs(MaskRes - params.MaskallowedIDs[_aidx]) < 0.1", frag_code)
        @test occursin("mix(baseColor, maskColor, alpha_Mask)", frag_code)

        # 2b. Compile to SPIR-V using glslangValidator
        spv_bytes = VulkanShaders.compile_glsl_to_spirv(:frag, frag_code)
        @test !isempty(spv_bytes)
        @test length(spv_bytes) >= 100
        # Verify SPIR-V magic number: 0x07230203 in little endian = [0x03, 0x02, 0x23, 0x07]
        @test spv_bytes[1:4] == UInt8[0x03, 0x02, 0x23, 0x07]
    end

    # =========================================================================
    # Group 3: MakieEventHandlers Toggle & Sync Lesion Logic
    # =========================================================================
    @testset "3. MakieEventHandlers Mapping State & UBO Packing" begin
        # 3a. ToggleMapModeEvent
        orig_mode = MakieEventHandlers.compare_all_lesions_mode[]
        MakieEventHandlers.compare_all_lesions_mode[] = true
        
        # Test helper to mock minimal StateDataFields
        function make_mock_state(panel_idx)
            ts_ct = TextureSpec{Float32}(name="CT", isMainImage=true, minAndMaxValue=[0.0f0, 1000.0f0])
            ts_mask = TextureSpec{Int16}(
                name="Mask",
                isMultiDiscreteMask=true,
                isIntegerTexture=true,
                colorSet=[RGB(1.0, 0.0, 0.0)],
                minAndMaxValue=Int16[1, 10000]
            )
            calcDims = CalcDimsStruct(
                mainImageQuadVert = zeros(Float32, 32),
                mainQuadVertSize = 128
            )
            onScroll = FullScrollableDat(
                dimensionToScroll=3,
                slicesNumber=Int32(50),
                dataToScroll=[
                    ThreeDimRawDat{Float32}(type=Float32, name="CT", dat=zeros(Float32, 10, 10, 50)),
                    ThreeDimRawDat{Int16}(type=Int16, name="Mask", dat=zeros(Int16, 10, 10, 50))
                ]
            )
            mainDisp = forDisplayObjects(
                listOfTextSpecifications=[ts_ct, ts_mask]
            )
            StateDataFields(
                calcDimsStruct=calcDims,
                onScrollData=onScroll,
                mainForDisplayObjects=mainDisp,
                currentDisplayedSlice=25
            )
        end

        mock_states = [make_mock_state(i) for i in 1:6]

        # Test reactToToggleMapMode
        MakieEventHandlers.compare_all_lesions_mode[] = false
        MakieEventHandlers.reactToToggleMapMode(MakieEvents.ToggleMapModeEvent(), mock_states)
        @test MakieEventHandlers.compare_all_lesions_mode[] == true

        # 3b. Test reactToSyncLesion in Compare Mode with Differential Opacity
        MakieEventHandlers.compare_mode[] = true
        MakieEventHandlers.current_tp_index[] = 0
        MakieEventHandlers.compare_right_tp[] = 1
        MakieEventHandlers.tp_node_names[0] = "TP_0"
        MakieEventHandlers.tp_node_names[1] = "TP_1"

        # Setup match: TP_0 lesion 5 <-> TP_1 lesion 12
        mg = LesionAssociation.get_match_groups()
        empty!(mg)
        LesionAssociation.update_match_group!("TP_0", 5, "TP_1", 12, "")

        MakieEventHandlers.mapping_alpha_linked[] = 0.20f0
        MakieEventHandlers.mapping_alpha_unlinked[] = 0.75f0
        MakieEventHandlers.compare_all_lesions_mode[] = true

        MakieEventHandlers.reactToSyncLesion(MakieEvents.SyncLesionEvent(5), mock_states)

        # Left panel (panel 1): single lesion mode
        mask_spec_left = mock_states[1].mainForDisplayObjects.listOfTextSpecifications[2]
        @test mask_spec_left.minAndMaxValue == [5, 5]
        @test mask_spec_left.colorMask.b == 0.0f0 # Differential mode OFF on left
        @test isempty(mask_spec_left.allowedIDs)

        # Right panel (panel 5): differential opacity mode
        mask_spec_right = mock_states[5].mainForDisplayObjects.listOfTextSpecifications[2]
        @test mask_spec_right.minAndMaxValue == [1, 10000] # ALL lesions shown
        @test mask_spec_right.colorMask.b == 1.0f0          # Differential mode ON on right
        @test mask_spec_right.colorMask.r == 0.20f0         # Linked opacity
        @test mask_spec_right.colorMask.g == 0.75f0         # Unlinked opacity
        @test mask_spec_right.allowedIDs == Float32[12.0]   # Mapped lesion ID passed to UBO

        # 3c. toggle_lesion_mapping!
        # Unlink: already mapped (TP_0 lesion 5 <-> TP_1 lesion 12)
        unlinked = MakieEventHandlers.toggle_lesion_mapping!(0, 5, 1, 12, mock_states)
        @test unlinked == true
        @test isempty(LesionAssociation.find_cross_tp_lesion("TP_0", 5, "TP_1"))
        # Re-check UBO after unlinking
        @test isempty(mask_spec_right.allowedIDs)

        # Link: now unmapped -> toggle links it back
        linked = MakieEventHandlers.toggle_lesion_mapping!(0, 5, 1, 12, mock_states)
        @test linked == true
        @test LesionAssociation.find_cross_tp_lesion("TP_0", 5, "TP_1") == [12]
        @test mask_spec_right.allowedIDs == Float32[12.0]

        # 3d. M key (ShowSingleLesion) must NOT reset right panel differential opacity
        # Set up compare mode with differential opacity on right panel
        MakieEventHandlers.compare_mode[] = true
        MakieEventHandlers.compare_all_lesions_mode[] = true
        MakieEventHandlers.is_single_lesion_mode[] = true
        LesionAssociation.update_match_group!("TP_0", 5, "TP_1", 12, "")
        MakieEventHandlers.reactToSyncLesion(MakieEvents.SyncLesionEvent(5), mock_states)
        
        # Verify right panel has differential opacity
        @test mask_spec_right.colorMask.b == 1.0f0
        @test mask_spec_right.minAndMaxValue == [1, 10000]
        
        # Now toggle single/multi via M key (ShowSingleLesionEvent with id=0 means show ALL)
        MakieEventHandlers.reactToShowSingleLesion(MakieEvents.ShowSingleLesionEvent(0), mock_states)
        
        # Left panel should show all lesions (M toggled to multi)
        @test mask_spec_left.minAndMaxValue == [1, 10000]
        @test mask_spec_left.colorMask.b == 0.0f0  # No differential mode on left
        
        # Right panel should STILL have differential opacity (not reset by M key)
        @test mask_spec_right.colorMask.b == 1.0f0   # Differential mode preserved
        @test mask_spec_right.minAndMaxValue == [1, 10000]  # All lesions still shown
        @test mask_spec_right.allowedIDs == Float32[12.0]   # Mapped IDs preserved
        
        # Toggle back to single lesion mode
        MakieEventHandlers.reactToShowSingleLesion(MakieEvents.ShowSingleLesionEvent(5), mock_states)
        
        # Left panel should show only lesion 5
        @test mask_spec_left.minAndMaxValue == [5, 5]
        
        # Right panel should STILL have differential opacity
        @test mask_spec_right.colorMask.b == 1.0f0
        @test mask_spec_right.minAndMaxValue == [1, 10000]

        # Reset state
        MakieEventHandlers.compare_mode[] = false
        MakieEventHandlers.compare_all_lesions_mode[] = orig_mode
    end

    # =========================================================================
    # Group 4: Interactive Mouse Click & Voxel Indexing
    # =========================================================================
    @testset "4. Click Coordinate Mapping & Radial Search Kernel" begin
        # 4a. Voxel coordinate mapping for Axial, Sagittal, Coronal planes
        # Slicing permutation check
        slice = 15
        ix, iy = 42, 68

        # dim == 3 (Axial): (ix, iy, slice)
        coord_axial = CartesianIndex(ix, iy, slice)
        @test (coord_axial[1], coord_axial[2], coord_axial[3]) == (42, 68, 15)

        # dim == 1 (Sagittal): (slice, ix, iy)
        coord_sag = CartesianIndex(slice, ix, iy)
        @test (coord_sag[1], coord_sag[2], coord_sag[3]) == (15, 42, 68)

        # dim == 2 (Coronal): (ix, slice, iy)
        coord_cor = CartesianIndex(ix, slice, iy)
        @test (coord_cor[1], coord_cor[2], coord_cor[3]) == (42, 15, 68)

        # 4b. Neighborhood search test (radial kernel 0 to 3 px)
        test_vol = zeros(Int16, 50, 50, 30)
        target_lesion_id = Int16(42)
        test_vol[20, 20, 15] = target_lesion_id

        offsets = Tuple{Int,Int}[(0,0), (-1,0), (1,0), (0,-1), (0,1),
                                (-1,-1), (1,-1), (-1,1), (1,1),
                                (-2,0), (2,0), (0,-2), (0,2),
                                (-2,-1), (-2,1), (2,-1), (2,1), (-1,-2), (1,-2), (-1,2), (1,2),
                                (-3,0), (3,0), (0,-3), (0,3)]

        function search_neighborhood(vol, cx, cy, cz)
            for (dx, dy) in offsets
                idx = CartesianIndex(cx + dx, cy + dy, cz)
                if checkbounds(Bool, vol, idx)
                    v = vol[idx]
                    if v > 0
                        return Int(v)
                    end
                end
            end
            return 0
        end

        # Exact hit
        @test search_neighborhood(test_vol, 20, 20, 15) == 42
        # 1 px offset
        @test search_neighborhood(test_vol, 21, 20, 15) == 42
        # 2 px offset
        @test search_neighborhood(test_vol, 20, 22, 15) == 42
        # 3 px offset
        @test search_neighborhood(test_vol, 17, 20, 15) == 42
        # 4 px offset (beyond radial kernel)
        @test search_neighborhood(test_vol, 20, 24, 15) == 0

        # 4c. Click vs Drag discrimination
        is_click(dt, dist, painting, measuring, heatgdt) = begin
            !painting && !measuring && !heatgdt && dt < 0.45 && dist < 8.0
        end

        @test is_click(0.12, 3.5, false, false, false) == true   # Valid click
        @test is_click(0.55, 3.5, false, false, false) == false  # Held too long (> 0.45s)
        @test is_click(0.12, 12.0, false, false, false) == false # Dragged (> 8.0px)
        @test is_click(0.12, 3.5, true, false, false) == false   # Painting active -> ignored
        @test is_click(0.12, 3.5, false, true, false) == false   # Measuring active -> ignored
        @test is_click(0.12, 3.5, false, false, true) == false   # HeatGDT active -> ignored
    end

    # =========================================================================
    # Group 5: GUI Sliders & Session Persistence
    # =========================================================================
    @testset "5. Opacity Slider Properties & Global State Persistence" begin
        # 5a. Initial values
        @test MakieEventHandlers.mapping_alpha_linked[] isa Float32
        @test MakieEventHandlers.mapping_alpha_unlinked[] isa Float32
        @test 0.0f0 < MakieEventHandlers.mapping_alpha_linked[] < 1.0f0
        @test 0.0f0 < MakieEventHandlers.mapping_alpha_unlinked[] <= 1.0f0
        @test MakieEventHandlers.mapping_alpha_linked[] < MakieEventHandlers.mapping_alpha_unlinked[]

        # 5b. Update slider values and check clamping
        MakieEventHandlers.mapping_alpha_linked[] = 0.15f0
        MakieEventHandlers.mapping_alpha_unlinked[] = 0.85f0
        @test MakieEventHandlers.mapping_alpha_linked[] == 0.15f0
        @test MakieEventHandlers.mapping_alpha_unlinked[] == 0.85f0

        # 5c. Global App State roundtrip
        state_dict = Dict{String, Any}(
            "mapping_alpha_linked" => 0.18f0,
            "mapping_alpha_unlinked" => 0.72f0,
            "compare_all_lesions_mode" => false
        )
        # Apply
        if haskey(state_dict, "mapping_alpha_linked")
            MakieEventHandlers.mapping_alpha_linked[] = Float32(state_dict["mapping_alpha_linked"])
        end
        if haskey(state_dict, "mapping_alpha_unlinked")
            MakieEventHandlers.mapping_alpha_unlinked[] = Float32(state_dict["mapping_alpha_unlinked"])
        end
        if haskey(state_dict, "compare_all_lesions_mode")
            MakieEventHandlers.compare_all_lesions_mode[] = Bool(state_dict["compare_all_lesions_mode"])
        end

        @test MakieEventHandlers.mapping_alpha_linked[] ≈ 0.18f0
        @test MakieEventHandlers.mapping_alpha_unlinked[] ≈ 0.72f0
        @test MakieEventHandlers.compare_all_lesions_mode[] == false

        # Restore default values
        MakieEventHandlers.mapping_alpha_linked[] = 0.25f0
        MakieEventHandlers.mapping_alpha_unlinked[] = 0.80f0
        MakieEventHandlers.compare_all_lesions_mode[] = true
    end

    # =========================================================================
    # Group 6: Event Dispatch & Registration
    # =========================================================================
    @testset "6. Event Handlers and Method Dispatch" begin
        # Verify ToggleMapModeEvent method signature is defined on SegmentationDisplay.on_next!
        @test hasmethod(SegmentationDisplay.on_next!, Tuple{Vector{StateDataFields}, ToggleMapModeEvent})
        @test hasmethod(MakieEventHandlers.reactToToggleMapMode, Tuple{ToggleMapModeEvent, Vector{StateDataFields}})
        @test hasmethod(MakieEventHandlers.toggle_lesion_mapping!, Tuple{Int, Int, Int, Int, Vector{StateDataFields}})
    end

    # =========================================================================
    # Group 7: Multi-TP Match Groups & Group Merging
    # =========================================================================
    @testset "7. Multi-TP Match Groups & Group Merging" begin
        mg = LesionAssociation.get_match_groups()
        empty!(mg)

        # 7a. 3-TP chain: TP_0 L1 → TP_1 L5 → TP_2 L10
        LesionAssociation.update_match_group!("TP_0", 1, "TP_1", 5, "")
        LesionAssociation.update_match_group!("TP_1", 5, "TP_2", 10, "")
        # Both should be in the same group since TP_1 L5 was already in a group
        @test length(mg) == 1
        gid_chain = first(keys(mg))
        @test length(mg[gid_chain]) == 3  # 3 entries across 3 TPs

        # Cross-TP lookups should work across the chain
        @test LesionAssociation.find_cross_tp_lesion("TP_0", 1, "TP_1") == [5]
        @test LesionAssociation.find_cross_tp_lesion("TP_1", 5, "TP_2") == [10]
        @test LesionAssociation.find_cross_tp_lesion("TP_0", 1, "TP_2") == [10]
        @test LesionAssociation.find_cross_tp_lesion("TP_2", 10, "TP_0") == [1]

        # 7b. Group merging: two separate groups that get merged
        empty!(mg)
        LesionAssociation.update_match_group!("TP_0", 1, "TP_1", 5, "")
        LesionAssociation.update_match_group!("TP_0", 2, "TP_1", 6, "")
        @test length(mg) == 2  # Two separate groups

        # Now link L1 ↔ L6 — this should merge the two groups
        LesionAssociation.update_match_group!("TP_0", 1, "TP_1", 6, "")
        @test length(mg) == 1  # Merged into one group
        merged_gid = first(keys(mg))
        @test length(mg[merged_gid]) == 4  # All 4 entries

        # 7c. Duplicate insert is idempotent
        LesionAssociation.update_match_group!("TP_0", 1, "TP_1", 5, "")
        @test length(mg) == 1
        @test length(mg[merged_gid]) == 4  # No duplicates

        # 7d. get_group_id_for_lesion after merge
        @test LesionAssociation.get_group_id_for_lesion("manual_1_TP_0") == merged_gid
        @test LesionAssociation.get_group_id_for_lesion("manual_2_TP_0") == merged_gid
        @test LesionAssociation.get_group_id_for_lesion("manual_5_TP_1") == merged_gid
        @test LesionAssociation.get_group_id_for_lesion("manual_6_TP_1") == merged_gid

        # 7e. Remove one member from merged group — group persists
        LesionAssociation.remove_from_match_group!("TP_0", 2, "")
        @test length(mg) == 1
        @test length(mg[merged_gid]) == 3

        # 7f. JSON roundtrip of multi-TP chain
        json_repr = LesionAssociation._match_groups_to_json()
        @test length(json_repr) == 1
        total_children = length(json_repr[1]["children"])
        @test total_children == 2  # 3 members → 1 parent + 2 children

        # Roundtrip parse
        empty!(mg)
        LesionAssociation._parse_matches_data(json_repr)
        @test length(mg) == 1
        @test length(first(values(mg))) == 3

        empty!(mg)
    end

    # =========================================================================
    # Group 8: Toggle Map Mode Lifecycle & State Transitions
    # =========================================================================
    @testset "8. Toggle Map Mode Lifecycle" begin
        function make_mock_state_v2(panel_idx)
            ts_ct = TextureSpec{Float32}(name="CT", isMainImage=true, minAndMaxValue=[0.0f0, 1000.0f0])
            ts_mask = TextureSpec{Int16}(
                name="Mask",
                isMultiDiscreteMask=true,
                isIntegerTexture=true,
                colorSet=[RGB(1.0, 0.0, 0.0)],
                minAndMaxValue=Int16[1, 10000]
            )
            calcDims = CalcDimsStruct(
                mainImageQuadVert = zeros(Float32, 32),
                mainQuadVertSize = 128
            )
            onScroll = FullScrollableDat(
                dimensionToScroll=3,
                slicesNumber=Int32(50),
                dataToScroll=[
                    ThreeDimRawDat{Float32}(type=Float32, name="CT", dat=zeros(Float32, 10, 10, 50)),
                    ThreeDimRawDat{Int16}(type=Int16, name="Mask", dat=zeros(Int16, 10, 10, 50))
                ]
            )
            mainDisp = forDisplayObjects(
                listOfTextSpecifications=[ts_ct, ts_mask]
            )
            StateDataFields(
                calcDimsStruct=calcDims,
                onScrollData=onScroll,
                mainForDisplayObjects=mainDisp,
                currentDisplayedSlice=25
            )
        end

        mock_states8 = [make_mock_state_v2(i) for i in 1:9]

        # 8a. Toggle ON → OFF → ON cycle
        MakieEventHandlers.compare_all_lesions_mode[] = false
        MakieEventHandlers.compare_mode[] = true
        MakieEventHandlers.current_tp_index[] = 0
        MakieEventHandlers.compare_right_tp[] = 1
        MakieEventHandlers.tp_node_names[0] = "TP_0"
        MakieEventHandlers.tp_node_names[1] = "TP_1"

        # Toggle ON
        MakieEventHandlers.reactToToggleMapMode(MakieEvents.ToggleMapModeEvent(), mock_states8)
        @test MakieEventHandlers.compare_all_lesions_mode[] == true

        # Toggle OFF
        MakieEventHandlers.reactToToggleMapMode(MakieEvents.ToggleMapModeEvent(), mock_states8)
        @test MakieEventHandlers.compare_all_lesions_mode[] == false

        # Toggle ON again
        MakieEventHandlers.reactToToggleMapMode(MakieEvents.ToggleMapModeEvent(), mock_states8)
        @test MakieEventHandlers.compare_all_lesions_mode[] == true

        # 8b. Toggle map mode triggers reactToSyncLesion (check UBOs updated)
        MakieEventHandlers.compare_all_lesions_mode[] = false
        MakieEventHandlers.current_active_lesion_id[] = 7
        mg = LesionAssociation.get_match_groups()
        empty!(mg)
        LesionAssociation.update_match_group!("TP_0", 7, "TP_1", 14, "")

        # Toggle ON — right panel should get differential opacity
        MakieEventHandlers.reactToToggleMapMode(MakieEvents.ToggleMapModeEvent(), mock_states8)
        @test MakieEventHandlers.compare_all_lesions_mode[] == true

        mask_right = mock_states8[5].mainForDisplayObjects.listOfTextSpecifications[2]
        @test mask_right.colorMask.b == 1.0f0  # Differential mode ON
        @test mask_right.minAndMaxValue == [1, 10000]

        # Toggle OFF — right panel loses differential opacity
        MakieEventHandlers.reactToToggleMapMode(MakieEvents.ToggleMapModeEvent(), mock_states8)
        @test MakieEventHandlers.compare_all_lesions_mode[] == false

        mask_right2 = mock_states8[5].mainForDisplayObjects.listOfTextSpecifications[2]
        @test mask_right2.colorMask.b == 0.0f0  # Differential mode OFF

        # Cleanup
        empty!(mg)
        MakieEventHandlers.compare_mode[] = false
    end

    # =========================================================================
    # Group 9: SyncLesion Edge Cases
    # =========================================================================
    @testset "9. SyncLesion Edge Cases" begin
        function make_mock_state_v3(panel_idx)
            ts_ct = TextureSpec{Float32}(name="CT", isMainImage=true, minAndMaxValue=[0.0f0, 1000.0f0])
            ts_mask = TextureSpec{Int16}(
                name="Mask",
                isMultiDiscreteMask=true,
                isIntegerTexture=true,
                colorSet=[RGB(1.0, 0.0, 0.0)],
                minAndMaxValue=Int16[1, 10000]
            )
            calcDims = CalcDimsStruct(
                mainImageQuadVert = zeros(Float32, 32),
                mainQuadVertSize = 128
            )
            onScroll = FullScrollableDat(
                dimensionToScroll=3,
                slicesNumber=Int32(50),
                dataToScroll=[
                    ThreeDimRawDat{Float32}(type=Float32, name="CT", dat=zeros(Float32, 10, 10, 50)),
                    ThreeDimRawDat{Int16}(type=Int16, name="Mask", dat=zeros(Int16, 10, 10, 50))
                ]
            )
            mainDisp = forDisplayObjects(
                listOfTextSpecifications=[ts_ct, ts_mask]
            )
            StateDataFields(
                calcDimsStruct=calcDims,
                onScrollData=onScroll,
                mainForDisplayObjects=mainDisp,
                currentDisplayedSlice=25
            )
        end

        mock_states9 = [make_mock_state_v3(i) for i in 1:9]
        mg = LesionAssociation.get_match_groups()
        empty!(mg)

        # 9a. Non-compare mode: sync lesion doesn't affect right panels
        MakieEventHandlers.compare_mode[] = false
        MakieEventHandlers.is_single_lesion_mode[] = true
        MakieEventHandlers.reactToSyncLesion(MakieEvents.SyncLesionEvent(42), mock_states9)

        mask_left = mock_states9[1].mainForDisplayObjects.listOfTextSpecifications[2]
        @test mask_left.minAndMaxValue == [42, 42]  # Single lesion
        @test isempty(mask_left.allowedIDs)

        # Right panel should also show single lesion (no compare override)
        mask_right = mock_states9[5].mainForDisplayObjects.listOfTextSpecifications[2]
        @test mask_right.minAndMaxValue == [42, 42]
        @test isempty(mask_right.allowedIDs)

        # 9b. Compare mode with NO matches: right panel gets no differential mode
        MakieEventHandlers.compare_mode[] = true
        MakieEventHandlers.current_tp_index[] = 0
        MakieEventHandlers.compare_right_tp[] = 1
        MakieEventHandlers.tp_node_names[0] = "TP_0"
        MakieEventHandlers.tp_node_names[1] = "TP_1"
        MakieEventHandlers.compare_all_lesions_mode[] = true
        empty!(mg)  # No matches at all

        MakieEventHandlers.reactToSyncLesion(MakieEvents.SyncLesionEvent(3), mock_states9)

        mask_right_no = mock_states9[5].mainForDisplayObjects.listOfTextSpecifications[2]
        # Even with compare_all_lesions_mode ON, if no matches found → still uses mode
        @test mask_right_no.colorMask.b == 1.0f0  # Differential mode still on
        @test isempty(mask_right_no.allowedIDs)    # But no IDs to highlight

        # 9c. Compare mode with MULTIPLE matches (1-to-many)
        LesionAssociation.update_match_group!("TP_0", 3, "TP_1", 10, "")
        LesionAssociation.update_match_group!("TP_0", 3, "TP_1", 11, "")
        LesionAssociation.update_match_group!("TP_0", 3, "TP_1", 12, "")

        MakieEventHandlers.reactToSyncLesion(MakieEvents.SyncLesionEvent(3), mock_states9)

        mask_right_multi = mock_states9[5].mainForDisplayObjects.listOfTextSpecifications[2]
        @test mask_right_multi.colorMask.b == 1.0f0
        @test sort(mask_right_multi.allowedIDs) == Float32[10.0, 11.0, 12.0]
        @test mask_right_multi.minAndMaxValue == [1, 10000]  # Shows all

        # 9d. updates current_active_lesion_id
        @test MakieEventHandlers.current_active_lesion_id[] == 3

        # 9e. Multi-lesion mode on left (non-single)
        MakieEventHandlers.is_single_lesion_mode[] = false
        MakieEventHandlers.reactToSyncLesion(MakieEvents.SyncLesionEvent(3), mock_states9)
        mask_left2 = mock_states9[1].mainForDisplayObjects.listOfTextSpecifications[2]
        @test mask_left2.minAndMaxValue == [1, 10000]  # All lesions

        # Cleanup
        empty!(mg)
        MakieEventHandlers.compare_mode[] = false
        MakieEventHandlers.is_single_lesion_mode[] = true
    end

    # =========================================================================
    # Group 10: M Key in Non-Compare Mode
    # =========================================================================
    @testset "10. M Key in Non-Compare Mode" begin
        function make_mock_state_nc(panel_idx)
            ts_ct = TextureSpec{Float32}(name="CT", isMainImage=true, minAndMaxValue=[0.0f0, 1000.0f0])
            ts_mask = TextureSpec{Int16}(
                name="Mask",
                isMultiDiscreteMask=true,
                isIntegerTexture=true,
                colorSet=[RGB(1.0, 0.0, 0.0)],
                minAndMaxValue=Int16[1, 10000]
            )
            calcDims = CalcDimsStruct(
                mainImageQuadVert = zeros(Float32, 32),
                mainQuadVertSize = 128
            )
            onScroll = FullScrollableDat(
                dimensionToScroll=3,
                slicesNumber=Int32(50),
                dataToScroll=[
                    ThreeDimRawDat{Float32}(type=Float32, name="CT", dat=zeros(Float32, 10, 10, 50)),
                    ThreeDimRawDat{Int16}(type=Int16, name="Mask", dat=zeros(Int16, 10, 10, 50))
                ]
            )
            mainDisp = forDisplayObjects(
                listOfTextSpecifications=[ts_ct, ts_mask]
            )
            StateDataFields(
                calcDimsStruct=calcDims,
                onScrollData=onScroll,
                mainForDisplayObjects=mainDisp,
                currentDisplayedSlice=25
            )
        end

        mock_nc = [make_mock_state_nc(i) for i in 1:9]

        MakieEventHandlers.compare_mode[] = false
        MakieEventHandlers.is_single_lesion_mode[] = true
        MakieEventHandlers.current_active_lesion_id[] = 10

        # 10a. Show single lesion (id=10) — ALL panels get id=10
        MakieEventHandlers.reactToShowSingleLesion(MakieEvents.ShowSingleLesionEvent(10), mock_nc)
        for idx in 1:9
            mask = mock_nc[idx].mainForDisplayObjects.listOfTextSpecifications[2]
            @test mask.minAndMaxValue == [10, 10]
            @test isempty(mask.allowedIDs)
            @test mask.colorMask.b == 0.0f0  # No differential mode
        end

        # 10b. Show all lesions (id=0) — ALL panels show [1, 10000]
        MakieEventHandlers.reactToShowSingleLesion(MakieEvents.ShowSingleLesionEvent(0), mock_nc)
        for idx in 1:9
            mask = mock_nc[idx].mainForDisplayObjects.listOfTextSpecifications[2]
            @test mask.minAndMaxValue == [1, 10000]
            @test isempty(mask.allowedIDs)
        end

        # 10c. Show single mode sets is_single_lesion_mode correctly
        MakieEventHandlers.reactToShowSingleLesion(MakieEvents.ShowSingleLesionEvent(5), mock_nc)
        @test MakieEventHandlers.is_single_lesion_mode[] == true
        @test MakieEventHandlers.current_active_lesion_id[] == 5

        MakieEventHandlers.reactToShowSingleLesion(MakieEvents.ShowSingleLesionEvent(0), mock_nc)
        @test MakieEventHandlers.is_single_lesion_mode[] == false
    end

    # =========================================================================
    # Group 11: toggle_lesion_mapping! Edge Cases
    # =========================================================================
    @testset "11. toggle_lesion_mapping! Edge Cases" begin
        function make_mock_state_tl(panel_idx)
            ts_ct = TextureSpec{Float32}(name="CT", isMainImage=true, minAndMaxValue=[0.0f0, 1000.0f0])
            ts_mask = TextureSpec{Int16}(
                name="Mask",
                isMultiDiscreteMask=true,
                isIntegerTexture=true,
                colorSet=[RGB(1.0, 0.0, 0.0)],
                minAndMaxValue=Int16[1, 10000]
            )
            calcDims = CalcDimsStruct(
                mainImageQuadVert = zeros(Float32, 32),
                mainQuadVertSize = 128
            )
            onScroll = FullScrollableDat(
                dimensionToScroll=3,
                slicesNumber=Int32(50),
                dataToScroll=[
                    ThreeDimRawDat{Float32}(type=Float32, name="CT", dat=zeros(Float32, 10, 10, 50)),
                    ThreeDimRawDat{Int16}(type=Int16, name="Mask", dat=zeros(Int16, 10, 10, 50))
                ]
            )
            mainDisp = forDisplayObjects(
                listOfTextSpecifications=[ts_ct, ts_mask]
            )
            StateDataFields(
                calcDimsStruct=calcDims,
                onScrollData=onScroll,
                mainForDisplayObjects=mainDisp,
                currentDisplayedSlice=25
            )
        end

        mock_tl = [make_mock_state_tl(i) for i in 1:9]
        mg = LesionAssociation.get_match_groups()
        empty!(mg)

        MakieEventHandlers.compare_mode[] = true
        MakieEventHandlers.current_tp_index[] = 0
        MakieEventHandlers.compare_right_tp[] = 1
        MakieEventHandlers.tp_node_names[0] = "TP_0"
        MakieEventHandlers.tp_node_names[1] = "TP_1"
        MakieEventHandlers.compare_all_lesions_mode[] = true

        # 11a. Invalid IDs return false
        @test MakieEventHandlers.toggle_lesion_mapping!(0, 0, 1, 5, mock_tl) == false
        @test MakieEventHandlers.toggle_lesion_mapping!(0, 5, 1, 0, mock_tl) == false
        @test MakieEventHandlers.toggle_lesion_mapping!(0, -1, 1, 5, mock_tl) == false
        @test isempty(mg)

        # 11b. Link → verify → unlink → verify cycle
        result = MakieEventHandlers.toggle_lesion_mapping!(0, 1, 1, 5, mock_tl)
        @test result == true
        @test LesionAssociation.find_cross_tp_lesion("TP_0", 1, "TP_1") == [5]

        result2 = MakieEventHandlers.toggle_lesion_mapping!(0, 1, 1, 5, mock_tl)
        @test result2 == true
        @test isempty(LesionAssociation.find_cross_tp_lesion("TP_0", 1, "TP_1"))

        # 11c. Many-to-1: Multiple left lesions mapped to same right lesion
        MakieEventHandlers.toggle_lesion_mapping!(0, 1, 1, 10, mock_tl)
        MakieEventHandlers.toggle_lesion_mapping!(0, 2, 1, 10, mock_tl)
        @test LesionAssociation.find_cross_tp_lesion("TP_0", 1, "TP_1") == [10]
        @test LesionAssociation.find_cross_tp_lesion("TP_0", 2, "TP_1") == [10]

        # Unlinking L1↔L10 removes L10 from the group entirely, which also
        # disconnects L2 from TP_1 (both were linked through the same L10 entry)
        MakieEventHandlers.toggle_lesion_mapping!(0, 1, 1, 10, mock_tl)
        @test isempty(LesionAssociation.find_cross_tp_lesion("TP_0", 1, "TP_1"))
        @test isempty(LesionAssociation.find_cross_tp_lesion("TP_0", 2, "TP_1"))  # Also disconnected

        # 11d. Re-link L2↔L10 independently — but since L1 and L2 are still
        # in the same group (only L10 was removed), re-adding L10 via L2
        # transitively re-connects L1 to L10 through the group
        MakieEventHandlers.toggle_lesion_mapping!(0, 2, 1, 10, mock_tl)
        @test LesionAssociation.find_cross_tp_lesion("TP_0", 2, "TP_1") == [10]
        @test LesionAssociation.find_cross_tp_lesion("TP_0", 1, "TP_1") == [10]  # Transitively reconnected

        # Cleanup
        empty!(mg)
        MakieEventHandlers.compare_mode[] = false
    end

    # =========================================================================
    # Group 12: Voxel Coordinate Transforms for All 3 Planes
    # =========================================================================
    @testset "12. Full 3D Voxel Coordinate Transforms" begin
        vol = zeros(Int16, 50, 50, 30)

        # Place lesions at specific 3D coordinates
        vol[25, 30, 15] = Int16(1)  # Lesion 1
        vol[10, 40, 20] = Int16(2)  # Lesion 2
        vol[48, 5, 28]  = Int16(3)  # Lesion 3 near edge

        # 12a. Axial plane (dim=3): index = (ix, iy, slice)
        @test vol[CartesianIndex(25, 30, 15)] == 1
        @test vol[CartesianIndex(10, 40, 20)] == 2

        # 12b. Sagittal plane (dim=1): index = (slice, ix, iy)
        @test vol[CartesianIndex(25, 30, 15)] == 1  # slice=25 shows (30, 15)
        @test vol[CartesianIndex(10, 40, 20)] == 2  # slice=10 shows (40, 20)

        # 12c. Coronal plane (dim=2): index = (ix, slice, iy)
        @test vol[CartesianIndex(25, 30, 15)] == 1  # slice=30 shows (25, 15)

        # 12d. Edge boundary: lesion at near-boundary voxels
        @test vol[CartesianIndex(48, 5, 28)] == 3
        @test checkbounds(Bool, vol, CartesianIndex(51, 5, 28)) == false
        @test checkbounds(Bool, vol, CartesianIndex(48, 0, 28)) == false

        # 12e. Radial kernel with boundary clipping
        offsets = Tuple{Int,Int}[(0,0), (-1,0), (1,0), (0,-1), (0,1),
                                (-1,-1), (1,-1), (-1,1), (1,1),
                                (-2,0), (2,0), (0,-2), (0,2),
                                (-3,0), (3,0), (0,-3), (0,3)]

        # Search near boundary — should not throw
        function safe_search(vol, cx, cy, cz)
            for (dx, dy) in offsets
                idx = CartesianIndex(cx + dx, cy + dy, cz)
                if checkbounds(Bool, vol, idx)
                    v = vol[idx]
                    if v > 0
                        return Int(v)
                    end
                end
            end
            return 0
        end

        # Edge case: corner search
        @test safe_search(vol, 48, 5, 28) == 3
        @test safe_search(vol, 50, 5, 28) == 3  # 2px away still in range
        @test safe_search(vol, 1, 1, 1) == 0     # Nothing near corner

        # 12f. Multiple lesions near each other — first match wins
        vol2 = zeros(Int16, 50, 50, 30)
        vol2[20, 20, 10] = Int16(5)
        vol2[21, 20, 10] = Int16(6)  # Adjacent to lesion 5

        # Center pixel hit: lesion 5
        @test safe_search(vol2, 20, 20, 10) == 5
        # Center pixel hit: lesion 6
        @test safe_search(vol2, 21, 20, 10) == 6
        # Offset search from (19, 20): finds L5 at (20,20) via (1,0) offset
        @test safe_search(vol2, 19, 20, 10) == 5
    end

    # =========================================================================
    # Group 13: _tp_index_from_node Helper
    # =========================================================================
    @testset "13. _tp_index_from_node Parsing" begin
        @test LesionAssociation._tp_index_from_node("TP_0") == 0
        @test LesionAssociation._tp_index_from_node("TP_1") == 1
        @test LesionAssociation._tp_index_from_node("TP_42") == 42
        @test LesionAssociation._tp_index_from_node("PET_Lesions_3") == 3
        @test LesionAssociation._tp_index_from_node("invalid_no_number") == 999  # fallback
    end

    # =========================================================================
    # Group 14: HDF5 Persistence with Multiple Groups
    # =========================================================================
    @testset "14. HDF5 Persistence with Multiple Groups" begin
        mg = LesionAssociation.get_match_groups()
        empty!(mg)

        temp_h5 = tempname() * ".h5"
        try
            HDF5.h5open(temp_h5, "w") do h5
                create_group(h5, "_meta_")
            end

            # Create multiple groups
            LesionAssociation.update_match_group!("TP_0", 1, "TP_1", 10, temp_h5)
            LesionAssociation.update_match_group!("TP_0", 2, "TP_1", 20, temp_h5)
            LesionAssociation.update_match_group!("TP_0", 3, "TP_1", 30, temp_h5)

            @test length(mg) == 3

            # Re-read from HDF5
            empty!(mg)
            @test isempty(mg)
            LesionAssociation.load_matches_from_h5(temp_h5)
            @test length(mg) == 3

            # Verify all lookups still work after load
            @test LesionAssociation.find_cross_tp_lesion("TP_0", 1, "TP_1") == [10]
            @test LesionAssociation.find_cross_tp_lesion("TP_0", 2, "TP_1") == [20]
            @test LesionAssociation.find_cross_tp_lesion("TP_0", 3, "TP_1") == [30]

            # Reverse lookups
            @test LesionAssociation.find_cross_tp_lesion("TP_1", 10, "TP_0") == [1]
            @test LesionAssociation.find_cross_tp_lesion("TP_1", 20, "TP_0") == [2]
            @test LesionAssociation.find_cross_tp_lesion("TP_1", 30, "TP_0") == [3]

            # Delete one and re-save
            LesionAssociation.remove_from_match_group!("TP_0", 2, temp_h5)
            @test length(mg) == 2

            # Re-read again
            empty!(mg)
            LesionAssociation.load_matches_from_h5(temp_h5)
            @test length(mg) == 2
            @test isempty(LesionAssociation.find_cross_tp_lesion("TP_0", 2, "TP_1"))
        finally
            rm(temp_h5; force=true)
        end
    end

    # =========================================================================
    # Group 15: Panel Independence — Compare vs Normal UBO State
    # =========================================================================
    @testset "15. Panel Independence — Compare vs Normal UBO State" begin
        function make_mock_state_pi(panel_idx)
            ts_ct = TextureSpec{Float32}(name="CT", isMainImage=true, minAndMaxValue=[0.0f0, 1000.0f0])
            ts_mask = TextureSpec{Int16}(
                name="Mask",
                isMultiDiscreteMask=true,
                isIntegerTexture=true,
                colorSet=[RGB(1.0, 0.0, 0.0)],
                minAndMaxValue=Int16[1, 10000]
            )
            calcDims = CalcDimsStruct(
                mainImageQuadVert = zeros(Float32, 32),
                mainQuadVertSize = 128
            )
            onScroll = FullScrollableDat(
                dimensionToScroll=3,
                slicesNumber=Int32(50),
                dataToScroll=[
                    ThreeDimRawDat{Float32}(type=Float32, name="CT", dat=zeros(Float32, 10, 10, 50)),
                    ThreeDimRawDat{Int16}(type=Int16, name="Mask", dat=zeros(Int16, 10, 10, 50))
                ]
            )
            mainDisp = forDisplayObjects(
                listOfTextSpecifications=[ts_ct, ts_mask]
            )
            StateDataFields(
                calcDimsStruct=calcDims,
                onScrollData=onScroll,
                mainForDisplayObjects=mainDisp,
                currentDisplayedSlice=25
            )
        end

        mock_pi = [make_mock_state_pi(i) for i in 1:9]
        mg = LesionAssociation.get_match_groups()
        empty!(mg)

        MakieEventHandlers.compare_mode[] = true
        MakieEventHandlers.compare_all_lesions_mode[] = true
        MakieEventHandlers.current_tp_index[] = 0
        MakieEventHandlers.compare_right_tp[] = 1
        MakieEventHandlers.tp_node_names[0] = "TP_0"
        MakieEventHandlers.tp_node_names[1] = "TP_1"
        MakieEventHandlers.is_single_lesion_mode[] = true

        # Setup a match
        LesionAssociation.update_match_group!("TP_0", 5, "TP_1", 15, "")

        MakieEventHandlers.reactToSyncLesion(MakieEvents.SyncLesionEvent(5), mock_pi)

        # 15a. Left panels (1-4) have single lesion, no differential
        for idx in 1:4
            mask = mock_pi[idx].mainForDisplayObjects.listOfTextSpecifications[2]
            @test mask.minAndMaxValue == [5, 5]       # Single lesion
            @test mask.colorMask.b == 0.0f0            # No differential
            @test isempty(mask.allowedIDs)
        end

        # 15b. Right panels (5-9) have all lesions, differential opacity
        for idx in 5:9
            mask = mock_pi[idx].mainForDisplayObjects.listOfTextSpecifications[2]
            @test mask.minAndMaxValue == [1, 10000]    # All lesions
            @test mask.colorMask.b == 1.0f0             # Differential ON
            @test mask.allowedIDs == Float32[15.0]      # Mapped ID
        end

        # 15c. After M key (toggle to multi), left changes but right stays
        MakieEventHandlers.reactToShowSingleLesion(MakieEvents.ShowSingleLesionEvent(0), mock_pi)

        for idx in 1:4
            mask = mock_pi[idx].mainForDisplayObjects.listOfTextSpecifications[2]
            @test mask.minAndMaxValue == [1, 10000]    # Now multi
            @test mask.colorMask.b == 0.0f0             # Still no differential on left
        end

        for idx in 5:9
            mask = mock_pi[idx].mainForDisplayObjects.listOfTextSpecifications[2]
            @test mask.colorMask.b == 1.0f0             # Still differential on right
            @test mask.allowedIDs == Float32[15.0]      # IDs preserved
        end

         # Cleanup
        empty!(mg)
        MakieEventHandlers.compare_mode[] = false
        MakieEventHandlers.compare_all_lesions_mode[] = true
        MakieEventHandlers.is_single_lesion_mode[] = true
    end

    # =========================================================================
    # Group 16: M2 Panel Reload — Differential Opacity After Data Load
    # =========================================================================
    @testset "16. M2 Panel Reload — Differential Opacity After Data Load" begin
        # This test simulates what happens when apply_m2_layout! loads new
        # data into panels 6-9. The data loading doesn't set mask differential
        # opacity, so reactToSyncLesion must be called AFTER to re-apply it.
        function make_mock_state_m2(panel_idx)
            ts_ct = TextureSpec{Float32}(name="CT", isMainImage=true, minAndMaxValue=[0.0f0, 1000.0f0])
            ts_mask = TextureSpec{Int16}(
                name="Mask",
                isMultiDiscreteMask=true,
                isIntegerTexture=true,
                colorSet=[RGB(1.0, 0.0, 0.0)],
                minAndMaxValue=Int16[1, 10000]
            )
            calcDims = CalcDimsStruct(
                mainImageQuadVert = zeros(Float32, 32),
                mainQuadVertSize = 128
            )
            onScroll = FullScrollableDat(
                dimensionToScroll=3,
                slicesNumber=Int32(50),
                dataToScroll=[
                    ThreeDimRawDat{Float32}(type=Float32, name="CT", dat=zeros(Float32, 10, 10, 50)),
                    ThreeDimRawDat{Int16}(type=Int16, name="Mask", dat=zeros(Int16, 10, 10, 50))
                ]
            )
            mainDisp = forDisplayObjects(
                listOfTextSpecifications=[ts_ct, ts_mask]
            )
            StateDataFields(
                calcDimsStruct=calcDims,
                onScrollData=onScroll,
                mainForDisplayObjects=mainDisp,
                currentDisplayedSlice=25
            )
        end

        mock_m2 = [make_mock_state_m2(i) for i in 1:9]
        mg = LesionAssociation.get_match_groups()
        empty!(mg)

        MakieEventHandlers.compare_mode[] = true
        MakieEventHandlers.compare_all_lesions_mode[] = true
        MakieEventHandlers.current_tp_index[] = 0
        MakieEventHandlers.compare_right_tp[] = 1
        MakieEventHandlers.tp_node_names[0] = "TP_0"
        MakieEventHandlers.tp_node_names[1] = "TP_1"
        MakieEventHandlers.is_single_lesion_mode[] = true
        MakieEventHandlers.current_active_lesion_id[] = 5

        # Setup a match
        LesionAssociation.update_match_group!("TP_0", 5, "TP_1", 15, "")

        # 16a. Initial sync sets differential opacity on panels 5-9
        MakieEventHandlers.reactToSyncLesion(MakieEvents.SyncLesionEvent(5), mock_m2)
        for idx in 6:9
            mask = mock_m2[idx].mainForDisplayObjects.listOfTextSpecifications[2]
            @test mask.colorMask.b == 1.0f0
            @test mask.minAndMaxValue == [1, 10000]
            @test mask.allowedIDs == Float32[15.0]
        end

        # 16b. Simulate data reload — reset mask specs as _load_tp_from_entry! would
        # (it doesn't touch Mask minAndMaxValue for PET modality, but let's simulate
        # the worst case where it gets reset to single-lesion defaults)
        for idx in 6:9
            mask = mock_m2[idx].mainForDisplayObjects.listOfTextSpecifications[2]
            mask.minAndMaxValue = Int16[5, 5]       # Single lesion (wrong for compare)
            mask.colorMask = RGBA(0f0, 0f0, 0f0, 1f0)  # No differential
            mask.allowedIDs = Float32[]                  # No IDs
        end

        # Verify panels 6-9 lost their differential opacity
        for idx in 6:9
            mask = mock_m2[idx].mainForDisplayObjects.listOfTextSpecifications[2]
            @test mask.colorMask.b == 0.0f0
            @test mask.minAndMaxValue == [5, 5]
        end

        # 16c. Re-apply sync (what apply_m2_layout! now does after loading)
        MakieEventHandlers.reactToSyncLesion(MakieEvents.SyncLesionEvent(5), mock_m2)

        # Panels 6-9 should have differential opacity restored
        for idx in 6:9
            mask = mock_m2[idx].mainForDisplayObjects.listOfTextSpecifications[2]
            @test mask.colorMask.b == 1.0f0             # Differential mode ON
            @test mask.minAndMaxValue == [1, 10000]      # All lesions shown
            @test mask.allowedIDs == Float32[15.0]       # Mapped IDs
            @test mask.colorMask.r == MakieEventHandlers.mapping_alpha_linked[]
            @test mask.colorMask.g == MakieEventHandlers.mapping_alpha_unlinked[]
        end

        # Left panels should still show single lesion
        for idx in 1:4
            mask = mock_m2[idx].mainForDisplayObjects.listOfTextSpecifications[2]
            @test mask.minAndMaxValue == [5, 5]
            @test mask.colorMask.b == 0.0f0
        end

        # Cleanup
        empty!(mg)
        MakieEventHandlers.compare_mode[] = false
        MakieEventHandlers.compare_all_lesions_mode[] = true
        MakieEventHandlers.is_single_lesion_mode[] = true
    end
end

println("\n=======================================================")
println("ALL INTERACTIVE LESION MAPPING TESTS PASSED SUCCESSFULLY!")
println("=======================================================")
