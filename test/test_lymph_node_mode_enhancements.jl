using Test
using MedEye3d
using HDF5
using JSON

const MEH = MedEye3d.SegmentationDisplay.MakieEventHandlers
const LA  = MedEye3d.LesionAssociation
const ME  = MedEye3d.MakieEvents
const DataStructs = MedEye3d.ForDisplayStructs
const StructsManag = MedEye3d.StructsManag
const RMCD = MedEye3d.ReactOnMouseClickAndDrag

@testset "Lymph Node Mode Enhancements: M-Key Visibility, Hover Readout, and Autosave" begin
    tmp_dir = mktempdir()
    test_h5 = joinpath(tmp_dir, "test_ln_enhancements.h5")

    sz = (32, 32, 32)
    ct_vol   = zeros(Float32, sz)
    pet_vol  = fill(1.0f0, sz)
    mask_vol = zeros(Int16, sz)
    anat_vol = zeros(UInt16, sz)

    # Place 2 lesions in mask_vol (IDs 1 and 2)
    mask_vol[4:6, 4:6, 4:6] .= Int16(1)
    mask_vol[10:12, 10:12, 10:12] .= Int16(2)

    # Place 3 lymph node areas in anat_vol (IDs 5, 6, 7)
    anat_vol[15:18, 15:18, 15:18] .= UInt16(5)  # LN 5
    anat_vol[20:23, 20:23, 20:23] .= UInt16(6)  # LN 6
    anat_vol[25:27, 25:27, 25:27] .= UInt16(7)  # LN 7
    # Place 1 non-LN organ in anat_vol (ID 120 = kidney_left)
    anat_vol[2:4, 2:4, 2:4] .= UInt16(120)

    anat_labels = Dict(
        "5" => "Abdominal_Obturator_Left",
        "6" => "Abdominal_Obturator_Right",
        "7" => "Thoracic_Station_2_UpperParatracheal_Right",
        "120" => "kidney_left"
    )

    HDF5.h5open(test_h5, "w") do f
        meta = HDF5.create_group(f, "_meta_")
        meta["preflipped"] = Int32(1)
        meta["max_anatomy_labels.json"] = JSON.json(anat_labels)

        base = HDF5.create_group(f, "BASELINE")
        base["ct.nii.gz"] = ct_vol
        base["pet.nii.gz"] = pet_vol
        base["mask.nii.gz"] = mask_vol
        base["max_anatomy.nii.gz"] = anat_vol

        atlas = HDF5.create_group(f, "ATLAS")
        atlas["max_anatomy"] = anat_vol
    end

    # Setup MEH global state
    MEH._mask_autosave_task_started[] = true # Suppress concurrent autosave task in headless unit test
    MEH.h5_path_ref[] = test_h5
    MEH.h5_save_path_ref[] = test_h5
    studies_list = [("PET", 0, "20250101", "ct.nii.gz", "pet.nii.gz", "mask.nii.gz", "PET_Lesions_0", "", "", "", "", "")]
    MEH.studies_ref[] = studies_list
    MEH.tp_node_names[0] = "PET_Lesions_0"
    empty!(MEH.dirty_anatomy_tps)
    empty!(MEH.dirty_mask_tps)
    empty!(MEH.tp_data_cache)
    MEH.current_tp_index[] = 0
    MEH.volume_z_size[] = sz[3]

    ts_names_dict = Dict{Int, String}(parse(Int, k) => v for (k, v) in anat_labels)
    MEH.global_ts_names[] = ts_names_dict
    MEH.anatomy_labels_cache[0] = ts_names_dict
    MEH.global_ts_atlas[] = copy(anat_vol)

    entry0 = MEH.TpCacheEntry(
        copy(ct_vol),
        copy(pet_vol),
        copy(mask_vol),
        zeros(Int8, sz),
        copy(anat_vol),
        copy(mask_vol),
        Int16.(anat_vol)
    )
    lock(MEH._tp_cache_lock) do
        MEH.tp_data_cache[0] = entry0
    end
    MEH.precompute_anatomy_ln_centroids!(entry0.anat_i16, 0, "PET_Lesions_0")

    # Build mock panel state
    function make_mock_panel(dim::Int, p_idx::Int)
        st = DataStructs.StateDataFields()
        st.switchIndex = p_idx
        ts_ct = DataStructs.TextureSpec{Float32}(name="CT", isMainImage=true, minAndMaxValue=Float32[-150, 250])
        ts_pet = DataStructs.TextureSpec{Float32}(name="PET", isMainImage=false, minAndMaxValue=Float32[0.1, 10.0])
        ts_mask = DataStructs.TextureSpec{Int16}(name="Mask", isMainImage=false, isMultiDiscreteMask=true, isIntegerTexture=true, isEditable=true, isVisible=true, minAndMaxValue=Int16[1, 100])
        ts_bone = DataStructs.TextureSpec{Int8}(name="Bone_Overlay", isMainImage=false, isIntegerTexture=true, isVisible=false, minAndMaxValue=Int8[0, 3])
        ts_anat = DataStructs.TextureSpec{Int16}(name="Anatomy", isMainImage=false, isMultiDiscreteMask=true, isIntegerTexture=true, isEditable=true, isVisible=false, minAndMaxValue=Int16[0, 400])
        st.mainForDisplayObjects.listOfTextSpecifications = DataStructs.TextureSpec[ts_ct, ts_pet, ts_mask, ts_bone, ts_anat]
        st.textureToModifyVec = DataStructs.TextureSpec[ts_mask]

        voxels = Any[("CT", entry0.ct), ("PET", entry0.pet), ("Mask", entry0.mask_i16), ("Bone_Overlay", entry0.bone_mask), ("Anatomy", entry0.anat_i16)]
        d2s = StructsManag.getThreeDims(voxels)
        st.onScrollData.dataToScroll = d2s
        st.onScrollData.nameIndexes = DataStructs.getLocationDict(d2s)
        st.onScrollData.dimensionToScroll = 3
        st.onScrollData.slicesNumber = Int32(sz[3])
        st.currentDisplayedSlice = 16
        twoDimList = map(tdd -> StructsManag.threeToTwoDimm(tdd.type, Int64(16), 3, tdd), d2s)
        st.currentlyDispDat = DataStructs.SingleSliceDat(listOfDataAndImageNames=twoDimList, sliceNumber=16)
        st.calcDimsStruct.imageTextureWidth = Int32(sz[1])
        st.calcDimsStruct.imageTextureHeight = Int32(sz[2])
        st.calcDimsStruct.windowWidth = Int32(800)
        st.calcDimsStruct.windowHeight = Int32(600)
        st.calcDimsStruct.fractionOfMainIm = 1.0f0
        st.calcDimsStruct.heightToWithRatio = 1.0f0
        st.calcDimsStruct = StructsManag.getMainVerticies(st.calcDimsStruct, DataStructs.QuadImage, p_idx)
        return st
    end

    stateObjects = [make_mock_panel(3, i) for i in 1:4]
    MEH._main_obj_ref[] = stateObjects[1].mainForDisplayObjects

    # ──────────────────────────────────────────────────────────────────────────
    # PART 1: 'm' Key Toggling in Lymph Node Mode
    # ──────────────────────────────────────────────────────────────────────────
    @testset "Part 1: 'm' Key Toggling in Lymph Node Mode" begin
        # Activate LN mode on LN 5
        MEH.current_active_lesion_id[] = 5
        MEH.reactToSetLNValidationMode(ME.SetLNValidationModeEvent(true), stateObjects)

        @test MEH.lymph_node_validation_mode[] == true
        @test MEH.is_single_lesion_mode[] == true

        # In single lesion mode, Anatomy must display ONLY active LN 5
        anat_ts = first(filter(t -> t.name == "Anatomy", stateObjects[1].mainForDisplayObjects.listOfTextSpecifications))
        mask_ts = first(filter(t -> t.name == "Mask", stateObjects[1].mainForDisplayObjects.listOfTextSpecifications))
        @test anat_ts.isVisible == true
        @test mask_ts.isVisible == false
        @test anat_ts.minAndMaxValue == Int16[5, 5]

        # Verify target texture for editing is Anatomy
        @test stateObjects[1].textureToModifyVec[1].name == "Anatomy"

        # Now press 'm' (ToggleSingleMultiLesionEvent)
        MEH.reactToToggleSingleMultiLesion(ME.ToggleSingleMultiLesionEvent(), stateObjects)

        # After pressing 'm', is_single_lesion_mode must be FALSE (All LN areas mode)
        @test MEH.is_single_lesion_mode[] == false

        # In all LN mode, minAndMaxValue must encompass ONLY the lymph node label range (5..7),
        # and MUST NOT be forced to 400!
        @test anat_ts.minAndMaxValue == Int16[5, 7]
        # Verify that kidney_left (ID 120) is EXCLUDED by this range
        @test 120 > anat_ts.minAndMaxValue[2]

        # Press 'm' again to return to single LN mode
        MEH.reactToToggleSingleMultiLesion(ME.ToggleSingleMultiLesionEvent(), stateObjects)
        @test MEH.is_single_lesion_mode[] == true
        @test anat_ts.minAndMaxValue == Int16[5, 5]
    end

    # ──────────────────────────────────────────────────────────────────────────
    # PART 2: Top Panel Hover Readout in Lymph Node Mode
    # ──────────────────────────────────────────────────────────────────────────
    @testset "Part 2: Top Panel Cursor Info Readout in LN Mode" begin
        # Test simulated cursor hovering over LN 5 voxel (16, 16, 16)
        panelState = stateObjects[1]
        panelState.currentDisplayedSlice = 16

        mousestr_ln5 = DataStructs.MouseStruct(
            isLeftButtonDown=false,
            isRightButtonDown=false,
            actualWindowWidth=800,
            actualWindowHeight=600,
            lastCoordinates=[CartesianIndex(200, 150)]
        )

        RMCD._last_cursor_readout_time[] = 0.0 # Force readout without throttle
        RMCD.reactToMouseDrag(mousestr_ln5, stateObjects)

        info = MEH.cursor_info_text[]
        @test occursin("Obturator Left", info) || occursin("Obturator", info)
        @test occursin("LN 5", info)
        # Normal mask lesion should NOT appear in LN mode
        @test !occursin("Lesion 1", info)
        @test !occursin("Lesion 2", info)

        # In single LN mode (active=5), hovering over LN 6 voxel at (22, 22, 22)
        # should NOT show LN 6
        panelState.currentDisplayedSlice = 22
        mousestr_ln6 = DataStructs.MouseStruct(
            isLeftButtonDown=false,
            isRightButtonDown=false,
            actualWindowWidth=800,
            actualWindowHeight=600,
            lastCoordinates=[CartesianIndex(240, 90)]
        )
        RMCD._last_cursor_readout_time[] = 0.0
        RMCD.reactToMouseDrag(mousestr_ln6, stateObjects)
        info_single = MEH.cursor_info_text[]
        @test !occursin("LN 6", info_single)

        # Now toggle 'm' to All LNs mode
        MEH.reactToToggleSingleMultiLesion(ME.ToggleSingleMultiLesionEvent(), stateObjects)
        @test MEH.is_single_lesion_mode[] == false

        # Now hover over LN 6 voxel again in All LNs mode
        RMCD._last_cursor_readout_time[] = 0.0
        RMCD.reactToMouseDrag(mousestr_ln6, stateObjects)
        info_multi = MEH.cursor_info_text[]
        # In All LNs mode, LN 6 IS visible and its name MUST appear!
        @test occursin("Obturator Right", info_multi) || occursin("Obturator", info_multi)
        @test occursin("LN 6", info_multi)

        # Reset back to single mode
        MEH.reactToToggleSingleMultiLesion(ME.ToggleSingleMultiLesionEvent(), stateObjects)
    end

    # ──────────────────────────────────────────────────────────────────────────
    # PART 3: Autosave of Lymph Node Area Mask Modifications
    # ──────────────────────────────────────────────────────────────────────────
    @testset "Part 3: Autosave of Lymph Node Area Mask Modifications" begin
        # 1. Modify Anatomy voxels (simulate brush painting on LN 5)
        # Expand LN 5 into voxels (14, 14, 16) and (14, 15, 16)
        panelState = stateObjects[1]
        entry0.anat_i16[14, 14, 16] = Int16(5)
        entry0.anat_i16[14, 15, 16] = Int16(5)
        entry0.anatomy[14, 14, 16] = UInt16(5)
        entry0.anatomy[14, 15, 16] = UInt16(5)

        # Trigger dirty marking as done by brush paint handler
        MEH.mark_tp_anatomy_dirty!(0)

        # Verify that mark_tp_anatomy_dirty! marked TP 0 as dirty
        @test 0 in MEH.dirty_anatomy_tps
        @test entry0.anat_i16[14, 14, 16] == Int16(5)
        @test entry0.anat_i16[14, 15, 16] == Int16(5)

        # Trigger flush
        success = MEH.save_tp_anatomy_to_h5(0)
        @test success == true
        @test !(0 in MEH.dirty_anatomy_tps)

        # Verify persistence to HDF5 file
        HDF5.h5open(test_h5, "r") do f
            @test haskey(f, "BASELINE/max_anatomy.nii.gz_expert")
            saved_anat = read(f["BASELINE/max_anatomy.nii.gz_expert"])
            @test saved_anat[14, 14, 16] == UInt16(5)
            @test saved_anat[14, 15, 16] == UInt16(5)

            # Check ATLAS/max_anatomy_expert was also updated for TP 0
            @test haskey(f, "ATLAS/max_anatomy_expert")
            atlas_anat = read(f["ATLAS/max_anatomy_expert"])
            @test atlas_anat[14, 14, 16] == UInt16(5)
            @test atlas_anat[14, 15, 16] == UInt16(5)

            # Check CENTROIDS group has updated coordinates
            @test haskey(f, "CENTROIDS/tp0_ln5")
            c5 = read(f["CENTROIDS/tp0_ln5"])
            @test length(c5) == 3
            @test c5[1] > 0 && c5[2] > 0 && c5[3] > 0
        end

        # 2. Test deleting / clearing an LN area in LN mode
        # Delete LN 7
        MEH.reactToDeleteLesion(ME.DeleteLesionEvent(7), stateObjects)

        # Verify that LN 7 voxels were zeroed in entry0.anat_i16 and marked dirty
        @test count(==(Int16(7)), entry0.anat_i16) == 0
        @test 0 in MEH.dirty_anatomy_tps

        # Flush deletion to HDF5
        MEH.save_tp_anatomy_to_h5(0)
        HDF5.h5open(test_h5, "r") do f
            saved_anat = read(f["BASELINE/max_anatomy.nii.gz_expert"])
            @test count(==(UInt16(7)), saved_anat) == 0
        end

        # 3. Test HeatGDT commit in LN mode
        # Simulate HeatGDT stop committing active LN 6 (ID 6)
        MEH.current_active_lesion_id[] = 6
        MEH._vk_heat_active[] = true
        MEH._vk_heat_start_time[] = time() - 1.0

        # Create mock heat_state
        mock_heat_state = (
            current_box_min = (19, 19, 19),
            current_box_max = (24, 24, 24),
            cpu_box_buffer = fill(0.05f0, 6, 6, 6) # Above commit_theta = 0.001
        )
        MEH._vk_heat_state[] = mock_heat_state
        mock_vk_ctx = (staging_pool = nothing,)
        MEH._vk_heat_ctx[] = mock_vk_ctx

        VHD = MEH.VulkanHeatDiffusion
        @eval VHD begin
            run_incremental_steps!(hs::NamedTuple, ctx::NamedTuple, steps::Int; kwargs...) = nothing
            apply_box_mask_to_segvol!(hs::NamedTuple, seg_vol, label_val; kwargs...) = begin
                seg_vol[19:24, 19:24, 19:24] .= label_val
                return 36
            end
        end

        # Trigger reactToHeatGDTStop
        MEH.reactToHeatGDTStop(ME.HeatGDTStopEvent(1), stateObjects)

        # Verify Anatomy in tp_cache received label 6 and was marked dirty
        @test entry0.anat_i16[19, 19, 19] == Int16(6)
        @test 0 in MEH.dirty_anatomy_tps

        # Flush to HDF5
        MEH.save_tp_anatomy_to_h5(0)
        HDF5.h5open(test_h5, "r") do f
            saved_anat = read(f["BASELINE/max_anatomy.nii.gz_expert"])
            @test saved_anat[19, 19, 19] == UInt16(6)
        end
        empty!(MEH._async_suv_pending)
    end
end
sleep(0.6)
println("✅ ALL LYMPH NODE ENHANCEMENT TESTS PASSED!")
