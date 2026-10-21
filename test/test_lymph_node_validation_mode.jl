using Test
using MedEye3d
using HDF5
using JSON
using GLMakie

const MEH = MedEye3d.SegmentationDisplay.MakieEventHandlers
const LMW = MedEye3d.LesionMetadataWindow
const LA  = MedEye3d.LesionAssociation
const ME  = MedEye3d.MakieEvents
const DataStructs = MedEye3d.ForDisplayStructs
const StructsManag = MedEye3d.StructsManag
const RMCD = MedEye3d.ReactOnMouseClickAndDrag

@testset "Lymph Node Validation Mode & HDF5 Persistence" begin
    tmp_dir = mktempdir()
    test_h5 = joinpath(tmp_dir, "test_ln_validation.h5")

    sz = (32, 32, 32)
    ct_vol   = zeros(Float32, sz)
    pet_vol  = fill(1.5f0, sz)
    mask_vol = zeros(Int16, sz)
    anat_vol = zeros(UInt16, sz)

    # Create 2 lesions in mask_vol (IDs 1 and 2)
    mask_vol[5:7, 5:7, 5:7] .= Int16(1)
    mask_vol[20:22, 20:22, 20:22] .= Int16(2)

    # Create 3 lymph node areas in anat_vol (IDs 1, 2, 3) and 1 non-LN organ (ID 100 = liver)
    anat_vol[8:10, 8:10, 8:10] .= UInt16(1)
    pet_vol[8:10, 8:10, 8:10] .= 6.5f0
    anat_vol[14:16, 14:16, 14:16] .= UInt16(2)
    pet_vol[14:16, 14:16, 14:16] .= 9.0f0
    anat_vol[22:24, 22:24, 22:24] .= UInt16(3)
    pet_vol[22:24, 22:24, 22:24] .= 3.2f0
    anat_vol[26:28, 26:28, 26:28] .= UInt16(100) # liver (not a lymph node)

    anat_labels = Dict(
        "1" => "LN_Inguinal_Superficial_L",
        "2" => "LN_External_Iliac_R",
        "3" => "LN_Obturator_L",
        "100" => "liver"
    )
    init_lesion_annotations = Dict(
        "1" => Dict("ObservationState" => "ACCEPTED", "Comment" => "Lesion 1 note"),
        "2" => Dict("ObservationState" => "UNREVIEWED", "Comment" => "Lesion 2 note")
    )

    HDF5.h5open(test_h5, "w") do f
        meta = HDF5.create_group(f, "_meta_")
        meta["preflipped"] = Int32(1)
        meta["max_anatomy_labels.json"] = JSON.json(anat_labels)
        meta["lesion_annotations.json"] = JSON.json(init_lesion_annotations)

        base = HDF5.create_group(f, "BASELINE")
        base["ct.nii.gz"] = ct_vol
        base["pet.nii.gz"] = pet_vol
        base["mask.nii.gz"] = mask_vol
        base["max_anatomy.nii.gz"] = anat_vol

        atlas = HDF5.create_group(f, "ATLAS")
        atlas["max_anatomy"] = anat_vol
    end

    # Populate MEH state
    MEH.h5_path_ref[] = test_h5
    MEH.h5_save_path_ref[] = test_h5
    studies_list = [("PET", 0, "20250101", "ct.nii.gz", "pet.nii.gz", "mask.nii.gz", "BASE_NODE", "", "", "", "", "")]
    MEH.studies_ref[] = studies_list
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
    MEH.pet_volumes_cache[0] = copy(pet_vol)
    MEH.precompute_mask_centroids!(entry0.mask_i16, 0, "BASE_NODE")
    MEH.precompute_anatomy_ln_centroids!(entry0.anat_i16, 0, "BASE_NODE")

    # Build 4 stateObjects (Axial, PET, Sagittal, Coronal)
    function make_panel_state(dim::Int, p_idx::Int)
        st = DataStructs.StateDataFields()
        st.switchIndex = p_idx
        ts_ct = DataStructs.TextureSpec{Float32}(name="CT", isMainImage=true, minAndMaxValue=Float32[-150, 250])
        ts_pet = DataStructs.TextureSpec{Float32}(name="PET", isMainImage=false, isNuclearMask=true, minAndMaxValue=Float32[0.1, 10.0])
        ts_mask = DataStructs.TextureSpec{Int16}(name="Mask", isMainImage=false, isMultiDiscreteMask=true, isIntegerTexture=true, isEditable=true, isVisible=true, minAndMaxValue=Int16[1, 100])
        ts_bone = DataStructs.TextureSpec{Int8}(name="Bone_Overlay", isMainImage=false, isIntegerTexture=true, isVisible=false, minAndMaxValue=Int8[0, 3])
        ts_anat = DataStructs.TextureSpec{Int16}(name="Anatomy", isMainImage=false, isMultiDiscreteMask=true, isIntegerTexture=true, isEditable=true, isVisible=false, minAndMaxValue=Int16[0, 400])
        st.mainForDisplayObjects.listOfTextSpecifications = DataStructs.TextureSpec[ts_ct, ts_pet, ts_mask, ts_bone, ts_anat]
        st.textureToModifyVec = DataStructs.TextureSpec[ts_mask]

        voxels = if p_idx == 3
            Any[("CT", PermutedDimsArray(entry0.ct, (2,3,1))), ("PET", PermutedDimsArray(entry0.pet, (2,3,1))), ("Mask", PermutedDimsArray(entry0.mask_i16, (2,3,1))), ("Bone_Overlay", PermutedDimsArray(entry0.bone_mask, (2,3,1))), ("Anatomy", PermutedDimsArray(entry0.anat_i16, (2,3,1)))]
        elseif p_idx == 4
            Any[("CT", PermutedDimsArray(entry0.ct, (1,3,2))), ("PET", PermutedDimsArray(entry0.pet, (1,3,2))), ("Mask", PermutedDimsArray(entry0.mask_i16, (1,3,2))), ("Bone_Overlay", PermutedDimsArray(entry0.bone_mask, (1,3,2))), ("Anatomy", PermutedDimsArray(entry0.anat_i16, (1,3,2)))]
        else
            Any[("CT", entry0.ct), ("PET", entry0.pet), ("Mask", entry0.mask_i16), ("Bone_Overlay", entry0.bone_mask), ("Anatomy", entry0.anat_i16)]
        end
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
        st.calcDimsStruct.mainQuadVertSize = Int32(16)
        st.calcDimsStruct.mainImageQuadVert = Float32[-1, -1, 1, 1]
        return st
    end

    stateObjects = [make_panel_state(3, i) for i in 1:4]
    MEH._main_obj_ref[] = stateObjects

    # Create GUI panel via create_metadata_window
    ch = Channel{Any}(128)
    active_lesion_obs = Observable("1: Lesion 1")
    lesion_ids_obs = Observable(["1: Lesion 1", "2: Lesion 2"])
    res = LMW.create_metadata_window(active_lesion_obs, lesion_ids_obs, ch; save_path = joinpath(tmp_dir, "companion.json"))
    obs = LMW._lmw_observables

    # 1. Verify Default Mode is OFF (Normal Lesion Mode)
    @test MEH.lymph_node_validation_mode[] == false
    @test obs[:btn_ln_validation].label[] == "LN Validation Mode: OFF"
    @test length(obs[:lesion_ids][]) == 2
    @test LMW.parse_lesion_id(obs[:lesion_ids][][1]) == 1
    @test LMW.parse_lesion_id(obs[:lesion_ids][][2]) == 2

    # Helper to drain channel events into stateObjects
    function drain_events!()
        while isready(ch)
            ev = take!(ch)
            if ev isa ME.SetLNValidationModeEvent
                MEH.reactToSetLNValidationMode(ev, stateObjects)
            elseif ev isa ME.SyncLesionEvent
                MEH.reactToSyncLesion(ev, stateObjects)
            elseif ev isa ME.ShowSingleLesionEvent
                MEH.reactToShowSingleLesion(ev, stateObjects)
            elseif ev isa ME.PaintValEvent
                MEH.reactToPaintVal(ev, stateObjects)
            elseif ev isa ME.RevertToAIEvent
                MEH.reactToRevertToAI(ev, stateObjects)
            end
        end
    end

    # 2. Click LN Validation Mode button to toggle ON
    obs[:btn_ln_validation].clicks[] += 1
    drain_events!()

    @test MEH.lymph_node_validation_mode[] == true
    @test obs[:btn_ln_validation].label[] == "LN Validation Mode: ON"

    # 3. Verify dropdown now iterates ALL lymph node areas (1, 2, 3) and excludes non-LN organ (100 = liver)
    ln_opts = obs[:lesion_ids][]
    @test length(ln_opts) == 3
    @test [LMW.parse_lesion_id(o) for o in ln_opts] == [1, 2, 3]
    @test occursin("Inguinal", ln_opts[1])
    @test occursin("External Iliac", ln_opts[2])
    @test occursin("Obturator", ln_opts[3])

    # Verify 3D viewer synchronized to LN area 1 (centroid (9, 9, 9)) and Anatomy texture is visible & isolated to ID 1
    anat_ts = first(filter(t -> t.name == "Anatomy", stateObjects[1].mainForDisplayObjects.listOfTextSpecifications))
    mask_ts = first(filter(t -> t.name == "Mask", stateObjects[1].mainForDisplayObjects.listOfTextSpecifications))
    @test anat_ts.isVisible == true
    @test mask_ts.isVisible == false
    @test anat_ts.minAndMaxValue == Int16[1, 1]
    @test stateObjects[1].currentDisplayedSlice == 9

    # 4. Accept LN area 1, advance to LN area 2, Reject LN area 2, advance to LN area 3, Correct LN area 3
    obs[:btn_accept].clicks[] += 1
    drain_events!()
    @test get(get(obs[:lesion_db][], "1", Dict()), "ObservationState", "") == "ACCEPTED"

    # Ensure we are on LN area 2 (auto-advance or explicit next)
    if LMW.parse_lesion_id(obs[:active_lesion_id][]) != 2
        obs[:btn_next].clicks[] += 1
        drain_events!()
    end
    @test LMW.parse_lesion_id(obs[:active_lesion_id][]) == 2
    @test anat_ts.minAndMaxValue == Int16[2, 2]
    @test stateObjects[1].currentDisplayedSlice == 15

    obs[:btn_reject].clicks[] += 1
    drain_events!()
    @test get(get(obs[:lesion_db][], "2", Dict()), "ObservationState", "") == "REJECTED"

    if LMW.parse_lesion_id(obs[:active_lesion_id][]) != 3
        obs[:btn_next].clicks[] += 1
        drain_events!()
    end
    @test LMW.parse_lesion_id(obs[:active_lesion_id][]) == 3
    @test anat_ts.minAndMaxValue == Int16[3, 3]
    @test stateObjects[1].currentDisplayedSlice == 23

    # 5. Manual segmentation editing (Paint & Erase) on LN area 3 in LN Validation Mode
    MEH.reactToPaintVal(ME.PaintValEvent(3, true), stateObjects)
    @test stateObjects[1].textureToModifyVec[1].name == "Anatomy"

    orig_count_ln3 = count(==(Int16(3)), entry0.anat_i16)
    @test orig_count_ln3 == 27

    # Paint new voxels on slice 23 via react_to_draw
    ms_paint = DataStructs.MouseStruct(
        isLeftButtonDown = true,
        lastCoordinates = [CartesianIndex(18, 14)],
        actualWindowWidth = 32,
        actualWindowHeight = 32,
        window_id = 1
    )
    RMCD.react_to_draw([ms_paint], stateObjects)
    @test count(==(Int16(3)), entry0.anat_i16) > orig_count_ln3
    @test entry0.anat_i16[18, 18, 23] == Int16(3)
    @test entry0.anatomy[18, 18, 23] == UInt16(3)
    @test 0 in MEH.dirty_anatomy_tps

    # Flush dirty anatomy to HDF5 and verify persistence in BASELINE/max_anatomy.nii.gz_expert and ATLAS/max_anatomy_expert
    MEH.flush_all_dirty_masks!()
    @test isempty(MEH.dirty_anatomy_tps)

    HDF5.h5open(test_h5, "r") do f
        @test haskey(f, "BASELINE/max_anatomy.nii.gz_expert")
        @test haskey(f, "ATLAS/max_anatomy_expert")
        saved_anat = read(f["BASELINE/max_anatomy.nii.gz_expert"])
        @test saved_anat[18, 18, 23] == UInt16(3)
    end

    # Test Revert to AI on LN area 3
    MEH.reactToRevertToAI(ME.RevertToAIEvent(3, 0), stateObjects)
    @test count(==(Int16(3)), entry0.anat_i16) == orig_count_ln3
    @test entry0.anat_i16[18, 18, 23] == Int16(0)

    # Mark LN area 3 as CORRECTED and save validations to HDF5
    obs[:btn_correct].clicks[] += 1
    drain_events!()
    LMW.save_ln_validations(obs[:lesion_db][], test_h5)

    # Verify HDF5 contains _meta_/lymph_node_validations.json with ACCEPTED, REJECTED, CORRECTED
    # AND _meta_/lesion_annotations.json is untouched!
    HDF5.h5open(test_h5, "r") do f
        @test haskey(f, "_meta_/lymph_node_validations.json")
        ln_saved = JSON.parse(String(read(f["_meta_/lymph_node_validations.json"])))
        @test get(get(ln_saved, "1", Dict()), "ObservationState", "") == "ACCEPTED"
        @test get(get(ln_saved, "2", Dict()), "ObservationState", "") == "REJECTED"
        @test get(get(ln_saved, "3", Dict()), "ObservationState", "") == "CORRECTED"

        les_saved = JSON.parse(String(read(f["_meta_/lesion_annotations.json"])))
        @test get(get(les_saved, "1", Dict()), "Comment", "") == "Lesion 1 note"
        @test get(get(les_saved, "2", Dict()), "ObservationState", "") == "UNREVIEWED"
    end

    # 6. Toggle LN Validation Mode OFF and verify restoration of normal Lesion Mode
    obs[:btn_ln_validation].clicks[] += 1
    drain_events!()

    @test MEH.lymph_node_validation_mode[] == false
    @test obs[:btn_ln_validation].label[] == "LN Validation Mode: OFF"
    @test [LMW.parse_lesion_id(o) for o in obs[:lesion_ids][]] == [1, 2]
    @test anat_ts.isVisible == false
    @test mask_ts.isVisible == true
    @test get(get(obs[:lesion_db][], "1", Dict()), "Comment", "") == "Lesion 1 note"
end

ccall(:_exit, Cvoid, (Cint,), 0)
