using Test
using GLMakie
using Colors: RGB
using MedEye3d
import MedEye3d.LesionMetadataWindow as LMW
import MedEye3d.SegmentationDisplay.MakieEventHandlers as MEH
import MedEye3d.MakieEvents

@testset "Up/Down Arrow Lesion Navigation & Dropdown Preservation" begin
    # Configure MEH state with 25 lesions at TP 0 (mimicking Pat3 + newly created Lesion 25)
    MEH.current_tp_index[] = 0
    MEH.volume_z_size[] = 326
    empty!(LMW._cached_lesion_ids)
    empty!(MEH.tp_segment_names)
    lock(MEH._tp_cache_lock) do
        empty!(MEH.tp_data_cache)
    end
    lock(MEH._centroids_lock) do
        empty!(MEH.lesion_centroids_cache)
    end

    seg_names = Dict{Int, String}()
    for i in 1:24
        seg_names[i] = i == 5 ? "prostate" : (i == 24 ? "sartorius_right" : "femur_right")
        MEH.global_organ_mapping[][i] = seg_names[i]
    end
    seg_names[25] = "Thigh Adductor Compartment"
    MEH.global_organ_mapping[][25] = "Thigh Adductor Compartment"
    MEH.tp_segment_names[0] = seg_names

    initial_opts = String[]
    for i in 1:25
        if i == 1
            push!(initial_opts, "1: Femur Right")
        elseif i == 5
            # Deliberately generic name to trigger apply_state auto-rename
            push!(initial_opts, "5: Lesion 5")
        elseif i == 24
            push!(initial_opts, "24: Sartorius Right")
        elseif i == 25
            push!(initial_opts, "25: New Lesion")
        else
            push!(initial_opts, "$i: Femur Right ($i)")
        end
    end

    active_lesion = Observable(initial_opts[1])
    lesion_ids = Observable(copy(initial_opts))

    save_json_path = tempname() * ".json"
    win = LMW.create_metadata_window(active_lesion, lesion_ids, nothing; save_path = save_json_path)

    # Populate cache AFTER create_metadata_window (which clears _cached_lesion_ids)
    LMW._cached_lesion_ids[0] = collect(1:25)

    # Allow async DB init and GUI preload to complete
    t0 = time()
    while !get(LMW._lmw_observables, :gui_preload_done, Ref(false))[] && time() - t0 < 20.0
        sleep(0.1)
    end
    sleep(0.5)

    obs_next = LMW._lmw_observables[:obs_next_lesion]
    obs_prev = LMW._lmw_observables[:obs_prev_lesion]
    lesion_db = LMW._lmw_observables[:lesion_db]

    # ── Test 1: Updating lesion_db with only keys "1" and "25" must NOT strip 2..24 ──
    partial_db = Dict{String, Any}(
        "1" => Dict{String, Any}("_display_name" => "1: Femur Right", "BaseAnatomy" => "femur_right"),
        "25" => Dict{String, Any}("_display_name" => "25: Thigh Adductor Compartment", "BaseAnatomy" => "Thigh Adductor Compartment"),
    )
    lesion_db[] = partial_db
    @test length(lesion_ids[]) == 25
    parsed_all = [LMW.parse_lesion_id(o) for o in lesion_ids[]]
    @test parsed_all == collect(1:25)

    # ── Test 2: Stepping Up/Down traverses all 25 lesions sequentially without sticking ──
    active_lesion[] = lesion_ids[][25]
    sleep(0.1)
    visited_prev = Int[]
    for step in 1:25
        obs_prev[] = obs_prev[] + 1
        push!(visited_prev, LMW.parse_lesion_id(active_lesion[]))
    end
    @test visited_prev == vcat(collect(24:-1:1), [25])

    # Verify auto-rename on generic Lesion 5 kept active_lesion[] synced with lesion_ids[][5]
    active_lesion[] = lesion_ids[][4]
    obs_next[] = obs_next[] + 1
    @test LMW.parse_lesion_id(active_lesion[]) == 5
    @test active_lesion[] in lesion_ids[]
    obs_next[] = obs_next[] + 1
    @test LMW.parse_lesion_id(active_lesion[]) == 6

    # Rapid forward traversal 1 -> 2 -> ... -> 25 -> 1 via obs_next
    active_lesion[] = lesion_ids[][1]
    visited_next = Int[]
    for step in 1:25
        obs_next[] = obs_next[] + 1
        push!(visited_next, LMW.parse_lesion_id(active_lesion[]))
    end
    @test visited_next == vcat(collect(2:25), [1])

    # ── Test 3: Self-healing fallback when no 3D mask is in tp_data_cache ──
    lesion_ids[] = ["1: Femur Right", "25: Thigh Adductor Compartment"]
    active_lesion[] = "25: Thigh Adductor Compartment"
    obs_prev[] = obs_prev[] + 1
    @test length(lesion_ids[]) == 25
    @test LMW.parse_lesion_id(active_lesion[]) == 24

    # ── Test 4: Ghost lesion pruning against active 3D mask in tp_data_cache ──
    mask_3d = zeros(Int16, 300, 300, 100)
    mask_3d[249, 299, 52] = Int16(16)
    mask_3d[258, 289, 65] = Int16(24)
    mask_3d[223, 277, 90] = Int16(26)
    empty_f32 = zeros(Float32, 1, 1, 1)
    empty_i8 = zeros(Int8, 1, 1, 1)
    lock(MEH._tp_cache_lock) do
        MEH.tp_data_cache[0] = MEH.TpCacheEntry(empty_f32, empty_f32, mask_3d, empty_i8, nothing, mask_3d, nothing)
    end
    empty!(LMW._cached_lesion_ids)

    # Inject 26 ghost+real options while on ghost Lesion 25
    ghost_opts = ["$i: Ghost $i" for i in 1:26]
    MEH.tp_segment_names[0] = Dict(i => "Ghost $i" for i in 1:26)
    lesion_ids[] = ghost_opts
    active_lesion[] = "25: Ghost 25"

    # Stepping Up from ghost 25 prunes all 0-voxel ghosts and lands strictly on 24
    obs_prev[] = obs_prev[] + 1
    @test [LMW.parse_lesion_id(o) for o in lesion_ids[]] == [16, 24, 26]
    @test LMW.parse_lesion_id(active_lesion[]) == 24

    # Continuing Up / Down cycles strictly within [16, 24, 26]
    obs_prev[] = obs_prev[] + 1
    @test LMW.parse_lesion_id(active_lesion[]) == 16
    obs_prev[] = obs_prev[] + 1
    @test LMW.parse_lesion_id(active_lesion[]) == 26
    obs_next[] = obs_next[] + 1
    @test LMW.parse_lesion_id(active_lesion[]) == 16

    # Leaving an unpainted (0-voxel) newly created lesion removes it automatically
    db_with_27 = copy(lesion_db[])
    db_with_27["27"] = Dict{String, Any}("_display_name" => "27: New Lesion")
    lesion_db[] = db_with_27
    opts_with_27 = vcat(copy(lesion_ids[]), ["27: New Lesion"])
    lesion_ids[] = opts_with_27
    active_lesion[] = "27: New Lesion"
    sleep(0.05)
    active_lesion[] = lesion_ids[][1] # leave 27 (0 voxels) -> 16
    sleep(0.05)
    @test !(27 in [LMW.parse_lesion_id(o) for o in lesion_ids[]])
    @test !haskey(lesion_db[], "27")

    # ── Test 5: reactToSyncLesion detects stale centroid and jumps to true voxel slice ──
    function make_visible_mock_state()
        ts_ct = MEH.TextureSpec{Float32}(name="CT", isMainImage=true, minAndMaxValue=[0.0f0, 1000.0f0])
        ts_mask = MEH.TextureSpec{Int16}(
            name="Mask",
            isMultiDiscreteMask=true,
            isIntegerTexture=true,
            colorSet=[RGB(1.0, 0.0, 0.0)],
            minAndMaxValue=Int16[1, 10000]
        )
        calcDims = MEH.CalcDimsStruct(
            mainImageQuadVert = ones(Float32, 32),
            mainQuadVertSize = 128
        )
        onScroll = MEH.FullScrollableDat(
            dimensionToScroll=3,
            slicesNumber=Int32(100),
            dataToScroll=[
                MEH.ThreeDimRawDat{Float32}(type=Float32, name="CT", dat=zeros(Float32, 300, 300, 100)),
                MEH.ThreeDimRawDat{Int16}(type=Int16, name="Mask", dat=mask_3d)
            ]
        )
        mainDisp = MEH.forDisplayObjects(listOfTextSpecifications=[ts_ct, ts_mask])
        MEH.StateDataFields(
            calcDimsStruct=calcDims,
            onScrollData=onScroll,
            mainForDisplayObjects=mainDisp,
            currentDisplayedSlice=1
        )
    end
    mock_states = [make_visible_mock_state() for _ in 1:4]
    MEH.compare_mode[] = false
    MEH.mri_compare_mode[] = false
    MEH.mri_main_multimodal_mode[] = false
    MEH.lymph_node_validation_mode[] = false
    MEH.tp_node_names[0] = "PET_Lesions_0"

    # Inject wrong stale centroid pointing to empty slice Z=87 for Lesion 24 (true voxel is [258, 289, 65])
    lock(MEH._centroids_lock) do
        MEH.lesion_centroids_cache[(0, 24)] = [189, 288, 87]
        MEH.lesion_centroids_cache[24] = [189, 288, 87]
        MEH.lesion_centroids_cache[(0, 25)] = [189, 288, 87]
        MEH.lesion_centroids_cache[25] = [189, 288, 87]
    end
    MEH.reactToSyncLesion(MakieEvents.SyncLesionEvent(24, true), mock_states)
    @test mock_states[1].currentDisplayedSlice == 65
    @test MEH.lesion_centroids_cache[(0, 24)] == [258, 289, 65]

    # Ghost Lesion 25 (0 voxels) must clear its stale centroid and NOT jump slice
    prev_slice = mock_states[1].currentDisplayedSlice
    MEH.reactToSyncLesion(MakieEvents.SyncLesionEvent(25, true), mock_states)
    @test mock_states[1].currentDisplayedSlice == prev_slice
    @test !haskey(MEH.lesion_centroids_cache, (0, 25))

    # ── Test 6: Multi-component lesion centroid snaps to actual dominant slice ──
    multi_vol = zeros(Int16, 20, 20, 100)
    # 15 voxels at slice Z=20, 20 voxels at slice Z=80 (arithmetic mean Z=54 is completely empty!)
    multi_vol[1:5, 1:3, 20] .= Int16(42)
    multi_vol[8:12, 10:13, 80] .= Int16(42)
    c42 = MEH.find_lesion_center(multi_vol, 42.0f0)
    @test c42 !== nothing
    @test c42[3] == 80
    @test multi_vol[c42[1], c42[2], c42[3]] == Int16(42)

    MEH.precompute_mask_centroids!(multi_vol, 0, "PET_Lesions_0")
    @test MEH.lesion_centroids_cache[(0, 42)][3] == 80
    c_cached = MEH.lesion_centroids_cache[(0, 42)]
    @test multi_vol[c_cached[1], c_cached[2], c_cached[3]] == Int16(42)

    # ── Test 7: Pressing 'c' (Compare Curr/Next TP) then 'y' (ToggleMRIMultimodalEvent) on M2 ──
    old_m2_vk = MedEye3d.SegmentationDisplay._m2_vk_ref[]
    old_m2_mode = MedEye3d.SegmentationDisplay._current_m2_mode[]
    old_tp_labels = copy(MEH.tp_labels)
    old_tp_mods = copy(MEH.tp_modalities)
    old_cmp_right = MEH.compare_right_tp[]
    try
        empty!(MEH.tp_labels)
        empty!(MEH.tp_modalities)
        MEH.tp_labels[0] = "PET BASELINE (TP 0)"; MEH.tp_modalities[0] = "PET"
        MEH.tp_labels[1] = "T2 TP 1";             MEH.tp_modalities[1] = "T2"
        MEH.tp_labels[2] = "ADC TP 2";            MEH.tp_modalities[2] = "ADC"
        MEH.tp_labels[3] = "DWI TP 3";            MEH.tp_modalities[3] = "DWI"
        MEH.tp_labels[4] = "T1 TP 4";             MEH.tp_modalities[4] = "T1"

        mri_sz = (16, 16, 100)
        ct0_vol = fill(-1000.0f0, mri_sz)
        t2_vol  = zeros(Float32, mri_sz); t2_vol[5:10, 5:10, 45:55] .= 500.0f0
        adc_vol = zeros(Float32, mri_sz); adc_vol[5:10, 5:10, 45:55] .= 1200.0f0
        dwi_vol = zeros(Float32, mri_sz); dwi_vol[5:10, 5:10, 45:55] .= 300.0f0
        t1_vol  = zeros(Float32, mri_sz); t1_vol[5:10, 5:10, 45:55] .= 400.0f0

        function make_tp_entry(vol, marker_lid)
            m16 = zeros(Int16, mri_sz)
            m16[7:8, 7:8, 50] .= Int16(marker_lid)
            a16 = zeros(Int16, mri_sz)
            a16[6:9, 6:9, 48:52] .= Int16(1)
            MEH.TpCacheEntry(
                vol,
                zeros(Float32, mri_sz),
                m16,
                zeros(Int8, mri_sz),
                UInt16.(a16),
                m16,
                a16
            )
        end

        lock(MEH._tp_cache_lock) do
            empty!(MEH.tp_data_cache)
            MEH.tp_data_cache[0] = make_tp_entry(ct0_vol, 7)
            MEH.tp_data_cache[1] = make_tp_entry(t2_vol, 1)
            MEH.tp_data_cache[2] = make_tp_entry(adc_vol, 1)
            MEH.tp_data_cache[3] = make_tp_entry(dwi_vol, 1)
            MEH.tp_data_cache[4] = make_tp_entry(t1_vol, 1)
        end
        lock(MEH._centroids_lock) do
            empty!(MEH.lesion_centroids_cache)
            # TP 0 lesion 7 is at Z=71 (outside MRI signal Z=45..55)
            MEH.lesion_centroids_cache[(0, 7)] = [8, 8, 71]
        end

        function make_m2_panel_state(vol)
            ts_ct = MEH.TextureSpec{Float32}(name="CT", isMainImage=true, minAndMaxValue=Float32[-160.0, 240.0], isVisible=true, maskContribution=0.0f0)
            ts_pet = MEH.TextureSpec{Float32}(name="PET", isMainImage=false, isNuclearMask=true, minAndMaxValue=Float32[0.1, 10.0], isVisible=true, maskContribution=0.5f0)
            ts_mask = MEH.TextureSpec{Int16}(name="Mask", isMultiDiscreteMask=true, isIntegerTexture=true, colorSet=[RGB(1.0, 0.0, 0.0)], minAndMaxValue=Int16[1, 10000], isVisible=true, maskContribution=0.5f0)
            ts_anat = MEH.TextureSpec{Int16}(name="Anatomy", isMultiDiscreteMask=true, isIntegerTexture=true, colorSet=[RGB(0.0, 1.0, 0.0)], minAndMaxValue=Int16[1, 400], isVisible=false, maskContribution=0.35f0)
            calcDims = MEH.CalcDimsStruct(
                mainImageQuadVert = ones(Float32, 32),
                mainQuadVertSize = 128,
                imageTextureWidth = Int32(16),
                imageTextureHeight = Int32(16),
                windowWidth = 1200,
                windowHeight = 800,
                avWindWidtForMain = Int32(1200),
                avWindHeightForMain = Int32(800),
            )
            onScroll = MEH.FullScrollableDat(
                dimensionToScroll=3,
                slicesNumber=Int32(100),
                dataToScrollDims=MEH.DataToScrollDims(imageSize=mri_sz, voxelSize=(1.0, 1.0, 1.0), dimensionToScroll=3),
                dataToScroll=[
                    MEH.ThreeDimRawDat{Float32}(type=Float32, name="CT", dat=vol),
                    MEH.ThreeDimRawDat{Float32}(type=Float32, name="PET", dat=zeros(Float32, mri_sz)),
                    MEH.ThreeDimRawDat{Int16}(type=Int16, name="Mask", dat=zeros(Int16, mri_sz)),
                    MEH.ThreeDimRawDat{Int16}(type=Int16, name="Anatomy", dat=zeros(Int16, mri_sz)),
                ]
            )
            mainDisp = MEH.forDisplayObjects(listOfTextSpecifications=MEH.TextureSpec[ts_ct, ts_pet, ts_mask, ts_anat])
            MEH.StateDataFields(
                calcDimsStruct=calcDims,
                onScrollData=onScroll,
                mainForDisplayObjects=mainDisp,
                currentDisplayedSlice=71
            )
        end

        mock_m2_states = [make_m2_panel_state(ct0_vol) for _ in 1:10]
        MedEye3d.SegmentationDisplay._m2_vk_ref[] = :mock_m2_active
        MEH.current_tp_index[] = 0
        MEH.current_active_lesion_id[] = 7
        MEH.compare_mode[] = true
        MEH.compare_right_tp[] = 1
        MEH.mri_compare_mode[] = false
        MEH.mri_main_multimodal_mode[] = false
        MEH.mri_compare_plane[] = 3

        # Step A: 'c' opens M2 in "Compare Curr/Next TP" showing TP 1 (T2)
        MedEye3d.SegmentationDisplay.apply_m2_layout!(mock_m2_states, "Compare Curr/Next TP", 0; load_data=true)
        @test MEH.mri_compare_mode[] == false
        @test mock_m2_states[6].onScrollData.currentTpIndex == 1
        @test mock_m2_states[6].currentDisplayedSlice in 45:55
        p7_pet_cmp = first(t for t in mock_m2_states[7].mainForDisplayObjects.listOfTextSpecifications if t.name == "PET")
        @test p7_pet_cmp.maskContribution == 0.0f0

        # Step B: 'y' turns MRI Multimodal ON on M2 (panels 6..9 -> TPs 1..4)
        MEH.reactToToggleMRIMultimodal(MakieEvents.ToggleMRIMultimodalEvent(true, :auto), mock_m2_states)
        @test MEH.mri_compare_mode[] == true
        @test [mock_m2_states[p].onScrollData.currentTpIndex for p in 6:9] == [1, 2, 3, 4]
        @test mock_m2_states[6].currentDisplayedSlice in 45:55
        @test mock_m2_states[7].currentDisplayedSlice in 45:55
        @test mock_m2_states[8].currentDisplayedSlice in 45:55
        @test mock_m2_states[9].currentDisplayedSlice in 45:55

        expected_wins = Dict(6 => Float32[0, 1000], 7 => Float32[0, 2200], 8 => Float32[0, 600], 9 => Float32[0, 700])
        for p in 6:9
            ct_spec = first(t for t in mock_m2_states[p].mainForDisplayObjects.listOfTextSpecifications if t.name == "CT")
            pet_spec = first(t for t in mock_m2_states[p].mainForDisplayObjects.listOfTextSpecifications if t.name == "PET")
            @test ct_spec.isVisible == true
            @test ct_spec.minAndMaxValue == expected_wins[p]
            @test pet_spec.maskContribution == 0.0f0
            if p == 7
                @test pet_spec.isVisible == false
            end
            # Loaded slice must actually belong to the MRI volume (> 0), not empty/CT
            disp_ct_slice = mock_m2_states[p].currentlyDispDat.listOfDataAndImageNames[1].dat
            @test any(>(0.0f0), disp_ct_slice)
        end

        # Step C: 'y' turns MRI Multimodal OFF -> must restore Compare TP (TP 1 T2 MRI), NEVER TP 0 CT with T2 windowing
        MEH.reactToToggleMRIMultimodal(MakieEvents.ToggleMRIMultimodalEvent(true, :auto), mock_m2_states)
        @test MEH.mri_compare_mode[] == false
        @test all(mock_m2_states[p].onScrollData.currentTpIndex == 1 for p in 6:9)
        @test mock_m2_states[6].currentDisplayedSlice in 45:55
        p6_ct_restored = mock_m2_states[6].currentlyDispDat.listOfDataAndImageNames[1].dat
        @test !any(==(-1000.0f0), p6_ct_restored)
        @test maximum(p6_ct_restored) == 500.0f0
        p6_ct_spec = first(t for t in mock_m2_states[6].mainForDisplayObjects.listOfTextSpecifications if t.name == "CT")
        @test p6_ct_spec.isVisible == true
        @test p6_ct_spec.minAndMaxValue == Float32[0.0, 1000.0]
        p7_pet_restored = first(t for t in mock_m2_states[7].mainForDisplayObjects.listOfTextSpecifications if t.name == "PET")
        @test p7_pet_restored.isVisible == false
        @test p7_pet_restored.maskContribution == 0.0f0
    finally
        MedEye3d.SegmentationDisplay._m2_vk_ref[] = old_m2_vk
        MedEye3d.SegmentationDisplay._current_m2_mode[] = old_m2_mode
        empty!(MEH.tp_labels); merge!(MEH.tp_labels, old_tp_labels)
        empty!(MEH.tp_modalities); merge!(MEH.tp_modalities, old_tp_mods)
        MEH.compare_right_tp[] = old_cmp_right
        MEH.compare_mode[] = false
        MEH.mri_compare_mode[] = false
        MEH.mri_main_multimodal_mode[] = false
        empty!(MEH.mri_compare_tps)
    end

    # Cleanup shared MEH state
    lock(MEH._tp_cache_lock) do
        empty!(MEH.tp_data_cache)
    end
    lock(MEH._centroids_lock) do
        empty!(MEH.lesion_centroids_cache)
    end
end

@testset "Test 8: Lacrimal Gland & Seminal Vesicle Mapping and Physiological Lacrimal Filtering" begin
    LA = MedEye3d.LesionAssociation
    ESR = MedEye3d.EPSMAStructuredReport

    # 1. Seminal vesicle priority and prostate recognition
    @test LA.is_prostate_structure("seminal_vesicle") == true
    @test LA.is_prostate_structure("seminal_vesicle_left") == true
    @test LA.is_prostate_structure("seminal_vesicle_right") == true
    @test LA.classify_tissue_priority("seminal_vesicle") == 1
    @test LA.classify_organ_to_lesion_type("seminal_vesicle") == "Prostate"

    # 2. Lacrimal gland recognition and priority over orbital skull
    @test LA.is_lacrimal_gland_structure("lacrimal_gland_left") == true
    @test LA.is_lacrimal_gland_structure("lacrimal_gland_right") == true
    @test LA.is_physiological_uptake_organ("lacrimal_gland_left") == true
    @test LA.classify_tissue_priority("lacrimal_gland_left") == 4

    ts_names = Dict{Int, String}(
        10 => "head",
        20 => "skin",
        30 => "skull",
        40 => "nvseg_skull",
        50 => "eye_left",
        60 => "urinary_bladder",
        70 => "rectum",
        531 => "seminal_vesicle",
        533 => "lacrimal_gland_left",
        534 => "lacrimal_gland_right",
    )

    # Orbital lesion touching skull/nvseg_skull + eye_left + lacrimal_gland_left -> MUST pick lacrimal_gland_left (never skull!)
    orb_counts = Dict{Int, Int}(10 => 200, 20 => 60, 30 => 45, 40 => 30, 50 => 80, 533 => 55)
    @test LA.pick_best_organ(orb_counts, ts_names) == "lacrimal_gland_left"

    # Seminal vesicle lesion touching urinary_bladder + rectum -> MUST pick seminal_vesicle
    sv_counts = Dict{Int, Int}(60 => 150, 70 => 80, 531 => 45)
    @test LA.pick_best_organ(sv_counts, ts_names) == "seminal_vesicle"

    # 3. Physiological lacrimal lesion filtering removes lacrimal lesions and preserves true lesions
    mask = zeros(Int16, 16, 16, 20)
    atlas = zeros(Int16, 16, 16, 20)
    # Atlas: right lacrimal gland at (4:6, 4:6, 17:18), orbital skull at (3:7, 3:7, 16:18), seminal vesicle at (7:9, 7:9, 3:4)
    atlas[3:7, 3:7, 16:18] .= Int16(40)   # nvseg_skull
    atlas[4:6, 4:6, 17:18] .= Int16(534)  # lacrimal_gland_right
    atlas[7:9, 7:9, 3:4]   .= Int16(531)  # seminal_vesicle
    atlas[10:12, 10:12, 10:11] .= Int16(30) # skull (true calvarial bone lesion)

    # Lesion 1: seminal vesicle lesion (keep)
    mask[7:9, 7:9, 3:4] .= Int16(1)
    # Lesion 2: lacrimal gland + adjacent orbital skull physiological uptake (remove!)
    mask[3:6, 3:6, 17:18] .= Int16(2)
    # Lesion 3: isolated calvarial skull metastasis far from lacrimal gland (keep)
    mask[10:12, 10:12, 10:11] .= Int16(3)

    removed = LA.filter_lacrimal_lesions!(mask, atlas, ts_names)
    @test removed == [2]
    @test any(==(Int16(1)), mask)
    @test !any(==(Int16(2)), mask)
    @test any(==(Int16(3)), mask)

    # 4. E-PSMA TNM staging for seminal_vesicle and lacrimal_gland
    @test ESR.classify_mitnm("seminal_vesicle", "", "") == "miT3b"
    @test ESR.classify_mitnm("lacrimal_gland_left", "", "") == "Artifact"
end
