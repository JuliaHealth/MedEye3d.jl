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

    # Test 5b: Semantic Filter GUI Toggle Button + Dropdown Filtering in Radiological Report Section
    btn_sf = LMW._lmw_observables[:btn_semantic_filter]
    lbl_sf = LMW._lmw_observables[:lbl_semantic_filter]
    @test occursin("OFF", btn_sf.label[])
    @test MEH.semantic_filter_active[] == false

    prev_sf_mentions = copy(MEH.semantic_mentioned_organs[])
    prev_sf_allowed = deepcopy(MEH.semantic_allowed_lesion_ids)
    prev_sf_tp = MEH.current_tp_index[]
    try
        lock(MEH._tp_cache_lock) do
            empty!(MEH.tp_data_cache)
        end
        MEH.current_tp_index[] = 0
        empty!(LMW._cached_lesion_ids)
        LMW._cached_lesion_ids[0] = collect(1:25)
        MEH.semantic_mentioned_organs[] = ["prostate"]
        empty!(MEH.semantic_allowed_lesion_ids)
        MEH.semantic_allowed_lesion_ids[0] = Set([5])

        # Toggle ON via actual button click -> dropdown filters down to lesion 5 (prostate) only
        btn_sf.clicks[] = btn_sf.clicks[] + 1
        @test MEH.semantic_filter_active[] == true
        @test occursin("ON", btn_sf.label[])
        @test occursin("1 report organs", lbl_sf.text[])
        kept_ids = [LMW.parse_lesion_id(o) for o in lesion_ids[]]
        @test kept_ids == [5]

        # Toggle OFF via actual button click -> all 25 lesions restored in dropdown
        btn_sf.clicks[] = btn_sf.clicks[] + 1
        @test MEH.semantic_filter_active[] == false
        @test occursin("OFF", btn_sf.label[])
        restored_ids = [LMW.parse_lesion_id(o) for o in lesion_ids[]]
        @test length(restored_ids) == 25
    finally
        MEH.current_tp_index[] = prev_sf_tp
        MEH.semantic_filter_active[] = false
        MEH.semantic_mentioned_organs[] = prev_sf_mentions
        empty!(MEH.semantic_allowed_lesion_ids)
        merge!(MEH.semantic_allowed_lesion_ids, prev_sf_allowed)
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

@testset "Test 9: Non-Compare Startup & Radiological Report Reload" begin
    lmw_src = read(joinpath(@__DIR__, "..", "src", "display", "LesionMetadataWindow.jl"), String)
    app_src = read(joinpath(@__DIR__, "..", "src", "packaging", "AppMain.jl"), String)

    # 1. refresh_tp_dropdown_options! must guard menu_tp_right.options[] with is_syncing_tp[] = true
    m_refresh = match(r"function refresh_tp_dropdown_options!\(\)(.*?)sync_tp_menus_to_current!\(\)"s, lmw_src)
    @test m_refresh !== nothing
    @test occursin("is_syncing_tp[] = true", m_refresh.captures[1])
    @test occursin("is_syncing_tp[] = false", m_refresh.captures[1])

    # 2. AppMain startup must explicitly enforce non-compare mode
    @test occursin("if makie_win.set_compare_mode[]", app_src)
    @test occursin("put!(mainViewer.channel, CompareTimePointsEvent(false))", app_src)

    # 3. XNAT case switch must reload tp_descriptions and tp_english_descriptions & refresh report UI
    @test occursin("empty!(MEH.tp_descriptions)", app_src)
    @test occursin("empty!(MEH.tp_english_descriptions)", app_src)
    @test occursin("obs_d[:update_old_report_display]()", app_src)
end

@testset "Test 10: F1/F2/F3 CT Windowing Must Affect Only CT (Never PET or MRI)" begin
    old_tp_mods = copy(MEH.tp_modalities)
    old_win = copy(MEH.current_windowing)
    old_user_mods = copy(MEH._user_windowing_modalities)
    old_cfg_env = get(ENV, "MEDEYE3D_DISPLAY_CONFIG_PATH", nothing)
    tmp_cfg = tempname() * "_test10_display_cfg.json"
    ENV["MEDEYE3D_DISPLAY_CONFIG_PATH"] = tmp_cfg
    old_cur_tp = MEH.current_tp_index[]
    old_cmp = MEH.compare_mode[]
    old_cmp_right = MEH.compare_right_tp[]
    old_mri_cmp = MEH.mri_compare_mode[]
    old_mri_main = MEH.mri_main_multimodal_mode[]

    try
        empty!(MEH.tp_modalities)
        MEH.tp_modalities[0] = "PET"   # Baseline PET/CT
        MEH.tp_modalities[1] = "T2"    # MRI Follow-Up
        MEH.current_tp_index[] = 0
        MEH.compare_mode[] = true
        MEH.compare_right_tp[] = 1
        MEH.mri_compare_mode[] = false
        MEH.mri_main_multimodal_mode[] = false

        function make_win_state(tp_i, specs)
            calc = MEH.CalcDimsStruct(mainImageQuadVert = ones(Float32, 32), mainQuadVertSize = 128)
            scr = MEH.FullScrollableDat(dimensionToScroll=3, slicesNumber=Int32(10), currentTpIndex=tp_i)
            disp = MEH.forDisplayObjects(listOfTextSpecifications=specs)
            return MEH.StateDataFields(calcDimsStruct=calc, onScrollData=scr, mainForDisplayObjects=disp, currentDisplayedSlice=1)
        end

        # Panel 1: Fused CT + PET overlay at TP 0
        p1_ct = MEH.TextureSpec{Float32}(name="CT", isMainImage=true, studyType="CT", minAndMaxValue=Float32[-160, 240])
        p1_pet = MEH.TextureSpec{Float32}(name="PET", isMainImage=false, isNuclearMask=true, studyType="PET", minAndMaxValue=Float32[0, 10])
        # Panel 2: Pure PET panel at TP 0 (isMainImage=true, isNuclearMask=false, studyType="" like AppMain)
        p2_pet = MEH.TextureSpec{Float32}(name="PET", isMainImage=true, isNuclearMask=false, studyType="", minAndMaxValue=Float32[0, 10])
        # Panel 3: MRI panel at TP 1 (reuses TextureSpec name="CT", studyType="CT" for MRI display!)
        p3_mri = MEH.TextureSpec{Float32}(name="CT", isMainImage=true, studyType="CT", minAndMaxValue=Float32[0, 1000])
        p3_pet = MEH.TextureSpec{Float32}(name="PET", isMainImage=false, isNuclearMask=true, studyType="PET", minAndMaxValue=Float32[0, 10])

        states = [
            make_win_state(0, [p1_ct, p1_pet]),
            make_win_state(0, [p2_pet]),
            make_win_state(0, [p1_ct, p1_pet]),
            make_win_state(0, [p1_ct, p1_pet]),
            make_win_state(0, [p1_ct, p1_pet]),
            make_win_state(1, [p3_mri, p3_pet]),  # Panel 6 compares MRI TP 1
        ]

        # Simulate F2/F3 presets (which ReactToKeyboard dispatches as WindowingEvent("CT", ...))
        MEH.reactToWindowing(MakieEvents.WindowingEvent("CT", -450.0f0, 1050.0f0), states)
        @test p1_ct.minAndMaxValue == Float32[-450, 1050]
        @test p1_pet.minAndMaxValue == Float32[0, 10]
        @test p2_pet.minAndMaxValue == Float32[0, 10]
        @test p3_mri.minAndMaxValue == Float32[0, 1000]
        @test p3_pet.minAndMaxValue == Float32[0, 10]

        MEH.reactToWindowing(MakieEvents.WindowingEvent("CT", -1350.0f0, 150.0f0), states)
        @test p1_ct.minAndMaxValue == Float32[-1350, 150]
        @test p1_pet.minAndMaxValue == Float32[0, 10]
        @test p2_pet.minAndMaxValue == Float32[0, 10]
        @test p3_mri.minAndMaxValue == Float32[0, 1000]

        # WindowControll legacy F1/F2/F3 path must also never alter PET
        WC = MedEye3d.WindowControll
        WC.setTextureWindow(p2_pet, states[2], MedEye3d.DataStructs.WindowControlStruct(letterCode="F1"))
        WC.setTextureWindow(p2_pet, states[2], MedEye3d.DataStructs.WindowControlStruct(letterCode="F2"))
        WC.setTextureWindow(p2_pet, states[2], MedEye3d.DataStructs.WindowControlStruct(letterCode="F3"))
        @test p2_pet.minAndMaxValue == Float32[0, 10]
    finally
        empty!(MEH.tp_modalities); merge!(MEH.tp_modalities, old_tp_mods)
        empty!(MEH.current_windowing); merge!(MEH.current_windowing, old_win)
        empty!(MEH._user_windowing_modalities); union!(MEH._user_windowing_modalities, old_user_mods)
        if old_cfg_env === nothing
            delete!(ENV, "MEDEYE3D_DISPLAY_CONFIG_PATH")
        else
            ENV["MEDEYE3D_DISPLAY_CONFIG_PATH"] = old_cfg_env
        end
        try rm(tmp_cfg; force=true) catch; end
        MEH.current_tp_index[] = old_cur_tp
        MEH.compare_mode[] = old_cmp
        MEH.compare_right_tp[] = old_cmp_right
        MEH.mri_compare_mode[] = old_mri_cmp
        MEH.mri_main_multimodal_mode[] = old_mri_main
    end
end

@testset "Test 11: Default Prostata_bimodal_PIPELINE Project & Typeable Patient Picker Overlap Search" begin
    # 1. Default XNAT project must be Prostata_bimodal_PIPELINE
    @test LMW.ALLOWED_XNAT_PROJECTS[1] == "FDM_DPI-2024-7-KRN_Prostata_bimodal_PIPELINE"
    proj_menu = LMW._lmw_observables[:xnat_proj_menu]
    @test proj_menu.selection[] == "FDM_DPI-2024-7-KRN_Prostata_bimodal_PIPELINE"

    # 2. Best-overlap ranking prefers exact/tightest identifier match first
    sample_subjs = [
        "[RAW] FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat127",
        "[PRE] FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat27",
        "[OK: alice] FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat2",
        "[PRE] FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat20",
        "[PRE] FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat3",
    ]
    rank_fn = LMW._lmw_observables[:rank_best_overlap]
    ranked_27 = rank_fn("Pat27", sample_subjs; limit=10)
    @test !isempty(ranked_27)
    @test occursin("__Pat27", ranked_27[1])
    @test !any(s -> occursin("__Pat3", s), ranked_27)

    ranked_2 = rank_fn("Pat2", sample_subjs; limit=10)
    @test occursin("__Pat2", ranked_2[1]) && !occursin("__Pat20", ranked_2[1])

    # 2b. Subject ID badge stripping handles [OK: user] colons cleanly
    @test LMW._parse_subject_id("[OK: jm] FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat1") == "FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat1"
    @test LMW._parse_subject_id("[PRE] FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat27") == "FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat27"
    @test LMW._parse_subject_id("[OK] pat6 (PROCESSED)") == "pat6"

    # 3. In-memory FTS5 index + live searchable_menu typing narrows patient dropdown options
    LMW._build_fts!("patient_fts", sample_subjs)
    fts_hits = LMW.fts_search("patient_fts", "pat27"; limit=20)
    @test any(s -> occursin("Pat27", s), fts_hits)

    subj_menu = LMW._lmw_observables[:xnat_subj_menu]
    cache = LMW._lmw_observables[:raw_subjects_cache]
    filt_fn = LMW._lmw_observables[:filter_and_set_subjects]
    prev_cache = get(cache, "FDM_DPI-2024-7-KRN_Prostata_bimodal_PIPELINE", nothing)
    try
        cache["FDM_DPI-2024-7-KRN_Prostata_bimodal_PIPELINE"] = copy(sample_subjs)
        filt_fn(query = "")
        @test length(subj_menu.options[]) == 5

        # Typing in patient search textbox narrows and ranks options
        filt_fn(query = "Pat27")
        @test occursin("__Pat27", subj_menu.options[][1])
        @test length(subj_menu.options[]) <= 2

        # Reset full list, open searchable_menu dropdown and type directly into menu
        filt_fn(query = "")
        subj_menu.is_open[] = true
        for ch in ('p', 'a', 't', '2', '7')
            events(subj_menu.blockscene).unicode_input[] = ch
        end
        sleep(0.25)  # wait past 80ms debounce
        @test occursin("__Pat27", subj_menu.options[][1])
        @test length(subj_menu.options[]) <= 2
        subj_menu.is_open[] = false
    finally
        if prev_cache === nothing
            delete!(cache, "FDM_DPI-2024-7-KRN_Prostata_bimodal_PIPELINE")
        else
            cache["FDM_DPI-2024-7-KRN_Prostata_bimodal_PIPELINE"] = prev_cache
        end
    end
end

@testset "Test 12: Case Switch Pat51 -> Pat1 Clears Stale Organ Mapping & Keeps Quadriceps Femoris Left" begin
    import HDF5, JSON

    tmp_pat1_h5 = tempname() * "_Pat1_switch_test.h5"
    old_h5_path = MEH.h5_path_ref[]
    old_h5_save = MEH.h5_save_path_ref[]
    old_pat_id = MEH.patient_id[]
    try
        HDF5.h5open(tmp_pat1_h5, "w") do f
            mg = HDF5.create_group(f, "_meta_")
            mg["organ_mapping"] = JSON.json(Dict(
                "0" => Dict("3" => "quadriceps_femoris_left", "4" => "gluteus_maximus_left")
            ))
            mg["segment_names.json"] = JSON.json(Dict(
                "0" => Dict("3" => "NNUNET_OR_TP_unknown_3_CT", "4" => "gluteus_maximus_left")
            ))
            mg["max_anatomy_labels.json"] = JSON.json(Dict(
                "9" => "quadriceps_femoris_left", "51" => "gluteus_maximus_left"
            ))
            cg = HDF5.create_group(f, "CENTROIDS")
            cg["tp0_lid3"] = Int32[120, 140, 12]
            cg["tp0_lid4"] = Int32[110, 160, 25]
        end

        # Simulate stale in-memory state from Pat51 where lesion 3 was Prostate
        MEH.current_tp_index[] = 0
        MEH.tp_organ_mapping[0] = Dict(3 => "nvseg_prostate", 4 => "prostate")
        MEH.global_organ_mapping[] = Dict(3 => "nvseg_prostate", 4 => "prostate")
        MEH.tp_segment_names[0] = Dict(3 => "Prostate Gland", 4 => "Prostate")
        MEH.tp_descriptions[0] = "2023-02-07 Pat51 stale dictation"
        MEH.tp_english_descriptions[0] = "Pat51 stale English dictation"

        # Switch to tmp_pat1_h5
        LMW._lmw_observables[:switch_to_hdf5](tmp_pat1_h5)

        # 1. Stale Pat51 organ_mapping and dictations must be cleared and replaced by Pat1
        @test MEH.tp_organ_mapping[0][3] == "quadriceps_femoris_left"
        @test MEH.global_organ_mapping[][3] == "quadriceps_femoris_left"
        @test !haskey(MEH.tp_descriptions, 0) || isempty(MEH.tp_descriptions[0])

        # 2. First lesion (lid=3) must display Quadriceps Femoris Left, never Prostate
        act_str = LMW._lmw_observables[:active_lesion_id][]
        @test LMW.parse_lesion_id(act_str) == 3
        @test !occursin("prostate", lowercase(act_str))
        @test occursin("quadriceps", lowercase(act_str))

        # 3. Metadata widgets (BaseAnatomy, LesionType, Anatomic Location) must reflect Quadriceps Femoris Left
        menu_ba = LMW._lmw_observables[:menu_base_anat]
        ba_val = lowercase(string(menu_ba.selection[]))
        @test occursin("quadriceps femoris", ba_val)
        @test !occursin("prostate", ba_val)
        @test LMW._lmw_observables[:active_lesion_type][] == "Technical Artifact"

        fw = LMW._lmw_observables[:field_widgets]
        loc_val = string(fw["Anatomic Location"].selection[])
        @test occursin("Muscle", loc_val) || occursin("Soft Tissue", loc_val)

        # 4. Wait past autosave debounce and verify target HDF5 was NOT corrupted with Pat51 / Prostate
        sleep(0.45)
        HDF5.h5open(tmp_pat1_h5, "r") do f
            om_after = JSON.parse(read(f["_meta_/organ_mapping"]))
            @test om_after["0"]["3"] == "quadriceps_femoris_left"
            if haskey(f, "_meta_/lesion_annotations.json")
                ann_after = JSON.parse(read(f["_meta_/lesion_annotations.json"]))
                if haskey(ann_after, "3")
                    @test !occursin("prostate", lowercase(string(get(ann_after["3"], "BaseAnatomy", ""))))
                end
            end
        end
    finally
        MEH.h5_path_ref[] = old_h5_path
        MEH.h5_save_path_ref[] = old_h5_save
        MEH.patient_id[] = old_pat_id
        try rm(tmp_pat1_h5; force=true) catch; end
    end
end

@testset "Test 13: XNAT Remote Freshness Comparison & Full Cohort Prefetch Cache" begin
    import Dates, HDF5, JSON
    XC = MedEye3d.XnatClient
    XP = MedEye3d.XnatPreloader

    # 1. Timestamp parser handles both ISO-8601 and RFC-7231 HTTP Last-Modified headers
    ts_iso = XC._parse_remote_timestamp_utc("2026-11-18T14:30:15Z")
    ts_http = XC._parse_remote_timestamp_utc("Wed, 18 Nov 2026 14:30:15 GMT")
    @test ts_iso == Dates.DateTime(2026, 11, 18, 14, 30, 15)
    @test ts_http == Dates.DateTime(2026, 11, 18, 14, 30, 15)
    @test XC._parse_remote_timestamp_utc("") === nothing

    # 2. Sidecar XNAT metadata helper round-trip & local autosave preservation
    tmp_dummy_h5 = tempname() * ".h5"
    write(tmp_dummy_h5, zeros(UInt8, 4096))
    try
        XC._write_local_xnat_meta(tmp_dummy_h5, "8c62c57c69b46590cc9797bc5594cfb2"; last_modified="2026-11-18T14:30:15Z")
        @test XC._read_local_xnat_digest(tmp_dummy_h5) == "8c62c57c69b46590cc9797bc5594cfb2"

        # Simulate local MedEye3d autosave growing local file size while XNAT baseline stays identical
        write(tmp_dummy_h5, zeros(UInt8, 8192))
        same_remote = (found=true, size=Int64(4096), digest="8c62c57c69b46590cc9797bc5594cfb2", last_modified_utc=nothing, uri="/x")
        @test XC._compare_remote_against_local(same_remote, tmp_dummy_h5) == (false, "up_to_date")

        newer_remote = (found=true, size=Int64(4096), digest="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", last_modified_utc=nothing, uri="/x")
        @test XC._compare_remote_against_local(newer_remote, tmp_dummy_h5) == (true, "digest_changed")
    finally
        rm(tmp_dummy_h5; force=true)
        rm(XC._xnat_meta_path(tmp_dummy_h5); force=true)
    end

    # 3. Local Prostata_bimodal cache must contain all 126 repo Pat* cases
    repo_root = joinpath(dirname(@__DIR__), "data", "cases", "Prostata_bimodal")
    cache_root = joinpath(homedir(), ".medeye3d", "xnat_cache", "Prostata_bimodal_PIPELINE")
    repo_pats = sort(filter(d -> occursin(r"^Pat\d+$", d) && isfile(joinpath(repo_root, d, "preprocessed_volumes.h5")), readdir(repo_root));
        by = d -> parse(Int, d[4:end]))
    @test length(repo_pats) == 126

    missing_cached = String[]
    for pat in repo_pats
        found = false
        if isdir(cache_root)
            for sub in readdir(cache_root)
                if endswith(sub, "__$pat") && isdir(joinpath(cache_root, sub))
                    for fn in readdir(joinpath(cache_root, sub))
                        fp = joinpath(cache_root, sub, fn)
                        if endswith(lowercase(fn), ".h5") && !occursin("lesion_edits", lowercase(fn)) && filesize(fp) > 1_000_000
                            found = true
                            break
                        end
                    end
                end
                found && break
            end
        end
        !found && push!(missing_cached, pat)
    end
    @test isempty(missing_cached)

    # 4. Verify ground-truth Pat1 has clean Quadriceps Femoris Left on lid=3 and no stale Prostate annotation
    pat1_repo_h5 = joinpath(repo_root, "Pat1", "preprocessed_volumes.h5")
    @test isfile(pat1_repo_h5)
    HDF5.h5open(pat1_repo_h5, "r") do f
        om = JSON.parse(read(f["_meta_/organ_mapping"]))
        @test om["0"]["3"] == "quadriceps_femoris_left"
        @test !haskey(f, "_meta_/lesion_annotations.json")
    end
end

@testset "Test 14: External Lymph Node Repository Boundary & Pat1 LN Mask Integrity" begin
    prep_dir = joinpath(dirname(@__DIR__), "scripts", "preprocessing")
    @test !isdir(joinpath(prep_dir, "lymph_node_rules"))
    @test !isfile(joinpath(prep_dir, "run_ln_worker.jl"))
    @test isfile(joinpath(dirname(@__DIR__), "data", "cases", "Prostata_bimodal", "Pat1", "Pat1_Normal_Lymph_Node_Rules.mrb"))

    pat1_h5 = joinpath(dirname(@__DIR__), "data", "cases", "Prostata_bimodal", "Pat1", "preprocessed_volumes.h5")
    @test isfile(pat1_h5)
    HDF5.h5open(pat1_h5, "r") do f
        lbls = Dict{Int, String}(parse(Int, string(k)) => lowercase(string(v)) for (k, v) in JSON.parse(read(f["_meta_/max_anatomy_labels.json"])))
        ln_ids = [k for (k, v) in lbls if 400 <= k < 500]
        @test length(ln_ids) >= 50
        # Ensure non-LN helper/exclusion structures never leak into LN label space
        @test !any(v -> occursin("exclusion_zone", v) || occursin("torso_fat", v) || occursin("helper", v), values(lbls))
        # Ensure cervical/thoracic/abdominal/pelvic LN stations exist in BASELINE/max_anatomy.nii.gz
        anat = read(f["BASELINE/max_anatomy.nii.gz"])
        @test any(400 .<= anat .< 500)
    end
end

@testset "Test 15: Anatomy Display Includes Pat1 Lymph Node IDs (> 400) & Case Switch Clears Stale LN Cache" begin
    @test MEH.ANATOMY_MAX_LABEL_ID >= 10000

    # Verify Pat1's 69 LN IDs (428..496) are ALL <= ANATOMY_MAX_LABEL_ID
    pat1_h5 = joinpath(dirname(@__DIR__), "data", "cases", "Prostata_bimodal", "Pat1", "preprocessed_volumes.h5")
    @test isfile(pat1_h5)
    pat1_labels = HDF5.h5open(pat1_h5, "r") do f
        Dict{Int, String}(parse(Int, string(k)) => string(v) for (k, v) in JSON.parse(read(f["_meta_/max_anatomy_labels.json"])))
    end

    old_anat_labels = deepcopy(MEH.anatomy_labels_cache)
    old_global_ts = copy(MEH.global_ts_names[])
    old_ln_val = MEH.lymph_node_validation_mode[]
    old_anat_vis = MEH._anatomy_active[]
    old_single = MEH.is_single_lesion_mode[]
    old_tp = MEH.current_tp_index[]
    old_cmp = MEH.compare_mode[]

    try
        MEH.current_tp_index[] = 0
        MEH.compare_mode[] = false
        MEH.global_ts_names[] = pat1_labels
        MEH.anatomy_labels_cache[0] = pat1_labels

        ln_ids = MEH.get_lymph_node_label_ids(0)
        @test length(ln_ids) == 69
        @test minimum(ln_ids) == 428
        @test maximum(ln_ids) == 496

        ts_ct = MEH.TextureSpec{Float32}(name="CT", isMainImage=true, studyType="CT", minAndMaxValue=Float32[-160, 240])
        ts_mask = MEH.TextureSpec{Int16}(name="Mask", isMainImage=false, isMultiDiscreteMask=true, isIntegerTexture=true, minAndMaxValue=Int16[1, 10000])
        ts_anat = MEH.TextureSpec{Int16}(name="Anatomy", isMainImage=false, isMultiDiscreteMask=true, isIntegerTexture=true,
            minAndMaxValue=Int16[1, MEH.ANATOMY_MAX_LABEL_ID], isEditable=true, isVisible=false, maskContribution=0.5f0)
        calc = MEH.CalcDimsStruct(mainImageQuadVert=ones(Float32, 32), mainQuadVertSize=128)
        scr = MEH.FullScrollableDat(dimensionToScroll=3, slicesNumber=Int32(4), currentTpIndex=0)
        disp = MEH.forDisplayObjects(listOfTextSpecifications=MEH.TextureSpec[ts_ct, ts_mask, ts_anat])
        st = MEH.StateDataFields(calcDimsStruct=calc, onScrollData=scr, mainForDisplayObjects=disp, currentDisplayedSlice=1)
        states = [st]

        # 1. Anatomy: ON (ShowMaskLayerEvent(4, true)) must keep max >= 496 (Pat1 max LN ID)
        MEH.reactToShowMaskLayer(MakieEvents.ShowMaskLayerEvent(4, true), states)
        @test ts_anat.isVisible == true
        @test ts_anat.minAndMaxValue[1] <= 428
        @test ts_anat.minAndMaxValue[2] >= 496

        # 2. _load_tp_from_entry! (called on SetTimePointEvent / case switch) must keep max >= 496
        dummy_ct = zeros(Float32, 4, 4, 4)
        dummy_mask = zeros(Int16, 4, 4, 4)
        dummy_bone = zeros(Int8, 4, 4, 4)
        dummy_anat = zeros(UInt16, 4, 4, 4)
        dummy_anat[2, 2, 2] = UInt16(435)
        entry = MEH.TpCacheEntry(dummy_ct, dummy_ct, dummy_mask, dummy_bone, dummy_anat, dummy_mask, Int16.(dummy_anat))
        MEH._load_tp_from_entry!(states, entry, 1)
        @test ts_anat.minAndMaxValue[2] >= 496

        # 3. Normal reactToSyncLesion with Anatomy ON must keep max >= 496
        MEH.lymph_node_validation_mode[] = false
        MEH.reactToSyncLesion(MakieEvents.SyncLesionEvent(3, true), states)
        @test ts_anat.minAndMaxValue[2] >= 496

        # 4. LN Validation Mode All-LNs and OFF branch must keep max >= 496
        MEH.lymph_node_validation_mode[] = true
        MEH.is_single_lesion_mode[] = false
        MEH.reactToSyncLesion(MakieEvents.SyncLesionEvent(435, false), states)
        @test ts_anat.isVisible == true
        @test ts_anat.minAndMaxValue == Int16[428, 496]

        MEH.reactToSetLNValidationMode(MakieEvents.SetLNValidationModeEvent(false), states)
        @test ts_anat.isVisible == false
        @test ts_anat.minAndMaxValue[2] >= 496
    finally
        empty!(MEH.anatomy_labels_cache)
        merge!(MEH.anatomy_labels_cache, old_anat_labels)
        MEH.global_ts_names[] = old_global_ts
        MEH.lymph_node_validation_mode[] = old_ln_val
        MEH._anatomy_active[] = old_anat_vis
        MEH.is_single_lesion_mode[] = old_single
        MEH.current_tp_index[] = old_tp
        MEH.compare_mode[] = old_cmp
    end
end

@testset "Test 16: Quad-MRI Fallback CT Windowing (<4 MRIs) & Pat1 T2/ADC/DWI/T1 Modalities" begin
    old_tp_labels = copy(MEH.tp_labels)
    old_tp_mods = copy(MEH.tp_modalities)
    old_mri_cmp = MEH.mri_compare_mode[]
    old_mri_main = MEH.mri_main_multimodal_mode[]
    old_cmp = MEH.compare_mode[]
    old_tp = MEH.current_tp_index[]
    old_m2_vk = MedEye3d.SegmentationDisplay._m2_vk_ref[]
    old_m2_mode = MedEye3d.SegmentationDisplay._current_m2_mode[]
    old_win = deepcopy(MEH.current_windowing)
    try
        empty!(MEH.tp_labels)
        empty!(MEH.tp_modalities)
        MEH.tp_labels[0] = "PET BASELINE (TP 0)"; MEH.tp_modalities[0] = "PET"
        MEH.tp_labels[1] = "T2 TP 1";             MEH.tp_modalities[1] = "T2"
        MEH.tp_labels[2] = "DWI TP 2";            MEH.tp_modalities[2] = "DWI"
        MEH.tp_labels[3] = "T1 TP 3";             MEH.tp_modalities[3] = "T1"

        mri_sz = (8, 8, 8)
        function mk_tp(v)
            m16 = zeros(Int16, mri_sz)
            a16 = zeros(Int16, mri_sz)
            MEH.TpCacheEntry(v, zeros(Float32, mri_sz), m16, zeros(Int8, mri_sz), UInt16.(a16), m16, a16)
        end
        lock(MEH._tp_cache_lock) do
            empty!(MEH.tp_data_cache)
            MEH.tp_data_cache[0] = mk_tp(fill(-500.0f0, mri_sz))
            MEH.tp_data_cache[1] = mk_tp(fill(500.0f0, mri_sz))
            MEH.tp_data_cache[2] = mk_tp(fill(300.0f0, mri_sz))
            MEH.tp_data_cache[3] = mk_tp(fill(400.0f0, mri_sz))
        end

        function mk_panel()
            ts_ct = MEH.TextureSpec{Float32}(name="CT", isMainImage=true, minAndMaxValue=Float32[-160.0, 240.0], isVisible=true, maskContribution=0.0f0)
            ts_pet = MEH.TextureSpec{Float32}(name="PET", isMainImage=false, isNuclearMask=true, minAndMaxValue=Float32[0.1, 10.0], isVisible=true, maskContribution=0.5f0)
            calc = MEH.CalcDimsStruct(mainImageQuadVert=ones(Float32, 32), mainQuadVertSize=128)
            scr = MEH.FullScrollableDat(
                dimensionToScroll=3, slicesNumber=Int32(8), currentTpIndex=0,
                dataToScrollDims=MEH.DataToScrollDims(imageSize=mri_sz, voxelSize=(1.0, 1.0, 1.0), dimensionToScroll=3),
                dataToScroll=[
                    MEH.ThreeDimRawDat{Float32}(type=Float32, name="CT", dat=fill(-500.0f0, mri_sz)),
                    MEH.ThreeDimRawDat{Float32}(type=Float32, name="PET", dat=zeros(Float32, mri_sz)),
                ]
            )
            MEH.StateDataFields(
                calcDimsStruct=calc, onScrollData=scr,
                mainForDisplayObjects=MEH.forDisplayObjects(listOfTextSpecifications=MEH.TextureSpec[ts_ct, ts_pet]),
                currentDisplayedSlice=4,
            )
        end

        states = [mk_panel() for _ in 1:4]
        MedEye3d.SegmentationDisplay._m2_vk_ref[] = nothing
        MEH.current_tp_index[] = 0
        MEH.compare_mode[] = false
        MEH.mri_compare_mode[] = false
        MEH.mri_main_multimodal_mode[] = false
        MEH.current_windowing["CT"] = Float32[-160.0, 240.0]

        @test MEH.get_mri_comparison_tps(0) == [1, 2, 3, 0]
        MEH.reactToToggleMRIMultimodal(MakieEvents.ToggleMRIMultimodalEvent(true, :auto), states)

        # Panel 4 falls back to TP 0 (CT baseline): must get CT windowing [-160, 240] (not [0, 1000])
        p4_ct = first(t for t in states[4].mainForDisplayObjects.listOfTextSpecifications if t.name == "CT")
        @test p4_ct.minAndMaxValue == Float32[-160.0, 240.0]
        p4_pet = first(t for t in states[4].mainForDisplayObjects.listOfTextSpecifications if t.name == "PET")
        @test p4_pet.minAndMaxValue == Float32[0.0, 10.0]
        for p in 1:4
            p_pet = first(t for t in states[p].mainForDisplayObjects.listOfTextSpecifications if t.name == "PET")
            @test p_pet.isVisible == false
            @test p_pet.maskContribution == 0.0f0
        end

        # Verify repaired Pat1 HDF5 now contains [T2, ADC, DWI, T1]
        pat1_h5 = joinpath(dirname(@__DIR__), "data", "cases", "Prostata_bimodal", "Pat1", "preprocessed_volumes.h5")
        @test isfile(pat1_h5)
        sh = HDF5.h5open(pat1_h5, "r") do f
            JSON.parse(read(f["_meta_/scene_hierarchy.json"]))
        end
        mri_mods = [n["modality"] for n in sh if get(n, "type", "") == "vtkMRMLLinearTransformNode"]
        @test length(mri_mods) >= 4
        @test mri_mods[1:4] == ["T2", "ADC", "DWI", "T1"]
    finally
        empty!(MEH.tp_labels); merge!(MEH.tp_labels, old_tp_labels)
        empty!(MEH.tp_modalities); merge!(MEH.tp_modalities, old_tp_mods)
        empty!(MEH.current_windowing); merge!(MEH.current_windowing, old_win)
        MEH.mri_compare_mode[] = old_mri_cmp
        MEH.mri_main_multimodal_mode[] = old_mri_main
        MEH.compare_mode[] = old_cmp
        MEH.current_tp_index[] = old_tp
        MedEye3d.SegmentationDisplay._m2_vk_ref[] = old_m2_vk
        MedEye3d.SegmentationDisplay._current_m2_mode[] = old_m2_mode
        empty!(MEH.mri_compare_tps)
        lock(MEH._tp_cache_lock) do; empty!(MEH.tp_data_cache); end
    end
end

@testset "Test 17: Persistent Installation-Wide Manual Windowing Across Cases & Restarts" begin
    old_win = deepcopy(MEH.current_windowing)
    old_user_mods = copy(MEH._user_windowing_modalities)
    old_tp_mods = copy(MEH.tp_modalities)
    old_h5_path = MEH.h5_path_ref[]
    old_h5_save = MEH.h5_save_path_ref[]
    old_mri_main = MEH.mri_main_multimodal_mode[]
    old_mri_cmp = MEH.mri_compare_mode[]
    old_cmp = MEH.compare_mode[]
    old_cur_tp = MEH.current_tp_index[]
    old_cfg_env = get(ENV, "MEDEYE3D_DISPLAY_CONFIG_PATH", nothing)

    tmp_cfg = tempname() * "_persist_win_cfg.json"
    tmp_h5 = tempname() * "_persist_win_case2.h5"
    tmp_json = replace(tmp_h5, ".h5" => "_annotations.json")
    ENV["MEDEYE3D_DISPLAY_CONFIG_PATH"] = tmp_cfg

    try
        empty!(MEH._user_windowing_modalities)
        empty!(MEH.tp_modalities)
        MEH.tp_modalities[0] = "PET"
        MEH.tp_modalities[1] = "T2"
        MEH.tp_modalities[2] = "ADC"
        MEH.current_tp_index[] = 0
        MEH.mri_main_multimodal_mode[] = false
        MEH.mri_compare_mode[] = false
        MEH.compare_mode[] = false

        function _mk_state(tp_i, specs)
            calc = MEH.CalcDimsStruct(mainImageQuadVert = ones(Float32, 32), mainQuadVertSize = 128)
            scr = MEH.FullScrollableDat(dimensionToScroll=3, slicesNumber=Int32(8), currentTpIndex=tp_i)
            disp = MEH.forDisplayObjects(listOfTextSpecifications=specs)
            return MEH.StateDataFields(calcDimsStruct=calc, onScrollData=scr, mainForDisplayObjects=disp, currentDisplayedSlice=1)
        end

        p1_ct = MEH.TextureSpec{Float32}(name="CT", isMainImage=true, studyType="CT", minAndMaxValue=Float32[-160, 240])
        p1_pet = MEH.TextureSpec{Float32}(name="PET", isMainImage=false, isNuclearMask=true, studyType="PET", minAndMaxValue=Float32[0.1, 10.0])
        p2_t2 = MEH.TextureSpec{Float32}(name="CT", isMainImage=true, studyType="CT", minAndMaxValue=Float32[0, 1000])
        p3_adc = MEH.TextureSpec{Float32}(name="CT", isMainImage=true, studyType="CT", minAndMaxValue=Float32[0, 2200])
        states = [_mk_state(0, [p1_ct, p1_pet]), _mk_state(1, [p2_t2]), _mk_state(2, [p3_adc])]

        # 1. Manual windowing adjustments must persist immediately to global config
        MEH.reactToWindowing(MakieEvents.WindowingEvent("CT", -450.0f0, 1050.0f0), states)
        MEH.reactToWindowing(MakieEvents.WindowingEvent("PET", 0.5f0, 18.0f0), states)
        MEH.reactToWindowing(MakieEvents.WindowingEvent("T2", 120.0f0, 880.0f0), states)
        MEH.reactToWindowing(MakieEvents.WindowingEvent("ADC", 500.0f0, 1950.0f0), states)
        @test isfile(tmp_cfg)

        # 2. Saving non-windowing display settings (e.g. label opacity / pet blend) must NOT clobber windowing
        LMW.save_display_config(Dict{String, Any}("label_opacity" => 0.65, "pet_ct_blend" => 0.45))
        saved_raw = JSON.parse(read(tmp_cfg, String))
        @test saved_raw["label_opacity"] ≈ 0.65
        @test saved_raw["windowing"]["CT"] == [-450.0, 1050.0]
        @test saved_raw["windowing"]["PET"] == [0.5, 18.0]
        @test saved_raw["windowing"]["T2"] == [120.0, 880.0]
        @test saved_raw["windowing"]["ADC"] == [500.0, 1950.0]

        # 3. Simulated app restart: clear in-memory windowing -> load_persistent_windowing!() restores all values
        empty!(MEH._user_windowing_modalities)
        for (k, v) in MEH.DEFAULT_WINDOWING
            MEH.current_windowing[k] = copy(v)
        end
        @test MEH.load_persistent_windowing!() == true
        @test MEH.has_persistent_windowing("CT")
        @test MEH.has_persistent_windowing("PET")
        @test MEH.has_persistent_windowing("T2")
        @test MEH.has_persistent_windowing("ADC")
        @test MEH.current_windowing["CT"] == Float32[-450.0, 1050.0]
        @test MEH.current_windowing["PET"] == Float32[0.5, 18.0]
        @test MEH.current_windowing["T2"] == Float32[120.0, 880.0]
        @test MEH.current_windowing["MRI"] == Float32[120.0, 880.0]
        @test MEH.current_windowing["ADC"] == Float32[500.0, 1950.0]

        # 4. Opening a second case whose HDF5 carries different _GLOBAL_APP_STATE windowing must NOT overwrite saved values
        #    (Test apply_global_state directly on the live LMW from Test 11 without spawning a duplicate GLMakie window)
        HDF5.h5open(tmp_h5, "w") do f
            mg = HDF5.create_group(f, "_meta_")
            mg["lesion_annotations.json"] = JSON.json(Dict(
                "_GLOBAL_APP_STATE" => Dict(
                    "CT_Min" => "-160.0", "CT_Max" => "240.0",
                    "PET_Min" => "0.1", "PET_Max" => "10.0",
                    "T2_Min" => "0.0", "T2_Max" => "1000.0",
                    "ADC_Min" => "0.0", "ADC_Max" => "2200.0"
                )
            ))
        end
        MEH.h5_path_ref[] = tmp_h5
        MEH.h5_save_path_ref[] = tmp_h5

        @test haskey(LMW._lmw_observables, :apply_global_state)
        @test haskey(LMW._lmw_observables, :sync_windowing_gui)
        LMW._lmw_observables[:apply_global_state](Dict(
            "CT_Min" => "-160.0", "CT_Max" => "240.0",
            "PET_Min" => "0.1", "PET_Max" => "10.0",
            "T2_Min" => "0.0", "T2_Max" => "1000.0",
            "ADC_Min" => "0.0", "ADC_Max" => "2200.0"
        ))

        @test MEH.current_windowing["CT"] == Float32[-450.0, 1050.0]
        @test MEH.current_windowing["PET"] == Float32[0.5, 18.0]
        @test MEH.current_windowing["T2"] == Float32[120.0, 880.0]
        @test MEH.current_windowing["ADC"] == Float32[500.0, 1950.0]
        @test LMW._lmw_observables[:tb_ct_min].stored_string[] == "-450.0"
        @test LMW._lmw_observables[:tb_ct_max].stored_string[] == "1050.0"
        @test LMW._lmw_observables[:tb_t2_min].stored_string[] == "120.0"
        @test LMW._lmw_observables[:tb_adc_min].stored_string[] == "500.0"

        # 5. apply_current_windowing_to_states! applies persisted windowing to fresh case panels
        p1_ct.minAndMaxValue = Float32[-160, 240]
        p1_pet.minAndMaxValue = Float32[0.1, 10.0]
        p2_t2.minAndMaxValue = Float32[0, 1000]
        p3_adc.minAndMaxValue = Float32[0, 2200]
        mv_test = MedEye3d.ForDisplayStructs.MainMedEye3d(channel=Channel{Any}(8), states=states)
        MEH.apply_current_windowing_to_states!(mv_test.states)
        @test p1_ct.minAndMaxValue == Float32[-450.0, 1050.0]
        @test p1_pet.minAndMaxValue == Float32[0.5, 18.0]
        @test p2_t2.minAndMaxValue == Float32[120.0, 880.0]
        @test p3_adc.minAndMaxValue == Float32[500.0, 1950.0]

        # 6. sync_windowing_gui updates CT/MRI widgets cleanly without errors
        LMW._lmw_observables[:sync_windowing_gui]("CT", -1350.0f0, 150.0f0)
        @test LMW._lmw_observables[:tb_ct_min].stored_string[] == "-1350.0"
        @test LMW._lmw_observables[:tb_ct_max].stored_string[] == "150.0"
    finally
        empty!(MEH.tp_modalities); merge!(MEH.tp_modalities, old_tp_mods)
        empty!(MEH.current_windowing); merge!(MEH.current_windowing, old_win)
        empty!(MEH._user_windowing_modalities); union!(MEH._user_windowing_modalities, old_user_mods)
        if haskey(LMW._lmw_observables, :sync_windowing_gui)
            LMW._lmw_observables[:sync_windowing_gui]("CT", MEH.current_windowing["CT"][1], MEH.current_windowing["CT"][2])
            LMW._lmw_observables[:sync_windowing_gui]("PET", MEH.current_windowing["PET"][1], MEH.current_windowing["PET"][2])
            LMW._lmw_observables[:sync_windowing_gui]("T2", MEH.current_windowing["T2"][1], MEH.current_windowing["T2"][2])
            LMW._lmw_observables[:sync_windowing_gui]("ADC", MEH.current_windowing["ADC"][1], MEH.current_windowing["ADC"][2])
        end
        MEH.h5_path_ref[] = old_h5_path
        MEH.h5_save_path_ref[] = old_h5_save
        MEH.mri_main_multimodal_mode[] = old_mri_main
        MEH.mri_compare_mode[] = old_mri_cmp
        MEH.compare_mode[] = old_cmp
        MEH.current_tp_index[] = old_cur_tp
        if old_cfg_env === nothing
            delete!(ENV, "MEDEYE3D_DISPLAY_CONFIG_PATH")
        else
            ENV["MEDEYE3D_DISPLAY_CONFIG_PATH"] = old_cfg_env
        end
        try rm(tmp_cfg; force=true) catch; end
        try rm(tmp_h5; force=true) catch; end
        try rm(tmp_json; force=true) catch; end
    end
end

@testset "Test 18: Main Window MRI Multimodal Zoom-to-Fit Uses Unified T2 Reference Across Panels 1-4" begin
    old_mri_main = MEH.mri_main_multimodal_mode[]
    old_mri_cmp = MEH.mri_compare_mode[]
    old_tp_mods = copy(MEH.tp_modalities)
    old_cur_tp = MEH.current_tp_index[]
    try
        empty!(MEH.tp_modalities)
        MEH.tp_modalities[1] = "T2"
        MEH.tp_modalities[2] = "ADC"
        MEH.tp_modalities[3] = "DWI"
        MEH.tp_modalities[4] = "T1"
        MEH.current_tp_index[] = 1
        MEH.mri_main_multimodal_mode[] = true
        MEH.mri_compare_mode[] = false

        function _mk_mri_panel(tp_i, x_rng, y_rng)
            vol = zeros(Float32, 64, 64, 4)
            vol[x_rng, y_rng, :] .= 500.0f0
            calc = MEH.CalcDimsStruct(
                mainImageQuadVert = ones(Float32, 32),
                mainQuadVertSize = 128,
                imageTextureWidth = 64,
                imageTextureHeight = 64,
                zoom = 1.0f0, panX = 0.0f0, panY = 0.0f0
            )
            scr = MEH.FullScrollableDat(
                dataToScroll = MEH.ThreeDimRawDat[MEH.ThreeDimRawDat{Float32}(Float32, "CT", vol)],
                dimensionToScroll = 3,
                slicesNumber = Int32(4),
                currentTpIndex = tp_i
            )
            return MEH.StateDataFields(calcDimsStruct=calc, onScrollData=scr, currentDisplayedSlice=2)
        end

        # T2/ADC/DWI share smaller pelvic FOV (21..40), while T1 covers wider FOV (5..60)
        states = [
            _mk_mri_panel(1, 21:40, 21:40),
            _mk_mri_panel(2, 21:40, 21:40),
            _mk_mri_panel(3, 21:40, 21:40),
            _mk_mri_panel(4, 5:60, 5:60),
        ]

        MEH.reactToZoomToFit(MakieEvents.ZoomToFitEvent(1), states)
        @test states[1].calcDimsStruct.zoom > 2.0f0
        @test states[4].calcDimsStruct.zoom == states[1].calcDimsStruct.zoom
        @test states[4].calcDimsStruct.panX == states[1].calcDimsStruct.panX
        @test states[4].calcDimsStruct.panY == states[1].calcDimsStruct.panY
        @test states[2].calcDimsStruct.zoom == states[1].calcDimsStruct.zoom
        @test states[3].calcDimsStruct.zoom == states[1].calcDimsStruct.zoom

        # Second Z press resets all 4 panels uniformly back to 1.0x
        MEH.reactToZoomToFit(MakieEvents.ZoomToFitEvent(1), states)
        @test all(s -> s.calcDimsStruct.zoom == 1.0f0 && s.calcDimsStruct.panX == 0.0f0 && s.calcDimsStruct.panY == 0.0f0, states)
    finally
        MEH.mri_main_multimodal_mode[] = old_mri_main
        MEH.mri_compare_mode[] = old_mri_cmp
        empty!(MEH.tp_modalities); merge!(MEH.tp_modalities, old_tp_mods)
        MEH.current_tp_index[] = old_cur_tp
    end
end
