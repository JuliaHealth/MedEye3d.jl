using Test
using HDF5
using JSON
using MedEye3d
import MedEye3d.LesionMetadataWindow as LMW
import MedEye3d.SegmentationDisplay.MakieEventHandlers as MEH
import MedEye3d.InferenceClient

@testset "Heat-GDT Edge Modality & Modification Status Invariants" begin
    @test MEH.heatgdt_edge_modality[] === :ct_only

    # Test modification status read/write round-trip on temp HDF5
    tmp_h5 = tempname() * ".h5"
    h5open(tmp_h5, "w") do f
        meta = create_group(f, "_meta_")
        LMW.write_h5_modification_status!(meta, false, ""; ts="")
    end
    m1, u1, t1 = LMW.read_h5_modification_status(tmp_h5)
    @test m1 == false
    @test u1 == ""
    @test t1 == ""

    h5open(tmp_h5, "r+") do f
        LMW.write_h5_modification_status!(f["_meta_"], true, "test_reviewer"; ts="2026-11-17T20:00:00")
    end
    m2, u2, t2 = LMW.read_h5_modification_status(tmp_h5)
    @test m2 == true
    @test u2 == "test_reviewer"
    @test t2 == "2026-11-17T20:00:00"

    # Test per-modality diffusivity loading (:ct_only default, :combined, :pet_only)
    h5open(tmp_h5, "r+") do f
        bl = create_group(f, "BASELINE")
        bl["Fixed_CT_Volume_0.nii.gz"] = zeros(Float32, 4, 4, 4)
        bl["diffusivity"] = fill(Float32(0.11), 4, 4, 4)
        bl["diffusivity_ct"] = fill(Float32(0.22), 4, 4, 4)
        bl["diffusivity_pet"] = fill(Float32(0.33), 4, 4, 4)
    end
    old_ref = MEH.h5_path_ref[]
    old_save = MEH.h5_save_path_ref[]
    old_mod = MEH.heatgdt_edge_modality[]
    try
        empty!(MEH._tp_diffusivity_cache)
        MEH.h5_path_ref[] = tmp_h5
        MEH.h5_save_path_ref[] = tmp_h5

        MEH.heatgdt_edge_modality[] = :ct_only
        MEH.reload_heatgdt_diffusivity_for_tp!(0; force=true)
        @test MEH.heatgdt_current_loaded_modality[] === :ct_only
        @test isapprox(InferenceClient._heatgdt_diffusivity[][1], 0.22f0; atol=1e-5)

        MEH.heatgdt_edge_modality[] = :combined
        MEH.reload_heatgdt_diffusivity_for_tp!(0; force=true)
        @test MEH.heatgdt_current_loaded_modality[] === :combined
        @test isapprox(InferenceClient._heatgdt_diffusivity[][1], 0.11f0; atol=1e-5)

        MEH.heatgdt_edge_modality[] = :pet_only
        MEH.reload_heatgdt_diffusivity_for_tp!(0; force=true)
        @test MEH.heatgdt_current_loaded_modality[] === :pet_only
        @test isapprox(InferenceClient._heatgdt_diffusivity[][1], 0.33f0; atol=1e-5)
    finally
        MEH.h5_path_ref[] = old_ref
        MEH.h5_save_path_ref[] = old_save
        MEH.heatgdt_edge_modality[] = old_mod
        empty!(MEH._tp_diffusivity_cache)
        rm(tmp_h5; force=true)
    end
end

@testset "Prostata_bimodal Cohort HDF5 Pipeline Invariants" begin
    cohort_dir = joinpath(@__DIR__, "..", "data", "cases", "Prostata_bimodal")
    @test isdir(cohort_dir)

    pat_dirs = sort(
        filter(d -> occursin(r"^Pat\d+$", d) && isfile(joinpath(cohort_dir, d, "preprocessed_volumes.h5")), readdir(cohort_dir)),
        by = d -> parse(Int, match(r"\d+", d).match)
    )
    @test length(pat_dirs) >= 117

    excluded_substrings = (
        "torso_fat", "oblique_hyoid", "facial_bone", "nasal_cavity",
        "_temp", "helper", "dilation", "buffer", "_cavity", "chest_wall",
        "exclusion_zone", "skeleton"
    )

    for pat in pat_dirs
        h5p = joinpath(cohort_dir, pat, "preprocessed_volumes.h5")
        h5open(h5p, "r") do f
            @test haskey(f, "BASELINE")
            @test haskey(f, "ATLAS")
            @test haskey(f, "_meta_")
            bl = f["BASELINE"]

            # All 3 edge diffusivities present
            @test haskey(bl, "diffusivity")
            @test haskey(bl, "diffusivity_ct")
            @test haskey(bl, "diffusivity_pet")

            # Clean unmodified state & zero *_expert datasets
            mod_flag, mod_user, _ = LMW.read_h5_modification_status(h5p)
            @test mod_flag == false
            @test isempty(mod_user)
            @test !any(endswith(k, "_expert") for k in keys(bl))

            # Lesions & radiological descriptions present (+ zero first-2/last-2 Z-slice artifact lesions)
            les = read(bl["PET_Lesions_0.nii.gz"])
            @test any(>(0), les)
            @test size(les, 3) > 4
            @test all(==(0), @view les[:, :, 1:2])
            @test all(==(0), @view les[:, :, (end - 1):end])
            meta = f["_meta_"]
            @test haskey(meta, "metadata.json")
            md = JSON.parse(String(read(meta["metadata.json"])))
            has_desc = !isempty(strip(String(get(md, "EnglishDescription", "")))) ||
                any(v -> v isa AbstractDict && !isempty(strip(String(get(v, "EnglishDescription", "")))), values(md))
            @test has_desc
            @test haskey(meta, "semantic_filter.json")
            sf = JSON.parse(String(read(meta["semantic_filter.json"])))
            @test haskey(sf, "mentioned_max_anatomy")
            @test haskey(sf, "allowed_lesion_ids_by_tp")
            if has_desc
                @test !isempty(sf["mentioned_max_anatomy"])
            end
            @test haskey(meta, "organ_mapping")
            @test haskey(meta, "segment_names.json")
            @test haskey(meta, "matches.json")

            # Anatomy label sanity (mirrors cads_anatomy_wiring.is_excluded_non_organ_mask; allows temporalis/temporal_lobe)
            lbls = JSON.parse(String(read(meta["max_anatomy_labels.json"])))
            for (_, v) in pairs(lbls)
                vl = lowercase(strip(String(v)))
                is_bad = vl == "temp" || startswith(vl, "temp_") || endswith(vl, "_plane") ||
                    occursin("_plane_", vl) || any(p -> occursin(p, vl), excluded_substrings)
                @test !is_bad
            end
        end
    end
end

@testset "Semantic Filter Non-Destructive Masking & Organ Matching" begin
    mentions = ["prostate", "nvseg_prostate", "iliac_left", "thyroid_gland", "liver_segment_6", "hip_left"]
    @test MEH.organ_matches_semantic_mentions("Prostate", mentions)
    @test MEH.organ_matches_semantic_mentions("nvseg_prostate", mentions)
    @test MEH.organ_matches_semantic_mentions("Prostatic Urethra", mentions)
    @test MEH.organ_matches_semantic_mentions("Left Iliac Lymph Node", mentions)
    @test MEH.organ_matches_semantic_mentions("Thyroid Gland", mentions)
    @test MEH.organ_matches_semantic_mentions("nvseg_liver", mentions)
    @test MEH.organ_matches_semantic_mentions("pelvis_bone_left", mentions)
    @test !MEH.organ_matches_semantic_mentions("nvseg_right_hip", mentions)
    @test !MEH.organ_matches_semantic_mentions("Urinary Bladder", mentions)
    @test !MEH.organ_matches_semantic_mentions("Unknown", mentions)

    # Synthetic HDF5 + TpCacheEntry non-destructive toggle test
    tmp_sf_h5 = tempname() * ".h5"
    h5open(tmp_sf_h5, "w") do f
        meta = create_group(f, "_meta_")
        meta["semantic_filter.json"] = JSON.json(Dict(
            "version" => "1.0",
            "model" => "qwen3.8-27b",
            "mentioned_max_anatomy" => ["prostate"],
            "allowed_lesion_ids_by_tp" => Dict("0" => [1]),
            "reasoning" => "Prostate tumor only"
        ))
    end

    # Snapshot globals to restore cleanly
    prev_active = MEH.semantic_filter_active[]
    prev_mentions = copy(MEH.semantic_mentioned_organs[])
    prev_reason = MEH.semantic_reasoning[]
    prev_allowed = deepcopy(MEH.semantic_allowed_lesion_ids)
    prev_unfilt = copy(MEH.unfiltered_lesion_masks)
    prev_entry0 = get(MEH.tp_data_cache, 0, nothing)

    try
        h5open(tmp_sf_h5, "r") do f
            MEH.load_semantic_filter_from_h5!(f)
        end
        @test MEH.semantic_filter_active[] == false
        @test MEH.semantic_mentioned_organs[] == ["prostate"]
        @test MEH.semantic_allowed_lesion_ids[0] == Set([1])
        @test MEH.is_semantic_lesion_allowed(0, 2)  # true when filter OFF

        m_i16 = zeros(Int16, 4, 4, 4)
        m_i16[1, 1, 2] = Int16(1)
        m_i16[2, 2, 2] = Int16(2)
        m_i8 = Int8.(m_i16)
        ent = MEH.TpCacheEntry(
            zeros(Float32, 4, 4, 4),
            zeros(Float32, 4, 4, 4),
            m_i8,
            zeros(Int8, 4, 4, 4),
            nothing,
            m_i16,
            nothing,
        )
        MEH.tp_data_cache[0] = ent

        # Toggle ON -> lesion 2 masked out in display arrays, kept in unfiltered backup & save helper
        MEH.apply_semantic_filter!(true)
        @test MEH.semantic_filter_active[] == true
        @test !MEH.is_semantic_lesion_allowed(0, 2)
        @test ent.mask_i16[1, 1, 2] == Int16(1)
        @test ent.mask_i16[2, 2, 2] == Int16(0)
        @test ent.mask[2, 2, 2] == Int8(0)
        @test MEH.get_unfiltered_mask_for_save(0, ent)[2, 2, 2] == Int16(2)

        # Simulate editing kept lesion 1 while filter ON, then toggle OFF -> restores lesion 2 AND keeps edit!
        ent.mask_i16[1, 2, 2] = Int16(1)
        MEH.apply_semantic_filter!(false)
        @test MEH.semantic_filter_active[] == false
        @test ent.mask_i16[1, 1, 2] == Int16(1)
        @test ent.mask_i16[1, 2, 2] == Int16(1)
        @test ent.mask_i16[2, 2, 2] == Int16(2)
        @test ent.mask[2, 2, 2] == Int8(2)
        @test !haskey(MEH.unfiltered_lesion_masks, 0)

        # Verify loading real preprocessed HDF5 semantic_filter.json (Pat51)
        pat51_h5 = joinpath(@__DIR__, "..", "data", "cases", "Prostata_bimodal", "Pat51", "preprocessed_volumes.h5")
        loaded_orgs = MEH.load_semantic_filter_from_path!(pat51_h5)
        @test !isempty(loaded_orgs)
        @test any(x -> occursin("prostate", lowercase(x)), loaded_orgs)
        @test !isempty(get(MEH.semantic_allowed_lesion_ids, 0, Set{Int}()))
    finally
        MEH.semantic_filter_active[] = prev_active
        MEH.semantic_mentioned_organs[] = prev_mentions
        MEH.semantic_reasoning[] = prev_reason
        empty!(MEH.semantic_allowed_lesion_ids)
        merge!(MEH.semantic_allowed_lesion_ids, prev_allowed)
        empty!(MEH.unfiltered_lesion_masks)
        merge!(MEH.unfiltered_lesion_masks, prev_unfilt)
        if prev_entry0 === nothing
            delete!(MEH.tp_data_cache, 0)
        else
            MEH.tp_data_cache[0] = prev_entry0
        end
        isfile(tmp_sf_h5) && rm(tmp_sf_h5; force=true)
    end
end
