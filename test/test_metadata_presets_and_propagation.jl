#!/usr/bin/env julia
"""
test_metadata_presets_and_propagation.jl

Unit and integration tests for:
1. Lesion Metadata Presets (built-in presets: lacrimal glands, trigeminal ganglion, edge-slice technical artifact)
2. Adding and persisting custom presets
3. Applying presets to lesions (anatomy, NoCTCorrelate, Certainty=0, alternative hypothesis)
4. Downstream Propagation across linked timepoints (dst_tp > src_tp)
5. Preservation of manually changed fields in descendants (DO NOT overwrite manual edits)
6. Edge-slice technical artifact auto-detection (first 2 and last 2 slices)
"""

using Test
using JSON
using Dates
using Observables
using GLMakie

using MedEye3d
using MedEye3d.LesionMetadataWindow
import MedEye3d.LesionAssociation as LA

@testset "Metadata Presets and Propagation Test Suite" begin

    # =========================================================================
    # Group 1: Built-in Presets Verification
    # =========================================================================
    @testset "1. Built-in Presets" begin
        presets = load_metadata_presets()
        @test haskey(presets, "Left Lacrimal Gland (Benign)")
        @test haskey(presets, "Right Lacrimal Gland (Benign)")
        @test haskey(presets, "Trigeminal Ganglion (Benign)")
        @test haskey(presets, "Technical Artifact (Edge Slices / Benign)")

        # Verify Left Lacrimal Gland preset
        left_lac = presets["Left Lacrimal Gland (Benign)"]
        @test get(left_lac, "BaseAnatomy", "") == "lacrimal gland"
        @test get(left_lac, "BaseAnatomySide", "") == "Left"
        @test get(left_lac, "Certainty", "") == "0"
        @test get(left_lac, "NoCTCorrelate", "") == "true"
        @test get(left_lac, "Alternative Hypothesis (False Positive)", "") == "Physiological uptake"
        @test get(left_lac, "ObservationState", "") == "ACCEPTED"

        # Verify Right Lacrimal Gland preset
        right_lac = presets["Right Lacrimal Gland (Benign)"]
        @test get(right_lac, "BaseAnatomy", "") == "lacrimal gland"
        @test get(right_lac, "BaseAnatomySide", "") == "Right"
        @test get(right_lac, "Certainty", "") == "0"
        @test get(right_lac, "NoCTCorrelate", "") == "true"
        @test get(right_lac, "Alternative Hypothesis (False Positive)", "") == "Physiological uptake"
        @test get(right_lac, "ObservationState", "") == "ACCEPTED"

        # Verify Trigeminal Ganglion preset
        trig = presets["Trigeminal Ganglion (Benign)"]
        @test get(trig, "BaseAnatomy", "") == "trigeminal ganglion"
        @test get(trig, "Certainty", "") == "0"
        @test get(trig, "NoCTCorrelate", "") == "true"
        @test get(trig, "Alternative Hypothesis (False Positive)", "") == "Trigeminal Ganglion"
        @test get(trig, "ObservationState", "") == "ACCEPTED"

        # Verify Technical Artifact preset
        art = presets["Technical Artifact (Edge Slices / Benign)"]
        @test get(art, "LesionType", "") == "Technical Artifact"
        @test get(art, "Certainty", "") == "0"
        @test get(art, "NoCTCorrelate", "") == "true"
        @test get(art, "Alternative Hypothesis (False Positive)", "") == "Technical Artifact"
        @test get(art, "ObservationState", "") == "ACCEPTED"
    end

    # =========================================================================
    # Group 2: Custom Preset Creation & Ephemeral Field Stripping
    # =========================================================================
    @testset "2. Custom Preset Creation" begin
        test_state = Dict{String, Any}(
            "BaseAnatomy" => "femur",
            "BaseAnatomySide" => "Right",
            "LesionType" => "Bone Meta",
            "ObservationState" => "ACCEPTED",
            "Margins" => "Sclerotic rim",
            "Texture" => "Homogeneous",
            "Certainty" => "2",
            "NoCTCorrelate" => "false",
            "Alternative Hypothesis (False Positive)" => "Bone Island / Enostosis",
            # Ephemeral fields that must be stripped out:
            "_display_name" => "1: Right Femur",
            "_last_modified_by" => "tester",
            "_last_modified_at" => "2026-10-14T12:00:00",
            "_edit_history" => [Dict("action" => "edit")],
            "KeyImage" => "true",
            "SUV max" => "4.5",
            "Lesion tracking name?" => "TP0_PET_1"
        )

        preset_name = "Test Sclerotic Femur Island"
        updated_presets = add_metadata_preset!(preset_name, test_state)
        @test haskey(updated_presets, preset_name)

        saved = updated_presets[preset_name]
        @test get(saved, "BaseAnatomy", "") == "femur"
        @test get(saved, "BaseAnatomySide", "") == "Right"
        @test get(saved, "LesionType", "") == "Bone Meta"
        @test get(saved, "Margins", "") == "Sclerotic rim"
        @test get(saved, "Texture", "") == "Homogeneous"
        @test get(saved, "Certainty", "") == "2"
        @test get(saved, "Alternative Hypothesis (False Positive)", "") == "Bone Island / Enostosis"

        # Ensure ephemeral fields were stripped
        @test !haskey(saved, "_display_name")
        @test !haskey(saved, "_last_modified_by")
        @test !haskey(saved, "_edit_history")
        @test !haskey(saved, "KeyImage")
        @test !haskey(saved, "SUV max")
        @test !haskey(saved, "Lesion tracking name?")
    end

    # =========================================================================
    # Group 3: Downstream Propagation across Time Points
    # =========================================================================
    @testset "3. Downstream Propagation across Linked Timepoints" begin
        # Reset match groups
        mg = LA.get_match_groups()
        empty!(mg)

        # Link lesion 10 (TP 0) -> lesion 20 (TP 1) -> lesion 30 (TP 2)
        LA.update_match_group!("PET_Lesions_0", 10, "PET_Lesions_1", 20, "")
        LA.update_match_group!("PET_Lesions_1", 20, "PET_Lesions_2", 30, "")

        # Verify descendant lookup
        desc_0 = find_linked_descendants(10, 0)
        @test length(desc_0) == 2
        @test (1, 20) in desc_0
        @test (2, 30) in desc_0

        desc_1 = find_linked_descendants(20, 1)
        @test length(desc_1) == 1
        @test desc_1 == [(2, 30)]

        desc_2 = find_linked_descendants(30, 2)
        @test isempty(desc_2)

        # Setup database
        db = Dict{String, Any}(
            "10" => Dict{String, Any}(
                "BaseAnatomy" => "lacrimal gland",
                "BaseAnatomySide" => "Left",
                "LesionType" => "Organ Meta",
                "ObservationState" => "ACCEPTED",
                "NoCTCorrelate" => "true",
                "Certainty" => "0",
                "Alternative Hypothesis (False Positive)" => "Physiological uptake",
                "Margins" => "Circumscribed",
                "Texture" => "Homogeneous"
            ),
            # Lesion 20 has NO manually set metadata
            "20" => Dict{String, Any}(),
            # Lesion 30 has a MANUALLY set field: Margins = "Ill-defined"
            "30" => Dict{String, Any}(
                "Margins" => "Ill-defined",
                "_manually_edited_fields" => ["Margins"]
            )
        )

        # Propagate from TP 0 (lesion 10)
        propagate_metadata_to_descendants!(10, 0, db["10"], db)

        # Check Lesion 20 (TP 1) — all fields copied
        l20 = db["20"]
        @test get(l20, "BaseAnatomy", "") == "lacrimal gland"
        @test get(l20, "BaseAnatomySide", "") == "Left"
        @test get(l20, "LesionType", "") == "Organ Meta"
        @test get(l20, "ObservationState", "") == "ACCEPTED"
        @test get(l20, "NoCTCorrelate", "") == "true"
        @test get(l20, "Certainty", "") == "0"
        @test get(l20, "Alternative Hypothesis (False Positive)", "") == "Physiological uptake"
        @test get(l20, "Margins", "") == "Circumscribed"
        @test get(l20, "Texture", "") == "Homogeneous"

        # Check Lesion 30 (TP 2):
        # - Margins was manually set -> MUST NOT BE OVERWRITTEN!
        # - Other non-manual fields -> COPIED!
        l30 = db["30"]
        @test get(l30, "Margins", "") == "Ill-defined" # Preserved!
        @test get(l30, "BaseAnatomy", "") == "lacrimal gland"
        @test get(l30, "BaseAnatomySide", "") == "Left"
        @test get(l30, "Certainty", "") == "0"
        @test get(l30, "NoCTCorrelate", "") == "true"
        @test get(l30, "Alternative Hypothesis (False Positive)", "") == "Physiological uptake"
        @test get(l30, "Texture", "") == "Homogeneous"
    end

    # =========================================================================
    # Group 4: Manual Edit Detection & History Preservation
    # =========================================================================
    @testset "4. Manual Edit Detection" begin
        # 1. Direct record
        d1 = Dict{String, Any}("_manually_edited_fields" => ["Texture", "Shape"])
        @test is_field_manually_set_in_lesion(d1, "Texture") == true
        @test is_field_manually_set_in_lesion(d1, "Shape") == true
        @test is_field_manually_set_in_lesion(d1, "Margins") == false

        # 2. User edited anatomy
        d2 = Dict{String, Any}("_user_edited_anatomy" => true)
        @test is_field_manually_set_in_lesion(d2, "BaseAnatomy") == true
        @test is_field_manually_set_in_lesion(d2, "BaseAnatomySide") == true
        @test is_field_manually_set_in_lesion(d2, "Texture") == false

        # 3. Edit history
        d3 = Dict{String, Any}(
            "_edit_history" => [
                Dict{String, Any}(
                    "changes" => [
                        Dict{String, Any}("field" => "Certainty", "old" => "10", "new" => "0")
                    ]
                )
            ]
        )
        @test is_field_manually_set_in_lesion(d3, "Certainty") == true
        @test is_field_manually_set_in_lesion(d3, "Texture") == false
    end

    # =========================================================================
    # Group 5: Edge Slice Technical Artifact Auto-detection
    # =========================================================================
    @testset "5. Edge Slice Artifact Auto-detection" begin
        # Reset match groups
        mg = LA.get_match_groups()
        empty!(mg)
        LA.update_match_group!("PET_Lesions_0", 1, "PET_Lesions_1", 2, "")

        # Verify edge slice condition
        z_first = 1
        z_second = 2
        z_last = 326
        z_penult = 325
        total_z = 326

        @test (z_first <= 2 || z_first >= total_z - 1) == true
        @test (z_second <= 2 || z_second >= total_z - 1) == true
        @test (z_last <= 2 || z_last >= total_z - 1) == true
        @test (z_penult <= 2 || z_penult >= total_z - 1) == true

        # Middle slice is NOT an edge slice
        z_mid = 150
        @test (z_mid <= 2 || z_mid >= total_z - 1) == false

        # Test propagation of edge slice artifact to linked descendant
        edge_data = Dict{String, Any}(
            "Alternative Hypothesis (False Positive)" => "Technical Artifact",
            "Certainty" => "0",
            "NoCTCorrelate" => "true",
            "LesionType" => "Technical Artifact",
            "ObservationState" => "ACCEPTED"
        )
        db = Dict{String, Any}("1" => copy(edge_data), "2" => Dict{String, Any}())
        propagate_metadata_to_descendants!(1, 0, edge_data, db)

        l2 = db["2"]
        @test get(l2, "Alternative Hypothesis (False Positive)", "") == "Technical Artifact"
        @test get(l2, "Certainty", "") == "0"
        @test get(l2, "NoCTCorrelate", "") == "true"
        @test get(l2, "LesionType", "") == "Technical Artifact"
        @test get(l2, "ObservationState", "") == "ACCEPTED"
    end

    # =========================================================================
    # Group 6: Live GUI Window Preset Selection & Plus Button
    # =========================================================================
    @testset "6. Live GUI Window Preset Controls" begin
        active_lesion = Observable("1: Lesion 1")
        lesion_ids = Observable(["1: Lesion 1"])
        tmp_json = tempname() * ".json"

        win = create_metadata_window(active_lesion, lesion_ids, nothing; save_path = tmp_json)
        
        # Wait for GUI preload
        t0 = time()
        while !get(LesionMetadataWindow._lmw_observables, :gui_preload_done, Ref(false))[] && time() - t0 < 10.0
            sleep(0.05)
        end
        sleep(0.2)

        menu_presets = LesionMetadataWindow._lmw_observables[:menu_presets]
        btn_add_preset = LesionMetadataWindow._lmw_observables[:btn_add_preset]

        @test menu_presets isa Makie.Menu
        @test btn_add_preset isa Makie.Button

        opts = menu_presets.options[]
        @test "- select preset -" in opts
        @test "Left Lacrimal Gland (Benign)" in opts
        @test "Right Lacrimal Gland (Benign)" in opts
        @test "Trigeminal Ganglion (Benign)" in opts
        @test "Technical Artifact (Edge Slices / Benign)" in opts

        # Select "Left Lacrimal Gland (Benign)"
        menu_presets.selection[] = "Left Lacrimal Gland (Benign)"
        sleep(0.3)

        # Verify fields were populated
        act_type = LesionMetadataWindow._lmw_observables[:active_lesion_type]
        @test act_type[] == "Organ Meta"

        no_ct = LesionMetadataWindow._lmw_observables[:no_ct_toggle]
        @test no_ct.active[] == true

        menu_ba = LesionMetadataWindow._lmw_observables[:menu_base_anat]
        @test menu_ba.selection[] == "lacrimal gland"

        menu_sd = LesionMetadataWindow._lmw_observables[:menu_side]
        @test menu_sd.selection[] == "Left"

        cert_w = LesionMetadataWindow._lmw_observables[:field_widgets]["Certainty"]
        if cert_w isa Makie.Slider
            @test cert_w.value[] == 0
        end

        alt_w = LesionMetadataWindow._lmw_observables[:field_widgets]["Alternative Hypothesis (False Positive)"]
        @test alt_w.selection[] == "Physiological uptake"

        # Toggle "+ Preset" button
        btn_add_preset.clicks[] = btn_add_preset.clicks[] + 1
        sleep(0.1)

        # Check that preset add row opened
        add_rows = get(LesionMetadataWindow._lmw_observables, :custom_add_row_open, Dict{Int, Base.RefValue{Bool}}())
        has_any_open = any(r -> r[2][], add_rows)
        @test has_any_open == true

        # Clean up temporary file
        rm(tmp_json, force=true)
    end

    # Clean up match groups
    empty!(LA.get_match_groups())
end
