using Test
using MedEye3d
using MedEye3d.DataStructs
using MedEye3d.ForDisplayStructs
using MedEye3d.StructsManag
using MedEye3d.MakieEvents
using MedEye3d.ReactOnMouseClickAndDrag
using MedEye3d.LesionMetadataWindow
using HDF5
using JSON

const MEH = MedEye3d.SegmentationDisplay.MakieEventHandlers
const LMW = LesionMetadataWindow

@testset "AI Mask Immutability and Versioning Tests" begin
    # Create temp HDF5 file
    tmp_dir = mktempdir()
    tmp_json = joinpath(tmp_dir, "test_annotations.json")

    @testset "1. Expert Edit Tracking Flags" begin
        # Test initial state of observables
        LMW.current_has_expert_edits[] = false
        LMW.current_seg_origin[] = "AI_PRESEGMENTATION"
        @test LMW.current_has_expert_edits[] == false
        @test LMW.current_seg_origin[] == "AI_PRESEGMENTATION"
    end

    @testset "2. mark_expert_correction! Sets Flags" begin
        LMW.current_has_expert_edits[] = false
        LMW.current_seg_origin[] = "AI_PRESEGMENTATION"

        LMW.mark_expert_correction!()
        @test LMW.current_has_expert_edits[] == true
        @test LMW.current_seg_origin[] == "EXPERT_CORRECTION"
    end

    @testset "3. mark_prompt_segmentation! Sets Origin" begin
        LMW.current_has_expert_edits[] = false
        LMW.current_seg_origin[] = "AI_PRESEGMENTATION"

        LMW.mark_prompt_segmentation!()
        @test LMW.current_seg_origin[] == "PROMPT_SEGMENTATION"
    end

    @testset "4. mark_reverted_to_ai! Resets Flags" begin
        # Set to expert correction first
        LMW.mark_expert_correction!()
        @test LMW.current_has_expert_edits[] == true
        @test LMW.current_seg_origin[] == "EXPERT_CORRECTION"

        # Revert to AI
        LMW.mark_reverted_to_ai!()
        @test LMW.current_has_expert_edits[] == false
        @test LMW.current_seg_origin[] == "AI_PRESEGMENTATION"
    end

    @testset "5. Segmentation Origin Lifecycle" begin
        # Simulate a full lifecycle: AI -> Expert Edit -> Prompt Segment -> Revert
        LMW.mark_reverted_to_ai!()
        @test LMW.current_seg_origin[] == "AI_PRESEGMENTATION"
        @test LMW.current_has_expert_edits[] == false

        # Expert edits the mask
        LMW.mark_expert_correction!()
        @test LMW.current_seg_origin[] == "EXPERT_CORRECTION"
        @test LMW.current_has_expert_edits[] == true

        # User runs prompt segmentation
        LMW.mark_prompt_segmentation!()
        @test LMW.current_seg_origin[] == "PROMPT_SEGMENTATION"

        # Expert edits the result
        LMW.mark_expert_correction!()
        @test LMW.current_seg_origin[] == "EXPERT_CORRECTION"
        @test LMW.current_has_expert_edits[] == true

        # Revert everything
        LMW.mark_reverted_to_ai!()
        @test LMW.current_seg_origin[] == "AI_PRESEGMENTATION"
        @test LMW.current_has_expert_edits[] == false
    end

    @testset "6. Observable Registry Contains Expert Edit Keys" begin
        @test haskey(LMW._lmw_observables, :has_expert_edits)
        @test haskey(LMW._lmw_observables, :seg_origin)
        @test LMW._lmw_observables[:has_expert_edits] === LMW.current_has_expert_edits
        @test LMW._lmw_observables[:seg_origin] === LMW.current_seg_origin
    end

    @testset "7. Metadata Persistence to JSON" begin
        test_db = Dict{String,Any}(
            "5" => Dict{String, Any}(
                "LesionType" => "Bone Meta",
                "ObservationState" => "CORRECTED",
                "has_expert_edits" => true,
                "SegmentationOrigin" => "EXPERT_CORRECTION"
            ),
            "6" => Dict{String, Any}(
                "LesionType" => "Organ Meta",
                "ObservationState" => "UNREVIEWED",
                "has_expert_edits" => false,
                "SegmentationOrigin" => "AI_PRESEGMENTATION"
            )
        )

        # Save & load JSON
        LMW.save_annotations(test_db, tmp_json)
        loaded_json = LMW.load_annotations(tmp_json)
        @test loaded_json["5"]["has_expert_edits"] == true
        @test loaded_json["5"]["SegmentationOrigin"] == "EXPERT_CORRECTION"
        @test loaded_json["6"]["has_expert_edits"] == false
        @test loaded_json["6"]["SegmentationOrigin"] == "AI_PRESEGMENTATION"
    end

    @testset "8. Metadata Persistence to HDF5" begin
        test_db = Dict{String,Any}(
            "5" => Dict{String, Any}(
                "LesionType" => "Bone Meta",
                "ObservationState" => "CORRECTED",
                "has_expert_edits" => true,
                "SegmentationOrigin" => "EXPERT_CORRECTION"
            ),
            "6" => Dict{String, Any}(
                "LesionType" => "Organ Meta",
                "ObservationState" => "UNREVIEWED",
                "has_expert_edits" => false,
                "SegmentationOrigin" => "AI_PRESEGMENTATION"
            )
        )

        test_h5_annot = joinpath(tmp_dir, "test_annot.h5")
        LMW.save_annotations_hdf5(test_db, test_h5_annot)
        loaded_h5 = LMW.load_annotations_hdf5(test_h5_annot)
        @test (loaded_h5["5"]["has_expert_edits"] == true || loaded_h5["5"]["has_expert_edits"] == "true")
        @test loaded_h5["5"]["SegmentationOrigin"] == "EXPERT_CORRECTION"
        @test (loaded_h5["6"]["has_expert_edits"] == false || loaded_h5["6"]["has_expert_edits"] == "false")
        @test loaded_h5["6"]["SegmentationOrigin"] == "AI_PRESEGMENTATION"
    end

    @testset "9. get_lesion_state defaults" begin
        mock_db = Dict{String, Any}(
            "5" => Dict{String, Any}(
                "LesionType" => "Bone Meta",
                "ObservationState" => "UNREVIEWED"
            )
        )
        st = LMW.get_lesion_state(mock_db, "5")
        seg_orig = get(st, "SegmentationOrigin", "AI_PRESEGMENTATION")
        @test seg_orig == "AI_PRESEGMENTATION"
        @test get(st, "has_expert_edits", false) == false
    end

    # Cleanup
    rm(tmp_dir, recursive=true, force=true)
end
