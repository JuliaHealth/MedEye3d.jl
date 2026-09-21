using Test
using MedEye3d
using MedEye3d.MakieEvents
using MedEye3d.ScientificWorkflow
using MedEye3d.ConflictChecker
using MedEye3d.ResearchExport
using JSON

@testset "New Features Comprehensive Tests" begin

    # ============================================================================
    # 1. ScientificWorkflow Data Model Tests
    # ============================================================================
    @testset "ScientificWorkflow Data Model" begin

        @testset "LesionObservationState enum (7 values)" begin
            @test UNREVIEWED isa LesionObservationState
            @test ACCEPTED isa LesionObservationState
            @test REJECTED isa LesionObservationState
            @test CORRECTED isa LesionObservationState
            @test UNCERTAIN isa LesionObservationState
            @test NEW isa LesionObservationState
            @test RESOLVED isa LesionObservationState
            @test length(instances(LesionObservationState)) == 7
        end

        @testset "AnnotationWorkflowState enum (7 values)" begin
            @test WF_CASE_LOADING isa AnnotationWorkflowState
            @test WF_LESION_REVIEW isa AnnotationWorkflowState
            @test WF_EDIT_MASK isa AnnotationWorkflowState
            @test WF_PROMPT_SEGMENT isa AnnotationWorkflowState
            @test WF_REGISTRATION_REVIEW isa AnnotationWorkflowState
            @test WF_CASE_QC isa AnnotationWorkflowState
            @test WF_CASE_COMPLETE isa AnnotationWorkflowState
            @test length(instances(AnnotationWorkflowState)) == 7
        end

        @testset "ClinicalPhase enum (6 values)" begin
            @test PHASE_CASE_SETUP isa ClinicalPhase
            @test PHASE_READ isa ClinicalPhase
            @test PHASE_ASSESS isa ClinicalPhase
            @test PHASE_REPORT_DRAFT isa ClinicalPhase
            @test PHASE_VALIDATION isa ClinicalPhase
            @test PHASE_SIGNED isa ClinicalPhase
            @test length(instances(ClinicalPhase)) == 6
        end

        @testset "CaseProfile enum (6 values)" begin
            @test PROFILE_INITIAL_STAGING isa CaseProfile
            @test PROFILE_BCR isa CaseProfile
            @test PROFILE_PRE_RLT isa CaseProfile
            @test PROFILE_POST_RLT isa CaseProfile
            @test PROFILE_RESPONSE isa CaseProfile
            @test PROFILE_GENERAL isa CaseProfile
            @test length(instances(CaseProfile)) == 6
        end

        @testset "RegistrationQC struct" begin
            # Default keyword constructor
            rqc = RegistrationQC()
            @test rqc.status == "UNREVIEWED"
            @test rqc.offset_x == 0.0f0
            @test rqc.offset_y == 0.0f0
            @test rqc.offset_z == 0.0f0
            @test rqc.comment == ""

            # Custom values
            rqc2 = RegistrationQC(status="GOOD", offset_x=1.5, offset_y=-2.0, offset_z=0.3, comment="looks fine")
            @test rqc2.status == "GOOD"
            @test rqc2.offset_x ≈ 1.5f0
            @test rqc2.offset_y ≈ -2.0f0
            @test rqc2.offset_z ≈ 0.3f0
            @test rqc2.comment == "looks fine"

            # Positional constructor
            rqc3 = RegistrationQC("QUESTIONABLE", 0.0, 0.0, 0.0, "shifted")
            @test rqc3.status == "QUESTIONABLE"
            @test rqc3.comment == "shifted"
        end

        @testset "SegmentationVersion struct" begin
            # Default keyword constructor
            sv = SegmentationVersion()
            @test sv.version_id == ""
            @test sv.timestamp == ""
            @test sv.author == ""
            @test sv.origin_model == ""
            @test sv.mask_reference === nothing

            # Custom values
            sv2 = SegmentationVersion(
                version_id="v001",
                timestamp="2026-09-21T05:00:00",
                author="expert_1",
                origin_model="nnUNet",
                mask_reference="path/to/mask"
            )
            @test sv2.version_id == "v001"
            @test sv2.author == "expert_1"
            @test sv2.origin_model == "nnUNet"
            @test sv2.mask_reference == "path/to/mask"
        end

        @testset "LesionObservation struct" begin
            # Default keyword constructor
            lo = LesionObservation()
            @test lo.observation_id == ""
            @test lo.timepoint_index == 0
            @test lo.state == UNREVIEWED
            @test lo.active_mask_version == ""
            @test lo.versions isa Vector{SegmentationVersion}
            @test isempty(lo.versions)
            @test lo.registration_qc isa RegistrationQC
            @test lo.suv_max ≈ 0.0f0
            @test lo.suv_mean ≈ 0.0f0
            @test lo.volume_ml ≈ 0.0f0

            # Custom values
            sv = SegmentationVersion(version_id="v1", timestamp="2026-01-01", author="AI", origin_model="autopet", mask_reference=nothing)
            rqc = RegistrationQC(status="GOOD")
            lo2 = LesionObservation(
                observation_id="obs_1",
                timepoint_index=2,
                state=ACCEPTED,
                active_mask_version="v1",
                versions=[sv],
                registration_qc=rqc,
                suv_max=12.5,
                suv_mean=6.3,
                volume_ml=2.1
            )
            @test lo2.observation_id == "obs_1"
            @test lo2.timepoint_index == 2
            @test lo2.state == ACCEPTED
            @test length(lo2.versions) == 1
            @test lo2.suv_max ≈ 12.5f0
            @test lo2.suv_mean ≈ 6.3f0
            @test lo2.volume_ml ≈ 2.1f0

            # Mutable state change
            lo2.state = CORRECTED
            @test lo2.state == CORRECTED
        end

        @testset "LesionTrack struct" begin
            # Default keyword constructor
            lt = LesionTrack()
            @test lt.track_id == ""
            @test lt.lesion_class == ""
            @test lt.anatomy_label == ""
            @test lt.observations isa Dict{Int, LesionObservation}
            @test isempty(lt.observations)

            # With observations
            obs = LesionObservation(observation_id="obs1", timepoint_index=0, state=ACCEPTED)
            lt2 = LesionTrack(
                track_id="L5",
                lesion_class="Bone Meta",
                anatomy_label="femur_left",
                observations=Dict(0 => obs)
            )
            @test lt2.track_id == "L5"
            @test lt2.lesion_class == "Bone Meta"
            @test lt2.anatomy_label == "femur_left"
            @test haskey(lt2.observations, 0)
            @test lt2.observations[0].state == ACCEPTED

            # Three-arg constructor (no observations)
            lt3 = LesionTrack("L10", "Organ Meta", "liver")
            @test lt3.track_id == "L10"
            @test isempty(lt3.observations)
        end

        @testset "AnnotationWorkflowController struct" begin
            # Default keyword constructor
            awc = AnnotationWorkflowController()
            @test awc.case_id == ""
            @test awc.tracks isa Vector{LesionTrack}
            @test isempty(awc.tracks)
            @test awc.current_track_id == ""
            @test awc.current_tp_idx == 0

            # With tracks
            track1 = LesionTrack("L1", "Bone Meta", "rib")
            track2 = LesionTrack("L2", "Organ Meta", "liver")
            awc2 = AnnotationWorkflowController(
                case_id="CASE_001",
                tracks=[track1, track2],
                current_track_id="L1",
                current_tp_idx=0
            )
            @test awc2.case_id == "CASE_001"
            @test length(awc2.tracks) == 2
            @test awc2.current_track_id == "L1"
            @test awc2.current_tp_idx == 0

            # Mutable field change
            awc2.current_track_id = "L2"
            @test awc2.current_track_id == "L2"
        end
    end

    # ============================================================================
    # 2. MakieEvents Tests
    # ============================================================================
    @testset "MakieEvents Event Types" begin

        @testset "Lesion decision events" begin
            @test AcceptLesionEvent() isa AcceptLesionEvent
            @test RejectLesionEvent() isa RejectLesionEvent
            @test MarkUncertainEvent() isa MarkUncertainEvent
            @test MarkResolvedEvent() isa MarkResolvedEvent
        end

        @testset "Lesion navigation events" begin
            @test CenterLesionEvent() isa CenterLesionEvent
            @test NextLesionEvent() isa NextLesionEvent
            @test PrevLesionEvent() isa PrevLesionEvent
        end

        @testset "ToggleMaskVisibilityEvent" begin
            evt_show = ToggleMaskVisibilityEvent(true)
            @test evt_show.visible == true
            evt_hide = ToggleMaskVisibilityEvent(false)
            @test evt_hide.visible == false
        end

        @testset "RevertToAIEvent" begin
            evt1 = RevertToAIEvent(5)
            @test evt1.lesion_id == 5
            @test evt1.tp_index == 0  # default

            evt2 = RevertToAIEvent(10, 3)
            @test evt2.lesion_id == 10
            @test evt2.tp_index == 3
        end

        @testset "Registration events" begin
            @test FlagRegistrationEvent() isa FlagRegistrationEvent

            evt = SetRegistrationQCEvent("GOOD")
            @test evt.status == "GOOD"
            evt2 = SetRegistrationQCEvent("QUESTIONABLE")
            @test evt2.status == "QUESTIONABLE"
        end

        @testset "Flicker and Overlay events" begin
            # These are not exported but defined in the module
            @test MakieEvents.ToggleFlickerEvent() isa MakieEvents.ToggleFlickerEvent
            @test MakieEvents.ToggleOverlayEvent() isa MakieEvents.ToggleOverlayEvent
        end

        @testset "SetM2ReferenceEvent" begin
            evt_neg = SetM2ReferenceEvent(-1)
            @test evt_neg.tp_index == -1
            evt_zero = SetM2ReferenceEvent(0)
            @test evt_zero.tp_index == 0
        end

        @testset "Validate and QC events" begin
            @test ValidateReportEvent() isa ValidateReportEvent
            @test CaseQCEvent() isa CaseQCEvent
        end

        @testset "Phase events" begin
            @test NextPhaseEvent() isa NextPhaseEvent
            @test PrevPhaseEvent() isa PrevPhaseEvent

            evt = SetPhaseEvent("READ")
            @test evt.phase == "READ"
            evt2 = SetPhaseEvent("SIGNED")
            @test evt2.phase == "SIGNED"
        end
    end

    # ============================================================================
    # 3. ConflictChecker Tests
    # ============================================================================
    @testset "ConflictChecker Validation Rules" begin

        @testset "Empty lesion list returns no issues" begin
            report = Dict{String,Any}("final_mitnm" => "M0", "tmtv_cc" => 0.0)
            issues = run_conflict_checks(report, Dict{String,Any}[])
            @test isempty(issues)
        end

        @testset "UNREVIEWED_LESIONS rule (BLOCKING)" begin
            report = Dict{String,Any}("final_mitnm" => "M1b", "tmtv_cc" => 5.0)
            lesions = [
                Dict{String,Any}("LesionName" => "L1", "ObservationState" => "UNREVIEWED",
                                  "BaseAnatomy" => "Prostate", "LesionType" => "Primary")
            ]
            issues = run_conflict_checks(report, lesions)
            blocking = filter(i -> i.code == "UNREVIEWED_LESIONS", issues)
            @test length(blocking) == 1
            @test blocking[1].severity == BLOCKING
            @test contains(blocking[1].message, "1 unreviewed")
        end

        @testset "BONE_M_MISMATCH rule (BLOCKING)" begin
            report = Dict{String,Any}("final_mitnm" => "M0", "tmtv_cc" => 5.0)
            lesions = [
                Dict{String,Any}("LesionName" => "L2", "ObservationState" => "ACCEPTED",
                                  "BaseAnatomy" => "Bone", "LesionType" => "Bone Meta")
            ]
            issues = run_conflict_checks(report, lesions)
            bone_issues = filter(i -> i.code == "BONE_M_MISMATCH", issues)
            @test length(bone_issues) == 1
            @test bone_issues[1].severity == BLOCKING
            @test contains(bone_issues[1].details, "M0")
        end

        @testset "BONE_M_MISMATCH not triggered when M1b present" begin
            report = Dict{String,Any}("final_mitnm" => "T2N1M1b", "tmtv_cc" => 5.0)
            lesions = [
                Dict{String,Any}("LesionName" => "L2", "ObservationState" => "ACCEPTED",
                                  "BaseAnatomy" => "Bone", "LesionType" => "Bone Meta")
            ]
            issues = run_conflict_checks(report, lesions)
            bone_issues = filter(i -> i.code == "BONE_M_MISMATCH", issues)
            @test isempty(bone_issues)
        end

        @testset "NEW_LESION_PRESENT rule (WARNING)" begin
            report = Dict{String,Any}("final_mitnm" => "M1a", "tmtv_cc" => 3.0)
            lesions = [
                Dict{String,Any}("LesionName" => "L3", "ObservationState" => "NEW",
                                  "BaseAnatomy" => "Lung", "LesionType" => "Organ Meta")
            ]
            issues = run_conflict_checks(report, lesions)
            new_issues = filter(i -> i.code == "NEW_LESION_PRESENT", issues)
            @test length(new_issues) == 1
            @test new_issues[1].severity == WARNING
        end

        @testset "UNCERTAIN_LESIONS rule (WARNING)" begin
            report = Dict{String,Any}("final_mitnm" => "M1a", "tmtv_cc" => 2.0)
            lesions = [
                Dict{String,Any}("LesionName" => "L4", "ObservationState" => "UNCERTAIN",
                                  "BaseAnatomy" => "LN", "LesionType" => "LN Meta")
            ]
            issues = run_conflict_checks(report, lesions)
            uncertain_issues = filter(i -> i.code == "UNCERTAIN_LESIONS", issues)
            @test length(uncertain_issues) == 1
            @test uncertain_issues[1].severity == WARNING
        end

        @testset "LATERALITY_MISMATCH rule (WARNING)" begin
            report = Dict{String,Any}("final_mitnm" => "M1b", "tmtv_cc" => 5.0)
            lesions = [
                Dict{String,Any}("LesionName" => "L5", "ObservationState" => "ACCEPTED",
                                  "Side" => "Left", "BaseAnatomy" => "right femur",
                                  "LesionType" => "Bone Meta")
            ]
            issues = run_conflict_checks(report, lesions)
            lat_issues = filter(i -> i.code == "LATERALITY_MISMATCH", issues)
            @test length(lat_issues) == 1
            @test lat_issues[1].severity == WARNING
            @test contains(lat_issues[1].message, "L5")
        end

        @testset "LATERALITY_MISMATCH not triggered for Midline" begin
            report = Dict{String,Any}("final_mitnm" => "M1b", "tmtv_cc" => 5.0)
            lesions = [
                Dict{String,Any}("LesionName" => "L6", "ObservationState" => "ACCEPTED",
                                  "Side" => "Midline", "BaseAnatomy" => "right femur",
                                  "LesionType" => "Bone Meta")
            ]
            issues = run_conflict_checks(report, lesions)
            lat_issues = filter(i -> i.code == "LATERALITY_MISMATCH", issues)
            @test isempty(lat_issues)
        end

        @testset "REGISTRATION_ISSUES rule (INFORMATIONAL)" begin
            report = Dict{String,Any}("final_mitnm" => "M1a", "tmtv_cc" => 5.0)
            lesions = [
                Dict{String,Any}("LesionName" => "L7", "ObservationState" => "ACCEPTED",
                                  "RegistrationQC" => "QUESTIONABLE",
                                  "BaseAnatomy" => "Lung", "LesionType" => "Organ Meta")
            ]
            issues = run_conflict_checks(report, lesions)
            reg_issues = filter(i -> i.code == "REGISTRATION_ISSUES", issues)
            @test length(reg_issues) == 1
            @test reg_issues[1].severity == INFORMATIONAL
        end

        @testset "REGISTRATION_ISSUES for POOR_MANUAL_MATCH" begin
            report = Dict{String,Any}("final_mitnm" => "M1a", "tmtv_cc" => 5.0)
            lesions = [
                Dict{String,Any}("LesionName" => "L8", "ObservationState" => "ACCEPTED",
                                  "RegistrationQC" => "POOR_MANUAL_MATCH",
                                  "BaseAnatomy" => "Lung", "LesionType" => "Organ Meta")
            ]
            issues = run_conflict_checks(report, lesions)
            reg_issues = filter(i -> i.code == "REGISTRATION_ISSUES", issues)
            @test length(reg_issues) == 1
        end

        @testset "ZERO_TMTV rule (WARNING)" begin
            report = Dict{String,Any}("final_mitnm" => "M1a", "tmtv_cc" => 0.0)
            lesions = [
                Dict{String,Any}("LesionName" => "L9", "ObservationState" => "ACCEPTED",
                                  "BaseAnatomy" => "Lung", "LesionType" => "Organ Meta")
            ]
            issues = run_conflict_checks(report, lesions)
            tmtv_issues = filter(i -> i.code == "ZERO_TMTV", issues)
            @test length(tmtv_issues) == 1
            @test tmtv_issues[1].severity == WARNING
        end

        @testset "ZERO_TMTV not triggered when tmtv > 0" begin
            report = Dict{String,Any}("final_mitnm" => "M1a", "tmtv_cc" => 15.3)
            lesions = [
                Dict{String,Any}("LesionName" => "L9", "ObservationState" => "ACCEPTED",
                                  "BaseAnatomy" => "Lung", "LesionType" => "Organ Meta")
            ]
            issues = run_conflict_checks(report, lesions)
            tmtv_issues = filter(i -> i.code == "ZERO_TMTV", issues)
            @test isempty(tmtv_issues)
        end

        @testset "Clean case: no blocking issues" begin
            report = Dict{String,Any}("final_mitnm" => "T2N1M1b", "tmtv_cc" => 25.0)
            lesions = [
                Dict{String,Any}("LesionName" => "L1", "ObservationState" => "ACCEPTED",
                                  "BaseAnatomy" => "Bone", "LesionType" => "Bone Meta",
                                  "Side" => "Left", "RegistrationQC" => "GOOD"),
                Dict{String,Any}("LesionName" => "L2", "ObservationState" => "ACCEPTED",
                                  "BaseAnatomy" => "Lung", "LesionType" => "Organ Meta",
                                  "Side" => "Midline", "RegistrationQC" => "GOOD"),
                Dict{String,Any}("LesionName" => "L3", "ObservationState" => "CORRECTED",
                                  "BaseAnatomy" => "liver", "LesionType" => "Organ Meta",
                                  "Side" => "Midline", "RegistrationQC" => "GOOD")
            ]
            issues = run_conflict_checks(report, lesions)
            blocking_issues = filter(i -> i.severity == BLOCKING, issues)
            @test isempty(blocking_issues)
            # Also no warnings
            warning_issues = filter(i -> i.severity == WARNING, issues)
            @test isempty(warning_issues)
        end

        @testset "Multiple issues combined" begin
            report = Dict{String,Any}("final_mitnm" => "M0", "tmtv_cc" => 0.0)
            lesions = [
                Dict{String,Any}("LesionName" => "L1", "ObservationState" => "UNREVIEWED",
                                  "BaseAnatomy" => "Bone", "LesionType" => "Bone Meta",
                                  "Side" => "Left", "RegistrationQC" => "QUESTIONABLE"),
                Dict{String,Any}("LesionName" => "L2", "ObservationState" => "NEW",
                                  "BaseAnatomy" => "Lung", "LesionType" => "Organ Meta"),
                Dict{String,Any}("LesionName" => "L3", "ObservationState" => "UNCERTAIN",
                                  "BaseAnatomy" => "LN", "LesionType" => "LN Meta")
            ]
            issues = run_conflict_checks(report, lesions)
            codes = Set([i.code for i in issues])
            @test "UNREVIEWED_LESIONS" in codes
            @test "BONE_M_MISMATCH" in codes
            @test "NEW_LESION_PRESENT" in codes
            @test "UNCERTAIN_LESIONS" in codes
            @test "REGISTRATION_ISSUES" in codes
            # Note: ZERO_TMTV only fires for ACCEPTED/CORRECTED/NEW.
            # L2 is NEW so ZERO_TMTV should fire
            @test "ZERO_TMTV" in codes
        end

        @testset "ValidationIssue struct fields" begin
            issue = ValidationIssue(BLOCKING, "TEST_CODE", "Test message", "Test details")
            @test issue.severity == BLOCKING
            @test issue.code == "TEST_CODE"
            @test issue.message == "Test message"
            @test issue.details == "Test details"
        end

        @testset "ConflictSeverity enum" begin
            @test BLOCKING isa ConflictSeverity
            @test WARNING isa ConflictSeverity
            @test INFORMATIONAL isa ConflictSeverity
            @test length(instances(ConflictSeverity)) == 3
        end
    end

    # ============================================================================
    # 4. ResearchExport Tests
    # ============================================================================
    @testset "ResearchExport CSV and JSON" begin
        tmp_dir = mktempdir()

        @testset "CSV export basic" begin
            entries = [
                Dict{String,Any}(
                    "case_id" => "CASE_001",
                    "timepoint_index" => 0,
                    "lesion_id" => "5",
                    "lesion_name" => "L5",
                    "lesion_type" => "Bone Meta",
                    "base_anatomy" => "femur_left",
                    "side" => "Left",
                    "observation_state" => "ACCEPTED",
                    "segmentation_origin" => "AI_PRESEGMENTATION",
                    "has_expert_edits" => false,
                    "suv_max" => 12.5,
                    "suv_mean" => 6.3,
                    "volume_cc" => 2.1,
                    "diameter_mm" => 15.0,
                    "registration_qc" => "GOOD",
                    "registration_comment" => "",
                    "reviewer_timestamp" => "2026-09-21T05:00:00",
                    "comment" => "clean"
                ),
                Dict{String,Any}(
                    "case_id" => "CASE_001",
                    "timepoint_index" => 1,
                    "lesion_id" => "5",
                    "lesion_name" => "L5",
                    "lesion_type" => "Bone Meta",
                    "base_anatomy" => "femur_left",
                    "side" => "Left",
                    "observation_state" => "CORRECTED",
                    "segmentation_origin" => "EXPERT_CORRECTION",
                    "has_expert_edits" => true,
                    "suv_max" => 14.0,
                    "suv_mean" => 7.1,
                    "volume_cc" => 2.5,
                    "diameter_mm" => 16.0,
                    "registration_qc" => "GOOD",
                    "registration_comment" => "",
                    "reviewer_timestamp" => "2026-09-21T06:00:00",
                    "comment" => "edited mask"
                )
            ]

            csv_path = joinpath(tmp_dir, "test_export.csv")
            result_path = export_lesion_data_csv(entries, csv_path)
            @test result_path == csv_path
            @test isfile(csv_path)

            lines = readlines(csv_path)
            @test length(lines) == 3  # header + 2 data rows
            @test startswith(lines[1], "case_id,")
            @test contains(lines[2], "CASE_001")
            @test contains(lines[2], "ACCEPTED")
            @test contains(lines[3], "CORRECTED")
        end

        @testset "CSV escaping" begin
            entries = [
                Dict{String,Any}(
                    "case_id" => "CASE_002",
                    "comment" => "has, comma",
                    "registration_comment" => "he said \"hello\"",
                    "lesion_name" => "line\nbreak"
                )
            ]
            csv_path = joinpath(tmp_dir, "test_escape.csv")
            export_lesion_data_csv(entries, csv_path)
            content = read(csv_path, String)
            # Values with commas/quotes/newlines should be quoted
            @test contains(content, "\"has, comma\"")
            @test contains(content, "\"he said \"\"hello\"\"\"")
            @test contains(content, "\"line\nbreak\"")
        end

        @testset "CSV empty data" begin
            csv_path = joinpath(tmp_dir, "test_empty.csv")
            export_lesion_data_csv(Dict{String,Any}[], csv_path)
            lines = readlines(csv_path)
            @test length(lines) == 1  # header only
            @test startswith(lines[1], "case_id,")
        end

        @testset "JSON export basic" begin
            entries = [
                Dict{String,Any}(
                    "case_id" => "CASE_001",
                    "lesion_id" => "5",
                    "suv_max" => 12.5,
                    "observation_state" => "ACCEPTED"
                ),
                Dict{String,Any}(
                    "case_id" => "CASE_001",
                    "lesion_id" => "6",
                    "suv_max" => 8.0,
                    "observation_state" => "REJECTED"
                )
            ]

            json_path = joinpath(tmp_dir, "test_export.json")
            result_path = export_lesion_data_json(entries, json_path)
            @test result_path == json_path
            @test isfile(json_path)

            # Parse the JSON back
            content = read(json_path, String)
            parsed = JSON.parse(content)
            @test parsed isa Vector
            @test length(parsed) == 2
            @test parsed[1]["case_id"] == "CASE_001"
            @test parsed[1]["suv_max"] == 12.5
            @test parsed[2]["observation_state"] == "REJECTED"
        end

        @testset "JSON empty data" begin
            json_path = joinpath(tmp_dir, "test_empty.json")
            export_lesion_data_json(Dict{String,Any}[], json_path)
            content = read(json_path, String)
            parsed = JSON.parse(content)
            @test parsed isa Vector
            @test isempty(parsed)
        end

        @testset "JSON escaping" begin
            entries = [
                Dict{String,Any}(
                    "comment" => "he said \"hello\"",
                    "case_id" => "CASE_003"
                )
            ]
            json_path = joinpath(tmp_dir, "test_json_escape.json")
            export_lesion_data_json(entries, json_path)
            content = read(json_path, String)
            parsed = JSON.parse(content)
            @test parsed[1]["comment"] == "he said \"hello\""
        end

        rm(tmp_dir, recursive=true, force=true)
    end

    # ============================================================================
    # 5. AuditEvent Creation Tests
    # ============================================================================
    @testset "AuditEvent Construction" begin

        @testset "Default keyword constructor" begin
            ae = AuditEvent()
            @test ae.event_id isa String
            @test ae.event_id == ""
            @test ae.timestamp isa String
            @test ae.timestamp == ""
            @test ae.case_id == ""
            @test ae.lesion_track_id == 0
            @test ae.timepoint_index == 0
            @test ae.event_type == ""
            @test ae.previous_state == ""
            @test ae.new_state == ""
            @test ae.segmentation_version == ""
            @test ae.tool_used == ""
            @test ae.comment == ""
        end

        @testset "Full custom construction" begin
            ae = AuditEvent(
                event_id="EVT_001",
                timestamp="2026-09-21T05:00:00Z",
                case_id="CASE_001",
                lesion_track_id=5,
                timepoint_index=0,
                event_type="ACCEPT",
                previous_state="UNREVIEWED",
                new_state="ACCEPTED",
                segmentation_version="v001",
                tool_used="keyboard",
                comment="Looks correct"
            )
            @test ae.event_id == "EVT_001"
            @test ae.timestamp == "2026-09-21T05:00:00Z"
            @test ae.case_id == "CASE_001"
            @test ae.lesion_track_id == 5
            @test ae.timepoint_index == 0
            @test ae.event_type == "ACCEPT"
            @test ae.previous_state == "UNREVIEWED"
            @test ae.new_state == "ACCEPTED"
            @test ae.segmentation_version == "v001"
            @test ae.tool_used == "keyboard"
            @test ae.comment == "Looks correct"
        end

        @testset "Various event_type values" begin
            for evt_type in ["ACCEPT", "REJECT", "CORRECT", "UNCERTAIN", "RESOLVED",
                             "NEW_LESION", "EDIT_START", "EDIT_APPLY", "REVERT_TO_AI",
                             "PROMPT_SEGMENT", "NAVIGATE"]
                ae = AuditEvent(event_type=evt_type)
                @test ae.event_type == evt_type
            end
        end

        @testset "Positional constructor" begin
            ae = AuditEvent("E1", "2026-01-01", "C1", 1, 0, "ACCEPT", "UNREVIEWED", "ACCEPTED", "v1", "kbd", "ok")
            @test ae.event_id == "E1"
            @test ae.case_id == "C1"
            @test ae.lesion_track_id == 1
        end

        @testset "Immutability check" begin
            ae = AuditEvent(event_id="test")
            # AuditEvent is a struct (not mutable), so fields cannot be changed
            @test_throws ErrorException ae.event_id = "changed"
        end
    end

end  # top-level testset
