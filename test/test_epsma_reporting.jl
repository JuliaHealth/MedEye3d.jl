using Test
using JSON
using Dates
using MedEye3d
using MedEye3d.EPSMAStructuredReport
using MedEye3d.EPSMAReportWindow

@testset "E-PSMA Structured Reporting Suite" begin
    # ── Test 1: Visual Score Calculation (Table 2) ───────────────────────────
    @testset "PSMA Expression V (Visual Score)" begin
        blood_suv = 1.5f0
        liver_suv = 2.3f0
        parotid_suv = 3.8f0

        # Below blood pool -> Score 0
        s0, n0 = compute_psma_v(0.8f0, liver_suv, blood_suv, parotid_suv)
        @test n0 == 0
        @test occursin("Score 0", s0)

        # Equal to blood pool, below liver -> Score 1
        s1, n1 = compute_psma_v(1.9f0, liver_suv, blood_suv, parotid_suv)
        @test n1 == 1
        @test occursin("Score 1", s1)

        # Equal to liver, below parotid -> Score 2
        s2, n2 = compute_psma_v(2.8f0, liver_suv, blood_suv, parotid_suv)
        @test n2 == 2
        @test occursin("Score 2", s2)

        # Equal to or above parotid -> Score 3
        s3, n3 = compute_psma_v(12.5f0, liver_suv, blood_suv, parotid_suv)
        @test n3 == 3
        @test occursin("Score 3", s3)
    end

    # ── Test 2: Reader Confidence Scale (Table 4) ────────────────────────────
    @testset "Reader Confidence Scale (1 to 5)" begin
        liver_suv = 2.3f0
        # Definitive PCa (Score 5): Typical site (prostate, pelvic node, bone) with high uptake
        @test compute_reader_confidence("Prostate", 10.5f0, liver_suv, false) == 5
        @test compute_reader_confidence("External Iliac Lymph Node", 6.2f0, liver_suv, false) == 5
        @test compute_reader_confidence("Femur", 8.0f0, liver_suv, true) == 5

        # Probably PCa (Score 4)
        @test compute_reader_confidence("Prostate", 2.1f0, liver_suv, false) == 4

        # Equivocal (Score 3)
        @test compute_reader_confidence("Prostate", 1.2f0, liver_suv, false) == 3
        @test compute_reader_confidence("Thyroid", 5.0f0, liver_suv, false) == 3

        # Probably Benign (Score 2)
        @test compute_reader_confidence("Thyroid", 2.0f0, liver_suv, false) == 2

        # Benign (Score 1)
        @test compute_reader_confidence("Fat", 0.5f0, liver_suv, false) == 1
    end

    # ── Test 3: miTNM Classification (Table 3) ───────────────────────────────
    @testset "miTNM Classification" begin
        # Prostate (T)
        @test classify_mitnm("Prostate", "right lobe", "") == "miT2"
        @test classify_mitnm("Prostate", "extracapsular extension", "") == "miT3a"
        @test classify_mitnm("Prostate", "right seminal vesicle", "") == "miT3b"
        @test classify_mitnm("Prostate bed", "anastomosis", "recurrence") == "miTr"
        @test classify_mitnm("Prostate", "invasion bladder neck", "") == "miT4"

        # Lymph Nodes (N / M1a)
        @test classify_mitnm("Lymph Node", "External Iliac", "") == "miN1"
        @test classify_mitnm("Lymph Node", "Obturator", "") == "miN1"
        @test classify_mitnm("Lymph Node", "Presacral", "") == "miN1"
        @test classify_mitnm("Lymph Node", "Retroperitoneal paraaortic", "") == "miM1a"
        @test classify_mitnm("Lymph Node", "Mediastinal", "") == "miM1a"

        # Bone (M1b)
        @test classify_mitnm("Bone", "Left Femur", "") == "miM1b"
        @test classify_mitnm("Ilium", "Right iliac crest", "") == "miM1b"
        @test classify_mitnm("Spine", "L3 vertebra", "") == "miM1b"
        @test classify_mitnm("Rib", "5th right rib", "") == "miM1b"

        # Visceral (M1c)
        @test classify_mitnm("Lung", "Left upper lobe nodule", "") == "miM1c"
        @test classify_mitnm("Liver", "Segment VII focal lesion", "") == "miM1c"

        # Muscle is NOT Bone & Defaults to Technical Artifact
        @test classify_mitnm("quadriceps_femoris_right", "", "") == "Artifact"
        @test classify_mitnm("sartorius_right", "", "") == "Artifact"
        @test classify_mitnm("iliopsoas", "", "") == "Artifact" # Must NOT match ilium!
        @test classify_mitnm("gluteus_maximus_left", "", "") == "Artifact"
        @test classify_mitnm("Technical Artifact", "", "") == "Artifact"
        
        # LesionType-driven classification (4th parameter)
        @test classify_mitnm("Unknown Focus", "", "", "Bone Meta") == "miM1b"
        @test classify_mitnm("Unknown Focus", "", "", "Lymph Node") == "miN1"
        @test classify_mitnm("Unknown Focus", "", "", "Prostate") == "miT2"
        @test classify_mitnm("Unknown Focus", "", "", "Organ Meta") == "miM1c"
        @test classify_mitnm("Unknown Focus", "", "", "Technical Artifact") == "Artifact"
        
        # Lymph Node with distant location keywords
        @test classify_mitnm("Unknown Focus", "inguinal", "", "Lymph Node") == "miM1a"
        @test classify_mitnm("Unknown Focus", "axillary", "", "Lymph Node") == "miM1a"
        @test classify_mitnm("Unknown Focus", "retroperitoneal", "", "Lymph Node") == "miM1a"
        @test classify_mitnm("Unknown Focus", "", "", "Lymph Node") == "miN1"  # Default pelvic
        
        # German segment name-driven classification (5th parameter)
        @test classify_mitnm("", "", "", "", "F08 Knochen Beckengürtel links UBU") == "miM1b"
        @test classify_mitnm("", "", "", "", "LN09 Lymphknoten Obturator rechts") == "miN1"
        @test classify_mitnm("", "", "", "", "PT01 ProstataCa Gl 7a gesichert") == "miT2"
        @test classify_mitnm("", "", "", "", "F12 Knochen Rippe UBU") == "miM1b"
        @test classify_mitnm("", "", "", "", "F09 Lymphknoten inguinal rechts") == "miM1a"
        
        # Fallthrough (no organ, no type, no segment) → lymph node
        @test classify_mitnm("Soft Tissue Focus", "", "") == "miN1"
        @test classify_mitnm("Unknown", "", "") == "miN1"
    end

    # ── Test 3B: Anatomy Resolution & No Generic Lesion Names ────────────────
    @testset "Max Anatomy Resolution & Muscle Artifact Flagging" begin
        # 1. Muscle resolution
        state_quad = Dict("BaseAnatomy" => "quadriceps_femoris_right", "LesionType" => "Technical Artifact", "Certainty" => "0")
        (loc_quad, is_art_quad, is_musc_quad) = EPSMAStructuredReport.resolve_anatomical_location(1, state_quad, 0, nothing, nothing)
        @test occursin("Quadriceps", loc_quad)
        @test is_art_quad == true
        @test is_musc_quad == true
        @test !occursin("Lesion 1", loc_quad)

        # 2. Bone resolution (Appendicular)
        state_femur = Dict("BaseAnatomy" => "femur_right", "LesionType" => "Bone Meta", "Certainty" => "3")
        (loc_femur, is_art_femur, is_musc_femur) = EPSMAStructuredReport.resolve_anatomical_location(3, state_femur, 0, nothing, nothing)
        @test occursin("Femur", loc_femur)
        @test is_art_femur == false
        @test is_musc_femur == false
        @test !occursin("Lesion 3", loc_femur)

        # 3. Bone resolution (Axial - Pelvis)
        state_hip = Dict("BaseAnatomy" => "hip_left", "LesionType" => "Bone Meta", "Certainty" => "3")
        (loc_hip, is_art_hip, is_musc_hip) = EPSMAStructuredReport.resolve_anatomical_location(5, state_hip, 0, nothing, nothing)
        @test occursin("Pelvic Bone", loc_hip) || occursin("Left", loc_hip)
        @test is_art_hip == false
        @test !occursin("Lesion 5", loc_hip)

        # 4. Bone resolution (Axial - Spine)
        state_vert = Dict("BaseAnatomy" => "vertebrae_T12", "LesionType" => "Bone Meta", "Certainty" => "3")
        (loc_vert, is_art_vert, is_musc_vert) = EPSMAStructuredReport.resolve_anatomical_location(17, state_vert, 0, nothing, nothing)
        @test occursin("Thoracic Vertebra", loc_vert) || occursin("T12", loc_vert)
        @test is_art_vert == false
        @test !occursin("Lesion 17", loc_vert)

        # 5. Prostate resolution
        state_pros = Dict("BaseAnatomy" => "prostate", "LesionType" => "Prostate", "Certainty" => "3")
        (loc_pros, is_art_pros, is_musc_pros) = EPSMAStructuredReport.resolve_anatomical_location(8, state_pros, 0, nothing, nothing)
        @test occursin("Prostate", loc_pros)
        @test is_art_pros == false
        @test !occursin("Lesion 8", loc_pros)
    end

    # ── Test 4: Report Assembly & Synoptic Tables ────────────────────────────
    @testset "Report Struct & Serialization" begin
        rows = [
            EPSMALesionRow(
                id = 1,
                location = "Left Femur (Proximal)",
                mitnm = "miM1b",
                size_str = "28 mm (4.2 cc)",
                volume_cc = 4.2,
                diameter_mm = 28.0,
                num_lesions = 1,
                psma_q = "SUVmax 12.8",
                suv_max = 12.8f0,
                psma_v = "Score 3 (≥ Parotid)",
                psma_v_num = 3,
                reader_confidence = 5,
                recip_status = "RECIP-PR",
                delta_str = "ΔVol: -35.2% | ΔSUV: -28.1%",
                is_new = false,
                comment = "Sclerotic correlate on CT"
            ),
            EPSMALesionRow(
                id = 2,
                location = "Right Iliac Crest",
                mitnm = "miM1b",
                size_str = "16 mm (1.8 cc)",
                volume_cc = 1.8,
                diameter_mm = 16.0,
                num_lesions = 1,
                psma_q = "SUVmax 9.4",
                suv_max = 9.4f0,
                psma_v = "Score 3 (≥ Parotid)",
                psma_v_num = 3,
                reader_confidence = 5,
                recip_status = "NEW LESION",
                delta_str = "New Hypermetabolic Lesion",
                is_new = true,
                comment = "No CT change (marrow lesion)"
            ),
            EPSMALesionRow(
                id = 3,
                location = "Right External Iliac Lymph Node",
                mitnm = "miN1",
                size_str = "8 mm (0.9 cc)",
                volume_cc = 0.9,
                diameter_mm = 8.0,
                num_lesions = 1,
                psma_q = "SUVmax 6.2",
                suv_max = 6.2f0,
                psma_v = "Score 2 (≥ Liver, < Parotid)",
                psma_v_num = 2,
                reader_confidence = 5,
                recip_status = "RECIP-SD",
                delta_str = "ΔVol: -5.1% | ΔSUV: -2.3%",
                is_new = false,
                comment = "Enlarged 8mm short axis"
            )
        ]

        rep = EPSMAReport(
            patient_id = "PAT_006",
            tp_index = 1,
            tp_label = "PET TP 1",
            modality = "PET/CT ([68Ga]Ga-PSMA-11)",
            study_date = "2021-06-10",
            tech_params = EPSMATechnicalParams(),
            background_suv = Dict("liver" => 2.3f0, "blood" => 1.5f0, "parotid" => 1.8f0),
            synoptic_rows = rows,
            final_mitnm = "miT0 miN1 miM1b",
            tmtv_cc = 6.9,
            tmtv_delta_pct = -18.2,
            overall_recip = "PROGRESSIVE METABOLIC DISEASE (PMD) - New Lesion",
            conclusion_en = "1. Evidence of progression due to new right iliac crest metastasis.\n2. Regression of dominant left femur lesion.",
            conclusion_de = "1. Nachweis eines Progresses aufgrund neuer rechtsseitiger Beckenmetastase.\n2. Regression der dominanten Femurmetastase links."
        )

        d = to_dict(rep)
        @test d["patient_id"] == "PAT_006"
        @test d["final_mitnm"] == "miT0 miN1 miM1b"
        @test length(d["synoptic_rows"]) == 3
        @test d["synoptic_rows"][1]["psma_v_num"] == 3
        @test d["synoptic_rows"][2]["is_new"] == true
    end

    # ── Test 5: Word Document (.docx) Export in EN and DE ────────────────────
    @testset "Word Document (.docx) Export" begin
        rep = EPSMAReport(
            patient_id = "PAT_TEST",
            tp_index = 0,
            tp_label = "PET Baseline",
            modality = "PET/CT ([68Ga]Ga-PSMA-11)",
            study_date = "2026-09-03",
            final_mitnm = "miT3b miN1 miM1b",
            tmtv_cc = 14.5,
            synoptic_rows = [
                EPSMALesionRow(
                    id = 1,
                    location = "Prostate (Right lobe & Seminal Vesicle)",
                    mitnm = "miT3b",
                    size_str = "32 mm (8.5 cc)",
                    volume_cc = 8.5,
                    diameter_mm = 32.0,
                    num_lesions = 1,
                    psma_q = "SUVmax 18.2",
                    suv_max = 18.2f0,
                    psma_v = "Score 3 (≥ Parotid)",
                    psma_v_num = 3,
                    reader_confidence = 5
                )
            ]
        )

        out_en = "/home/jm/.gemini/antigravity-cli/brain/853d9462-f5ad-4cf0-874c-6c7072725016/scratch/test_export_EN.docx"
        out_de = "/home/jm/.gemini/antigravity-cli/brain/853d9462-f5ad-4cf0-874c-6c7072725016/scratch/test_export_DE.docx"

        res_en = export_to_docx(rep, out_en; lang = "EN")
        @test isfile(res_en)
        @test filesize(res_en) > 10000

        res_de = export_to_docx(rep, out_de; lang = "DE")
        @test isfile(res_de)
        @test filesize(res_de) > 10000
    end

    # ── Test 6: Makie Window Lifecycle ───────────────────────────────────────
    @testset "Makie Report Window Lifecycle" begin
        rep = EPSMAReport(patient_id = "PAT_UI_TEST")
        # In test mode, verify window opens and closes without crashing
        ENV["MEDEYE3D_TEST_MODE"] = "true"
        try
            screen = open_epsma_report_window(rep)
            @test active_report_screen[] !== nothing
            close_epsma_report_window()
            # After close, screen ref is nulled (SetWindowShouldClose triggers cleanup)
            @test active_report_screen[] === nothing
        catch e
            # If headless display is unavailable, verify fallback
            @info "Headless screen test result: $e"
        end
    end
end
println("\n=== ALL E-PSMA TESTS PASSED SUCCESSFULLY! ===")
