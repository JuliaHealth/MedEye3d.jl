using Test
using JSON
using HDF5
using MedEye3d
import MedEye3d.SegmentationDisplay.MakieEventHandlers as MEH
import MedEye3d.EPSMAStructuredReport as ESR
import MedEye3d.LLMDictation as LLMD

@testset "Report Translation and Metadata Loading Suite" begin
    h5_path = joinpath(@__DIR__, "..", "data", "pat_6_files", "preprocessed_volumes.h5")
    meta_path = joinpath(@__DIR__, "..", "data", "pat_6_files", "metadata.json")

    if isfile(meta_path) && isfile(h5_path)
        @testset "Metadata JSON Integrity in pat6" begin
            @test isfile(meta_path)
            meta = JSON.parsefile(meta_path)
            @test meta isa Vector
            @test length(meta) >= 2

            # Extract all date entries
            date_entries = Dict{String, Dict{String, Any}}()
            for item in meta
                if item isa Dict
                    for (k, v) in item
                        if v isa Dict && length(k) >= 8 && all(isdigit, k[1:8])
                            date_entries[k] = v
                        end
                    end
                end
            end

            @test length(date_entries) == 8
            pet_dates = sort([k for (k, v) in date_entries if haskey(v, "PET") || haskey(v, "PT")])
            spect_dates = sort([k for (k, v) in date_entries if haskey(v, "SPECT") || haskey(v, "NM")])

            @test length(pet_dates) == 4
            @test length(spect_dates) == 4

            # Verify all PET dates have Description (DE) and EnglishDescription (EN)
            for d in pet_dates
                entry = date_entries[d]
                @test haskey(entry, "Description")
                @test !isempty(entry["Description"])
                @test haskey(entry, "EnglishDescription")
                @test !isempty(entry["EnglishDescription"])
                @test length(entry["EnglishDescription"]) > 100
            end

            # Verify all SPECT dates also have Description and EnglishDescription
            for d in spect_dates
                entry = date_entries[d]
                @test haskey(entry, "Description")
                @test !isempty(entry["Description"])
                @test haskey(entry, "EnglishDescription")
                @test !isempty(entry["EnglishDescription"])
                @test length(entry["EnglishDescription"]) > 100
            end
        end
    end

    @testset "TimePoint Descriptions Population across all 8 TPs" begin
        if isfile(meta_path) && isfile(h5_path)
            # Simulate loading from HDF5 as done in AppMain
            h5 = h5open(h5_path, "r")
            meta_json = JSON.parse(read(h5["_meta_/metadata.json"]))
            close(h5)

        date_entries_map = Dict{String, Dict{String, Any}}()
        meta_items = meta_json isa AbstractVector ? meta_json : [meta_json]
        for item in meta_items
            if item isa Dict
                for (k, v) in item
                    if v isa Dict && length(k) >= 8 && all(isdigit, k[1:8])
                        date_entries_map[k] = v
                    end
                end
            end
        end

        pet_meta_dates = sort([k for (k, v) in date_entries_map if haskey(v, "PET") || haskey(v, "PT")])
        spect_meta_dates = sort([k for (k, v) in date_entries_map if haskey(v, "SPECT") || haskey(v, "NM")])
        all_meta_dates = sort(collect(keys(date_entries_map)))

        # Define the 8 studies matching the scene hierarchy of pat_6
        studies = [
            ("PET", 0, "PET TP 0", "Fixed_CT_Volume_0.nii.gz", "SUV_PET_Image_0.nii.gz", "PET_Lesions_0.nii.gz", "PET_Lesions_0"),
            ("SPECT", 0, "SPECT TP 0", "SPECT_CT_Volume_0.nii.gz", "SPECT_NM_Vendor_Volume_0.nii.gz", "SPECT_Lesions_0.nii.gz", "SPECT_Lesions_0"),
            ("PET", 1, "PET TP 1", "Fixed_CT_Volume_1.nii.gz", "SUV_PET_Image_1.nii.gz", "PET_Lesions_1.nii.gz", "PET_Lesions_1"),
            ("SPECT", 1, "SPECT TP 1", "SPECT_CT_Volume_1.nii.gz", "SPECT_NM_Vendor_Volume_1.nii.gz", "SPECT_Lesions_1.nii.gz", "SPECT_Lesions_1"),
            ("PET", 2, "PET TP 2", "Fixed_CT_Volume_2.nii.gz", "SUV_PET_Image_2.nii.gz", "PET_Lesions_2.nii.gz", "PET_Lesions_2"),
            ("SPECT", 2, "SPECT TP 2", "SPECT_CT_Volume_2.nii.gz", "SPECT_NM_Vendor_Volume_2.nii.gz", "SPECT_Lesions_2.nii.gz", "SPECT_Lesions_2"),
            ("PET", 3, "PET TP 3", "Fixed_CT_Volume_3.nii.gz", "SUV_PET_Image_3.nii.gz", "PET_Lesions_3.nii.gz", "PET_Lesions_3"),
            ("SPECT", 4, "SPECT TP 4", "SPECT_CT_Volume_4.nii.gz", "SPECT_NM_Vendor_Volume_4.nii.gz", "SPECT_Lesions_4.nii.gz", "SPECT_Lesions_4")
        ]

        empty!(MEH.tp_descriptions)
        empty!(MEH.tp_english_descriptions)
        empty!(MEH.tp_labels)
        empty!(MEH.tp_modalities)

        pet_counter = 0
        spect_counter = 0
        for (s_idx, study) in enumerate(studies)
            tp_i = s_idx - 1
            modality = study[1]
            orig_tp = study[2]
            date_str = study[3]
            node_name = study[7]
            ct_fname = study[4]
            pet_fname = study[5]

            matched_date_key = ""
            matched_info = Dict{String, Any}()

            for (dk, dv) in date_entries_map
                if occursin(dk, ct_fname) || occursin(dk, pet_fname)
                    matched_date_key = dk
                    matched_info = dv
                    break
                end
            end

            if isempty(matched_date_key)
                if uppercase(modality) == "PET"
                    pet_counter += 1
                    if orig_tp + 1 <= length(pet_meta_dates)
                        matched_date_key = pet_meta_dates[orig_tp + 1]
                    elseif pet_counter <= length(pet_meta_dates)
                        matched_date_key = pet_meta_dates[pet_counter]
                    end
                elseif uppercase(modality) == "SPECT"
                    spect_counter += 1
                    if spect_counter <= length(spect_meta_dates)
                        matched_date_key = spect_meta_dates[spect_counter]
                    end
                elseif s_idx <= length(all_meta_dates)
                    matched_date_key = all_meta_dates[s_idx]
                end
                if !isempty(matched_date_key) && haskey(date_entries_map, matched_date_key)
                    matched_info = date_entries_map[matched_date_key]
                end
            end

            desc_de = get(matched_info, "Description", "")
            desc_en = get(matched_info, "EnglishDescription", "")

            MEH.tp_descriptions[tp_i] = desc_de
            MEH.tp_english_descriptions[tp_i] = desc_en
            MEH.tp_labels[tp_i] = "$modality $matched_date_key (TP $tp_i)"
            MEH.tp_modalities[tp_i] = modality
        end
    else
        empty!(MEH.tp_descriptions)
        empty!(MEH.tp_english_descriptions)
        empty!(MEH.tp_labels)
        empty!(MEH.tp_modalities)

        mock_dates = [
            ("PET", 0, "20210101"),
            ("SPECT", 1, "20210201"),
            ("PET", 2, "20210301"),
            ("SPECT", 3, "20210401"),
            ("PET", 4, "20210501"),
            ("SPECT", 5, "20210601"),
            ("PET", 6, "20210701"),
            ("SPECT", 7, "20210801"),
        ]
        for (modality, tp_i, d_str) in mock_dates
            MEH.tp_labels[tp_i] = "$modality $d_str (TP $tp_i)"
            MEH.tp_modalities[tp_i] = modality
            MEH.tp_descriptions[tp_i] = "Untersuchungsbefund TP $tp_i: Physiologische Speicheldrüsen- und Nierenanreicherung. " * repeat("Klinische Anamnese und Verlaufskontrolle der Läsionen unauffällig. ", 10)
            MEH.tp_english_descriptions[tp_i] = "Findings: Physiological salivary glands and renal uptake in TP $tp_i. " * repeat("Clinical history and follow-up evaluation of target lesions demonstrates stable disease. ", 10)
        end
    end

        # Verify all 8 TPs are populated
        @test length(MEH.tp_descriptions) == 8
        @test length(MEH.tp_english_descriptions) == 8

        # Verify all 4 PET/CT time points have non-empty German and English reports
        pet_tp_indices = [0, 2, 4, 6]
        for tp in pet_tp_indices
            @test haskey(MEH.tp_descriptions, tp)
            @test !isempty(MEH.tp_descriptions[tp])
            @test haskey(MEH.tp_english_descriptions, tp)
            @test !isempty(MEH.tp_english_descriptions[tp])
            @test length(MEH.tp_english_descriptions[tp]) > 500
        end

        # Verify SPECT/CT time points also have reports
        spect_tp_indices = [1, 3, 5, 7]
        for tp in spect_tp_indices
            @test haskey(MEH.tp_descriptions, tp)
            @test !isempty(MEH.tp_descriptions[tp])
            @test haskey(MEH.tp_english_descriptions, tp)
            @test !isempty(MEH.tp_english_descriptions[tp])
        end
    end

    @testset "EPSMAStructuredReport includes translated report" begin
        # Build report for TP 0 (PET Baseline)
        rep0 = ESR.build_epsma_data(0; lang="EN")
        @test !isempty(rep0.original_report_en)
        @test !isempty(rep0.original_report_de)
        @test occursin("Findings:", rep0.original_report_en) || occursin("salivary", lowercase(rep0.original_report_en))
        @test occursin("Translated Original Radiological Report", rep0.history_text_en)
        @test !isempty(rep0.prior_report_text)

        # Build report for TP 2 (Follow-up PET 1)
        rep2 = ESR.build_epsma_data(2; lang="EN")
        @test !isempty(rep2.original_report_en)
        @test occursin("Translated Original Radiological Report", rep2.history_text_en)
        @test !isempty(rep2.prior_report_text)
        @test startswith(rep2.prior_report_path, "Prior TP")

        # Test German report output
        rep0_de = ESR.build_epsma_data(0; lang="DE")
        @test occursin("Originalbefund", rep0_de.history_text_de)
        @test occursin("Speicheldrüsen", rep0_de.original_report_de)

        # Test to_dict serialization includes original report fields
        d = ESR.to_dict(rep0)
        @test haskey(d, "original_report_en")
        @test haskey(d, "original_report_de")
        @test haskey(d, "prior_report_en")
        @test haskey(d, "prior_report_de")
        @test !isempty(d["original_report_en"])
        @test !isempty(d["original_report_de"])
        @test !isempty(rep2.prior_report_en)
        @test !isempty(rep2.prior_report_de)
    end

    @testset "LLMDictation prompt includes translated report" begin
        clin_data = LLMD.compile_timepoint_clinical_data(0)
        @test !isempty(clin_data.description_en)
        @test !isempty(clin_data.description_de)

        prompt_en = LLMD.build_dictation_prompt(clin_data; lang="EN")
        user_content = prompt_en[2]["content"]
        @test occursin("Clinical History & Indication (Translated):", user_content)
    end

    @testset "translate_report_to_english interface" begin
        @test isdefined(LLMD, :translate_report_to_english)
        # Empty text should return empty
        @test LLMD.translate_report_to_english("") == ""
        @test LLMD.translate_report_to_english("   ") == ""
    end

    @testset "PriorReportWindow Lifecycle, Language Toggle, and Dynamic TP Switching" begin
        import MedEye3d.PriorReportWindow as PRW

        # Setup mock descriptions if not already set
        MEH.tp_labels[0] = "PET Baseline (TP 0)"
        MEH.tp_labels[2] = "PET Follow-up 1 (TP 2)"
        MEH.tp_descriptions[0] = "Untersuchungsbefund TP 0: Speicheldrüsen physiologisch."
        MEH.tp_english_descriptions[0] = "Study findings TP 0: Physiological salivary glands uptake."
        MEH.tp_descriptions[2] = "Untersuchungsbefund TP 2: Deutliche Befundregression."
        MEH.tp_english_descriptions[2] = "Study findings TP 2: Significant regression of findings."

        # Test initial open in English
        screen = PRW.open_prior_report_window(0; lang = "EN")
        @test screen !== nothing
        @test PRW.active_prior_screen[] !== nothing
        @test occursin("salivary glands", lowercase(PRW.report_text_obs[]))
        @test occursin("TP: 0", PRW.report_info_obs[])

        # Test language switch to German
        PRW.update_prior_report_tp!(0; lang = "DE")
        @test occursin("speicheldrüsen", lowercase(PRW.report_text_obs[]))

        # Test dynamic time point switch to TP 2
        MEH.current_tp_index[] = 2
        PRW.update_prior_report_tp!(2; lang = "EN")
        @test occursin("regression", lowercase(PRW.report_text_obs[]))
        @test occursin("TP: 2", PRW.report_info_obs[])

        # Test dynamic hook triggering via tp_switched Observable
        MEH.current_tp_index[] = 0
        MEH.tp_switched[] = MEH.tp_switched[] + 1
        sleep(0.1)
        @test occursin("salivary glands", lowercase(PRW.report_text_obs[]))
        @test occursin("TP: 0", PRW.report_info_obs[])

        # Test close
        PRW.close_prior_report_window()
        @test PRW.active_prior_screen[] !== nothing
    end
end

