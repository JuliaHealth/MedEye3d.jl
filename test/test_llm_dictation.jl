using Test
using MedEye3d
using MedEye3d.LLMDictation

@testset "LLMDictation Data Structs and Prompt Builder" begin
    # Mock clinical context
    lesion1 = LesionFinding(
        1, "Lesion 1", "Femur", "Left", "Proximal Diaphysis", "Bone Metastasis",
        "RID28620", "Cortical breach visible", [128, 256, 45], 45,
        4.25, 28.4, 12.8f0, 5.56f0, 8.53f0,
        1, true, "PET TP 0", 7.10, 18.5f0,
        -2.85, -40.1, -5.7f0, -30.8,
        "RECIP-PR", false
    )

    lesion2 = LesionFinding(
        2, "Lesion 2", "Ilium", "Right", "Posterior Spine", "Bone Metastasis",
        "RID28564", "New osteolytic lesion", [340, 210, 72], 72,
        1.80, 16.0, 9.4f0, 4.08f0, 6.26f0,
        0, false, "", 0.0, 0.0f0,
        0.0, 0.0, 0.0f0, 0.0,
        "NEW LESION", true
    )

    resolved1 = ResolvedLesion("PET TP 0", 3, "Rib 6 Right", 1.20, 6.4f0)

    bg = Dict("liver" => 2.3f0, "blood" => 1.5f0, "parotid" => 1.8f0)

    data = TimePointClinicalData(
        "PAT_TEST_001", 1, "PET TP 1 (Follow-up)", "PET/CT",
        "Vorbekanntes metastasiertes Prostatakarzinom", "Known metastatic prostate cancer under therapy",
        bg, [lesion1, lesion2], [resolved1], 2,
        6.05, 7.10, -14.8,
        "PROGRESSIVE METABOLIC DISEASE (PMD / RECIP-PD) - New Lesion Emerged", true
    )

    @test data.patient_id == "PAT_TEST_001"
    @test data.total_lesions == 2
    @test length(data.resolved_lesions) == 1
    @test data.lesions[2].is_new == true

    # Test prompt generation (English)
    prompts_en = build_dictation_prompt(data; lang="EN")
    @test length(prompts_en) == 2
    @test prompts_en[1]["role"] == "system"
    @test prompts_en[2]["role"] == "user"
    user_txt_en = prompts_en[2]["content"]
    @test occursin("PAT_TEST_001", user_txt_en)
    @test occursin("Femur (Left)", user_txt_en)
    @test occursin("RECIP-PR", user_txt_en)
    @test occursin("*** NEW HYPERMETABOLIC LESION ***", user_txt_en)
    @test occursin("RESOLVED LESIONS", user_txt_en)
    @test occursin("Liver SUVmean: 2.3", user_txt_en)

    # Test prompt generation (German)
    prompts_de = build_dictation_prompt(data; lang="DE")
    @test length(prompts_de) == 2
    @test occursin("Facharzt für Radiologie", prompts_de[1]["content"])
    @test occursin("KLINISCHE ANGABEN", prompts_de[2]["content"])
end

@testset "LLMDictation API Completion Test" begin
    # Test a direct call to AcademicCloud with qwen3.5-397b-a17b
    test_msgs = [
        Dict("role" => "system", "content" => "You are a radiologist assistant. Be concise."),
        Dict("role" => "user", "content" => "Confirm receipt of clinical data for PAT_TEST_001 in one sentence.")
    ]
    resp = call_academiccloud_chat(test_msgs; model="qwen3.5-397b-a17b", max_tokens=1500)
    println("API Response: ", resp)
    @test !isempty(resp)
    @test !startswith(resp, "API Error")
    @test !startswith(resp, "Connection Error")
end

println("ALL LLM DICTATION TESTS PASSED ✓")
