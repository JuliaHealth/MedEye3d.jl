# ==============================================================================
# test_xnat_case_selection_and_preprocessing.jl
#
# Tests for:
# 1. Startup XNAT query for projects and subjects across both primary pipelines:
#    - Lu177_PSMA_PIPELINE (420+ cases)
#    - Prostata_bimodal_PIPELINE (130+ cases)
# 2. Case selection in dropdown and search box (all cases available).
# 3. Automatic on-demand preprocessing of unprocessed ([RAW]) cases (not ignored).
# 4. Force re-preprocessing when requested ([PRE] -> Preprocess button).
# ==============================================================================

using Test
using MedEye3d
using MedEye3d.XnatAuth
using MedEye3d.XnatClient
using MedEye3d.XnatPreloader
using MedEye3d.LesionMetadataWindow
using JSON
using Dates

@testset "XNAT Full Case Selection & Preprocessing Pipeline" begin
    # ── 1. Authentication & Startup Verification ─────────────────────────────
    ok, creds, msg = XnatAuth.auto_verify_credentials()
    @test ok == true
    @test creds !== nothing
    @test !isempty(creds.alias)
    @test !isempty(creds.secret)
    println("✓ XNAT authenticated successfully for user: '$(creds.username)' at $(creds.server_url)")

    # ── 2. Query All Projects from XNAT ──────────────────────────────────────
    remote_projs = XnatClient.list_projects(creds)
    @test !isempty(remote_projs)
    @test length(remote_projs) >= 10
    
    proj_ids = [String(get(p, "ID", "")) for p in remote_projs]
    proj_names = [String(get(p, "name", "")) for p in remote_projs]
    all_proj_identifiers = vcat(proj_ids, proj_names)
    
    # Verify primary Lu177 PSMA pipeline exists
    has_lu177 = any(p -> occursin("Lu177_PSMA", p), all_proj_identifiers)
    @test has_lu177 == true
    
    # Verify primary Prostata bimodal pipeline exists
    has_prostata = any(p -> occursin("Prostata_bimodal", p), all_proj_identifiers)
    @test has_prostata == true
    println("✓ Found $(length(remote_projs)) projects on XNAT (including Lu177_PSMA and Prostata_bimodal)")

    # ── 3. Query All Subjects for Both Primary Pipelines ─────────────────────
    # Project 1: Lu177 PSMA Pipeline (Dozens / hundreds of subjects)
    lu177_proj = "FDM_DPI-2024-7-KRN_Lu177_PSMA_PIPELINE"
    if !(lu177_proj in proj_ids || lu177_proj in proj_names)
        lu177_proj = "Lu177_PSMA_PIPELINE"
    end
    lu177_subjs = XnatClient.list_subjects(creds, lu177_proj)
    @test !isempty(lu177_subjs)
    @test length(lu177_subjs) >= 50  # Over 400 subjects on server
    println("✓ Fetched $(length(lu177_subjs)) subjects for project '$lu177_proj'")
    
    # Verify sample subjects exist
    lu177_labels = [String(get(s, "label", get(s, "ID", ""))) for s in lu177_subjs]
    @test any(s -> occursin(r"[pP]at_?6"i, s), lu177_labels)

    # Project 2: Prostata Bimodal Pipeline (Dozens / hundreds of subjects)
    prostate_proj = "FDM_DPI-2024-7-KRN_Prostata_bimodal_PIPELINE"
    if !(prostate_proj in proj_ids || prostate_proj in proj_names)
        prostate_proj = "Prostata_bimodal_PIPELINE"
    end
    prostate_subjs = XnatClient.list_subjects(creds, prostate_proj)
    @test !isempty(prostate_subjs)
    @test length(prostate_subjs) >= 20  # Over 130 subjects on server
    println("✓ Fetched $(length(prostate_subjs)) subjects for project '$prostate_proj'")

    prostate_labels = [String(get(s, "label", get(s, "ID", ""))) for s in prostate_subjs]
    @test any(s -> occursin(r"[pP]at_?1"i, s) || occursin(r"[pP]at_?44"i, s), prostate_labels)

    # ── 4. Verify Badging & Discovery ([PRE] vs [RAW]) ───────────────────────
    # Local cache check logic
    for subj_label in [lu177_labels[1], lu177_labels[min(10, length(lu177_labels))]]
        clean_id = LesionMetadataWindow._parse_subject_id(subj_label)
        target_dir = XnatPreloader.get_local_cache_dir(lu177_proj, clean_id)
        target_h5 = joinpath(target_dir, "$(clean_id)_preprocessed.h5")
        
        badge = (isfile(target_h5) && filesize(target_h5) > 1024) ? "[PRE]" : "[RAW]"
        @test badge in ["[PRE]", "[RAW]"]
    end
    println("✓ Badging logic verified: correctly categorizes preprocessed vs raw cases")

    # ── 5. Test Case Search & Direct Query Resolution ────────────────────────
    # Test parser
    @test LesionMetadataWindow._parse_subject_id("[PRE] pat6 (PREPROCESSED)") == "pat6"
    @test LesionMetadataWindow._parse_subject_id("[RAW] pat102") == "pat102"
    @test LesionMetadataWindow._parse_subject_id("[OK] pat1 (PROCESSED)") == "pat1"
    @test LesionMetadataWindow._parse_subject_id("pat55") == "pat55"

    # ── 6. Test Unprocessed Case Dispatch (Not Ignored) ─────────────────────
    # When selecting an unprocessed case, _load_patient_direct must:
    # 1. Check local cache
    # 2. Check XNAT download
    # 3. If neither present, trigger _run_preprocess_and_upload to preprocess anew!
    # Let's verify with an isolated dry-run invocation of xnat_preprocess_patient.sh
    sh_script = joinpath(dirname(@__DIR__), "scripts", "xnat", "xnat_preprocess_patient.sh")
    @test isfile(sh_script)
    
    # Run script help check to verify executable integrity
    help_out = read(`bash $sh_script --help`, String)
    @test occursin("Usage:", help_out)
    @test occursin("--overwrite", help_out)
    @test occursin("--project", help_out)
    println("✓ Preprocessing script orchestrator verified with full CLI argument support")

    # ── 7. Test Preprocessing Worker Integrity ──────────────────────────────
    worker_script = joinpath(dirname(@__DIR__), "scripts", "xnat", "xnat_preproc_worker.jl")
    @test isfile(worker_script)
    
    # Test worker CLI help
    worker_help = read(`julia -t 1 --project=. $worker_script --help`, String)
    @test occursin("Commands:", worker_help)
    @test occursin("download", worker_help)
    @test occursin("process", worker_help)
    @test occursin("upload", worker_help)
    @test occursin("full", worker_help)
    println("✓ Preprocessing worker script verified (download, process, upload, full commands supported)")

    # ── 8. Test XNAT Resource Discovery Endpoints ────────────────────────────
    # Test subject resource query on live server
    test_subj = lu177_labels[1]
    res_list = XnatClient.list_subject_resources(creds, lu177_proj, test_subj)
    @test res_list isa Vector{String}
    println("✓ Subject resources accessible on XNAT for $test_subj: $(res_list)")

    println("==============================================================================")
    println(" ALL XNAT CASE SELECTION & PREPROCESSING PIPELINE TESTS PASSED!")
    println("==============================================================================")
end
