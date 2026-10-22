using Test
using MedEye3d
using MedEye3d.XnatAuth
using MedEye3d.XnatClient
using JSON
using Dates

@testset "XNAT Integration Unit Tests" begin
    # 1. Credentials path
    p = XnatAuth.get_credentials_path()
    @test !isempty(p)
    @test endswith(p, "xnat_credentials.json")

    # 2. XnatCredentials constructor & URL cleanup
    creds = XnatCredentials(
        server_url = "https://xnat.example.org/ ",
        alias = " test_alias ",
        secret = " secret_123 ",
        username = " tester "
    )
    @test creds.server_url == "https://xnat.example.org"
    @test creds.alias == "test_alias"
    @test creds.secret == "secret_123"
    @test creds.username == "tester"

    # 3. Local token validity check
    @test XnatAuth.is_token_valid_locally(creds) == true
    
    expired_creds = XnatCredentials(
        server_url = "https://xnat.example.org",
        alias = "expired_alias",
        secret = "expired_secret",
        expires_at = time() - 100.0
    )
    @test XnatAuth.is_token_valid_locally(expired_creds) == false

    # 4. Auth headers
    headers = XnatAuth.get_auth_headers(creds)
    @test any(h -> h.first == "Authorization" && startswith(h.second, "Basic "), headers)

    # 5. Save & Load Cached Credentials in isolated temp environment
    test_creds_file = tempname() * "_xnat_creds.json"
    ENV["MEDEYE3D_XNAT_CREDS"] = test_creds_file
    try
        @test XnatAuth.load_cached_credentials() === nothing
        @test XnatAuth.save_cached_credentials(creds) == true
        @test isfile(test_creds_file)

        loaded = XnatAuth.load_cached_credentials()
        @test loaded !== nothing
        @test loaded.server_url == creds.server_url
        @test loaded.alias == creds.alias
        @test loaded.secret == creds.secret
        @test loaded.username == creds.username

        # Clear credentials
        @test XnatAuth.clear_cached_credentials() == true
        @test !isfile(test_creds_file)
        @test XnatAuth.load_cached_credentials() === nothing
    finally
        delete!(ENV, "MEDEYE3D_XNAT_CREDS")
        rm(test_creds_file, force=true)
    end

    # 6. CaseStatusInfo
    raw_status = CaseStatusInfo(:RAW)
    @test raw_status.status == :RAW
    @test isempty(raw_status.user)

    proc_status = CaseStatusInfo(:PROCESSED, "dr_mitura", "2026-10-22T05:00:00Z", 4, Dict{String, Any}("summary" => 1))
    @test proc_status.status == :PROCESSED
    @test proc_status.user == "dr_mitura"
    @test proc_status.num_lesions == 4

    # 7. Subject ID Parsing from UI options
    parse_subj = MedEye3d.LesionMetadataWindow._parse_subject_id
    @test parse_subj("🟢 pat6 (PROCESSED)") == "pat6"
    @test parse_subj("🟡 pat7 (PREPROCESSED)") == "pat7"
    @test parse_subj("🔴 pat8_followup (RAW)") == "pat8_followup"
    @test parse_subj("🔵 pat9-case (IN_PROGRESS)") == "pat9-case"
    @test parse_subj("pat6") == "pat6"
    @test parse_subj("  pat6  ") == "pat6"

    # 8. Local cache dir formatting
    cache_dir = MedEye3d.XnatPreloader.get_local_cache_dir("FDM_DPI-2024-7-KRN_Lu177_PSMA_PIPELINE", "pat6")
    @test occursin("FDM_DPI-2024-7-KRN_Lu177_PSMA_PIPELINE", cache_dir)
    @test endswith(cache_dir, "pat6")

    # 9. Preloader notification & queue tracking
    test_queue = ["pat6", "pat7", "pat8"]
    MedEye3d.XnatPreloader.notify_patient_selected(creds, "FDM_DPI-2024-7-KRN_Lu177_PSMA_PIPELINE", "pat6", test_queue)
    # Give a tiny slice for preloader task startup
    sleep(0.1)
    MedEye3d.XnatPreloader.cancel_preload!()
end

ccall(:_exit, Cvoid, (Cint,), 0)
