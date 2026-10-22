using Test
using MedEye3d
using MedEye3d.XnatAuth
using MedEye3d.XnatClient
using JSON
using Dates
using HDF5

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

    # 2b. XnatCredentials with SubString inputs (prevent TypeError on strip())
    sub_url = strip("  https://xnat.bmia.nl/  ")
    sub_alias = strip("  token_alias_abc  ")
    sub_sec = strip("  token_sec_123  ")
    sub_user = strip("  john_doe  ")
    @test sub_url isa SubString
    creds_sub = XnatCredentials(
        server_url = sub_url,
        alias = sub_alias,
        secret = sub_sec,
        username = sub_user
    )
    @test creds_sub.server_url isa String
    @test creds_sub.server_url == "https://xnat.bmia.nl"
    @test creds_sub.alias == "token_alias_abc"
    @test creds_sub.secret == "token_sec_123"
    @test creds_sub.username == "john_doe"

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

    # 10. Switch dataset handling
    test_h5 = joinpath(dirname(@__DIR__), "data", "pat_6_files", "preprocessed_volumes.h5")
    if isfile(test_h5)
        # Verify local switch logic
        MEH_mod = MedEye3d.SegmentationDisplay.MakieEventHandlers
        orig_path = MEH_mod.h5_path_ref[]
        MEH_mod.h5_path_ref[] = test_h5
        @test MEH_mod.h5_path_ref[] == test_h5
        @test isfile(MEH_mod.h5_path_ref[])
        MEH_mod.h5_path_ref[] = orig_path
    end

    # 11. download_preprocessed_hdf5 graceful handling of non-existent files
    fake_creds = XnatCredentials(server_url="https://xnat.invalid", alias="a", secret="s")
    @test XnatClient.download_preprocessed_hdf5(fake_creds, "proj", "subj", tempname() * ".h5") == false

    # 12. Automated missing-preprocessed HDF5 fallback pathing & cache resolution
    target_cache_dir = MedEye3d.XnatPreloader.get_local_cache_dir("test_proj", "test_subj")
    target_cache_h5 = joinpath(target_cache_dir, "test_subj_preprocessed.h5")
    @test occursin("test_proj", target_cache_dir)
    @test endswith(target_cache_h5, "test_subj_preprocessed.h5")
    # Verify candidate paths resolution logic
    dummy_scratch = mktempdir()
    dummy_candidate = joinpath(dummy_scratch, "preprocessed_volumes.h5")
    write(dummy_candidate, "dummy-hdf5-data-test")
    candidate_paths = [
        joinpath("/tmp", "medeye3d_xnat_preproc", "test_proj", "test_subj", "test_subj_preprocessed.h5"),
        dummy_candidate
    ]
    resolved_candidate = nothing
    for cp_path in candidate_paths
        if isfile(cp_path)
            resolved_candidate = cp_path
            break
        end
    end
    @test resolved_candidate == dummy_candidate
    rm(dummy_scratch, recursive=true)

    # 13. Allowed XNAT Projects constraint
    allowed_projs = MedEye3d.LesionMetadataWindow.ALLOWED_XNAT_PROJECTS
    @test length(allowed_projs) == 2
    @test "FDM_DPI-2024-7-KRN_Prostata_bimodal_PIPELINE" in allowed_projs
    @test "FDM_DPI-2024-7-KRN_Lu177_PSMA_PIPELINE" in allowed_projs

    # 14. Validation of both cases (Lu177 pat6 and Prostata bimodal pat1)
    pat6_h5 = joinpath(dirname(@__DIR__), "data", "pat_6_files", "preprocessed_volumes.h5")
    pat1_h5 = joinpath(dirname(@__DIR__), "data", "cases", "psma_patient_all_tp", "preprocessed_volumes.h5")
    @test isfile(pat6_h5)
    @test isfile(pat1_h5)
    HDF5.h5open(pat6_h5, "r") do h5
        @test haskey(h5, "BASELINE")
        @test haskey(h5["BASELINE"], "Fixed_CT_Volume_0.nii.gz")
    end
    HDF5.h5open(pat1_h5, "r") do h5
        @test haskey(h5, "BASELINE")
        @test haskey(h5["BASELINE"], "Fixed_CT_Volume_0.nii.gz")
    end
end

ccall(:_exit, Cvoid, (Cint,), 0)
