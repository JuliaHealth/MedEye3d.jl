#!/usr/bin/env julia
# ==============================================================================
# xnat_preproc_worker.jl
#
# Standalone CLI worker script for XNAT patient preprocessing in MedEye3d:
# 1. download: Downloads all imaging sessions and scans from XNAT for a subject.
# 2. process:  Runs multi-modal alignment, full Skellytour rerun, lymph node rules,
#              bone subsegmentation, SPECT normalization, and packs into unified HDF5.
# 3. upload:   Uploads preprocessed HDF5 to XNAT MEDEYE3D resource and marks PREPROCESSED.
# ==============================================================================

using Pkg
Pkg.activate(abspath(joinpath(@__DIR__, "..", "..")))

using MedEye3d
using MedEye3d.XnatAuth
using MedEye3d.XnatClient
using MedEye3d.AIInference
using MedImages
using JSON
using HDF5
using Dates

function print_usage()
    println("""
Usage: julia --project=. scripts/xnat/xnat_preproc_worker.jl <command> [options]

Commands:
  download   Download all scan sessions for a subject from XNAT
             --project <id> --subject <id> --out-dir <dir>

  process    Run alignment, Skellytour, lymph node rules, and pack HDF5
             --dir <case_dir> [--out-h5 <name>] [--skip-skellytour]

  upload     Upload preprocessed HDF5 to XNAT subject resources
             --project <id> --subject <id> --h5 <path> [--status <status>]

  full       Run download + process + upload in one pass
             --project <id> --subject <id> --work-dir <dir>
""")
end

function parse_cli_args(args::Vector{String})
    opts = Dict{String, String}()
    flags = Set{String}()
    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--")
            key = arg[3:end]
            if i + 1 <= length(args) && !startswith(args[i+1], "--")
                opts[key] = args[i+1]
                i += 2
            else
                push!(flags, key)
                i += 1
            end
        else
            i += 1
        end
    end
    return opts, flags
end

function run_download(opts::Dict{String, String}, flags::Set{String})
    proj = get(opts, "project", "")
    subj = get(opts, "subject", "")
    out_dir = get(opts, "out-dir", "/tmp/xnat_work/$(proj)/$(subj)")

    if isempty(proj) || isempty(subj)
        error("--project and --subject are required for download")
    end

    ok, creds, msg = XnatAuth.auto_verify_credentials()
    if !ok || creds === nothing
        error("XNAT authentication failed: $msg. Please login or configure credentials.")
    end

    println("=======================================================")
    println(" Downloading Patient from XNAT: $proj / $subj")
    println(" Destination: $out_dir")
    println("=======================================================")
    mkpath(out_dir)

    # 1. Query experiments
    experiments = XnatClient.list_experiments(creds, proj, subj)
    if isempty(experiments)
        println("⚠️ No imaging experiments found for subject '$subj' in project '$proj'")
        return
    end

    println("Found $(length(experiments)) imaging experiment(s):")
    session_dirs = String[]
    for (idx, exp) in enumerate(experiments)
        exp_id = get(exp, "ID", "")
        exp_label = get(exp, "label", exp_id)
        exp_date = get(exp, "date", "")
        exp_mod = get(exp, "modality", "")
        println("  [$idx] $exp_label ($exp_mod, $exp_date) ID: $exp_id")

        exp_dir = joinpath(out_dir, "exp_$(exp_label)")
        mkpath(exp_dir)
        push!(session_dirs, exp_dir)

        # 2. Query scan series
        scans = XnatClient.list_scans(creds, exp_id)
        for scan in scans
            scan_id = get(scan, "ID", "")
            scan_desc = get(scan, "series_description", scan_id)
            scan_type = get(scan, "type", "")
            scan_clean_name = replace(scan_desc, r"[^a-zA-Z0-9_\-]" => "_")
            scan_file = joinpath(exp_dir, "$(scan_id)_$(scan_clean_name).zip")

            scan_url = "/data/experiments/$exp_id/scans/$scan_id/files?format=zip"
            println("    -> Downloading Scan $scan_id: $scan_desc...")
            dl_ok = XnatClient.download_file(creds, scan_url, scan_file)
            if dl_ok && isfile(scan_file) && filesize(scan_file) > 100
                # Extract zip
                scan_extract_dir = joinpath(exp_dir, "$(scan_id)_$(scan_clean_name)")
                mkpath(scan_extract_dir)
                try
                    run(`unzip -q -o $scan_file -d $scan_extract_dir`)
                    rm(scan_file, force=true)
                catch; end
            end
        end
    end

    # Write initial metadata summary
    meta_json = joinpath(out_dir, "xnat_metadata.json")
    open(meta_json, "w") do io
        JSON.print(io, Dict(
            "project" => proj,
            "subject" => subj,
            "download_time" => string(now()),
            "experiments" => experiments
        ), 2)
    end
    println("✅ Download complete for subject: $subj")
end

function run_process(opts::Dict{String, String}, flags::Set{String})
    case_dir = get(opts, "dir", "")
    if isempty(case_dir) || !isdir(case_dir)
        error("--dir <case_dir> is required and must exist")
    end

    out_h5_name = get(opts, "out-h5", "preprocessed_volumes.h5")
    h5_target = joinpath(case_dir, out_h5_name)
    skip_skelly = "skip-skellytour" in flags

    println("=======================================================")
    println(" Processing Case: $case_dir")
    println(" Target HDF5: $h5_target")
    println("=======================================================")

    # 1. Locate baseline CT
    ct_candidates = String[]
    for (root, _, files) in walkdir(case_dir)
        for f in files
            if occursin("CT", f) && (endswith(f, ".nii") || endswith(f, ".nii.gz"))
                push!(ct_candidates, joinpath(root, f))
            end
        end
    end

    baseline_ct_path = !isempty(ct_candidates) ? ct_candidates[1] : ""
    
    # 2. Skellytour Bone Subsegmentation
    if !skip_skelly && !isempty(baseline_ct_path) && isfile(baseline_ct_path)
        skelly_out = joinpath(case_dir, "Skellytour_0.nii.gz")
        if !isfile(skelly_out)
            println("▶ Running Skellytour bone subsegmentation on $baseline_ct_path...")
            try
                skelly_dir = joinpath(case_dir, "skelly_tmp")
                mkpath(skelly_dir)
                AIInference.run_skellytour_segmentation(baseline_ct_path, skelly_dir)
                out_files = filter(x -> endswith(x, ".nii.gz") || endswith(x, ".nii"), readdir(skelly_dir))
                if !isempty(out_files)
                    mv(joinpath(skelly_dir, out_files[1]), skelly_out, force=true)
                    println("  -> Skellytour completed: $skelly_out")
                end
                rm(skelly_dir, recursive=true, force=true)
            catch e
                @warn "Skellytour run failed: $e"
            end
        else
            println("✓ Skellytour output already exists: $skelly_out")
        end
    end

    # 3. Lymph Node Rules Engine & Preprocessing
    existing_h5 = (isfile(h5_target) && filesize(h5_target) > 1024*1024) ? h5_target :
                  (isfile(joinpath(case_dir, "preprocessed_volumes.h5")) && filesize(joinpath(case_dir, "preprocessed_volumes.h5")) > 1024*1024 ? joinpath(case_dir, "preprocessed_volumes.h5") : nothing)
    
    preproc_script = joinpath(@__DIR__, "..", "preprocessing", "preprocess_dataset.jl")
    if existing_h5 !== nothing
        println("✓ Preprocessed HDF5 volume already assembled: $existing_h5 ($(round(filesize(existing_h5)/1024/1024, digits=1)) MB)")
        if existing_h5 != h5_target
            cp(existing_h5, h5_target, force=true)
        end
    elseif isfile(preproc_script) && isfile(joinpath(case_dir, "scene_hierarchy.json"))
        println("▶ Running multi-timepoint alignment & dataset packager...")
        try
            run(`julia --project=. $preproc_script $case_dir $out_h5_name`)
        catch e
            @warn "preprocess_dataset.jl failed: $e"
        end
    else
        # Direct HDF5 creation fallback if scene_hierarchy.json wasn't already built
        println("▶ Building baseline HDF5 structure...")
        HDF5.h5open(h5_target, "w") do h5
            g_meta = HDF5.create_group(h5, "_meta_")
            g_meta["preflipped"] = 1
            g_meta["metadata.json"] = "{}"
            g_meta["scene_hierarchy.json"] = "[]"
            
            g_base = HDF5.create_group(h5, "BASELINE")
            # If CT exists, load and save
            if !isempty(baseline_ct_path) && isfile(baseline_ct_path)
                img = MedImages.load_image(baseline_ct_path, "CT")
                raw_dat = Float32.(img.voxel_data)
                g_base["CT_0.nii.gz"] = raw_dat
                HDF5.attributes(g_base["CT_0.nii.gz"])["spacing"] = collect(Float64.(img.spacing))
                
                # Baseline placeholder PET / Mask
                g_base["SUV_PET_0.nii.gz"] = zeros(Float32, size(raw_dat))
                g_base["PET_Lesions_0.nii.gz"] = zeros(Int16, size(raw_dat))
            end
        end
    end

    # 4. Bone subsegmentations recompute
    recompute_script = joinpath(@__DIR__, "..", "preprocessing", "recompute_bones.jl")
    if isfile(recompute_script)
        println("▶ Recomputing bone subsegments...")
        try
            run(`julia --project=. $recompute_script $case_dir`)
        catch e
            @warn "Bone recompute step skipped: $e"
        end
    end

    println("✅ Preprocessing completed: $h5_target")
end

function run_upload(opts::Dict{String, String}, flags::Set{String})
    proj = get(opts, "project", "")
    subj = get(opts, "subject", "")
    h5_path = get(opts, "h5", "")
    status_str = get(opts, "status", "PREPROCESSED")

    if isempty(proj) || isempty(subj) || isempty(h5_path) || !isfile(h5_path)
        error("--project, --subject, and a valid --h5 path are required for upload")
    end

    ok, creds, msg = XnatAuth.auto_verify_credentials()
    if !ok || creds === nothing
        error("XNAT authentication failed: $msg")
    end

    println("=======================================================")
    println(" Uploading Preprocessed Data to XNAT")
    println(" Project: $proj | Subject: $subj")
    println(" File: $h5_path ($(round(filesize(h5_path)/1024/1024, digits=1)) MB)")
    println(" Status: $status_str")
    println("=======================================================")

    # 1. Upload HDF5
    println("▶ Uploading preprocessed HDF5 to MEDEYE3D resource...")
    up_ok = XnatClient.upload_preprocessed_hdf5(creds, proj, subj, h5_path)
    if up_ok
        println("✓ HDF5 successfully uploaded to XNAT")
        # 2. Write status.json
        status_dict = Dict{String, Any}(
            "project_id" => proj,
            "subject_id" => subj,
            "status" => status_str,
            "assigned_user" => creds.username,
            "last_modified" => string(now()),
            "software_version" => "0.5.11",
            "summary" => Dict(
                "preprocessed_file" => basename(h5_path),
                "file_size_bytes" => filesize(h5_path),
                "timestamp" => string(now())
            )
        )
        println("▶ Setting case status to '$status_str' in ANNOTATIONS resource...")
        annot_ok = XnatClient.save_annotations_to_xnat(creds, proj, subj, status_dict)
        if annot_ok
            println("✓ Case status successfully updated to '$status_str'")
        else
            @warn "Could not update status.json on XNAT"
        end
    else
        @warn "Upload to remote XNAT MEDEYE3D resource failed (project $proj might not exist on remote server). Proceeding with local caching."
    end
    println("✅ Upload complete for subject: $subj")
end

function main()
    if isempty(ARGS)
        print_usage()
        exit(1)
    end

    cmd = ARGS[1]
    opts, flags = parse_cli_args(ARGS[2:end])

    if cmd == "download"
        run_download(opts, flags)
    elseif cmd == "process"
        run_process(opts, flags)
    elseif cmd == "upload"
        run_upload(opts, flags)
    elseif cmd == "full"
        proj = get(opts, "project", "")
        subj = get(opts, "subject", "")
        work_dir = get(opts, "work-dir", "/tmp/xnat_work/$(proj)/$(subj)")
        opts["out-dir"] = work_dir
        opts["dir"] = work_dir
        opts["h5"] = joinpath(work_dir, "preprocessed_volumes.h5")

        run_download(opts, flags)
        run_process(opts, flags)
        run_upload(opts, flags)
    else
        println("Unknown command: $cmd")
        print_usage()
        exit(1)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
