"""
    XnatClient

High-performance REST API client for XNAT integration in MedEye3d.
Provides project and patient discovery, resource catalog querying, case status checking
(RAW vs PREPROCESSED vs PROCESSED), asynchronous streaming downloads, and uploads.
"""
module XnatClient

using JSON
using Downloads
using ..XnatAuth

export list_projects,
       list_subjects,
       list_experiments,
       list_project_experiments,
       list_scans,
       list_subject_resources,
       get_case_status,
       download_file,
       upload_file,
       ensure_subject_resource,
       download_preprocessed_hdf5,
       upload_preprocessed_hdf5,
       save_annotations_to_xnat,
       resolve_project_id,
       resolve_subject_label,
       CaseStatusInfo

"""
    CaseStatusInfo

Structured status metadata for a given patient.
"""
struct CaseStatusInfo
    status::Symbol  # :RAW, :PREPROCESSED, :IN_PROGRESS, :PROCESSED
    user::String
    last_modified::String
    num_lesions::Int
    raw_dict::Dict{String, Any}
end

CaseStatusInfo(status::Symbol) = CaseStatusInfo(status, "", "", 0, Dict{String, Any}())

# ── Helper for XNAT JSON REST requests ────────────────────────────────────────

function _xnat_get_json(creds::XnatCredentials, endpoint::AbstractString; timeout::Float64=15.0)::Union{Dict{String, Any}, Nothing}
    url = "$(creds.server_url)$(endpoint)"
    headers = XnatAuth.get_auth_headers(creds)
    try
        buf = IOBuffer()
        resp = Downloads.request(url; method="GET", headers=headers, output=buf, timeout=timeout)
        if resp.status == 200
            body = String(take!(buf))
            return JSON.parse(body)
        else
            @debug "XNAT GET $url returned HTTP $(resp.status)"
            return nothing
        end
    catch e
        @warn "XNAT GET error on $url: $e"
        return nothing
    end
end

"""
    resolve_project_id(project_id::AbstractString) -> String

Maps pipeline display names to their actual XNAT project IDs (e.g. `FDM_DPI-2024-7-KRN_Lu177_PSMA_PIPELINE` -> `Lu177_PSMA_PIPELINE`).
"""
function resolve_project_id(project_id::AbstractString)::String
    p = strip(project_id)
    if p == "FDM_DPI-2024-7-KRN_Lu177_PSMA_PIPELINE" || p == "Lu177_PSMA" || p == "FDM_DPI-2024-7-KRN_Lu177_PSMA"
        return "Lu177_PSMA_PIPELINE"
    elseif p == "FDM_DPI-2024-7-KRN_Prostata_bimodal_PIPELINE" || p == "Prostata_bimodal_PIPELINE"
        return "Prostata_bimodal_PIPELINE"
    elseif p == "Prostata_bimodal" || p == "FDM_DPI-2024-7-KRN_Prostata_bimodal"
        return "Prostata_bimodal"
    end
    return String(p)
end

"""
    resolve_subject_label(creds::XnatCredentials, project_id::AbstractString, subject_id::AbstractString) -> String

Finds the matching subject label on XNAT. If `subject_id` (e.g. `pat6` or `pat1`) is a shorthand,
resolves it to the full XNAT subject label (e.g. `FDM_DPI-2024-7-KRN_Lu177_PSMA__PETPSMA_0__Pat6`).
"""
function resolve_subject_label(creds::XnatCredentials, project_id::AbstractString, subject_id::AbstractString)::String
    s = strip(subject_id)
    p = resolve_project_id(project_id)
    subjects = list_subjects(creds, p)
    # 1. Exact match on ID or label
    for subj in subjects
        lbl = get(subj, "label", "")
        id = get(subj, "ID", "")
        if lbl == s || id == s
            return !isempty(lbl) ? lbl : id
        end
    end
    # 2. Case-insensitive suffix or pattern match (e.g. "pat6" or "Pat6")
    m_num = match(r"[pP]at_?(\d+)", s)
    target_num = m_num !== nothing ? m_num.captures[1] : ""
    if !isempty(target_num)
        for subj in subjects
            lbl = get(subj, "label", "")
            id = get(subj, "ID", "")
            cand = !isempty(lbl) ? lbl : id
            if occursin(Regex("[pP]at0*$(target_num)(?:[\\s_]|\$)", "i"), cand)
                return cand
            end
        end
    end
    # Fallback to input string
    return String(s)
end

"""
    list_projects(creds::XnatCredentials) -> Vector{Dict{String, Any}}

Queries `GET /data/projects?format=json` and returns all projects accessible to the user.
"""
function list_projects(creds::XnatCredentials)::Vector{Dict{String, Any}}
    data = _xnat_get_json(creds, "/data/projects?format=json")
    data === nothing && return Dict{String, Any}[]
    try
        res = data["ResultSet"]["Result"]
        return Vector{Dict{String, Any}}(res)
    catch e
        @warn "Failed to parse projects list: $e"
        return Dict{String, Any}[]
    end
end

"""
    list_subjects(creds::XnatCredentials, project_id) -> Vector{Dict{String, Any}}

Queries `GET /data/projects/{project_id}/subjects?format=json`.
"""
function list_subjects(creds::XnatCredentials, project_id::AbstractString)::Vector{Dict{String, Any}}
    clean_proj = resolve_project_id(project_id)
    isempty(clean_proj) && return Dict{String, Any}[]
    data = _xnat_get_json(creds, "/data/projects/$(clean_proj)/subjects?format=json")
    data === nothing && return Dict{String, Any}[]
    try
        res = data["ResultSet"]["Result"]
        # Check if subject labels are empty (e.g. Prostata_bimodal where label is "")
        has_labels = any(s -> !isempty(strip(get(s, "label", ""))), res)
        if !has_labels
            exp_res = list_project_experiments(creds, clean_proj)
            if !isempty(exp_res)
                patient_map = Dict{String, Dict{String, Any}}()
                for e in exp_res
                    elbl = get(e, "label", "")
                    m = match(r"[pP]at\d+", elbl)
                    if m !== nothing
                        pat_id = m.match
                        if !haskey(patient_map, pat_id)
                            patient_map[pat_id] = Dict{String, Any}(
                                "ID" => pat_id,
                                "label" => pat_id,
                                "project" => clean_proj,
                                "experiment_id" => get(e, "ID", ""),
                                "experiment_label" => elbl
                            )
                        end
                    end
                end
                if !isempty(patient_map)
                    sorted_keys = sort(collect(keys(patient_map)), by=k -> begin
                        m = match(r"\d+", k)
                        m !== nothing ? parse(Int, m.match) : 0
                    end)
                    return [patient_map[k] for k in sorted_keys]
                end
            end
        end
        return Vector{Dict{String, Any}}(res)
    catch e
        @warn "Failed to parse subjects list for $project_id: $e"
        return Dict{String, Any}[]
    end
end

"""
    list_project_experiments(creds::XnatCredentials, project_id) -> Vector{Dict{String, Any}}

Queries all imaging sessions (experiments) under a project.
"""
function list_project_experiments(creds::XnatCredentials, project_id::AbstractString)::Vector{Dict{String, Any}}
    clean_proj = resolve_project_id(project_id)
    isempty(clean_proj) && return Dict{String, Any}[]
    data = _xnat_get_json(creds, "/data/projects/$(clean_proj)/experiments?format=json")
    data === nothing && return Dict{String, Any}[]
    try
        res = data["ResultSet"]["Result"]
        return Vector{Dict{String, Any}}(res)
    catch e
        @warn "Failed to parse experiments list for $project_id: $e"
        return Dict{String, Any}[]
    end
end

"""
    list_experiments(creds::XnatCredentials, project_id, subject_id) -> Vector{Dict{String, Any}}

Queries imaging sessions (experiments) for a subject.
"""
function list_experiments(creds::XnatCredentials, project_id::AbstractString, subject_id::AbstractString)::Vector{Dict{String, Any}}
    p = resolve_project_id(project_id)
    s = resolve_subject_label(creds, p, subject_id)
    endpoint = "/data/projects/$p/subjects/$s/experiments?format=json"
    data = _xnat_get_json(creds, endpoint)
    if data !== nothing && haskey(data, "ResultSet") && haskey(data["ResultSet"], "Result")
        res = data["ResultSet"]["Result"]
        if !isempty(res)
            return Vector{Dict{String, Any}}(res)
        end
    end
    # Fallback: query project experiments and filter by patient pattern
    all_exps = list_project_experiments(creds, p)
    m_num = match(r"[pP]at_?(\d+)", s)
    target_pat = m_num !== nothing ? "Pat$(m_num.captures[1])" : s
    filtered = filter(e -> begin
        lbl = get(e, "label", "")
        occursin(Regex("\\b$(target_pat)\\b", "i"), lbl) || occursin(s, lbl)
    end, all_exps)
    return filtered
end

"""
    list_scans(creds::XnatCredentials, experiment_id) -> Vector{Dict{String, Any}}

Queries all scan series for an imaging session.
"""
function list_scans(creds::XnatCredentials, experiment_id::AbstractString)::Vector{Dict{String, Any}}
    endpoint = "/data/experiments/$(strip(experiment_id))/scans?format=json"
    data = _xnat_get_json(creds, endpoint)
    data === nothing && return Dict{String, Any}[]
    try
        res = data["ResultSet"]["Result"]
        return Vector{Dict{String, Any}}(res)
    catch; return Dict{String, Any}[]; end
end

"""
    list_subject_resources(creds::XnatCredentials, project_id, subject_id) -> Vector{String}

Returns a list of resource folder names attached to the subject (e.g. `["MEDEYE3D", "ANNOTATIONS"]`).
"""
function list_subject_resources(creds::XnatCredentials, project_id::AbstractString, subject_id::AbstractString)::Vector{String}
    p = resolve_project_id(project_id)
    s = resolve_subject_label(creds, p, subject_id)
    endpoint = "/data/projects/$p/subjects/$s/resources?format=json"
    data = _xnat_get_json(creds, endpoint)
    data === nothing && return String[]
    try
        results = data["ResultSet"]["Result"]
        names = String[]
        for item in results
            lbl = get(item, "label", get(item, "Name", ""))
            !isempty(lbl) && push!(names, lbl)
        end
        return names
    catch; return String[]; end
end

"""
    get_case_status(creds::XnatCredentials, project_id, subject_id) -> CaseStatusInfo

Determines whether a patient's case is:
- `:PROCESSED`: Validated by expert (`resources/ANNOTATIONS/files/status.json` has `status == "PROCESSED"`)
- `:PREPROCESSED`: `{Subject}_preprocessed.h5` exists in `resources/MEDEYE3D/files/`
- `:RAW`: Only raw DICOM scans exist on XNAT
"""
function get_case_status(creds::XnatCredentials, project_id::AbstractString, subject_id::AbstractString)::CaseStatusInfo
    p = resolve_project_id(project_id)
    s = resolve_subject_label(creds, p, subject_id)
    if isempty(p) || isempty(s)
        return CaseStatusInfo(:RAW)
    end

    # 1. Check if status.json exists in ANNOTATIONS resource
    status_url = "/data/projects/$p/subjects/$s/resources/ANNOTATIONS/files/status.json"
    status_json = _xnat_get_json(creds, status_url; timeout=5.0)
    if status_json !== nothing
        st_str = uppercase(get(status_json, "status", ""))
        user = get(status_json, "assigned_user", get(status_json, "user", ""))
        ts = get(status_json, "last_modified", "")
        summary = get(status_json, "summary", Dict{String, Any}())
        n_les = get(summary, "num_lesions", 0)
        
        st_sym = if st_str == "PROCESSED" || st_str == "VALIDATED" || st_str == "COMPLETED"
            :PROCESSED
        elseif st_str == "IN_PROGRESS"
            :IN_PROGRESS
        elseif st_str == "PREPROCESSED"
            :PREPROCESSED
        else
            :RAW
        end
        return CaseStatusInfo(st_sym, user, ts, n_les, status_json)
    end

    # 2. Check if preprocessed HDF5 file exists in MEDEYE3D resource
    medeye_files = _xnat_get_json(creds, "/data/projects/$p/subjects/$s/resources/MEDEYE3D/files?format=json"; timeout=5.0)
    if medeye_files !== nothing
        try
            results = medeye_files["ResultSet"]["Result"]
            for r in results
                fname = get(r, "Name", "")
                if endswith(fname, ".h5") || endswith(fname, ".hdf5")
                    return CaseStatusInfo(:PREPROCESSED)
                end
            end
        catch; end
    end

    # 3. Fallback: Check experiment-level resources (e.g. for Prostata_bimodal)
    exps = list_experiments(creds, p, s)
    for exp in exps
        exp_id = get(exp, "ID", "")
        isempty(exp_id) && continue
        
        # Check experiment ANNOTATIONS
        exp_status = _xnat_get_json(creds, "/data/experiments/$exp_id/resources/ANNOTATIONS/files/status.json"; timeout=5.0)
        if exp_status !== nothing
            st_str = uppercase(get(exp_status, "status", ""))
            user = get(exp_status, "assigned_user", get(exp_status, "user", ""))
            ts = get(exp_status, "last_modified", "")
            summary = get(exp_status, "summary", Dict{String, Any}())
            n_les = get(summary, "num_lesions", 0)
            st_sym = st_str in ("PROCESSED", "VALIDATED", "COMPLETED") ? :PROCESSED : (st_str == "IN_PROGRESS" ? :IN_PROGRESS : :PREPROCESSED)
            return CaseStatusInfo(st_sym, user, ts, n_les, exp_status)
        end
        
        # Check experiment MEDEYE3D
        exp_medeye = _xnat_get_json(creds, "/data/experiments/$exp_id/resources/MEDEYE3D/files?format=json"; timeout=5.0)
        if exp_medeye !== nothing
            try
                for r in exp_medeye["ResultSet"]["Result"]
                    fname = get(r, "Name", "")
                    if endswith(fname, ".h5") || endswith(fname, ".hdf5")
                        return CaseStatusInfo(:PREPROCESSED)
                    end
                end
            catch; end
        end
    end

    return CaseStatusInfo(:RAW)
end

const _xnat_downloader = Ref{Union{Downloads.Downloader, Nothing}}(nothing)
function get_xnat_downloader()::Downloads.Downloader
    if _xnat_downloader[] === nothing
        d = Downloads.Downloader()
        d.easy_hook = (easy, info) -> begin
            Downloads.Curl.setopt(easy, Downloads.Curl.CURLOPT_LOW_SPEED_TIME, 0)
            Downloads.Curl.setopt(easy, Downloads.Curl.CURLOPT_LOW_SPEED_LIMIT, 0)
        end
        _xnat_downloader[] = d
    end
    return _xnat_downloader[]
end

"""
    download_file(creds::XnatCredentials, remote_endpoint_or_url, local_dest; progress_cb=nothing) -> Bool

Streams a file from XNAT to local disk, optionally invoking `progress_cb(total_bytes, dl_bytes)` during download.
"""
function download_file(
    creds::XnatCredentials,
    remote_url_or_endpoint::AbstractString,
    local_dest::AbstractString;
    progress_cb::Union{Function, Nothing}=nothing
)::Bool
    full_url = startswith(remote_url_or_endpoint, "http") ? String(remote_url_or_endpoint) : "$(creds.server_url)$(remote_url_or_endpoint)"
    dest_path = String(local_dest)
    tmp_path = "$(dest_path).tmp.$(getpid())_$(round(Int, time()*1000))"
    headers = XnatAuth.get_auth_headers(creds)
    downloader = get_xnat_downloader()
    download_ok = false
    try
        mkpath(dirname(dest_path))
        if progress_cb !== nothing
            Downloads.download(full_url, tmp_path; headers=headers, progress=progress_cb, timeout=1800.0, downloader=downloader)
        else
            Downloads.download(full_url, tmp_path; headers=headers, timeout=1800.0, downloader=downloader)
        end
        download_ok = isfile(tmp_path) && filesize(tmp_path) > 0
    catch e
        @warn "Downloads.download failed from $full_url: $e; falling back to curl..."
    end

    if !download_ok
        try
            auth_str = "$(creds.alias):$(creds.secret)"
            cmd = `curl -fSL -u $auth_str --connect-timeout 60 --speed-time 300 --speed-limit 100 -o $tmp_path $full_url`
            run(cmd)
            download_ok = isfile(tmp_path) && filesize(tmp_path) > 0
        catch e2
            @warn "Curl download also failed: $e2"
        end
    end

    if download_ok
        mv(tmp_path, dest_path; force=true)
        return true
    else
        isfile(tmp_path) && rm(tmp_path; force=true)
        return false
    end
end

"""
    upload_file(creds::XnatCredentials, remote_endpoint_or_url, local_src; content_type="application/octet-stream") -> Bool

Uploads a local file to XNAT via HTTP PUT.
"""
function upload_file(
    creds::XnatCredentials,
    remote_url_or_endpoint::AbstractString,
    local_src::AbstractString;
    content_type::AbstractString="application/octet-stream"
)::Bool
    src_path = String(local_src)
    !isfile(src_path) && return false
    full_url = startswith(remote_url_or_endpoint, "http") ? String(remote_url_or_endpoint) : "$(creds.server_url)$(remote_url_or_endpoint)"
    # Ensure inbody=true for XNAT REST API payload upload
    if !occursin("inbody=true", full_url)
        sep = occursin('?', full_url) ? "&" : "?"
        full_url = full_url * sep * "inbody=true"
    end
    headers = [
        XnatAuth.get_auth_headers(creds)...,
        "Content-Type" => String(content_type)
    ]
    try
        open(src_path, "r") do in_io
            buf = IOBuffer()
            resp = Downloads.request(full_url; method="PUT", headers=headers, input=in_io, output=buf, timeout=1800.0)
            return resp.status in (200, 201)
        end
    catch e
        @warn "Upload failed to $full_url: $e"
        return false
    end
end

"""
    ensure_subject_resource(creds::XnatCredentials, project_id, subject_id, resource_name) -> Bool

Creates the specified resource folder under the subject if it does not yet exist.
"""
function ensure_subject_resource(creds::XnatCredentials, project_id::AbstractString, subject_id::AbstractString, resource_name::AbstractString)::Bool
    p = resolve_project_id(project_id)
    s = resolve_subject_label(creds, p, subject_id)
    r = strip(resource_name)
    existing = list_subject_resources(creds, p, s)
    if r in existing
        return true
    end

    res_url = "$(creds.server_url)/data/projects/$p/subjects/$s/resources/$r"
    headers = XnatAuth.get_auth_headers(creds)
    try
        buf = IOBuffer()
        resp = Downloads.request(res_url; method="PUT", headers=headers, output=buf, timeout=10.0)
        return resp.status in (200, 201)
    catch e
        @warn "Error ensuring resource $r for $p/$s: $e"
        return false
    end
end

"""
    download_preprocessed_hdf5(creds::XnatCredentials, project_id, subject_id, local_dest; force=false, progress_cb=nothing) -> Bool

Downloads preprocessed HDF5 from XNAT subject or experiment resource `MEDEYE3D`.
Skips download if file already exists locally and `force == false`.
"""
function download_preprocessed_hdf5(
    creds::XnatCredentials,
    project_id::AbstractString,
    subject_id::AbstractString,
    local_dest::AbstractString;
    force::Bool=false,
    progress_cb::Union{Function, Nothing}=nothing
)::Bool
    dest_path = String(local_dest)
    p = resolve_project_id(project_id)
    s = resolve_subject_label(creds, p, subject_id)

    # 1. Search for preprocessed hdf5 in subject MEDEYE3D resource
    medeye_files = _xnat_get_json(creds, "/data/projects/$p/subjects/$s/resources/MEDEYE3D/files?format=json")
    if medeye_files !== nothing
        try
            results = medeye_files["ResultSet"]["Result"]
            for r in results
                fn = get(r, "Name", "")
                if endswith(fn, ".h5") || endswith(fn, ".hdf5")
                    remote_sz = tryparse(Int64, string(get(r, "Size", "0")))
                    if !force && isfile(dest_path) && filesize(dest_path) > 1024
                        if remote_sz === nothing || remote_sz <= 0 || filesize(dest_path) == remote_sz
                            return true
                        end
                        println("[XNAT] Cached HDF5 size ($(filesize(dest_path))) != remote XNAT size ($remote_sz); re-downloading fresh copy...")
                    end
                    uri = get(r, "URI", "/data/projects/$p/subjects/$s/resources/MEDEYE3D/files/$fn")
                    if download_file(creds, uri, dest_path; progress_cb=progress_cb)
                        return true
                    end
                end
            end
        catch; end
    end

    # 2. Fallback: Search in experiment MEDEYE3D resources (e.g. for Prostata_bimodal)
    exps = list_experiments(creds, p, s)
    for exp in exps
        exp_id = get(exp, "ID", "")
        isempty(exp_id) && continue
        exp_files = _xnat_get_json(creds, "/data/experiments/$exp_id/resources/MEDEYE3D/files?format=json")
        if exp_files !== nothing && haskey(exp_files, "ResultSet") && haskey(exp_files["ResultSet"], "Result")
            for r in exp_files["ResultSet"]["Result"]
                fn = get(r, "Name", "")
                if endswith(fn, ".h5") || endswith(fn, ".hdf5")
                    uri = get(r, "URI", "/data/experiments/$exp_id/resources/MEDEYE3D/files/$fn")
                    println("[XNAT] Downloading preprocessed volume from experiment $exp_id: $fn")
                    if download_file(creds, uri, dest_path; progress_cb=progress_cb)
                        return true
                    end
                end
            end
        end
    end

    return false
end

"""
    upload_preprocessed_hdf5(creds::XnatCredentials, project_id, subject_id, local_h5) -> Bool

Uploads preprocessed HDF5 to `/data/projects/{proj}/subjects/{subj}/resources/MEDEYE3D/files/{subj}_preprocessed.h5?overwrite=true`,
or falls back to matching imaging session experiment resource if subject-level resource creation is disallowed.
"""
function upload_preprocessed_hdf5(
    creds::XnatCredentials,
    project_id::AbstractString,
    subject_id::AbstractString,
    local_h5::AbstractString
)::Bool
    h5_path = String(local_h5)
    !isfile(h5_path) && return false
    p = resolve_project_id(project_id)
    s = resolve_subject_label(creds, p, subject_id)
    
    # 1. Try subject-level resource
    if ensure_subject_resource(creds, p, s, "MEDEYE3D")
        remote_url = "/data/projects/$p/subjects/$s/resources/MEDEYE3D/files/$(s)_preprocessed.h5?overwrite=true&inbody=true"
        if upload_file(creds, remote_url, h5_path)
            return true
        end
    end

    # 2. Fallback: upload to matching PET/CT experiment
    exps = list_experiments(creds, p, s)
    pet_exps = filter(e -> occursin("PET", uppercase(get(e, "label", ""))) || occursin("PET", uppercase(get(e, "xsiType", ""))), exps)
    target_exp = !isempty(pet_exps) ? first(pet_exps) : (!isempty(exps) ? first(exps) : nothing)
    if target_exp !== nothing
        exp_id = get(target_exp, "ID", "")
        # Ensure resource on experiment
        res_url = "$(creds.server_url)/data/experiments/$exp_id/resources/MEDEYE3D"
        headers = XnatAuth.get_auth_headers(creds)
        try
            Downloads.request(res_url; method="PUT", headers=headers, timeout=10.0)
        catch; end
        
        fname = basename(local_h5)
        remote_url = "/data/experiments/$exp_id/resources/MEDEYE3D/files/$(fname)?overwrite=true&inbody=true"
        return upload_file(creds, remote_url, h5_path)
    end

    @warn "Cannot access or create MEDEYE3D resource on remote XNAT for $p/$s. Remote upload skipped."
    return false
end

"""
    save_annotations_to_xnat(creds::XnatCredentials, project_id, subject_id, status_dict; local_expert_h5=nothing) -> Bool

Uploads `status.json` and optionally the expert modifications HDF5 to `resources/ANNOTATIONS`.
"""
function save_annotations_to_xnat(
    creds::XnatCredentials,
    project_id::AbstractString,
    subject_id::AbstractString,
    status_dict::Dict{String, Any};
    local_expert_h5::Union{AbstractString, Nothing}=nothing
)::Bool
    p = resolve_project_id(project_id)
    s = resolve_subject_label(creds, p, subject_id)
    if !ensure_subject_resource(creds, p, s, "ANNOTATIONS")
        @warn "Cannot access or create ANNOTATIONS resource on remote XNAT for $p/$s. Remote upload skipped."
        return false
    end

    # 1. Upload status.json
    tmp_json = tempname() * ".json"
    open(tmp_json, "w") do io
        JSON.print(io, status_dict, 2)
    end
    status_ok = upload_file(creds, "/data/projects/$p/subjects/$s/resources/ANNOTATIONS/files/status.json?overwrite=true&inbody=true", tmp_json; content_type="application/json")
    try rm(tmp_json, force=true) catch end

    # 2. Upload expert HDF5 if provided
    if local_expert_h5 !== nothing && isfile(local_expert_h5)
        upload_file(creds, "/data/projects/$p/subjects/$s/resources/ANNOTATIONS/files/$(s)_annotations.h5?overwrite=true&inbody=true", String(local_expert_h5))
    end

    return status_ok
end

end # module XnatClient
