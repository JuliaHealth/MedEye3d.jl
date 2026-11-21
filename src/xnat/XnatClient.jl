"""
    XnatClient

High-performance REST API client for XNAT integration in MedEye3d.
Provides project and patient discovery, resource catalog querying, case status checking
(RAW vs PREPROCESSED vs PROCESSED), asynchronous streaming downloads, and uploads.
"""
module XnatClient

using JSON
using Downloads
using Dates
using ..XnatAuth

export list_projects,
       list_subjects,
       list_experiments,
       list_project_experiments,
       list_scans,
       list_subject_resources,
       get_case_status,
       fetch_project_case_statuses,
       extract_status_users,
       get_remote_preprocessed_info,
       is_remote_newer_than_local,
       download_file,
       upload_file,
       ensure_subject_resource,
       download_preprocessed_hdf5,
       upload_preprocessed_hdf5,
       download_lesion_edits_hdf5,
       upload_lesion_edits_hdf5,
       save_annotations_to_xnat,
       resolve_project_id,
       resolve_subject_label,
       prefetch_all_project_hdf5,
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

function _push_unique_users!(out::Vector{String}, raw_val)
    if raw_val isa AbstractVector
        for item in raw_val
            _push_unique_users!(out, item)
        end
    elseif raw_val !== nothing
        for tok in split(string(raw_val), r"[,;]+")
            u = String(strip(tok))
            isempty(u) && continue
            lc = lowercase(u)
            (lc in ("none", "unknown", "null", "nothing", "unmodified", "na", "n/a")) && continue
            any(x -> lowercase(x) == lc, out) || push!(out, u)
        end
    end
    return out
end

"""
    extract_status_users(raw_dict::AbstractDict) -> Vector{String}

Extracts all unique non-empty usernames who modified or uploaded a case from
XNAT `status.json` (or resource file metadata dictionaries), supporting both
single-user (`modified_by`, `assigned_user`, `user`) and multi-user (`modified_users`,
`users`, `contributors`) representations.
"""
function extract_status_users(raw_dict::AbstractDict)::Vector{String}
    out = String[]
    for k in ("modified_users", "users", "contributors", "uploaded_by", "modified_by", "assigned_user", "user", "created_by", "insert_user")
        haskey(raw_dict, k) && _push_unique_users!(out, raw_dict[k])
    end
    sum_d = get(raw_dict, "summary", nothing)
    if sum_d isa AbstractDict
        for k in ("modified_users", "modified_by", "users")
            haskey(sum_d, k) && _push_unique_users!(out, sum_d[k])
        end
    end
    return out
end

function _extract_xnat_file_users(files_json)::Vector{String}
    out = String[]
    (files_json isa AbstractDict && haskey(files_json, "ResultSet") && haskey(files_json["ResultSet"], "Result")) || return out
    for r in files_json["ResultSet"]["Result"]
        r isa AbstractDict || continue
        for k in ("created_by", "insert_user", "modified_by", "uploaded_by", "user")
            haskey(r, k) && _push_unique_users!(out, r[k])
        end
    end
    return out
end

"""
    get_case_status(creds::XnatCredentials, project_id, subject_id; include_file_users=true, resolve_subject=true) -> CaseStatusInfo

Determines whether a patient's case is:
- `:PROCESSED`: Validated / modified (`resources/ANNOTATIONS/files/status.json` has `status == "PROCESSED"` or `manually_modified == true`)
- `:PREPROCESSED`: `{Subject}_preprocessed.h5` exists in `resources/MEDEYE3D/files/`
- `:RAW`: Only raw DICOM scans exist on XNAT

Collects all modifying/uploading users into `raw_dict["modified_users"]` and joins them in `info.user`.
"""
function get_case_status(
    creds::XnatCredentials,
    project_id::AbstractString,
    subject_id::AbstractString;
    include_file_users::Bool=true,
    resolve_subject::Bool=true
)::CaseStatusInfo
    p = resolve_project_id(project_id)
    s = resolve_subject ? resolve_subject_label(creds, p, subject_id) : String(strip(subject_id))
    if isempty(p) || isempty(s)
        return CaseStatusInfo(:RAW)
    end

    function _parse_status_dict(st_d::AbstractDict, file_users::Vector{String})::CaseStatusInfo
        st_str = uppercase(string(get(st_d, "status", "")))
        ts = string(get(st_d, "last_modified", ""))
        summary = get(st_d, "summary", Dict{String, Any}())
        n_les = summary isa AbstractDict ? Int(get(summary, "num_lesions", 0)) : 0
        is_mod = Bool(get(st_d, "manually_modified", st_str in ("PROCESSED", "VALIDATED", "COMPLETED")))
        users = extract_status_users(st_d)
        for u in file_users
            any(x -> lowercase(x) == lowercase(u), users) || push!(users, u)
        end
        raw_copy = Dict{String, Any}(string(k) => v for (k, v) in st_d)
        raw_copy["modified_users"] = users
        st_sym = if is_mod || st_str in ("PROCESSED", "VALIDATED", "COMPLETED")
            :PROCESSED
        elseif st_str == "IN_PROGRESS"
            :IN_PROGRESS
        elseif st_str == "PREPROCESSED"
            :PREPROCESSED
        else
            :RAW
        end
        return CaseStatusInfo(st_sym, join(users, ", "), ts, n_les, raw_copy)
    end

    # 1. Check if status.json exists in ANNOTATIONS resource
    status_url = "/data/projects/$p/subjects/$s/resources/ANNOTATIONS/files/status.json"
    status_json = _xnat_get_json(creds, status_url; timeout=5.0)
    if status_json !== nothing
        f_users = String[]
        if include_file_users
            af = _xnat_get_json(creds, "/data/projects/$p/subjects/$s/resources/ANNOTATIONS/files?format=json"; timeout=4.0)
            mf = _xnat_get_json(creds, "/data/projects/$p/subjects/$s/resources/MEDEYE3D/files?format=json"; timeout=4.0)
            append!(f_users, _extract_xnat_file_users(af))
            append!(f_users, _extract_xnat_file_users(mf))
        end
        return _parse_status_dict(status_json, f_users)
    end

    # 2. Check if preprocessed HDF5 file exists in MEDEYE3D resource
    medeye_files = _xnat_get_json(creds, "/data/projects/$p/subjects/$s/resources/MEDEYE3D/files?format=json"; timeout=5.0)
    if medeye_files !== nothing
        try
            results = medeye_files["ResultSet"]["Result"]
            for r in results
                fname = get(r, "Name", "")
                if endswith(fname, ".h5") || endswith(fname, ".hdf5")
                    f_users = _extract_xnat_file_users(medeye_files)
                    raw_d = Dict{String, Any}("modified_users" => f_users)
                    return CaseStatusInfo(:PREPROCESSED, join(f_users, ", "), "", 0, raw_d)
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
            f_users = String[]
            if include_file_users
                af = _xnat_get_json(creds, "/data/experiments/$exp_id/resources/ANNOTATIONS/files?format=json"; timeout=4.0)
                mf = _xnat_get_json(creds, "/data/experiments/$exp_id/resources/MEDEYE3D/files?format=json"; timeout=4.0)
                append!(f_users, _extract_xnat_file_users(af))
                append!(f_users, _extract_xnat_file_users(mf))
            end
            return _parse_status_dict(exp_status, f_users)
        end
        
        # Check experiment MEDEYE3D
        exp_medeye = _xnat_get_json(creds, "/data/experiments/$exp_id/resources/MEDEYE3D/files?format=json"; timeout=5.0)
        if exp_medeye !== nothing
            try
                for r in exp_medeye["ResultSet"]["Result"]
                    fname = get(r, "Name", "")
                    if endswith(fname, ".h5") || endswith(fname, ".hdf5")
                        f_users = _extract_xnat_file_users(exp_medeye)
                        raw_d = Dict{String, Any}("modified_users" => f_users)
                        return CaseStatusInfo(:PREPROCESSED, join(f_users, ", "), "", 0, raw_d)
                    end
                end
            catch; end
        end
    end

    return CaseStatusInfo(:RAW)
end

"""
    fetch_project_case_statuses(creds, project_id, subjects; include_file_users=false) -> Dict{String, CaseStatusInfo}

Concurrently fetches `CaseStatusInfo` (including multi-user modification history) across
subjects in `project_id` so the XNAT Manager can badge and filter by user / combinations.
"""
function fetch_project_case_statuses(
    creds::XnatCredentials,
    project_id::AbstractString,
    subjects::Vector{String};
    include_file_users::Bool=false
)::Dict{String, CaseStatusInfo}
    out = Dict{String, CaseStatusInfo}()
    isempty(subjects) && return out
    p = resolve_project_id(project_id)
    pairs = asyncmap(subjects; ntasks=8) do s
        info = try
            get_case_status(creds, p, s; include_file_users=include_file_users, resolve_subject=false)
        catch
            CaseStatusInfo(:RAW)
        end
        s => info
    end
    for (s, info) in pairs
        out[s] = info
    end
    return out
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

@inline function _is_lesion_edits_h5_filename(fn::AbstractString)::Bool
    low = lowercase(String(fn))
    return (endswith(low, ".h5") || endswith(low, ".hdf5")) && occursin("lesion_edits", low)
end

@inline function _is_preprocessed_h5_filename(fn::AbstractString)::Bool
    low = lowercase(String(fn))
    return (endswith(low, ".h5") || endswith(low, ".hdf5")) && !occursin("lesion_edits", low)
end

const _HTTP_MONTHS = Dict{String, Int}(
    "jan"=>1, "feb"=>2, "mar"=>3, "apr"=>4, "may"=>5, "jun"=>6,
    "jul"=>7, "aug"=>8, "sep"=>9, "oct"=>10, "nov"=>11, "dec"=>12
)

function _parse_remote_timestamp_utc(s::AbstractString)::Union{DateTime, Nothing}
    st = strip(String(s))
    isempty(st) && return nothing
    # 1. ISO-8601 (e.g. "2026-11-18T10:15:30Z" or "2026-11-18T10:15:30.123")
    m_iso = match(r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})", st)
    if m_iso !== nothing
        try
            return DateTime(
                parse(Int, m_iso.captures[1]), parse(Int, m_iso.captures[2]), parse(Int, m_iso.captures[3]),
                parse(Int, m_iso.captures[4]), parse(Int, m_iso.captures[5]), parse(Int, m_iso.captures[6])
            )
        catch; end
    end
    # 2. HTTP RFC 7231 Last-Modified (e.g. "Wed, 18 Nov 2026 10:15:30 GMT")
    m_http = match(r"(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})\s+(\d{2}):(\d{2}):(\d{2})", st)
    if m_http !== nothing
        try
            mon = get(_HTTP_MONTHS, lowercase(m_http.captures[2]), 0)
            mon > 0 && return DateTime(
                parse(Int, m_http.captures[3]), mon, parse(Int, m_http.captures[1]),
                parse(Int, m_http.captures[4]), parse(Int, m_http.captures[5]), parse(Int, m_http.captures[6])
            )
        catch; end
    end
    return nothing
end

@inline function _xnat_meta_path(local_h5::AbstractString)::String
    return string(local_h5) * ".xnat_meta.json"
end

function _read_local_xnat_baseline(local_h5::AbstractString)
    mp = _xnat_meta_path(local_h5)
    if isfile(mp)
        try
            meta = JSON.parsefile(mp)
            d = lowercase(strip(string(get(meta, "digest", ""))))
            sz = Int64(get(meta, "size", -1))
            mt = Int64(get(meta, "mtime", -1))
            return (digest=d, size=sz, mtime=mt)
        catch; end
    end
    return (digest="", size=Int64(-1), mtime=Int64(-1))
end

function _read_local_xnat_digest(local_h5::AbstractString)::String
    !isfile(local_h5) && return ""
    cur_sz = Int64(filesize(local_h5))
    cur_mt = round(Int64, mtime(local_h5))
    b = _read_local_xnat_baseline(local_h5)
    if !isempty(b.digest) && b.size == cur_sz && b.mtime == cur_mt
        return b.digest
    end
    return ""
end

function _write_local_xnat_meta(local_h5::AbstractString, digest::AbstractString; last_modified::AbstractString="")
    !isfile(local_h5) && return
    d = lowercase(strip(String(digest)))
    try
        open(_xnat_meta_path(local_h5), "w") do io
            JSON.print(io, Dict(
                "size" => Int64(filesize(local_h5)),
                "mtime" => round(Int64, mtime(local_h5)),
                "digest" => d,
                "last_modified" => String(last_modified)
            ))
        end
    catch; end
end

function _compute_local_md5(local_h5::AbstractString)::String
    (!isfile(local_h5) || Sys.which("md5sum") === nothing) && return ""
    try
        out = read(`md5sum $local_h5`, String)
        parts = split(strip(out))
        return !isempty(parts) ? lowercase(parts[1]) : ""
    catch
        return ""
    end
end

"""
    get_remote_preprocessed_info(creds, project_id, subject_id)

Returns metadata `(found, size, digest, last_modified_utc, uri)` for the subject's preprocessed HDF5 on XNAT.
Uses XNAT resource JSON metadata (Size + MD5 digest) and `ANNOTATIONS/files/status.json` timestamp without
performing HTTP HEAD on multi-GB binary downloads (which can block on XNAT servers).
"""
function get_remote_preprocessed_info(
    creds::XnatCredentials,
    project_id::AbstractString,
    subject_id::AbstractString
)
    p = resolve_project_id(project_id)
    s = resolve_subject_label(creds, p, subject_id)
    if isempty(p) || isempty(s)
        return (found=false, size=Int64(0), digest="", last_modified_utc=nothing, uri="")
    end

    function _fetch_status_ts()::Union{DateTime, Nothing}
        st_j = _xnat_get_json(creds, "/data/projects/$p/subjects/$s/resources/ANNOTATIONS/files/status.json"; timeout=4.0)
        if st_j !== nothing
            return _parse_remote_timestamp_utc(string(get(st_j, "updated_at", get(st_j, "last_modified", ""))))
        end
        return nothing
    end

    # 1. Subject-level MEDEYE3D resource
    medeye_files = _xnat_get_json(creds, "/data/projects/$p/subjects/$s/resources/MEDEYE3D/files?format=json"; timeout=8.0)
    if medeye_files !== nothing && haskey(medeye_files, "ResultSet") && haskey(medeye_files["ResultSet"], "Result")
        for r in medeye_files["ResultSet"]["Result"]
            fn = get(r, "Name", "")
            if _is_preprocessed_h5_filename(fn)
                remote_sz = something(tryparse(Int64, string(get(r, "Size", "0"))), Int64(0))
                remote_digest = lowercase(strip(string(get(r, "digest", ""))))
                uri = get(r, "URI", "/data/projects/$p/subjects/$s/resources/MEDEYE3D/files/$fn")
                ts_utc = _fetch_status_ts()
                return (found=true, size=remote_sz, digest=remote_digest, last_modified_utc=ts_utc, uri=uri)
            end
        end
    end

    # 2. Experiment-level MEDEYE3D fallback
    for exp in list_experiments(creds, p, s)
        exp_id = get(exp, "ID", "")
        isempty(exp_id) && continue
        exp_files = _xnat_get_json(creds, "/data/experiments/$exp_id/resources/MEDEYE3D/files?format=json"; timeout=8.0)
        if exp_files !== nothing && haskey(exp_files, "ResultSet") && haskey(exp_files["ResultSet"], "Result")
            for r in exp_files["ResultSet"]["Result"]
                fn = get(r, "Name", "")
                if _is_preprocessed_h5_filename(fn)
                    remote_sz = something(tryparse(Int64, string(get(r, "Size", "0"))), Int64(0))
                    remote_digest = lowercase(strip(string(get(r, "digest", ""))))
                    uri = get(r, "URI", "/data/experiments/$exp_id/resources/MEDEYE3D/files/$fn")
                    ts_utc = _fetch_status_ts()
                    return (found=true, size=remote_sz, digest=remote_digest, last_modified_utc=ts_utc, uri=uri)
                end
            end
        end
    end

    return (found=false, size=Int64(0), digest="", last_modified_utc=nothing, uri="")
end

"""
    is_remote_newer_than_local(creds, project_id, subject_id, local_h5) -> (needs_download::Bool, reason::String)

Checks if the remote XNAT HDF5 is newer (by `status.json` timestamp) or differs in byte size / MD5 digest.
"""
function _compare_remote_against_local(info, local_h5::AbstractString)::Tuple{Bool, String}
    if !isfile(local_h5) || filesize(local_h5) <= 1024
        return (true, "not_cached")
    end
    !info.found && return (false, "no_remote")

    local_sz = Int64(filesize(local_h5))
    local_mtime_utc = unix2datetime(mtime(local_h5))
    baseline = _read_local_xnat_baseline(local_h5)

    # When sidecar records the synced XNAT baseline digest (and optional size),
    # local autosaves that grow/touch local_h5 do not make an unchanged XNAT file "newer".
    if !isempty(info.digest) && !isempty(baseline.digest)
        remote_unchanged = (baseline.digest == info.digest) &&
            (info.size <= 0 || baseline.size <= 0 || baseline.size == info.size)
        if remote_unchanged
            if info.last_modified_utc !== nothing && info.last_modified_utc > local_mtime_utc + Second(5)
                return (true, "remote_newer")
            end
            return (false, "up_to_date")
        elseif info.last_modified_utc === nothing || info.last_modified_utc >= local_mtime_utc - Second(5)
            return (true, "digest_changed")
        else
            return (false, "up_to_date")
        end
    end

    if info.last_modified_utc !== nothing
        if info.last_modified_utc > local_mtime_utc + Second(5)
            return (true, "remote_newer")
        elseif info.size > 0 && local_sz != info.size && info.last_modified_utc >= local_mtime_utc - Second(5)
            return (true, "size_changed")
        end
    elseif info.size > 0 && local_sz != info.size
        return (true, "size_changed")
    end

    # When size matches (and local file is not newer than status.json), verify MD5 digest
    if !isempty(info.digest)
        local_dig = _read_local_xnat_digest(local_h5)
        if !isempty(local_dig)
            if local_dig != info.digest
                # Only treat digest mismatch as remote_newer if remote wasn't explicitly older than local edit
                if info.last_modified_utc === nothing || info.last_modified_utc >= local_mtime_utc - Second(5)
                    return (true, "digest_changed")
                end
            end
            return (false, "up_to_date")
        elseif info.last_modified_utc === nothing
            # No remote timestamp and no cached sidecar digest yet: compute once and cache sidecar
            comp_dig = _compute_local_md5(local_h5)
            if !isempty(comp_dig)
                _write_local_xnat_meta(local_h5, comp_dig)
                if comp_dig != info.digest
                    return (true, "digest_changed")
                end
            end
        end
    end

    return (false, "up_to_date")
end

function is_remote_newer_than_local(
    creds::XnatCredentials,
    project_id::AbstractString,
    subject_id::AbstractString,
    local_h5::AbstractString
)::Tuple{Bool, String}
    if !isfile(local_h5) || filesize(local_h5) <= 1024
        return (true, "not_cached")
    end
    info = get_remote_preprocessed_info(creds, project_id, subject_id)
    return _compare_remote_against_local(info, local_h5)
end

function _download_from_medeye3d_resource(
    creds::XnatCredentials,
    project_id::AbstractString,
    subject_id::AbstractString,
    local_dest::AbstractString,
    filename_pred::Function;
    force::Bool=false,
    progress_cb::Union{Function, Nothing}=nothing
)::Bool
    dest_path = String(local_dest)
    p = resolve_project_id(project_id)
    s = resolve_subject_label(creds, p, subject_id)

    function _should_skip_existing(remote_sz, remote_digest)::Bool
        (force || !isfile(dest_path) || filesize(dest_path) <= 1024) && return false
        local_sz = Int64(filesize(dest_path))
        rd = lowercase(strip(string(remote_digest)))
        baseline = _read_local_xnat_baseline(dest_path)
        if !isempty(rd) && !isempty(baseline.digest) && baseline.digest == rd &&
           (remote_sz === nothing || remote_sz <= 0 || baseline.size <= 0 || baseline.size == remote_sz)
            return true
        end
        if remote_sz !== nothing && remote_sz > 0 && local_sz != remote_sz
            println("[XNAT] Cached HDF5 size ($local_sz) != remote XNAT size ($remote_sz); downloading new version..."); flush(stdout)
            return false
        end
        if !isempty(rd)
            ld = _read_local_xnat_digest(dest_path)
            if !isempty(ld) && ld != rd
                println("[XNAT] Cached HDF5 MD5 ($ld) != remote XNAT MD5 ($rd); downloading new version..."); flush(stdout)
                return false
            end
        end
        return true
    end

    # 1. Search in subject MEDEYE3D resource
    medeye_files = _xnat_get_json(creds, "/data/projects/$p/subjects/$s/resources/MEDEYE3D/files?format=json")
    if medeye_files !== nothing
        try
            results = medeye_files["ResultSet"]["Result"]
            for r in results
                fn = get(r, "Name", "")
                if filename_pred(fn)
                    remote_sz = tryparse(Int64, string(get(r, "Size", "0")))
                    remote_dig = string(get(r, "digest", ""))
                    uri = get(r, "URI", "/data/projects/$p/subjects/$s/resources/MEDEYE3D/files/$fn")
                    _should_skip_existing(remote_sz, remote_dig) && return true
                    if download_file(creds, uri, dest_path; progress_cb=progress_cb)
                        !isempty(remote_dig) && _write_local_xnat_meta(dest_path, remote_dig)
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
                if filename_pred(fn)
                    remote_sz = tryparse(Int64, string(get(r, "Size", "0")))
                    remote_dig = string(get(r, "digest", ""))
                    uri = get(r, "URI", "/data/experiments/$exp_id/resources/MEDEYE3D/files/$fn")
                    _should_skip_existing(remote_sz, remote_dig) && return true
                    println("[XNAT] Downloading $fn from experiment $exp_id")
                    if download_file(creds, uri, dest_path; progress_cb=progress_cb)
                        !isempty(remote_dig) && _write_local_xnat_meta(dest_path, remote_dig)
                        return true
                    end
                end
            end
        end
    end

    return false
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
    return _download_from_medeye3d_resource(creds, project_id, subject_id, local_dest, _is_preprocessed_h5_filename; force=force, progress_cb=progress_cb)
end

"""
    download_lesion_edits_hdf5(creds::XnatCredentials, project_id, subject_id, local_dest; force=false, progress_cb=nothing) -> Bool

Downloads the separate per-lesion edit history HDF5 (`*_lesion_edits.h5`) from XNAT `MEDEYE3D` resource if present.
"""
function download_lesion_edits_hdf5(
    creds::XnatCredentials,
    project_id::AbstractString,
    subject_id::AbstractString,
    local_dest::AbstractString;
    force::Bool=false,
    progress_cb::Union{Function, Nothing}=nothing
)::Bool
    return _download_from_medeye3d_resource(creds, project_id, subject_id, local_dest, _is_lesion_edits_h5_filename; force=force, progress_cb=progress_cb)
end

function _upload_to_medeye3d_resource(
    creds::XnatCredentials,
    project_id::AbstractString,
    subject_id::AbstractString,
    local_h5::AbstractString,
    subject_remote_fname::AbstractString
)::Bool
    h5_path = String(local_h5)
    !isfile(h5_path) && return false
    p = resolve_project_id(project_id)
    s = resolve_subject_label(creds, p, subject_id)

    # 1. Try subject-level resource
    if ensure_subject_resource(creds, p, s, "MEDEYE3D")
        remote_url = "/data/projects/$p/subjects/$s/resources/MEDEYE3D/files/$(subject_remote_fname)?overwrite=true&inbody=true"
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
        res_url = "$(creds.server_url)/data/experiments/$exp_id/resources/MEDEYE3D"
        headers = XnatAuth.get_auth_headers(creds)
        try
            Downloads.request(res_url; method="PUT", headers=headers, timeout=10.0)
        catch; end

        fname = basename(h5_path)
        remote_url = "/data/experiments/$exp_id/resources/MEDEYE3D/files/$(fname)?overwrite=true&inbody=true"
        return upload_file(creds, remote_url, h5_path)
    end

    @warn "Cannot access or create MEDEYE3D resource on remote XNAT for $p/$s. Remote upload skipped."
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
    p = resolve_project_id(project_id)
    s = resolve_subject_label(creds, p, subject_id)
    return _upload_to_medeye3d_resource(creds, p, s, local_h5, "$(s)_preprocessed.h5")
end

"""
    upload_lesion_edits_hdf5(creds::XnatCredentials, project_id, subject_id, local_edits_h5) -> Bool

Uploads the separate lesion-editing HDF5 (`lesion_edits.h5`) to XNAT `MEDEYE3D` resource as `{subj}_lesion_edits.h5`.
"""
function upload_lesion_edits_hdf5(
    creds::XnatCredentials,
    project_id::AbstractString,
    subject_id::AbstractString,
    local_edits_h5::AbstractString
)::Bool
    p = resolve_project_id(project_id)
    s = resolve_subject_label(creds, p, subject_id)
    return _upload_to_medeye3d_resource(creds, p, s, local_edits_h5, "$(s)_lesion_edits.h5")
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

    # 1. Merge prior remote modified_users with current payload so multi-user history is preserved
    merged_status = deepcopy(status_dict)
    prior_users = String[]
    try
        rem_status = _xnat_get_json(creds, "/data/projects/$p/subjects/$s/resources/ANNOTATIONS/files/status.json"; timeout=4.0)
        if rem_status !== nothing
            append!(prior_users, extract_status_users(rem_status))
        end
    catch; end
    cur_users = extract_status_users(merged_status)
    all_users = copy(prior_users)
    for u in cur_users
        any(x -> lowercase(x) == lowercase(u), all_users) || push!(all_users, u)
    end
    if !isempty(all_users)
        merged_status["modified_users"] = all_users
        if get(merged_status, "manually_modified", false) == true && isempty(strip(string(get(merged_status, "modified_by", ""))))
            merged_status["modified_by"] = last(all_users)
        end
        sum_d = get(merged_status, "summary", nothing)
        if sum_d isa AbstractDict
            sum_d["modified_users"] = all_users
        end
    end

    tmp_json = tempname() * ".json"
    open(tmp_json, "w") do io
        JSON.print(io, merged_status, 2)
    end
    status_ok = upload_file(creds, "/data/projects/$p/subjects/$s/resources/ANNOTATIONS/files/status.json?overwrite=true&inbody=true", tmp_json; content_type="application/json")
    try rm(tmp_json, force=true) catch end

    # 2. Upload expert HDF5 if provided
    if local_expert_h5 !== nothing && isfile(local_expert_h5)
        upload_file(creds, "/data/projects/$p/subjects/$s/resources/ANNOTATIONS/files/$(s)_annotations.h5?overwrite=true&inbody=true", String(local_expert_h5))
    end

    return status_ok
end

"""
    prefetch_all_project_hdf5(creds, project_id; cache_base, progress_cb, cancel_ref) -> (ok::Int, skip::Int, fail::Int)

Downloads ALL preprocessed HDF5 files from every subject in a project to local cache.
Skips subjects whose HDF5 is already locally cached (size match).

* `progress_cb(done, total, subject_id, status_msg)` — called after each subject.
* `cancel_ref::Ref{Bool}` — set to `true` to abort early.

Returns (downloaded_count, skipped_count, failed_count).
"""
function prefetch_all_project_hdf5(
    creds::XnatCredentials,
    project_id::AbstractString;
    cache_base::AbstractString = "",
    progress_cb::Union{Function, Nothing} = nothing,
    cancel_ref::Ref{Bool} = Ref(false)
)::Tuple{Int, Int, Int}
    p = resolve_project_id(project_id)

    # 1. List all subjects
    subjects = list_subjects(creds, p)
    total = length(subjects)
    if total == 0
        @info "[XNAT-PREFETCH] No subjects found in project $p"
        return (0, 0, 0)
    end

    # Sort subjects by label for deterministic ordering
    sort!(subjects, by = s -> get(s, "label", get(s, "ID", "")))

    # 2. Determine cache base
    if isempty(cache_base)
        if Sys.iswindows()
            appdata = get(ENV, "APPDATA", "")
            cache_base = !isempty(appdata) ? joinpath(appdata, "MedEye3D", "xnat_cache") : joinpath(homedir(), "AppData", "Roaming", "MedEye3D", "xnat_cache")
        else
            cache_base = joinpath(homedir(), ".medeye3d", "xnat_cache")
        end
    end

    clean_p = replace(String(p), r"[^a-zA-Z0-9_\-]" => "_")
    ok_count = 0
    skip_count = 0
    fail_count = 0

    println("[XNAT-PREFETCH] Starting prefetch of $total subjects from $p")
    flush(stdout)

    for (i, subj_dict) in enumerate(subjects)
        cancel_ref[] && break

        subj_label = get(subj_dict, "label", get(subj_dict, "ID", ""))
        clean_s = replace(String(subj_label), r"[^a-zA-Z0-9_\-]" => "_")
        cache_dir = joinpath(cache_base, clean_p, clean_s)
        target_h5 = joinpath(cache_dir, "$(clean_s)_preprocessed.h5")

        # Check remote XNAT freshness: download if missing, newer timestamp, or changed size
        was_cached = isfile(target_h5) && filesize(target_h5) > 1024
        needs_dl, dl_reason = try
            is_remote_newer_than_local(creds, p, subj_label, target_h5)
        catch
            (!was_cached, was_cached ? "up_to_date" : "not_cached")
        end
        if was_cached && !needs_dl
            skip_count += 1
            progress_cb !== nothing && progress_cb(i, total, subj_label, "cached")
            continue
        end

        # Attempt download (force=true when remote timestamp is newer so updated files are fetched)
        try
            mkpath(cache_dir)
            dl_ok = download_preprocessed_hdf5(creds, p, subj_label, target_h5; force=(dl_reason == "remote_newer"))
            if dl_ok && isfile(target_h5) && filesize(target_h5) > 1024
                try
                    edits_dest = joinpath(cache_dir, "$(clean_s)_preprocessed_lesion_edits.h5")
                    download_lesion_edits_hdf5(creds, p, subj_label, edits_dest; force=false)
                catch; end
                ok_count += 1
                sz_mb = round(filesize(target_h5) / 1024 / 1024, digits=1)
                tag_dl = was_cached ? "updated" : "downloaded"
                println("[XNAT-PREFETCH] ($i/$total) ✓ $subj_label ($sz_mb MB, $tag_dl: $dl_reason)")
                progress_cb !== nothing && progress_cb(i, total, subj_label, tag_dl)
            elseif was_cached
                skip_count += 1
                progress_cb !== nothing && progress_cb(i, total, subj_label, "cached")
            else
                fail_count += 1
                println("[XNAT-PREFETCH] ($i/$total) ○ $subj_label — no HDF5 on XNAT")
                progress_cb !== nothing && progress_cb(i, total, subj_label, "no_hdf5")
            end
        catch e
            fail_count += 1
            @warn "[XNAT-PREFETCH] ($i/$total) ✗ $subj_label — error" exception=e
            progress_cb !== nothing && progress_cb(i, total, subj_label, "error")
        end

        flush(stdout)
    end

    cancelled = cancel_ref[]
    println("[XNAT-PREFETCH] $(cancelled ? "Cancelled" : "Complete"): $ok_count downloaded, $skip_count cached, $fail_count no-data (total $total)")
    flush(stdout)
    return (ok_count, skip_count, fail_count)
end

end # module XnatClient
