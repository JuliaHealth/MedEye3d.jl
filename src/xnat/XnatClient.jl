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
       list_scans,
       list_subject_resources,
       get_case_status,
       download_file,
       upload_file,
       ensure_subject_resource,
       download_preprocessed_hdf5,
       upload_preprocessed_hdf5,
       save_annotations_to_xnat,
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

function _xnat_get_json(creds::XnatCredentials, endpoint::String; timeout::Float64=15.0)::Union{Dict{String, Any}, Nothing}
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
    list_subjects(creds::XnatCredentials, project_id::String) -> Vector{Dict{String, Any}}

Queries `GET /data/projects/{project_id}/subjects?format=json`.
"""
function list_subjects(creds::XnatCredentials, project_id::String)::Vector{Dict{String, Any}}
    clean_proj = strip(project_id)
    isempty(clean_proj) && return Dict{String, Any}[]
    data = _xnat_get_json(creds, "/data/projects/$(clean_proj)/subjects?format=json")
    data === nothing && return Dict{String, Any}[]
    try
        res = data["ResultSet"]["Result"]
        return Vector{Dict{String, Any}}(res)
    catch e
        @warn "Failed to parse subjects list for $project_id: $e"
        return Dict{String, Any}[]
    end
end

"""
    list_experiments(creds::XnatCredentials, project_id::String, subject_id::String) -> Vector{Dict{String, Any}}

Queries imaging sessions (experiments) for a subject.
"""
function list_experiments(creds::XnatCredentials, project_id::String, subject_id::String)::Vector{Dict{String, Any}}
    endpoint = "/data/projects/$(strip(project_id))/subjects/$(strip(subject_id))/experiments?format=json"
    data = _xnat_get_json(creds, endpoint)
    data === nothing && return Dict{String, Any}[]
    try
        res = data["ResultSet"]["Result"]
        return Vector{Dict{String, Any}}(res)
    catch; return Dict{String, Any}[]; end
end

"""
    list_scans(creds::XnatCredentials, experiment_id::String) -> Vector{Dict{String, Any}}

Queries all scan series for an imaging session.
"""
function list_scans(creds::XnatCredentials, experiment_id::String)::Vector{Dict{String, Any}}
    endpoint = "/data/experiments/$(strip(experiment_id))/scans?format=json"
    data = _xnat_get_json(creds, endpoint)
    data === nothing && return Dict{String, Any}[]
    try
        res = data["ResultSet"]["Result"]
        return Vector{Dict{String, Any}}(res)
    catch; return Dict{String, Any}[]; end
end

"""
    list_subject_resources(creds::XnatCredentials, project_id::String, subject_id::String) -> Vector{String}

Returns a list of resource folder names attached to the subject (e.g. `["MEDEYE3D", "ANNOTATIONS"]`).
"""
function list_subject_resources(creds::XnatCredentials, project_id::String, subject_id::String)::Vector{String}
    endpoint = "/data/projects/$(strip(project_id))/subjects/$(strip(subject_id))/resources?format=json"
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
    get_case_status(creds::XnatCredentials, project_id::String, subject_id::String) -> CaseStatusInfo

Determines whether a patient's case is:
- `:PROCESSED`: Validated by expert (`resources/ANNOTATIONS/files/status.json` has `status == "PROCESSED"`)
- `:PREPROCESSED`: `{Subject}_preprocessed.h5` exists in `resources/MEDEYE3D/files/`
- `:RAW`: Only raw DICOM scans exist on XNAT
"""
function get_case_status(creds::XnatCredentials, project_id::String, subject_id::String)::CaseStatusInfo
    p = strip(project_id)
    s = strip(subject_id)
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

    return CaseStatusInfo(:RAW)
end

"""
    download_file(creds::XnatCredentials, remote_endpoint_or_url::String, local_dest::String) -> Bool

Streams a file from XNAT to local disk.
"""
function download_file(creds::XnatCredentials, remote_url_or_endpoint::String, local_dest::String)::Bool
    full_url = startswith(remote_url_or_endpoint, "http") ? remote_url_or_endpoint : "$(creds.server_url)$(remote_url_or_endpoint)"
    headers = XnatAuth.get_auth_headers(creds)
    try
        mkpath(dirname(local_dest))
        Downloads.download(full_url, local_dest; headers=headers)
        return isfile(local_dest) && filesize(local_dest) > 0
    catch e
        @warn "Download failed from $full_url: $e"
        return false
    end
end

"""
    upload_file(creds::XnatCredentials, remote_endpoint_or_url::String, local_src::String; content_type::String="application/octet-stream") -> Bool

Uploads a local file to XNAT via HTTP PUT.
"""
function upload_file(
    creds::XnatCredentials,
    remote_url_or_endpoint::String,
    local_src::String;
    content_type::String="application/octet-stream"
)::Bool
    !isfile(local_src) && return false
    full_url = startswith(remote_url_or_endpoint, "http") ? remote_url_or_endpoint : "$(creds.server_url)$(remote_url_or_endpoint)"
    headers = [
        XnatAuth.get_auth_headers(creds)...,
        "Content-Type" => content_type
    ]
    try
        open(local_src, "r") do in_io
            buf = IOBuffer()
            resp = Downloads.request(full_url; method="PUT", headers=headers, input=in_io, output=buf, timeout=600.0)
            return resp.status in (200, 201)
        end
    catch e
        @warn "Upload failed to $full_url: $e"
        return false
    end
end

"""
    ensure_subject_resource(creds::XnatCredentials, project_id::String, subject_id::String, resource_name::String) -> Bool

Creates the specified resource folder under the subject if it does not yet exist.
"""
function ensure_subject_resource(creds::XnatCredentials, project_id::String, subject_id::String, resource_name::String)::Bool
    res_url = "$(creds.server_url)/data/projects/$(strip(project_id))/subjects/$(strip(subject_id))/resources/$(strip(resource_name))"
    headers = XnatAuth.get_auth_headers(creds)
    try
        buf = IOBuffer()
        resp = Downloads.request(res_url; method="PUT", headers=headers, output=buf, timeout=10.0)
        return resp.status in (200, 201)
    catch e
        @warn "Error ensuring resource $resource_name: $e"
        return false
    end
end

"""
    download_preprocessed_hdf5(creds::XnatCredentials, project_id::String, subject_id::String, local_dest::String; force::Bool=false) -> Bool

Downloads `{subject_id}_preprocessed.h5` from XNAT subject resource `MEDEYE3D`.
Skips download if file already exists locally and `force == false`.
"""
function download_preprocessed_hdf5(
    creds::XnatCredentials,
    project_id::String,
    subject_id::String,
    local_dest::String;
    force::Bool=false
)::Bool
    if !force && isfile(local_dest) && filesize(local_dest) > 1024
        return true
    end

    p = strip(project_id)
    s = strip(subject_id)
    # Search for preprocessed hdf5 in MEDEYE3D resource
    medeye_files = _xnat_get_json(creds, "/data/projects/$p/subjects/$s/resources/MEDEYE3D/files?format=json")
    target_name = "$(s)_preprocessed.h5"
    if medeye_files !== nothing
        try
            for r in medeye_files["ResultSet"]["Result"]
                fn = get(r, "Name", "")
                if endswith(fn, ".h5") || endswith(fn, ".hdf5")
                    target_name = fn
                    break
                end
            end
        catch; end
    end

    remote_file_url = "/data/projects/$p/subjects/$s/resources/MEDEYE3D/files/$(target_name)"
    return download_file(creds, remote_file_url, local_dest)
end

"""
    upload_preprocessed_hdf5(creds::XnatCredentials, project_id::String, subject_id::String, local_h5::String) -> Bool

Uploads preprocessed HDF5 to `/data/projects/{proj}/subjects/{subj}/resources/MEDEYE3D/files/{subj}_preprocessed.h5?overwrite=true`.
"""
function upload_preprocessed_hdf5(
    creds::XnatCredentials,
    project_id::String,
    subject_id::String,
    local_h5::String
)::Bool
    !isfile(local_h5) && return false
    p = strip(project_id)
    s = strip(subject_id)
    ensure_subject_resource(creds, p, s, "MEDEYE3D")
    
    remote_url = "/data/projects/$p/subjects/$s/resources/MEDEYE3D/files/$(s)_preprocessed.h5?overwrite=true"
    return upload_file(creds, remote_url, local_h5)
end

"""
    save_annotations_to_xnat(creds::XnatCredentials, project_id::String, subject_id::String, status_dict::Dict{String, Any}; local_expert_h5::Union{String, Nothing}=nothing) -> Bool

Uploads `status.json` and optionally the expert modifications HDF5 to `resources/ANNOTATIONS`.
"""
function save_annotations_to_xnat(
    creds::XnatCredentials,
    project_id::String,
    subject_id::String,
    status_dict::Dict{String, Any};
    local_expert_h5::Union{String, Nothing}=nothing
)::Bool
    p = strip(project_id)
    s = strip(subject_id)
    ensure_subject_resource(creds, p, s, "ANNOTATIONS")

    # 1. Upload status.json
    tmp_json = tempname() * ".json"
    open(tmp_json, "w") do io
        JSON.print(io, status_dict, 2)
    end
    status_ok = upload_file(creds, "/data/projects/$p/subjects/$s/resources/ANNOTATIONS/files/status.json?overwrite=true", tmp_json; content_type="application/json")
    try rm(tmp_json, force=true) catch end

    # 2. Upload expert HDF5 if provided
    if local_expert_h5 !== nothing && isfile(local_expert_h5)
        upload_file(creds, "/data/projects/$p/subjects/$s/resources/ANNOTATIONS/files/$(s)_annotations.h5?overwrite=true", local_expert_h5)
    end

    return status_ok
end

end # module XnatClient
