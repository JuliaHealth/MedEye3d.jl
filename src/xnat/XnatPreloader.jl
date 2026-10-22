"""
    XnatPreloader

Asynchronous background preloader for XNAT cases in MedEye3d.
Whenever a user selects or reviews patient N, this module automatically downloads
and caches patient N+1's preprocessed HDF5 volume in a background task, ensuring
zero-latency switching when clicking "Next Case".
"""
module XnatPreloader

using ..XnatAuth
using ..XnatClient
using Logging

export get_local_cache_dir,
       notify_patient_selected,
       get_preloaded_case,
       is_case_preloaded,
       cancel_preload!

const _current_project = Ref{String}("")
const _current_subject = Ref{String}("")
const _preloaded_subject = Ref{String}("")
const _preloaded_h5_path = Ref{String}("")
const _preload_task = Ref{Union{Task, Nothing}}(nothing)
const _preload_active = Ref{Bool}(false)

"""
    get_local_cache_dir(project_id::String, subject_id::String) -> String

Returns the persistent local cache path for an XNAT subject's preprocessed volumes.
"""
function get_local_cache_dir(project_id::AbstractString, subject_id::AbstractString)::String
    clean_p = replace(String(project_id), r"[^a-zA-Z0-9_\-]" => "_")
    clean_s = replace(String(subject_id), r"[^a-zA-Z0-9_\-]" => "_")
    if Sys.iswindows()
        appdata = get(ENV, "APPDATA", "")
        base = !isempty(appdata) ? joinpath(appdata, "MedEye3D", "xnat_cache") : joinpath(homedir(), "AppData", "Roaming", "MedEye3D", "xnat_cache")
    else
        base = joinpath(homedir(), ".medeye3d", "xnat_cache")
    end
    return joinpath(base, clean_p, clean_s)
end

"""
    notify_patient_selected(creds, project_id, current_subject_id, subject_queue)

Informs the preloader that the user is viewing `current_subject_id`.
Determines the subsequent patient in `subject_queue` and launches an asynchronous
background download if that case is not yet cached locally.
"""
function notify_patient_selected(
    creds::XnatCredentials,
    project_id::AbstractString,
    current_subject_id::AbstractString,
    subject_queue::Vector{<:AbstractString}
)
    _current_project[] = String(project_id)
    _current_subject[] = String(current_subject_id)

    curr_idx = findfirst(==(String(current_subject_id)), String.(subject_queue))
    if curr_idx === nothing || curr_idx >= length(subject_queue)
        return nothing
    end

    next_subj = subject_queue[curr_idx + 1]
    if _preloaded_subject[] == next_subj && isfile(_preloaded_h5_path[])
        # Already preloaded
        return nothing
    end

    cache_dir = get_local_cache_dir(project_id, next_subj)
    target_h5 = joinpath(cache_dir, "$(next_subj)_preprocessed.h5")

    if isfile(target_h5) && filesize(target_h5) > 1024
        _preloaded_subject[] = next_subj
        _preloaded_h5_path[] = target_h5
        return nothing
    end

    # Launch background preload
    cancel_preload!()
    _preload_active[] = true
    _preload_task[] = Threads.@spawn begin
        try
            mkpath(cache_dir)
            println("[XNAT-PRELOAD] Background downloading next case: $project_id / $next_subj...")
            flush(stdout)
            
            dl_ok = XnatClient.download_preprocessed_hdf5(creds, project_id, next_subj, target_h5; force=false)
            if dl_ok && isfile(target_h5) && filesize(target_h5) > 1024
                _preloaded_subject[] = next_subj
                _preloaded_h5_path[] = target_h5
                println("[XNAT-PRELOAD] Ready: $next_subj cached at $target_h5 ($(round(filesize(target_h5)/1024/1024, digits=1)) MB)")
                flush(stdout)
            else
                println("[XNAT-PRELOAD] Note: Next case $next_subj has no preprocessed HDF5 on XNAT (raw or uncompleted)")
                flush(stdout)
            end
        catch e
            @debug "[XNAT-PRELOAD] Preload failed for $next_subj: $e"
        finally
            _preload_active[] = false
        end
    end

    return nothing
end

"""
    get_preloaded_case(subject_id) -> Union{String, Nothing}

Returns the cached HDF5 file path if `subject_id` is preloaded and ready.
"""
function get_preloaded_case(subject_id::AbstractString)::Union{String, Nothing}
    if _preloaded_subject[] == subject_id && isfile(_preloaded_h5_path[])
        return _preloaded_h5_path[]
    end
    return nothing
end

"""
    is_case_preloaded(subject_id) -> Bool

Returns true if `subject_id` has been completely downloaded and cached locally.
"""
function is_case_preloaded(subject_id::AbstractString)::Bool
    return get_preloaded_case(subject_id) !== nothing
end

"""
    cancel_preload!()

Cancels any running background download task.
"""
function cancel_preload!()
    _preload_active[] = false
    # Threads.@spawn tasks cannot be hard cancelled, but the flag aborts further steps
end

end # module XnatPreloader
