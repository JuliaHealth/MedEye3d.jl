module InferenceClient

using Sockets
using JSON
using Base64
using MedImages
using ..ConnectedComponents

export start_python_worker, run_helpnet_inference, run_nninteractive, run_bone_subsegmentation_remote,
       insert_patch!, preload_ct_for_nninteractive, send_json_request,
       prompt_start_ai_models, is_worker_reachable, is_ai_enabled, set_ai_enabled!,
       get_last_ai_error, set_last_ai_error!, find_ai_script, get_inference_dir

global PYTHON_PROC = nothing
const AI_ENABLED = Ref{Bool}(true)
const LAST_AI_ERROR = Ref{String}("")

"""
    get_last_ai_error() -> String

Returns the most recent descriptive error encountered when starting or communicating with the AI worker.
"""
function get_last_ai_error()::String
    return LAST_AI_ERROR[]
end

"""
    set_last_ai_error!(err::String)

Sets the most recent descriptive error message for AI worker diagnostics.
"""
function set_last_ai_error!(err::String)
    LAST_AI_ERROR[] = err
    return err
end

"""
    find_ai_script(script_name::String) -> String

Finds the absolute path to an AI helper script across development, bundled, installed, and user directories.
"""
function find_ai_script(script_name::String)::String
    candidates = [
        joinpath(dirname(Sys.BINDIR), "scripts", "ai", script_name),
        joinpath(Sys.BINDIR, "scripts", "ai", script_name),
        normpath(joinpath(@__DIR__, "..", "..", "scripts", "ai", script_name)),
        joinpath(pwd(), "scripts", "ai", script_name),
        joinpath(homedir(), ".medeye3d", "scripts", "ai", script_name)
    ]
    for c in candidates
        if isfile(c)
            return normpath(c)
        end
    end
    return ""
end

"""
    get_inference_dir() -> String

Returns a verified writable directory for temporary inference volume data exchange with the AI container.
"""
function get_inference_dir()::String
    # 1. Dev directory if exists and writable
    dev_dir = normpath(joinpath(@__DIR__, "..", "..", "tmp_inference"))
    if isdir(dirname(dev_dir))
        try
            mkpath(dev_dir)
            test_file = joinpath(dev_dir, ".perm_test")
            write(test_file, "test")
            rm(test_file; force=true)
            return dev_dir
        catch
        end
    end
    # 2. User profile directory
    user_dir = joinpath(homedir(), ".medeye3d", "tmp_inference")
    try
        mkpath(user_dir)
        return user_dir
    catch
    end
    # 3. System temp directory
    tmp_dir = joinpath(tempdir(), "medeye3d_inference")
    mkpath(tmp_dir)
    return tmp_dir
end

"""
    is_ai_enabled() -> Bool

Returns whether AI inference models are currently enabled for this session.
"""
function is_ai_enabled()::Bool
    return AI_ENABLED[]
end

"""
    set_ai_enabled!(val::Bool)

Enables or disables AI inference models for this session.
"""
function set_ai_enabled!(val::Bool)
    AI_ENABLED[] = val
    if !val
        ENV["MEDEYE3D_START_AI"] = "0"
    else
        ENV["MEDEYE3D_START_AI"] = "1"
    end
    return val
end

"""
    get_ai_host()::String

Returns the remote or local AI worker hostname / IP address configured via `MEDEYE3D_AI_HOST`
(defaults to `"127.0.0.1"` for local execution or SSH port-forwarded tunnel).
"""
function get_ai_host()::String
    return get(ENV, "MEDEYE3D_AI_HOST", "127.0.0.1")
end

"""
    get_ai_port(default_port=5005)::Int

Returns the AI worker TCP port configured via `MEDEYE3D_AI_PORT` (defaults to 5005).
"""
function get_ai_port(default_port=5005)::Int
    return parse(Int, get(ENV, "MEDEYE3D_AI_PORT", string(default_port)))
end

"""
    is_worker_reachable(; host=get_ai_host(), port=get_ai_port())::Bool

Quickly tests if the AI worker TCP server is reachable and responding to commands.
"""
function is_worker_reachable(; host=get_ai_host(), port=get_ai_port())::Bool
    try
        conn = connect(host, port)
        write(conn, JSON.json(Dict("command" => "ping")))
        resp_str = read(conn, String)
        close(conn)
        resp = JSON.parse(resp_str)
        return get(resp, "status", "") == "success"
    catch
        return false
    end
end

"""
    prompt_start_ai_models(args::Vector{String}=ARGS) -> Bool

Prompts the user on startup whether to run the local AI inference models (nnInteractive & HELPNet).
Checks CLI flags (`--ai` / `--no-ai`), environment variables (`MEDEYE3D_START_AI`),
persistent configuration, and native OS dialogs.
"""
function prompt_start_ai_models(args::Vector{String}=ARGS)::Bool
    # 1. CLI flags have highest priority
    if any(a -> a in ("--no-ai", "--disable-ai", "--without-ai"), args)
        println("[InferenceClient] AI inference models disabled via CLI flag.")
        set_ai_enabled!(false)
        return false
    end
    if any(a -> a in ("--ai", "--enable-ai", "--with-ai"), args)
        println("[InferenceClient] AI inference models enabled via CLI flag.")
        set_ai_enabled!(true)
        return true
    end

    # 2. Environment variable
    env_ai = lowercase(strip(get(ENV, "MEDEYE3D_START_AI", "")))
    if env_ai in ("0", "false", "no", "disable", "disabled")
        println("[InferenceClient] AI inference models disabled via MEDEYE3D_START_AI.")
        set_ai_enabled!(false)
        return false
    elseif env_ai in ("1", "true", "yes", "enable", "enabled")
        println("[InferenceClient] AI inference models enabled via MEDEYE3D_START_AI.")
        set_ai_enabled!(true)
        return true
    end

    # 3. Headless / automated / CI environment
    if haskey(ENV, "CI") || get(ENV, "MEDEYE3D_NONINTERACTIVE", "") in ("1", "true")
        println("[InferenceClient] Headless / CI environment detected. Defaulting to viewer mode.")
        set_ai_enabled!(false)
        return false
    end

    # 4. Persistent configuration
    cfg_path = joinpath(homedir(), ".medeye3d_display_config.json")
    if isfile(cfg_path)
        try
            cfg = JSON.parse(read(cfg_path, String))
            pref = lowercase(strip(get(cfg, "start_ai_models", "ask")))
            if pref in ("always", "true", "yes", "enable")
                println("[InferenceClient] AI inference models enabled via saved preference.")
                set_ai_enabled!(true)
                return true
            elseif pref in ("never", "false", "no", "disable")
                println("[InferenceClient] AI inference models disabled via saved preference.")
                set_ai_enabled!(false)
                return false
            end
        catch e
            @debug "Failed to read display config: $e"
        end
    end

    # 5. If already reachable, inform user and use it
    if is_worker_reachable()
        println("[InferenceClient] AI Worker is already active at $(get_ai_host()):$(get_ai_port()).")
        set_ai_enabled!(true)
        return true
    end

    # 6. Native dialog prompt on Windows
    if Sys.iswindows()
        prompt_text = "Would you like to run the AI inference models (nnInteractive & HELPNet) locally?\\n\\nIn case of capable hardware (GPU / Docker / Python), running the models locally enables interactive AI segmentation tools automatically.\\n\\n• Click 'Yes' to run/connect local AI models\\n• Click 'No' for viewer-only mode"
        prompt_title = "MedEye3D - AI Inference Models"
        ps_cmd = """
        Add-Type -AssemblyName System.Windows.Forms
        \$res = [System.Windows.Forms.MessageBox]::Show(
            "$prompt_text",
            "$prompt_title",
            [System.Windows.Forms.MessageBoxButtons]::YesNo,
            [System.Windows.Forms.MessageBoxIcon]::Question
        )
        Write-Output \$res
        """
        try
            out = read(`powershell -NoProfile -Command $ps_cmd`, String)
            result = strip(out)
            if result == "Yes"
                println("[InferenceClient] User selected YES to run AI inference models.")
                set_ai_enabled!(true)
                return true
            else
                println("[InferenceClient] User selected NO to AI inference models.")
                set_ai_enabled!(false)
                return false
            end
        catch e
            @warn "[InferenceClient] Failed to display native dialog: $e"
            set_ai_enabled!(false)
            return false
        end
    elseif isinteractive()
        print("Would you like to run the AI inference models (nnInteractive & HELPNet)? [y/N]: ")
        flush(stdout)
        ans = readline()
        selected = lowercase(strip(ans)) in ("y", "yes")
        set_ai_enabled!(selected)
        return selected
    end

    set_ai_enabled!(false)
    return false
end

function send_json_request(req::Dict; port=get_ai_port())
    host = get_ai_host()
    try
        conn = connect(host, port)
        write(conn, JSON.json(req))
        resp_str = read(conn, String)
        close(conn)
        return JSON.parse(resp_str)
    catch e
        return Dict("status" => "error", "message" => string(e))
    end
end

"""
    start_python_worker(worker_script_path::String = "") -> Bool

Ensures the MedEye3d AI inference worker is reachable at host:port.
If not running locally, attempts to start it via Docker (or local Python).
Returns `true` if connected successfully, or `false` otherwise.
"""
function start_python_worker(worker_script_path::String = "")::Bool
    host = get_ai_host()
    port = get_ai_port()
    println("[InferenceClient] Ensuring MedEye3d AI Worker is reachable at $host:$port..."); flush(stdout)

    # 1. If already connected, return immediately
    if is_worker_reachable(; host=host, port=port)
        println("[InferenceClient] AI Worker is ready and connected at $host:$port."); flush(stdout)
        set_ai_enabled!(true)
        set_last_ai_error!("")
        try verify_docker_code_sync() catch end
        return true
    end

    fatal_launcher_error = false

    # 2. Try to start local worker if host is local
    if host in ("127.0.0.1", "localhost")
        if Sys.iswindows()
            # Try start_docker_worker.ps1
            ps_script = find_ai_script("start_docker_worker.ps1")
            inf_dir = get_inference_dir()
            if !isempty(ps_script)
                println("[InferenceClient] Launching Windows Docker AI worker ($ps_script)..."); flush(stdout)
                try
                    out_buf = IOBuffer()
                    err_buf = IOBuffer()
                    cmd = `powershell -NoProfile -ExecutionPolicy Bypass -File $ps_script -Port $port -InferenceDir $inf_dir`
                    proc = run(pipeline(ignorestatus(cmd), stdout=out_buf, stderr=err_buf), wait=true)
                    out_str = String(take!(out_buf))
                    err_str = String(take!(err_buf))
                    exit_code = proc.exitcode

                    if !isempty(out_str)
                        print("[InferenceClient] Docker launcher: $out_str"); flush(stdout)
                    end
                    if !isempty(err_str)
                        print("[InferenceClient] Docker launcher stderr: $err_str"); flush(stdout)
                    end

                    if exit_code != 0
                        err_reason = if exit_code == 1
                            "Docker CLI not found in PATH. Please install Docker Desktop (https://www.docker.com) and ensure 'docker' is in your system PATH."
                        elseif exit_code == 2
                            "Docker Desktop daemon is not running. Please start Docker Desktop and ensure the Docker engine is running."
                        elseif exit_code == 3
                            "Failed to build Docker image 'medeye3d-ai:latest'. Check Docker Desktop logs and disk space."
                        elseif exit_code == 4
                            "Docker image 'medeye3d-ai:latest' not found."
                        elseif exit_code == 5
                            "Failed to start Docker container 'medeye3d-ai'. Check Docker Desktop logs."
                        else
                            "Docker launch script exited with code $exit_code: $(strip(err_str))"
                        end
                        set_last_ai_error!(err_reason)
                        println("[InferenceClient ERROR] $err_reason"); flush(stdout)
                        fatal_launcher_error = true
                    end
                catch e
                    err_msg = "Failed to run Docker launcher script: $e"
                    set_last_ai_error!(err_msg)
                    println("[InferenceClient ERROR] $err_msg"); flush(stdout)
                    fatal_launcher_error = true
                end
            else
                err_msg = "Could not find 'start_docker_worker.ps1' in application bundle or repository."
                set_last_ai_error!(err_msg)
                println("[InferenceClient WARNING] $err_msg"); flush(stdout)
            end

            # If Docker launch had fatal error and port is not open, check local Python worker fallback
            if !is_worker_reachable(; host=host, port=port)
                py_script = !isempty(worker_script_path) && isfile(worker_script_path) ?
                    worker_script_path :
                    find_ai_script("python_worker.py")
                if !isempty(py_script) && isfile(py_script)
                    has_py = try
                        run(pipeline(`python -c "import torch"`, stdout=devnull, stderr=devnull), wait=true)
                        true
                    catch
                        false
                    end
                    if has_py
                        println("[InferenceClient] Attempting local python_worker.py fallback..."); flush(stdout)
                        try
                            run(`python $py_script`, wait=false)
                            fatal_launcher_error = false
                        catch e
                            println("[InferenceClient] Local python worker launch notice: $e"); flush(stdout)
                        end
                    end
                end
            end
        else
            docker_script = find_ai_script("start_docker_worker.sh")
            if !isempty(docker_script)
                out_buf = IOBuffer()
                err_buf = IOBuffer()
                cmd = `bash $docker_script`
                proc = run(pipeline(ignorestatus(cmd), stdout=out_buf, stderr=err_buf), wait=true)
                out_str = String(take!(out_buf))
                err_str = String(take!(err_buf))
                if proc.exitcode != 0
                    err_reason = "Docker script 'start_docker_worker.sh' failed with exit code $(proc.exitcode): $(strip(err_str))"
                    set_last_ai_error!(err_reason)
                    println("[InferenceClient ERROR] $err_reason"); flush(stdout)
                    fatal_launcher_error = true
                end
            else
                err_msg = "Could not find 'start_docker_worker.sh'"
                set_last_ai_error!(err_msg)
                println("[InferenceClient WARNING] $err_msg"); flush(stdout)
            end
        end
    end

    # If launcher failed fatally and port is definitely not reachable, return immediately without 45s sleep loop
    if fatal_launcher_error && !is_worker_reachable(; host=host, port=port)
        println("[InferenceClient] Aborting AI connection: Docker launcher encountered fatal error and AI port $port is offline."); flush(stdout)
        return false
    end

    # 3. Wait up to 45 seconds for the Python TCP server to be reachable & responsive
    connected = false
    for i in 1:45
        if is_worker_reachable(; host=host, port=port)
            println("[InferenceClient] AI Worker is ready at $host:$port."); flush(stdout)
            connected = true
            set_ai_enabled!(true)
            set_last_ai_error!("")
            try verify_docker_code_sync() catch end
            break
        end
        if i % 10 == 0
            println("[InferenceClient] Waiting for AI Worker to initialize ($i/45s)..."); flush(stdout)
        end
        sleep(1)
    end

    if !connected
        timeout_msg = "AI Worker not responsive at $host:$port after 45s (GPU model loading timeout or container exited)."
        set_last_ai_error!(timeout_msg)
        println("[InferenceClient] Notice: $timeout_msg"); flush(stdout)
        println("[InferenceClient] For remote GPU server over SSH, ensure tunnel is active: ssh -N -L $port:localhost:$port user@server"); flush(stdout)
        return false
    end
    return true
end

"""
Verify the Docker container is running the correct python_worker.py code.
Checks that INFERENCE_LOCK exists — if missing, the container is running stale code.
"""
function verify_docker_code_sync()
    try
        result = read(`docker exec medeye3d-ai grep -c INFERENCE_LOCK /app/python_worker.py`, String)
        count = parse(Int, strip(result))
        if count < 2
            @error "[InferenceClient] CRITICAL: Docker container running OLD python_worker.py without INFERENCE_LOCK! Run: docker rm -f medeye3d-ai && bash scripts/ai/start_docker_worker.sh"
        else
            println("[InferenceClient] ✓ Docker code sync verified (INFERENCE_LOCK present)"); flush(stdout)
        end
    catch e
        @warn "[InferenceClient] Could not verify Docker code sync: $e"
    end
end

function extract_patch(vol::Array{Float32, 3}, cx::Int, cy::Int, cz::Int; patch_size::Int=64, pad_val::Float32=0.0f0)
    w, h, d = size(vol)
    patch = fill(pad_val, patch_size, patch_size, patch_size)
    
    hw = patch_size ÷ 2
    
    src_x1 = max(1, cx - hw)
    src_x2 = min(w, cx + (patch_size - hw - 1))
    dst_x1 = 1 + (src_x1 - (cx - hw))
    dst_x2 = patch_size - ((cx + (patch_size - hw - 1)) - src_x2)

    src_y1 = max(1, cy - hw)
    src_y2 = min(h, cy + (patch_size - hw - 1))
    dst_y1 = 1 + (src_y1 - (cy - hw))
    dst_y2 = patch_size - ((cy + (patch_size - hw - 1)) - src_y2)

    src_z1 = max(1, cz - hw)
    src_z2 = min(d, cz + (patch_size - hw - 1))
    dst_z1 = 1 + (src_z1 - (cz - hw))
    dst_z2 = patch_size - ((cz + (patch_size - hw - 1)) - src_z2)
    
    patch[dst_x1:dst_x2, dst_y1:dst_y2, dst_z1:dst_z2] .= vol[src_x1:src_x2, src_y1:src_y2, src_z1:src_z2]
    return patch
end

function insert_patch!(vol::AbstractArray{T, 3}, patch::AbstractArray{<:Real, 3}, cx::Int, cy::Int, cz::Int; label_val::T=T(1)) where T
    w, h, d = size(vol)
    pw, ph, pd = size(patch)
    
    hw = pw ÷ 2
    hh = ph ÷ 2
    hd = pd ÷ 2
    
    src_x1 = max(1, cx - hw)
    src_x2 = min(w, cx + (pw - hw - 1))
    dst_x1 = 1 + (src_x1 - (cx - hw))
    dst_x2 = pw - ((cx + (pw - hw - 1)) - src_x2)

    src_y1 = max(1, cy - hh)
    src_y2 = min(h, cy + (ph - hh - 1))
    dst_y1 = 1 + (src_y1 - (cy - hh))
    dst_y2 = ph - ((cy + (ph - hh - 1)) - src_y2)

    src_z1 = max(1, cz - hd)
    src_z2 = min(d, cz + (pd - hd - 1))
    dst_z1 = 1 + (src_z1 - (cz - hd))
    dst_z2 = pd - ((cz + (pd - hd - 1)) - src_z2)
    
    mask_slice = patch[dst_x1:dst_x2, dst_y1:dst_y2, dst_z1:dst_z2]
    
    @views target_slice = vol[src_x1:src_x2, src_y1:src_y2, src_z1:src_z2]
    for i in eachindex(mask_slice)
        if mask_slice[i] > 0
            target_slice[i] = label_val
        end
    end
    vol[src_x1:src_x2, src_y1:src_y2, src_z1:src_z2] .= target_slice
end

function run_helpnet_inference(ct_vol::Array{Float32, 3}, pet_vol::Array{Float32, 3}, points_vol::Union{Nothing, Array{Float32, 3}}, cx::Int, cy::Int, cz::Int; port=get_ai_port())
    if !is_ai_enabled()
        println("[InferenceClient] HELPNet inference skipped: AI models are disabled."); flush(stdout)
        return nothing
    end

    # Docker container (medeye3d-ai) is started once at app startup — here we only communicate via TCP

    out_dir = get_inference_dir()
    mkpath(out_dir)
    
    ct_patch = extract_patch(ct_vol, cx, cy, cz, pad_val=-1000.0f0)
    pet_patch = extract_patch(pet_vol, cx, cy, cz, pad_val=0.0f0)
    # HELPNet expects a SINGLE center point as its 3rd input channel, NOT a full scribble mask.
    # When given multiple scribble voxels, HELPNet's attention is confused and returns 0 predictions.
    # The scribble coordinates are used to determine (cx,cy,cz) which centers the 64³ patch — 
    # the model then segments around this single point.
    point_patch = zeros(Float32, 64, 64, 64)
    point_patch[33, 33, 33] = 1.0f0  # single center point in the 64³ patch
    
    ct_path = joinpath(out_dir, "ct_in.nii.gz")
    pet_path = joinpath(out_dir, "pet_in.nii.gz")
    point_path = joinpath(out_dir, "point_in.nii.gz")
    
    dummy_sp = (1.0, 1.0, 1.0)
    dummy_or = (0.0, 0.0, 0.0)
    dummy_dir = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    
    im_ct = MedImage(voxel_data=ct_patch, spacing=dummy_sp, origin=dummy_or, direction=dummy_dir, image_type=MedImages.MedImage_data_struct.MRI_type, image_subtype=MedImages.MedImage_data_struct.CT_subtype, patient_id="dummy")
    im_pet = MedImage(voxel_data=pet_patch, spacing=dummy_sp, origin=dummy_or, direction=dummy_dir, image_type=MedImages.MedImage_data_struct.MRI_type, image_subtype=MedImages.MedImage_data_struct.CT_subtype, patient_id="dummy")
    im_pt = MedImage(voxel_data=point_patch, spacing=dummy_sp, origin=dummy_or, direction=dummy_dir, image_type=MedImages.MedImage_data_struct.MRI_type, image_subtype=MedImages.MedImage_data_struct.CT_subtype, patient_id="dummy")
    
    MedImages.create_nii_from_medimage(im_ct, ct_path)
    MedImages.create_nii_from_medimage(im_pet, pet_path)
    MedImages.create_nii_from_medimage(im_pt, point_path)
    
    req = Dict(
        "command" => "helpnet",
        "ct_path" => "/tmp/medeye3d_inference/$(basename(ct_path))",
        "pet_path" => "/tmp/medeye3d_inference/$(basename(pet_path))",
        "point_path" => "/tmp/medeye3d_inference/$(basename(point_path))",
        "out_dir" => "/tmp/medeye3d_inference"
    )
    
    host = get_ai_host()
    conn = nothing
    for attempt in 1:3
        try
            conn = connect(host, port)
            break
        catch e
            if attempt < 3
                sleep(1.0)
            end
        end
    end
    if conn === nothing
        last_err = get_last_ai_error()
        detail = isempty(last_err) ? "Is Docker container 'medeye3d-ai' running? Check Docker Desktop or run start_docker_worker.ps1." : last_err
        set_last_ai_error!("Connection refused at $host:$port ($detail)")
        println("[InferenceClient ERROR] Failed to connect to Python Worker at $host:$port: connection refused. $detail"); flush(stdout)
        return nothing
    end

    try
        write(conn, JSON.json(req))
        resp_str = read(conn, String)
        close(conn)
        
        resp = JSON.parse(resp_str)
        if resp["status"] == "success"
            pred_file = basename(resp["prediction_path"])
            local_pred_path = joinpath(out_dir, pred_file)
            pred_im = MedImages.load_image(local_pred_path, "unknown")
            raw_mask = Array{UInt8}(pred_im.voxel_data)
            # Post-process: extract only largest connected component using GPU/CPU KernelAbstractions
            clean_mask = try
                ConnectedComponents.extract_largest_connected_component(raw_mask)
            catch lcc_err
                println("[InferenceClient] HELPNet post-processing (LCC) fallback to raw mask: $lcc_err"); flush(stdout)
                raw_mask
            end
            println("[InferenceClient] HELPNet post-processing (LCC): $(count(raw_mask .> 0)) -> $(count(clean_mask .> 0)) voxels"); flush(stdout)
            set_last_ai_error!("")
            return clean_mask
        else
            err_msg = string(get(resp, "message", "Unknown error from HELPNet worker"))
            set_last_ai_error!("HELPNet error: $err_msg")
            println("[InferenceClient ERROR] Python Worker Error: $err_msg"); flush(stdout)
            return nothing
        end
    catch e
        set_last_ai_error!("Failed to communicate with Python Worker at $host:$port: $e")
        println("[InferenceClient ERROR] Failed to communicate with Python Worker at $host:$port: $e"); flush(stdout)
        return nothing
    end
end

function run_helpnet_inference(ct_vol::Array{Float32, 3}, pet_vol::Array{Float32, 3}, cx::Int, cy::Int, cz::Int; port=get_ai_port())
    return run_helpnet_inference(ct_vol, pet_vol, nothing, cx, cy, cz; port=port)
end

"""
    run_nninteractive(ct_vol, pet_vol, scribble_coords, cx, cy, cz; port=get_ai_port(), autozoom=true)

Run nnInteractive segmentation. `scribble_coords` is a `Vector{Vector{Int}}` of
0-indexed [x,y,z] coordinates — avoids the expensive `findall` + full-volume allocation.
Supports inline base64 mask transfer from Docker (skips NIfTI file I/O).
"""
function run_nninteractive(ct_vol::Array{Float32, 3}, pet_vol::Array{Float32, 3},
                          scribble_coords::Vector{Vector{Int}},
                          cx::Int, cy::Int, cz::Int;
                          port=get_ai_port(), autozoom=true)
    if !is_ai_enabled()
        println("[InferenceClient] nnInteractive inference skipped: AI models are disabled."); flush(stdout)
        return nothing
    end

    out_dir = get_inference_dir()
    mkpath(out_dir)
    
    if isempty(scribble_coords)
        error("No scribble coordinates provided for NNInteractive. No fallbacks allowed.")
    end
    
    ct_hash = hash(ct_vol)
    ct_path = joinpath(out_dir, "nn_ct_$(ct_hash).nii.gz")
    
    if !isfile(ct_path)
        dummy_sp = (1.0, 1.0, 1.0); dummy_or = (0.0, 0.0, 0.0)
        dummy_dir = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        im_ct = MedImage(voxel_data=ct_vol, spacing=dummy_sp, origin=dummy_or, direction=dummy_dir, image_type=MedImages.MedImage_data_struct.MRI_type, image_subtype=MedImages.MedImage_data_struct.CT_subtype, patient_id="dummy")
        MedImages.create_nii_from_medimage(im_ct, ct_path)
    end
    
    req = Dict(
        "command" => "nninteractive",
        "ct_path" => "/tmp/medeye3d_inference/$(basename(ct_path))",
        "scribble_coords" => scribble_coords,
        "out_dir" => "/tmp/medeye3d_inference",
        "autozoom" => autozoom,
        "inline_result" => true  # Request inline base64 mask transfer
    )
    
    host = get_ai_host()
    conn = nothing
    for attempt in 1:3
        try
            conn = connect(host, port)
            break
        catch e
            if attempt < 3
                sleep(1.0)
            end
        end
    end
    if conn === nothing
        last_err = get_last_ai_error()
        detail = isempty(last_err) ? "Is Docker container 'medeye3d-ai' running? Check Docker Desktop or run start_docker_worker.ps1." : last_err
        set_last_ai_error!("Connection refused at $host:$port ($detail)")
        println("[InferenceClient ERROR] Failed to connect to Python Worker at $host:$port: connection refused. $detail"); flush(stdout)
        return nothing
    end

    try
        write(conn, JSON.json(req))
        resp_str = read(conn, String)
        close(conn)
        
        resp = JSON.parse(resp_str)
        if resp["status"] == "success"
            set_last_ai_error!("")
            # Prefer inline base64 transfer (no file I/O)
            if haskey(resp, "mask_b64")
                raw = base64decode(resp["mask_b64"])
                shape = Tuple(resp["mask_shape"])
                bbox = resp["bbox"]  # [[x1,x2], [y1,y2], [z1,z2]] in ZYX
                sub_mask = reshape(reinterpret(UInt8, raw), Tuple(resp["sub_shape"]))
                # Insert sub-mask into full-size output at bbox position
                full_mask = zeros(UInt8, shape)
                z1, z2 = bbox[1][1]+1, bbox[1][2]
                y1, y2 = bbox[2][1]+1, bbox[2][2]
                x1, x2 = bbox[3][1]+1, bbox[3][2]
                full_mask[x1:x2, y1:y2, z1:z2] .= sub_mask
                return full_mask
            else
                # Fallback: read from NIfTI file
                pred_file = basename(resp["prediction_path"])
                local_pred_path = joinpath(out_dir, pred_file)
                pred_im = MedImages.load_image(local_pred_path, "unknown")
                return Array{UInt8}(pred_im.voxel_data)
            end
        else
            err_msg = string(get(resp, "message", "Unknown error from nnInteractive worker"))
            set_last_ai_error!("nnInteractive error: $err_msg")
            println("[InferenceClient ERROR] Python Worker Error: $err_msg"); flush(stdout)
            return nothing
        end
    catch e
        set_last_ai_error!("Failed to communicate with Python Worker at $host:$port: $e")
        println("[InferenceClient ERROR] Failed to communicate with Python Worker at $host:$port: $e"); flush(stdout)
        return nothing
    end
end

# Legacy API: accept points_vol (3D volume) and extract coords internally
function run_nninteractive(ct_vol::Array{Float32, 3}, pet_vol::Array{Float32, 3}, points_vol::Union{Nothing, Array{Float32, 3}}, cx::Int, cy::Int, cz::Int; port=get_ai_port(), autozoom=true)
    if points_vol === nothing || count(points_vol .> 0) == 0
        error("No user-painted scribbles provided for NNInteractive. No fallbacks allowed.")
    end
    scribble_indices = findall(points_vol .> 0)
    scribble_coords = [[c[1]-1, c[2]-1, c[3]-1] for c in scribble_indices]
    return run_nninteractive(ct_vol, pet_vol, scribble_coords, cx, cy, cz; port=port, autozoom=autozoom)
end

"""
    preload_ct_for_nninteractive(ct_vol; port=get_ai_port())

Preload CT into Docker nnInteractive GPU memory for faster subsequent inference.
Fire-and-forget — runs in a background thread. Errors are logged but don't propagate.
"""
function preload_ct_for_nninteractive(ct_vol::Array{Float32, 3}; port=get_ai_port())
    if !is_ai_enabled()
        return nothing
    end
    Threads.@spawn begin
        try
            host = get_ai_host()
            # Fast check if AI server is listening before doing any heavy I/O
            test_conn = try
                connect(host, port)
            catch
                return nothing
            end
            close(test_conn)

            out_dir = get_inference_dir()
            mkpath(out_dir)
            
            ct_hash = hash(ct_vol)
            ct_path = joinpath(out_dir, "nn_ct_$(ct_hash).nii.gz")
            
            # Save CT to NIfTI if not already on disk
            if !isfile(ct_path)
                dummy_sp = (1.0, 1.0, 1.0)
                dummy_or = (0.0, 0.0, 0.0)
                dummy_dir = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
                im_ct = MedImage(voxel_data=ct_vol, spacing=dummy_sp, origin=dummy_or,
                    direction=dummy_dir,
                    image_type=MedImages.MedImage_data_struct.MRI_type,
                    image_subtype=MedImages.MedImage_data_struct.CT_subtype,
                    patient_id="dummy")
                MedImages.create_nii_from_medimage(im_ct, ct_path)
                println("[InferenceClient] CT saved for preload: $ct_path"); flush(stdout)
            end
            
            req = Dict(
                "command" => "preload_ct",
                "ct_path" => "/tmp/medeye3d_inference/$(basename(ct_path))",
                "out_dir" => "/tmp/medeye3d_inference"
            )
            
            conn = connect(host, port)
            write(conn, JSON.json(req))
            resp_str = read(conn, String)
            close(conn)
            
            resp = JSON.parse(resp_str)
            if resp["status"] == "success"
                println("[InferenceClient] CT preloaded into nnInteractive GPU ✓"); flush(stdout)
            else
                println("[InferenceClient] CT preload warning: $(resp["message"])"); flush(stdout)
            end
        catch e
            # Non-fatal — preload is an optimization, not a requirement
            println("[InferenceClient] CT preload failed (non-fatal): $e"); flush(stdout)
        end
    end
    return nothing
end

"""
    run_bone_subsegmentation_remote(lesion_mask::Array{UInt8, 3}, bone_mask::Array{UInt8, 3}, spacing; port=get_ai_port())

Run PyTorch-based bone subsegmentation remotely on the Docker container's GPU using Base64 inline transfer.
Returns `(surface_mask, marrow_mask)` as `Array{Bool, 3}`.
"""
function run_bone_subsegmentation_remote(lesion_mask::AbstractArray{T, 3}, bone_mask::AbstractArray{U, 3}, spacing; port=get_ai_port()) where {T, U}
    if !is_ai_enabled()
        return nothing, nothing
    end
    shape = size(lesion_mask)
    
    # Pack as UInt8
    lesion_uint8 = convert(Array{UInt8, 3}, lesion_mask .> 0)
    bone_uint8 = convert(Array{UInt8, 3}, bone_mask .> 0)
    
    lesion_b64 = base64encode(lesion_uint8)
    bone_b64 = base64encode(bone_uint8)
    
    req = Dict(
        "command" => "bone_subsegmentation",
        "shape" => collect(shape),
        "spacing" => collect(spacing),
        "lesion_mask_b64" => lesion_b64,
        "bone_mask_b64" => bone_b64
    )
    
    host = get_ai_host()
    try
        conn = connect(host, port)
        write(conn, JSON.json(req))
        resp_str = read(conn, String)
        close(conn)
        
        resp = JSON.parse(resp_str)
        if resp["status"] == "success"
            surf_raw = base64decode(resp["surf_mask_b64"])
            marr_raw = base64decode(resp["marr_mask_b64"])
            
            surf_arr = reshape(reinterpret(UInt8, surf_raw), shape) .> 0
            marr_arr = reshape(reinterpret(UInt8, marr_raw), shape) .> 0
            
            return surf_arr, marr_arr
        else
            println("[InferenceClient ERROR] Bone Subsegmentation failed: $(resp["message"])"); flush(stdout)
            return nothing, nothing
        end
    catch e
        println("[InferenceClient ERROR] Failed to communicate with Python Worker at $host:$port: $e"); flush(stdout)
        return nothing, nothing
    end
end

end # module
