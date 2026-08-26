module InferenceClient

using Sockets
using JSON
using Base64
using MedImages
using ..ConnectedComponents

export start_python_worker, run_helpnet_inference, run_nninteractive, run_bone_subsegmentation_remote, insert_patch!, preload_ct_for_nninteractive, send_json_request

global PYTHON_PROC = nothing

function send_json_request(req::Dict; port=5005)
    try
        conn = connect("127.0.0.1", port)
        write(conn, JSON.json(req))
        resp_str = read(conn, String)
        close(conn)
        return JSON.parse(resp_str)
    catch e
        return Dict("status" => "error", "message" => string(e))
    end
end

function start_python_worker(worker_script_path::String)
    # Since we are using Docker, we just run the bash script to orchestrate it
    println("[InferenceClient] Ensuring MedEye3d AI Docker Worker is running..."); flush(stdout)
    docker_script = joinpath(@__DIR__, "..", "..", "scripts", "ai", "start_docker_worker.sh")
    
    # We use run(..., wait=true) because the bash script itself runs docker in the background (-d)
    # or immediately exits if it's already running.
    run(pipeline(`bash $docker_script`, stdout="/tmp/medeye3d_docker_worker.log", stderr="/tmp/medeye3d_docker_worker.log"), wait=true)

    # Wait up to 15 seconds for the Python TCP server to initialize models and open port 5005 inside the container
    for i in 1:15
        try
            conn = connect("127.0.0.1", 5005)
            close(conn)
            println("[InferenceClient] Docker Python Worker is ready."); flush(stdout)
            verify_docker_code_sync()
            break
        catch
            sleep(1)
        end
    end
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

function insert_patch!(vol::Array{Float32, 3}, patch::Array{<:Real, 3}, cx::Int, cy::Int, cz::Int; label_val::Float32=1.0f0)
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

const INFERENCE_DIR = joinpath(dirname(dirname(@__DIR__)), "tmp_inference")

function run_helpnet_inference(ct_vol::Array{Float32, 3}, pet_vol::Array{Float32, 3}, points_vol::Union{Nothing, Array{Float32, 3}}, cx::Int, cy::Int, cz::Int; port=5005)
    # Docker container (medeye3d-ai) is started once at app startup — here we only communicate via TCP

    out_dir = INFERENCE_DIR
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
        "ct_path" => ct_path,
        "pet_path" => pet_path,
        "point_path" => point_path,
        "out_dir" => "/tmp/medeye3d_inference"
    )
    
    try
        conn = connect("127.0.0.1", port)
        write(conn, JSON.json(req))
        resp_str = read(conn, String)
        close(conn)
        
        resp = JSON.parse(resp_str)
        if resp["status"] == "success"
            pred_file = basename(resp["prediction_path"])
            local_pred_path = joinpath(INFERENCE_DIR, pred_file)
            pred_im = MedImages.load_image(local_pred_path, "unknown")
            raw_mask = Array{UInt8}(pred_im.voxel_data)
            # Post-process: extract only largest connected component using GPU KernelAbstractions
            clean_mask = ConnectedComponents.extract_largest_connected_component(raw_mask)
            println("[InferenceClient] HELPNet post-processing (LCC): $(count(raw_mask .> 0)) -> $(count(clean_mask .> 0)) voxels"); flush(stdout)
            return clean_mask
        else
            println("[InferenceClient ERROR] Python Worker Error: $(resp["message"])"); flush(stdout)
            return nothing
        end
    catch e
        println("[InferenceClient ERROR] Failed to communicate with Python Worker: $e"); flush(stdout)
        println(sprint(showerror, e, catch_backtrace())); flush(stdout)
        return nothing
    end
end

"""
    run_nninteractive(ct_vol, pet_vol, scribble_coords, cx, cy, cz; port=5005, autozoom=true)

Run nnInteractive segmentation. `scribble_coords` is a `Vector{Vector{Int}}` of
0-indexed [x,y,z] coordinates — avoids the expensive `findall` + full-volume allocation.
Supports inline base64 mask transfer from Docker (skips NIfTI file I/O).
"""
function run_nninteractive(ct_vol::Array{Float32, 3}, pet_vol::Array{Float32, 3},
                          scribble_coords::Vector{Vector{Int}},
                          cx::Int, cy::Int, cz::Int;
                          port=5005, autozoom=true)
    out_dir = INFERENCE_DIR
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
        "ct_path" => ct_path,
        "scribble_coords" => scribble_coords,
        "out_dir" => "/tmp/medeye3d_inference",
        "autozoom" => autozoom,
        "inline_result" => true  # Request inline base64 mask transfer
    )
    
    try
        conn = connect("127.0.0.1", port)
        write(conn, JSON.json(req))
        resp_str = read(conn, String)
        close(conn)
        
        resp = JSON.parse(resp_str)
        if resp["status"] == "success"
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
                local_pred_path = joinpath(INFERENCE_DIR, pred_file)
                pred_im = MedImages.load_image(local_pred_path, "unknown")
                return Array{UInt8}(pred_im.voxel_data)
            end
        else
            println("[InferenceClient ERROR] Python Worker Error: $(resp["message"])"); flush(stdout)
            return nothing
        end
    catch e
        println("[InferenceClient ERROR] Failed to communicate with Python Worker: $e"); flush(stdout)
        println(sprint(showerror, e, catch_backtrace())); flush(stdout)
        return nothing
    end
end

# Legacy API: accept points_vol (3D volume) and extract coords internally
function run_nninteractive(ct_vol::Array{Float32, 3}, pet_vol::Array{Float32, 3}, points_vol::Union{Nothing, Array{Float32, 3}}, cx::Int, cy::Int, cz::Int; port=5005, autozoom=true)
    if points_vol === nothing || count(points_vol .> 0) == 0
        error("No user-painted scribbles provided for NNInteractive. No fallbacks allowed.")
    end
    scribble_indices = findall(points_vol .> 0)
    scribble_coords = [[c[1]-1, c[2]-1, c[3]-1] for c in scribble_indices]
    return run_nninteractive(ct_vol, pet_vol, scribble_coords, cx, cy, cz; port=port, autozoom=autozoom)
end

"""
    preload_ct_for_nninteractive(ct_vol; port=5005)

Preload CT into Docker nnInteractive GPU memory for faster subsequent inference.
Fire-and-forget — runs in a background thread. Errors are logged but don't propagate.
"""
function preload_ct_for_nninteractive(ct_vol::Array{Float32, 3}; port=5005)
    Threads.@spawn begin
        try
            out_dir = INFERENCE_DIR
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
                "ct_path" => ct_path,
                "out_dir" => "/tmp/medeye3d_inference"
            )
            
            conn = connect("127.0.0.1", port)
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
    run_bone_subsegmentation_remote(lesion_mask::Array{UInt8, 3}, bone_mask::Array{UInt8, 3}, spacing; port=5005)

Run PyTorch-based bone subsegmentation remotely on the Docker container's GPU using Base64 inline transfer.
Returns `(surface_mask, marrow_mask)` as `Array{Bool, 3}`.
"""
function run_bone_subsegmentation_remote(lesion_mask::AbstractArray{T, 3}, bone_mask::AbstractArray{U, 3}, spacing; port=5005) where {T, U}
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
    
    try
        conn = connect("127.0.0.1", port)
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
        println("[InferenceClient ERROR] Failed to communicate with Python Worker: $e"); flush(stdout)
        return nothing, nothing
    end
end

end # module
