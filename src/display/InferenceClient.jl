module InferenceClient

using Sockets
using JSON
using MedImages
using ..ConnectedComponents

export start_python_worker, run_helpnet_inference, run_nninteractive, insert_patch!

global PYTHON_PROC = nothing

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

function run_nninteractive(ct_vol::Array{Float32, 3}, pet_vol::Array{Float32, 3}, points_vol::Union{Nothing, Array{Float32, 3}}, cx::Int, cy::Int, cz::Int; port=5005)
    # Docker container (medeye3d-ai) is started once at app startup — here we only communicate via TCP
    # NOTE: pet_vol is accepted for API compatibility but NOT used — nnInteractive operates on CT only.

    out_dir = INFERENCE_DIR
    mkpath(out_dir)
    
    if points_vol === nothing || count(points_vol .> 0) == 0
        error("No user-painted scribbles provided for NNInteractive. Paint a scribble before running AI. No fallbacks allowed.")
    end
    
    # Hash CT volume to avoid writing 300MB files to disk on every click
    ct_hash = hash(ct_vol)
    
    ct_path = joinpath(out_dir, "nn_ct_$(ct_hash).nii.gz")
    point_path = joinpath(out_dir, "nn_point_in.nii.gz")
    
    dummy_sp = (1.0, 1.0, 1.0)
    dummy_or = (0.0, 0.0, 0.0)
    dummy_dir = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    
    # Only write CT if it doesn't already exist for this patient
    if !isfile(ct_path)
        im_ct = MedImage(voxel_data=ct_vol, spacing=dummy_sp, origin=dummy_or, direction=dummy_dir, image_type=MedImages.MedImage_data_struct.MRI_type, image_subtype=MedImages.MedImage_data_struct.CT_subtype, patient_id="dummy")
        MedImages.create_nii_from_medimage(im_ct, ct_path)
    end
    
    # Extract non-zero scribble coordinates directly (0-indexed for Python)
    # This avoids expensive 3D NIfTI file generation for a few scribble points
    scribble_indices = findall(points_vol .> 0)
    scribble_coords = [[c[1]-1, c[2]-1, c[3]-1] for c in scribble_indices]
    
    req = Dict(
        "command" => "nninteractive",
        "ct_path" => ct_path,
        "scribble_coords" => scribble_coords,
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
            return Array{UInt8}(pred_im.voxel_data)
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

end # module
