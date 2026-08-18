module InferenceClient

using Sockets
using JSON
using MedImages

export start_python_worker, run_helpnet_inference, run_nninteractive, insert_patch!

global PYTHON_PROC = nothing

function start_python_worker(worker_script_path::String)
    global PYTHON_PROC
    if PYTHON_PROC === nothing || process_exited(PYTHON_PROC)
        @info "Starting Python Worker for Inference..."
        env = copy(ENV)
        env["CUDA_VISIBLE_DEVICES"] = "1"
        PYTHON_PROC = run(pipeline(setenv(`python3 $worker_script_path`, env), stdout=devnull, stderr=devnull), wait=false)
        sleep(3) # Give it time to load PyTorch and models
    else
        @info "Python Worker is already running."
    end
end

function extract_patch(vol::Array{Float32, 3}, cx::Int, cy::Int, cz::Int; pad_val::Float32=0.0f0)
    w, h, d = size(vol)
    patch = fill(pad_val, 64, 64, 64)
    
    src_x1 = max(1, cx - 32)
    src_x2 = min(w, cx + 31)
    dst_x1 = 1 + (src_x1 - (cx - 32))
    dst_x2 = 64 - ((cx + 31) - src_x2)

    src_y1 = max(1, cy - 32)
    src_y2 = min(h, cy + 31)
    dst_y1 = 1 + (src_y1 - (cy - 32))
    dst_y2 = 64 - ((cy + 31) - src_y2)

    src_z1 = max(1, cz - 32)
    src_z2 = min(d, cz + 31)
    dst_z1 = 1 + (src_z1 - (cz - 32))
    dst_z2 = 64 - ((cz + 31) - src_z2)
    
    patch[dst_x1:dst_x2, dst_y1:dst_y2, dst_z1:dst_z2] .= vol[src_x1:src_x2, src_y1:src_y2, src_z1:src_z2]
    return patch
end

function insert_patch!(vol::Array{Float32, 3}, patch::Array{UInt8, 3}, cx::Int, cy::Int, cz::Int; label_val::Float32=1.0f0)
    w, h, d = size(vol)
    
    src_x1 = max(1, cx - 32)
    src_x2 = min(w, cx + 31)
    dst_x1 = 1 + (src_x1 - (cx - 32))
    dst_x2 = 64 - ((cx + 31) - src_x2)

    src_y1 = max(1, cy - 32)
    src_y2 = min(h, cy + 31)
    dst_y1 = 1 + (src_y1 - (cy - 32))
    dst_y2 = 64 - ((cy + 31) - src_y2)

    src_z1 = max(1, cz - 32)
    src_z2 = min(d, cz + 31)
    dst_z1 = 1 + (src_z1 - (cz - 32))
    dst_z2 = 64 - ((cz + 31) - src_z2)
    
    mask_slice = patch[dst_x1:dst_x2, dst_y1:dst_y2, dst_z1:dst_z2]
    
    target_slice = vol[src_x1:src_x2, src_y1:src_y2, src_z1:src_z2]
    for i in eachindex(mask_slice)
        if mask_slice[i] > 0
            target_slice[i] = label_val
        end
    end
    vol[src_x1:src_x2, src_y1:src_y2, src_z1:src_z2] .= target_slice
end

function run_helpnet_inference(ct_vol::Array{Float32, 3}, pet_vol::Array{Float32, 3}, points_vol::Union{Nothing, Array{Float32, 3}}, cx::Int, cy::Int, cz::Int; port=5005)
    start_python_worker(joinpath(@__DIR__, "..", "..", "scripts", "python_worker.py"))

    out_dir = "/tmp/medeye3d_inference"
    mkpath(out_dir)
    
    ct_patch = extract_patch(ct_vol, cx, cy, cz, pad_val=-1000.0f0)
    pet_patch = extract_patch(pet_vol, cx, cy, cz, pad_val=0.0f0)
    
    point_patch = if points_vol !== nothing && any(points_vol .> 0)
        p_patch = extract_patch(points_vol, cx, cy, cz, pad_val=0.0f0)
        if count(p_patch .> 0) == 0
            p_patch[33, 33, 33] = 1.0f0
        end
        p_patch
    else
        p_patch = fill(0.0f0, 64, 64, 64)
        p_patch[33, 33, 33] = 1.0f0
        p_patch
    end
    
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
        "out_dir" => out_dir
    )
    
    try
        conn = connect("127.0.0.1", port)
        write(conn, JSON.json(req))
        resp_str = read(conn, String)
        close(conn)
        
        resp = JSON.parse(resp_str)
        if resp["status"] == "success"
            pred_path = resp["prediction_path"]
            pred_im = MedImages.load_image(pred_path, "unknown")
            return Array{UInt8}(pred_im.voxel_data)
        else
            @warn "Python Worker Error: $(resp["message"])"
            return nothing
        end
    catch e
        @warn "Failed to communicate with Python Worker: $e"
        return nothing
    end
end

function run_nninteractive(ct_vol::Array{Float32, 3}, pet_vol::Array{Float32, 3}, points_vol::Union{Nothing, Array{Float32, 3}}, cx::Int, cy::Int, cz::Int; port=5005)
    start_python_worker(joinpath(@__DIR__, "..", "..", "scripts", "python_worker.py"))

    out_dir = "/tmp/medeye3d_inference"
    mkpath(out_dir)
    
    ct_patch = extract_patch(ct_vol, cx, cy, cz, pad_val=-1000.0f0)
    pet_patch = extract_patch(pet_vol, cx, cy, cz, pad_val=0.0f0)
    
    point_patch = if points_vol !== nothing && any(points_vol .> 0)
        p_patch = extract_patch(points_vol, cx, cy, cz, pad_val=0.0f0)
        if count(p_patch .> 0) == 0
            p_patch[33, 33, 33] = 1.0f0
        end
        p_patch
    else
        p_patch = fill(0.0f0, 64, 64, 64)
        p_patch[33, 33, 33] = 1.0f0
        p_patch
    end
    
    ct_path = joinpath(out_dir, "nn_ct_in.nii.gz")
    pet_path = joinpath(out_dir, "nn_pet_in.nii.gz")
    point_path = joinpath(out_dir, "nn_point_in.nii.gz")
    
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
        "command" => "nninteractive",
        "ct_path" => ct_path,
        "pet_path" => pet_path,
        "point_path" => point_path,
        "out_dir" => out_dir
    )
    
    try
        conn = connect("127.0.0.1", port)
        write(conn, JSON.json(req))
        resp_str = read(conn, String)
        close(conn)
        
        resp = JSON.parse(resp_str)
        if resp["status"] == "success"
            pred_path = resp["prediction_path"]
            pred_im = MedImages.load_image(pred_path, "unknown")
            return Array{UInt8}(pred_im.voxel_data)
        else
            @warn "Python Worker Error: $(resp["message"])"
            return nothing
        end
    catch e
        @warn "Failed to communicate with Python Worker: $e"
        return nothing
    end
end

end # module
