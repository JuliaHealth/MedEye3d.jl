#!/usr/bin/env julia
# ==============================================================================
# run_ln_worker.jl
# Persistent GPU worker for recomputing lymph node rules via DagVm.run_dag_vm_pipeline
# using the current ruleset in /mnt/big/project_ssd/project_ssd/lymph_node_rules.
# Unpacks primary masks and landmarks directly in RAM from packed_anatomical_tensor.h5
# + Fixed_CT_Volume.nii.gz (eliminating the 180s primary_masks.h5 disk round-trip!).
# ==============================================================================

using Pkg
const LN_ROOT = get(ENV, "LN_RULES_DIR", isdir("/mnt/big/project_ssd/lymph_node_rules") ? "/mnt/big/project_ssd/lymph_node_rules" : "/workspaces/MedEye3d.jl/scripts/preprocessing/lymph_node_rules")
Pkg.activate(joinpath(LN_ROOT, "src/julia_gpu"))

using Printf
using JSON
using CUDA
using KernelAbstractions
using HDF5
using NIfTI
using Statistics

include(joinpath(LN_ROOT, "src/julia_gpu/preprocessing/MedImagesIO.jl"))
include(joinpath(LN_ROOT, "src/julia_gpu/preprocessing/Landmarks.jl"))
include(joinpath(LN_ROOT, "src/julia_gpu/preprocessing/PrimaryHDF5.jl"))
include(joinpath(LN_ROOT, "src/julia_gpu/preprocessing/coordinator.jl"))
include(joinpath(LN_ROOT, "src/julia_gpu/main/Evaluator.jl"))

include(joinpath(LN_ROOT, "src/julia_gpu/main/MaskPacker.jl"))
include(joinpath(LN_ROOT, "src/julia_gpu/main/StaticArena.jl"))
include(joinpath(LN_ROOT, "src/julia_gpu/main/GpuCCL.jl"))

include(joinpath(LN_ROOT, "src/julia_gpu/main/RuleExecutors.jl"))
using .RuleExecutors
include(joinpath(LN_ROOT, "src/julia_gpu/main/kernels_custom_rules.jl"))
using .CustomRules
include(joinpath(LN_ROOT, "src/julia_gpu/main/DagVm.jl"))
using .DagVm

using .MedImagesIO
using .Landmarks
using .PrimaryHDF5
using .Evaluator
using .Coordinator

# Load the canonical list of 325 primary mask names from Pat114
const PRIMARY_MASK_NAMES = let
    p114_h5 = joinpath(LN_ROOT, "data/processed_cases_restored/FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat114/primary_masks.h5")
    names = Set{String}()
    if isfile(p114_h5)
        h5open(p114_h5, "r") do f
            if haskey(f, "masks")
                for k in keys(f["masks"])
                    push!(names, k)
                end
            end
        end
    end
    names
end

function unpack_masks_and_landmarks_in_ram(raw_dir::String, backend)
    ct_path = joinpath(raw_dir, "Fixed_CT_Volume.nii.gz")
    anat_h5_path = joinpath(raw_dir, "packed_anatomical_tensor.h5")
    reg_json_path = joinpath(raw_dir, "packed_anatomical_tensor_registry.json")

    if !isfile(ct_path) || !isfile(anat_h5_path)
        error("Missing Fixed_CT_Volume.nii.gz or packed_anatomical_tensor.h5 in $raw_dir")
    end

    ct_arr, spacing, origin, direction, _ = PrimaryHDF5.safe_read_nii(ct_path)
    dims = size(ct_arr)
    ct_arr = nothing

    packed_data = h5read(anat_h5_path, "packed_tensor")
    reg_masks = Dict{String, Any}()
    if isfile(reg_json_path)
        reg_doc = JSON.parsefile(reg_json_path)
        reg_masks = get(reg_doc, "masks", reg_doc)
    else
        h5open(anat_h5_path, "r") do af
            if haskey(af, "registry")
                raw_s = read(af["registry"])
                s = raw_s isa Vector{UInt8} ? String(raw_s) : string(raw_s)
                reg_doc = JSON.parse(s)
                reg_masks = get(reg_doc, "masks", reg_doc)
            end
        end
    end

    masks = Dict{String, BitArray{3}}()
    landmark_u8_masks = Dict{String, Array{UInt8, 3}}()

    for (m_name, meta) in reg_masks
        if !(meta isa AbstractDict)
            continue
        end
        ch = Int(get(meta, "channel", 0))
        mid = UInt8(get(meta, "id", 0))
        if ch <= 0 || mid == 0 || ch > size(packed_data, 4)
            continue
        end

        is_primary = (!isempty(PRIMARY_MASK_NAMES) && (m_name in PRIMARY_MASK_NAMES)) || (
            !startswith(m_name, "helper_") &&
            !endswith(m_name, "_Base") &&
            !startswith(m_name, "Neck_") &&
            !startswith(m_name, "Thoracic_") &&
            !startswith(m_name, "Axillary_") &&
            !startswith(m_name, "Abdominal_") &&
            !startswith(m_name, "Deep_Inguinal") &&
            !startswith(m_name, "Superficial_Inguinal") &&
            !startswith(m_name, "Inguinal_") &&
            !startswith(m_name, "lev_vert_")
        )
        if !is_primary
            continue
        end

        clean_name = replace(m_name, r"^nvseg_" => "")
        if clean_name == "bladder"
            clean_name = "urinary_bladder"
        end
        if haskey(masks, clean_name)
            continue
        end

        bit_m = BitArray(view(packed_data, :, :, :, ch) .== mid)
        if count(bit_m) < 10
            continue
        end
        masks[clean_name] = bit_m

        # Prefix to postfix side normalization: "left_femur" -> "femur_left"
        m_side = match(r"^(left|right)_(.+)$", clean_name)
        if m_side !== nothing
            side_str = m_side.captures[1]
            base_str = m_side.captures[2]
            postfix_name = "$(base_str)_$(side_str)"
            masks[postfix_name] = bit_m
            
            # Special rib formats: "left_rib_1" -> "rib_1_left", "rib_left_1"
            if startswith(base_str, "rib_")
                rib_num = replace(base_str, "rib_" => "")
                masks["rib_$(rib_num)_$(side_str)"] = bit_m
                masks["rib_$(side_str)_$(rib_num)"] = bit_m
            end
        end

        # Postfix to prefix side normalization: "femur_left" -> "left_femur"
        m_post = match(r"^(.+)_(left|right)$", clean_name)
        if m_post !== nothing
            base_str = m_post.captures[1]
            side_str = m_post.captures[2]
            prefix_name = "$(side_str)_$(base_str)"
            masks[prefix_name] = bit_m
        end

        # Vertebrae uppercase normalization: e.g. "vertebrae_l5" -> "vertebrae_L5", "vertebrae_s1" -> "vertebrae_S1"
        m_vert = match(r"^(vertebrae|lev_vert)_([cslth])(\d+)$", clean_name)
        if m_vert !== nothing
            v_pfx = m_vert.captures[1]
            v_letter = uppercase(m_vert.captures[2])
            v_num = m_vert.captures[3]
            masks["$(v_pfx)_$(v_letter)$(v_num)"] = bit_m
        end
    end

    # Overlay fresh TotalSegmentator v2 segmentations if present locally (eliminating truncated XNAT vessels/organs)
    ts_path = joinpath(raw_dir, "totalseg_ct_fast.nii.gz")
    ts_map_path = "/workspaces/MedEye3d.jl/data/totalseg_class_map.json"
    if isfile(ts_path) && isfile(ts_map_path)
        println("[LN_WORKER] Overlaying fresh TotalSegmentator v2 base anatomy from $(basename(ts_path))...")
        ts_ni = NIfTI.niread(ts_path)
        ts_arr = ts_ni.raw
        ts_map = JSON.parsefile(ts_map_path)
        for (id_str, class_name) in ts_map
            cls_id = parse(Int, id_str)
            bit_m = BitArray(ts_arr .== cls_id)
            if count(bit_m) >= 10
                masks[class_name] = bit_m
                m_side = match(r"^(left|right)_(.+)$", class_name)
                if m_side !== nothing
                    side_str = m_side.captures[1]
                    base_str = m_side.captures[2]
                    masks["$(base_str)_$(side_str)"] = bit_m
                end
                m_post = match(r"^(.+)_(left|right)$", class_name)
                if m_post !== nothing
                    base_str = m_post.captures[1]
                    side_str = m_post.captures[2]
                    masks["$(side_str)_$(base_str)"] = bit_m
                end
                m_vert = match(r"^(vertebrae|lev_vert)_([cslth])(\d+)$", class_name)
                if m_vert !== nothing
                    v_pfx = m_vert.captures[1]
                    v_letter = uppercase(m_vert.captures[2])
                    v_num = m_vert.captures[3]
                    masks["$(v_pfx)_$(v_letter)$(v_num)"] = bit_m
                end
            end
        end
        ts_arr = nothing
        ts_ni = nothing
    end

    packed_data = nothing
    GC.gc(false)

    # Pass 2: Bilateral / Union masks
    # Femur:
    f_l = get(masks, "femur_left", get(masks, "left_femur", nothing))
    f_r = get(masks, "femur_right", get(masks, "right_femur", nothing))
    if f_l !== nothing && f_r !== nothing
        masks["femur"] = f_l .| f_r
    elseif f_l !== nothing
        masks["femur"] = copy(f_l)
    elseif f_r !== nothing
        masks["femur"] = copy(f_r)
    end

    # Hip / Pelvis:
    h_l = get(masks, "hip_left", get(masks, "left_hip", nothing))
    h_r = get(masks, "hip_right", get(masks, "right_hip", nothing))
    if h_l !== nothing && h_r !== nothing
        masks["hip"] = h_l .| h_r
    elseif h_l !== nothing
        masks["hip"] = copy(h_l)
    elseif h_r !== nothing
        masks["hip"] = copy(h_r)
    end
    if haskey(masks, "hip")
        masks["pelvis"] = haskey(masks, "sacrum") ? (masks["hip"] .| masks["sacrum"]) : copy(masks["hip"])
    end

    # Iliac vessels unions:
    for vtype in ("artery", "vena")
        vl = get(masks, "iliac_$(vtype)_left", get(masks, "left_iliac_$(vtype)", nothing))
        vr = get(masks, "iliac_$(vtype)_right", get(masks, "right_iliac_$(vtype)", nothing))
        if vl !== nothing && vr !== nothing
            masks["iliac_$(vtype)"] = vl .| vr
        elseif vl !== nothing
            masks["iliac_$(vtype)"] = copy(vl)
        elseif vr !== nothing
            masks["iliac_$(vtype)"] = copy(vr)
        end
    end

    # Iliopsoas:
    ip_l = get(masks, "iliopsoas_left", get(masks, "left_iliopsoas", nothing))
    ip_r = get(masks, "iliopsoas_right", get(masks, "right_iliopsoas", nothing))
    if ip_l !== nothing && ip_r !== nothing
        masks["iliopsoas"] = ip_l .| ip_r
    end

    # Kidneys:
    k_l = get(masks, "kidney_left", get(masks, "left_kidney", nothing))
    k_r = get(masks, "kidney_right", get(masks, "right_kidney", nothing))
    if k_l !== nothing && k_r !== nothing
        masks["kidneys"] = k_l .| k_r
    end

    # Fused vertebrae and vertebrae_body:
    vert_keys = [k for k in keys(masks) if startswith(k, "vertebrae_") && !endswith(k, "_fused") && !endswith(k, "_body")]
    if !isempty(vert_keys)
        v_union = falses(dims)
        for vk in vert_keys
            v_union .|= masks[vk]
        end
        masks["vertebrae"] = BitArray(v_union)
        masks["vertebrae_body"] = BitArray(v_union)
    end

    # Helper sacrum fused: sacrum + S1
    if haskey(masks, "sacrum")
        s1 = get(masks, "vertebrae_S1", get(masks, "vertebrae_s1", nothing))
        masks["helper_sacrum_fused"] = (s1 !== nothing) ? (masks["sacrum"] .| s1) : copy(masks["sacrum"])
    end

    # Pass 3: Iliac vessel splitting into Common and External (L5 bottom & Aorta bottom)
    l5_m = get(masks, "vertebrae_L5", get(masks, "vertebrae_l5", nothing))
    if l5_m !== nothing
        l5_slices = findall([any(view(l5_m, :, :, k)) for k in 1:dims[3]])
        if !isempty(l5_slices)
            l5_bottom = minimum(l5_slices)
            aorta_m = get(masks, "aorta", nothing)
            aorta_cap = dims[3]
            if aorta_m !== nothing
                ao_slices = findall([any(view(aorta_m, :, :, k)) for k in 1:dims[3]])
                if !isempty(ao_slices)
                    aorta_cap = minimum(ao_slices)
                end
            end

            for vtype in ("artery", "vena"), side in ("left", "right")
                whole_k = "iliac_$(vtype)_$(side)"
                if haskey(masks, whole_k)
                    whole_m = masks[whole_k]
                    common_m = falses(dims)
                    external_m = falses(dims)
                    for k in 1:dims[3]
                        if k < l5_bottom
                            view(external_m, :, :, k) .= view(whole_m, :, :, k)
                        elseif k <= aorta_cap
                            view(common_m, :, :, k) .= view(whole_m, :, :, k)
                        end
                    end
                    masks["iliac_$(vtype)_common_$(side)"] = BitArray(common_m)
                    masks["iliac_$(vtype)_external_$(side)"] = BitArray(external_m)
                    masks["ts_iliac_$(vtype)_$(side)"] = whole_m
                    masks["nv_iliac_$(vtype)_$(side)"] = whole_m
                end
            end

            # Bilateral common/external iliac unions
            for vtype in ("artery", "vena")
                c_l = get(masks, "iliac_$(vtype)_common_left", nothing)
                c_r = get(masks, "iliac_$(vtype)_common_right", nothing)
                if c_l !== nothing && c_r !== nothing
                    masks["iliac_$(vtype)_common"] = c_l .| c_r
                end
                e_l = get(masks, "iliac_$(vtype)_external_left", nothing)
                e_r = get(masks, "iliac_$(vtype)_external_right", nothing)
                if e_l !== nothing && e_r !== nothing
                    masks["iliac_$(vtype)_external"] = e_l .| e_r
                end
            end
        end
    end

    # Pass 4: Landmark masks
    for (k, bit_m) in masks
        if any(startswith(k, p) for p in ("trachea", "aorta", "iliac_artery", "mandible", "hyoid", "scapula", "rib_"))
            landmark_u8_masks[k] = UInt8.(bit_m)
        end
    end

    # Compute anatomical landmarks in RAM (matching PrimaryHDF5 + DagVm.run_pipeline)
    landmarks_dict = Dict{String, Any}()
    dir_22 = direction[9]
    if haskey(landmark_u8_masks, "trachea")
        car_z, _ = Landmarks.compute_carina_z(landmark_u8_masks["trachea"], spacing)
        if car_z <= 0
            trach_bounds = Landmarks.compute_mask_bounds(landmark_u8_masks["trachea"])
            car_z = trach_bounds[5]
        end
        if car_z > 0
            car_mm = origin[3] + (Float64(car_z) - 1.0) * spacing[3] * dir_22
            landmarks_dict["carina_computed"] = car_mm
            landmarks_dict["carina_computed_vox"] = car_z
        end
    end

    if haskey(landmark_u8_masks, "aorta")
        _, _, _, top_ao_idx, bottom_ao_idx = Landmarks.split_aorta_into_parts(landmark_u8_masks["aorta"])
        if top_ao_idx > 0 && bottom_ao_idx > 0
            arch_top_mm = origin[3] + (Float64(top_ao_idx) - 1.0) * spacing[3] * dir_22
            arch_bot_mm = origin[3] + (Float64(bottom_ao_idx) - 1.0) * spacing[3] * dir_22
            landmarks_dict["aortic_arch_computed"] = arch_top_mm
            landmarks_dict["aortic_arch_computed_vox"] = top_ao_idx
            landmarks_dict["aortic_arch_top"] = arch_top_mm
            landmarks_dict["aortic_arch_top_vox"] = top_ao_idx
            landmarks_dict["aortic_arch_bottom"] = arch_bot_mm
            landmarks_dict["aortic_arch_bottom_vox"] = bottom_ao_idx
        else
            arch_z, _ = Landmarks.compute_aortic_arch_z(landmark_u8_masks["aorta"], spacing)
            arch_mm = origin[3] + (Float64(arch_z) - 1.0) * spacing[3] * dir_22
            landmarks_dict["aortic_arch_computed"] = arch_mm
            landmarks_dict["aortic_arch_computed_vox"] = arch_z
        end
    end

    for side in ("left", "right")
        art_k = "iliac_artery_internal_$side"
        if !haskey(landmark_u8_masks, art_k) && haskey(landmark_u8_masks, "iliac_artery_$side")
            art_k = "iliac_artery_$side"
        end
        if haskey(landmark_u8_masks, art_k)
            p1, p2 = Landmarks.compute_internal_iliac_points(landmark_u8_masks[art_k], spacing, origin, direction)
            if p1 !== nothing
                landmarks_dict["internal_iliac_p1_$side"] = p1
                landmarks_dict["internal_iliac_p2_$side"] = p2
                landmarks_dict["internal_iliac_p1"] = p1
                landmarks_dict["internal_iliac_p2"] = p2
            end
        end
    end

    is_lps_y = direction[5] > 0.0
    for side in ("left", "right")
        m_arr = get(landmark_u8_masks, "mandible", nothing)
        h_arr = get(landmark_u8_masks, "hyoid", nothing)
        if m_arr !== nothing || h_arr !== nothing
            dt, db, dpl = Landmarks.compute_digastric_landmarks(m_arr, h_arr, spacing, origin, side; is_lps=is_lps_y)
            if dt !== nothing; landmarks_dict["dig_top_$side"] = dt; end
            if db !== nothing; landmarks_dict["dig_bot_$side"] = db; end
            if dpl !== nothing; landmarks_dict["digastric_plane_$side"] = JSON.parse(JSON.json((dt, dpl))); end
        end
    end
    if haskey(landmarks_dict, "dig_top_left") && haskey(landmarks_dict, "dig_top_right")
        zl = landmarks_dict["dig_top_left"][3]
        zr = landmarks_dict["dig_top_right"][3]
        landmarks_dict["dig_top"] = zl > zr ? landmarks_dict["dig_top_left"] : landmarks_dict["dig_top_right"]
    end

    for side in ("left", "right")
        scap = get(landmark_u8_masks, "scapula_$side", nothing)
        rib3 = get(landmark_u8_masks, "rib_$(side)_3", get(landmark_u8_masks, "rib_3_$side", get(landmark_u8_masks, "rib_3", nothing)))
        rib5 = get(landmark_u8_masks, "rib_$(side)_5", get(landmark_u8_masks, "rib_5_$side", get(landmark_u8_masks, "rib_5", nothing)))
        if scap !== nothing && rib3 !== nothing && rib5 !== nothing
            geom = Landmarks.compute_axillary_geometry(scap, rib3, rib5, spacing, origin, side)
            landmarks_dict["axillary_geometry_$side"] = JSON.parse(JSON.json(geom))
        end
    end

    # Coracoid process Z (most anterior voxel of scapula, from DagVm.run_pipeline)
    is_ras_y = direction[5] < 0.0
    for side in ["left", "right"]
        scap_m = get(masks, "scapula_$side", nothing)
        if scap_m === nothing; scap_m = get(masks, "scapula", nothing); end
        if scap_m !== nothing
            scap_m_gpu = backend isa CPU ? scap_m : KernelAbstractions.adapt(backend, scap_m)
            min_y_cpu, max_y_cpu = RuleExecutors.get_slice_y_bounds_gpu(backend, scap_m_gpu, dims)
            best_y = is_ras_y ? 0 : (dims[2] + 1)
            best_k = 0
            for k in 1:dims[3]
                if is_ras_y
                    if max_y_cpu[k] > best_y
                        best_y = max_y_cpu[k]
                        best_k = k
                    end
                else
                    if max_y_cpu[k] > 0 && min_y_cpu[k] < best_y
                        best_y = min_y_cpu[k]
                        best_k = k
                    end
                end
            end
            if best_k > 0
                z_mm = origin[3] + (best_k - 1) * spacing[3]
                landmarks_dict["coracoid_process_$side"] = [0.0, 0.0, z_mm]
            end
        end
    end
    if haskey(landmarks_dict, "coracoid_process_left") && haskey(landmarks_dict, "coracoid_process_right")
        z_l = landmarks_dict["coracoid_process_left"][3]
        z_r = landmarks_dict["coracoid_process_right"][3]
        landmarks_dict["coracoid_process"] = [0.0, 0.0, min(z_l, z_r)]
    elseif haskey(landmarks_dict, "coracoid_process_left")
        landmarks_dict["coracoid_process"] = landmarks_dict["coracoid_process_left"]
    elseif haskey(landmarks_dict, "coracoid_process_right")
        landmarks_dict["coracoid_process"] = landmarks_dict["coracoid_process_right"]
    end

    landmark_u8_masks = nothing
    GC.gc(false)
    println("[LN_WORKER] Unpacked $(length(masks)) primary masks and $(length(landmarks_dict)) landmarks in RAM in $raw_dir")
    flush(stdout)
    return masks, landmarks_dict, dims, spacing, origin, direction
end

function process_single_case(primary_h5_path::String, out_h5_path::String)
    t0 = time()
    raw_dir = isdir(primary_h5_path) ? primary_h5_path : dirname(primary_h5_path)
    backend = CUDA.functional() ? CUDABackend() : CPU()
    json_dir = joinpath(LN_ROOT, "jsons")

    masks, computed_landmarks, dims, spacing, origin, direction = unpack_masks_and_landmarks_in_ram(raw_dir, backend)
    println("[LN_WORKER] Running DagVm.run_dag_vm_pipeline on: $raw_dir")
    flush(stdout)
    gen_masks, timing_dict, packed_tensor = DagVm.run_dag_vm_pipeline(
        json_dir, masks, computed_landmarks, dims, spacing, origin, direction;
        backend=backend
    )

    tmp_out = out_h5_path * ".tmp"
    println("[LN_WORKER] Saving clinical lymph node stations to: $out_h5_path")
    flush(stdout)
    h5open(tmp_out, "w"; libver_bounds=(v"1.8", v"1.10")) do f
        for (k, v) in gen_masks
            if !startswith(k, "Neck_") && !startswith(k, "Thoracic_") && !startswith(k, "Axillary_") &&
               !startswith(k, "Abdominal_") && !startswith(k, "Deep_Inguinal") && !startswith(k, "Superficial_Inguinal")
                continue
            end
            try
                arr = collect(v)
                if eltype(arr) == Bool
                    arr = UInt8.(arr)
                elseif eltype(arr) != UInt8
                    arr = UInt8.(arr .> 0)
                end
                if ndims(arr) == 3
                    chk = (min(32, size(arr, 1)), min(64, size(arr, 2)), min(64, size(arr, 3)))
                    f[k, chunk=chk, compress=1] = arr
                end
            catch e
                println("[LN_WORKER] ERROR saving $k: $e")
            end
        end
    end
    mv(tmp_out, out_h5_path; force=true)

    # Free GPU arena memory between cases to keep VRAM clean for TotalSegmentator / PyTorch SWD
    try
        if haskey(StaticArena.VM_ARENA, :level_output) && StaticArena.VM_ARENA[:level_output] !== nothing
            if StaticArena.VM_ARENA[:level_output] isa CuArray
                try CUDA.unsafe_free!(StaticArena.VM_ARENA[:level_output]); catch; end
            end
            StaticArena.VM_ARENA[:level_output] = nothing
        end
        for (k, buf) in StaticArena.MASK_ARENA
            if buf isa CuArray
                try CUDA.unsafe_free!(buf); catch; end
            end
        end
        empty!(StaticArena.MASK_ARENA)
        empty!(StaticArena.MASK_IN_USE)
        if StaticArena.EMPTY_MASK isa CuArray
            try CUDA.unsafe_free!(StaticArena.EMPTY_MASK); catch; end
            StaticArena.EMPTY_MASK = nothing
        end
        for (k, buf) in StaticArena.TEMP_ARENA
            if buf isa CuArray
                try CUDA.unsafe_free!(buf); catch; end
            end
        end
        empty!(StaticArena.TEMP_ARENA)
        empty!(StaticArena.TEMP_IN_USE)
        StaticArena.free_edt_arena!()
        StaticArena.free_ft_arena!()
        for sym in [:labels, :counts, :output]
            if haskey(StaticArena.CCL_ARENA, sym) && StaticArena.CCL_ARENA[sym] isa CuArray
                try CUDA.unsafe_free!(StaticArena.CCL_ARENA[sym]); catch; end
                delete!(StaticArena.CCL_ARENA, sym)
            end
        end
        packed_tensor = nothing
        gen_masks = nothing
        masks = nothing
        GC.gc(true)
        CUDA.reclaim()
    catch e
        println("[LN_WORKER] Warning during VRAM cleanup: $e")
    end

    elapsed = round(time() - t0, digits=2)
    println("[LN_WORKER] Completed $out_h5_path in $(elapsed)s (pure GPU DAG: $(get(timing_dict, "pure_julia_processing_total_seconds", 0.0))s)")
    flush(stdout)
    return timing_dict
end

function run_queue_daemon(queue_dir::String)
    mkpath(queue_dir)
    println("[LN_WORKER] Daemon started. Watching queue directory: $queue_dir on GPU $(CUDA.device())")
    flush(stdout)
    ready_file = joinpath(queue_dir, "worker_ready")
    write(ready_file, "READY\n")

    while true
        files = sort(readdir(queue_dir))
        req_files = filter(f -> endswith(f, ".req.json"), files)
        if isempty(req_files)
            sleep(0.5)
            continue
        end
        for rf in req_files
            req_path = joinpath(queue_dir, rf)
            if !isfile(req_path)
                continue
            end
            job_id = replace(rf, ".req.json" => "")
            resp_path = joinpath(queue_dir, "$(job_id).done.json")
            try
                req = JSON.parsefile(req_path)
                rm(req_path; force=true)
                if get(req, "command", "") == "shutdown"
                    println("[LN_WORKER] Shutdown requested.")
                    write(resp_path, JSON.json(Dict("status" => "shutdown")))
                    return
                end
                primary_h5 = req["primary_h5"]
                out_h5 = req["out_h5"]
                t_dict = process_single_case(primary_h5, out_h5)
                write(resp_path, JSON.json(Dict(
                    "status" => "ok",
                    "out_h5" => out_h5,
                    "timing" => t_dict
                )))
            catch e
                err_msg = sprint(showerror, e, catch_backtrace())
                println("[LN_WORKER] ERROR processing job $job_id: $err_msg")
                flush(stdout)
                write(resp_path, JSON.json(Dict(
                    "status" => "error",
                    "error" => err_msg
                )))
                try
                    GC.gc(true)
                    CUDA.reclaim()
                catch
                end
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) == 1
        run_queue_daemon(ARGS[1])
    elseif length(ARGS) == 2
        process_single_case(ARGS[1], ARGS[2])
    else
        println("Usage: julia run_ln_worker.jl <queue_dir> OR julia run_ln_worker.jl <raw_dir_or_primary_h5> <out_h5>")
        exit(1)
    end
end
