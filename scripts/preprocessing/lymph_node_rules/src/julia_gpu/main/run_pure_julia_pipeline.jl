#!/usr/bin/env julia

using Printf
using JSON
using CUDA
using KernelAbstractions
using Adapt
using HDF5
using NIfTI
using Statistics

include("../preprocessing/MedImagesIO.jl")
include("../preprocessing/Landmarks.jl")
include("../preprocessing/PrimaryHDF5.jl")
include("Evaluator.jl")
include("../preprocessing/coordinator.jl")
include("DagVm.jl")
include("MaxAnatomyBuilder.jl")

using .MedImagesIO
using .Landmarks
using .PrimaryHDF5
using .Evaluator
using .Coordinator
using .DagVm
using .MaxAnatomyBuilder

function run_dag_execution(case_dir::String, h5_path::String)
    println("="^85)
    println("Pure Julia KernelAbstractions DAG Execution (Direct HDF5 Streaming)")
    println("Case: $(basename(case_dir))")
    println("Primary HDF5: $h5_path")
    println("="^85)
    
    gold_nrrd_path = joinpath(case_dir, "All_Lymph_Node_Areas.seg.nrrd")
    json_dir = normpath(joinpath(@__DIR__, "../../../jsons"))
    
    # 1. Compile Supporting Structures and DAG Rules
    println("\n[1/4] Scanning needed input masks from JSON rules...")
    all_rules = DagVm.load_all_rules(json_dir)
    
    base_organs = ["body", "trachea", "aorta", "esophagus", "skull", "mandible", "hyoid", "cricoid_cartilage", "sternum", "manubrium", "aortic_arch", "femur", "scapula", "pectoralis_major", "pectoralis_minor", "subclavian_artery", "subclavian_vein", "clavicula", "humerus", "pancreas", "rectum", "sacrum", "colon", "small_bowel", "internal_jugular_vein", "common_carotid_artery"]
    needed_masks = Set{String}()
    for org in base_organs
        push!(needed_masks, org)
        push!(needed_masks, "$(org)_left")
        push!(needed_masks, "$(org)_right")
        push!(needed_masks, "left_$org")
        push!(needed_masks, "right_$org")
    end
    for i in 1:12
        push!(needed_masks, "rib_$i")
        push!(needed_masks, "rib_left_$i")
        push!(needed_masks, "rib_right_$i")
        push!(needed_masks, "rib_$(i)_left")
        push!(needed_masks, "rib_$(i)_right")
    end
    
    for (rname, params) in all_rules
        for key in ["input", "input_image", "inputs", "base_landmark", "mask_name", "input_landmarks", "target", "obstacle", "exclude", "ancillary_landmarks", "depends_on", "superior_ref", "anterior_ref", "dilation_ref"]
            if haskey(params, key)
                val = params[key]
                if val isa Vector
                    for v in val
                        if v isa String
                            push!(needed_masks, v)
                            for s in ["left", "right", "Left", "Right"]
                                push!(needed_masks, v * "_" * s)
                                push!(needed_masks, replace(v, "\$side" => s))
                            end
                        end
                    end
                elseif val isa String
                    push!(needed_masks, val)
                    for s in ["left", "right", "Left", "Right"]
                        push!(needed_masks, val * "_" * s)
                        push!(needed_masks, replace(val, "\$side" => s))
                    end
                end
            end
        end
    end
    
    # 2. Fast Streaming from Primary HDF5
    println("\n[2/4] Streaming CT volume, spatial metadata and organ masks from Primary HDF5...")
    t_start = time()
    dims, spacing, origin, direction = PrimaryHDF5.load_primary_metadata(h5_path)
    println("  -> CT Dimensions: $dims, Spacing: $spacing, Origin: $origin")
    
    masks = PrimaryHDF5.load_primary_masks(h5_path; required_names=nothing)
    computed_landmarks = PrimaryHDF5.load_primary_landmarks(h5_path)
    t_load = round(time() - t_start, digits=3)
    println("  -> Loaded $(length(masks)) total organ/detail masks and $(length(computed_landmarks)) landmarks in $(t_load)s.")
    
    # 3. Pure Julia DAG VM Execution on GPU
    backend = CUDA.functional() ? CUDABackend() : CPU()
    gen_masks = DagVm.run_dag_vm_pipeline(
        json_dir,
        masks,
        computed_landmarks,
        dims,
        spacing,
        origin,
        direction;
        backend=backend
    )
    
    # 4. Save generated masks to disk as NIfTI files
    println("\n[3/4] Saving generated masks to lymph_node_outputs/...")
    output_base = joinpath(case_dir, "lymph_node_outputs")
    
    function get_output_subdir(mask_name::String)
        mn = lowercase(mask_name)
        
        # Check rule definition
        base_l = replace(mask_name, r"_Left$"i => "")
        base_r = replace(mask_name, r"_Right$"i => "")
        is_known_rule = haskey(all_rules, mask_name) || haskey(all_rules, base_l) || haskey(all_rules, base_r)
        
        # If it's not a rule (e.g. primary mask like parotid_gland), skip it
        if !is_known_rule
            return nothing
        end
        
        rule_def = get(all_rules, mask_name, get(all_rules, base_l, get(all_rules, base_r, Dict())))
        is_helper = get(rule_def, "is_helper", false) || occursin("helper", mn) || occursin("_base", mn) || occursin("_exclusion", mn) || occursin("_bridge", mn) || mask_name == "Thoracic_Chest_Wall" || mask_name == "Inguinal_Ligament" || startswith(mn, "inguinal_ligament")
        
        if is_helper
            return "helpers"
        elseif startswith(mn, "neck_") || startswith(mn, "parotid")
            return "head_neck"
        elseif startswith(mn, "thoracic_") || startswith(mn, "axillary_")
            return "thorax"
        elseif startswith(mn, "abdominal_") || startswith(mn, "inguinal_") || startswith(mn, "deep_inguinal") || startswith(mn, "superficial_inguinal")
            return "abdomen_pelvis"
        else
            return "other"
        end
    end
    
    # Obtain reference NIfTI header to ensure 100% exact spatial orientation with CT
    ct_path = joinpath(case_dir, "Fixed_CT_Volume.nii.gz")
    ref_hdr = if isfile(ct_path)
        hdr = deepcopy(NIfTI.niread(ct_path).header)
        hdr.datatype = Int16(2)  # UInt8
        hdr.bitpix = Int16(8)
        hdr
    else
        # Rigorous LPS -> RAS conversion
        hdr = NIfTI.NIfTI1Header()
        hdr.pixdim = NTuple{8, Float32}([1.0, Float32(spacing[1]), Float32(spacing[2]), Float32(spacing[3]), 0.0, 0.0, 0.0, 0.0])
        hdr.srow_x = NTuple{4, Float32}([-direction[1]*spacing[1], -direction[2]*spacing[2], -direction[3]*spacing[3], -origin[1]])
        hdr.srow_y = NTuple{4, Float32}([-direction[4]*spacing[1], -direction[5]*spacing[2], -direction[6]*spacing[3], -origin[2]])
        hdr.srow_z = NTuple{4, Float32}([ direction[7]*spacing[1],  direction[8]*spacing[2],  direction[9]*spacing[3],  origin[3]])
        hdr.qoffset_x = Float32(-origin[1])
        hdr.qoffset_y = Float32(-origin[2])
        hdr.qoffset_z = Float32(origin[3])
        hdr.quatern_d = 1.0f0
        hdr.sform_code = Int16(1)
        hdr.qform_code = Int16(1)
        hdr.datatype = Int16(2)
        hdr.bitpix = Int16(8)
        hdr
    end

    n_saved = 0
    for (mask_name, mask_arr) in gen_masks
        subdir = get_output_subdir(mask_name)
        subdir === nothing && continue
        
        out_dir = joinpath(output_base, subdir)
        mkpath(out_dir)
        out_path = joinpath(out_dir, mask_name * ".nii.gz")
        
        
            arr_u8 = UInt8.(adapt(Array, mask_arr) .> 0)
            ni = NIfTI.NIVolume(deepcopy(ref_hdr), arr_u8)
            NIfTI.niwrite(out_path, ni)
            n_saved += 1
    end
    println("  -> Saved $n_saved masks to $output_base")

    
    # 6. Pure Julia Evaluation
    eval_gold_path = isfile(joinpath(case_dir, "Pat44_Combined_All_Lymph_Nodes.seg.nrrd")) ? 
                     joinpath(case_dir, "Pat44_Combined_All_Lymph_Nodes.seg.nrrd") : gold_nrrd_path
    
    println("\n[5/5] Evaluating Dice scores against Gold Standard ($eval_gold_path)...")
    if isfile(eval_gold_path)

    println("\n[5.5/5] Building Max Anatomy in Pure Julia...")
    # Pass 'masks' (the base organs) and 'gen_masks' (the lymph nodes)
    MaxAnatomyBuilder.build_and_save_max_anatomy(masks, gen_masks, ref_hdr, case_dir)
        avg_dice = evaluate_lymph_nodes(gen_masks, eval_gold_path)
        println("Validation Complete. Overall Dice: ", avg_dice)
    else
        @warn "Gold standard NRRD not found at $eval_gold_path"
    end
end

function resolve_case_dir(case_arg::String)
    if isdir(case_arg)
        return case_arg
    end
    base_dir = "/mnt/big/project_ssd/project_ssd/lymph_node_rules/data/processed_cases_restored"
    direct_path = joinpath(base_dir, case_arg)
    if isdir(direct_path)
        return direct_path
    end
    if isdir(base_dir)
        for d in readdir(base_dir)
            if occursin(case_arg, d) && isdir(joinpath(base_dir, d))
                return joinpath(base_dir, d)
            end
        end
    end
    return direct_path
end

function main()
    case_path = "/mnt/big/project_ssd/project_ssd/lymph_node_rules/data/processed_cases_restored/FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat44"
    force_step1 = false
    force_step2 = false
    h5_input = ""
    h5_output = ""
    json_dir = "jsons"
    
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--case" || arg == "-c"
            case_path = ARGS[i+1]
            i += 1
        elseif arg == "--h5"
            h5_input = ARGS[i+1]
            i += 1
        elseif arg == "--out" || arg == "-o"
            h5_output = ARGS[i+1]
            i += 1
        elseif arg == "--jsons"
            json_dir = ARGS[i+1]
            i += 1
        elseif arg == "--force-step1"
            force_step1 = true
        elseif arg == "--force-step2"
            force_step2 = true
        elseif arg == "--refresh"
            force_step1 = true
            force_step2 = true
        elseif !startswith(arg, "-")
            case_path = arg
        end
        i += 1
    end

    if !isempty(h5_input)
        println("="^85)
        println("Direct HDF5 Streaming Execution: $h5_input")
        println("="^85)
        gen_masks = DagVm.run_pipeline(h5_input; json_dir=json_dir, resolve_overlaps=true)
        if isempty(h5_output)
            h5_output = joinpath(dirname(h5_input), "final_results_gpu.h5")
        end
        # Deduplicate masks: if both "Thoracic_Station_Hilar_Interlobar_Left" and
        # "Thoracic_Station_Hilar_Interlobar_left" exist (stale legacy vs computed),
        # keep only the computed one (which has the canonical CamelCase key from JSON rules).
        deduped = Dict{String, Pair{String, Any}}()  # lowercase_key => (original_key, mask)
        for (k, v) in gen_masks
            k_lower = lowercase(k)
            if haskey(deduped, k_lower)
                existing_key = deduped[k_lower].first
                # Prefer the key that matches a JSON rule name (has uppercase chars)
                if any(isuppercase, k) && !any(isuppercase, existing_key)
                    println("  Dedup: replacing stale '$existing_key' with computed '$k'")
                    deduped[k_lower] = k => v
                else
                    println("  Dedup: keeping '$existing_key', discarding stale '$k'")
                end
            else
                deduped[k_lower] = k => v
            end
        end
        println("\\nSaving $(length(deduped)) deduplicated masks to $h5_output (from $(length(gen_masks)) total)...")
        mkpath(dirname(abspath(h5_output)))
        h5open(h5_output, "w") do f
            for (k_lower, (k, v)) in deduped
                    arr = collect(v)
                    if eltype(arr) == Bool; arr = UInt8.(arr)
                    elseif eltype(arr) != UInt8; arr = UInt8.(arr .> 0); end
                    if ndims(arr) == 3
                        f[k, chunk=(64, 64, 32), compress=1] = arr
                    end
            end
        end
    used_rules = Dict{String, Any}()
    all_rules = DagVm.load_all_rules(json_dir)
    function add_rule_and_deps(name)
        if haskey(used_rules, name); return; end
        if !haskey(all_rules, name); return; end
        rule = all_rules[name]
        used_rules[name] = rule
        if haskey(rule, "components")
            for comp in rule["components"]; add_rule_and_deps(comp); end
        end
        if haskey(rule, "depends_on")
            for dep in rule["depends_on"]; add_rule_and_deps(dep); end
        end
        if haskey(rule, "z_plane_restriction")
            z = rule["z_plane_restriction"]
            if haskey(z, "superior"); add_rule_and_deps(z["superior"]); end
            if haskey(z, "inferior"); add_rule_and_deps(z["inferior"]); end
        end
    end
    for k in keys(gen_masks)
        add_rule_and_deps(k)
    end
    json_out_path = joinpath(dirname(h5_output), "used_lymph_node_rules.json")
    open(json_out_path, "w") do io
        JSON.print(io, used_rules, 4)
    end
    println("Done saving HDF5 to $h5_output and used rules to $json_out_path")
        return
    end

    case_dir = resolve_case_dir(case_path)
    Coordinator.coordinate_case_pipeline(case_dir; 
                                         force_step1=force_step1, 
                                         force_step2=force_step2, 
                                         run_dag_fn=run_dag_execution)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
