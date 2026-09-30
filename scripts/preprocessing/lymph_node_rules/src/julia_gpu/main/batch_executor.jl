using JSON
using NPZ
using Adapt
using KernelAbstractions: @index, @synchronize

@kernel function split_mask_kernel!(output, input, dims, axis_idx, split_val, keep_min)
    I, J, K = @index(Global, NTuple)
    if I <= dims[1] && J <= dims[2] && K <= dims[3]
        if input[I, J, K] > 0
            v = axis_idx == 3 ? K : (axis_idx == 2 ? J : I)
            if (keep_min && v <= split_val) || (!keep_min && v >= split_val)
                output[I, J, K] = 1
            else
                output[I, J, K] = 0
            end
        else
            output[I, J, K] = 0
        end
    end
end

function get_largest_connected_component!(arr::AbstractArray{UInt8, 3})
    host_arr = Array(arr)
    vox_count = count(host_arr .> 0)
    if vox_count <= 1
        return
    end
    dims = size(host_arr)
    visited = zeros(Bool, dims)
    best_component = Tuple{Int, Int, Int}[]
    curr_component = Tuple{Int, Int, Int}[]
    queue = Tuple{Int, Int, Int}[]
    sizehint!(queue, min(vox_count, 100000))
    sizehint!(curr_component, min(vox_count, 100000))
    
    for z in 1:dims[3], y in 1:dims[2], x in 1:dims[1]
        if host_arr[x, y, z] > 0 && !visited[x, y, z]
            empty!(curr_component)
            empty!(queue)
            
            visited[x, y, z] = true
            push!(queue, (x, y, z))
            push!(curr_component, (x, y, z))
            
            head = 1
            while head <= length(queue)
                cx, cy, cz = queue[head]
                head += 1
                
                # 6-connected neighbors
                for (dx, dy, dz) in ((-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1))
                    nx, ny, nz = cx + dx, cy + dy, cz + dz
                    if nx >= 1 && nx <= dims[1] && ny >= 1 && ny <= dims[2] && nz >= 1 && nz <= dims[3]
                        if host_arr[nx, ny, nz] > 0 && !visited[nx, ny, nz]
                            visited[nx, ny, nz] = true
                            push!(queue, (nx, ny, nz))
                            push!(curr_component, (nx, ny, nz))
                        end
                    end
                end
            end
            
            if length(curr_component) > length(best_component)
                best_component = copy(curr_component)
            end
        end
    end
    
    host_out = zeros(UInt8, dims)
    for (x, y, z) in best_component
        host_out[x, y, z] = 1
    end
    copyto!(arr, host_out)
end

function get_ancillary_view(tm::TensorManager, anc_dict::Dict, k::String)
    ch = get(anc_dict, k, 0)
    return ch > 0 ? get_channel_view(tm, ch) : get_empty_view(tm)
end

function process_batch_operations(tm::TensorManager, batch_json_str::String, computed_landmarks::Dict{String, Any} = Dict{String, Any}())
    batch_req = JSON.parse(batch_json_str)
    
    # 1. Load missing input masks into the tensor from NPY
    for load_req in get(batch_req, "load_masks", [])
        ch = load_req["channel"]
        path = load_req["path"]
        
        if isfile(path)
            mask_host = npzread(path)
            load_mask_to_channel!(tm, ch, mask_host)
        else
            println("WARNING: Cannot load mask from $path (Not found)")
        end
    end
    
    # 2. Execute Morphological/Boolean operations
    for op in get(batch_req, "operations", [])
        rule_type = op["rule"]
        target = op["target"]
        out_ch = op["out_channel"]
        params = op["params"]
        
        println("  -> Executing $rule_type for $target into channel $out_ch")
        
        out_view = get_channel_view(tm, out_ch)
        
        if rule_type == "DistanceExpansion"
            in_ch = op["input_channel"]
            
            sp_x = Float32(get(params, "spacing_x", 1.0))
            sp_y = Float32(get(params, "spacing_y", 1.0))
            sp_z = Float32(get(params, "spacing_z", 1.0))
            dist = Float32(get(params, "distance_mm", 0.0))
            margins = Dict("all" => dist)
            
            in_view = get_channel_view(tm, in_ch)
            out_view = get_channel_view(tm, out_ch)
            
            temp_in = KernelAbstractions.zeros(tm.backend, UInt8, tm.size_x, tm.size_y, tm.size_z)
            copyto!(temp_in, in_view)
            
            c_sum_temp = sum(adapt(Array, temp_in))
            println("    [DEBUG DistanceExpansion] target=$target in_ch=$in_ch sum(temp_in)=$(c_sum_temp)")
            
            for u_ch in get(op, "union_channels", [])
                u_view = get_channel_view(tm, u_ch)
                run_bitwise_or!(tm.backend, temp_in, temp_in, u_view)
            end
            
            run_anisotropic_dilation!(tm.backend, out_view, temp_in, sp_x, sp_y, sp_z, margins)
            
            for s_ch in get(op, "subtract_channels", [])
                s_view = get_channel_view(tm, s_ch)
                run_boolean_op!(tm.backend, out_view, out_view, s_view, "subtract")
            end
            
            for i_ch in get(op, "intersect_channels", [])
                i_view = get_channel_view(tm, i_ch)
                run_boolean_op!(tm.backend, out_view, out_view, i_view, "intersect")
            end
            
        elseif rule_type == "BitwiseOr" || rule_type == "Combine"
            in_ch = get(op, "input_channel", 0)
            out_view = get_channel_view(tm, out_ch)
            
            temp_in = KernelAbstractions.zeros(tm.backend, UInt8, tm.size_x, tm.size_y, tm.size_z)
            if in_ch > 0
                in_view = get_channel_view(tm, in_ch)
                copyto!(temp_in, in_view)
            end
            
            for u_ch in get(op, "union_channels", [])
                if u_ch > 0
                    u_view = get_channel_view(tm, u_ch)
                    run_bitwise_or!(tm.backend, temp_in, temp_in, u_view)
                end
            end
            
            for s_ch in get(op, "subtract_channels", [])
                if s_ch > 0
                    s_view = get_channel_view(tm, s_ch)
                    run_boolean_op!(tm.backend, temp_in, temp_in, s_view, "subtract")
                end
            end
            
            copyto!(out_view, temp_in)
            
        elseif rule_type == "AnisotropicMargin"
            sp_x = Float32(get(params, "spacing_x", 1.0))
            sp_y = Float32(get(params, "spacing_y", 1.0))
            sp_z = Float32(get(params, "spacing_z", 1.0))
            
            side = get(params, "side", "left")
            is_left = lowercase(side) == "left"
            
            out_view = get_channel_view(tm, out_ch)
            fill!(out_view, 0)
            
            bld = get(op, "base_landmark_dict", nothing)
            if bld !== nothing
                for (ch_str, margins_def) in bld
                    ch_int = parse(Int, ch_str)
                    in_view = get_channel_view(tm, ch_int)
                    
                    margins_mm = typeof(margins_def) <: Number ? Dict("all" => margins_def) : margins_def
                    all_default = Float32(get(margins_mm, "all", 0.0))
                    med_default = Float32(get(margins_mm, "medial", all_default))
                    lat_default = Float32(get(margins_mm, "lateral", all_default))
                    
                    ml = Float32(get(margins_mm, "left", is_left ? lat_default : med_default))
                    mr = Float32(get(margins_mm, "right", is_left ? med_default : lat_default))
                    ma = Float32(get(margins_mm, "anterior", all_default))
                    mp = Float32(get(margins_mm, "posterior", all_default))
                    ms = Float32(get(margins_mm, "superior", all_default))
                    mi = Float32(get(margins_mm, "inferior", all_default))
                    
                    margins = Dict("left" => ml, "right" => mr, "anterior" => ma, "posterior" => mp, "superior" => ms, "inferior" => mi)
                    
                    temp_out = adapt(tm.backend, in_view)
                    fill!(temp_out, 0)
                    run_anisotropic_dilation!(tm.backend, temp_out, in_view, sp_x, sp_y, sp_z, margins)
                    run_bitwise_or!(tm.backend, out_view, out_view, temp_out)
                end
            else
                in_ch = op["input_channel"]
                in_view = get_channel_view(tm, in_ch)
                temp_in = adapt(tm.backend, in_view)
                copyto!(temp_in, in_view)
                
                margins_mm = get(params, "margins_mm", Dict())
                if typeof(margins_mm) <: Number
                    margins_mm = Dict("all" => margins_mm)
                end
                all_default = Float32(get(margins_mm, "all", 0.0))
                med_default = Float32(get(margins_mm, "medial", all_default))
                lat_default = Float32(get(margins_mm, "lateral", all_default))
                
                ml = Float32(get(params, "margin_left", get(margins_mm, "left", is_left ? lat_default : med_default)))
                mr = Float32(get(params, "margin_right", get(margins_mm, "right", is_left ? med_default : lat_default)))
                ma = Float32(get(params, "margin_anterior", get(margins_mm, "anterior", all_default)))
                mp = Float32(get(params, "margin_posterior", get(margins_mm, "posterior", all_default)))
                ms = Float32(get(params, "margin_superior", get(margins_mm, "superior", all_default)))
                mi = Float32(get(params, "margin_inferior", get(margins_mm, "inferior", all_default)))
                
                margins = Dict("left" => ml, "right" => mr, "anterior" => ma, "posterior" => mp, "superior" => ms, "inferior" => mi)
                
                for u_ch in get(op, "union_channels", [])
                    u_view = get_channel_view(tm, u_ch)
                    run_bitwise_or!(tm.backend, temp_in, temp_in, u_view)
                end
                
                run_anisotropic_dilation!(tm.backend, out_view, temp_in, sp_x, sp_y, sp_z, margins)
            end
            
            for s_ch in get(op, "subtract_channels", [])
                s_view = get_channel_view(tm, s_ch)
                run_boolean_op!(tm.backend, out_view, out_view, s_view, "subtract")
            end
            
            for i_ch in get(op, "intersect_channels", [])
                i_view = get_channel_view(tm, i_ch)
                run_boolean_op!(tm.backend, out_view, out_view, i_view, "intersect")
            end
            
        elseif rule_type == "SplitMask"
            in_ch = op["input_channel"]
            in_view = get_channel_view(tm, in_ch)
            out_view = get_channel_view(tm, out_ch)
            
            side_str = lowercase(get(params, "side", get(op, "side", "")))
            axis_str = lowercase(get(params, "axis", get(op, "axis", "x")))
            keep_str = lowercase(get(params, "keep", get(op, "keep", "")))
            
            axis_idx = axis_str == "z" ? 3 : (axis_str == "y" ? 2 : 1)
            
            if !isempty(keep_str)
                keep_min = keep_str == "min" || keep_str == "negative" || keep_str == "right"
            elseif !isempty(side_str)
                keep_min = (side_str == "right") # X <= midline is Right, X > midline is Left
            else
                keep_min = true
            end
            
            dims = (tm.size_x, tm.size_y, tm.size_z)
            split_val = dims[axis_idx] ÷ 2
            
            kernel! = CustomRules.split_mask_kernel!(tm.backend)
            kernel!(out_view, in_view, dims, axis_idx, split_val, keep_min, ndrange=dims)
            KernelAbstractions.synchronize(tm.backend)
                                  
        elseif rule_type == "CoronalLateralGrowth"
            in_ch = op["input_channel"]
            in_view = get_channel_view(tm, in_ch)
            
            sp_x = get(params, "spacing_x", 1.0)
            margin_mm = get(params, "margin_mm", 0.0)
            
            # Move to GPU
            in_device = adapt(tm.backend, in_view)
            out_device = adapt(tm.backend, zeros(eltype(in_view), size(in_view)))
            
            run_coronal_lateral_growth!(tm.backend, out_device, in_device, Float32(sp_x), Float32(margin_mm))
            
            out_view = get_channel_view(tm, out_ch)
            copyto!(out_view, out_device)
            
        elseif rule_type == "BooleanOps"
            # target = op["input_channel"] is base
            # union/subtract/intersect channels
            in_ch = op["input_channel"]
            in_view = get_channel_view(tm, in_ch)
            
            out_device = adapt(tm.backend, in_view)
            
            for u_ch in get(op, "union_channels", [])
                u_view = get_channel_view(tm, u_ch)
                u_dev = adapt(tm.backend, u_view)
                run_boolean_op!(tm.backend, out_device, out_device, u_dev, "union")
            end
            
            for s_ch in get(op, "subtract_channels", [])
                s_view = get_channel_view(tm, s_ch)
                s_dev = adapt(tm.backend, s_view)
                run_boolean_op!(tm.backend, out_device, out_device, s_dev, "subtract")
            end
            
            for i_ch in get(op, "intersect_channels", [])
                i_view = get_channel_view(tm, i_ch)
                i_dev = adapt(tm.backend, i_view)
                run_boolean_op!(tm.backend, out_device, out_device, i_dev, "intersect")
            end
            
            out_view = get_channel_view(tm, out_ch)
            copyto!(out_view, out_device)
            
        elseif rule_type == "GenericMorphologyDAG"
            println("  -> Native GenericMorphologyDAG for $target")
            input_chs = get(op, "input_channels", Int[])
            in_ch = get(op, "input_channel", 0)
            if isempty(input_chs) && in_ch > 0
                push!(input_chs, in_ch)
            end
            
            out_view = get_channel_view(tm, out_ch)
            out_device = KernelAbstractions.zeros(tm.backend, UInt8, tm.size_x, tm.size_y, tm.size_z)
            
            # Step 1: Union all input channels
            for ch in input_chs
                if ch > 0
                    v = get_channel_view(tm, ch)
                    run_bitwise_or!(tm.backend, out_device, out_device, v)
                end
            end
            
            # Step 2: Apply morphological operations sequentially
            sp_x = Float32(get(params, "spacing_x", 1.0))
            sp_y = Float32(get(params, "spacing_y", 1.0))
            sp_z = Float32(get(params, "spacing_z", 1.0))
            
            ops = get(op, "operations", [])
            for mop in ops
                opc = get(mop, "opcode", 0)
                if opc == 12 # DILATE
                    radius = Float32(get(mop, "radius", 1.0))
                    temp_dev = KernelAbstractions.zeros(tm.backend, UInt8, tm.size_x, tm.size_y, tm.size_z)
                    run_anisotropic_dilation!(tm.backend, temp_dev, out_device, sp_x, sp_y, sp_z, Dict("all" => radius))
                    copyto!(out_device, temp_dev)
                elseif opc == 13 # ERODE
                    radius = Float32(get(mop, "radius", 1.0))
                    temp_dev = KernelAbstractions.zeros(tm.backend, UInt8, tm.size_x, tm.size_y, tm.size_z)
                    run_anisotropic_erosion!(tm.backend, temp_dev, out_device, sp_x, sp_y, sp_z, Dict("all" => radius))
                    copyto!(out_device, temp_dev)
                elseif opc == 14 # 2D CONVEX HULL
                    temp_dev = KernelAbstractions.zeros(tm.backend, UInt8, tm.size_x, tm.size_y, tm.size_z)
                    run_volumetric_convex_hull_3d!(tm.backend, temp_dev, out_device)
                    copyto!(out_device, temp_dev)
                end
            end
            
            copyto!(out_view, out_device)
            
        elseif rule_type == "CommonIliacCustom"
            println("  -> Native CommonIliacCustom for $target")
            out_view = get_channel_view(tm, out_ch)
            anc_dict = get(op, "ancillary_channels_dict", Dict{String, Any}())
            aorta_ch = get(anc_dict, "aorta", 0)
            # Prefer common iliac vessels, fall back to full iliac
            art_l_ch = get(anc_dict, "iliac_artery_common_left", get(anc_dict, "iliac_artery_left", 0))
            art_r_ch = get(anc_dict, "iliac_artery_common_right", get(anc_dict, "iliac_artery_right", 0))
            ven_l_ch = get(anc_dict, "iliac_vena_common_left", get(anc_dict, "iliac_vena_left", 0))
            ven_r_ch = get(anc_dict, "iliac_vena_common_right", get(anc_dict, "iliac_vena_right", 0))
            aorta_view = aorta_ch > 0 ? get_channel_view(tm, aorta_ch) : get_channel_view(tm, 0)
            art_l_view = art_l_ch > 0 ? get_channel_view(tm, art_l_ch) : get_channel_view(tm, 0)
            art_r_view = art_r_ch > 0 ? get_channel_view(tm, art_r_ch) : get_channel_view(tm, 0)
            ven_l_view = ven_l_ch > 0 ? get_channel_view(tm, ven_l_ch) : get_channel_view(tm, 0)
            ven_r_view = ven_r_ch > 0 ? get_channel_view(tm, ven_r_ch) : get_channel_view(tm, 0)
            CommonIliacCustom(tm, out_view, params, computed_landmarks, aorta_view, art_l_view, art_r_view, ven_l_view, ven_r_view)
            
        elseif rule_type == "ZPlaneIntersection"
            in_ch = op["input_channel"]
            z_min = get(params, "z_min", 1)
            z_max = get(params, "z_max", tm.size_z)
            
            in_view = get_channel_view(tm, in_ch)
            in_device = adapt(tm.backend, in_view)
            out_device = adapt(tm.backend, zeros(eltype(in_view), size(in_view)))
            
            run_z_plane_clip!(tm.backend, out_device, in_device, z_min, z_max)
            
            out_view = get_channel_view(tm, out_ch)
            copyto!(out_view, out_device)

        elseif rule_type == "LateralBridge"
            in_ch = op["input_channel"]
            in_view = get_channel_view(tm, in_ch)
            base_host = adapt(CPU(), in_view)
            
            target_device = adapt(tm.backend, zeros(eltype(in_view), size(in_view)))
            for u_ch in get(op, "union_channels", [])
                u_view = get_channel_view(tm, u_ch)
                u_dev = adapt(tm.backend, u_view)
                run_boolean_op!(tm.backend, target_device, target_device, u_dev, "union")
            end
            
            sp_x = get(params, "spacing_x", 1.0)
            max_mm = get(params, "max_mm", 100.0)
            
            output_device = adapt(tm.backend, zeros(eltype(in_view), size(in_view)))
            run_lateral_bridge!(tm.backend, output_device, base_host, target_device, Float32(sp_x), Float32(max_mm))
            
            out_view = get_channel_view(tm, out_ch)
            copyto!(out_view, output_device)

        elseif rule_type == "PrimaryVector"
            in_ch = op["input_channel"]
            in_view = get_channel_view(tm, in_ch)
            primary_host = adapt(CPU(), in_view)
            
            vector_ch = get(op, "target_channel", 0)
            if vector_ch == 0 && haskey(op, "union_channels") && !isempty(op["union_channels"])
                vector_ch = op["union_channels"][end]
            end
            if vector_ch == 0
                println("    [WARNING] PrimaryVector missing target_channel. Using input.")
                vector_ch = in_ch
            end
            
            vector_view = get_channel_view(tm, vector_ch)
            vector_host = adapt(CPU(), vector_view)
            
            output_device = adapt(tm.backend, zeros(eltype(in_view), size(in_view)))
            
            sp_x = get(params, "spacing_x", 1.0)
            sp_y = get(params, "spacing_y", 1.0)
            sp_z = get(params, "spacing_z", 1.0)
            base_margin = get(params, "base_margin_mm", 0.0)
            exp_margin = get(params, "expansion_mm", 0.0)
            
            run_primary_vector!(tm.backend, output_device, primary_host, vector_host, Float32(sp_x), Float32(sp_y), Float32(sp_z), Float32(base_margin), Float32(exp_margin))
            
            out_view = get_channel_view(tm, out_ch)
            copyto!(out_view, output_device)

        elseif rule_type == "SplitConnectedComponents"
            in_ch = op["input_channel"]
            in_view = get_channel_view(tm, in_ch)
            out_view = get_channel_view(tm, out_ch)
            
            side_val = get(params, "side", "left")
            side = side_val === nothing ? "left" : lowercase(String(side_val))
            is_left = side == "left"
            midline_x = get(params, "midline_x", div(tm.size_x, 2))
            keep_min = !is_left
            
            dims = size(out_view)
            kernel! = split_mask_kernel!(tm.backend)
            kernel!(out_view, in_view, dims, 1, midline_x, keep_min, ndrange=dims)
            KernelAbstractions.synchronize(tm.backend)

        elseif rule_type == "BooleanOps" || rule_type == "BooleanOperation" || rule_type == "BitwiseOr" || rule_type == "BitwiseAnd" || rule_type == "BitwiseSub"
            in_ch = get(op, "input_channel", 0)
            out_view = get_channel_view(tm, out_ch)
            
            if in_ch > 0
                in_view = get_channel_view(tm, in_ch)
                copyto!(out_view, in_view)
            else
                fill!(out_view, 0)
            end
            
            for i_ch in get(op, "intersect_channels", [])
                i_view = get_channel_view(tm, i_ch)
                run_boolean_op!(tm.backend, out_view, out_view, i_view, "intersect")
            end
            
            for u_ch in get(op, "union_channels", [])
                u_view = get_channel_view(tm, u_ch)
                run_boolean_op!(tm.backend, out_view, out_view, u_view, "union")
            end
            
            for s_ch in get(op, "subtract_channels", [])
                s_view = get_channel_view(tm, s_ch)
                run_boolean_op!(tm.backend, out_view, out_view, s_view, "subtract")
            end

        elseif rule_type == "DistanceExpansion"
            in_ch = get(op, "input_channel", 0)
            in_view = get_channel_view(tm, in_ch)
            out_view = get_channel_view(tm, out_ch)
            
            in_host = adapt(CPU(), in_view)
            println("    [DEBUG DistanceExpansion] target=$(get(op, "target", "")) in_ch=$in_ch sum(in_host)=$(sum(in_host))")
            if get(op, "target", "") == "Abdominal_Celiac"
                println("!!! Abdominal_Celiac input non-zero voxels: ", sum(in_host .> 0))
            end
            if sum(in_host) > 0
                sp_x = Float32(get(params, "spacing_x", 1.0))
                sp_y = Float32(get(params, "spacing_y", 1.0))
                sp_z = Float32(get(params, "spacing_z", 1.0))
                dist_mm = Float32(get(params, "distance_mm", 0.0))
                if dist_mm > 0
                    margins = Dict("all" => dist_mm)
                    run_anisotropic_dilation!(tm.backend, out_view, in_view, sp_x, sp_y, sp_z, margins)
                else
                    copyto!(out_view, in_view)
                end
            else
                fill!(out_view, 0)
            end

        elseif rule_type == "FusedPostProcessing"
            in_ch = op["input_channel"]
            in_view = get_channel_view(tm, in_ch)
            out_view = get_channel_view(tm, out_ch)
            
            copyto!(out_view, in_view)
            
            for u_ch in get(op, "union_channels", [])
                u_view = get_channel_view(tm, u_ch)
                run_boolean_op!(tm.backend, out_view, out_view, u_view, "union")
            end
            
            for s_ch in get(op, "subtract_channels", [])
                s_view = get_channel_view(tm, s_ch)
                run_boolean_op!(tm.backend, out_view, out_view, s_view, "subtract")
            end

        # ============================================================
        # Trivial rule aliases: map to BooleanOps
        # ============================================================
        elseif rule_type in ["BitwiseOr", "Union", "Combine"]
            # All are BooleanOps union of multiple channels
            in_ch = op["input_channel"]
            in_view = get_channel_view(tm, in_ch)
            out_view = get_channel_view(tm, out_ch)
            
            copyto!(out_view, in_view)
            
            for u_ch in get(op, "union_channels", [])
                u_view = get_channel_view(tm, u_ch)
                run_boolean_op!(tm.backend, out_view, out_view, u_view, "union")
            end
            
            for s_ch in get(op, "subtract_channels", [])
                s_view = get_channel_view(tm, s_ch)
                run_boolean_op!(tm.backend, out_view, out_view, s_view, "subtract")
            end
            
            for i_ch in get(op, "intersect_channels", [])
                i_view = get_channel_view(tm, i_ch)
                run_boolean_op!(tm.backend, out_view, out_view, i_view, "intersect")
            end

        elseif rule_type in ["Mask", "BooleanOperation", "BitwiseAnd", "BitwiseOr", "Combine"]
            # Mask = intersect base with target channels, then subtract exclusions
            in_ch = op["input_channel"]
            in_view = get_channel_view(tm, in_ch)
            out_view = get_channel_view(tm, out_ch)
            
            copyto!(out_view, in_view)
            
            for i_ch in get(op, "intersect_channels", [])
                i_view = get_channel_view(tm, i_ch)
                run_boolean_op!(tm.backend, out_view, out_view, i_view, "intersect")
            end
            
            for u_ch in get(op, "union_channels", [])
                u_view = get_channel_view(tm, u_ch)
                run_boolean_op!(tm.backend, out_view, out_view, u_view, "union")
            end
            
            for s_ch in get(op, "subtract_channels", [])
                s_view = get_channel_view(tm, s_ch)
                run_boolean_op!(tm.backend, out_view, out_view, s_view, "subtract")
            end

        elseif rule_type == "Base"
            # Simply copy input to output
            in_ch = op["input_channel"]
            in_view = get_channel_view(tm, in_ch)
            out_view = get_channel_view(tm, out_ch)
            copyto!(out_view, in_view)

        elseif rule_type == "SplitConnectedComponents"
            in_ch = op["input_channel"]
            in_view = get_channel_view(tm, in_ch)
            out_view = get_channel_view(tm, out_ch)
            
            side_val = get(params, "side", "left")
            side = side_val === nothing ? "left" : lowercase(String(side_val))
            keep_min = side == "right"
            dims = (tm.size_x, tm.size_y, tm.size_z)
            split_val = dims[1] ÷ 2
            
            kernel! = CustomRules.split_mask_kernel!(tm.backend)
            kernel!(out_view, in_view, dims, 1, split_val, keep_min, ndrange=dims)
            KernelAbstractions.synchronize(tm.backend)

        elseif rule_type == "VolumetricBoundary2D"
            structure_chs = get(op, "structure_channels", Int[])
            medial_chs = get(op, "medial_channels", Int[])
            lateral_chs = get(op, "lateral_channels", Int[])
            posterior_chs = get(op, "posterior_channels", Int[])
            obstacle_chs = get(op, "obstacle_channels", Int[])
            growth_mode = get(op, "growth_mode", "convex_hull")
            per_layer_margin = get(op, "per_layer_margin", false)
            ap_margin_only = get(op, "ap_margin_only", false)
            posterior_offset_mm = Float64(get(op, "posterior_offset_mm", 5.0))
            min_depth_mm = Float64(get(op, "min_depth_mm", 0.0))
            erode_iterations = Int(get(op, "erode_iterations", 0))
            sp_y = Float64(get(params, "spacing_y", 1.0))
            
            s_arrs = [get_channel_view(tm, c) for c in structure_chs]
            m_arrs = [get_channel_view(tm, c) for c in medial_chs]
            l_arrs = [get_channel_view(tm, c) for c in lateral_chs]
            p_arrs = [get_channel_view(tm, c) for c in posterior_chs]
            o_arrs = [get_channel_view(tm, c) for c in obstacle_chs]
            
            run_volumetric_boundary_2d!(get_channel_view(tm, out_ch), s_arrs, m_arrs, l_arrs, p_arrs, o_arrs;
                                        growth_mode=growth_mode, per_layer_margin=per_layer_margin,
                                        ap_margin_only=ap_margin_only, posterior_offset_mm=posterior_offset_mm,
                                        min_depth_mm=min_depth_mm, erode_iterations=erode_iterations, sp_y=sp_y)
            
        elseif rule_type == "VolumetricConvexHull3D"
            target_ch = get(op, "target_channel", 0)
            if target_ch > 0
                run_volumetric_convex_hull_3d!(tm.backend, get_channel_view(tm, out_ch), get_channel_view(tm, target_ch))
            end
            
        elseif rule_type == "ConvexHullBridge" || rule_type == "ConvexHullBridging"
            lm1_ch = get(op, "landmark_1_channel", 0)
            lm2_ch = get(op, "landmark_2_channel", 0)
            if lm1_ch > 0 && lm2_ch > 0
                run_convex_hull_bridge!(get_channel_view(tm, out_ch), get_channel_view(tm, lm1_ch), get_channel_view(tm, lm2_ch))
            end

        elseif rule_type == "GeometricPrimitive"
            primitive = get(op, "primitive", "Cylinder")
            radius_mm = Float32(get(op, "radius_mm", 20.0))
            
            p1_ch = get(op, "p1_channel", 0)
            p2_ch = get(op, "p2_channel", 0)
            
            if p1_ch > 0 && p2_ch > 0 && primitive == "Cylinder"
                sp_x = Float32(get(params, "spacing_x", 1.0))
                sp_y = Float32(get(params, "spacing_y", 1.0))
                sp_z = Float32(get(params, "spacing_z", 1.0))
                dir_00 = Float32(get(params, "direction_00", 1.0))
                
                is_left_side = get(params, "side", "") == "Left"
                
                # Fetch arrays to host to resolve points
                p1_arr = adapt(Array, get_channel_view(tm, p1_ch))
                p2_arr = adapt(Array, get_channel_view(tm, p2_ch))
                
                p1_split = get(op, "p1_split", "")
                p1_dir = get(op, "p1_direction", "")
                p1_pt = resolve_extreme_point(p1_arr, p1_split, p1_dir, is_left_side, sp_x, sp_y, sp_z, dir_00)
                
                p2_split = get(op, "p2_split", "")
                p2_dir = get(op, "p2_direction", "")
                p2_pt = resolve_extreme_point(p2_arr, p2_split, p2_dir, is_left_side, sp_x, sp_y, sp_z, dir_00)
                
                if !isnothing(p1_pt) && !isnothing(p2_pt)
                    length_mm = get(params, "length_mm", nothing)
                    p2_adj = [p2_pt[1], p2_pt[2], p2_pt[3]]
                    if length_mm !== nothing
                        dir_vec = p2_pt .- p1_pt
                        len = sqrt(sum(dir_vec .^ 2))
                        if len > 0
                            p2_adj = p1_pt .+ (dir_vec ./ len) .* Float64(length_mm)
                        end
                    end
                    out_view = get_channel_view(tm, out_ch)
                    fill!(out_view, 0)
                    orig_x = Float32(get(params, "origin_x", 0.0))
                    orig_y = Float32(get(params, "origin_y", 0.0))
                    orig_z = Float32(get(params, "origin_z", 0.0))
                    kernel! = CustomRules.cylinder_primitive_kernel!(tm.backend)
                    kernel!(out_view, Float32(p1_pt[1]), Float32(p1_pt[2]), Float32(p1_pt[3]),
                                      Float32(p2_adj[1]), Float32(p2_adj[2]), Float32(p2_adj[3]),
                                      radius_mm, orig_x, orig_y, orig_z, sp_x, sp_y, sp_z, ndrange=size(out_view))
                    KernelAbstractions.synchronize(tm.backend)
                end
            end

        elseif rule_type == "HilarAnteriorHelper"
            in_ch = op["input_channel"]
            in_view = get_channel_view(tm, in_ch)
            out_view = get_channel_view(tm, out_ch)
            HilarAnteriorHelper(tm, out_view, params, in_view)

        elseif rule_type == "InternalIliacCustom"
            out_view = get_channel_view(tm, out_ch)
            side = get(params, "side", "Left")
            p1_key = "internal_iliac_p1_" * lowercase(side)
            p2_key = "internal_iliac_p2_" * lowercase(side)
            
            p1 = get(computed_landmarks, p1_key, nothing)
            p2 = get(computed_landmarks, p2_key, nothing)
            radius_mm = Float32(get(params, "radius_mm", 7.0))
            
            if p1 !== nothing && p2 !== nothing
                sp_x = Float32(get(params, "spacing_x", 1.0))
                sp_y = Float32(get(params, "spacing_y", 1.0))
                sp_z = Float32(get(params, "spacing_z", 1.0))
                orig_x = Float32(get(params, "origin_x", 0.0))
                orig_y = Float32(get(params, "origin_y", 0.0))
                orig_z = Float32(get(params, "origin_z", 0.0))
                
                length_mm = get(params, "length_mm", nothing)
                p2_adj = [p2[1], p2[2], p2[3]]
                if length_mm !== nothing
                    dir_vec = p2 .- p1
                    len = sqrt(sum(dir_vec .^ 2))
                    if len > 0
                        p2_adj = p1 .+ (dir_vec ./ len) .* Float64(length_mm)
                    end
                end
                
                fill!(out_view, 0)
                kernel! = CustomRules.cylinder_primitive_kernel!(tm.backend)
                kernel!(out_view, Float32(p1[1]), Float32(p1[2]), Float32(p1[3]),
                                  Float32(p2_adj[1]), Float32(p2_adj[2]), Float32(p2_adj[3]),
                                  radius_mm, orig_x, orig_y, orig_z, sp_x, sp_y, sp_z, ndrange=size(out_view))
                KernelAbstractions.synchronize(tm.backend)
                
                # Apply dynamic midline split to ensure side separation
                is_left_side = lowercase(side) == "left"
                mid_x = tm.size_x ÷ 2
                split_k! = CustomRules.internal_iliac_split_kernel!(tm.backend)
                split_k!(out_view, out_view, size(out_view), mid_x, is_left_side, ndrange=size(out_view))
                KernelAbstractions.synchronize(tm.backend)
            else
                in_ch = get(op, "input_channel", 0)
                if in_ch > 0
                    in_view = get_channel_view(tm, in_ch)
                    anc_dict = get(op, "ancillary_channels_dict", Dict{String, Any}())
                    art_l_ch = get(anc_dict, "iliac_artery_left", 0)
                    art_r_ch = get(anc_dict, "iliac_artery_right", 0)
                    art_l_view = art_l_ch > 0 ? get_channel_view(tm, art_l_ch) : get_channel_view(tm, 0)
                    art_r_view = art_r_ch > 0 ? get_channel_view(tm, art_r_ch) : get_channel_view(tm, 0)
                    InternalIliacCustom(tm, out_view, params, in_view, art_l_view, art_r_view)
                end
            end

        elseif rule_type == "PropagateZ"
            in_ch = op["input_channel"]
            in_view = get_channel_view(tm, in_ch)
            out_view = get_channel_view(tm, out_ch)
            fill!(out_view, 0)
            
            direction = get(params, "direction", "inferior")
            is_inferior = lowercase(direction) == "inferior"
            
            terminus_ch = 0
            if haskey(op, "ancillary_channels_dict")
                anc = op["ancillary_channels_dict"]
                terminus_name = get(params, "terminus", "")
                if !isempty(terminus_name) && haskey(anc, terminus_name)
                    terminus_ch = anc[terminus_name]
                end
            end
            
            z_term = is_inferior ? 1 : tm.size_z
            if terminus_ch > 0
                term_view = get_channel_view(tm, terminus_ch)
                lim = resolve_landmark_limit([term_view], 3, is_inferior ? "min" : "max")
                if lim !== nothing
                    z_term = lim
                end
            end
            
            kernel! = CustomRules.propagate_z_kernel!(tm.backend)
            kernel!(out_view, in_view, size(out_view), is_inferior, z_term, ndrange=(size(out_view, 1), size(out_view, 2)))
            KernelAbstractions.synchronize(tm.backend)


        elseif rule_type == "AnteriorGrowthMask"
            in_ch = op["input_channel"]
            in_view = get_channel_view(tm, in_ch)
            out_view = get_channel_view(tm, out_ch)
            AnteriorGrowthMask(tm, out_view, params, in_view)

        elseif rule_type == "PosteriorGrowthMask"
            in_ch = op["input_channel"]
            in_view = get_channel_view(tm, in_ch)
            out_view = get_channel_view(tm, out_ch)
            PosteriorGrowthMask(tm, out_view, params, in_view)

        elseif rule_type in ["VolumetricBoundary2D", "VolumetricBoundary", "ConvexHull2D", "ConvexHull"]
            in_ch = op["input_channel"]
            in_view = in_ch > 0 ? get_channel_view(tm, in_ch) : get_empty_view(tm)
            out_view = get_channel_view(tm, out_ch)
            
            # Combine input and union channels as the bounding boundary envelope
            temp_in = KernelAbstractions.zeros(tm.backend, UInt8, tm.size_x, tm.size_y, tm.size_z)
            if in_ch > 0
                copyto!(temp_in, in_view)
            end
            u_chs = get(op, "union_channels", Int[])
            for k in ["structure_channels", "medial_channels", "lateral_channels", "posterior_channels", "anterior_channels"]
                append!(u_chs, get(op, k, Int[]))
            end
            for uc in u_chs
                if uc > 0
                    u_view = get_channel_view(tm, uc)
                    run_boolean_op!(tm.backend, temp_in, temp_in, u_view, "union")
                end
            end
            sum_temp_in = sum(adapt(Array, temp_in) .> 0)
            VolumetricBoundary2D(tm, out_view, params, temp_in)
            sum_out_view = sum(adapt(Array, out_view) .> 0)
            println("    [DEBUG VolumetricBoundary2D] target=$target sum_temp_in=$sum_temp_in sum_out_view=$sum_out_view")

        elseif rule_type == "VolumetricBoundary2DLateralGrowth"
            in_ch = get(op, "input_channel", 0)
            in_view = in_ch > 0 ? get_channel_view(tm, in_ch) : get_empty_view(tm)
            out_view = get_channel_view(tm, out_ch)
            anc_dict = get(op, "ancillary_channels_dict", Dict{String, Any}())
            
            side = lowercase(get(params, "side", ""))
            s = isempty(side) ? (occursin("right", lowercase(get(op, "target", ""))) ? "right" : "left") : side
            
            # Medial vessels
            vessel_in = KernelAbstractions.zeros(tm.backend, UInt8, tm.size_x, tm.size_y, tm.size_z)
            if in_ch > 0
                run_bitwise_or!(tm.backend, vessel_in, vessel_in, in_view)
            end
            for k in ["internal_jugular_vein_$s", "internal_jugular_vein", "internal_carotid_artery_$s", "internal_carotid_artery", "common_carotid_artery_$s", "common_carotid_artery"]
                v = get_ancillary_view(tm, anc_dict, k)
                if sum(v) > 0
                    run_bitwise_or!(tm.backend, vessel_in, vessel_in, v)
                end
            end
            
            # Lateral obstacle (SCM + Parotid)
            obs_in = KernelAbstractions.zeros(tm.backend, UInt8, tm.size_x, tm.size_y, tm.size_z)
            obs_ch = get(op, "obstacle_channel", 0)
            if obs_ch > 0
                run_bitwise_or!(tm.backend, obs_in, obs_in, get_channel_view(tm, obs_ch))
            end
            for k in ["sternocleidomastoid_$s", "sternocleidomastoid", "parotid_gland_$s", "parotid_gland"]
                v = get_ancillary_view(tm, anc_dict, k)
                if sum(v) > 0
                    run_bitwise_or!(tm.backend, obs_in, obs_in, v)
                end
            end
            
            ijv_v = get_ancillary_view(tm, anc_dict, "internal_jugular_vein_$s")
            if sum(ijv_v) == 0; ijv_v = get_ancillary_view(tm, anc_dict, "internal_jugular_vein"); end
            
            CustomRules.VolumetricBoundary2DLateralGrowth(tm, out_view, params, vessel_in, obs_in, ijv_v)

        elseif rule_type == "Station1LowCervical"
            in_ch = op["input_channel"]
            in_view = get_channel_view(tm, in_ch)
            out_view = get_channel_view(tm, out_ch)
            anc_dict = get(op, "ancillary_channels_dict", Dict{String, Any}())
            
            cricoid_v = get_ancillary_view(tm, anc_dict, "cricoid_cartilage")
            if sum(cricoid_v) == 0; cricoid_v = get_ancillary_view(tm, anc_dict, "cricoid"); end
            trachea_v = get_ancillary_view(tm, anc_dict, "trachea")
            manubrium_v = get_ancillary_view(tm, anc_dict, "manubrium")
            if sum(manubrium_v) == 0; manubrium_v = get_ancillary_view(tm, anc_dict, "sternum"); end
            clav_l = get_ancillary_view(tm, anc_dict, "clavicula_left")
            clav_r = get_ancillary_view(tm, anc_dict, "clavicula_right")
            clavicle_v = clav_l
            if sum(clav_r) > 0
                clavicle_v = clav_l .| clav_r
            end
            esophagus_v = get_ancillary_view(tm, anc_dict, "esophagus")
            thyroid_v = get_ancillary_view(tm, anc_dict, "thyroid_gland")
            if sum(thyroid_v) == 0; thyroid_v = get_ancillary_view(tm, anc_dict, "thyroid"); end
            lung_l_v = get_ancillary_view(tm, anc_dict, "lung_left")
            lung_r_v = get_ancillary_view(tm, anc_dict, "lung_right")
            scm_v = get_ancillary_view(tm, anc_dict, "sternocleidomastoid")
            scalene_v = get_ancillary_view(tm, anc_dict, "scalene_muscle")
            if sum(scalene_v) == 0; scalene_v = get_ancillary_view(tm, anc_dict, "scalene"); end
            
            CustomRules.Station1LowCervical(tm, out_view, params, in_view,
                                            cricoid_v, trachea_v, manubrium_v, clavicle_v,
                                            esophagus_v, thyroid_v, lung_l_v, lung_r_v,
                                            scm_v, scalene_v)

        elseif rule_type == "IliacBifurcationCustom"
            in_ch = get(op, "input_channel", 0)
            in_view = in_ch > 0 ? get_channel_view(tm, in_ch) : get_empty_view(tm)
            out_view = get_channel_view(tm, out_ch)
            anc_dict = get(op, "ancillary_channels_dict", Dict{String, Any}())
            art_l_ch = get(anc_dict, "iliac_artery_left", get(anc_dict, "iliac_artery", 0))
            art_r_ch = get(anc_dict, "iliac_artery_right", get(anc_dict, "iliac_artery", 0))
            ven_l_ch = get(anc_dict, "iliac_vena_left", get(anc_dict, "iliac_vena", 0))
            ven_r_ch = get(anc_dict, "iliac_vena_right", get(anc_dict, "iliac_vena", 0))
            
            println("    [DEBUG] IliacBifurcationCustom: art_l_ch=$art_l_ch, art_r_ch=$art_r_ch, ven_l_ch=$ven_l_ch, ven_r_ch=$ven_r_ch")
            
            art_l = art_l_ch > 0 ? get_channel_view(tm, art_l_ch) : in_view
            art_r = art_r_ch > 0 ? get_channel_view(tm, art_r_ch) : in_view
            ven_l = ven_l_ch > 0 ? get_channel_view(tm, ven_l_ch) : get_empty_view(tm)
            ven_r = ven_r_ch > 0 ? get_channel_view(tm, ven_r_ch) : get_empty_view(tm)
            
            IliacBifurcationCustom(tm, out_view, params, computed_landmarks, art_l, art_r, ven_l, ven_r)

        elseif rule_type == "ExternalIliacCustom"
            in_ch = op["input_channel"]
            in_view = get_channel_view(tm, in_ch)
            out_view = get_channel_view(tm, out_ch)
            anc_dict = get(op, "ancillary_channels_dict", Dict{String, Any}())
            femur_ch = get(anc_dict, "femur", 0)
            art_l_ch = get(anc_dict, "iliac_artery_left", 0)
            art_r_ch = get(anc_dict, "iliac_artery_right", 0)
            ven_l_ch = get(anc_dict, "iliac_vena_left", 0)
            ven_r_ch = get(anc_dict, "iliac_vena_right", 0)
            femur_view = femur_ch > 0 ? get_channel_view(tm, femur_ch) : get_channel_view(tm, 0)
            art_l_view = art_l_ch > 0 ? get_channel_view(tm, art_l_ch) : get_channel_view(tm, 0)
            art_r_view = art_r_ch > 0 ? get_channel_view(tm, art_r_ch) : get_channel_view(tm, 0)
            ven_l_view = ven_l_ch > 0 ? get_channel_view(tm, ven_l_ch) : get_channel_view(tm, 0)
            ven_r_view = ven_r_ch > 0 ? get_channel_view(tm, ven_r_ch) : get_channel_view(tm, 0)
            ExternalIliacCustom(tm, out_view, params, computed_landmarks, in_view, femur_view, art_l_view, art_r_view, ven_l_view, ven_r_view)

        elseif rule_type == "AxillaryHelperB"
            in_ch = op["input_channel"]
            in_view = get_channel_view(tm, in_ch)
            out_view = get_channel_view(tm, out_ch)
            anc_dict = get(op, "ancillary_channels_dict", Dict{String, Any}())
            pec_ch = get(anc_dict, "pectoralis_major_left", get(anc_dict, "pectoralis_major_right", 0))
            sub_ch = get(anc_dict, "helper_subscapularis_left", get(anc_dict, "helper_subscapularis_right", 0))
            pec_view = pec_ch > 0 ? get_channel_view(tm, pec_ch) : get_channel_view(tm, 0)
            sub_view = sub_ch > 0 ? get_channel_view(tm, sub_ch) : get_channel_view(tm, 0)
            AxillaryHelperB(tm, out_view, params, pec_view, sub_view)

        elseif rule_type == "PleuralSpaceCustom"
            in_ch = op["input_channel"]
            in_view = get_channel_view(tm, in_ch)
            out_view = get_channel_view(tm, out_ch)
            PleuralSpaceCustom(tm, out_view, params, in_view)

        elseif rule_type == "PresacralAnteriorCustom"
            in_ch = op["input_channel"]
            in_view = get_channel_view(tm, in_ch)
            out_view = get_channel_view(tm, out_ch)
            PresacralAnteriorCustom(tm, out_view, params, in_view)

        elseif rule_type == "SplitConnectedComponents"
            in_ch = op["input_channel"]
            in_view = get_channel_view(tm, in_ch)
            out_view = get_channel_view(tm, out_ch)
            SplitConnectedComponents(tm, out_view, params, in_view)

        elseif rule_type == "SplitConnectedComponents"
            in_ch = get(op, "input_channel", 0)
            in_view = in_ch > 0 ? get_channel_view(tm, in_ch) : get_empty_view(tm)
            out_view = get_channel_view(tm, out_ch)
            SplitConnectedComponents(tm, out_view, params, in_view)

        elseif rule_type == "AxillaryRTOG" || rule_type == "AxillaryRTOGRelaxed"
            in_ch = get(op, "input_channel", 0)
            in_view = in_ch > 0 ? get_channel_view(tm, in_ch) : get_empty_view(tm)
            out_view = get_channel_view(tm, out_ch)
            
            anc_dict = get(op, "ancillary_channels_dict", Dict{String, Any}())
            side = get(params, "side", "left")
            s = lowercase(side)
            pec_ch = get(anc_dict, "pectoralis_major_$s", 0)
            pm_ch = get(anc_dict, "pectoralis_minor_$s", 0)
            sub_ch = get(anc_dict, "subscapularis_$s", 0)
            art_ch = get(anc_dict, "subclavian_artery_$s", 0)
            chestwall_ch = get(anc_dict, "helper_chestwall_bridge", 0)
            
            pec_view = pec_ch > 0 ? get_channel_view(tm, pec_ch) : get_empty_view(tm)
            pm_view = pm_ch > 0 ? get_channel_view(tm, pm_ch) : get_empty_view(tm)
            sub_view = sub_ch > 0 ? get_channel_view(tm, sub_ch) : get_empty_view(tm)
            art_view = art_ch > 0 ? get_channel_view(tm, art_ch) : get_empty_view(tm)
            chestwall_view = chestwall_ch > 0 ? get_channel_view(tm, chestwall_ch) : get_empty_view(tm)
            
            if rule_type == "AxillaryRTOG"
                AxillaryRTOG(tm, out_view, params, in_view, computed_landmarks, pec_view, pm_view, sub_view, art_view, chestwall_view)
            else
                AxillaryRTOGRelaxed(tm, out_view, params, in_view, computed_landmarks, pec_view, pm_view, sub_view, art_view, chestwall_view)
            end

        elseif rule_type == "LimitZByLandmark"
            in_ch = op["input_channel"]
            in_view = get_channel_view(tm, in_ch)
            out_view = get_channel_view(tm, out_ch)
            
            lm_ch = get(op, "landmark_channel", 0)
            z_min = 1
            z_max = tm.size_z
            if lm_ch > 0
                lm_view = get_channel_view(tm, lm_ch)
                z_min_val = resolve_landmark_limit([lm_view], 3, "min")
                z_max_val = resolve_landmark_limit([lm_view], 3, "max")
                if z_min_val !== nothing; z_min = z_min_val; end
                if z_max_val !== nothing; z_max = z_max_val; end
            end
            
            in_sum = sum(Array(in_view) .> 0)
            lm_sum = lm_ch > 0 ? sum(Array(lm_view) .> 0) : 0
            println("    [DEBUG] LimitZByLandmark: in_ch=$in_ch (sum=$in_sum), lm_ch=$lm_ch (sum=$lm_sum), bounds=[$z_min, $z_max]")
            
            kernel! = CustomRules.limit_z_by_landmark_kernel!(tm.backend)
            kernel!(out_view, in_view, size(out_view), z_min, z_max, ndrange=size(out_view))
            KernelAbstractions.synchronize(tm.backend)

        elseif rule_type == "AnteriorExtrusion"
            in_ch = op["input_channel"]
            in_view = get_channel_view(tm, in_ch)
            out_view = get_channel_view(tm, out_ch)
            
            dist_mm = Float32(get(params, "distance_mm", get(params, "distance", 10.0)))
            sp_y = Float32(get(params, "spacing_y", 1.0))
            voxel_dist = Int(round(dist_mm / sp_y))
            
            fill!(out_view, 0)
            kernel! = CustomRules.anterior_growth_kernel!(tm.backend)
            kernel!(out_view, in_view, size(out_view), voxel_dist, ndrange=(size(out_view, 1), size(out_view, 3)))
            KernelAbstractions.synchronize(tm.backend)

        else
            println("    [WARNING] Batch executor does not support rule type '$rule_type' for target '$(op["target"])'. Skipping operation and delegating to Python fallback.")
        end
        # ============================================================
        # Post-rule constraint processing
        # ============================================================
        constraints = get(op, "constraints", [])
        if !isempty(constraints)
            sp_x = Float64(get(params, "spacing_x", 1.0))
            sp_y = Float64(get(params, "spacing_y", 1.0))
            sp_z = Float64(get(params, "spacing_z", 1.0))
            dir_00 = Float64(get(params, "direction_00", 1.0))
            dir_11 = Float64(get(params, "direction_11", 1.0))
            dir_22 = Float64(get(params, "direction_22", 1.0))
            orig_x = Float64(get(params, "origin_x", 0.0))
            orig_y = Float64(get(params, "origin_y", 0.0))
            orig_z = Float64(get(params, "origin_z", 0.0))
            process_constraints!(tm.backend, tm, out_ch, constraints, sp_x, sp_y, sp_z, dir_00, dir_11, dir_22, orig_x, orig_y, orig_z)
        end
        
        # ============================================================
        # Post-rule boolean operations (union/subtract/intersect)
        # Applied for ALL rule types after constraints
        # ============================================================
        if !(rule_type in ["BooleanOps", "BooleanOperation", "BitwiseOr", "BitwiseAnd", "BitwiseSub"])
            u_chs = get(op, "union_channels", Int[])
            s_chs = get(op, "subtract_channels", Int[])
            i_chs = get(op, "intersect_channels", Int[])
            
            if !isempty(u_chs) || !isempty(s_chs) || !isempty(i_chs)
                out_arr_gpu = get_channel_view(tm, out_ch)
                
                for u_ch in u_chs
                    u_view = get_channel_view(tm, u_ch)
                    run_boolean_op!(tm.backend, out_arr_gpu, out_arr_gpu, u_view, "union")
                end
                
                for s_ch in s_chs
                    s_view = get_channel_view(tm, s_ch)
                    run_boolean_op!(tm.backend, out_arr_gpu, out_arr_gpu, s_view, "subtract")
                end
                
                for i_ch in i_chs
                    i_view = get_channel_view(tm, i_ch)
                    run_boolean_op!(tm.backend, out_arr_gpu, out_arr_gpu, i_view, "intersect")
                end
            end
            
            if haskey(op, "layer_wise_propagation")
                lwp = op["layer_wise_propagation"]
                if haskey(lwp, "target_channels")
                    t_channels = lwp["target_channels"]
                    if !isempty(t_channels)
                        out_arr_gpu = get_channel_view(tm, out_ch)
                        out_arr_cpu = Array(out_arr_gpu)
                        
                        target_arr_cpu = zeros(eltype(out_arr_cpu), size(out_arr_cpu))
                        for t_ch in t_channels
                            if t_ch > 0
                                t_arr = Array(get_channel_view(tm, t_ch))
                                target_arr_cpu .|= t_arr
                            end
                        end
                        
                        dir = get(lwp, "direction", "inferior")
                        step_mm = Float64(get(lwp, "step_dilation_mm", 0.0))
                        max_mm = Float64(get(lwp, "max_mm", 300.0))
                        
                        sp_x = Float64(get(params, "spacing_x", 1.0))
                        sp_z = Float64(get(params, "spacing_z", 1.0))
                        step_px = round(Int, step_mm / sp_x)
                        max_slices = round(Int, max_mm / sp_z)
                        
                        Main.run_layer_wise_propagation_cpu!(out_arr_cpu, target_arr_cpu, dir, step_px, max_slices)
                        copyto!(out_arr_gpu, out_arr_cpu)
                    end
                elseif haskey(lwp, "target_channel") && lwp["target_channel"] > 0
                    target_ch = lwp["target_channel"]
                    out_arr_gpu = get_channel_view(tm, out_ch)
                    out_arr_cpu = Array(out_arr_gpu)
                    target_arr_gpu = get_channel_view(tm, target_ch)
                    target_arr_cpu = Array(target_arr_gpu)
                    
                    dir = get(lwp, "direction", "inferior")
                    step_mm = Float64(get(lwp, "step_dilation_mm", 0.0))
                    max_mm = Float64(get(lwp, "max_mm", 300.0))
                    
                    sp_x = Float64(get(params, "spacing_x", 1.0))
                    sp_z = Float64(get(params, "spacing_z", 1.0))
                    step_px = round(Int, step_mm / sp_x)
                    max_slices = round(Int, max_mm / sp_z)
                    
                    Main.run_layer_wise_propagation_cpu!(out_arr_cpu, target_arr_cpu, dir, step_px, max_slices)
                    copyto!(out_arr_gpu, out_arr_cpu)
                end
            end
            
            # Post-processing connected components filtering
        if !get(params, "keep_all_components", false) && !(rule_type in ["AxillaryRTOG", "AxillaryRTOGRelaxed"]) && !startswith(op["target"], "helper_")
            out_arr_gpu = get_channel_view(tm, out_ch)
            get_largest_connected_component!(out_arr_gpu)
        end
    end
    
    # Wait for all async kernels in this batch level to complete
    KernelAbstractions.synchronize(tm.backend)

    # ============================================================
    # Bilateral splitting
    # ============================================================
    for split_req in get(batch_req, "bilateral_splits", [])
        in_ch = split_req["input_channel"]
        spine_ch = split_req["spine_channel"]
        left_ch = split_req["left_channel"]
        right_ch = split_req["right_channel"]
        split_type = get(split_req, "split_type", "spine")
        split_lm_ch = get(split_req, "split_landmark_channel", 0)
        
        split_bilateral!(tm.backend, tm, in_ch, spine_ch, left_ch, right_ch;
                         split_landmark_ch=split_lm_ch, split_type=split_type)
        
        println("  -> Split bilateral: channel $in_ch -> left=$left_ch, right=$right_ch")
    end
    
    # 3. Save requested output masks to NPY
    for save_req in get(batch_req, "save_masks", [])
        ch = save_req["channel"]
        path = save_req["path"]
        
        mask_host = save_channel_to_mask(tm, ch)
        npzwrite(path, mask_host)
    end
    
    # 4. Free masks no longer needed
    for ch in get(batch_req, "free_masks", [])
        free_channel!(tm, ch)
    end
end
end
