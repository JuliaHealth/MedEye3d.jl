import Pkg
# Initialize dependencies dynamically if needed, though they should be in the main environment
using KernelAbstractions
using Adapt
using JSON
using NPZ

# Import morphology operations (Anisotropic Margin, etc.)
using ImageMorphology
include("kernels_morphology.jl")

# Import geometry operations (CoronalLateralGrowth, etc.)
include("kernels_geometry.jl")

# Import Boolean operations
include("kernels_boolean.jl")

# Import Mega Fused Kernel
include("kernels_fused_mega.jl")

# Import Constraint Kernels
include("kernels_constraints.jl")

# Import Bilateral Splitting Kernels
include("kernels_bilateral.jl")

include("kernels_convexhull.jl")

# Try to load CUDA / AMDGPU backends
global backend = CPU()
try
    using CUDA
    if CUDA.functional()
        global backend = CUDABackend()
        println("Using CUDABackend")
    end
catch e
    try
        using AMDGPU
        if AMDGPU.functional()
            global backend = ROCBackend()
            println("Using ROCBackend")
        end
    catch e2
        println("Using CPU Backend (fallback)")
    end
end

function process_gpu_rule(json_path::String, temp_dir::String)
    # 1. Read JSON
    rule_data = JSON.parsefile(json_path)
    
    # Expect format: { "RuleName": { "rule": "AnisotropicMargin", ... } }
    rule_name = first(keys(rule_data))
    params = rule_data[rule_name]
    rule_type = get(params, "rule", "")
    
    println("Julia Engine executing rule: $rule_name of type: $rule_type")
    
    # 2. Execute specific rule
    if rule_type == "AnisotropicMargin"
        input_path = joinpath(temp_dir, "input.npy")
        if isfile(input_path)
            input_host = npzread(input_path)
            # Move data to GPU device
            input_device = adapt(backend, input_host)
            output_device = adapt(backend, zeros(eltype(input_host), size(input_host)))
            
            sp_x = get(params, "spacing_x", 1.0)
            sp_y = get(params, "spacing_y", 1.0)
            sp_z = get(params, "spacing_z", 1.0)
            
            margins = get(params, "margins_mm", Dict("all" => 10.0))
            
            # Execute Kernel on GPU device arrays
            run_anisotropic_dilation!(backend, output_device, input_device, sp_x, sp_y, sp_z, margins)
            
            # Download result from GPU to CPU for saving
            output_host = Array(output_device)
            npzwrite(joinpath(temp_dir, "output.npy"), output_host)
        else
            println("Error: input.npy not found")
        end
    elseif rule_type == "DistanceExpansion"
        input_path = joinpath(temp_dir, "input.npy")
        if isfile(input_path)
            input_host = npzread(input_path)
            # Move data to GPU device
            input_device = adapt(backend, input_host)
            output_device = adapt(backend, zeros(eltype(input_host), size(input_host)))
            
            sp_x = get(params, "spacing_x", 1.0)
            sp_y = get(params, "spacing_y", 1.0)
            sp_z = get(params, "spacing_z", 1.0)
            
            dist = get(params, "distance_mm", 25.0)
            margins = Dict("left" => dist, "right" => dist, "anterior" => dist, "posterior" => dist, "superior" => dist, "inferior" => dist)
            
            # Execute Kernel on GPU device arrays
            run_anisotropic_dilation!(backend, output_device, input_device, sp_x, sp_y, sp_z, margins)
            
            # Download result from GPU to CPU for saving
            output_host = Array(output_device)
            npzwrite(joinpath(temp_dir, "output.npy"), output_host)
        else
            println("Error: input.npy not found for DistanceExpansion")
        end
        
    elseif rule_type == "BooleanOps"
        op_str = get(params, "op", get(params, "op_type", get(params, "operation", "Unknown")))
        
        # Load base_image
        input_path = joinpath(temp_dir, "input.npy")
        if !isfile(input_path)
            println("No base input for BooleanOps")
            return
        end
        
        base_host = npzread(input_path)
        current_device = adapt(backend, base_host)
        
        # We need to find all other inputs. They are saved as other_0.npy, other_1.npy...
        # Wait, how does python wrapper name them?
        # Let's assume the wrapper will name the base 'input' and others 'target_0', 'target_1', etc.
        # We can just iterate over target_*.npy
        for i in 0:100
            target_path = joinpath(temp_dir, "target_$i.npy")
            if isfile(target_path)
                target_host = npzread(target_path)
                target_device = adapt(backend, target_host)
                output_device = adapt(backend, zeros(eltype(base_host), size(base_host)))
                
                run_boolean_op!(backend, output_device, current_device, target_device, op_str)
                current_device = output_device
            else
                break
            end
        end
        
        output_host = Array(adapt(CPU(), current_device))
        npzwrite(joinpath(temp_dir, "output.npy"), output_host)

    elseif rule_type == "ZPlaneIntersection"
        # Load Z-plane landmark (base input)
        input_path = joinpath(temp_dir, "input.npy")
        if !isfile(input_path)
            println("No z_plane_image input for ZPlaneIntersection")
            return
        end
        
        z_plane_host = npzread(input_path)
        
        z_min = size(z_plane_host, 3)
        z_max = 1
        has_z = false
        for z in 1:size(z_plane_host, 3)
            if any(z_plane_host[:, :, z] .> 0)
                z_min = min(z_min, z)
                z_max = max(z_max, z)
                has_z = true
            end
        end
        
        if !has_z
            output_host = zeros(eltype(z_plane_host), size(z_plane_host))
            npzwrite(joinpath(temp_dir, "output.npy"), output_host)
            return
        end
        
        # Intersect all target inputs
        intersection_device = nothing
        for i in 0:100
            target_path = joinpath(temp_dir, "target_$i.npy")
            if isfile(target_path)
                target_host = npzread(target_path)
                target_device = adapt(backend, target_host)
                if intersection_device === nothing
                    intersection_device = target_device
                else
                    output_device = adapt(backend, zeros(eltype(z_plane_host), size(z_plane_host)))
                    run_boolean_op!(backend, output_device, intersection_device, target_device, "intersect")
                    intersection_device = output_device
                end
            else
                break
            end
        end
        
        if intersection_device === nothing
            output_host = zeros(eltype(z_plane_host), size(z_plane_host))
            npzwrite(joinpath(temp_dir, "output.npy"), output_host)
            return
        end
        
        # Clip Z
        final_device = adapt(backend, zeros(eltype(z_plane_host), size(z_plane_host)))
        run_z_plane_clip!(backend, final_device, intersection_device, z_min, z_max)
        
        output_host = Array(adapt(CPU(), final_device))
        npzwrite(joinpath(temp_dir, "output.npy"), output_host)
        
    elseif rule_type == "CoronalLateralGrowth"
        input_path = joinpath(temp_dir, "input.npy")
        if !isfile(input_path)
            return
        end
        
        input_host = npzread(input_path)
        input_device = adapt(backend, input_host)
        output_device = adapt(backend, zeros(eltype(input_host), size(input_host)))
        
        sp_x = get(params, "spacing_x", 1.0)
        margin_mm = get(params, "margin_mm", 0.0)
        
        run_coronal_lateral_growth!(backend, output_device, input_device, Float32(sp_x), Float32(margin_mm))
        
        output_host = Array(adapt(CPU(), output_device))
        npzwrite(joinpath(temp_dir, "output.npy"), output_host)

    elseif rule_type == "LateralBridge"
        input_path = joinpath(temp_dir, "input.npy")
        if !isfile(input_path)
            return
        end
        
        base_host = npzread(input_path)
        
        # Merge targets
        target_device = nothing
        for i in 0:100
            target_path = joinpath(temp_dir, "target_$i.npy")
            if isfile(target_path)
                t_host = npzread(target_path)
                t_dev = adapt(backend, t_host)
                if target_device === nothing
                    target_device = t_dev
                else
                    out_dev = adapt(backend, zeros(eltype(base_host), size(base_host)))
                    run_boolean_op!(backend, out_dev, target_device, t_dev, "union")
                    target_device = out_dev
                end
            else
                break
            end
        end
        
        if target_device === nothing
            target_device = adapt(backend, zeros(eltype(base_host), size(base_host)))
        end
        
        output_device = adapt(backend, zeros(eltype(base_host), size(base_host)))
        sp_x = get(params, "spacing_x", 1.0)
        max_mm = get(params, "max_mm", 100.0)
        
        # Move base to GPU device
        base_device = adapt(backend, base_host)
        run_lateral_bridge!(backend, output_device, base_device, target_device, Float32(sp_x), Float32(max_mm))
        
        output_host = Array(adapt(CPU(), output_device))
        npzwrite(joinpath(temp_dir, "output.npy"), output_host)

    elseif rule_type == "PrimaryVector"
        input_path = joinpath(temp_dir, "input.npy")
        vector_path = joinpath(temp_dir, "target_0.npy")
        
        if !isfile(input_path) || !isfile(vector_path)
            return
        end
        
        primary_host = npzread(input_path)
        vector_host = npzread(vector_path)
        
        # Move data to GPU device
        primary_device = adapt(backend, primary_host)
        vector_device = adapt(backend, vector_host)
        output_device = adapt(backend, zeros(eltype(primary_host), size(primary_host)))
        
        sp_x = get(params, "spacing_x", 1.0)
        sp_y = get(params, "spacing_y", 1.0)
        sp_z = get(params, "spacing_z", 1.0)
        base_margin = get(params, "base_margin_mm", 0.0)
        exp_margin = get(params, "expansion_mm", 0.0)
        
        run_primary_vector!(backend, output_device, primary_device, vector_device, Float32(sp_x), Float32(sp_y), Float32(sp_z), Float32(base_margin), Float32(exp_margin))
        
        output_host = Array(adapt(CPU(), output_device))
        npzwrite(joinpath(temp_dir, "output.npy"), output_host)

    elseif rule_type == "SplitConnectedComponents"
        input_path = joinpath(temp_dir, "input.npy")
        if !isfile(input_path)
            return
        end
        
        arr = npzread(input_path)
        side = lowercase(get(params, "side", "left"))
        
        dims = size(arr)
        labels = zeros(Int32, dims)
        current_label = 0
        
        n_nonzero = sum(arr .> 0)
        if n_nonzero > 0
            queue_I = zeros(Int, n_nonzero)
            queue_J = zeros(Int, n_nonzero)
            queue_K = zeros(Int, n_nonzero)
            
            for K in 1:dims[3], J in 1:dims[2], I in 1:dims[1]
                if arr[I,J,K] > 0 && labels[I,J,K] == 0
                    current_label += 1
                    labels[I,J,K] = current_label
                    
                    head = 1
                    tail = 2
                    queue_I[1] = I
                    queue_J[1] = J
                    queue_K[1] = K
                    
                    while head < tail
                        qi = queue_I[head]
                        qj = queue_J[head]
                        qk = queue_K[head]
                        head += 1
                        
                        for dk in -1:1, dj in -1:1, di in -1:1
                            if di == 0 && dj == 0 && dk == 0 continue end
                            ni, nj, nk = qi+di, qj+dj, qk+dk
                            if ni >= 1 && ni <= dims[1] && nj >= 1 && nj <= dims[2] && nk >= 1 && nk <= dims[3]
                                if arr[ni,nj,nk] > 0 && labels[ni,nj,nk] == 0
                                    labels[ni,nj,nk] = current_label
                                    queue_I[tail] = ni
                                    queue_J[tail] = nj
                                    queue_K[tail] = nk
                                    tail += 1
                                end
                            end
                        end
                    end
                end
            end
        end
        
        n_comp = current_label
        result_arr = zeros(UInt8, dims)
        
        origin_x = get(params, "origin_x", 0.0)
        dir_00 = get(params, "direction_00", 1.0)
        sp_x = get(params, "spacing_x", 1.0)
        
        if n_comp < 2
            for K in 1:dims[3], J in 1:dims[2], I in 1:dims[1]
                if arr[I,J,K] > 0
                    x_vox = I - 1
                    phys_x = origin_x + dir_00 * x_vox * sp_x
                    if side == "left"
                        if phys_x > 0 result_arr[I,J,K] = 1 end
                    else
                        if phys_x <= 0 result_arr[I,J,K] = 1 end
                    end
                end
            end
        else
            comps = []
            for comp_id in 1:n_comp
                sum_x = 0.0
                count = 0
                for K in 1:dims[3], J in 1:dims[2], I in 1:dims[1]
                    if labels[I,J,K] == comp_id
                        sum_x += (I - 1)
                        count += 1
                    end
                end
                x_mean = count > 0 ? sum_x / count : 0.0
                phys_x = origin_x + dir_00 * x_mean * sp_x
                push!(comps, (phys_x, comp_id))
            end
            
            sort!(comps, by = x -> x[1], rev=true)
            
            target_comp = side == "left" ? comps[1][2] : comps[2][2]
            
            for i in eachindex(arr)
                if labels[i] == target_comp
                    result_arr[i] = 1
                end
            end
        end
        
        npzwrite(joinpath(temp_dir, "output.npy"), result_arr)

    elseif rule_type == "FusedPostProcessing"
        input_path = joinpath(temp_dir, "input.npy")
        if !isfile(input_path) return end
        
        arr = npzread(input_path)
        output_device = adapt(backend, arr)
        
        union_imgs = get(params, "union_images", [])
        for u in union_imgs
            u_path = joinpath(temp_dir, u * ".npy")
            if isfile(u_path)
                u_arr = npzread(u_path)
                u_device = adapt(backend, u_arr)
                run_boolean_op!(backend, output_device, output_device, u_device, "union")
            end
        end
        
        subtract_imgs = get(params, "subtract_images", [])
        for s in subtract_imgs
            s_path = joinpath(temp_dir, s * ".npy")
            if isfile(s_path)
                s_arr = npzread(s_path)
                s_device = adapt(backend, s_arr)
                run_boolean_op!(backend, output_device, output_device, s_device, "subtract")
            end
        end
        
        intersect_imgs = get(params, "intersect_images", [])
        for i in intersect_imgs
            i_path = joinpath(temp_dir, i * ".npy")
            if isfile(i_path)
                i_arr = npzread(i_path)
                i_device = adapt(backend, i_arr)
                run_boolean_op!(backend, output_device, output_device, i_device, "intersect")
            end
        end
        
        output_host = Array(adapt(CPU(), output_device))
        npzwrite(joinpath(temp_dir, "output.npy"), output_host)
        
    elseif rule_type == "MegaFused"
        # 1. Load masks
        masks = AbstractArray[]
        idx = 0
        while true
            m_path = joinpath(temp_dir, "input_$idx.npy")
            if isfile(m_path)
                push!(masks, npzread(m_path))
                idx += 1
            else
                break
            end
        end
        
        # 2. Parse ops levels
        ops_levels_json = get(params, "ops_levels", [])
        ops_levels = Vector{VoxelOp}[]
        for level in ops_levels_json
            ops = VoxelOp[]
            for op in level
                op_type = Int32(get(op, "type", 0))
                t_id = UInt16(get(op, "target_idx", 0) + 1) # Julia is 1-indexed
                a_id = UInt16(get(op, "arg_idx", 0) + 1)
                push!(ops, VoxelOp(op_type, t_id, a_id))
            end
            push!(ops_levels, ops)
        end
        
        # 3. Execute
        if !isempty(masks)
            out_masks = run_mega_fused(backend, masks, ops_levels)
            for (i, m) in enumerate(out_masks)
                npzwrite(joinpath(temp_dir, "output_$(i-1).npy"), m)
            end
        end

    else
        println("Warning: Rule type $rule_type not specifically handled by Julia GPU engine yet.")
    end
end

# CLI entry point
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2
        println("Usage: julia engine_ka.jl <temp_dir> <rule_json_path>")
        exit(1)
    end
    temp_dir = ARGS[1]
    json_path = ARGS[2]
    process_gpu_rule(json_path, temp_dir)
end
