using JSON
using CUDA
using KernelAbstractions
using Adapt

include("vm_types.jl")
using .VmTypes
include("vm_kernel.jl")
using .VmKernel
include("jfa.jl")
include("hull_planes.jl")
using .HullPlanes
include("cc_kernel.jl")
using .CcKernel

mutable struct TensorManagerVM
    tensor::Any 
    backend::Any
    sp::Any
    dims::Tuple{Int, Int, Int}
    
    # Pre-allocated buffers for operations to prevent OOM
    jfa_seeds_A::Any
    jfa_seeds_B::Any
    jfa_distances::Any
    
    cc_labels_A::Any
    cc_labels_B::Any
end

const TM_VM = TensorManagerVM(nothing, nothing, nothing, (0,0,0), nothing, nothing, nothing, nothing, nothing)

function init_vm_tensor(size_x, size_y, size_z, num_channels, sp_x, sp_y, sp_z; use_gpu=true)
    backend = use_gpu && CUDA.functional() ? CUDABackend() : CPU()
    
    # Int32 ID tensor (4D: X, Y, Z, Channels)
    tensor = adapt(backend, zeros(UInt32, size_x, size_y, size_z, num_channels))
    
    TM_VM.tensor = tensor
    TM_VM.backend = backend
    TM_VM.sp = (Float32(sp_x), Float32(sp_y), Float32(sp_z))
    TM_VM.dims = (size_x, size_y, size_z)
    
    # Allocate JFA buffers (3 channels for x,y,z coordinates of nearest seed)
    TM_VM.jfa_seeds_A = adapt(backend, zeros(UInt16, 3, size_x, size_y, size_z))
    TM_VM.jfa_seeds_B = adapt(backend, zeros(UInt16, 3, size_x, size_y, size_z))
    TM_VM.jfa_distances = adapt(backend, fill(Inf32, size_x, size_y, size_z))
    
    TM_VM.cc_labels_A = adapt(backend, zeros(Int32, size_x, size_y, size_z))
    TM_VM.cc_labels_B = adapt(backend, zeros(Int32, size_x, size_y, size_z))
    
    return "Initialized VM Tensor on $(typeof(backend)) with $(num_channels) channels"
end

@kernel function insert_mask_vm_kernel!(tensor, in_mask, mask_id, channel_idx, dims)
    I, J, K = @index(Global, NTuple)
    if I <= dims[1] && J <= dims[2] && K <= dims[3]
        if in_mask[I, J, K] > 0
            tensor[I, J, K, channel_idx] |= (UInt32(1) << mask_id)
        end
    end
end

function insert_mask_vm(backend, tensor, in_mask, mask_id, channel_idx, dims)
    insert_mask_vm_kernel!(backend, 256)(tensor, in_mask, UInt32(mask_id), Int32(channel_idx), dims, ndrange=dims)
    KernelAbstractions.synchronize(backend)
end

@kernel function extract_mask_vm_kernel!(out_mask, tensor, mask_id, channel_idx, dims)
    I, J, K = @index(Global, NTuple)
    if I <= dims[1] && J <= dims[2] && K <= dims[3]
        out_mask[I, J, K] = UInt8((tensor[I, J, K, channel_idx] & (UInt32(1) << mask_id)) != 0)
    end
end

function extract_mask_vm(backend, tensor, mask_id, channel_idx, dims)
    out_mask = adapt(backend, zeros(UInt8, dims[1], dims[2], dims[3]))
    if mask_id == 0xFFFFFFFF || mask_id == -1
        return out_mask
    end
    extract_mask_vm_kernel!(backend, 256)(out_mask, tensor, UInt32(mask_id), Int32(channel_idx), dims, ndrange=dims)
    KernelAbstractions.synchronize(backend)
    return out_mask
end

function execute_vm_batch(batch_json_str)
    t0 = time_ns()
    batch_dict = JSON.parse(batch_json_str)
    batch = get(batch_dict, "operations", [])
    
    instructions = Instruction[]
    
    for op in batch
        opcode = UInt32(op["opcode"])
        in1_id = Int32(get(op, "in1_id", -1))
        in1_channel = Int32(get(op, "in1_channel", -1))
        in2_id = Int32(get(op, "in2_id", -1))
        in2_channel = Int32(get(op, "in2_channel", -1))
        out_id = Int32(get(op, "out_id", -1))
        out_channel = Int32(get(op, "out_channel", -1))
        farg1 = Float32(get(op, "farg1", 0.0))
        farg2 = Float32(get(op, "farg2", 0.0))
        farg3 = Float32(get(op, "farg3", 0.0))
        farg4 = Float32(get(op, "farg4", 0.0))
        
        push!(instructions, Instruction(
            opcode, in1_id, in1_channel, in2_id, in2_channel, 
            out_id, out_channel, farg1, farg2, farg3, farg4
        ))
    end
    
    t_parse = time_ns()
    
    if !isempty(instructions)
        dims = TM_VM.dims
        execute_vm_kernel!(TM_VM.backend, TM_VM.tensor, instructions, x_dim=dims[1], y_dim=dims[2], z_dim=dims[3])
    end
    
    t_kernel = time_ns()
    println("[BENCH] vm_batch: parse=$(round(Int, (t_parse-t0)/1e6))ms kernel=$(round(Int, (t_kernel-t_parse)/1e6))ms n_ops=$(length(instructions))")
    flush(stdout)
    
    return "Executed VM batch with $(length(instructions)) instructions"
end

# Execute a batch that includes half-space hull operations
# This is a 2-phase process:
# Phase 1: Run edge extraction instructions via standard kernel
# Phase 2: Extract edges to CPU, compute per-slice 2D hulls, build planes table, run hull kernel
function execute_vm_hull_batch(batch_json_str)
    batch_dict = JSON.parse(batch_json_str)
    
    # Phase 1: Edge extraction operations
    edge_ops = get(batch_dict, "edge_ops", [])
    # Phase 2: Hull operations that need planes
    hull_ops = get(batch_dict, "hull_ops", [])
    # Regular ops that can run in either phase
    regular_ops = get(batch_dict, "regular_ops", [])
    
    dims = TM_VM.dims
    
    # Execute Phase 1: edge extraction + regular ops
    phase1_insts = Instruction[]
    for op in vcat(regular_ops, edge_ops)
        push!(phase1_insts, parse_instruction(op))
    end
    if !isempty(phase1_insts)
        execute_vm_kernel!(TM_VM.backend, TM_VM.tensor, phase1_insts, x_dim=dims[1], y_dim=dims[2], z_dim=dims[3])
    end
    
    # Phase 2: For each hull operation, compute planes and execute
    for hull_op in hull_ops
        edge_mask_id = Int32(hull_op["edge_mask_id"])
        edge_mask_ch = Int32(hull_op["edge_mask_ch"])
        obstacle_ids = get(hull_op, "obstacle_ids", [])
        out_id = Int32(hull_op["out_id"])
        out_ch = Int32(hull_op["out_ch"])
        keep_all = get(hull_op, "keep_all", false)
        
        # Compute per-slice 2D convex hull planes directly on GPU
        planes_table, num_planes_arr = compute_sector_planes_gpu(TM_VM.backend, TM_VM.tensor, edge_mask_id, edge_mask_ch, dims)
        
        # We can still extract num_planes_arr to CPU just to check if it's empty
        num_planes_cpu = adapt(CPU(), num_planes_arr)
        println("    [Hull] Computed GPU planes table: slices_with_hull=$(count(num_planes_cpu .> 0))"); flush(stdout)
        
        if all(num_planes_cpu .== 0)
            println("    [Hull] WARNING: No hull planes computed, skipping hull fill"); flush(stdout)
            continue
        end
        
        # Build obstacle ID in tensor for the kernel's val2 check
        # We need to insert the combined obstacle as a temporary mask
        # Use a temporary slot
        temp_obs_id = out_id  # Reuse output slot temporarily
        temp_obs_ch = out_ch
        
        # Instead of inserting obstacle into tensor, we use the hull instruction's in2 to reference obstacles
        # For simplicity, insert combined obstacle as temp mask and reference it
        # Actually, we just create a single hull instruction with obstacle as in2
        
        # Create hull instruction  
        hull_inst = Instruction(
            OP_HALF_SPACE,
            Int32(-1), Int32(-1),  # in1 not used (hull test is self-contained)
            Int32(-1), Int32(-1),  # in2 = obstacles (handled via planes subtraction for now)
            out_id, out_ch,
            Float32(0), Float32(0), Float32(0), Float32(0)
        )
        
        # Execute hull kernel with planes
        execute_vm_kernel_with_planes!(
            TM_VM.backend, TM_VM.tensor, 
            [hull_inst], planes_table, num_planes_arr,
            x_dim=dims[1], y_dim=dims[2], z_dim=dims[3]
        )
        
        # Post-process: subtract obstacles on GPU if any
        if !isempty(obstacle_ids)
            obs_insts = Instruction[]
            for obs in obstacle_ids
                push!(obs_insts, Instruction(
                    OP_EXCLUDE,
                    out_id, out_ch,
                    Int32(obs["id"]), Int32(obs["ch"]),
                    out_id, out_ch,
                    0f0, 0f0, 0f0, 0f0, 0f0, 0f0
                ))
            end
            execute_vm_kernel!(TM_VM.backend, TM_VM.tensor, obs_insts, x_dim=dims[1], y_dim=dims[2], z_dim=dims[3])
        end
        
        if !keep_all
            println("    [Hull] Running GPU CC extraction..."); flush(stdout)
            execute_cc_vm(out_id, out_ch)
        end
        
        # For logging, count remaining (optional, but keep it lightweight)
        # out_mask_gpu = extract_mask_vm(TM_VM.backend, TM_VM.tensor, out_id, out_ch, dims)
        # out_mask_cpu = adapt(CPU(), out_mask_gpu)
        # println("    [Hull] Filled $(count(out_mask_cpu .> 0)) voxels (keep_all=$keep_all)"); flush(stdout)

    end
    
    return "Executed hull batch"
end

# Clear a specific mask bit from the tensor
@kernel function clear_mask_vm_kernel!(tensor, mask_id, channel_idx, dims)
    I, J, K = @index(Global, NTuple)
    if I <= dims[1] && J <= dims[2] && K <= dims[3]
        tensor[I, J, K, channel_idx] &= ~(UInt32(1) << mask_id)
    end
end

function clear_mask_vm(backend, tensor, mask_id, channel_idx, dims)
    clear_mask_vm_kernel!(backend, 256)(tensor, UInt32(mask_id), Int32(channel_idx), dims, ndrange=dims)
    KernelAbstractions.synchronize(backend)
end

function execute_cc_vm(mask_id, mask_ch)
    dims = TM_VM.dims
    x_dim, y_dim, z_dim = dims
    backend = TM_VM.backend
    
    labels_A = TM_VM.cc_labels_A
    labels_B = TM_VM.cc_labels_B
    tensor = TM_VM.tensor
    
    # Init
    cc_init_kernel!(backend, 256)(labels_A, tensor, UInt32(mask_id), Int32(mask_ch), Int32(x_dim), Int32(y_dim), Int32(z_dim), ndrange=dims)
    KernelAbstractions.synchronize(backend)
    
    curr_labels = labels_A
    next_labels = labels_B
    
    for iters in 1:12
        cc_local_propagate!(backend, 256)(next_labels, curr_labels, tensor, UInt32(mask_id), Int32(mask_ch), Int32(x_dim), Int32(y_dim), Int32(z_dim), ndrange=dims)
        KernelAbstractions.synchronize(backend)
        
        cc_pointer_jump!(backend, 256)(curr_labels, next_labels, tensor, UInt32(mask_id), Int32(mask_ch), Int32(x_dim), Int32(y_dim), Int32(z_dim), ndrange=dims)
        KernelAbstractions.synchronize(backend)
    end
    
    labels_cpu = adapt(CPU(), curr_labels)
    counts = Dict{Int32, Int}()
    for l in labels_cpu
        if l > 0
            counts[l] = get(counts, l, 0) + 1
        end
    end
    
    if isempty(counts)
        return
    end
    
    best_label = -1
    best_count = -1
    for (l, c) in counts
        if c > best_count
            best_count = c
            best_label = l
        end
    end
    
    clear_mask_vm(backend, tensor, mask_id, mask_ch, dims)
    
    extract_largest_cc!(backend, 256)(tensor, curr_labels, Int32(best_label), UInt32(mask_id), Int32(mask_ch), Int32(x_dim), Int32(y_dim), Int32(z_dim), ndrange=dims)
    KernelAbstractions.synchronize(backend)
end

# Simple seed-based flood fill on CPU (to be replaced by JFA on GPU later)
function jfa_seed_propagation(mask::Array{UInt8,3})
    dims = size(mask)
    
    # Find Z range with free-space voxels
    z_min = 0
    z_max = 0
    for z in 1:dims[3]
        if any(view(mask, :, :, z) .> 0)
            if z_min == 0
                z_min = z
            end
            z_max = z
        end
    end
    if z_min == 0
        return mask
    end
    
    # Find optimal seed slice: slice where the centroid of free space is farthest from boundary
    # (approximation of Python's EDT-based seed selection)
    best_z = -1
    best_score = -1.0
    best_seed = (0, 0)
    
    # Sample every 4th slice for speed, then refine around the best
    step = max(1, (z_max - z_min) ÷ 50)
    
    for z in z_min:step:z_max
        slice = view(mask, :, :, z)
        nz = count(slice .> 0)
        if nz < 10
            continue
        end
        
        # Find centroid of free space
        sx, sy, cnt = 0, 0, 0
        for x in 1:4:dims[1], y in 1:4:dims[2]
            if slice[x, y] > 0
                sx += x; sy += y; cnt += 1
            end
        end
        if cnt == 0 continue end
        cx = sx ÷ cnt
        cy = sy ÷ cnt
        
        # Ensure centroid is in free space; if not, find nearest
        if mask[cx, cy, z] == 0
            found = false
            for r in 1:30
                for dx in -r:r, dy in -r:r
                    nx, ny = cx + dx, cy + dy
                    if nx >= 1 && nx <= dims[1] && ny >= 1 && ny <= dims[2] && mask[nx, ny, z] > 0
                        cx, cy = nx, ny; found = true; break
                    end
                end
                if found break end
            end
            if !found continue end
        end
        
        # Approximate EDT: distance from centroid to boundary (4-dir)
        d_left = 0; while cx - d_left - 1 >= 1 && mask[cx - d_left - 1, cy, z] > 0; d_left += 1; end
        d_right = 0; while cx + d_right + 1 <= dims[1] && mask[cx + d_right + 1, cy, z] > 0; d_right += 1; end
        d_up = 0; while cy - d_up - 1 >= 1 && mask[cx, cy - d_up - 1, z] > 0; d_up += 1; end
        d_down = 0; while cy + d_down + 1 <= dims[2] && mask[cx, cy + d_down + 1, z] > 0; d_down += 1; end
        
        min_dist = Float64(min(d_left, d_right, d_up, d_down))
        if min_dist > best_score
            best_score = min_dist
            best_z = z
            best_seed = (cx, cy)
        end
    end
    
    if best_z == -1 || best_seed == (0, 0)
        return mask
    end
    
    println("    [Seed] Best slice: $best_z, clearance: $best_score px, seed: $best_seed")
    flush(stdout)
    
    result = zeros(UInt8, dims)
    
    # Solve a slice: 2D flood fill from seed within the free-space mask
    function solve_slice!(result, mask, z, seed_mask_2d)
        limit = view(mask, :, :, z)
        # Valid seed = seed from previous slice AND current slice's free space
        valid_seed = seed_mask_2d .& (limit .> 0)
        if count(valid_seed) == 0
            return false
        end
        
        # 2D BFS flood fill from valid_seed within limit (morphological reconstruction)
        visited = falses(dims[1], dims[2])
        queue = Tuple{Int,Int}[]
        for x in 1:dims[1], y in 1:dims[2]
            if valid_seed[x, y]
                push!(queue, (x, y))
                visited[x, y] = true
                result[x, y, z] = 1
            end
        end
        
        while !isempty(queue)
            x, y = popfirst!(queue)
            for (dx, dy) in ((1,0), (-1,0), (0,1), (0,-1))
                nx, ny = x + dx, y + dy
                if nx >= 1 && nx <= dims[1] && ny >= 1 && ny <= dims[2]
                    if !visited[nx, ny] && mask[nx, ny, z] > 0
                        visited[nx, ny] = true
                        result[nx, ny, z] = 1
                        push!(queue, (nx, ny))
                    end
                end
            end
        end
        
        return count(view(result, :, :, z) .> 0) > 0
    end
    
    # Solve seed slice first
    seed_2d = falses(dims[1], dims[2])
    seed_2d[best_seed[1], best_seed[2]] = true
    solve_slice!(result, mask, best_z, seed_2d)
    
    # Propagate upward (z+1, z+2, ...)
    for z in (best_z + 1):z_max
        prev_slice = view(result, :, :, z - 1) .> 0
        if !solve_slice!(result, mask, z, prev_slice)
            break
        end
    end
    
    # Propagate downward (z-1, z-2, ...)
    for z in (best_z - 1):-1:z_min
        prev_slice = view(result, :, :, z + 1) .> 0
        if !solve_slice!(result, mask, z, prev_slice)
            break
        end
    end
    
    return result
end

function parse_instruction(op)
    return Instruction(
        UInt32(op["opcode"]),
        Int32(get(op, "in1_id", -1)),
        Int32(get(op, "in1_channel", -1)),
        Int32(get(op, "in2_id", -1)),
        Int32(get(op, "in2_channel", -1)),
        Int32(get(op, "out_id", -1)),
        Int32(get(op, "out_channel", -1)),
        Float32(get(op, "farg1", 0.0)),
        Float32(get(op, "farg2", 0.0)),
        Float32(get(op, "farg3", 0.0)),
        Float32(get(op, "farg4", 0.0))
    )
end


include("../gpu_engine/kernels_morphology.jl")
include("../gpu_engine/kernels_convexhull.jl")

function execute_spatial_rule_vm(rule_type, params, in_id, in_ch, out_id, out_ch)
    t_total = time_ns()
    
    # Extract to CPU
    in_mask_cpu = adapt(CPU(), extract_mask_vm(TM_VM.backend, TM_VM.tensor, in_id, in_ch, TM_VM.dims))
    out_mask_cpu = zeros(UInt8, size(in_mask_cpu))
    
    t_extract = time_ns()
    
    if rule_type == "DistanceExpansion"
        sp_x = Float64(TM_VM.sp[1])
        sp_y = Float64(TM_VM.sp[2])
        sp_z = Float64(TM_VM.sp[3])
        dist = Float64(get(params, "distance_mm", 0.0))
        margins = Dict("left" => dist, "right" => dist, "anterior" => dist, "posterior" => dist, "superior" => dist, "inferior" => dist)
        run_anisotropic_dilation!(TM_VM.backend, out_mask_cpu, in_mask_cpu, sp_x, sp_y, sp_z, margins)
        t_compute = time_ns()
        println("[BENCH] DistanceExpansion: extract=$(round(Int, (t_extract-t_total)/1e6))ms dilation=$(round(Int, (t_compute-t_extract)/1e6))ms")
        flush(stdout)
    elseif rule_type == "AnisotropicMargin"
        sp_x = Float64(TM_VM.sp[1])
        sp_y = Float64(TM_VM.sp[2])
        sp_z = Float64(TM_VM.sp[3])
        margins = get(params, "margins_mm", Dict())
        run_anisotropic_dilation!(TM_VM.backend, out_mask_cpu, in_mask_cpu, sp_x, sp_y, sp_z, margins)
        t_compute = time_ns()
        println("[BENCH] AnisotropicMargin: extract=$(round(Int, (t_extract-t_total)/1e6))ms dilation=$(round(Int, (t_compute-t_extract)/1e6))ms")
        flush(stdout)
    elseif rule_type == "VolumetricBoundary2D"
        # NEW: 2-phase kernel approach
        deps = get(params, "deps_ids", [])
        medial_ids = get(params, "medial_ids", [])
        lateral_ids = get(params, "lateral_ids", [])
        growth_mode = get(params, "growth_mode", "default")
        keep_all = get(params, "keep_all_components", false)
        
        if growth_mode == "sideways" && !isempty(medial_ids) && !isempty(lateral_ids)
            medial_arrs = []
            for m in medial_ids
                push!(medial_arrs, adapt(CPU(), extract_mask_vm(TM_VM.backend, TM_VM.tensor, m["id"], m["ch"], TM_VM.dims)))
            end
            lateral_arrs = []
            for l in lateral_ids
                push!(lateral_arrs, adapt(CPU(), extract_mask_vm(TM_VM.backend, TM_VM.tensor, l["id"], l["ch"], TM_VM.dims)))
            end
            t_deps = time_ns()
            run_volumetric_boundary_2d!(out_mask_cpu, [], medial_arrs, lateral_arrs, "sideways", keep_all)
            t_vb2d = time_ns()
            println("[BENCH] VB2D-sideways: extract=$(round(Int, (t_extract-t_total)/1e6))ms deps=$(round(Int, (t_deps-t_extract)/1e6))ms vb2d=$(round(Int, (t_vb2d-t_deps)/1e6))ms")
            flush(stdout)
        else
            # Default mode: per-slice 2D convex hull with boundary subtraction + seed propagation
            # Python: hull_2d = convex_hull_image(slice_boundary)
            #         free_space = hull_2d & ~global_obstacles  (where obstacles = boundary structures)
            #         result = reconstruction(seed, free_space, method='dilation')
            combined_boundary = zeros(UInt8, size(out_mask_cpu))
            for d in deps
                arr = adapt(CPU(), extract_mask_vm(TM_VM.backend, TM_VM.tensor, d["id"], d["ch"], TM_VM.dims))
                combined_boundary .|= arr
            end
            t_deps = time_ns()
            
            # Per-slice 2D convex hull directly on the boundary mask
            planes_table, num_planes_arr = compute_planes_table_from_mask(combined_boundary)
            t_hull = time_ns()
            
            n_hull_slices = count(num_planes_arr .> 0)
            println("    [VB2D-Hull] Boundary voxels: $(count(combined_boundary .> 0)), hull slices: $n_hull_slices")
            flush(stdout)
            
            t_fill = t_hull
            t_seed = t_hull
            
            if n_hull_slices > 0
                fill_hull_cpu!(out_mask_cpu, planes_table, num_planes_arr)
                t_fill = time_ns()
                
                # Check for per_layer_margin logic
                per_layer = get(params, "per_layer_margin", false)
                ap_margin = get(params, "ap_margin_only", false)
                posterior_offset = Float64(get(params, "posterior_offset_mm", 0.0))
                
                posterior_ids = get(params, "posterior_ids", [])
                if !isempty(posterior_ids)
                    posterior_mask = zeros(UInt8, size(out_mask_cpu))
                    for p in posterior_ids
                        arr = adapt(CPU(), extract_mask_vm(TM_VM.backend, TM_VM.tensor, p["id"], p["ch"], TM_VM.dims))
                        posterior_mask .|= arr
                    end
                    
                    if per_layer
                        voxel_offset = round(Int, posterior_offset / TM_VM.sp[2])
                        
                        # We must dilate posterior_mask and intersect with hull
                        # If ap_margin_only, we only dilate along Y!
                        if voxel_offset > 0
                            dilated_post = zeros(UInt8, size(out_mask_cpu))
                            for z in 1:size(out_mask_cpu, 3), x in 1:size(out_mask_cpu, 1)
                                y_inds = findall(posterior_mask[x, :, z] .> 0)
                                if !isempty(y_inds)
                                    min_y = minimum(y_inds)
                                    max_y = maximum(y_inds)
                                    if ap_margin
                                        start_y = max(1, min_y - voxel_offset)
                                        end_y = min(size(out_mask_cpu, 2), max_y + voxel_offset)
                                        dilated_post[x, start_y:end_y, z] .= 1
                                    else
                                        # Full 2D dilation fallback (simplification: just do 1D for now since Python mostly uses ap_margin_only for presacral)
                                        start_y = max(1, min_y - voxel_offset)
                                        end_y = min(size(out_mask_cpu, 2), max_y + voxel_offset)
                                        dilated_post[x, start_y:end_y, z] .= 1
                                    end
                                end
                            end
                            out_mask_cpu .&= dilated_post
                        else
                            out_mask_cpu .&= posterior_mask
                        end
                    else
                        voxel_offset_y = round(Int, posterior_offset / TM_VM.sp[2])
                        min_depth_vox = round(Int, Float64(get(params, "min_depth_mm", 0.0)) / TM_VM.sp[2])
                        y_increasing = get(params, "y_increasing", true)
                        
                        for z in 1:size(out_mask_cpu, 3)
                            slice_post = posterior_mask[:, :, z]
                            if any(slice_post .> 0)
                                y_inds = findall(any(slice_post .> 0, dims=1)[1, :])
                                if isempty(y_inds) continue end
                                
                                slice_hull = out_mask_cpu[:, :, z]
                                slice_ant = (slice_hull .> 0) .& (slice_post .== 0)
                                has_ant = false
                                ant_min_y, ant_max_y = 1, 1
                                if any(slice_ant)
                                    ant_y_inds = findall(any(slice_ant, dims=1)[1, :])
                                    if !isempty(ant_y_inds)
                                        ant_min_y = minimum(ant_y_inds)
                                        ant_max_y = maximum(ant_y_inds)
                                        has_ant = true
                                    end
                                end
                                
                                if y_increasing
                                    vessel_front_y = minimum(y_inds)
                                    limit_y = vessel_front_y + voxel_offset_y
                                    if has_ant && min_depth_vox > 0
                                        limit_y = max(limit_y, ant_max_y + min_depth_vox)
                                    end
                                    if has_ant && limit_y > ant_min_y + 2
                                        limit_y = max(1, limit_y)
                                        out_mask_cpu[:, limit_y:end, z] .= 0
                                    end
                                else
                                    vessel_front_y = maximum(y_inds)
                                    limit_y = vessel_front_y - voxel_offset_y
                                    if has_ant && min_depth_vox > 0
                                        limit_y = min(limit_y, ant_min_y - min_depth_vox)
                                    end
                                    if has_ant && limit_y < ant_max_y - 2
                                        limit_y = min(size(out_mask_cpu, 2), limit_y + 1)
                                        out_mask_cpu[:, 1:limit_y, z] .= 0
                                    end
                                end
                            end
                        end
                    end
                end
                
                # We do NOT subtract combined_boundary! Python subtracts 'exclude' masks, which are handled in post-processing.
                # If seed propagation leaks without obstacles, we might need to handle it, but for Abdominal_Presacral, 
                # per_layer_margin restricts it to the base mask anyway.
                
                # Seed propagation was historically here, but Python does not do this before post-processing.
                # LCC is applied in apply_post_processing_vm.
            end
            t_seed = time_ns()
            
            println("[BENCH] VB2D-hull: extract=$(round(Int, (t_extract-t_total)/1e6))ms deps=$(round(Int, (t_deps-t_extract)/1e6))ms hull_compute=$(round(Int, (t_hull-t_deps)/1e6))ms fill=$(round(Int, (t_fill-t_hull)/1e6))ms seed=$(round(Int, (t_seed-t_fill)/1e6))ms")
            flush(stdout)
        end
        
        
        
    elseif rule_type == "ConvexHullBridging"
        deps = get(params, "deps_ids", [])
        if length(deps) >= 2
            combined_edges = zeros(UInt8, size(out_mask_cpu))
            z_mask = ones(Bool, size(out_mask_cpu, 3))
            
            for d in deps
                arr = adapt(CPU(), extract_mask_vm(TM_VM.backend, TM_VM.tensor, d["id"], d["ch"], TM_VM.dims))
                edges = extract_edges_cpu(arr)
                combined_edges .|= edges
                
                z_sums = dropdims(sum(arr, dims=(1,2)), dims=(1,2))
                z_mask .&= (z_sums .> 0)
            end
            
            for z in 1:size(combined_edges, 3)
                if !z_mask[z]
                    combined_edges[:, :, z] .= 0
                end
            end
            t_edge = time_ns()
            
            planes_table, num_planes_arr = compute_planes_table_from_mask(combined_edges)
            t_hull = time_ns()
            
            n_hull_slices = count(num_planes_arr .> 0)
            println("    [CHB-Hull] Combined edge voxels: $(count(combined_edges .> 0)), hull slices: $n_hull_slices")
            flush(stdout)
            
            t_fill = t_hull
            if n_hull_slices > 0
                fill_hull_cpu!(out_mask_cpu, planes_table, num_planes_arr)
                t_fill = time_ns()
            end
            t_sub = time_ns()
            
            println("[BENCH] CHB: extract=$(round(Int, (t_extract-t_total)/1e6))ms edge=$(round(Int, (t_edge-t_extract)/1e6))ms hull_compute=$(round(Int, (t_hull-t_edge)/1e6))ms fill=$(round(Int, (t_fill-t_hull)/1e6))ms")
            flush(stdout)
        end
        
    elseif rule_type == "PresacralAnteriorCustom"
        dist_mm = Float32(get(params, "distance_mm", 10.0))
        out_type = get(params, "output_type", "zone")
        in_id = get(params, "in_id", -1)
        in_ch = get(params, "in_ch", 1)
        
        in_arr = adapt(CPU(), extract_mask_vm(TM_VM.backend, TM_VM.tensor, in_id, in_ch, TM_VM.dims))
        dist_voxels = round(Int, dist_mm / TM_VM.sp[2])
        
        for z in 1:size(in_arr, 3)
            for x in 1:size(in_arr, 1)
                min_y = 0
                for y in 1:size(in_arr, 2)
                    if in_arr[x, y, z] > 0
                        min_y = y
                        break
                    end
                end
                
                if min_y > 0
                    start_y = max(1, min_y - dist_voxels)
                    if out_type == "front_line"
                        out_mask_cpu[x, min_y, z] = 1
                    elseif out_type == "moved_line"
                        out_mask_cpu[x, start_y, z] = 1
                    else
                        for y in start_y:min_y-1
                            out_mask_cpu[x, y, z] = 1
                        end
                    end
                end
            end
        end
        
        
    elseif rule_type == "VolumetricConvexHull3D"
        edge_mask = extract_edges_cpu(in_mask_cpu)
        t_edge = time_ns()
        
        planes_table, num_planes_arr = compute_planes_table_from_mask(edge_mask)
        t_hull = time_ns()
        
        n_hull_slices = count(num_planes_arr .> 0)
        println("    [VCH3D] Edge voxels: $(count(edge_mask .> 0)), hull slices: $n_hull_slices")
        flush(stdout)
        
        t_fill = t_hull
        if n_hull_slices > 0
            fill_hull_cpu!(out_mask_cpu, planes_table, num_planes_arr)
            t_fill = time_ns()
        end
        
        println("[BENCH] VCH3D: extract=$(round(Int, (t_extract-t_total)/1e6))ms edge=$(round(Int, (t_edge-t_extract)/1e6))ms hull_compute=$(round(Int, (t_hull-t_edge)/1e6))ms fill=$(round(Int, (t_fill-t_hull)/1e6))ms")
        flush(stdout)
    end
    
    # Insert back to GPU
    t_pre_insert = time_ns()
    in_mask_gpu = adapt(TM_VM.backend, out_mask_cpu)
    insert_mask_vm(TM_VM.backend, TM_VM.tensor, in_mask_gpu, out_id, out_ch, TM_VM.dims)
    t_insert = time_ns()
    
    println("[BENCH] $rule_type TOTAL: $(round(Int, (t_insert-t_total)/1e6))ms (insert=$(round(Int, (t_insert-t_pre_insert)/1e6))ms)")
    flush(stdout)
    return "Executed $rule_type"
end

function keep_largest_component_cpu!(mask::Array{UInt8, 3})
    dims = size(mask)
    visited = zeros(Bool, dims)
    max_size = 0
    max_seed = (0, 0, 0)
    
    # Pass 1: Find largest component
    for z in 1:dims[3], y in 1:dims[2], x in 1:dims[1]
        if mask[x, y, z] > 0 && !visited[x, y, z]
            q = [(x, y, z)]
            visited[x, y, z] = true
            sz = 0
            head = 1
            while head <= length(q)
                cx, cy, cz = q[head]
                head += 1
                sz += 1
                
                for (dx, dy, dz) in ((1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1))
                    nx, ny, nz = cx+dx, cy+dy, cz+dz
                    if 1 <= nx <= dims[1] && 1 <= ny <= dims[2] && 1 <= nz <= dims[3]
                        if mask[nx, ny, nz] > 0 && !visited[nx, ny, nz]
                            visited[nx, ny, nz] = true
                            push!(q, (nx, ny, nz))
                        end
                    end
                end
            end
            if sz > max_size
                max_size = sz
                max_seed = (x, y, z)
            end
        end
    end
    
    # Pass 2: Keep only largest component
    if max_size > 0
        new_mask = zeros(UInt8, dims)
        q = [max_seed]
        new_mask[max_seed...] = 1
        head = 1
        while head <= length(q)
            cx, cy, cz = q[head]
            head += 1
            for (dx, dy, dz) in ((1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1))
                nx, ny, nz = cx+dx, cy+dy, cz+dz
                if 1 <= nx <= dims[1] && 1 <= ny <= dims[2] && 1 <= nz <= dims[3]
                    if mask[nx, ny, nz] > 0 && new_mask[nx, ny, nz] == 0
                        new_mask[nx, ny, nz] = 1
                        push!(q, (nx, ny, nz))
                    end
                end
            end
        end
        mask .= new_mask
    end
end

function apply_post_processing_vm(mask_id, mask_ch, params_dict)
    t_start = time_ns()
    out_mask_cpu = adapt(CPU(), extract_mask_vm(TM_VM.backend, TM_VM.tensor, mask_id, mask_ch, TM_VM.dims))
    
    voxels_before = count(out_mask_cpu .> 0)
    
    # Apply intersect_with
    for iw in get(params_dict, "intersect_with_ids", [])
        iw_arr = adapt(CPU(), extract_mask_vm(TM_VM.backend, TM_VM.tensor, iw["id"], iw["ch"], TM_VM.dims))
        out_mask_cpu[iw_arr .== 0] .= 0
    end
    
    # Apply fat intersection
    if get(params_dict, "intersect_with_fat", false) && get(params_dict, "fat_id", -1) >= 0
        fat_arr = adapt(CPU(), extract_mask_vm(TM_VM.backend, TM_VM.tensor, params_dict["fat_id"], params_dict["fat_ch"], TM_VM.dims))
        out_mask_cpu[fat_arr .== 0] .= 0
    end
    
    # Apply exclusions
    for ex in get(params_dict, "exclude_ids", [])
        v_before = count(out_mask_cpu .> 0)
        ex_arr = adapt(CPU(), extract_mask_vm(TM_VM.backend, TM_VM.tensor, ex["id"], ex["ch"], TM_VM.dims))
        ex_voxels = count(ex_arr .> 0)
        out_mask_cpu[ex_arr .> 0] .= 0
        v_after = count(out_mask_cpu .> 0)
        println("    Julia: [VM Post-Processing] Excluded $(ex["id"]) (ch $(ex["ch"])): $v_before -> $v_after (ex_arr voxels: $ex_voxels)")
    end
    
    # Apply constraints
    for c in get(params_dict, "constraints", [])
        v_before = count(out_mask_cpu .> 0)
        c_type = c["constraint_type"]
        c_part = c["boundary_part"]
        lm_id = c["landmark_id"]
        lm_ch = c["landmark_ch"]
        
        lm_arr = adapt(CPU(), extract_mask_vm(TM_VM.backend, TM_VM.tensor, lm_id, lm_ch, TM_VM.dims))
        
        # Compute percentile bounds to match Python
        x_sums = dropdims(sum(lm_arr, dims=(2,3)), dims=(2,3))
        y_sums = dropdims(sum(lm_arr, dims=(1,3)), dims=(1,3))
        z_sums = dropdims(sum(lm_arr, dims=(1,2)), dims=(1,2))
        
        total_voxels = sum(z_sums)
        if total_voxels == 0 continue end
        target_v = c_part == "min" ? total_voxels * 0.01 : (c_part == "center" ? total_voxels * 0.50 : total_voxels * 0.99)
        
        ref_x, ref_y, ref_z = 1, 1, 1
        
        cum = 0; for i in 1:length(x_sums); cum += x_sums[i]; if cum >= target_v; ref_x = i; break; end; end
        cum = 0; for i in 1:length(y_sums); cum += y_sums[i]; if cum >= target_v; ref_y = i; break; end; end
        cum = 0; for i in 1:length(z_sums); cum += z_sums[i]; if cum >= target_v; ref_z = i; break; end; end
        
        z_increasing = get(params_dict, "z_increasing", true)
        x_increasing = get(params_dict, "x_increasing", true)
        y_increasing = get(params_dict, "y_increasing", true)
        
        println("  [VM Constraint] Applied $c_type on LM $lm_id ($c_part) -> ref_x=$ref_x, ref_y=$ref_y, ref_z=$ref_z. Z_inc=$z_increasing")
        
        slice_wise = get(c, "slice_wise", false)
        
        if slice_wise && (c_type == "AnteriorTo" || c_type == "PosteriorTo" || c_type == "LeftOf" || c_type == "RightOf")
            for z in 1:size(out_mask_cpu, 3)
                lm_slice = lm_arr[:, :, z]
                if !any(lm_slice .> 0) continue end
                
                if c_type == "AnteriorTo" || c_type == "PosteriorTo"
                    y_inds = findall(any(lm_slice .> 0, dims=1)[1, :])
                    if isempty(y_inds) continue end
                    
                    if c_part == "min"
                        s_ref = minimum(y_inds)
                    elseif c_part == "max"
                        s_ref = maximum(y_inds)
                    else
                        # Match Python: (min + max) // 2 (midpoint of extent, NOT centroid)
                        s_ref = (minimum(y_inds) + maximum(y_inds)) ÷ 2
                    end
                    
                    if c_type == "AnteriorTo"
                        if y_increasing
                            out_mask_cpu[:, s_ref:end, z] .= 0
                        else
                            out_mask_cpu[:, 1:s_ref, z] .= 0
                        end
                    else
                        if y_increasing
                            out_mask_cpu[:, 1:s_ref, z] .= 0
                        else
                            out_mask_cpu[:, s_ref:end, z] .= 0
                        end
                    end
                else
                    x_inds = findall(any(lm_slice .> 0, dims=2)[:, 1])
                    if isempty(x_inds) continue end
                    
                    if c_part == "min"
                        s_ref = minimum(x_inds)
                    elseif c_part == "max"
                        s_ref = maximum(x_inds)
                    else
                        # Match Python: (min + max) // 2 (midpoint of extent, NOT centroid)
                        s_ref = (minimum(x_inds) + maximum(x_inds)) ÷ 2
                    end
                    
                    if c_type == "LeftOf"
                        if x_increasing
                            out_mask_cpu[1:s_ref, :, z] .= 0
                        else
                            out_mask_cpu[s_ref:end, :, z] .= 0
                        end
                    else
                        if x_increasing
                            out_mask_cpu[s_ref:end, :, z] .= 0
                        else
                            out_mask_cpu[1:s_ref, :, z] .= 0
                        end
                    end
                end
            end
        else
            if c_type == "InferiorTo"
                # z_increasing=True means +z is Superior. We want to keep Inferior (-z), so clear Superior (+z).
                # +1 to make boundary inclusive (keep the ref_z slice itself, matching Python pipeline)
                if z_increasing
                    if ref_z + 1 <= size(out_mask_cpu, 3)
                        out_mask_cpu[:, :, ref_z+1:end] .= 0
                    end
                else
                    if ref_z - 1 >= 1
                        out_mask_cpu[:, :, 1:ref_z-1] .= 0
                    end
                end
            elseif c_type == "SuperiorTo"
                # z_increasing=True means +z is Superior. We want to keep Superior (+z), so clear Inferior (-z).
                # -1 to make boundary inclusive (keep the ref_z slice itself, matching Python pipeline)
                if z_increasing
                    if ref_z - 1 >= 1
                        out_mask_cpu[:, :, 1:ref_z-1] .= 0
                    end
                else
                    if ref_z + 1 <= size(out_mask_cpu, 3)
                        out_mask_cpu[:, :, ref_z+1:end] .= 0
                    end
                end
            elseif c_type == "LeftOf"
                # Keep left (+x if x_increasing). Clear right.
                if x_increasing
                    out_mask_cpu[1:ref_x, :, :] .= 0
                else
                    out_mask_cpu[ref_x:end, :, :] .= 0
                end
            elseif c_type == "RightOf"
                # Keep right (-x if x_increasing). Clear left.
                if x_increasing
                    out_mask_cpu[ref_x:end, :, :] .= 0
                else
                    out_mask_cpu[1:ref_x, :, :] .= 0
                end
            elseif c_type == "AnteriorTo"
                # Keep anterior (-y if y_increasing (LPS)). Clear posterior.
                if y_increasing
                    out_mask_cpu[:, ref_y:end, :] .= 0
                else
                    out_mask_cpu[:, 1:ref_y, :] .= 0
                end
            elseif c_type == "PosteriorTo"
                # Keep posterior (+y if y_increasing). Clear anterior.
                if y_increasing
                    out_mask_cpu[:, 1:ref_y, :] .= 0
                else
                    out_mask_cpu[:, ref_y:end, :] .= 0
                end
            end
        end
        
        v_after = count(out_mask_cpu .> 0)
        println("  [VM Constraint] $c_type on LM $lm_id: $v_before -> $v_after")
    end
    
    # Keep largest component
    if get(params_dict, "keep_all_components", false) == false
        keep_largest_component_cpu!(out_mask_cpu)
        v_after_lcc = count(out_mask_cpu .> 0)
        println("  [VM Post-Processing] LCC: $voxels_before -> $v_after_lcc")
    end
    
    voxels_after = count(out_mask_cpu .> 0)
    println("  [VM Post-Processing] Voxels: $voxels_before -> $voxels_after")
    
    in_mask_gpu = adapt(TM_VM.backend, out_mask_cpu)
    clear_mask_vm(TM_VM.backend, TM_VM.tensor, mask_id, mask_ch, TM_VM.dims)
    insert_mask_vm(TM_VM.backend, TM_VM.tensor, in_mask_gpu, mask_id, mask_ch, TM_VM.dims)
    
    t_end = time_ns()
    println("[BENCH] apply_post_processing_vm: $(round(Int, (t_end-t_start)/1e6))ms")
    flush(stdout)
    return "Post-processing applied"
end

# CPU-side edge extraction (6-connectivity)
function extract_edges_cpu(mask::Array{UInt8,3})
    dims = size(mask)
    edges = zeros(UInt8, dims)
    
    Threads.@threads for z in 1:dims[3]
        for y in 1:dims[2], x in 1:dims[1]
            if mask[x, y, z] == 0
                continue
            end
            is_edge = false
            if x > 1 && mask[x-1, y, z] == 0
                is_edge = true
            elseif x == 1
                is_edge = true
            end
            if !is_edge && x < dims[1] && mask[x+1, y, z] == 0
                is_edge = true
            end
            if !is_edge && y > 1 && mask[x, y-1, z] == 0
                is_edge = true
            elseif !is_edge && y == 1
                is_edge = true
            end
            if !is_edge && y < dims[2] && mask[x, y+1, z] == 0
                is_edge = true
            end
            if !is_edge && z > 1 && mask[x, y, z-1] == 0
                is_edge = true
            elseif !is_edge && z == 1
                is_edge = true
            end
            if !is_edge && z < dims[3] && mask[x, y, z+1] == 0
                is_edge = true
            end
            if is_edge
                edges[x, y, z] = 1
            end
        end
    end
    
    return edges
end

# CPU-side hull fill using half-space planes (mirrors kernel logic)
function fill_hull_cpu!(out_mask::Array{UInt8,3}, planes_table::Array{Float32,3}, num_planes_arr::Array{Int32,1})
    dims = size(out_mask)
    
    Threads.@threads for z in 1:dims[3]
        np = num_planes_arr[z]
        if np == 0
            continue
        end
        for y in 1:dims[2], x in 1:dims[1]
            inside = true
            for p in 1:np
                A = planes_table[p, 1, z]
                B = planes_table[p, 2, z]
                D = planes_table[p, 3, z]
                if A * Float32(x) + B * Float32(y) + D > 1f-5
                    inside = false
                    break
                end
            end
            if inside
                out_mask[x, y, z] = 1
            end
        end
    end
end
