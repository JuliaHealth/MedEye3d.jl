using JSON
using HDF5
using CUDA
using KernelAbstractions
using Adapt
using LinearAlgebra

include("mega_fused_dag_kernel_v2.jl")
include("../gpu_engine/kernels_convexhull.jl")
include("hull_planes.jl")
using .HullPlanes
include("jfa.jl")

mutable struct TensorManagerMegaV2
    tensor_in::Any   # Integer tensor (UInt16, 4D: C, X, Y, Z) 
    tensor_out::Any  # Binary tensor (UInt8, 4D: C, X, Y, Z) for the current step's outputs
    backend::Any
    sp::Any
    dims::Tuple{Int, Int, Int}
    
    dt_tensor::Any
    jfa_seeds_A::Any
    jfa_seeds_B::Any
end

const TM_MEGA2 = TensorManagerMegaV2(nothing, nothing, nothing, (0,0,0), (0,0,0), nothing, nothing, nothing)

function free_mega2_tensors!()
    for field in (:tensor_in, :tensor_out, :dt_tensor, :jfa_seeds_A, :jfa_seeds_B)
        arr = getfield(TM_MEGA2, field)
        if arr !== nothing
            try CUDA.unsafe_free!(arr) catch end
            setfield!(TM_MEGA2, field, nothing)
        end
    end
    GC.gc(true)
    if CUDA.functional()
        CUDA.reclaim()
        dev = CUDA.device()
        pool = CUDA.memory_pool(dev)
        CUDA.CUDACore.trim(pool, 0)
    end
end

function init_mega2_tensor(size_x, size_y, size_z, num_in_channels, num_out_channels, sp_x, sp_y, sp_z; use_gpu=true)
    backend = use_gpu && CUDA.functional() ? CUDABackend() : CPU()
    
    # Use CUDA.unsafe_free! for immediate, deterministic GPU memory release.
    if use_gpu && CUDA.functional()
        free_mega2_tensors!()
    end
    if use_gpu && CUDA.functional()
        tensor_in = CUDA.zeros(UInt16, num_in_channels, size_x, size_y, size_z)
        tensor_out = CUDA.zeros(UInt8, num_out_channels, size_x, size_y, size_z)
        dt_tensor = CUDA.zeros(UInt16, 3, 2, size_x, size_y, size_z)
        jfa_seeds_A = CUDA.zeros(UInt16, 3, size_x, size_y, size_z)
        jfa_seeds_B = CUDA.zeros(UInt16, 3, size_x, size_y, size_z)
    else
        tensor_in = zeros(UInt16, num_in_channels, size_x, size_y, size_z)
        tensor_out = zeros(UInt8, num_out_channels, size_x, size_y, size_z)
        dt_tensor = zeros(UInt16, 3, 2, size_x, size_y, size_z)
        jfa_seeds_A = zeros(UInt16, 3, size_x, size_y, size_z)
        jfa_seeds_B = zeros(UInt16, 3, size_x, size_y, size_z)
    end
    
    TM_MEGA2.tensor_in = tensor_in
    TM_MEGA2.tensor_out = tensor_out
    TM_MEGA2.backend = backend
    TM_MEGA2.sp = (Float32(sp_x), Float32(sp_y), Float32(sp_z))
    TM_MEGA2.dims = (size_x, size_y, size_z)
    TM_MEGA2.dt_tensor = dt_tensor
    TM_MEGA2.jfa_seeds_A = jfa_seeds_A
    TM_MEGA2.jfa_seeds_B = jfa_seeds_B
    
    return "Initialized MegaV2 Tensor on $(typeof(backend))"
end




function load_mega2_tensor(path, is_input=true)
    # Load from HDF5 (h5py-compatible)
    # IMPORTANT: Julia HDF5.jl reverses dimensions vs h5py/numpy.
    # Python writes (C, X, Y, Z) → Julia reads (Z, Y, X, C) → we permute back to (C, X, Y, Z)
    arr = h5open(path, "r") do f
        raw = read(f["tensor"])
        permutedims(raw, (4, 1, 2, 3))  # (X, Y, Z, C) → (C, X, Y, Z)
    end
    arr_dev = adapt(TM_MEGA2.backend, arr)
    if is_input
        if TM_MEGA2.tensor_in !== nothing
            try CUDA.unsafe_free!(TM_MEGA2.tensor_in) catch end
        end
        TM_MEGA2.tensor_in = arr_dev
        # Update dims from actual loaded tensor size (may differ from init if mask sizes differ from CT)
        loaded_xyz = (size(arr, 2), size(arr, 3), size(arr, 4))
        if loaded_xyz != TM_MEGA2.dims
            println("[WARN] load_mega2_tensor: dims mismatch! init=$(TM_MEGA2.dims) loaded=$(loaded_xyz). Updating dims.")
            TM_MEGA2.dims = loaded_xyz
            flush(stdout)
        else
            println("[INFO] load_mega2_tensor: dims match $(TM_MEGA2.dims), $(size(arr, 1)) channels loaded.")
        end
    else
        if TM_MEGA2.tensor_out !== nothing
            try CUDA.unsafe_free!(TM_MEGA2.tensor_out) catch end
        end
        TM_MEGA2.tensor_out = arr_dev
    end
end

function save_mega2_tensor(path, is_input=true)
    arr_dev = is_input ? TM_MEGA2.tensor_in : TM_MEGA2.tensor_out
    arr = Array(arr_dev)
    arr_save = permutedims(arr, (1, 4, 3, 2))  # (C, X, Y, Z) -> (C, Z, Y, X)
    # Save as plain HDF5 (h5py-compatible), no compression for speed
    h5open(path, "w") do f
        f["tensor"] = arr_save
    end
end

# Find the most anterior point (smallest Y physical coord) in a packed input tensor channel
# Returns (px_mm, py_mm, pz_mm) using voxel-to-mm with origin=0
function most_anterior_point_from_tensor(ch, id_val)
    dims = TM_MEGA2.dims
    sp = TM_MEGA2.sp
    arr = adapt(CPU(), TM_MEGA2.tensor_in[ch, :, :, :])
    best_y = Inf
    best = (0.0f0, 0.0f0, 0.0f0)
    for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
        if arr[i, j, k] == UInt16(id_val)
            py = Float32(j - 1) * sp[2]
            if py < best_y
                best_y = py
                best = (Float32(i - 1) * sp[1], py, Float32(k - 1) * sp[3])
            end
        end
    end
    return best
end

function execute_mega2_batch(batch_json_str)
    t0 = time_ns()
    batch_dict = JSON.parse(batch_json_str)
    batch = get(batch_dict, "operations", [])
    
    instructions = DagInstructionV2[]
    dt_idx = 1
    dims = TM_MEGA2.dims
    
    # Run pre-computations for DTs
    for op in batch
        opcode = Int32(op["opcode"])
        if opcode == OP_ANISOTROPIC_MARGIN || opcode == OP_DISTANCE_EXPANSION
            in_ch = Int32(op["in_ch"])
            in_id = UInt16(op["in_id"])
            
            # Extract mask to run JFA
            backend = KernelAbstractions.get_backend(TM_MEGA2.tensor_in)
            mask = adapt(backend, zeros(UInt8, dims[1], dims[2], dims[3]))
            
            @kernel function extract_tmp!(dst, src_in, src_out, ch, id, dims)
                I, J, K = @index(Global, NTuple)
                if I <= dims[1] && J <= dims[2] && K <= dims[3]
                    if ch > 0
                        if src_in[ch, I, J, K] == id
                            dst[I, J, K] = UInt8(1)
                        end
                    elseif ch < 0
                        if src_out[-ch, I, J, K] > 0
                            dst[I, J, K] = UInt8(1)
                        end
                    end
                end
            end
            
            if in_ch != 0
                extract_tmp!(backend, 256)(mask, TM_MEGA2.tensor_in, TM_MEGA2.tensor_out, in_ch, in_id, dims, ndrange=dims)
            end
            
            for u in get(op, "union_channels", [])
                u_ch = Int32(u["ch"])
                u_id = UInt16(u["id"])
                if u_ch != 0
                    extract_tmp!(backend, 256)(mask, TM_MEGA2.tensor_in, TM_MEGA2.tensor_out, u_ch, u_id, dims, ndrange=dims)
                end
            end
            
            KernelAbstractions.synchronize(backend)
            
            mask_sum = sum(mask)
            println("[DEBUG MEGA2 JFA] Extract mask for op $opcode: sum=$mask_sum")
            flush(stdout)
            
            dt = run_jfa_3d(backend, mask, TM_MEGA2.sp[1], TM_MEGA2.sp[2], TM_MEGA2.sp[3])
            
            @kernel function copy_dt_kernel!(dt_tensor, dt, dt_idx, dims)
                I, J, K = @index(Global, NTuple)
                if I <= dims[1] && J <= dims[2] && K <= dims[3]
                    dt_tensor[1, dt_idx, I, J, K] = dt[1, I, J, K]
                    dt_tensor[2, dt_idx, I, J, K] = dt[2, I, J, K]
                    dt_tensor[3, dt_idx, I, J, K] = dt[3, I, J, K]
                end
            end
            copy_dt_kernel!(backend, 256)(TM_MEGA2.dt_tensor, dt, Int32(dt_idx), dims, ndrange=dims)
            KernelAbstractions.synchronize(backend)
            
            op["farg1"] = Float32(dt_idx)
            dt_idx += 1
        end
    end
    
    for op in batch
        push!(instructions, DagInstructionV2(
            Int32(op["opcode"]),
            Int32(get(op, "in_ch", 0)), UInt16(get(op, "in_id", 0)),
            Int32(get(op, "limit_ch", 0)), UInt16(get(op, "limit_id", 0)),
            Int32(get(op, "out_ch", 0)),
            Float32(get(op, "farg1", 0.0)), Float32(get(op, "farg2", 0.0)),
            Float32(get(op, "farg3", 0.0)), Float32(get(op, "farg4", 0.0)),
            Float32(get(op, "farg5", 0.0)), Float32(get(op, "farg6", 0.0)),
            Float32(get(op, "farg7", 0.0)),
            Float32(get(op, "farg8", 0.0)), Float32(get(op, "farg9", 0.0)),
            Float32(get(op, "farg10", 0.0)), Float32(get(op, "farg11", 0.0)),
            Float32(get(op, "farg12", 0.0)), Float32(get(op, "farg13", 0.0)),
            Float32(get(op, "farg14", 0.0)), Float32(get(op, "farg15", 0.0)),
            Float32(get(op, "farg16", 0.0)), Float32(get(op, "farg17", 0.0)),
            Int32(get(op, "bb_min_x", 1)), Int32(get(op, "bb_max_x", dims[1])),
            Int32(get(op, "bb_min_y", 1)), Int32(get(op, "bb_max_y", dims[2])),
            Int32(get(op, "bb_min_z", 1)), Int32(get(op, "bb_max_z", dims[3]))
        ))
    end
    
    t_parse = time_ns()
    
    if !isempty(instructions)
        # ── CPU validation pass — check for OOB before launching GPU kernel ──
        n_in  = TM_MEGA2.tensor_in !== nothing ? size(TM_MEGA2.tensor_in,  1) : 0
        n_out = TM_MEGA2.tensor_out !== nothing ? size(TM_MEGA2.tensor_out, 1) : 0
        n_dt  = TM_MEGA2.dt_tensor !== nothing ? size(TM_MEGA2.dt_tensor,  2) : 0
        tin_xyz = TM_MEGA2.tensor_in !== nothing ? (size(TM_MEGA2.tensor_in, 2), size(TM_MEGA2.tensor_in, 3), size(TM_MEGA2.tensor_in, 4)) : (0, 0, 0)
        println("[VALIDATE] n_in=$n_in n_out=$n_out dims=$(TM_MEGA2.dims) tensor_in_xyz=$tin_xyz n_inst=$(length(instructions))")
        flush(stdout)
        for (idx, inst) in enumerate(instructions)
            bad = false
            if inst.in_ch > n_in
                println("[VALIDATE] inst $idx: in_ch=$(inst.in_ch) > tensor_in channels=$(n_in), opcode=$(inst.opcode)")
                bad = true
            end
            if inst.in_ch < 0 && (-inst.in_ch) > n_out
                println("[VALIDATE] inst $idx: in_ch=$(inst.in_ch) → tensor_out channel $(-inst.in_ch) > n_out=$n_out, opcode=$(inst.opcode)")
                bad = true
            end
            
            # Print ALL instructions that write to channel 13 (helper_Paraaortic_Aorta)
            # or any instruction that reads from it
            # To be safe, let's just print them all if it's a small DAG. But it has 150 instructions.
            # Let's print the whole instruction list for debugging:
            if idx <= 150
                println("[DEBUG-INST] idx=$idx op=$(inst.opcode) in=$(inst.in_ch):$(inst.in_id) lim=$(inst.limit_ch):$(inst.limit_id) out=$(inst.out_ch) bb_z=$(inst.bb_min_z)-$(inst.bb_max_z) arg1=$(inst.arg1)")
            end
            if inst.limit_ch > n_in
                println("[VALIDATE] inst $idx: limit_ch=$(inst.limit_ch) > tensor_in channels=$(n_in), opcode=$(inst.opcode)")
                bad = true
            end
            if inst.limit_ch < 0 && (-inst.limit_ch) > n_out
                println("[VALIDATE] inst $idx: limit_ch=$(inst.limit_ch) → tensor_out channel $(-inst.limit_ch) > n_out=$n_out, opcode=$(inst.opcode)")
                bad = true
            end
            if inst.out_ch > n_out
                println("[VALIDATE] inst $idx: out_ch=$(inst.out_ch) > tensor_out channels=$(n_out), opcode=$(inst.opcode)")
                bad = true
            end
            if inst.opcode == 1 || inst.opcode == 8
                dt_ch = Int32(round(inst.arg1))  # arg1 = dt_ch slot (stored as Float32, but is integer)
                if dt_ch > n_dt
                    println("[VALIDATE] inst $idx: dt_ch=$(dt_ch) > dt_tensor slots=$(n_dt), opcode=$(inst.opcode)")
                    bad = true
                end
            end
            if bad
                println("[VALIDATE] Full inst: in_ch=$(inst.in_ch) in_id=$(inst.in_id) limit_ch=$(inst.limit_ch) out_ch=$(inst.out_ch) bb=$(inst.bb_min_x)-$(inst.bb_max_x),$(inst.bb_min_y)-$(inst.bb_max_y),$(inst.bb_min_z)-$(inst.bb_max_z)")
                flush(stdout)
            end
        end
        flush(stdout)
        # ─────────────────────────────────────────────────────────────────────

        backend = TM_MEGA2.backend !== nothing ? TM_MEGA2.backend : CUDABackend()
        inst_device = adapt(backend, instructions)
        mega_fused_dag_kernel_v2!(backend, 256)(
            TM_MEGA2.tensor_in, TM_MEGA2.tensor_out, TM_MEGA2.dt_tensor, 
            inst_device, Int32(length(instructions)), dims, 
            TM_MEGA2.sp[1], TM_MEGA2.sp[2], TM_MEGA2.sp[3], 
            ndrange=dims
        )
        KernelAbstractions.synchronize(backend)
    end
    
    t_kernel = time_ns()
    println("[BENCH] mega2_batch: parse=$(round(Int, (t_parse-t0)/1e6))ms kernel=$(round(Int, (t_kernel-t_parse)/1e6))ms n_ops=$(length(instructions))")
    flush(stdout)
    
    counts_gpu = mapreduce(x -> Int32(x), +, TM_MEGA2.tensor_out; dims=(2,3,4), init=Int32(0))
    counts_cpu = Array(counts_gpu)
    counts_flat = dropdims(counts_cpu, dims=(2,3,4))
    
    return JSON.json(Dict("status" => "success", "counts" => counts_flat))
end

function extract_mega2_mask(ch)
    dims = TM_MEGA2.dims
    mask = adapt(TM_MEGA2.backend, zeros(UInt8, dims[1], dims[2], dims[3]))
    @kernel function extract_out!(dst, src, c, dims)
        I, J, K = @index(Global, NTuple)
        if I <= dims[1] && J <= dims[2] && K <= dims[3]
            dst[I, J, K] = src[c, I, J, K]
        end
    end
    extract_out!(TM_MEGA2.backend, 256)(mask, TM_MEGA2.tensor_out, Int32(ch), dims, ndrange=dims)
    KernelAbstractions.synchronize(TM_MEGA2.backend)
    return mask
end

function insert_mega2_mask!(mask, ch)
    dims = TM_MEGA2.dims
    @kernel function insert_out!(dst, src, c, dims)
        I, J, K = @index(Global, NTuple)
        if I <= dims[1] && J <= dims[2] && K <= dims[3]
            dst[c, I, J, K] = src[I, J, K]
        end
    end
    insert_out!(TM_MEGA2.backend, 256)(TM_MEGA2.tensor_out, mask, Int32(ch), dims, ndrange=dims)
    KernelAbstractions.synchronize(TM_MEGA2.backend)
    if typeof(mask) <: CuArray
        CUDA.unsafe_free!(mask)
    end
end

function execute_mega2_hull(batch_json_str)
    batch_dict = JSON.parse(batch_json_str)
    edge_ops = get(batch_dict, "edge_ops", [])
    hull_ops = get(batch_dict, "hull_ops", [])
    dims = TM_MEGA2.dims
    
    # 1. Run edge operations
    if !isempty(edge_ops)
        batch_str = JSON.json(Dict("operations" => edge_ops))
        execute_mega2_batch(batch_str)
    end
    
    # 2. Extract edges and compute convex hull (Pure GPU)
    for op in hull_ops
        edge_ch = Int32(op["edge_ch"])
        out_ch = Int32(op["out_ch"])
        
        planes_table, num_planes = HullPlanes.compute_sector_planes_mega_gpu(
            TM_MEGA2.backend, TM_MEGA2.tensor_out, edge_ch, dims)
        
        max_p = Int32(size(planes_table, 1))
        HullPlanes.fill_hull_gpu!(TM_MEGA2.backend, 256)(
            TM_MEGA2.tensor_out, out_ch, planes_table, num_planes,
            Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), max_p, ndrange=dims)
        KernelAbstractions.synchronize(TM_MEGA2.backend)
    end
    
    return "Executed MegaV2 hull batch"
end

# ─── Step 4: Merge output channels into input tensor (pure GPU, no HDF5 round-trip) ─────────────────
# This implements the user's step 4: after executing a DAG block, the output masks
# are merged back into tensor_in as new integer channels, so the next kernel pass
# can reference them directly.
#
# `assignments` is a vector of tuples: (out_ch::Int, target_in_ch::Int, new_id::UInt16)
#   out_ch:      source channel in tensor_out (1-based, binary mask)
#   target_in_ch: destination channel in tensor_in (1-based, may be new)
#   new_id:      integer id to write in tensor_in for this mask
#
# If tensor_in doesn't have enough channels, it is expanded.
function extend_mega2_inputs!(assignments::Vector{Tuple{Int,Int,UInt16}})
    dims = TM_MEGA2.dims
    backend = TM_MEGA2.backend

    # Find max target channel needed
    max_needed_ch = maximum(a[2] for a in assignments)
    current_in_ch = size(TM_MEGA2.tensor_in, 1)

    if max_needed_ch > current_in_ch
        new_total = max_needed_ch
        mem_needed = new_total * dims[1] * dims[2] * dims[3] * sizeof(UInt16)
        println("[extend_mega2_inputs!] Need $new_total channels ($(round(mem_needed/1e9, digits=2)) GB)")
        
        old_tensor = TM_MEGA2.tensor_in
        if backend isa CUDABackend
            new_tensor_dev = CUDA.zeros(UInt16, new_total, dims[1], dims[2], dims[3])
        else
            new_tensor_dev = zeros(UInt16, new_total, dims[1], dims[2], dims[3])
        end
        new_tensor_dev[1:current_in_ch, :, :, :] .= old_tensor
        TM_MEGA2.tensor_in = new_tensor_dev
        try CUDA.unsafe_free!(old_tensor) catch end
        println("[extend_mega2_inputs!] Allocated on GPU: $(size(new_tensor_dev))")
    end

    if isempty(assignments)
        return
    end
    
    backend = KernelAbstractions.get_backend(TM_MEGA2.tensor_in)
    
    @kernel function assign_kernel!(t_in, t_out, in_ch, out_ch, new_id, dims)
        I, J, K = @index(Global, NTuple)
        if I <= dims[1] && J <= dims[2] && K <= dims[3]
            if t_out[out_ch, I, J, K] > 0
                t_in[in_ch, I, J, K] = new_id
            end
        end
    end
    
    kernel! = assign_kernel!(backend, 256)
    for (out_ch, in_ch, new_id) in assignments
        kernel!(TM_MEGA2.tensor_in, TM_MEGA2.tensor_out, Int32(in_ch), Int32(out_ch), UInt16(new_id), TM_MEGA2.dims, ndrange=TM_MEGA2.dims)
    end
    KernelAbstractions.synchronize(backend)
    
    println("[extend_mega2_inputs!] Merged $(length(assignments)) outputs into tensor_in")
end

# Reset output tensor between DAG steps (clear all output channels to zero).
# Called before each new batch execution to avoid stale values.
function reset_mega2_outputs!(num_out_channels::Int)
    dims = TM_MEGA2.dims
    backend = TM_MEGA2.backend
    if TM_MEGA2.tensor_out === nothing || size(TM_MEGA2.tensor_out, 1) < num_out_channels
        if TM_MEGA2.tensor_out !== nothing
            try
                CUDA.unsafe_free!(TM_MEGA2.tensor_out)
            catch
            end
        end
        TM_MEGA2.tensor_out = adapt(backend, zeros(UInt8, num_out_channels, dims[1], dims[2], dims[3]))
    else
        fill!(TM_MEGA2.tensor_out, UInt8(0))
    end
end

function get_mega2_bboxes(channels)
    dims = TM_MEGA2.dims
    bboxes = Dict{String, Any}()
    
    # We can do a quick CPU reduction or a GPU reduction. 
    # Since we are reducing a boolean mask over a 3D volume, 
    # doing it on the GPU with mapreduce or just finding the limits is fast.
    # For simplicity and given the 4D array is on GPU, we can use CUDA mapreduce
    # or just copy the required channels to CPU if there are few.
    # Since channels are single bytes and we only need bounding box, 
    # downloading the channel to CPU and using Julia's findall is fine (a few ms).
    
    tensor_out_cpu = adapt(CPU(), TM_MEGA2.tensor_out)
    
    for ch_name in keys(channels)
        ch_idx = channels[ch_name]
        if ch_idx > 0 && ch_idx <= size(tensor_out_cpu, 1)
            arr = tensor_out_cpu[ch_idx, :, :, :]
            # Find bounding box
            indices = findall(x -> x > 0, arr)
            if isempty(indices)
                bboxes[ch_name] = nothing
            else
                min_x = minimum(i[1] for i in indices)
                max_x = maximum(i[1] for i in indices)
                min_y = minimum(i[2] for i in indices)
                max_y = maximum(i[2] for i in indices)
                min_z = minimum(i[3] for i in indices)
                max_z = maximum(i[3] for i in indices)
                # 0-based for Python
                bboxes[ch_name] = [[min_x-1, max_x-1], [min_y-1, max_y-1], [min_z-1, max_z-1]]
            end
        end
    end
    
    return bboxes
end
