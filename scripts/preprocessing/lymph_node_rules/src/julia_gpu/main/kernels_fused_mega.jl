using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using Adapt

# Instruction representation for the interpreter
struct VoxelOp
    op_type::Int32       # 0: Subtract/Overlap, 1: Union, 2: Conditional, etc
    target_id::UInt16    # Target mask ID to modify
    arg_id::UInt16       # Argument mask ID
end

struct BBox
    min_x::Int32
    max_x::Int32
    min_y::Int32
    max_y::Int32
    min_z::Int32
    max_z::Int32
end

# ---------------------------------------------------------
# Graph Coloring & Bounding Box Logic
# ---------------------------------------------------------

function calculate_bounding_box(mask::AbstractArray)
    inds = findall(x -> x > 0, mask)
    if isempty(inds)
        return BBox(0, 0, 0, 0, 0, 0)
    end
    min_x = minimum(i[1] for i in inds)
    max_x = maximum(i[1] for i in inds)
    min_y = minimum(i[2] for i in inds)
    max_y = maximum(i[2] for i in inds)
    min_z = minimum(i[3] for i in inds)
    max_z = maximum(i[3] for i in inds)
    return BBox(Int32(min_x), Int32(max_x), Int32(min_y), Int32(max_y), Int32(min_z), Int32(max_z))
end

function boxes_intersect(b1::BBox, b2::BBox)
    if (b1.max_x == 0 && b1.max_y == 0) || (b2.max_x == 0 && b2.max_y == 0)
        return false
    end
    if b1.max_x < b2.min_x || b1.min_x > b2.max_x return false end
    if b1.max_y < b2.min_y || b1.min_y > b2.max_y return false end
    if b1.max_z < b2.min_z || b1.min_z > b2.max_z return false end
    return true
end

function masks_overlap_ka(mask1::AbstractArray, b1::BBox, mask2::AbstractArray, b2::BBox; exact_voxel_overlap=true)
    if !boxes_intersect(b1, b2)
        return false
    end
    if !exact_voxel_overlap
        return true
    end
    rx = max(b1.min_x, b2.min_x):min(b1.max_x, b2.max_x)
    ry = max(b1.min_y, b2.min_y):min(b1.max_y, b2.max_y)
    rz = max(b1.min_z, b2.min_z):min(b1.max_z, b2.max_z)
    sub1 = @view mask1[rx, ry, rz]
    sub2 = @view mask2[rx, ry, rz]
    return any((sub1 .> 0) .& (sub2 .> 0))
end

function compute_graph_coloring(masks::Vector{<:AbstractArray}; exact_voxel_overlap=true)
    N = length(masks)
    bboxes = [calculate_bounding_box(m) for m in masks]
    
    adj = zeros(Bool, N, N)
    for i in 1:N
        for j in (i+1):N
            if masks_overlap_ka(masks[i], bboxes[i], masks[j], bboxes[j]; exact_voxel_overlap=exact_voxel_overlap)
                adj[i, j] = true
                adj[j, i] = true
            end
        end
    end
    
    colors = zeros(Int, N)
    for i in 1:N
        used_colors = Set{Int}()
        for j in 1:N
            if adj[i, j] && colors[j] > 0
                push!(used_colors, colors[j])
            end
        end
        
        c = 1
        while c in used_colors
            c += 1
        end
        colors[i] = c
    end
    
    return colors, bboxes
end

# ---------------------------------------------------------
# GPU Kernels (KernelAbstractions)
# ---------------------------------------------------------

@kernel function pack_single_mask_kernel!(tensor_4d, @Const(mask_3d), channel_idx, mask_id, dims)
    I, J, K = @index(Global, NTuple)
    if I <= dims[1] && J <= dims[2] && K <= dims[3]
        if mask_3d[I, J, K] > 0
            tensor_4d[channel_idx, I, J, K] = mask_id
        end
    end
end

@kernel function unpack_single_mask_kernel!(binary_out, @Const(tensor_4d), channel_idx, mask_id, dims)
    I, J, K = @index(Global, NTuple)
    if I <= dims[1] && J <= dims[2] && K <= dims[3]
        binary_out[I, J, K] = (tensor_4d[channel_idx, I, J, K] == mask_id) ? UInt8(1) : UInt8(0)
    end
end

@kernel function fused_mega_kernel!(tensor_4d, @Const(ops), @Const(bboxes), num_k)
    x, y, z = @index(Global, NTuple)
    
    for op in ops
        t_box = bboxes[op.target_id]
        if x < t_box.min_x || x > t_box.max_x || y < t_box.min_y || y > t_box.max_y || z < t_box.min_z || z > t_box.max_z
            continue
        end
        
        target_present = false
        target_k = 0
        
        for k in 1:num_k
            if tensor_4d[k, x, y, z] == op.target_id
                target_present = true
                target_k = k
            end
        end
        
        if target_present
            if op.op_type == 0 # SUBTRACT / PRECEDENCE OVERLAP
                a_box = bboxes[op.arg_id]
                if x < a_box.min_x || x > a_box.max_x || y < a_box.min_y || y > a_box.max_y || z < a_box.min_z || z > a_box.max_z
                    continue
                end
                
                arg_present = false
                for k in 1:num_k
                    if tensor_4d[k, x, y, z] == op.arg_id
                        arg_present = true
                    end
                end
                
                if arg_present
                    tensor_4d[target_k, x, y, z] = UInt16(0)
                end
            elseif op.op_type == 1 # UNION
                # Future expansion
            end
        end
    end
end

# Pure GPU execution wrapper with KernelAbstractions
function run_mega_fused(backend, masks::Vector{<:AbstractArray}, ops_levels::Vector{Vector{VoxelOp}})
    N = length(masks)
    if N == 0
        return []
    end
    
    dims = size(masks[1])
    size_x, size_y, size_z = dims
    
    # 1. Graph Coloring
    colors, bboxes = compute_graph_coloring(masks)
    num_k = maximum(colors)
    println("MegaFused: Packed $N masks into $num_k non-overlapping channels (Compression Ratio: $(round(N/num_k, digits=2))x)")
    
    # 2. Pack 4D Tensor directly on GPU using KernelAbstractions
    d_tensor = adapt(backend, zeros(UInt16, num_k, size_x, size_y, size_z))
    pack_kernel! = pack_single_mask_kernel!(backend, 256)
    
    for i in 1:N
        k = Int32(colors[i])
        mask_id = UInt16(i)
        mask_dev = adapt(backend, masks[i])
        pack_kernel!(d_tensor, mask_dev, k, mask_id, dims, ndrange=dims)
        KernelAbstractions.synchronize(backend)
    end
    
    d_bboxes = adapt(backend, bboxes)
    
    # 3. Execute Kernel per DAG Level on GPU
    ndrange = (size_x, size_y, size_z)
    kernel! = fused_mega_kernel!(backend, 256)
    
    level_idx = 1
    for ops in ops_levels
        if isempty(ops)
            continue
        end
        d_ops = adapt(backend, ops)
        
        println("MegaFused: Executing DAG Level $level_idx with $(length(ops)) operations...")
        kernel!(d_tensor, d_ops, d_bboxes, Int32(num_k), ndrange=ndrange)
        KernelAbstractions.synchronize(backend)
        level_idx += 1
    end
    
    # 4. Unpack directly on GPU using KernelAbstractions
    unpack_kernel! = unpack_single_mask_kernel!(backend, 256)
    out_masks = []
    for i in 1:N
        k = Int32(colors[i])
        mask_id = UInt16(i)
        out_mask = adapt(backend, zeros(UInt8, size_x, size_y, size_z))
        unpack_kernel!(out_mask, d_tensor, k, mask_id, dims, ndrange=dims)
        KernelAbstractions.synchronize(backend)
        push!(out_masks, adapt(CPU(), out_mask))
    end
    
    return out_masks
end

