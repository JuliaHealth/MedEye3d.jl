module MaskPacker

using Adapt
using KernelAbstractions
using KernelAbstractions: @index
using CUDA

export PackedTensor, pack_masks, add_mask!, add_masks!, unpack_mask, unpack_mask!, pack_level_outputs!, get_mask_bbox

"""
    PackedTensor
Stores non-overlapping 3D binary masks in a 4D integer tensor `(X, Y, Z, C)`
where each non-overlapping mask within channel `C` is assigned a unique integer ID >= 1.
"""
mutable struct PackedTensor
    data::AbstractArray{UInt8, 4}              # (X, Y, Z, C)
    num_channels_used::Int
    registry::Dict{String, Tuple{Int, UInt8}}  # mask_name -> (channel_idx, int_id)
    next_id::Vector{UInt8}                     # next available ID per channel
    dims::Tuple{Int, Int, Int}
    overlap_flag::AbstractArray{Int32, 1}      # Preallocated 1-element flag buffer for zero-allocation overlap checks
    bboxes::Dict{String, Tuple{Int, Int, Int, Int, Int, Int}} # mask_name -> (min_i, max_i, min_j, max_j, min_k, max_k)
end

function PackedTensor(dims::Tuple{Int, Int, Int}; capacity::Int=102, backend=CPU())
    if backend isa CPU || !CUDA.functional()
        init_data = zeros(UInt8, dims[1], dims[2], dims[3], capacity)
        overlap_flag = zeros(Int32, 1)
    else
        init_data = KernelAbstractions.zeros(backend, UInt8, dims[1], dims[2], dims[3], capacity)
        overlap_flag = KernelAbstractions.zeros(backend, Int32, 1)
    end
    registry = Dict{String, Tuple{Int, UInt8}}()
    next_id = ones(UInt8, capacity)
    bboxes = Dict{String, Tuple{Int, Int, Int, Int, Int, Int}}()
    return PackedTensor(init_data, 1, registry, next_id, dims, overlap_flag, bboxes)
end

@kernel function write_mask_id_kernel!(
    packed_data,
    @Const(new_mask),
    assigned_channel::Int32,
    assigned_id::UInt8,
    dims_x::Int32, dims_y::Int32, dims_z::Int32
)
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        if new_mask[i, j, k] > 0
            packed_data[i, j, k, assigned_channel] = assigned_id
        end
    end
end

@kernel function check_overlap_kernel!(
    @Const(packed_data),
    @Const(new_mask),
    channel::Int32,
    overlap_flag,
    dims_x::Int32, dims_y::Int32, dims_z::Int32
)
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        if new_mask[i, j, k] > 0 && packed_data[i, j, k, channel] > 0
            overlap_flag[1] = Int32(1)
        end
    end
end

"""
    pack_masks(masks::AbstractDict; backend=CPU()) -> PackedTensor
"""
function pack_masks(masks::AbstractDict; backend=CPU())
    if isempty(masks)
        error("Cannot pack empty mask dictionary")
    end
    
    first_mask = first(values(masks))
    dims = (size(first_mask, 1), size(first_mask, 2), size(first_mask, 3))
    packed = PackedTensor(dims; capacity=96, backend=CPU())
    
    # Pre-count voxels once per mask
    counts = Dict{String, Int}()
    for (k, m) in masks
        counts[k] = count(m .> 0)
    end
    sorted_names = sort(collect(keys(masks)), by=k -> counts[k], rev=true)
    
    for name in sorted_names
        add_mask_cpu!(packed, name, masks[name])
    end
    println("  [MaskPacker] Packed $(length(masks)) masks into $(packed.num_channels_used) channels (preallocated capacity=$(size(packed.data, 4))).")
    
    if !(backend isa CPU) && CUDA.functional()
        gpu_data = CuArray(packed.data)
        packed.data = gpu_data
        packed.overlap_flag = adapt(backend, packed.overlap_flag)
    end
    
    return packed
end

"""
    add_mask_cpu!(packed::PackedTensor, name::String, mask::AbstractArray)
"""
function add_mask_cpu!(packed::PackedTensor, name::String, mask::AbstractArray)
    dims = packed.dims
    mask_host = mask isa Array ? (mask .> 0) : adapt(Array, mask .> 0)
    
    indices = findall(mask_host)
    if isempty(indices)
        packed.registry[name] = (1, UInt8(0))
        return
    end
    
    packed.bboxes[name] = (
        minimum(ci[1] for ci in indices), maximum(ci[1] for ci in indices),
        minimum(ci[2] for ci in indices), maximum(ci[2] for ci in indices),
        minimum(ci[3] for ci in indices), maximum(ci[3] for ci in indices)
    )
    
    indices_1d = [ci[1] + (ci[2]-1)*dims[1] + (ci[3]-1)*dims[1]*dims[2] for ci in indices]
    
    assigned_channel = 0
    vol_size = length(mask_host)
    
    for c in 1:packed.num_channels_used
        offset = (c - 1) * vol_size
        has_overlap = false
        @inbounds for idx in indices_1d
            if packed.data[idx + offset] > 0
                has_overlap = true
                break
            end
        end
        if !has_overlap
            assigned_channel = c
            break
        end
    end
    
    if assigned_channel == 0
        packed.num_channels_used += 1
        if packed.num_channels_used > size(packed.data, 4)
            new_cap = size(packed.data, 4) + 2
            new_data = zeros(UInt8, dims[1], dims[2], dims[3], new_cap)
            new_data[:, :, :, 1:size(packed.data, 4)] .= packed.data
            packed.data = new_data
            for _ in 1:2 push!(packed.next_id, UInt8(1)) end
        end
        assigned_channel = packed.num_channels_used
    end
    
    assigned_id = packed.next_id[assigned_channel]
    packed.next_id[assigned_channel] += UInt8(1)
    
    offset = (assigned_channel - 1) * vol_size
    @inbounds for idx in indices_1d
        packed.data[idx + offset] = assigned_id
    end
    
    packed.registry[name] = (assigned_channel, assigned_id)
end

"""
    add_mask_gpu!(packed::PackedTensor, name::String, mask::AbstractArray; backend=CUDABackend())
Adds a mask to PackedTensor entirely on GPU without CPU roundtrips.
"""
function add_mask_gpu!(packed::PackedTensor, name::String, mask::AbstractArray; backend=CUDA.functional() ? CUDABackend() : CPU())
    dims = packed.dims
    mask_gpu = (mask isa CuArray) ? mask : adapt(backend, UInt8.(mask .> 0))
    if count(x -> x > 0, mask_gpu) == 0
        packed.registry[name] = (1, UInt8(0))
        return
    end

    assigned_channel = 0
    overlap_flag = packed.overlap_flag
    overlap_kernel! = check_overlap_kernel!(backend)

    for c in 1:packed.num_channels_used
        if backend isa CUDABackend
            CUDA.fill!(overlap_flag, Int32(0))
        else
            fill!(overlap_flag, Int32(0))
        end
        overlap_kernel!(
            packed.data, mask_gpu, Int32(c), overlap_flag,
            Int32(dims[1]), Int32(dims[2]), Int32(dims[3]),
            ndrange=dims
        )
        KernelAbstractions.synchronize(backend)
        flag_val = (overlap_flag isa CuArray) ? Array(overlap_flag)[1] : overlap_flag[1]
        if flag_val == 0
            assigned_channel = c
            break
        end
    end

    if assigned_channel == 0
        packed.num_channels_used += 1
        if packed.num_channels_used > size(packed.data, 4)
            error("PackedTensor capacity $(size(packed.data, 4)) exceeded! Please increase initial capacity.")
        end
        assigned_channel = packed.num_channels_used
    end

    assigned_id = packed.next_id[assigned_channel]
    packed.next_id[assigned_channel] += UInt8(1)

    w_kernel! = write_mask_id_kernel!(backend)
    w_kernel!(
        packed.data, mask_gpu, Int32(assigned_channel), assigned_id,
        Int32(dims[1]), Int32(dims[2]), Int32(dims[3]),
        ndrange=dims
    )
    KernelAbstractions.synchronize(backend)

    packed.registry[name] = (assigned_channel, assigned_id)
end

"""
    pack_level_outputs!(packed::PackedTensor, level_output, level_nodes; backend=CUDABackend())
Incrementally repacks all rules generated in a single MegaKernel level into PackedTensor
with zero temporary array allocations.
"""
function pack_level_outputs!(packed::PackedTensor, level_output, level_nodes; backend=CUDA.functional() ? CUDABackend() : CPU())
    dims = packed.dims
    overlap_kernel! = check_overlap_kernel!(backend)
    w_kernel! = write_mask_id_kernel!(backend)
    overlap_flag = packed.overlap_flag
    
    for (r, name) in enumerate(level_nodes)
        mask_view = view(level_output, :, :, :, r)
        
        assigned_channel = 0
        for c in 1:packed.num_channels_used
            if backend isa CUDABackend
                CUDA.fill!(overlap_flag, Int32(0))
            else
                fill!(overlap_flag, Int32(0))
            end
            
            overlap_kernel!(
                packed.data, mask_view, Int32(c), overlap_flag,
                Int32(dims[1]), Int32(dims[2]), Int32(dims[3]),
                ndrange=dims
            )
            KernelAbstractions.synchronize(backend)
            
            flag_val = (overlap_flag isa CuArray) ? (CUDA.@allowscalar overlap_flag[1]) : overlap_flag[1]
            if flag_val == 0
                assigned_channel = c
                break
            end
        end
        
        if assigned_channel == 0
            packed.num_channels_used += 1
            if packed.num_channels_used > size(packed.data, 4)
                error("PackedTensor capacity $(size(packed.data, 4)) exceeded! Please increase initial capacity.")
            end
            assigned_channel = packed.num_channels_used
        end
        
        assigned_id = packed.next_id[assigned_channel]
        packed.next_id[assigned_channel] += UInt8(1)
        
        w_kernel!(
            packed.data, mask_view, Int32(assigned_channel), assigned_id,
            Int32(dims[1]), Int32(dims[2]), Int32(dims[3]),
            ndrange=dims
        )
        KernelAbstractions.synchronize(backend)
        
        packed.registry[name] = (assigned_channel, assigned_id)
    end
end

"""
    add_masks!(packed::PackedTensor, new_masks::AbstractDict; backend=CPU())
Fast GPU-resident batch repacking of level outputs.
"""
function add_masks!(packed::PackedTensor, new_masks::AbstractDict; backend=CUDA.functional() ? CUDABackend() : CPU())
    if isempty(new_masks); return; end
    is_gpu = !(packed.data isa Array)
    if is_gpu
        for (k, v) in new_masks
            v_gpu = (v isa CuArray) ? v : adapt(backend, UInt8.(v .> 0))
            add_mask_gpu!(packed, k, v_gpu; backend=backend)
        end
        GC.gc(false)  # Light GC only — avoid CUDA.reclaim() which flushes the pool cache
    else
        for (k, v) in new_masks
            add_mask_cpu!(packed, k, v)
        end
    end
end

"""
    add_mask!(packed::PackedTensor, name::String, mask::AbstractArray; backend=CPU())
"""
function add_mask!(packed::PackedTensor, name::String, mask::AbstractArray; backend=CUDA.functional() ? CUDABackend() : CPU())
    add_masks!(packed, Dict(name => mask); backend=backend)
end

"""
    unpack_mask(packed::PackedTensor, name::String) -> AbstractArray{UInt8, 3}
"""
function unpack_mask(packed::PackedTensor, name::String)
    if !haskey(packed.registry, name)
        error("Mask '$name' not found in PackedTensor registry")
    end
    
    ch, id = packed.registry[name]
    if id == 0
        if packed.data isa Array
            return zeros(UInt8, packed.dims)
        else
            return CUDA.zeros(UInt8, packed.dims)
        end
    end
    
    ch_data = @view packed.data[:, :, :, ch]
    return map(x -> x == id ? UInt8(1) : UInt8(0), ch_data)
end

function unpack_mask!(out::AbstractArray{UInt8, 3}, packed::PackedTensor, name::String)
    ch_info = get(packed.registry, name, nothing)
    if ch_info === nothing
        fill!(out, UInt8(0))
        return false
    end
    
    ch, val = ch_info
    out .= UInt8.(view(packed.data, :, :, :, ch) .== val)
    return true
end

function get_mask_bbox(packed::PackedTensor, name::String)
    return get(packed.bboxes, name, nothing)
end

end # module
