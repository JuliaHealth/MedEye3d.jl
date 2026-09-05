module ConnectedComponents

using KernelAbstractions
using CUDA

export extract_largest_connected_component, label_connected_components

"""
Initialize labels: each foreground voxel gets a unique 1-based linear index, background gets 0.
"""
@kernel function init_labels_kernel!(labels, @Const(mask))
    I = @index(Global, Cartesian)
    if mask[I] > 0
        dims = size(labels)
        # Linear index
        labels[I] = Int32(I[1] + (I[2] - 1) * dims[1] + (I[3] - 1) * dims[1] * dims[2])
    else
        labels[I] = Int32(0)
    end
end

"""
Parallel label propagation across 26-connected neighbors.
"""
@kernel function propagate_labels_26!(labels_out, @Const(labels_in), changed)
    I = @index(Global, Cartesian)
    cur = labels_in[I]
    if cur > Int32(0)
        min_lbl = cur
        dims = size(labels_in)
        ix, iy, iz = I[1], I[2], I[3]
        
        for dz in -1:1, dy in -1:1, dx in -1:1
            nx = ix + dx
            ny = iy + dy
            nz = iz + dz
            if 1 <= nx <= dims[1] && 1 <= ny <= dims[2] && 1 <= nz <= dims[3]
                nlbl = labels_in[nx, ny, nz]
                if nlbl > Int32(0) && nlbl < min_lbl
                    min_lbl = nlbl
                end
            end
        end
        
        labels_out[I] = min_lbl
        if min_lbl < cur
            changed[1] = Int32(1)
        end
    else
        labels_out[I] = Int32(0)
    end
end

"""
Parallel label propagation across 6-connected neighbors.
"""
@kernel function propagate_labels_6!(labels_out, @Const(labels_in), changed)
    I = @index(Global, Cartesian)
    cur = labels_in[I]
    if cur > Int32(0)
        min_lbl = cur
        dims = size(labels_in)
        ix, iy, iz = I[1], I[2], I[3]
        
        # 6-neighborhood offsets
        if ix > 1 && labels_in[ix-1, iy, iz] > Int32(0) && labels_in[ix-1, iy, iz] < min_lbl
            min_lbl = labels_in[ix-1, iy, iz]
        end
        if ix < dims[1] && labels_in[ix+1, iy, iz] > Int32(0) && labels_in[ix+1, iy, iz] < min_lbl
            min_lbl = labels_in[ix+1, iy, iz]
        end
        if iy > 1 && labels_in[ix, iy-1, iz] > Int32(0) && labels_in[ix, iy-1, iz] < min_lbl
            min_lbl = labels_in[ix, iy-1, iz]
        end
        if iy < dims[2] && labels_in[ix, iy+1, iz] > Int32(0) && labels_in[ix, iy+1, iz] < min_lbl
            min_lbl = labels_in[ix, iy+1, iz]
        end
        if iz > 1 && labels_in[ix, iy, iz-1] > Int32(0) && labels_in[ix, iy, iz-1] < min_lbl
            min_lbl = labels_in[ix, iy, iz-1]
        end
        if iz < dims[3] && labels_in[ix, iy, iz+1] > Int32(0) && labels_in[ix, iy, iz+1] < min_lbl
            min_lbl = labels_in[ix, iy, iz+1]
        end
        
        labels_out[I] = min_lbl
        if min_lbl < cur
            changed[1] = Int32(1)
        end
    else
        labels_out[I] = Int32(0)
    end
end

"""
Extract voxels with matching target_label into a binary mask.
"""
@kernel function filter_largest_kernel!(out_mask, @Const(labels), target_label, output_val)
    I = @index(Global, Cartesian)
    if labels[I] == target_label && target_label > Int32(0)
        out_mask[I] = output_val
    else
        out_mask[I] = UInt8(0)
    end
end

"""
    label_connected_components(mask::AbstractArray{<:Real, 3}; connectivity::Int=26, use_gpu::Bool=CUDA.functional()) -> Array{Int32, 3}

Labels connected components in a 3D binary/integer mask using KernelAbstractions.
Returns a 3D Int32 array with distinct component IDs.
"""
function label_connected_components(mask::AbstractArray{<:Real, 3}; connectivity::Int=26, use_gpu::Bool=CUDA.functional())
    if count(mask .> 0) == 0
        return zeros(Int32, size(mask))
    end
    
    dims = size(mask)
    if use_gpu && CUDA.functional()
        mask_d = CUDA.CuArray(mask .> 0)
        labels_a = CUDA.zeros(Int32, dims)
        labels_b = CUDA.zeros(Int32, dims)
        changed_d = CUDA.zeros(Int32, 1)
        backend = KernelAbstractions.get_backend(labels_a)
    else
        mask_d = Array{Bool, 3}(mask .> 0)
        labels_a = zeros(Int32, dims)
        labels_b = zeros(Int32, dims)
        changed_d = zeros(Int32, 1)
        backend = KernelAbstractions.get_backend(labels_a)
    end
    
    k_init = init_labels_kernel!(backend)
    k_init(labels_a, mask_d, ndrange=dims)
    KernelAbstractions.synchronize(backend)
    
    k_prop = connectivity == 6 ? propagate_labels_6!(backend) : propagate_labels_26!(backend)
    
    cur_in = labels_a
    cur_out = labels_b
    
    max_iters = max(dims...) * 4
    iter = 0
    while iter < max_iters
        iter += 1
        if use_gpu && CUDA.functional()
            CUDA.fill!(changed_d, Int32(0))
        else
            fill!(changed_d, Int32(0))
        end
        
        k_prop(cur_out, cur_in, changed_d, ndrange=dims)
        KernelAbstractions.synchronize(backend)
        
        h_changed = Array(changed_d)[1]
        cur_in, cur_out = cur_out, cur_in
        if h_changed == 0
            break
        end
    end
    
    return Array(cur_in)
end

"""
    extract_largest_connected_component(mask::AbstractArray{<:Real, 3}; connectivity::Int=26, output_val::UInt8=UInt8(1), use_gpu::Bool=CUDA.functional()) -> Array{UInt8, 3}

Extracts ONLY the largest connected component from a 3D binary/integer mask.
Uses GPU KernelAbstractions for parallel multi-pass label propagation.
"""
function extract_largest_connected_component(mask::AbstractArray{<:Real, 3}; connectivity::Int=26, output_val::UInt8=UInt8(1), use_gpu::Bool=CUDA.functional())
    if count(mask .> 0) == 0
        return zeros(UInt8, size(mask))
    end
    
    dims = size(mask)
    if use_gpu && CUDA.functional()
        mask_d = CUDA.CuArray(mask .> 0)
        labels_a = CUDA.zeros(Int32, dims)
        labels_b = CUDA.zeros(Int32, dims)
        changed_d = CUDA.zeros(Int32, 1)
        out_mask_d = CUDA.zeros(UInt8, dims)
        backend = KernelAbstractions.get_backend(labels_a)
    else
        mask_d = Array{Bool, 3}(mask .> 0)
        labels_a = zeros(Int32, dims)
        labels_b = zeros(Int32, dims)
        changed_d = zeros(Int32, 1)
        out_mask_d = zeros(UInt8, dims)
        backend = KernelAbstractions.get_backend(labels_a)
    end
    
    k_init = init_labels_kernel!(backend)
    k_init(labels_a, mask_d, ndrange=dims)
    KernelAbstractions.synchronize(backend)
    
    k_prop = connectivity == 6 ? propagate_labels_6!(backend) : propagate_labels_26!(backend)
    
    cur_in = labels_a
    cur_out = labels_b
    
    max_iters = max(dims...) * 4
    iter = 0
    while iter < max_iters
        iter += 1
        if use_gpu && CUDA.functional()
            CUDA.fill!(changed_d, Int32(0))
        else
            fill!(changed_d, Int32(0))
        end
        
        k_prop(cur_out, cur_in, changed_d, ndrange=dims)
        KernelAbstractions.synchronize(backend)
        
        h_changed = Array(changed_d)[1]
        cur_in, cur_out = cur_out, cur_in
        if h_changed == 0
            break
        end
    end
    
    final_labels = Array(cur_in)
    fg_indices = findall(final_labels .> Int32(0))
    if isempty(fg_indices)
        return zeros(UInt8, dims)
    end
    
    # Calculate component voxel counts
    counts = Dict{Int32, Int}()
    for idx in fg_indices
        lbl = final_labels[idx]
        counts[lbl] = get(counts, lbl, 0) + 1
    end
    
    best_lbl = Int32(0)
    max_c = 0
    for (lbl, c) in counts
        if c > max_c
            max_c = c
            best_lbl = lbl
        end
    end
    
    k_filter = filter_largest_kernel!(backend)
    k_filter(out_mask_d, cur_in, best_lbl, output_val, ndrange=dims)
    KernelAbstractions.synchronize(backend)
    
    return Array(out_mask_d)
end

end # module
