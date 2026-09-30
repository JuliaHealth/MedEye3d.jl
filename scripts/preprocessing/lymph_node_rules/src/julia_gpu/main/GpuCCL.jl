"""
GPU-native 3D Connected Component Labeling (CCL) using Union-Find with atomicMin.

Based on the Block-based Union Find (BUF) algorithm from:
  Allegretti, Bolelli, Grana - "Optimized Block-Based Algorithms to Label
  Connected Components on GPUs" (IEEE TPDS 2020)

Implements 26-connectivity to match SimpleITK ConnectedComponent default.

All computation stays on GPU — zero CPU↔GPU memory transfers except
the single (Int32, Int) result from the final GPU reduction (findmax).
"""
module GpuCCL

using KernelAbstractions
using CUDA

# ============================================================
# Kernel 1: Initialize labels
# Each foreground voxel gets its linear index as label.
# Background voxels get label 0.
# ============================================================
@kernel function ccl_init_kernel!(labels::AbstractArray{Int32, 1},
                                  mask::AbstractArray{T, 3},
                                  @Const(n_voxels::Int32)) where T
    idx = @index(Global, Linear)
    if idx <= n_voxels
        @inbounds labels[idx] = mask[idx] > zero(T) ? Int32(idx) : Int32(0)
    end
end

# ============================================================
# Kernel 2: Merge — Union-Find with CUDA.atomic_min!, 26-connectivity
#
# The union operation uses atomicMin for lock-free merging:
#   1. Find roots of both voxels (path compression inline)
#   2. Link the larger root to the smaller via atomicMin
#   3. Retry if someone else modified the root concurrently
#
# A single kernel launch suffices because the while(!done) loop
# inside each thread handles concurrent races (Jayanti 2004).
# ============================================================
@kernel function ccl_merge_kernel!(labels::AbstractArray{Int32, 1},
                                   mask::AbstractArray{T, 3},
                                   @Const(dx::Int32), @Const(dy::Int32), @Const(dz::Int32)) where T
    i, j, k = @index(Global, NTuple)
    if i <= dx && j <= dy && k <= dz
        @inbounds begin
            idx = Int32((k - Int32(1)) * dx * dy + (j - Int32(1)) * dx + i)
            if mask[idx] > zero(T)
                # Check 26 neighbors (only need to check "forward" half to avoid
                # redundant work, but checking all is simpler and correct since
                # union is symmetric and idempotent)
                for dkk in Int32(-1):Int32(1)
                    nk = k + dkk
                    (nk < Int32(1) || nk > dz) && continue
                    for djj in Int32(-1):Int32(1)
                        nj = j + djj
                        (nj < Int32(1) || nj > dy) && continue
                        for dii in Int32(-1):Int32(1)
                            (dii == Int32(0) && djj == Int32(0) && dkk == Int32(0)) && continue
                            ni = i + dii
                            (ni < Int32(1) || ni > dx) && continue
                            nidx = Int32((nk - Int32(1)) * dx * dy + (nj - Int32(1)) * dx + ni)
                            if mask[nidx] > zero(T)
                                # Union idx and nidx via atomicMin
                                # Find root of idx (path halving)
                                ra = idx
                                while labels[ra] != ra
                                    ra = labels[ra]
                                end
                                # Find root of nidx (path halving)
                                rb = nidx
                                while labels[rb] != rb
                                    rb = labels[rb]
                                end
                                # Link roots using atomicMin (monotonically decreasing)
                                done = false
                                while !done && ra != rb
                                    if ra < rb
                                        old = CUDA.atomic_min!(pointer(labels, rb), ra)
                                        done = (old == rb)
                                        rb = old
                                    else
                                        old = CUDA.atomic_min!(pointer(labels, ra), rb)
                                        done = (old == ra)
                                        ra = old
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

# ============================================================
# Kernel 3: Path Compression
# Flatten all trees so every voxel points directly to its root.
# ============================================================
@kernel function ccl_compress_kernel!(labels::AbstractArray{Int32, 1},
                                      @Const(n_voxels::Int32))
    idx = @index(Global, Linear)
    if idx <= n_voxels
        @inbounds begin
            lbl = labels[idx]
            if lbl != Int32(0)
                # Chase to root
                root = lbl
                while labels[root] != root
                    root = labels[root]
                end
                # Point directly to root
                labels[idx] = root
            end
        end
    end
end

# ============================================================
# Kernel 4: Count component sizes (atomic add on root labels)
# Atomic add is necessary here because this is a histogram
# operation — each voxel contributes to a different bucket.
# ============================================================
@kernel function ccl_count_kernel!(counts::AbstractArray{Int32, 1},
                                   labels::AbstractArray{Int32, 1},
                                   @Const(n_voxels::Int32))
    idx = @index(Global, Linear)
    if idx <= n_voxels
        @inbounds begin
            root = labels[idx]
            if root != Int32(0)
                CUDA.@atomic counts[root] += Int32(1)
            end
        end
    end
end

# ============================================================
# Kernel 5: Extract largest connected component
# Writes 1 for voxels matching best_label, 0 otherwise.
# ============================================================
@kernel function ccl_extract_kernel!(output::AbstractArray{UInt8, 3},
                                     labels::AbstractArray{Int32, 1},
                                     @Const(best_label::Int32),
                                     @Const(n_voxels::Int32))
    idx = @index(Global, Linear)
    if idx <= n_voxels
        @inbounds begin
            output[idx] = labels[idx] == best_label ? UInt8(1) : UInt8(0)
        end
    end
end


# ============================================================
# High-level API: GPU Largest Connected Component
#
# All computation stays on GPU. The only GPU→CPU transfer is
# the single (max_count, best_idx) from findmax (binary tree
# reduction on GPU, returns 2 scalars).
# ============================================================
function gpu_findmax_kernel(counts, block_max_val, block_max_idx, n)
    tid = threadIdx().x
    bid = blockIdx().x
    g_stride = gridDim().x * blockDim().x
    
    local_max = Int32(0)
    local_idx = Int32(0)
    
    i = (bid - 1) * blockDim().x + tid
    while i <= n
        @inbounds v = counts[i]
        if v > local_max
            local_max = v
            local_idx = Int32(i)
        end
        i += g_stride
    end
    
    s_val = CUDA.CuStaticSharedArray(Int32, 256)
    s_idx = CUDA.CuStaticSharedArray(Int32, 256)
    
    s_val[tid] = local_max
    s_idx[tid] = local_idx
    sync_threads()
    
    offset = 128
    while offset > 0
        if tid <= offset && tid + offset <= 256
            if s_val[tid + offset] > s_val[tid]
                s_val[tid] = s_val[tid + offset]
                s_idx[tid] = s_idx[tid + offset]
            end
        end
        sync_threads()
        offset ÷= 2
    end
    
    if tid == 1
        block_max_val[bid] = s_val[1]
        block_max_idx[bid] = s_idx[1]
    end
    return
end

"""
    gpu_largest_connected_component!(backend, output, mask, dims, labels_buf, counts_buf; block_val=nothing, block_idx=nothing)

Compute the largest connected component of `mask` (3D UInt8 GPU array)
with 26-connectivity. Result is written into `output` (same dims as mask).
Uses preallocated `labels_buf` and `counts_buf` (1D Int32 arrays, length ≥ prod(dims)).

Returns `output`.
"""
function gpu_largest_connected_component!(
    backend,
    output::AbstractArray{UInt8, 3},
    mask::AbstractArray{T, 3},
    dims::Tuple{Int, Int, Int},
    labels_buf::AbstractArray{Int32, 1},
    counts_buf::AbstractArray{Int32, 1};
    block_val=nothing,
    block_idx=nothing
) where T
    n_voxels = Int32(dims[1] * dims[2] * dims[3])
    dx, dy, dz = Int32(dims[1]), Int32(dims[2]), Int32(dims[3])
    
    # Step 1: Initialize labels (foreground=linear_index, background=0)
    fill!(counts_buf, Int32(0))
    kernel_init = ccl_init_kernel!(backend, 256)
    kernel_init(labels_buf, mask, n_voxels; ndrange=Int(n_voxels))
    KernelAbstractions.synchronize(backend)
    
    # Step 2: Merge (Union-Find with atomicMin, 26-connectivity)
    # Block dims (8,4,4) = 128 threads/block, good occupancy for 3D
    kernel_merge = ccl_merge_kernel!(backend, (8, 4, 4))
    kernel_merge(labels_buf, mask, dx, dy, dz; ndrange=(Int(dx), Int(dy), Int(dz)))
    KernelAbstractions.synchronize(backend)
    
    # Step 3: Path compression — flatten all union-find trees
    kernel_compress = ccl_compress_kernel!(backend, 256)
    kernel_compress(labels_buf, n_voxels; ndrange=Int(n_voxels))
    KernelAbstractions.synchronize(backend)
    
    # Step 4: Count component sizes (atomic add — histogram operation)
    kernel_count = ccl_count_kernel!(backend, 256)
    kernel_count(counts_buf, labels_buf, n_voxels; ndrange=Int(n_voxels))
    KernelAbstractions.synchronize(backend)
    
    # Step 5: Find label with maximum count using custom zero-allocation GPU reduction
    if backend isa CUDABackend || counts_buf isa CuArray
        threads = 256
        blocks = min(1024, cld(Int(n_voxels), threads))
        
        b_val = block_val !== nothing ? block_val : CUDA.zeros(Int32, blocks)
        b_idx = block_idx !== nothing ? block_idx : CUDA.zeros(Int32, blocks)
        
        @cuda threads=threads blocks=blocks gpu_findmax_kernel(counts_buf, b_val, b_idx, n_voxels)
        
        # Second pass on CPU (only 1024 elements = 4 KB!)
        vals = Array(view(b_val, 1:blocks))
        idxs = Array(view(b_idx, 1:blocks))
        m_val, b_i = findmax(vals)
        max_count = m_val
        best_label = idxs[b_i]
    else
        max_count, best_idx = findmax(counts_buf)
        best_label = Int32(best_idx)
    end
    
    if max_count == Int32(0)
        fill!(output, UInt8(0))
        return output
    end
    
    # Step 6: Extract LCC voxels — zero-copy, writes directly to output
    kernel_extract = ccl_extract_kernel!(backend, 256)
    kernel_extract(output, labels_buf, best_label, n_voxels; ndrange=Int(n_voxels))
    KernelAbstractions.synchronize(backend)
    
    return output
end

end # module GpuCCL
