"""
Geometric constraint kernels for the GPU DAG pipeline.
Constraints are pre-resolved to voxel index limits by the Python orchestrator.
Each constraint clips voxels along a single axis.
"""

using KernelAbstractions
using KernelAbstractions: @index

# ============================================================
# Per-voxel constraint kernel (GPU-compatible)
# ============================================================

@kernel function constraint_clip_z_kernel!(arr, limit_z::Int32, keep_above::Int32)
    I, J, K = @index(Global, NTuple)
    # keep_above == 1: keep voxels where K >= limit_z (SuperiorTo in ascending Z)
    # keep_above == 0: keep voxels where K <= limit_z (InferiorTo in ascending Z)
    if keep_above == Int32(1)
        if K < limit_z
            arr[I, J, K] = UInt8(0)
        end
    else
        if K > limit_z
            arr[I, J, K] = UInt8(0)
        end
    end
end

@kernel function constraint_clip_y_kernel!(arr, limit_y::Int32, keep_above::Int32)
    I, J, K = @index(Global, NTuple)
    if keep_above == Int32(1)
        if J < limit_y
            arr[I, J, K] = UInt8(0)
        end
    else
        if J > limit_y
            arr[I, J, K] = UInt8(0)
        end
    end
end

@kernel function constraint_clip_x_kernel!(arr, limit_x::Int32, keep_above::Int32)
    I, J, K = @index(Global, NTuple)
    if keep_above == Int32(1)
        if I < limit_x
            arr[I, J, K] = UInt8(0)
        end
    else
        if I > limit_x
            arr[I, J, K] = UInt8(0)
        end
    end
end

# Slice-wise Y constraint: for each Z slice, clip at a per-slice Y limit
# limits_y is a 1D array of length dims[3], where limits_y[k] = the Y limit for slice k
# A value of 0 means "no constraint for this slice"
@kernel function constraint_clip_x_slicewise_kernel!(arr, limits_x, keep_above::Int32)
    I, J, K = @index(Global, NTuple)
    lim = limits_x[K]
    if lim > Int32(0)
        if keep_above == Int32(1)
            if I < lim
                arr[I, J, K] = UInt8(0)
            end
        else
            if I > lim
                arr[I, J, K] = UInt8(0)
            end
        end
    end
end

@kernel function constraint_clip_y_slicewise_kernel!(arr, limits_y, keep_above::Int32)
    I, J, K = @index(Global, NTuple)
    lim = limits_y[K]
    if lim > Int32(0)
        if keep_above == Int32(1)
            if J < lim
                arr[I, J, K] = UInt8(0)
            end
        else
            if J > lim
                arr[I, J, K] = UInt8(0)
            end
        end
    end
end

@kernel function plane_clip_kernel!(arr, orig_x, orig_y, orig_z, sp_x, sp_y, sp_z, px, py, pz, nx, ny, nz, keep_positive::Int32)
    I, J, K = @index(Global, NTuple)
    dims = size(arr)
    if I <= dims[1] && J <= dims[2] && K <= dims[3]
        if arr[I, J, K] > UInt8(0)
            x = Float32(orig_x) + (Float32(I) - 1.0f0) * Float32(sp_x)
            y = Float32(orig_y) + (Float32(J) - 1.0f0) * Float32(sp_y)
            z = Float32(orig_z) + (Float32(K) - 1.0f0) * Float32(sp_z)
            
            d = (x - Float32(px)) * Float32(nx) + (y - Float32(py)) * Float32(ny) + (z - Float32(pz)) * Float32(nz)
            if keep_positive == Int32(1)
                if d < 0.0f0
                    arr[I, J, K] = UInt8(0)
                end
            else
                if d > 0.0f0
                    arr[I, J, K] = UInt8(0)
                end
            end
        end
    end
end

# ============================================================
# Constraint application using landmark channel
# Resolves the bounding box of a landmark channel to get the limit index
# ============================================================

"""
    resolve_landmark_limit(arr, axis, boundary_part)

Given a 3D mask array `arr`, find the bounding box limit along `axis`.
- axis: 1=X, 2=Y, 3=Z
- boundary_part: "min", "max", "center"
Returns the voxel index (1-based).
"""
function resolve_landmark_limit(arrs::Vector, axis::Int, boundary_part::String)
    min_idx = typemax(Int)
    max_idx = 0
    
    for a in arrs
        if axis == 3
            z_sum = Array(sum(a, dims=(1, 2)))[:]
            zs = findall(z_sum .> 0)
            if !isempty(zs)
                min_idx = min(min_idx, minimum(zs))
                max_idx = max(max_idx, maximum(zs))
            end
        elseif axis == 2
            y_sum = Array(sum(a, dims=(1, 3)))[:]
            ys = findall(y_sum .> 0)
            if !isempty(ys)
                min_idx = min(min_idx, minimum(ys))
                max_idx = max(max_idx, maximum(ys))
            end
        elseif axis == 1
            x_sum = Array(sum(a, dims=(2, 3)))[:]
            xs = findall(x_sum .> 0)
            if !isempty(xs)
                min_idx = min(min_idx, minimum(xs))
                max_idx = max(max_idx, maximum(xs))
            end
        end
    end
    
    if max_idx == 0
        return nothing  # landmark is empty
    end
    
    if boundary_part == "min" || boundary_part == "anterior"
        return min_idx
    elseif boundary_part == "max" || boundary_part == "posterior"
        return max_idx
    elseif boundary_part == "center"
        return div(min_idx + max_idx, 2)
    else
        return min_idx  # default
    end
end

"""
    resolve_landmark_limit_per_slice(arr, slice_axis, limit_axis, boundary_part)

For each slice along `slice_axis`, find the bounding box limit along `limit_axis`.
Returns a Vector{Int32} of per-slice limits. 0 means no data in that slice.
"""
function resolve_landmark_limit_per_slice(arrs::Vector, slice_axis::Int, limit_axis::Int, boundary_part::String)
    cpu_arrs = [adapt(CPU(), a) for a in arrs]
    dims = size(cpu_arrs[1])
    n_slices = dims[slice_axis]
    limits = zeros(Int32, n_slices)
    
    for s in 1:n_slices
        s_min = typemax(Int)
        s_max = 0
        
        for arr in cpu_arrs
            if slice_axis == 3  # iterate over Z slices
                for J in 1:dims[2], I in 1:dims[1]
                    if arr[I, J, s] > 0
                        idx = limit_axis == 1 ? I : J
                        s_min = min(s_min, idx)
                        s_max = max(s_max, idx)
                    end
                end
            elseif slice_axis == 2  # iterate over Y slices
                for K in 1:dims[3], I in 1:dims[1]
                    if arr[I, s, K] > 0
                        idx = limit_axis == 1 ? I : K
                        s_min = min(s_min, idx)
                        s_max = max(s_max, idx)
                    end
                end
            else  # slice_axis == 1
                for K in 1:dims[3], J in 1:dims[2]
                    if arr[s, J, K] > 0
                        idx = limit_axis == 2 ? J : K
                        s_min = min(s_min, idx)
                        s_max = max(s_max, idx)
                    end
                end
            end
        end
        
        if s_max > 0
            if boundary_part == "min" || boundary_part == "anterior"
                limits[s] = Int32(s_min)
            elseif boundary_part == "max" || boundary_part == "posterior"
                limits[s] = Int32(s_max)
            elseif boundary_part == "center"
                limits[s] = Int32(div(s_min + s_max, 2))
            else
                limits[s] = Int32(s_min)
            end
        end
    end
    
    # Fill internal gaps only (do not extrapolate outside the landmark range)
    first_valid_s = findfirst(limits .> 0)
    last_valid_s = findlast(limits .> 0)
    if first_valid_s !== nothing && last_valid_s !== nothing
        last_val = limits[first_valid_s]
        for s in first_valid_s:last_valid_s
            if limits[s] > 0
                last_val = limits[s]
            else
                limits[s] = last_val
            end
        end
    end
    
    return limits
end

# ============================================================
# High-level constraint application
# ============================================================

"""
    apply_constraint!(backend, arr, constraint_type, axis, limit_idx, offset_vox)

Apply a single constraint to a 3D mask array.
- constraint_type: "SuperiorTo", "InferiorTo", "AnteriorTo", "PosteriorTo", "LeftOf", "RightOf"
- axis: 1=X, 2=Y, 3=Z
- limit_idx: the voxel index at which to clip
- offset_vox: additional offset in voxels (from offset_mm / spacing)
"""
function apply_constraint!(backend, arr::AbstractArray{UInt8,3}, 
                           constraint_type::String, axis::Int, 
                           limit_idx::Int, offset_vox::Int,
                           dir_00::Float64, dir_11::Float64, dir_22::Float64)
    dims = size(arr)
    
    adjusted_limit = limit_idx + offset_vox
    adjusted_limit = clamp(adjusted_limit, 1, dims[axis])
    
    keep_above = false
    if constraint_type in ["SuperiorTo", "InferiorTo"]
        keep_above = constraint_type == "SuperiorTo"
        if dir_22 < 0.0
            keep_above = !keep_above
        end
    elseif constraint_type in ["AnteriorTo", "PosteriorTo"]
        keep_above = constraint_type == "PosteriorTo"
        if dir_11 > 0.0 # RAS
            keep_above = !keep_above
        end
    elseif constraint_type in ["LeftOf", "RightOf"]
        keep_above = constraint_type == "LeftOf"
        if dir_00 > 0.0 # RAS
            keep_above = !keep_above
        end
    end
    
    k_above = keep_above ? Int32(1) : Int32(0)
    
    if axis == 3
        kernel! = constraint_clip_z_kernel!(backend)
        kernel!(arr, Int32(adjusted_limit), k_above, ndrange=size(arr))
    elseif axis == 2
        kernel! = constraint_clip_y_kernel!(backend)
        kernel!(arr, Int32(adjusted_limit), k_above, ndrange=size(arr))
    elseif axis == 1
        kernel! = constraint_clip_x_kernel!(backend)
        kernel!(arr, Int32(adjusted_limit), k_above, ndrange=size(arr))
    end
    
    KernelAbstractions.synchronize(backend)
    println("  -> Constraint $constraint_type axis=$axis limit=$adjusted_limit keep_above=$keep_above")
end

"""
    apply_constraint_slicewise!(backend, arr, constraint_type, limits_per_slice, offset_vox, dir_00, dir_11, dir_22)

Apply a per-slice Y constraint (e.g., AnteriorTo with slice_wise=true).
"""
function apply_constraint_slicewise!(backend, arr::AbstractArray{UInt8,3}, 
                                     constraint_type::String, 
                                     axis::Int,
                                     limits_per_slice::Vector{Int32}, 
                                     offset_vox::Int,
                                     dir_00::Float64, dir_11::Float64, dir_22::Float64)
    dims = size(arr)
    
    keep_above = false
    if constraint_type in ["AnteriorTo", "PosteriorTo"]
        keep_above = constraint_type == "PosteriorTo"
    elseif constraint_type in ["LeftOf", "RightOf"]
        keep_above = constraint_type == "LeftOf"
    end
    
    k_above = keep_above ? Int32(1) : Int32(0)
    
    # Pre-adjust limits with offset and copy to GPU
    adj_limits = zeros(Int32, dims[3])
    limit_dim = axis == 1 ? dims[1] : dims[2]
    for K in 1:dims[3]
        lim = limits_per_slice[K]
        if lim > 0
            adj_limits[K] = Int32(clamp(lim + offset_vox, 1, limit_dim))
        end
    end
    adj_limits_gpu = adapt(backend, adj_limits)
    
    if axis == 1
        kernel! = constraint_clip_x_slicewise_kernel!(backend)
        kernel!(arr, adj_limits_gpu, k_above, ndrange=size(arr))
    else
        kernel! = constraint_clip_y_slicewise_kernel!(backend)
        kernel!(arr, adj_limits_gpu, k_above, ndrange=size(arr))
    end
    KernelAbstractions.synchronize(backend)
end

"""
    process_constraints!(backend, tm, out_ch, constraints, spacing)

Process a list of constraint specifications for a given output channel.
Each constraint dict has:
  - "type": constraint type string
  - "landmark_channel": channel index of the landmark mask
  - "axis": 1/2/3 for X/Y/Z
  - "boundary_part": "min"/"max"/"center"
  - "offset_mm": offset in mm
  - "slice_wise": true/false (optional, for AnteriorTo/PosteriorTo)
"""
function process_constraints!(backend::KernelAbstractions.Backend, tm, out_ch::Int, constraints::Vector{Any}, sp_x::Float64, sp_y::Float64, sp_z::Float64, dir_00::Float64, dir_11::Float64, dir_22::Float64, orig_x::Float64=0.0, orig_y::Float64=0.0, orig_z::Float64=0.0)
    out_arr_gpu = get_channel_view(tm, out_ch)

    for c in constraints
        c_type = get(c, "type", "")
        if c_type == "PlaneLimit"
            side_to_keep = get(c, "side_to_keep", "negative")
            keep_pos = (side_to_keep == "positive" || side_to_keep == ">=") ? Int32(1) : Int32(0)
            
            # Check if geometric point & normal are provided directly
            if haskey(c, "point") && haskey(c, "normal")
                p = c["point"]
                n = c["normal"]
                kernel! = plane_clip_kernel!(backend)
                kernel!(out_arr_gpu, Float32(orig_x), Float32(orig_y), Float32(orig_z),
                        Float32(sp_x), Float32(sp_y), Float32(sp_z),
                        Float32(p[1]), Float32(p[2]), Float32(p[3]),
                        Float32(n[1]), Float32(n[2]), Float32(n[3]),
                        keep_pos, ndrange=size(out_arr_gpu))
                KernelAbstractions.synchronize(backend)
                continue
            end
            
            for lm_ch in get(c, "landmark_channels", [])
                if lm_ch > 0
                    lm_view = get_channel_view(tm, lm_ch)
                    op_str = side_to_keep == "negative" ? "subtract" : "intersect"
                    run_boolean_op!(backend, out_arr_gpu, out_arr_gpu, lm_view, op_str)
                end
            end
            continue
        end
        
        axis = get(c, "axis", 3)
        b_part = get(c, "boundary_part", "center")
        offset_mm = Float64(get(c, "offset_mm", 0.0))
        slice_wise = get(c, "slice_wise", false) || get(c, "mode", "") == "per_layer"
        
        if c_type == "LeftOf" || c_type == "RightOf"
            axis = 1
            direction = dir_00
        elseif c_type == "AnteriorTo" || c_type == "PosteriorTo"
            axis = 2
            direction = dir_11
        elseif c_type == "SuperiorTo" || c_type == "InferiorTo"
            axis = 3
            direction = dir_22
        else
            direction = 1.0
        end
        
        sp = axis == 1 ? sp_x : (axis == 2 ? sp_y : sp_z)
        offset_vox = round(Int, offset_mm / sp)
        
        lm_chs = Int[]
        for lm in get(c, "landmark_channels", [])
            ch = Int(lm)
            if ch > 0
                push!(lm_chs, ch)
            end
        end
        
        # If a pre-resolved limit index is provided, use it directly
        if haskey(c, "limit_idx")
            limit_idx = c["limit_idx"]
            apply_constraint!(backend, out_arr_gpu, c_type, axis, limit_idx, offset_vox, dir_00, dir_11, dir_22)
            continue
        end
        
        # Otherwise, resolve from landmark channels
        if isempty(lm_chs)
            # fallback to single channel
            ch = get(c, "landmark_channel", 0)
            if ch > 0
                push!(lm_chs, ch)
            end
        end
        
        if isempty(lm_chs)
            println("WARNING: Constraint $c_type has no landmark_channels or limit_idx. Skipping.")
            continue
        end
        
        lm_arrs = [get_channel_view(tm, ch) for ch in lm_chs]
        
        if slice_wise && (axis == 1 || axis == 2)
            # Per-slice constraint
            limits = resolve_landmark_limit_per_slice(lm_arrs, 3, axis, b_part)
            apply_constraint_slicewise!(backend, out_arr_gpu, c_type, axis, limits, offset_vox, dir_00, dir_11, dir_22)
        else
            limit = resolve_landmark_limit(lm_arrs, axis, b_part)
            if limit === nothing
                println("WARNING: Landmark channels $(lm_chs) are empty for constraint $c_type. Skipping.")
                continue
            end
            apply_constraint!(backend, out_arr_gpu, c_type, axis, limit, offset_vox, dir_00, dir_11, dir_22)
        end
    end
end
