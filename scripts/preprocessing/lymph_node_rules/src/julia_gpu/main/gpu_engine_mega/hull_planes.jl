# hull_planes.jl — CPU-side per-slice 2D convex hull → half-space planes table
# Called between kernel phases to compute plane equations from edge masks

module HullPlanes

using KernelAbstractions
using Adapt
using CUDA

export compute_planes_table_from_mask, compute_sector_planes_gpu, compute_sector_planes_mega_gpu, fill_hull_gpu!

@kernel function compute_centroids_kernel!(slice_stats, tensor, mask_id, mask_ch, x_dim, y_dim, z_dim)
    i, j, k = @index(Global, NTuple)
    if i <= x_dim && j <= y_dim && k <= z_dim
        if (tensor[i, j, k, mask_ch] & (UInt32(1) << mask_id)) != 0
            CUDA.@atomic slice_stats[1, k] += Int32(i)
            CUDA.@atomic slice_stats[2, k] += Int32(j)
            CUDA.@atomic slice_stats[3, k] += Int32(1)
        end
    end
end

@kernel function find_extremes_kernel!(sector_extremes, tensor, mask_id, mask_ch, slice_stats, dir_x, dir_y, x_dim, y_dim, z_dim, n_sectors)
    i, j, k = @index(Global, NTuple)
    if i <= x_dim && j <= y_dim && k <= z_dim
        if (tensor[i, j, k, mask_ch] & (UInt32(1) << mask_id)) != 0
            cnt = slice_stats[3, k]
            if cnt > 0
                cx = Float32(slice_stats[1, k]) / Float32(cnt)
                cy = Float32(slice_stats[2, k]) / Float32(cnt)
                
                for s in 1:n_sectors
                    proj = (Float32(i) - cx) * dir_x[s] + (Float32(j) - cy) * dir_y[s]
                    proj_int = round(Int64, (proj + 10000f0) * 1024f0)
                    if proj_int < 0; proj_int = 0; end
                    val = (UInt64(proj_int) << 32) | (UInt64(i) << 16) | UInt64(j)
                    
                    CUDA.@atomic sector_extremes[s, k] = max(sector_extremes[s, k], val)
                end
            end
        end
    end
end

@kernel function compute_planes_kernel!(planes_table, num_planes, ext_x, ext_y, sector_extremes, slice_stats, z_dim, n_sectors)
    k = @index(Global, Linear)
    if k <= z_dim
        cnt = slice_stats[3, k]
        if cnt >= 3
            cx = Float32(slice_stats[1, k]) / Float32(cnt)
            cy = Float32(slice_stats[2, k]) / Float32(cnt)
            
            n_unique = 0
            for s in 1:n_sectors
                val = sector_extremes[s, k]
                if val > 0
                    x = Float32((val >> 16) & 0xFFFF)
                    y = Float32(val & 0xFFFF)
                    
                    is_dup = false
                    if n_unique > 0
                        if ext_x[n_unique, k] == x && ext_y[n_unique, k] == y
                            is_dup = true
                        end
                    end
                    
                    if !is_dup
                        n_unique += 1
                        ext_x[n_unique, k] = x
                        ext_y[n_unique, k] = y
                    end
                end
            end
            
            if n_unique > 1 && ext_x[1, k] == ext_x[n_unique, k] && ext_y[1, k] == ext_y[n_unique, k]
                n_unique -= 1
            end
            
            if n_unique >= 3
                valid_planes = 0
                for i in 1:n_unique
                    j = (i % n_unique) + 1
                    x1 = ext_x[i, k]
                    y1 = ext_y[i, k]
                    x2 = ext_x[j, k]
                    y2 = ext_y[j, k]
                    
                    dx = x2 - x1
                    dy = y2 - y1
                    
                    A = -dy
                    B = dx
                    len = sqrt(A*A + B*B)
                    if len >= 1f-10
                        A /= len
                        B /= len
                        
                        D = -(A * x1 + B * y1)
                        
                        if A * cx + B * cy + D > 0
                            A = -A
                            B = -B
                            D = -D
                        end
                        
                        valid_planes += 1
                        planes_table[valid_planes, 1, k] = A
                        planes_table[valid_planes, 2, k] = B
                        planes_table[valid_planes, 3, k] = D
                    end
                end
                num_planes[k] = valid_planes
            else
                num_planes[k] = 0
            end
        else
            num_planes[k] = 0
        end
    end
end

function compute_sector_planes_gpu(backend, tensor, mask_id, mask_ch, dims; n_sectors::Int=32)
    x_dim, y_dim, z_dim = dims
    
    slice_stats = adapt(backend, zeros(Int32, 3, z_dim))
    cc_kernel! = compute_centroids_kernel!(backend, 256)
    cc_kernel!(slice_stats, tensor, UInt32(mask_id), Int32(mask_ch), Int32(x_dim), Int32(y_dim), Int32(z_dim), ndrange=dims)
    KernelAbstractions.synchronize(backend)
    
    sector_extremes = adapt(backend, zeros(UInt64, n_sectors, z_dim))
    sector_angles = range(0f0, 2f0 * Float32(π), length=n_sectors+1)[1:n_sectors]
    dir_x = adapt(backend, cos.(sector_angles))
    dir_y = adapt(backend, sin.(sector_angles))
    
    fe_kernel! = find_extremes_kernel!(backend, 256)
    fe_kernel!(sector_extremes, tensor, UInt32(mask_id), Int32(mask_ch), slice_stats, dir_x, dir_y, Int32(x_dim), Int32(y_dim), Int32(z_dim), Int32(n_sectors), ndrange=dims)
    KernelAbstractions.synchronize(backend)
    
    planes_table = adapt(backend, zeros(Float32, n_sectors, 3, z_dim))
    num_planes = adapt(backend, zeros(Int32, z_dim))
    ext_x = adapt(backend, zeros(Float32, n_sectors, z_dim))
    ext_y = adapt(backend, zeros(Float32, n_sectors, z_dim))
    
    cp_kernel! = compute_planes_kernel!(backend, 256)
    cp_kernel!(planes_table, num_planes, ext_x, ext_y, sector_extremes, slice_stats, Int32(z_dim), Int32(n_sectors), ndrange=z_dim)
    KernelAbstractions.synchronize(backend)
    
    return planes_table, num_planes
end

# --- Mega V2 specific kernels (CH, X, Y, Z layout) ---
@kernel function compute_centroids_mega_kernel!(slice_stats, tensor_out, out_ch, x_dim, y_dim, z_dim)
    i, j, k = @index(Global, NTuple)
    I = Int64(i); J = Int64(j); K = Int64(k); C = Int64(out_ch)
    if I <= Int64(x_dim) && J <= Int64(y_dim) && K <= Int64(z_dim)
        if tensor_out[C, I, J, K] > UInt8(0)
            CUDA.@atomic slice_stats[1, K] += Int32(I)
            CUDA.@atomic slice_stats[2, K] += Int32(J)
            CUDA.@atomic slice_stats[3, K] += Int32(1)
        end
    end
end

@kernel function find_extremes_mega_kernel!(sector_extremes, tensor_out, out_ch, slice_stats, dir_x, dir_y, x_dim, y_dim, z_dim, n_sectors)
    i, j, k = @index(Global, NTuple)
    I = Int64(i); J = Int64(j); K = Int64(k); C = Int64(out_ch)
    if I <= Int64(x_dim) && J <= Int64(y_dim) && K <= Int64(z_dim)
        if tensor_out[C, I, J, K] > UInt8(0)
            cnt = slice_stats[3, K]
            if cnt > 0
                cx = Float32(slice_stats[1, K]) / Float32(cnt)
                cy = Float32(slice_stats[2, K]) / Float32(cnt)
                
                for s in 1:Int64(n_sectors)
                    proj = (Float32(I) - cx) * dir_x[s] + (Float32(J) - cy) * dir_y[s]
                    proj_int = round(Int64, (proj + 10000f0) * 1024f0)
                    if proj_int < 0; proj_int = 0; end
                    val = (UInt64(proj_int) << 32) | (UInt64(I) << 16) | UInt64(J)
                    
                    CUDA.@atomic sector_extremes[s, K] = max(sector_extremes[s, K], val)
                end
            end
        end
    end
end

function compute_sector_planes_mega_gpu(backend, tensor_out, out_ch, dims; n_sectors::Int=32)
    x_dim, y_dim, z_dim = dims
    
    slice_stats = adapt(backend, zeros(Int32, 3, z_dim))
    cc_kernel! = compute_centroids_mega_kernel!(backend, 256)
    cc_kernel!(slice_stats, tensor_out, Int32(out_ch), Int32(x_dim), Int32(y_dim), Int32(z_dim), ndrange=dims)
    KernelAbstractions.synchronize(backend)
    
    sector_extremes = adapt(backend, zeros(UInt64, n_sectors, z_dim))
    sector_angles = range(0f0, 2f0 * Float32(π), length=n_sectors+1)[1:n_sectors]
    dir_x = adapt(backend, cos.(sector_angles))
    dir_y = adapt(backend, sin.(sector_angles))
    
    fe_kernel! = find_extremes_mega_kernel!(backend, 256)
    fe_kernel!(sector_extremes, tensor_out, Int32(out_ch), slice_stats, dir_x, dir_y, Int32(x_dim), Int32(y_dim), Int32(z_dim), Int32(n_sectors), ndrange=dims)
    KernelAbstractions.synchronize(backend)
    
    planes_table = adapt(backend, zeros(Float32, n_sectors, 3, z_dim))
    num_planes = adapt(backend, zeros(Int32, z_dim))
    ext_x = adapt(backend, zeros(Float32, n_sectors, z_dim))
    ext_y = adapt(backend, zeros(Float32, n_sectors, z_dim))
    
    cp_kernel! = compute_planes_kernel!(backend, 256)
    cp_kernel!(planes_table, num_planes, ext_x, ext_y, sector_extremes, slice_stats, Int32(z_dim), Int32(n_sectors), ndrange=z_dim)
    KernelAbstractions.synchronize(backend)
    
    return planes_table, num_planes
end

@kernel function fill_hull_gpu!(tensor_out, out_ch, planes_table, num_planes, dims_x::Int32, dims_y::Int32, dims_z::Int32, max_planes::Int32)
    I, J, K = @index(Global, NTuple)
    if I <= dims_x && J <= dims_y && K <= dims_z
        np = num_planes[K]
        if np > 0
            inside = true
            @inbounds for p in 1:max_planes
                if p > np; break; end
                A = planes_table[p, 1, K]
                B = planes_table[p, 2, K]
                D = planes_table[p, 3, K]
                if A * Float32(I) + B * Float32(J) + D > 1f-5
                    inside = false
                    break
                end
            end
            if inside
                @inbounds tensor_out[out_ch, I, J, K] = UInt8(1)
            end
        end
    end
end

"""
    compute_planes_table_from_mask(edge_mask::Array{UInt8,3})

Given a 3D binary edge mask, compute per-slice 2D convex hulls and return
a planes table for the half-space kernel.

Returns (planes_table, num_planes) where:
  - planes_table: Float32[max_planes, 3, Z] — A, B, D per plane per slice
  - num_planes:   Int32[Z] — count of active planes per slice
"""
function compute_planes_table_from_mask(edge_mask::Array{UInt8,3}; n_sectors::Int=32)
    X, Y, Z = size(edge_mask)
    
    # Sector direction vectors (pre-computed)
    sector_angles = range(0f0, 2f0 * Float32(π), length=n_sectors+1)[1:n_sectors]
    dir_x = cos.(sector_angles)
    dir_y = sin.(sector_angles)
    
    hulls = Vector{Vector{Tuple{Float32,Float32,Float32}}}(undef, Z)
    max_np = 0
    
    for z in 1:Z
        pts = extract_2d_points(edge_mask, z)
        if length(pts) < 3
            hulls[z] = Tuple{Float32,Float32,Float32}[]
            continue
        end
        
        # Compute centroid
        cx = sum(p[1] for p in pts) / length(pts)
        cy = sum(p[2] for p in pts) / length(pts)
        
        # Find extreme point for each sector
        extreme_pts = Tuple{Float32,Float32}[]
        for s in 1:n_sectors
            best_proj = -Inf32
            best_pt = (Float32(pts[1][1]), Float32(pts[1][2]))
            for p in pts
                proj = (Float32(p[1]) - Float32(cx)) * dir_x[s] + 
                       (Float32(p[2]) - Float32(cy)) * dir_y[s]
                if proj > best_proj
                    best_proj = proj
                    best_pt = (Float32(p[1]), Float32(p[2]))
                end
            end
            push!(extreme_pts, best_pt)
        end
        
        # Remove duplicates (adjacent sectors may pick same point)
        unique_pts = [extreme_pts[1]]
        for i in 2:n_sectors
            if extreme_pts[i] != unique_pts[end]
                push!(unique_pts, extreme_pts[i])
            end
        end
        # Also check wrap-around
        if length(unique_pts) > 1 && unique_pts[end] == unique_pts[1]
            pop!(unique_pts)
        end
        
        if length(unique_pts) < 3
            hulls[z] = Tuple{Float32,Float32,Float32}[]
            continue
        end
        
        # Compute half-space planes from extreme polygon
        planes = hull_to_planes_from_tuples(unique_pts, Float32(cx), Float32(cy))
        hulls[z] = planes
        max_np = max(max_np, length(planes))
    end
    
    # Pack into planes_table (same format as existing)
    if max_np == 0; max_np = 1; end
    planes_table = zeros(Float32, max_np, 3, Z)
    num_planes_arr = zeros(Int32, Z)
    
    for z in 1:Z
        np = length(hulls[z])
        num_planes_arr[z] = Int32(np)
        for p in 1:np
            A, B, D = hulls[z][p]
            planes_table[p, 1, z] = A
            planes_table[p, 2, z] = B
            planes_table[p, 3, z] = D
        end
    end
    
    return planes_table, num_planes_arr
end

function hull_to_planes_from_tuples(hull::Vector{Tuple{Float32,Float32}}, cx::Float32, cy::Float32)
    n = length(hull)
    planes = Tuple{Float32,Float32,Float32}[]
    for i in 1:n
        j = (i % n) + 1
        x1, y1 = hull[i][1], hull[i][2]
        x2, y2 = hull[j][1], hull[j][2]
        
        dx = x2 - x1
        dy = y2 - y1
        
        A = -dy
        B = dx
        len = sqrt(A*A + B*B)
        if len < 1f-10
            continue
        end
        A /= len
        B /= len
        
        D = -(A * x1 + B * y1)
        
        centroid_val = A * cx + B * cy + D
        if centroid_val > 0
            A = -A
            B = -B
            D = -D
        end
        
        push!(planes, (A, B, D))
    end
    return planes
end

"""
Extract 2D (x, y) coordinates from a Z-slice of the edge mask.
"""
function extract_2d_points(mask::Array{UInt8,3}, z::Int)
    X, Y = size(mask, 1), size(mask, 2)
    pts = Tuple{Int,Int}[]
    @inbounds for y in 1:Y, x in 1:X
        if mask[x, y, z] > 0
            push!(pts, (x, y))
        end
    end
    return pts
end

"""
Compute 2D convex hull using monotone chain algorithm (Andrew's).
Returns hull vertices in counter-clockwise order.
"""
function convex_hull_2d(pts::Vector{Tuple{Int,Int}})
    n = length(pts)
    if n < 3
        return pts
    end
    
    # Sort by x then y
    sorted = sort(pts, by=p -> (p[1], p[2]))
    
    # Remove duplicates
    unique_pts = Tuple{Int,Int}[sorted[1]]
    for i in 2:n
        if sorted[i] != sorted[i-1]
            push!(unique_pts, sorted[i])
        end
    end
    sorted = unique_pts
    n = length(sorted)
    if n < 3
        return sorted
    end
    
    # Build lower hull
    lower = Tuple{Int,Int}[]
    for p in sorted
        while length(lower) >= 2 && cross2d(lower[end-1], lower[end], p) <= 0
            pop!(lower)
        end
        push!(lower, p)
    end
    
    # Build upper hull
    upper = Tuple{Int,Int}[]
    for p in Iterators.reverse(sorted)
        while length(upper) >= 2 && cross2d(upper[end-1], upper[end], p) <= 0
            pop!(upper)
        end
        push!(upper, p)
    end
    
    # Remove last point of each half (it's repeated)
    pop!(lower)
    pop!(upper)
    
    return vcat(lower, upper)
end

"""
2D cross product for convex hull: (b-a) × (c-a)
Positive = counter-clockwise turn.
"""
function cross2d(a::Tuple{Int,Int}, b::Tuple{Int,Int}, c::Tuple{Int,Int})
    return (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1])
end

"""
Convert convex hull polygon vertices to half-space plane equations.
Each edge (v_i, v_{i+1}) generates a plane: Ax + By + D <= 0 for points inside.
Normal points INWARD (toward centroid).
"""
function hull_to_planes(hull::Vector{Tuple{Int,Int}})
    n = length(hull)
    if n < 3
        return Tuple{Float32,Float32,Float32}[]
    end
    
    # Compute centroid for orientation
    cx = sum(p[1] for p in hull) / n
    cy = sum(p[2] for p in hull) / n
    
    planes = Tuple{Float32,Float32,Float32}[]
    for i in 1:n
        j = (i % n) + 1
        x1, y1 = Float32(hull[i][1]), Float32(hull[i][2])
        x2, y2 = Float32(hull[j][1]), Float32(hull[j][2])
        
        # Edge direction
        dx = x2 - x1
        dy = y2 - y1
        
        # Normal perpendicular to edge (outward = left turn for CCW hull)
        # Outward normal: (-dy, dx) for CCW polygon
        A = -dy
        B = dx
        
        # Normalize
        len = sqrt(A*A + B*B)
        if len < 1f-10
            continue
        end
        A /= len
        B /= len
        
        # D such that A*x1 + B*y1 + D = 0 (point on the edge)
        D = -(A * x1 + B * y1)
        
        # Check orientation: centroid should be on the NEGATIVE side (inside)
        centroid_val = A * Float32(cx) + B * Float32(cy) + D
        if centroid_val > 0
            # Flip normal to point outward
            A = -A
            B = -B
            D = -D
        end
        
        push!(planes, (A, B, D))
    end
    
    return planes
end

end  # module HullPlanes
