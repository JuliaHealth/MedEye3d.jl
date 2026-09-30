module Landmarks

using CUDA
using KernelAbstractions
using LinearAlgebra
using Statistics

export compute_mask_bounds, compute_mask_centroid, compute_carina_z,
       compute_aortic_arch_z, compute_internal_iliac_points, compute_axillary_geometry,
       compute_digastric_landmarks

"""
    compute_mask_bounds(arr) -> (min_x, max_x, min_y, max_y, min_z, max_z)
Finds the 1-based 3D bounding box of non-zero voxels in array `arr`.
"""
function compute_mask_bounds(arr::AbstractArray{T, 3}) where T
    dims = size(arr)
    any_vox = any(arr .> 0)
    if !any_vox
        return (0, 0, 0, 0, 0, 0)
    end
    
    # Slice projections
    z_proj = dropdims(any(arr .> 0, dims=(1, 2)), dims=(1, 2))
    z_min = findfirst(z_proj)
    z_max = findlast(z_proj)
    
    y_proj = dropdims(any(arr .> 0, dims=(1, 3)), dims=(1, 3))
    y_min = findfirst(y_proj)
    y_max = findlast(y_proj)
    
    x_proj = dropdims(any(arr .> 0, dims=(2, 3)), dims=(2, 3))
    x_min = findfirst(x_proj)
    x_max = findlast(x_proj)
    
    return (Int(x_min), Int(x_max), Int(y_min), Int(y_max), Int(z_min), Int(z_max))
end

"""
    compute_mask_centroid(arr, spacing) -> (cx, cy, cz) [physical mm]
"""
function compute_mask_centroid(arr::AbstractArray{T, 3}, spacing::NTuple{3, Float64}) where T
    indices = findall(arr .> 0)
    if isempty(indices)
        return (0.0, 0.0, 0.0)
    end
    
    cx = mean(idx[1] for idx in indices)
    cy = mean(idx[2] for idx in indices)
    cz = mean(idx[3] for idx in indices)
    
    # 0-based index to physical mm
    return ((cx - 1.0) * spacing[1], (cy - 1.0) * spacing[2], (cz - 1.0) * spacing[3])
end

"""
    compute_carina_z(trachea_arr, spacing) -> (z_vox, z_mm)
"""
function compute_carina_z(trachea_arr::AbstractArray{T, 3}, spacing::NTuple{3, Float64}) where T
    bounds = compute_mask_bounds(trachea_arr)
    if bounds[5] == 0
        return (0, 0.0)
    end
    z_min = bounds[5]
    z_max = bounds[6]
    
    # LPS: Z=1 is inferior, Z=end is superior.
    # Scan from superior (z_max) down to inferior (z_min).
    carina_z = z_min # default
    
    cpu_arr = trachea_arr isa CuArray ? Array(trachea_arr) : trachea_arr
    
    for z in z_max:-1:z_min
        slice_2d = cpu_arr[:, :, z] .> 0
        sum_v = sum(slice_2d)
        if sum_v < 10
            continue
        end
        
        labeled = zeros(Int, size(slice_2d))
        label_count = 0
        for j in 1:size(slice_2d, 2), i in 1:size(slice_2d, 1)
            if slice_2d[i, j] && labeled[i, j] == 0
                label_count += 1
                queue = [(i, j)]
                labeled[i, j] = label_count
                while !isempty(queue)
                    ci, cj = popfirst!(queue)
                    for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
                        ni, nj = ci + di, cj + dj
                        if 1 <= ni <= size(slice_2d, 1) && 1 <= nj <= size(slice_2d, 2)
                            if slice_2d[ni, nj] && labeled[ni, nj] == 0
                                labeled[ni, nj] = label_count
                                push!(queue, (ni, nj))
                            end
                        end
                    end
                end
            end
        end
        
        sizes = zeros(Int, label_count)
        for j in 1:size(slice_2d, 2), i in 1:size(slice_2d, 1)
            if labeled[i, j] > 0
                sizes[labeled[i, j]] += 1
            end
        end
        
        sig_count = sum(sizes .>= 10)
        if sig_count >= 2
            carina_z = z
            break
        end
    end
    
    z_mm = (carina_z - 1.0) * spacing[3]
    return (carina_z, z_mm)
end

"""
    compute_aortic_arch_z(aorta_arr, spacing) -> (z_vox, z_mm)
"""
function compute_aortic_arch_z(aorta_arr::AbstractArray{T, 3}, spacing::NTuple{3, Float64}) where T
    bounds = compute_mask_bounds(aorta_arr)
    if bounds[6] == 0
        return (0, 0.0)
    end
    z_max = bounds[6]
    z_mm = (z_max - 1.0) * spacing[3]
    return (z_max, z_mm)
end

"""
    compute_internal_iliac_p1(art_arr, l5_arr, spacing, origin, direction)
"""

function compute_internal_iliac_p1(art_arr, l5_arr, spacing::NTuple{3, Float64}, origin::NTuple{3, Float64}=(0.0, 0.0, 0.0), direction::NTuple{9, Float64}=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    backend = (art_arr isa CuArray) ? CUDABackend() : CPU()
    art_u8 = (art_arr isa CuArray) ? art_arr : ((art_arr isa Array{UInt8, 3}) ? art_arr : Array{UInt8}(art_arr .> 0))
    dims = size(art_u8)
    
    art_sx, art_sy, art_c = Main.DagVm.RuleExecutors.get_centroid_per_slice_gpu(backend, art_u8, dims)
    
    art_z = findall(art_c .> 0)
    if isempty(art_z)
        return nothing
    end
    
    dir_00 = direction[1]; dir_11 = direction[5]; dir_22 = direction[9]
    
    function idx_to_phys(x, y, z)
        return [origin[1] + (x - 1.0)*spacing[1]*dir_00, 
                origin[2] + (y - 1.0)*spacing[2]*dir_11, 
                origin[3] + (z - 1.0)*spacing[3]*dir_22]
    end
    
    if l5_arr !== nothing
        l5_u8 = (l5_arr isa CuArray) ? l5_arr : ((l5_arr isa Array{UInt8, 3}) ? l5_arr : Array{UInt8}(l5_arr .> 0))
        l5_sx, l5_sy, l5_c = Main.DagVm.RuleExecutors.get_centroid_per_slice_gpu(backend, l5_u8, dims)
        l5_z = findall(l5_c .> 0)
        
        if length(l5_z) > 3
            # 1. Gather L5 centerline in physical space
            l5_pts = []
            for z in l5_z
                cx = Float64(l5_sx[z]) / l5_c[z]
                cy = Float64(l5_sy[z]) / l5_c[z]
                push!(l5_pts, idx_to_phys(cx, cy, Float64(z)))
            end
            
            # 2. PCA to find primary axis
            pts_mat = hcat(l5_pts...) # 3 x N
            com = sum(pts_mat, dims=2) ./ size(pts_mat, 2)
            centered = pts_mat .- com
            cov_mat = (centered * centered') ./ size(pts_mat, 2)
            
            # Eigen decomposition
            # LinearAlgebra.eigen returns sorted eigenvalues in ascending order
            vals, vecs = LinearAlgebra.eigen(cov_mat)
            v1 = vecs[:, 3] # Principal axis
            
            # Ensure normal points Superiorly (+Z in physical, but depends on dir_22)
            # If Z physical increases cranially, we want it pointing up
            if v1[3] < 0
                v1 = -v1
            end
            
            # 3. Find the anatomical bottom of L5 (lowest Z physical, or lowest projected)
            # Projection onto v1
            projections = [LinearAlgebra.dot(p - com[:, 1], v1) for p in l5_pts]
            min_idx = argmin(projections)
            p_bottom = l5_pts[min_idx]
            
            # 4. Find intersection of this oblique plane with the artery centerline
            # We want to find the artery slice where distance to plane crosses 0
            best_z = -1
            min_dist = Inf
            
            for z in art_z
                cx = Float64(art_sx[z]) / art_c[z]
                cy = Float64(art_sy[z]) / art_c[z]
                art_pt = idx_to_phys(cx, cy, Float64(z))
                
                dist = abs(LinearAlgebra.dot(art_pt - p_bottom, v1))
                if dist < min_dist
                    min_dist = dist
                    best_z = z
                end
            end
            
            if best_z != -1 && min_dist < 20.0
                cx = Float64(art_sx[best_z]) / art_c[best_z]
                cy = Float64(art_sy[best_z]) / art_c[best_z]
                return idx_to_phys(cx, cy, Float64(best_z))
            end
        end
    end
    
    # Fallback to Z-min of artery if L5 intersection fails or L5 is missing
    min_art_z = minimum(art_z)
    cx = Float64(art_sx[min_art_z]) / art_c[min_art_z]
    cy = Float64(art_sy[min_art_z]) / art_c[min_art_z]
    return idx_to_phys(cx, cy, Float64(min_art_z))
end

function compute_internal_iliac_p2(sacrum_arr, hip_arr, spacing::NTuple{3, Float64}, origin::NTuple{3, Float64}=(0.0, 0.0, 0.0), direction::NTuple{9, Float64}=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    backend = (sacrum_arr isa CuArray) ? CUDABackend() : CPU()
    s_u8 = (sacrum_arr isa CuArray) ? sacrum_arr : ((sacrum_arr isa Array{UInt8, 3}) ? sacrum_arr : Array{UInt8}(sacrum_arr .> 0))
    h_u8 = (hip_arr isa CuArray) ? hip_arr : ((hip_arr isa Array{UInt8, 3}) ? hip_arr : Array{UInt8}(hip_arr .> 0))
    dims = size(s_u8)
    
    s_sx, s_sy, s_c = Main.DagVm.RuleExecutors.get_centroid_per_slice_gpu(backend, s_u8, dims)
    s_z = findall(s_c .> 0)
    if isempty(s_z)
        return nothing
    end
    
    dir_00 = direction[1]; dir_11 = direction[5]; dir_22 = direction[9]
    min_z = dir_22 > 0 ? minimum(s_z) : maximum(s_z)
    
    h_slice = Array(view(h_u8, :, :, min_z))
    s_slice = Array(view(s_u8, :, :, min_z))
    
    if sum(h_slice) == 0
        for d in 1:15
            z_down = max(1, min_z - d)
            z_up = min(dims[3], min_z + d)
            if sum(Array(view(h_u8, :, :, z_down))) > 0
                min_z = z_down
                h_slice = Array(view(h_u8, :, :, min_z))
                s_slice = Array(view(s_u8, :, :, min_z))
                break
            elseif sum(Array(view(h_u8, :, :, z_up))) > 0
                min_z = z_up
                h_slice = Array(view(h_u8, :, :, min_z))
                s_slice = Array(view(s_u8, :, :, min_z))
                break
            end
        end
    end
    
    if sum(h_slice) == 0; return nothing; end
    
    s_coords = findall(s_slice .> 0)
    h_coords = findall(h_slice .> 0)
    if isempty(s_coords); return nothing; end
    
    min_dist = Inf
    best_h = h_coords[1]
    
    for hc in h_coords[1:3:end]
        hx, hy = hc[1], hc[2]
        for sc in s_coords[1:3:end]
            sx, sy = sc[1], sc[2]
            d = (hx - sx)^2 + (hy - sy)^2
            if d < min_dist
                min_dist = d
                best_h = hc
            end
        end
    end
    
    return [origin[1] + (best_h[1]-1.0)*spacing[1]*dir_00, origin[2] + (best_h[2]-1.0)*spacing[2]*dir_11, origin[3] + (Float64(min_z)-1.0)*spacing[3]*dir_22]
end

"""
    compute_internal_iliac_points(internal_iliac_arr, spacing, origin)
Computes bifurcation point p1 and terminal point p2 for internal iliac artery/vein.
"""
function compute_internal_iliac_points(vessel_arr::AbstractArray{T, 3}, spacing::NTuple{3, Float64}, origin::NTuple{3, Float64}=(0.0, 0.0, 0.0), direction::NTuple{9, Float64}=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)) where T
    indices = findall(vessel_arr .> 0)
    if isempty(indices)
        return (nothing, nothing)
    end
    
    dir_00 = direction[1]
    dir_11 = direction[5]
    dir_22 = direction[9]
    
    min_z = minimum(idx[3] for idx in indices)
    max_z = maximum(idx[3] for idx in indices)
    
    top_indices = filter(idx -> idx[3] == max_z, indices)
    p1_x = mean([Float64(idx[1]) for idx in top_indices])
    p1_y = mean([Float64(idx[2]) for idx in top_indices])
    p1_z = Float64(max_z)
    p1 = [origin[1] + (p1_x - 1.0) * spacing[1] * dir_00, origin[2] + (p1_y - 1.0) * spacing[2] * dir_11, origin[3] + (p1_z - 1.0) * spacing[3] * dir_22]
    
    bottom_indices = filter(idx -> idx[3] == min_z, indices)
    p2_x = mean([Float64(idx[1]) for idx in bottom_indices])
    p2_y = mean([Float64(idx[2]) for idx in bottom_indices])
    p2_z = Float64(min_z)
    p2 = [origin[1] + (p2_x - 1.0) * spacing[1] * dir_00, origin[2] + (p2_y - 1.0) * spacing[2] * dir_11, origin[3] + (p2_z - 1.0) * spacing[3] * dir_22]
    
    return (p1, p2)
end

"""
    compute_axillary_geometry(scapula_arr, rib3_arr, rib5_arr, spacing, origin, direction, side)
"""
function compute_axillary_geometry(scapula_arr::AbstractArray{T, 3},
                                   rib3_arr::AbstractArray{T, 3},
                                   rib5_arr::AbstractArray{T, 3},
                                   spacing::NTuple{3, Float64},
                                   origin::NTuple{3, Float64},
                                   side::String,
                                   direction::NTuple{9, Float64}=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)) where T
    scap_bounds = compute_mask_bounds(scapula_arr)
    rib3_bounds = compute_mask_bounds(rib3_arr)
    rib5_bounds = compute_mask_bounds(rib5_arr)
    
    dir_00 = direction[1]
    dir_11 = direction[5]
    dir_22 = direction[9]
    
    coracoid_x = origin[1] + ((scap_bounds[1] + scap_bounds[2]) / 2.0 - 1.0) * spacing[1] * dir_00
    coracoid_y = origin[2] + (scap_bounds[3] - 1.0) * spacing[2] * dir_11 # anterior edge
    coracoid_z = origin[3] + (scap_bounds[6] - 1.0) * spacing[3] * dir_22 # superior edge
    
    is_left = lowercase(side) == "left"
    lat_sign = is_left ? 1.0 : -1.0
    
    rib3_x_idx, rib3_min_y_idx, rib3_best_z_idx = get_most_anterior_point(rib3_arr)
    rib5_x_idx, rib5_min_y_idx, rib5_best_z_idx = get_most_anterior_point(rib5_arr)
    
    rib5_x = origin[1] + (rib5_x_idx - 1.0) * spacing[1] * dir_00
    rib5_y = origin[2] + (rib5_min_y_idx - 1.0) * spacing[2] * dir_11
    rib5_z = origin[3] + (rib5_best_z_idx - 1.0) * spacing[3] * dir_22
    
    rib3_x = origin[1] + (rib3_x_idx - 1.0) * spacing[1] * dir_00
    rib3_y = origin[2] + (rib3_min_y_idx - 1.0) * spacing[2] * dir_11
    rib3_z = origin[3] + (rib3_best_z_idx - 1.0) * spacing[3] * dir_22
    
    v1_x = rib3_x - coracoid_x
    v1_y = rib3_y - coracoid_y
    v1_z = rib3_z - coracoid_z
    
    v2_x = rib5_x - coracoid_x
    v2_y = rib5_y - coracoid_y
    v2_z = rib5_z - coracoid_z
    
    pm_nx = v1_y * v2_z - v1_z * v2_y
    pm_ny = v1_z * v2_x - v1_x * v2_z
    pm_nz = v1_x * v2_y - v1_y * v2_x
    
    norm_len = sqrt(pm_nx^2 + pm_ny^2 + pm_nz^2)
    if norm_len < 1e-6
        pm_nx, pm_ny, pm_nz = 0.0, 1.0, 0.0
    else
        pm_nx /= norm_len
        pm_ny /= norm_len
        pm_nz /= norm_len
    end
    
    if pm_ny < 0
        pm_nx = -pm_nx
        pm_ny = -pm_ny
        pm_nz = -pm_nz
    end
    
    return Dict{String, Float64}(
        "coracoid_x" => coracoid_x,
        "coracoid_y" => coracoid_y,
        "coracoid_z" => coracoid_z,
        "rib5_x" => rib5_x,
        "rib5_y" => rib5_y,
        "rib5_z" => rib5_z,
        "rib3_x" => rib3_x,
        "rib3_y" => rib3_y,
        "rib3_z" => rib3_z,
        "pm_nx" => pm_nx,
        "pm_ny" => pm_ny,
        "pm_nz" => pm_nz,
        "lat_nx" => lat_sign * 0.7071,
        "lat_ny" => 0.7071,
        "lat_nz" => 0.0,
        "med_nx" => -lat_sign * 0.7071,
        "med_ny" => 0.7071,
        "med_nz" => 0.0,
        "scap_post_y" => origin[2] + (scap_bounds[4] - 1.0) * spacing[2]
    )
end

"""
    compute_digastric_landmarks(mandible_arr, hyoid_arr, spacing, side; is_lps=true)
Computes dig_top (mandible based) and dig_bot (hyoid based).
"""
function compute_digastric_landmarks(mandible_arr::Union{AbstractArray{T, 3}, Nothing},
                                     hyoid_arr::Union{AbstractArray{T, 3}, Nothing},
                                     spacing::NTuple{3, Float64},
                                     origin::NTuple{3, Float64},
                                     side::String;
                                     is_lps::Bool=true) where T
    dig_top = nothing
    if mandible_arr !== nothing
        indices = findall(mandible_arr .> 0)
        if !isempty(indices)
            xs = [idx[1] for idx in indices]
            ys = [idx[2] for idx in indices]
            zs = [idx[3] for idx in indices]
            
            med_x = median(xs)
            midline_mask = abs.(xs .- med_x) .< 15.0
            if !any(midline_mask)
                midline_mask = trues(length(xs))
            end
            
            # Filter indices that are in midline
            midline_indices = indices[midline_mask]
            
            # Find anterior point
            # In LPS, anterior is minimum Y. In RAS, anterior is maximum Y.
            if is_lps
                ant_idx = argmin([idx[2] for idx in midline_indices])
            else
                ant_idx = argmax([idx[2] for idx in midline_indices])
            end
            p_ant_idx = midline_indices[ant_idx]
            
            p_ant = [origin[1] + (p_ant_idx[1] - 1.0) * spacing[1], origin[2] + (p_ant_idx[2] - 1.0) * spacing[2], origin[3] + (p_ant_idx[3] - 1.0) * spacing[3]]
            
            # Menton + 15mm Superior + 15mm Lateral
            lat_off = 0.0
            if is_lps
                lat_off = side == "left" ? 15.0 : -15.0
            else
                lat_off = side == "right" ? 15.0 : -15.0
            end
            
            dig_top = [p_ant[1] + lat_off, p_ant[2], p_ant[3] + 15.0]
        end
    end
    
    dig_bot = nothing
    if hyoid_arr !== nothing
        indices = findall(hyoid_arr .> 0)
        if !isempty(indices)
            xs = [idx[1] for idx in indices]
            ys = [idx[2] for idx in indices]
            zs = [idx[3] for idx in indices]
            
            y_mean = mean(ys)
            # Anterior half mask (in LPS min Y is anterior, in RAS max Y is anterior)
            ant_h_mask = is_lps ? (ys .< y_mean) : (ys .> y_mean)
            
            ant_h_indices = indices[ant_h_mask]
            if !isempty(ant_h_indices)
                ant_xs = [idx[1] for idx in ant_h_indices]
                if side == "left"
                    extreme_idx = argmax(ant_xs)
                else
                    extreme_idx = argmin(ant_xs)
                end
                
                p_ex = ant_h_indices[extreme_idx]
                dig_bot = [origin[1] + (p_ex[1] - 1.0) * spacing[1], origin[2] + (p_ex[2] - 1.0) * spacing[2], origin[3] + (p_ex[3] - 1.0) * spacing[3]]
            end
        end
    end
    
    # Compute plane if both exist
    digastric_plane = nothing
    if dig_top !== nothing && dig_bot !== nothing
        v_line = dig_bot .- dig_top
        # v_perp = cross(v_line, [0, 0, 1])
        # In Julia cross of 3-vectors:
        v_perp = [v_line[2]*1.0 - v_line[3]*0.0, v_line[3]*0.0 - v_line[1]*1.0, v_line[1]*0.0 - v_line[2]*0.0]
        
        # Ensure normal always points MEDIAL
        # In LPS, +X is Left. Left Medial is -X. Right Medial is +X.
        if is_lps
            if side == "left" && v_perp[1] > 0
                v_perp = -v_perp
            elseif side == "right" && v_perp[1] < 0
                v_perp = -v_perp
            end
        else
            if side == "left" && v_perp[1] < 0
                v_perp = -v_perp
            elseif side == "right" && v_perp[1] > 0
                v_perp = -v_perp
            end
        end
        # Normalize
        norm_val = norm(v_perp)
        if norm_val > 0
            v_perp = v_perp ./ norm_val
        end
        digastric_plane = v_perp
    end
    
    return dig_top, dig_bot, digastric_plane
end


function get_most_anterior_point(arr::AbstractArray{T, 3}) where T
    sz = size(arr)
    min_y = sz[2]
    best_x = 1
    best_z = sz[3]
    for z in 1:sz[3]
        for y in 1:sz[2]
            for x in 1:sz[1]
                if arr[x, y, z] > 0
                    if y < min_y
                        min_y = y
                        best_x = x
                        best_z = z
                    end
                end
            end
        end
    end
    return Float64(best_x), Float64(min_y), Float64(best_z)
end

end # module

function compute_scapula_anterior_z(scapula_arr, spacing::NTuple{3, Float64}, origin::NTuple{3, Float64}=(0.0, 0.0, 0.0), direction::NTuple{9, Float64}=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    # We want the Z coordinate of the most anterior voxel (minimum Y in LPS, or depends on direction[5])
    # LPS: +Y is Posterior. Anterior is min Y.
    # RAS: +Y is Anterior. Anterior is max Y.
    backend = (scapula_arr isa CuArray) ? CUDABackend() : CPU()
    s_u8 = (scapula_arr isa CuArray) ? scapula_arr : ((scapula_arr isa Array{UInt8, 3}) ? scapula_arr : Array{UInt8}(scapula_arr .> 0))
    
    dims = size(s_u8)
    if backend isa CUDABackend
        # GPU path: use 1D Y-projection to find anterior bound (~2 KB transfer)
        y_proj = Array(dropdims(any(s_u8 .> UInt8(0), dims=(1,3)), dims=(1,3)))
        y_inds = findall(y_proj)
        if isempty(y_inds); return nothing; end
        
        is_ras_y = direction[5] > 0.0
        best_y = is_ras_y ? maximum(y_inds) : minimum(y_inds)
        
        # Find the Z of voxels at that Y
        z_proj_at_y = Array(dropdims(any(view(s_u8, :, best_y:best_y, :) .> UInt8(0), dims=(1,2)), dims=(1,2)))
        z_inds = findall(z_proj_at_y)
        if isempty(z_inds); return nothing; end
        
        best_z = z_inds[1]  # Any Z at the most anterior Y
        dir_22 = direction[9]
        return origin[3] + (Float64(best_z) - 1.0) * spacing[3] * dir_22
    else
        # CPU path
        s_cpu = s_u8
        coords = findall(s_cpu .> 0)
        if isempty(coords); return nothing; end
        
        is_ras_y = direction[5] > 0.0
        if is_ras_y
            best_coord = coords[argmax([c[2] for c in coords])]
        else
            best_coord = coords[argmin([c[2] for c in coords])]
        end
        
        dir_22 = direction[9]
        return origin[3] + (Float64(best_coord[3]) - 1.0) * spacing[3] * dir_22
    end
end

function compute_posterior_triangle_base_z(cricoid_arr, lung_l_arr, lung_r_arr, spacing::NTuple{3, Float64}, origin::NTuple{3, Float64}=(0.0, 0.0, 0.0), direction::NTuple{9, Float64}=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    backend = (cricoid_arr isa CuArray) ? CUDABackend() : CPU()
    dir_22 = direction[9]
    base_z = -9999.0
    
    if cricoid_arr !== nothing
        c_u8 = (cricoid_arr isa CuArray) ? cricoid_arr : ((cricoid_arr isa Array{UInt8, 3}) ? cricoid_arr : Array{UInt8}(cricoid_arr .> 0))
        _, _, c_c = Main.DagVm.RuleExecutors.get_centroid_per_slice_gpu(backend, c_u8, size(c_u8))
        z_idx = findall(c_c .> 0)
        if !isempty(z_idx)
            z_min = origin[3] + (Float64(minimum(z_idx)) - 1.0) * spacing[3] * dir_22
            base_z = max(base_z, z_min)
        end
    end
    
    for lung in [lung_l_arr, lung_r_arr]
        if lung !== nothing
            l_u8 = (lung isa CuArray) ? lung : ((lung isa Array{UInt8, 3}) ? lung : Array{UInt8}(lung .> 0))
            _, _, l_c = Main.DagVm.RuleExecutors.get_centroid_per_slice_gpu(backend, l_u8, size(l_u8))
            z_idx = findall(l_c .> 0)
            if !isempty(z_idx)
                z_max = origin[3] + (Float64(maximum(z_idx)) - 1.0) * spacing[3] * dir_22
                base_z = max(base_z, z_max)
            end
        end
    end
    
    return base_z == -9999.0 ? nothing : base_z
end
