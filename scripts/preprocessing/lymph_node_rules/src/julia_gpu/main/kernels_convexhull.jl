using KernelAbstractions
using ImageMorphology

export run_volumetric_boundary_2d!, run_convex_hull_bridge!, run_volumetric_convex_hull_3d!

function fill_convexhull!(out_slice, pts)
    n = length(pts)
    if n < 3
        for p in pts
            out_slice[p] = true
        end
        return
    end
    dims = size(out_slice)
    for y in 1:dims[2]
        min_x = dims[1] + 1
        max_x = 0
        for i in 1:n
            p1 = pts[i]
            p2 = pts[i == n ? 1 : i+1]
            y1 = p1[2]
            y2 = p2[2]
            x1 = p1[1]
            x2 = p2[1]
            if (y1 <= y && y2 > y) || (y2 <= y && y1 > y)
                x_int = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                x_int_round = round(Int, x_int)
                if x_int_round < min_x
                    min_x = x_int_round
                end
                if x_int_round > max_x
                    max_x = x_int_round
                end
            end
            if y1 == y
                if x1 < min_x
                    min_x = x1
                end
                if x1 > max_x
                    max_x = x1
                end
            end
        end
        if min_x <= max_x
            min_x = max(1, min_x)
            max_x = min(dims[1], max_x)
            out_slice[min_x:max_x, y] .= true
        end
    end
end

# ==========================================================
# VolumetricBoundary2D
# Matches Python reference: convex hull of boundaries, subtract obstacles,
# seed from max-distance point, propagate slice-by-slice via flood fill.
# ==========================================================

# Simple 2D flood fill: fills connected region from seed within limit_mask
function flood_fill_2d!(result, seed_mask, limit_mask, dims_xy)
    # Use a queue-based flood fill
    queue = Tuple{Int,Int}[]
    for x in 1:dims_xy[1], y in 1:dims_xy[2]
        if seed_mask[x, y] && limit_mask[x, y]
            push!(queue, (x, y))
            result[x, y] = true
        end
    end
    while !isempty(queue)
        (cx, cy) = popfirst!(queue)
        for (dx, dy) in ((1,0), (-1,0), (0,1), (0,-1))
            nx, ny = cx + dx, cy + dy
            if nx >= 1 && nx <= dims_xy[1] && ny >= 1 && ny <= dims_xy[2]
                if limit_mask[nx, ny] && !result[nx, ny]
                    result[nx, ny] = true
                    push!(queue, (nx, ny))
                end
            end
        end
    end
end

function run_volumetric_boundary_2d!(out_arr, structure_arrs, medial_arrs, lateral_arrs, posterior_arrs=Any[], obstacle_arrs=Any[];
                                    growth_mode="convex_hull", per_layer_margin=false, ap_margin_only=false,
                                    posterior_offset_mm=5.0, min_depth_mm=0.0, erode_iterations=0, sp_y=1.0)
    dims = size(out_arr)
    
    # Download arrays to CPU
    struct_cpu = [adapt(CPU(), arr) for arr in structure_arrs]
    medial_cpu = [adapt(CPU(), arr) for arr in medial_arrs]
    lateral_cpu = [adapt(CPU(), arr) for arr in lateral_arrs]
    post_cpu = [adapt(CPU(), arr) for arr in posterior_arrs]
    obs_cpu = [adapt(CPU(), arr) for arr in obstacle_arrs]
    
    out_host = zeros(UInt8, dims)
    
    boundary_mask = zeros(Bool, dims)
    for arr in struct_cpu
        boundary_mask .|= (arr .> 0)
    end
    
    posterior_mask = zeros(Bool, dims)
    for arr in post_cpu
        posterior_mask .|= (arr .> 0)
    end
    
    medial_mask = zeros(Bool, dims)
    for arr in medial_cpu
        medial_mask .|= (arr .> 0)
    end
    
    lateral_mask = zeros(Bool, dims)
    for arr in lateral_cpu
        lateral_mask .|= (arr .> 0)
    end
    
    obstacle_mask = zeros(Bool, dims)
    for arr in obs_cpu
        obstacle_mask .|= (arr .> 0)
    end
    
    voxel_offset_y = Int(round(posterior_offset_mm / max(0.1, sp_y)))
    min_depth_vox = Int(round(min_depth_mm / max(0.1, sp_y)))
    
    if growth_mode == "sideways" && (any(medial_mask) || any(lateral_mask))
        Threads.@threads for z in 1:dims[3]
            m_slice = view(medial_mask, :, :, z)
            l_slice = view(lateral_mask, :, :, z)
            obs_slice = view(obstacle_mask, :, :, z)
            
            if !any(m_slice) || !any(l_slice)
                continue
            end
            
            m_y_indices = Int[]
            l_y_indices = Int[]
            for y in 1:dims[2]
                if any(view(m_slice, :, y)); push!(m_y_indices, y); end
                if any(view(l_slice, :, y)); push!(l_y_indices, y); end
            end
            
            if isempty(m_y_indices) || isempty(l_y_indices)
                continue
            end
            
            all_y = intersect(m_y_indices, l_y_indices)
            if isempty(all_y)
                all_y = union(m_y_indices, l_y_indices)
            end
            
            for y in all_y
                # Find closest Y in medial
                closest_m_y = m_y_indices[argmin(abs.(m_y_indices .- y))]
                m_x = findall(view(m_slice, :, closest_m_y))
                
                # Find closest Y in lateral
                closest_l_y = l_y_indices[argmin(abs.(l_y_indices .- y))]
                l_x = findall(view(l_slice, :, closest_l_y))
                
                if !isempty(m_x) && !isempty(l_x)
                    min_x = min(minimum(m_x), minimum(l_x))
                    max_x = max(maximum(m_x), maximum(l_x))
                    for x in min_x:max_x
                        if !obs_slice[x, y]
                            out_host[x, y, z] = 1
                        end
                    end
                end
            end
        end
        copyto!(out_arr, out_host)
        return
    end
    
    # 2D Convex Hull per slice
    z_indices = Int[]
    for z in 1:dims[3]
        if any(view(boundary_mask, :, :, z))
            push!(z_indices, z)
        end
    end
    
    if isempty(z_indices)
        copyto!(out_arr, out_host)
        return
    end
    
    Threads.@threads for z in z_indices
        slice_boundary = view(boundary_mask, :, :, z)
        slice_post = view(posterior_mask, :, :, z)
        slice_obs = view(obstacle_mask, :, :, z)
        
        if count(slice_boundary) < 3
            continue
        end
        
        hull_slice = zeros(Bool, dims[1], dims[2])
        try
            pts = convexhull(slice_boundary)
            fill_convexhull!(hull_slice, pts)
        catch
            continue
        end
        
        if any(slice_post)
            if per_layer_margin
                if voxel_offset_y > 0
                    dilated_post = zeros(Bool, dims[1], dims[2])
                    if ap_margin_only
                        for y in 1:dims[2], x in 1:dims[1]
                            if slice_post[x, y]
                                y_start = max(1, y - voxel_offset_y)
                                y_end = min(dims[2], y + voxel_offset_y)
                                dilated_post[x, y_start:y_end] .= true
                            end
                        end
                    else
                        for y in 1:dims[2], x in 1:dims[1]
                            if slice_post[x, y]
                                x_start = max(1, x - voxel_offset_y)
                                x_end = min(dims[1], x + voxel_offset_y)
                                y_start = max(1, y - voxel_offset_y)
                                y_end = min(dims[2], y + voxel_offset_y)
                                dilated_post[x_start:x_end, y_start:y_end] .= true
                            end
                        end
                    end
                    hull_slice .&= dilated_post
                else
                    hull_slice .&= slice_post
                end
            else
                y_post_indices = Int[]
                for y in 1:dims[2]
                    if any(view(slice_post, :, y)); push!(y_post_indices, y); end
                end
                
                slice_ant = slice_boundary .& .!slice_post
                y_ant_indices = Int[]
                for y in 1:dims[2]
                    if any(view(slice_ant, :, y)); push!(y_ant_indices, y); end
                end
                
                if !isempty(y_post_indices)
                    vessel_front_y = minimum(y_post_indices)
                    limit_y = vessel_front_y + voxel_offset_y
                    if !isempty(y_ant_indices)
                        ant_min_y = minimum(y_ant_indices)
                        ant_max_y = maximum(y_ant_indices)
                        if min_depth_vox > 0
                            limit_y = max(limit_y, ant_max_y + min_depth_vox)
                        end
                        if limit_y > ant_min_y + 2 && limit_y <= dims[2]
                            hull_slice[:, limit_y:end] .= false
                        end
                    end
                end
            end
        end
        
        free_space = hull_slice .& .!slice_obs
        out_host[:, :, z] .= UInt8.(free_space)
    end
    
    if erode_iterations > 0
        # Simple morphological erosion
        eroded = copy(out_host)
        for iter in 1:erode_iterations
            prev = copy(eroded)
            for z in 1:dims[3], y in 2:dims[2]-1, x in 2:dims[1]-1
                if prev[x, y, z] > 0
                    if prev[x-1, y, z] == 0 || prev[x+1, y, z] == 0 ||
                       prev[x, y-1, z] == 0 || prev[x, y+1, z] == 0
                        eroded[x, y, z] = 0
                    end
                end
            end
        end
        out_host .= eroded
    end
    
    copyto!(out_arr, out_host)
end

# ==========================================================
# ConvexHullBridge
# ==========================================================

function run_convex_hull_bridge!(out_arr, lm1_arr, lm2_arr)
    # Forward to the new hybrid GPU implementation in RuleExecutors
    backend = KernelAbstractions.get_backend(out_arr)
    res = Main.DagVm.RuleExecutors.execute_convex_hull_bridge(backend, lm1_arr, lm2_arr, size(out_arr), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    copyto!(out_arr, res)
end


function run_volumetric_convex_hull_3d!(backend, out_arr, in_arr)
    dims = size(in_arr)
    in_host = adapt(CPU(), in_arr)
    out_host = zeros(UInt8, dims)
    
    Threads.@threads for z in 1:dims[3]
        slice_2d = in_host[:, :, z] .> 0
        if any(slice_2d)
            try
                pts = convexhull(slice_2d)
                v = view(out_host, :, :, z)
                fill_convexhull!(v, pts)
            catch
                # Fallback to original slice if convex hull fails
                out_host[:, :, z] .= slice_2d
            end
        end
    end
    
    copyto!(out_arr, out_host)
end
