module RuleExecutors
import Statistics: mean


using KernelAbstractions
using KernelAbstractions: @index
using Adapt
using CUDA
using ImageMorphology
using ..MaskPacker
using ..StaticArena
using ..GpuCCL
include("gpu_edt.jl")
include("gpu_feature_transform.jl")

@kernel function rotter_fill_kernel_internal!(flood, pec_major, pec_minor, dims_x::Int32, dims_y::Int32, dims_z::Int32)
    i, k = @index(Global, NTuple)
    if i <= dims_x && k <= dims_z
        pm_max_y = Int32(0)
        for j in dims_y:-1:1
            if pec_major[i, j, k] > UInt8(0)
                pm_max_y = Int32(j)
                break
            end
        end
        
        pmin_min_y = Int32(dims_y + 1)
        for j in 1:dims_y
            if pec_minor[i, j, k] > UInt8(0)
                pmin_min_y = Int32(j)
                break
            end
        end
        
        if pm_max_y > Int32(0) && pmin_min_y <= dims_y && pm_max_y < pmin_min_y
            for j in pm_max_y:pmin_min_y
                flood[i, j, k] = UInt8(1)
            end
        end
    end
end


export execute_axillary_rtog, execute_relative_geometric_region,
       execute_station_3a_prevascular,
       execute_dynamic_surface_split, execute_iliac_bifurcation,
       execute_external_iliac, execute_common_iliac,
       execute_anisotropic_expansion, execute_cylinder_primitive,
       execute_ellipsoid_primitive, execute_convex_hull_bridge,
       execute_primary_vector, apply_constraints, apply_z_plane_restriction,
       find_most_anterior_point, get_z_bounds_physical, gpu_bounding_box,
       exact_anisotropic_ft, exact_anisotropic_dt, get_largest_connected_component,
       execute_z_propagation, execute_split_mask,
       gpu_exact_edt!, edt_assign_overlap_kernel!

# ============================================================
# GPU Bounding Box & Projection Kernels
# ============================================================

@kernel function bbox_projection_kernel!(
    @Const(mask),
    x_has_fg,
    y_has_fg,
    z_has_fg,
    dims_x::Int32, dims_y::Int32, dims_z::Int32
)
    j, k = @index(Global, NTuple)
    if j <= dims_y && k <= dims_z
        has_any = false
        for i in Int32(1):dims_x
            if mask[i, j, k] > 0
                has_any = true
                x_has_fg[i] = UInt8(1)
            end
        end
        if has_any
            y_has_fg[j] = UInt8(1)
            z_has_fg[k] = UInt8(1)
        end
    end
end

"""
    gpu_bounding_box(backend, mask) -> (x_min, x_max, y_min, y_max, z_min, z_max) or nothing
Computes 1-based voxel index bounding box entirely on GPU via fast 2D projection kernel.
Transfers only 1.3 KB of projected profile to CPU.
"""
function gpu_bounding_box(backend::KernelAbstractions.Backend, mask::AbstractArray)
    dims = size(mask)
    if !(mask isa CuArray) || backend isa CPU
        idx = findall(mask .> 0)
        if isempty(idx)
            return nothing
        end
        return (
            minimum(ci[1] for ci in idx), maximum(ci[1] for ci in idx),
            minimum(ci[2] for ci in idx), maximum(ci[2] for ci in idx),
            minimum(ci[3] for ci in idx), maximum(ci[3] for ci in idx)
        )
    end
    
    x_has_fg = KernelAbstractions.zeros(backend, UInt8, dims[1])
    y_has_fg = KernelAbstractions.zeros(backend, UInt8, dims[2])
    z_has_fg = KernelAbstractions.zeros(backend, UInt8, dims[3])
    
    kernel! = bbox_projection_kernel!(backend)
    kernel!(
        mask, x_has_fg, y_has_fg, z_has_fg,
        Int32(dims[1]), Int32(dims[2]), Int32(dims[3]),
        ndrange=(dims[2], dims[3])
    )
    KernelAbstractions.synchronize(backend)
    
    x_cpu = adapt(Array, x_has_fg)
    y_cpu = adapt(Array, y_has_fg)
    z_cpu = adapt(Array, z_has_fg)
    
    z_min = findfirst(==(UInt8(1)), z_cpu)
    if z_min === nothing
        return nothing
    end
    z_max = findlast(==(UInt8(1)), z_cpu)
    y_min = findfirst(==(UInt8(1)), y_cpu)
    y_max = findlast(==(UInt8(1)), y_cpu)
    x_min = findfirst(==(UInt8(1)), x_cpu)
    x_max = findlast(==(UInt8(1)), x_cpu)
    
    return (Int(x_min), Int(x_max), Int(y_min), Int(y_max), Int(z_min), Int(z_max))
end

@kernel function find_voxel_with_y_kernel!(
    @Const(mask),
    target_y::Int32,
    found_coords,
    dims_x::Int32, dims_z::Int32
)
    i, k = @index(Global, NTuple)
    if i <= dims_x && k <= dims_z
        if mask[i, target_y, k] > 0
            found_coords[1] = i
            found_coords[2] = target_y
            found_coords[3] = k
        end
    end
end

function find_most_anterior_point(mask::AbstractArray, dims, spacing, origin, direction; backend=CUDA.functional() ? CUDABackend() : CPU())
    bbox = gpu_bounding_box(backend, mask)
    if bbox === nothing
        return nothing
    end
    min_i, max_i, min_j, max_j, min_k, max_k = bbox
    
    dir_11 = Float32(direction[5])
    target_j = dir_11 > 0 ? min_j : max_j
    
    mask_gpu = (backend isa CUDABackend && !(mask isa CUDA.CuArray)) ? adapt(backend, UInt8.(mask .> 0)) : mask
    found_coords = KernelAbstractions.zeros(backend, Int32, 3)
    k! = find_voxel_with_y_kernel!(backend)
    k!(mask_gpu, Int32(target_j), found_coords, Int32(dims[1]), Int32(dims[3]), ndrange=(dims[1], dims[3]))
    KernelAbstractions.synchronize(backend)
    
    coords = adapt(Array, found_coords)
    i, j, k_idx = coords[1], coords[2], coords[3]
    if i == 0
        return nothing
    end
    
    dir_00 = Float32(direction[1]); dir_22 = Float32(direction[9])
    
    px = Float32(origin[1]) + Float32(i - 1) * Float32(spacing[1]) * dir_00
    py = Float32(origin[2]) + Float32(j - 1) * Float32(spacing[2]) * dir_11
    pz = Float32(origin[3]) + Float32(k_idx - 1) * Float32(spacing[3]) * dir_22
    
    return (px, py, pz)
end

function get_z_bounds_physical(mask::AbstractArray, dims, spacing, origin, direction; backend=CUDA.functional() ? CUDABackend() : CPU())
    bbox = gpu_bounding_box(backend, mask)
    if bbox === nothing
        return (0f0, 0f0)
    end
    dir_22 = Float32(direction[9])
    k_min = bbox[5] - 1
    k_max = bbox[6] - 1
    z_min = Float32(origin[3]) + Float32(k_min) * Float32(spacing[3]) * dir_22
    z_max = Float32(origin[3]) + Float32(k_max) * Float32(spacing[3]) * dir_22
    return (min(z_min, z_max), max(z_min, z_max))
end

function get_x_bounds_physical(mask::AbstractArray, dims, spacing, origin, direction; backend=CUDA.functional() ? CUDABackend() : CPU())
    bbox = gpu_bounding_box(backend, mask)
    if bbox === nothing
        return (0f0, 0f0)
    end
    dir_00 = Float32(direction[1])
    i_min = bbox[1] - 1
    i_max = bbox[2] - 1
    x_min = Float32(origin[1]) + Float32(i_min) * Float32(spacing[1]) * dir_00
    x_max = Float32(origin[1]) + Float32(i_max) * Float32(spacing[1]) * dir_00
    return (min(x_min, x_max), max(x_min, x_max))
end

function get_y_bounds_physical(mask::AbstractArray, dims, spacing, origin, direction; backend=CUDA.functional() ? CUDABackend() : CPU())
    bbox = gpu_bounding_box(backend, mask)
    if bbox === nothing
        return (0f0, 0f0)
    end
    dir_11 = Float32(direction[5])
    j_min = bbox[3] - 1
    j_max = bbox[4] - 1
    y_min = Float32(origin[2]) + Float32(j_min) * Float32(spacing[2]) * dir_11
    y_max = Float32(origin[2]) + Float32(j_max) * Float32(spacing[2]) * dir_11
    return (min(y_min, y_max), max(y_min, y_max))
end

# ============================================================
# Axillary RTOG Evaluation
# ============================================================

@kernel function axillary_rtog_eval_kernel!(
    output_i::AbstractArray{UInt8, 3},
    output_ii::AbstractArray{UInt8, 3},
    output_iii::AbstractArray{UInt8, 3},
    output_rotter::AbstractArray{UInt8, 3},
    rect_mask::AbstractArray{UInt8, 3},
    pec_dist::AbstractArray{Float32, 3},
    art_dist::AbstractArray{Float32, 3},
    exclusion::AbstractArray{UInt8, 3},
    packed_data::AbstractArray{UInt8, 4},
    excl_ch::AbstractArray{Int32, 1},
    excl_id::AbstractArray{UInt8, 1},
    num_excl::Int32,
    dims_x::Int32, dims_y::Int32, dims_z::Int32,
    sp_x::Float32, sp_y::Float32, sp_z::Float32,
    orig_x::Float32, orig_y::Float32, orig_z::Float32,
    dir_00::Float32, dir_11::Float32, dir_22::Float32,
    coracoid_x::Float32, coracoid_y::Float32, coracoid_z::Float32,
    rib3_x::Float32, rib3_y::Float32, rib3_z::Float32,
    rib5_x::Float32, rib5_y::Float32, rib5_z::Float32,
    norm_x::Float32, norm_y::Float32, norm_z::Float32,
    max_z_clav::Float32, min_z_bound::Float32,
    is_left::Bool,
    has_art::Bool
)
    i, j, k = @index(Global, NTuple)

    if i <= dims_x && j <= dims_y && k <= dims_z
        pos_x = orig_x + (Float32(i) - 1.0f0) * sp_x * dir_00
        pos_y = orig_y + (Float32(j) - 1.0f0) * sp_y * dir_11
        pos_z = orig_z + (Float32(k) - 1.0f0) * sp_z * dir_22

        # 1. Bounding check
        in_bounds = (pos_z <= max_z_clav) && (pos_z >= min_z_bound)
        is_excl = exclusion[i, j, k] > 0
        
        if in_bounds && !is_excl
            for idx in 1:num_excl
                if packed_data[i, j, k, excl_ch[idx]] == excl_id[idx]
                    is_excl = true
                    break
                end
            end
        end

        if in_bounds && !is_excl
            # Lateral boundary line from coracoid to rib 5
            dz_lat = coracoid_z - rib5_z
            dz_lat = abs(dz_lat) < 0.00001f0 ? 0.00001f0 : dz_lat
            t_lat = (pos_z - rib5_z) / dz_lat
            x_lat = rib5_x + t_lat * (coracoid_x - rib5_x)

            # Medial boundary line from coracoid to rib 3
            dz_med = coracoid_z - rib3_z
            dz_med = abs(dz_med) < 0.00001f0 ? 0.00001f0 : dz_med
            t_med = (pos_z - rib3_z) / dz_med
            x_med = rib3_x + t_med * (coracoid_x - rib3_x)

            # Level spatial masks in LPS
            level_1 = is_left ? (pos_x >= x_lat) : (pos_x <= x_lat)
            level_2 = is_left ? ((pos_x < x_lat) && (pos_x >= x_med)) : ((pos_x > x_lat) && (pos_x <= x_med))
            level_3 = is_left ? (pos_x < x_med) : (pos_x > x_med)

            # Distance to PM plane
            plane_dist = (pos_x - coracoid_x) * norm_x + (pos_y - coracoid_y) * norm_y + (pos_z - coracoid_z) * norm_z
            deep_to_pm = plane_dist >= -10.0f0
            ant_to_pm  = plane_dist < 10.0f0

            p_dist = pec_dist[i, j, k]
            a_dist = art_dist[i, j, k]

            # Level I
            if level_1 && rect_mask[i, j, k] > 0
                output_i[i, j, k] = UInt8(1)
            end

            # Level II
            if level_2 && rect_mask[i, j, k] > 0
                output_ii[i, j, k] = UInt8(1)
            end

            # Level III
            mid_phys_x = orig_x + Float32(div(dims_x, Int32(2))) * sp_x * dir_00
            medial_limit_iii = is_left ? ((pos_x >= (x_med - 35.0f0)) && (pos_x >= (mid_phys_x + 15.0f0))) :
                                         ((pos_x <= (x_med + 35.0f0)) && (pos_x <= (mid_phys_x - 15.0f0)))
            z_bound_iii = pos_z >= (coracoid_z - 55.0f0)
            flood_iii = has_art ? (a_dist <= 25.0f0) : (p_dist <= 30.0f0)

            if level_3 && deep_to_pm && medial_limit_iii && z_bound_iii && flood_iii
                output_iii[i, j, k] = UInt8(1)
            end

            
        end
    end
end

"""
    compute_distance_to_mask(backend, mask, dims, spacing)
Computes 3D Euclidean distance (in mm) from every voxel to the nearest foreground voxel of mask.
"""
function compute_distance_to_mask!(backend::KernelAbstractions.Backend, out::AbstractArray{Float32, 3}, mask::AbstractArray, dims, spacing)
    mask_gpu = mask isa CUDA.CuArray ? mask : adapt(backend, mask)
    has_fg = false
    if backend isa CUDABackend
        # Use GPU reduction without @allowscalar or temporary 3D allocation
        has_fg = sum(mask_gpu) > 0
    else
        has_fg = any(mask_gpu .> UInt8(0))
    end
    if !has_fg
        if backend isa CUDABackend; CUDA.fill!(out, 1000.0f0); else fill!(out, 1000.0f0); end
        return out
    end
    seed_gpu = mask_gpu isa CuArray{UInt8, 3} ? mask_gpu : adapt(backend, UInt8.(mask_gpu .> 0))
    gpu_exact_edt!(out, backend, seed_gpu, Float32.(spacing))
    out .= sqrt.(out)
    return out
end

function compute_distance_to_mask(backend::KernelAbstractions.Backend, mask::AbstractArray, dims, spacing)
    out = KernelAbstractions.allocate(backend, Float32, dims)
    return compute_distance_to_mask!(backend, out, mask, dims, spacing)
end

@kernel function dist_field_kernel!(
    out_dist::AbstractArray{Float32, 3},
    fg_x::AbstractArray{Float32, 1},
    fg_y::AbstractArray{Float32, 1},
    fg_z::AbstractArray{Float32, 1},
    num_fg::Int32,
    dims_x::Int32, dims_y::Int32, dims_z::Int32,
    sp_x::Float32, sp_y::Float32, sp_z::Float32
)
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        px = (Float32(i) - 1.0f0) * sp_x
        py = (Float32(j) - 1.0f0) * sp_y
        pz = (Float32(k) - 1.0f0) * sp_z

        min_d_sq = 1.0f8
        for idx in 1:num_fg
            dx = px - fg_x[idx]
            dy = py - fg_y[idx]
            dz = pz - fg_z[idx]
            d_sq = dx*dx + dy*dy + dz*dz
            if d_sq < min_d_sq
                min_d_sq = d_sq
            end
        end
        out_dist[i, j, k] = sqrt(min_d_sq)
    end
end

"""
    execute_axillary_rtog(backend, get_mask_fn, side, dims, spacing, origin, direction)
"""

@kernel function axillary_rectangle_kernel!(output, pec, sub, scapula, has_scapula, dims, is_left, medial_vox, lateral_vox)
    K = @index(Global, Linear)
    if K <= dims[3]
        pec_min_y = dims[2] + 1
        pec_max_y = 0
        pec_sum_x = 0
        pec_count = 0
        
        sub_min_y = dims[2] + 1
        sub_max_y = 0
        sub_sum_x = 0
        sub_count = 0
        
        scap_min_y = dims[2] + 1
        scap_count = 0
        
        for j in 1:dims[2]
            for i in 1:dims[1]
                if pec[i, j, K] > 0
                    pec_min_y = min(pec_min_y, j)
                    pec_max_y = max(pec_max_y, j)
                    pec_sum_x += i
                    pec_count += 1
                end
                if sub[i, j, K] > 0
                    sub_min_y = min(sub_min_y, j)
                    sub_max_y = max(sub_max_y, j)
                    sub_sum_x += i
                    sub_count += 1
                end
                if has_scapula && scapula[i, j, K] > 0
                    scap_min_y = min(scap_min_y, j)
                    scap_count += 1
                end
            end
        end
        
        if pec_count > 0 && sub_count > 0 && sub_max_y > pec_min_y
            y_start = pec_max_y
            y_end = scap_count > 0 ? scap_min_y : sub_max_y
            if y_start < y_end
                pec_cx = pec_sum_x ÷ pec_count
                sub_cx = sub_sum_x ÷ sub_count
                ref_x = (pec_cx + sub_cx) ÷ 2
                
                x_med = is_left ? ref_x - medial_vox : ref_x + medial_vox
                x_lat = is_left ? ref_x + lateral_vox : ref_x - lateral_vox
                
                x_min = min(x_med, x_lat)
                x_max = max(x_med, x_lat)
                
                x_min = max(1, x_min)
                x_max = min(dims[1], x_max)
                
                for j in y_start:y_end
                    for i in x_min:x_max
                        output[i, j, K] = 1
                    end
                end
            end
        end
    end
end

function execute_axillary_rtog(
    backend::KernelAbstractions.Backend,
    get_mask_fn::Function,
    get_mask_into_fn::Function,
    packed_tensor,
    side::String,
    dims::Tuple{Int, Int, Int},
    spacing::Tuple{Float64, Float64, Float64},
    origin::Tuple{Float64, Float64, Float64},
    direction::Tuple
)
    s_lower = lowercase(side)
    is_left = (s_lower == "left")
    is_ras = direction[1] > 0.0
    keep_larger = is_ras ? !is_left : is_left
    keep_side_str = keep_larger ? "max" : "min"

    # Reclaim GPU memory before heavy EDT allocations
    GC.gc(false)
    if backend isa CUDABackend; CUDA.reclaim(); end


    scapula = get_mask_fn("scapula_$s_lower")
    if scapula === nothing
        scapula = get_mask_fn("scapula")
        if scapula !== nothing
            # split bilateral mask
            scapula = execute_split_mask(backend, scapula, "x", "image_center", keep_side_str, dims)
        end
    end
    
    rib3 = get_mask_fn("rib_$(s_lower)_3")
    if rib3 === nothing; rib3 = get_mask_fn("rib_3_$s_lower"); end
    if rib3 === nothing
        rib3 = get_mask_fn("rib_3")
        if rib3 !== nothing
            # split bilateral mask
            rib3 = execute_split_mask(backend, rib3, "x", "image_center", keep_side_str, dims)
        end
    end
    
    rib5 = get_mask_fn("rib_$(s_lower)_5")
    if rib5 === nothing; rib5 = get_mask_fn("rib_5_$s_lower"); end
    if rib5 === nothing
        rib5 = get_mask_fn("rib_5")
        if rib5 !== nothing
            # split bilateral mask
            rib5 = execute_split_mask(backend, rib5, "x", "image_center", keep_side_str, dims)
        end
    end
    
    pec_major = get_mask_fn("pectoralis_major_$s_lower")
    if pec_major === nothing; pec_major = get_mask_fn("pectoralis_major"); end
    
    pec_minor = get_mask_fn("pectoralis_minor_$s_lower")
    if pec_minor === nothing; pec_minor = get_mask_fn("pectoralis_minor"); end
    
    
    subclavian = get_mask_fn("subclavian_artery_$s_lower")
    if subclavian === nothing; subclavian = get_mask_fn("subclavian_artery"); end
    
    subscapularis = get_mask_fn("subscapularis_$s_lower")
    if subscapularis === nothing; subscapularis = get_mask_fn("subscapularis"); end

    
    clavicula = get_mask_fn("clavicula_$s_lower")
    if clavicula === nothing
        clavicula = get_mask_fn("clavicula")
        if clavicula !== nothing
            # split bilateral mask
            clavicula = execute_split_mask(backend, clavicula, "x", "image_center", keep_side_str, dims)
        end
    end
    
    if scapula === nothing || rib3 === nothing || rib5 === nothing || pec_major === nothing
        println("  [AxillaryRTOG] Missing critical landmarks for side $side. Skipping.")
        return Dict{String, AbstractArray{UInt8, 3}}()
    end
    
    coracoid_pt = find_most_anterior_point(scapula, dims, spacing, origin, direction; backend=backend)
    rib3_pt = find_most_anterior_point(rib3, dims, spacing, origin, direction; backend=backend)
    rib5_pt = find_most_anterior_point(rib5, dims, spacing, origin, direction; backend=backend)
    
    if coracoid_pt === nothing || rib3_pt === nothing || rib5_pt === nothing
        println("  [AxillaryRTOG] Failed to find anterior landmark points for side $side.")
        return Dict{String, AbstractArray{UInt8, 3}}()
    end
    
    v1 = (rib3_pt[1] - coracoid_pt[1], rib3_pt[2] - coracoid_pt[2], rib3_pt[3] - coracoid_pt[3])
    v2 = (rib5_pt[1] - coracoid_pt[1], rib5_pt[2] - coracoid_pt[2], rib5_pt[3] - coracoid_pt[3])
    
    nx = v1[2]*v2[3] - v1[3]*v2[2]
    ny = v1[3]*v2[1] - v1[1]*v2[3]
    nz = v1[1]*v2[2] - v1[2]*v2[1]
    n_len = sqrt(nx*nx + ny*ny + nz*nz)
    
    if n_len < 1e-5
        nx = 0f0; ny = 1f0; nz = 0f0
    else
        nx /= n_len; ny /= n_len; nz /= n_len
    end
    # Find center of pectoralis major to orient the normal robustly
    pec_bbox = RuleExecutors.get_z_bounds_physical(pec_major, dims, spacing, origin, direction; backend=backend)
    pec_cx = origin[1] + (dims[1] ÷ 2) * spacing[1] * direction[1] # approx
    # Actually, let's just find the first voxel of pec_major
    pec_pt = RuleExecutors.find_most_anterior_point(pec_major, dims, spacing, origin, direction; backend=backend)
    if pec_pt !== nothing
        # dot product of pec_pt with normal
        d_pec = (pec_pt[1] - coracoid_pt[1]) * nx + (pec_pt[2] - coracoid_pt[2]) * ny + (pec_pt[3] - coracoid_pt[3]) * nz
        # We want the normal to point POSTERIORLY (away from pec_major, which is anterior).
        # So if d_pec > 0, the normal points towards pec_major (Anterior). We should flip it.
        if d_pec > 0
            nx = -nx; ny = -ny; nz = -nz
        end
    else
        if ny < 0
            nx = -nx; ny = -ny; nz = -nz
        end
    end
    
    max_z_clav = coracoid_pt[3] # Task 4: Explicitly use coracoid process Z as upper bound


    
    if rib5 !== nothing
        min_z_bound = get_z_bounds_physical(rib5, dims, spacing, origin, direction; backend=backend)[1]
    else
        min_z_bound = rib3_pt[3] - 60f0
    end

    
    exclusion, _excl_key = StaticArena.acquire_mask(backend)
    excl_names = [
        "lung", "lung_$s_lower",
        "clavicula", "clavicula_$s_lower",
        "scapula", "scapula_$s_lower",
        "humerus", "humerus_$s_lower",
        "pectoralis_major", "pectoralis_major_$s_lower",
        "latissimus_dorsi", "latissimus_dorsi_$s_lower",
        "serratus_anterior", "serratus_anterior_$s_lower",
        "subclavian_artery", "subclavian_artery_$s_lower",
        "subscapularis", "subscapularis_$s_lower",
        "deltoid", "deltoid_$s_lower",
        "teres_major", "teres_major_$s_lower",
        "teres_minor", "teres_minor_$s_lower",
        "infraspinatus", "infraspinatus_$s_lower",
        "supraspinatus", "supraspinatus_$s_lower",
        "coracobrachialis", "coracobrachialis_$s_lower",
        "helper_subscapularis_band_$s_lower"
    ]
    for i in 1:10
        push!(excl_names, "rib_$(i)_$s_lower")
        push!(excl_names, "rib_$(s_lower)_$i")
        push!(excl_names, "rib_$i")
    end
    m_buf = nothing
    _mbuf_key = nothing
    
    excl_ch = Int32[]
    excl_id = UInt8[]
    
    for name in excl_names
        ch_info = packed_tensor !== nothing ? get(packed_tensor.registry, name, nothing) : nothing
        if ch_info !== nothing
            push!(excl_ch, Int32(ch_info[1]))
            push!(excl_id, UInt8(ch_info[2]))
        else
            if m_buf === nothing
                m_buf, _mbuf_key = StaticArena.acquire_mask(backend)
            end
            if get_mask_into_fn(name, m_buf)
                exclusion .|= m_buf
            end
        end
    end
    if _mbuf_key !== nothing; StaticArena.release_mask(_mbuf_key); end
    num_excl = Int32(length(excl_ch))
    if num_excl == 0
        push!(excl_ch, Int32(-1))
        push!(excl_id, UInt8(0))
    end
    excl_ch_gpu = adapt(backend, excl_ch)
    excl_id_gpu = adapt(backend, excl_id)
    
    pec_dist = KernelAbstractions.allocate(backend, Float32, dims)
    if pec_major !== nothing
        compute_distance_to_mask!(backend, pec_dist, pec_major, dims, spacing)
    else
        if backend isa CUDABackend; CUDA.fill!(pec_dist, 1000.0f0); else fill!(pec_dist, 1000.0f0); end
    end
    has_art = (subclavian !== nothing && gpu_bounding_box(backend, subclavian) !== nothing)
    art_dist = pec_dist
    if has_art
        art_dist = KernelAbstractions.allocate(backend, Float32, dims)
        compute_distance_to_mask!(backend, art_dist, subclavian, dims, spacing)
    end
    
    out_i      = KernelAbstractions.zeros(backend, UInt8, dims)
    out_ii     = KernelAbstractions.zeros(backend, UInt8, dims)
    out_iii    = KernelAbstractions.zeros(backend, UInt8, dims)
    out_rotter = KernelAbstractions.zeros(backend, UInt8, dims)
    
    sp_x = Float32(spacing[1]); sp_y = Float32(spacing[2]); sp_z = Float32(spacing[3])
    orig_x = Float32(origin[1]); orig_y = Float32(origin[2]); orig_z = Float32(origin[3])
    dir_00 = Float32(direction[1]); dir_11 = Float32(direction[5]); dir_22 = Float32(direction[9])
    
    
    rect_mask = KernelAbstractions.zeros(backend, UInt8, dims)
    if pec_major !== nothing && subscapularis !== nothing
        medial_vox = Int32(round(40.0 / spacing[1]))
        lateral_vox = Int32(round(300.0 / spacing[1]))
        kernel_rect! = axillary_rectangle_kernel!(backend)
        has_scap = scapula !== nothing
        dummy_scap = has_scap ? scapula : pec_major
        kernel_rect!(rect_mask, pec_major, subscapularis, dummy_scap, has_scap, Int32.(dims), is_left, medial_vox, lateral_vox, ndrange=dims[3])
        KernelAbstractions.synchronize(backend)
    end
    
    kernel! = axillary_rtog_eval_kernel!(backend)

    kernel!(
        out_i, out_ii, out_iii, out_rotter, rect_mask,
        pec_dist, art_dist, exclusion, packed_tensor !== nothing ? packed_tensor.data : KernelAbstractions.zeros(backend, UInt8, (1, 1, 1, 1)), excl_ch_gpu, excl_id_gpu, num_excl,
        Int32(dims[1]), Int32(dims[2]), Int32(dims[3]),
        sp_x, sp_y, sp_z,
        orig_x, orig_y, orig_z,
        dir_00, dir_11, dir_22,
        coracoid_pt[1], coracoid_pt[2], coracoid_pt[3],
        rib3_pt[1], rib3_pt[2], rib3_pt[3],
        rib5_pt[1], rib5_pt[2], rib5_pt[3],
        nx, ny, nz,
        Float32(max_z_clav), Float32(min_z_bound),
        is_left,
        has_art,
        ndrange=dims
    )
    
    KernelAbstractions.synchronize(backend)
    
    out_rotter = KernelAbstractions.zeros(backend, UInt8, dims)
    if pec_major !== nothing && pec_minor !== nothing
        rotter_fill_kernel_internal!(backend)(out_rotter, pec_major, pec_minor, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=dims)
        KernelAbstractions.synchronize(backend)
    end
    
    StaticArena.release_mask(_excl_key)

    
    prefix = is_left ? "Left" : "Right"
    res = Dict{String, AbstractArray{UInt8, 3}}(
        "helper_Axillary_Level_I_base_$prefix" => out_i,
        "Axillary_Level_II_$prefix" => out_ii,
        "Axillary_Level_III_$prefix" => out_iii,
        "Axillary_Rotter_$prefix" => out_rotter
    )
    return res
end

# ============================================================
# GPU Gather Dilation Kernel (Anisotropic Expansion)
# ============================================================

function exact_anisotropic_ft(img::AbstractArray{Bool, 3}, spacing::NTuple{3, Float64})
    dims = size(img)
    F = fill(CartesianIndex(0,0,0), dims)
    sp_sq = (spacing[1]^2, spacing[2]^2, spacing[3]^2)
    
    # 1. First dimension (X)
    Threads.@threads for k in 1:dims[3]
        for j in 1:dims[2]
            last_i = 0
            for i in 1:dims[1]
                if img[i, j, k]; last_i = i; end
                if last_i > 0; F[i, j, k] = CartesianIndex(last_i, j, k); end
            end
            last_i = 0
            for i in dims[1]:-1:1
                if img[i, j, k]; last_i = i; end
                if last_i > 0
                    if F[i, j, k] == CartesianIndex(0,0,0) || abs(i - last_i) < abs(i - F[i, j, k].I[1])
                        F[i, j, k] = CartesianIndex(last_i, j, k)
                    end
                end
            end
        end
    end
    
    # 2. Second dimension (Y)
    w_y_sq = sp_sq[2]
    Threads.@threads for k in 1:dims[3]
        tmp_g = CartesianIndex{3}[]
        for i in 1:dims[1]
            empty!(tmp_g)
            for j in 1:dims[2]
                fi = F[i, j, k]
                if fi != CartesianIndex(0,0,0)
                    while length(tmp_g) >= 2
                        u = tmp_g[end-1]; v = tmp_g[end]; w = fi
                        u_d = u.I[2]; v_d = v.I[2]; w_d = w.I[2]
                        a = v_d - u_d; b = w_d - v_d; c = w_d - u_d
                        u_d2 = (u.I[1] - i)^2 * sp_sq[1]
                        v_d2 = (v.I[1] - i)^2 * sp_sq[1]
                        w_d2 = (w.I[1] - i)^2 * sp_sq[1]
                        if c * v_d2 - b * u_d2 - a * w_d2 > w_y_sq * a * b * c
                            pop!(tmp_g)
                        else
                            break
                        end
                    end
                    push!(tmp_g, fi)
                end
            end
            
            if !isempty(tmp_g)
                l = 1
                fthis = tmp_g[l]
                for j in 1:dims[2]
                    d2this = (fthis.I[1] - i)^2 * sp_sq[1] + (fthis.I[2] - j)^2 * sp_sq[2]
                    while l < length(tmp_g)
                        fnext = tmp_g[l+1]
                        d2next = (fnext.I[1] - i)^2 * sp_sq[1] + (fnext.I[2] - j)^2 * sp_sq[2]
                        if d2this > d2next
                            d2this = d2next
                            fthis = fnext
                            l += 1
                        else
                            break
                        end
                    end
                    F[i, j, k] = fthis
                end
            end
        end
    end
    
    # 3. Third dimension (Z)
    w_z_sq = sp_sq[3]
    Threads.@threads for j in 1:dims[2]
        tmp_g = CartesianIndex{3}[]
        for i in 1:dims[1]
            empty!(tmp_g)
            for k in 1:dims[3]
                fi = F[i, j, k]
                if fi != CartesianIndex(0,0,0)
                    while length(tmp_g) >= 2
                        u = tmp_g[end-1]; v = tmp_g[end]; w = fi
                        u_d = u.I[3]; v_d = v.I[3]; w_d = w.I[3]
                        a = v_d - u_d; b = w_d - v_d; c = w_d - u_d
                        u_d2 = (u.I[1] - i)^2 * sp_sq[1] + (u.I[2] - j)^2 * sp_sq[2]
                        v_d2 = (v.I[1] - i)^2 * sp_sq[1] + (v.I[2] - j)^2 * sp_sq[2]
                        w_d2 = (w.I[1] - i)^2 * sp_sq[1] + (w.I[2] - j)^2 * sp_sq[2]
                        if c * v_d2 - b * u_d2 - a * w_d2 > w_z_sq * a * b * c
                            pop!(tmp_g)
                        else
                            break
                        end
                    end
                    push!(tmp_g, fi)
                end
            end
            
            if !isempty(tmp_g)
                l = 1
                fthis = tmp_g[l]
                for k in 1:dims[3]
                    d2this = (fthis.I[1] - i)^2 * sp_sq[1] + (fthis.I[2] - j)^2 * sp_sq[2] + (fthis.I[3] - k)^2 * sp_sq[3]
                    while l < length(tmp_g)
                        fnext = tmp_g[l+1]
                        d2next = (fnext.I[1] - i)^2 * sp_sq[1] + (fnext.I[2] - j)^2 * sp_sq[2] + (fnext.I[3] - k)^2 * sp_sq[3]
                        if d2this > d2next
                            d2this = d2next
                            fthis = fnext
                            l += 1
                        else
                            break
                        end
                    end
                    F[i, j, k] = fthis
                end
            end
        end
    end
    
    return F
end

function exact_anisotropic_dt(F::Array{CartesianIndex{3}, 3}, spacing::NTuple{3, Float64})
    dims = size(F)
    D = zeros(Float32, dims)
    sp_sq = (Float32(spacing[1]^2), Float32(spacing[2]^2), Float32(spacing[3]^2))
    
    Threads.@threads for k in 1:dims[3]
        for j in 1:dims[2], i in 1:dims[1]
            fi = F[i, j, k]
            if fi != CartesianIndex(0,0,0)
                dx = Float32(fi.I[1] - i)
                dy = Float32(fi.I[2] - j)
                dz = Float32(fi.I[3] - k)
                D[i, j, k] = sqrt(dx^2 * sp_sq[1] + dy^2 * sp_sq[2] + dz^2 * sp_sq[3])
            else
                D[i, j, k] = Float32(Inf)
            end
        end
    end
    return D
end

function resolve_directional_margins(margins_mm::Any, side::Union{String, Nothing}=nothing)
    resolved = Dict{String, Float32}(
        "left" => 0.0f0, "right" => 0.0f0,
        "anterior" => 0.0f0, "posterior" => 0.0f0,
        "superior" => 0.0f0, "inferior" => 0.0f0
    )
    if margins_mm isa Number
        for k in keys(resolved)
            resolved[k] = Float32(margins_mm)
        end
        return resolved
    elseif !(margins_mm isa AbstractDict)
        for k in keys(resolved)
            resolved[k] = 10.0f0
        end
        return resolved
    end
    
    # Check if this is an all-negative (erosion) specification
    all_neg = any(string(k) != "slice_wise" for k in keys(margins_mm)) && all((v isa Number && Float32(v) < 0f0) for (k, v) in margins_mm if string(k) != "slice_wise")
    if all_neg && haskey(margins_mm, "all")
        v = Float32(margins_mm["all"])
        for axis in keys(resolved)
            resolved[axis] = v
        end
        return resolved
    end
    
    is_left_side = side !== nothing && lowercase(side) == "left"
    
    for (key, val) in margins_mm
        if string(key) == "slice_wise" || !(val isa Number)
            continue
        end
        v = Float32(val)
        if v <= 0f0
            continue
        end
        k = lowercase(string(key))
        if haskey(resolved, k)
            resolved[k] = max(resolved[k], v)
        elseif k == "lateral"
            target = is_left_side ? "left" : "right"
            resolved[target] = max(resolved[target], v)
        elseif k == "medial"
            target = is_left_side ? "right" : "left"
            resolved[target] = max(resolved[target], v)
        elseif k == "all"
            for axis in keys(resolved)
                resolved[axis] = max(resolved[axis], v)
            end
        elseif occursin("anterolateral", k)
            resolved["anterior"] = max(resolved["anterior"], v)
            lat_target = is_left_side ? "left" : "right"
            resolved[lat_target] = max(resolved[lat_target], v)
        elseif occursin("anteromedial", k)
            resolved["anterior"] = max(resolved["anterior"], v)
            med_target = is_left_side ? "right" : "left"
            resolved[med_target] = max(resolved[med_target], v)
        elseif occursin("posterolateral", k)
            resolved["posterior"] = max(resolved["posterior"], v)
            lat_target = is_left_side ? "left" : "right"
            resolved[lat_target] = max(resolved[lat_target], v)
        elseif occursin("posteromedial", k)
            resolved["posterior"] = max(resolved["posterior"], v)
            med_target = is_left_side ? "right" : "left"
            resolved[med_target] = max(resolved[med_target], v)
        end
    end
    return resolved
end


# ── GPU Surface Scatter Dilation Kernel (Fastest for massive expansions) ─────
@kernel function extract_surface_kernel!(surf, @Const(in_mask), dims_x::Int32, dims_y::Int32, dims_z::Int32)
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        if in_mask[i, j, k] > 0
            is_surf = false
            for dz in Int32(-1):Int32(1), dy in Int32(-1):Int32(1), dx in Int32(-1):Int32(1)
                if dx == 0 && dy == 0 && dz == 0 continue end
                cx = i + dx; cy = j + dy; cz = k + dz
                if cx >= Int32(1) && cx <= dims_x && cy >= Int32(1) && cy <= dims_y && cz >= Int32(1) && cz <= dims_z
                    if in_mask[cx, cy, cz] == UInt8(0)
                        is_surf = true
                        break
                    end
                else
                    is_surf = true
                    break
                end
            end
            if is_surf
                surf[i, j, k] = UInt8(1)
            end
        end
    end
end

@kernel function gpu_scatter_dilate_from_mask_kernel!(
    output, @Const(surf_mask),
    sp_x::Float32, sp_y::Float32, sp_z::Float32,
    mx_pos::Float32, mx_neg::Float32,
    my_pos::Float32, my_neg::Float32,
    mz_pos::Float32, mz_neg::Float32,
    rad_x_pos::Int32, rad_x_neg::Int32,
    rad_y_pos::Int32, rad_y_neg::Int32,
    rad_z_pos::Int32, rad_z_neg::Int32,
    dims_x::Int32, dims_y::Int32, dims_z::Int32
)
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        if surf_mask[i, j, k] > 0
            # Original voxel always stays true
            output[i, j, k] = UInt8(1)
            
            for dz_vox in -rad_z_neg:rad_z_pos
                cz = k + dz_vox
                if cz >= Int32(1) && cz <= dims_z
                    dz = Float32(dz_vox) * sp_z
                    mz = dz >= 0f0 ? mz_pos : mz_neg
                    
                    for dy_vox in -rad_y_neg:rad_y_pos
                        cy = j + dy_vox
                        if cy >= Int32(1) && cy <= dims_y
                            dy = Float32(dy_vox) * sp_y
                            my = dy >= 0f0 ? my_pos : my_neg
                            
                            for dx_vox in -rad_x_neg:rad_x_pos
                                cx = i + dx_vox
                                if cx >= Int32(1) && cx <= dims_x
                                    dx = Float32(dx_vox) * sp_x
                                    mx = dx >= 0f0 ? mx_pos : mx_neg
                                    
                                    abs_dx = abs(dx)
                                    abs_dy = abs(dy)
                                    abs_dz = abs(dz)
                                    tot = abs_dx + abs_dy + abs_dz + 1f-8
                                    
                                    thresh = (abs_dx / tot) * mx + (abs_dy / tot) * my + (abs_dz / tot) * mz
                                    dist_sq = dx*dx + dy*dy + dz*dz
                                    
                                    if dist_sq <= thresh * thresh && thresh > 0f0
                                        output[cx, cy, cz] = UInt8(1)
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


"""
    execute_anisotropic_expansion(backend, input_mask, dims, spacing, margins)
GPU-accelerated directional margin expansion using ROI-bounded gather kernel.
Computes directional expansion on GPU in milliseconds without CPU round-tripping.
"""
function execute_anisotropic_expansion(
    backend::KernelAbstractions.Backend,
    input_mask::AbstractArray,
    dims::Tuple{Int, Int, Int},
    spacing::Tuple{Float64, Float64, Float64},
    margins::Any;
    side::Union{String, Nothing}=nothing
)
    # Check if slice_wise mode
    is_slice_wise = margins isa AbstractDict && get(margins, "slice_wise", false)
    clean_margins = if is_slice_wise
        Dict{String, Any}(string(k) => v for (k, v) in margins if string(k) != "slice_wise")
    else
        margins
    end

    # ── Resolve directional margins ──────────────────────────────────────
    res_m = resolve_directional_margins(clean_margins, side)
    mx_pos = res_m["left"]
    mx_neg = res_m["right"]
    my_pos = res_m["posterior"]
    my_neg = res_m["anterior"]
    mz_pos = res_m["superior"]
    mz_neg = res_m["inferior"]
    
    sp_x = Float32(spacing[1]); sp_y = Float32(spacing[2]); sp_z = Float32(spacing[3])
    max_m = max(mx_pos, mx_neg, my_pos, my_neg, mz_pos, mz_neg)
    min_m = min(mx_pos, mx_neg, my_pos, my_neg, mz_pos, mz_neg)
    
    # ── Erosion or zero margin ───────────────────────────────────────────
    if max_m <= 0f0
        if min_m < 0f0 && mx_pos == min_m && mx_neg == min_m && my_pos == min_m && my_neg == min_m && mz_pos == min_m && mz_neg == min_m
            # Isotropic erosion: invert → expand → invert
            inv_m = map(x -> x == 0 ? UInt8(1) : UInt8(0), input_mask)
            pos_margins = Dict("all" => Float64(-min_m))
            exp_inv = execute_anisotropic_expansion(backend, inv_m, dims, spacing, pos_margins; side=side)
            return map(x -> x == 0 ? UInt8(1) : UInt8(0), exp_inv)
        else
            return (input_mask isa CuArray) ? copy(input_mask) : (backend isa CPU ? copy(input_mask) : adapt(backend, map(x -> x > 0 ? UInt8(1) : UInt8(0), input_mask)))
        end
    end
    
    # ── EDT Path (matches Python gold standard pipeline) ─────────────────
    # Uses Euclidean Distance Transform from nearest surface point.
    # More accurate than scatter: each voxel uses direction to its NEAREST 
    # surface point, not to ANY reachable surface point.
    if get(margins, "use_edt", false) || get(margins, "edt", false)
        t_edt_start = time()
        mask_cpu = input_mask isa Array ? (input_mask .> UInt8(0)) : Array(input_mask .> UInt8(0))
        
        # Crop to bounding box + padding for efficiency (matches Python approach)
        coords = findall(mask_cpu)
        if isempty(coords)
            output = KernelAbstractions.zeros(backend, UInt8, dims)
            println("    [EDT-AM] Empty input mask, returning zeros")
            return output
        end
        
        ci = [c[1] for c in coords]; cj = [c[2] for c in coords]; ck = [c[3] for c in coords]
        rad_x = ceil(Int, max_m / sp_x) + 2
        rad_y = ceil(Int, max_m / sp_y) + 2
        rad_z = ceil(Int, max_m / sp_z) + 2
        
        i_start = max(1, minimum(ci) - rad_x)
        i_end   = min(dims[1], maximum(ci) + rad_x)
        j_start = max(1, minimum(cj) - rad_y)
        j_end   = min(dims[2], maximum(cj) + rad_y)
        k_start = max(1, minimum(ck) - rad_z)
        k_end   = min(dims[3], maximum(ck) + rad_z)
        
        crop_mask = mask_cpu[i_start:i_end, j_start:j_end, k_start:k_end]
        crop_dims = size(crop_mask)
        
        # EDT on cropped region
        ft = ImageMorphology.feature_transform(crop_mask)
        edt = ImageMorphology.distance_transform(ft, (Float64(sp_x), Float64(sp_y), Float64(sp_z)))
        
        t_edt_mid = time()
        
        # Check for inferior taper (reduces lateral/medial margins for voxels below base)
        taper_inf = Float32(get(margins, "taper_inferior", 0.0))
        
        # Compute mask Z range for taper (use min Z as reference, taper as Z increases)
        z_min_crop = Float32(1)
        z_range_crop = Float32(crop_dims[3])
        if taper_inf > 0f0
            z_min_val = crop_dims[3]
            z_max_val = 1
            for k_idx in 1:crop_dims[3], j_idx in 1:crop_dims[2], i_idx in 1:crop_dims[1]
                if crop_mask[i_idx, j_idx, k_idx]
                    z_min_val = min(z_min_val, k_idx)
                    z_max_val = max(z_max_val, k_idx)
                end
            end
            z_min_crop = Float32(z_min_val)
            z_range_crop = mz_neg  # taper over just the inferior margin distance
            println("    [EDT-taper] Z min=$(z_min_val), Z max=$(z_max_val), Z range=$(round(z_range_crop,digits=1))mm, taper=$(taper_inf)")
        end
        
        # Directional threshold on cropped region
        result_crop = copy(crop_mask)
        Threads.@threads for k_idx in 1:crop_dims[3]
            for j_idx in 1:crop_dims[2], i_idx in 1:crop_dims[1]
                if crop_mask[i_idx, j_idx, k_idx]; continue; end
                d = edt[i_idx, j_idx, k_idx]
                if d > max_m; continue; end
                
                nearest = ft[i_idx, j_idx, k_idx]
                ni, nj, nk = Tuple(nearest)
                
                dx = Float32(i_idx - ni) * sp_x
                dy = Float32(j_idx - nj) * sp_y
                dz = Float32(k_idx - nk) * sp_z
                
                abs_dx = abs(dx)
                abs_dy = abs(dy)
                abs_dz = abs(dz)
                tot = abs_dx + abs_dy + abs_dz + 1f-8
                
                local_mx = dx >= 0f0 ? mx_pos : mx_neg
                local_my = dy >= 0f0 ? my_pos : my_neg
                local_mz = dz >= 0f0 ? mz_pos : mz_neg
                
                # Inferior taper: reduce MEDIAL margin based on absolute Z position
                # Gold's medial edge contracts inferiorly while lateral stays constant
                if taper_inf > 0f0 && Float32(k_idx) > z_min_crop
                    z_below_top = (Float32(k_idx) - z_min_crop) * sp_z  # mm below top of mask
                    inf_frac = min(1f0, z_below_top / (z_range_crop + 1f-8))  # 0..1 fraction
                    taper_scale = max(1f0 - taper_inf * inf_frac, 0.15f0)
                    # Only taper the smaller margin (medial side)
                    if mx_pos <= mx_neg
                        # mx_pos is the smaller (medial) margin
                        if dx >= 0f0
                            local_mx *= taper_scale
                        end
                    else
                        # mx_neg is the smaller (medial) margin
                        if dx < 0f0
                            local_mx *= taper_scale
                        end
                    end
                    # Also taper anterior margin
                    if dy < 0f0  # anterior direction (negative Y in LPS)
                        local_my *= taper_scale
                    end
                end
                
                thresh = (abs_dx / tot) * local_mx + (abs_dy / tot) * local_my + (abs_dz / tot) * local_mz
                
                if d <= thresh && thresh > 0f0
                    result_crop[i_idx, j_idx, k_idx] = true
                end
            end
        end
        
        # Paste back to full volume
        full_result = zeros(Bool, dims)
        full_result[i_start:i_end, j_start:j_end, k_start:k_end] .= result_crop
        
        output = adapt(backend, UInt8.(full_result))
        t_edt_end = time()
        println("    [EDT-AM] EDT-based expansion: $(sum(full_result)) voxels, crop=$(crop_dims), edt=$(round(t_edt_mid-t_edt_start,digits=2))s, thresh=$(round(t_edt_end-t_edt_mid,digits=2))s (margins: left=$(mx_pos) right=$(mx_neg) ant=$(my_neg) post=$(my_pos) sup=$(mz_pos) inf=$(mz_neg))")
        return output
    end

    # ── Fast GPU Path (Surface Scatter Dilation) ───────────────────────
    if backend isa CUDABackend || input_mask isa CuArray
        gpu_mask = input_mask isa CuArray{UInt8, 3} ? input_mask : adapt(backend, UInt8.(input_mask .> 0))
        bbox = gpu_bounding_box(backend, gpu_mask)
        if bbox === nothing
            output = KernelAbstractions.zeros(backend, UInt8, dims)
            GC.gc(false)
        return adapt(typeof(input_mask), output)
        end
        
        rad_x_pos = ceil(Int32, mx_pos / max(0.001f0, sp_x))
        rad_x_neg = ceil(Int32, mx_neg / max(0.001f0, sp_x))
        rad_y_pos = ceil(Int32, my_pos / max(0.001f0, sp_y))
        rad_y_neg = ceil(Int32, my_neg / max(0.001f0, sp_y))
        rad_z_pos = is_slice_wise ? Int32(0) : ceil(Int32, mz_pos / max(0.001f0, sp_z))
        rad_z_neg = is_slice_wise ? Int32(0) : ceil(Int32, mz_neg / max(0.001f0, sp_z))
        
        # 1. Extract surface points
        surf_mask, surf_key = StaticArena.acquire_mask(backend)
        extract_surface_kernel!(backend)(surf_mask, gpu_mask, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=dims)
        KernelAbstractions.synchronize(backend)
        
        # 2. Scatter from surface (no findall needed)
        output = KernelAbstractions.zeros(backend, UInt8, dims)
        scatter_kernel! = gpu_scatter_dilate_from_mask_kernel!(backend)
        scatter_kernel!(
            output, surf_mask,
            sp_x, sp_y, sp_z,
            mx_pos, mx_neg, my_pos, my_neg, mz_pos, mz_neg,
            rad_x_pos, rad_x_neg, rad_y_pos, rad_y_neg, rad_z_pos, rad_z_neg,
            Int32(dims[1]), Int32(dims[2]), Int32(dims[3]),
            ndrange=dims
        )
        KernelAbstractions.synchronize(backend)
        StaticArena.release_mask(surf_key)

        KernelAbstractions.synchronize(backend)
        
        # Mask inside the original volume is also true
        output .= output .| gpu_mask
        
        return adapt(typeof(input_mask), output)
    end
    
    # ── CPU Path with Bounding-Box Crop (fastest for typical anatomy masks) ──
    mask_host = input_mask isa Array ? input_mask : adapt(Array, input_mask)
    idx = findall(mask_host .> 0)
    if isempty(idx)
        output = KernelAbstractions.zeros(backend, UInt8, dims)
        return output
    end
    
    rad_x = ceil(Int, max_m / max(0.001f0, sp_x)) + 2
    rad_y = ceil(Int, max_m / max(0.001f0, sp_y)) + 2
    rad_z = ceil(Int, max_m / max(0.001f0, sp_z)) + 2
    
    min_i = max(1, minimum(c[1] for c in idx) - rad_x)
    max_i = min(dims[1], maximum(c[1] for c in idx) + rad_x)
    min_j = max(1, minimum(c[2] for c in idx) - rad_y)
    max_j = min(dims[2], maximum(c[2] for c in idx) + rad_y)
    min_k = max(1, minimum(c[3] for c in idx) - rad_z)
    max_k = min(dims[3], maximum(c[3] for c in idx) + rad_z)
    
    crop = mask_host[min_i:max_i, min_j:max_j, min_k:max_k] .> 0
    crop_dims = size(crop)
    
    ft = exact_anisotropic_ft(crop, (Float64(sp_x), Float64(sp_y), Float64(sp_z)))
    
    host_out = zeros(UInt8, dims)
    Threads.@threads for k in 1:crop_dims[3]
        for j in 1:crop_dims[2], i in 1:crop_dims[1]
            nearest = ft[i, j, k]
            if nearest != CartesianIndex(0,0,0)
                sx, sy, sz = nearest.I
                dx = Float32(i - sx) * sp_x
                dy = Float32(j - sy) * sp_y
                dz = Float32(k - sz) * sp_z
                # Seed voxels: nearest == self → dist = 0 → always include
                if dx == 0f0 && dy == 0f0 && dz == 0f0
                    host_out[i + min_i - 1, j + min_j - 1, k + min_k - 1] = UInt8(1)
                else
                    # Directional threshold formula — matches old IPC engine kernel
                    abs_dx = abs(dx)
                    abs_dy = abs(dy)
                    abs_dz = abs(dz)
                    dist = sqrt(dx*dx + dy*dy + dz*dz)
                    total = abs_dx + abs_dy + abs_dz + 1f-8
                    mx = dx > 0f0 ? mx_pos : mx_neg
                    my = dy > 0f0 ? my_pos : my_neg
                    mz = dz > 0f0 ? mz_pos : mz_neg
                    threshold = (abs_dx / total) * mx + (abs_dy / total) * my + (abs_dz / total) * mz
                    if dist <= threshold && threshold > 0f0
                        host_out[i + min_i - 1, j + min_j - 1, k + min_k - 1] = UInt8(1)
                    end
                end
            end
        end
    end
    
    GC.gc(false)
    return adapt(typeof(input_mask), host_out)
end

# ============================================================
# Landmark Resolution & Restriction Bounds
# ============================================================

function resolve_landmark_z(
    lm_spec::Any,
    part::String,
    off::Real,
    get_mask_fn::Function,
    computed_landmarks::Dict,
    dims::Tuple{Int, Int, Int},
    spacing::Tuple{Float64, Float64, Float64},
    origin::Tuple{Float64, Float64, Float64},
    direction::NTuple{9, Float64};
    side::String="",
    selection_mode::String="first"
)
    lm_items = if lm_spec isa Vector
        [x isa Dict ? x : Dict("landmark" => string(x)) for x in lm_spec]
    elseif lm_spec isa Dict
        [lm_spec]
    elseif lm_spec isa String && !isempty(lm_spec)
        [Dict("landmark" => lm_spec)]
    else
        Dict{String, Any}[]
    end
    
    found_z = Float32[]
    
    for item in lm_items
        lm_name = get(item, "landmark", "")
        isempty(lm_name) && continue
        
        item_off = Float32(get(item, "offset_mm", off))
        eff_part = get(item, "part", get(item, "boundary_part", part))
        base_lm = lm_name
        if get_mask_fn(lm_name) === nothing
            for sfx in ["_bottom", "_min", "_inferior"]
                if endswith(lowercase(lm_name), sfx)
                    base_lm = lm_name[1:end-length(sfx)]
                    eff_part = "min"
                    break
                end
            end
            for sfx in ["_top", "_max", "_superior"]
                if endswith(lowercase(lm_name), sfx)
                    base_lm = lm_name[1:end-length(sfx)]
                    eff_part = "max"
                    break
                end
            end
        end
        
        # 0. Check computed landmarks directly first (skip if inguinal to avoid matching ligamentum arteriosum)
        if !occursin("inguinal", lowercase(base_lm)) && !occursin("inguinal", lowercase(lm_name))
            lm_cands = [lm_name, base_lm, "$(base_lm)_computed"]
            if lowercase(base_lm) in ["aortic_split_location", "aortic_bifurcation", "aortic_split"]
                push!(lm_cands, "aortic_bifurcation", "aortic_split_location", "aortic_split")
            end
            if !isempty(side)
                s_low = lowercase(side)
                s_up = uppercase(first(side))
                push!(lm_cands, "$(lm_name)_$s_low", "$(base_lm)_$s_low", "$(base_lm)$(s_up)_inf_z", "$(base_lm)$(s_up)_sup_z")
            end
            if occursin("station_2", lowercase(base_lm))
                if !isempty(side)
                    s_low = lowercase(side)
                    s_up = uppercase(first(side))
                    push!(lm_cands, "station_2_inf_z_$s_low", "station_2_sup_z_$s_low", "station_2$(s_up)_inf_z", "station_2$(s_up)_sup_z")
                end
                push!(lm_cands, "station_2_inf_z_left", "station_2_inf_z_right", "station_2L_inf_z", "station_2R_inf_z", "station_2_sup_z_left", "station_2_sup_z_right", "station_2L_sup_z", "station_2R_sup_z")
            end
            
            found_in_computed = false
            for cand in lm_cands
                if haskey(computed_landmarks, cand)
                    val = computed_landmarks[cand]
                    z_val = if val isa Number
                        Float32(val)
                    elseif val isa Vector && length(val) >= 3
                        Float32(val[3])
                    else
                        nothing
                    end
                    if z_val !== nothing
                        # Skip dummy/placeholder values like -1.0 for aortic bifurcation
                        if cand in ["aortic_bifurcation", "aortic_split_location", "aortic_split"] && (z_val == -1.0f0 || z_val == 0.0f0)
                            continue
                        end
                        dir_22 = Float32(direction[9])
                        z_min_vol = Float32(origin[3])
                        z_max_vol = Float32(origin[3]) + Float32(dims[3] - 1) * Float32(spacing[3]) * dir_22
                        vol_lo = min(z_min_vol, z_max_vol) - 50.0f0
                        vol_hi = max(z_min_vol, z_max_vol) + 50.0f0
                        if z_val < vol_lo || z_val > vol_hi
                            vox_key = "$(cand)_vox"
                            if haskey(computed_landmarks, vox_key)
                                vox_idx = computed_landmarks[vox_key]
                                z_val = Float32(origin[3]) + (Float32(vox_idx) - 1.0f0) * Float32(spacing[3]) * dir_22
                            end
                        end
                        push!(found_z, z_val + item_off)
                        found_in_computed = true
                        if selection_mode == "first"
                            return z_val + item_off
                        end
                        break
                    end
                end
            end
            if found_in_computed continue end
        end

        # Anatomical Aliases
        if lowercase(base_lm) in ["aortic_split_location", "aortic_bifurcation", "aortic_split"]
            base_lm = "aorta"
            eff_part = "min"
        elseif lowercase(base_lm) in ["carina", "carina_computed"]
            base_lm = "trachea"
            eff_part = "min"
        elseif lowercase(base_lm) in ["hyoid_bone", "hyoid"]
            base_lm = "hyoid"
        elseif lowercase(base_lm) in ["first_rib_top", "first_rib"]
            base_lm = "rib_1"
        elseif lowercase(base_lm) in ["skull_base", "skull"]
            base_lm = "skull"
        elseif lowercase(base_lm) in ["celiac_trunk", "celiac_artery", "celiac"]
            base_lm = "celiac_trunk"
        end
        
        # 1. Check 3D masks first
        mask_candidates = String[]
        if occursin("inguinal", lowercase(base_lm)) || occursin("inguinal", lowercase(lm_name))
            if !isempty(side)
                s_low = lowercase(side)
                push!(mask_candidates, "femur_$s_low", "femur", "inguinal_plane_superior_$s_low", "inguinal_plane_$s_low", "acetabulum_proxy")
            end
            push!(mask_candidates, "femur", "femur_left", "femur_right", "inguinal_plane_superior_left", "inguinal_plane_superior_right", "inguinal_plane_superior", "acetabulum_proxy")
        end
        if base_lm == "rib_1" || lm_name == "rib_1"
            empty!(mask_candidates)
            push!(mask_candidates, "rib_1", "rib_left_1", "rib_right_1")
        else
            is_midline = occursin(r"(?i)(vertebra|skull|spine|trachea|esophagus|sternum|hyoid|larynx|aorta|cord)", base_lm)
            if is_midline
                push!(mask_candidates, base_lm, lm_name)
            elseif !isempty(side)
                push!(mask_candidates, "$(base_lm)_$(lowercase(side))", "$(base_lm)_$(uppercase(side))", "$(lm_name)_$(lowercase(side))")
            end
            push!(mask_candidates, base_lm, lm_name)
            if base_lm == "hyoid" || lm_name == "hyoid_bone"
                push!(mask_candidates, "hyoid", "hyoid_bottom_plane")
            end
            if base_lm == "cricoid_cartilage_proxy" || base_lm == "cricoid"
                push!(mask_candidates, "cricoid_cartilage_proxy", "cricoid", "upper_cricoid_plane", "low_cricoid_plane")
            end
            if occursin("aortic_arch", base_lm)
                push!(mask_candidates, "aortic_arch_computed")
            end
            if occursin("carina", base_lm)
                push!(mask_candidates, "carina_computed", "trachea")
            end
            if occursin("vertebrae_l5", lowercase(base_lm))
                push!(mask_candidates, "vertebrae_L5", "vertebrae_l5")
            end
            push!(mask_candidates, "$(base_lm)_left", "$(base_lm)_right", "$(base_lm)_plane_left", "$(base_lm)_plane_right", "$(base_lm)_plane")
        end
        
        matched_mask = false
        for mc in mask_candidates
            m = get_mask_fn(mc)
            if m === nothing && haskey(computed_landmarks, mc)
                m = computed_landmarks[mc]
            end
            if m !== nothing
                if m isa Number
                    z_base = Float32(m)
                    matched_mask = true
                    break
                elseif (m isa Tuple || m isa Vector) && length(m) >= 3 && m[3] isa Number
                    z_base = Float32(m[3])
                    matched_mask = true
                    break
                elseif (m isa Tuple || m isa Vector) && length(m) >= 1 && m[end] isa Number
                    z_base = Float32(m[end])
                    matched_mask = true
                    break
                elseif m isa AbstractArray
                    m_clean = if lowercase(mc) in ["celiac_trunk", "adrenal_gland_left", "adrenal_gland_right", "adrenal_gland", "cricoid", "hyoid", "pancreas"]
                        output_buf = get(StaticArena.CCL_ARENA, :output, nothing)
                        get_largest_connected_component(m; dest=output_buf)
                    else
                        m
                    end
                    bbox = gpu_bounding_box(CUDA.functional() ? CUDABackend() : CPU(), m_clean)
                    if bbox !== nothing
                        dir_22 = Float32(direction[9])
                        k_min = bbox[5] - 1
                        k_max = bbox[6] - 1
                        z_min_phys = Float32(origin[3]) + Float32(k_min) * Float32(spacing[3]) * dir_22
                        z_max_phys = Float32(origin[3]) + Float32(k_max) * Float32(spacing[3]) * dir_22
                    
                    phys_lower = min(z_min_phys, z_max_phys)
                    phys_upper = max(z_min_phys, z_max_phys)
                    
                    # Fix 21: Use robust percentile Z boundaries to match Python's resolve_point.
                    # Python uses np.percentile(zs, 99.0) for max_z and np.percentile(zs, 1.0) for min_z
                    # instead of absolute bounding box extremes. This avoids edge noise and fixes cases
                    # like Station 2 where aorta's absolute max Z is above rib_1 center by 1 voxel.
                    z_base = if eff_part in ["absolute_min", "abs_min"]
                        phys_lower
                    elseif eff_part in ["min", "inferior", "min_z"]
                        # Robust min: 1st percentile of Z indices (matching Python, GPU reduced)
                        slice_counts = (m_clean isa CuArray || m_clean isa SubArray) ?
                            Array(dropdims(sum(m_clean .> UInt8(0), dims=(1, 2)), dims=(1, 2))) :
                            [count(m_clean[:, :, kk] .> 0) for kk in 1:size(m_clean, 3)]
                        total_vox = sum(slice_counts)
                        if total_vox > 0
                            threshold = max(1, Int(ceil(total_vox * 0.01)))
                            cum = 0; robust_k = 1
                            for kk in 1:length(slice_counts)
                                cum += slice_counts[kk]
                                if cum >= threshold
                                    robust_k = kk; break
                                end
                            end
                            Float32(origin[3]) + Float32(robust_k - 1) * Float32(spacing[3]) * dir_22
                        else
                            phys_lower
                        end
                    elseif eff_part in ["absolute_max", "abs_max"]
                        phys_upper
                    elseif eff_part in ["max", "superior", "max_z"]
                        # Robust max: 99th percentile of Z indices (matching Python, GPU reduced)
                        slice_counts = (m_clean isa CuArray || m_clean isa SubArray) ?
                            Array(dropdims(sum(m_clean .> UInt8(0), dims=(1, 2)), dims=(1, 2))) :
                            [count(m_clean[:, :, kk] .> 0) for kk in 1:size(m_clean, 3)]
                        total_vox = sum(slice_counts)
                        if total_vox > 0
                            threshold = max(1, Int(ceil(total_vox * 0.99)))
                            cum = 0; robust_k = length(slice_counts)
                            for kk in 1:length(slice_counts)
                                cum += slice_counts[kk]
                                if cum >= threshold
                                    robust_k = kk; break
                                end
                            end
                            Float32(origin[3]) + Float32(robust_k - 1) * Float32(spacing[3]) * dir_22
                        else
                            phys_upper
                        end
                    elseif eff_part in ["mid", "mean", "center"]
                        # Fix 21b: Use bounding box center for "center" to match Python's resolve_point.
                        # Python uses (zs.min()+zs.max())/2.0 for center_z, NOT center of mass.
                        # This matters for asymmetric masks like rib_right_1 where centroid ≠ bbox_center.
                        (phys_lower + phys_upper) / 2.0f0
                    else
                        phys_upper
                    end

                    push!(found_z, z_base + item_off)
                    matched_mask = true
                    if selection_mode == "first"
                        return z_base + item_off
                    end
                    break
                    end
                end
            end
        end
        if matched_mask continue end
        
        # 2. Check computed landmarks
        landmark_candidates = [lm_name, base_lm, "$(base_lm)_computed"]
        if !isempty(side)
            s_low = lowercase(side)
            s_up = uppercase(first(side))
            push!(landmark_candidates, "$(lm_name)_$s_low", "$(base_lm)_$s_low", "$(base_lm)$(s_up)_inf_z", "$(base_lm)$(s_up)_sup_z")
        end
        if occursin("station_2", lowercase(base_lm))
            if !isempty(side)
                s_low = lowercase(side)
                s_up = uppercase(first(side))
                push!(landmark_candidates, "station_2_inf_z_$s_low", "station_2_sup_z_$s_low", "station_2$(s_up)_inf_z", "station_2$(s_up)_sup_z")
            end
            push!(landmark_candidates, "station_2_inf_z_left", "station_2_inf_z_right", "station_2L_inf_z", "station_2R_inf_z", "station_2_sup_z_left", "station_2_sup_z_right", "station_2L_sup_z", "station_2R_sup_z")
        end
        
        if occursin("acetabulum", lowercase(base_lm))
            push!(landmark_candidates, "acetabulum_top_left", "acetabulum_top_right", "acetabulum_top")
        end
        
        for cand in landmark_candidates
            if haskey(computed_landmarks, cand)
                val = computed_landmarks[cand]
                z_val = if val isa Number
                    Float32(val)
                elseif val isa Vector && length(val) >= 3
                    Float32(val[3])
                elseif val isa String && isfile(val)
                    try
                        nii = NIfTI.niread(val)
                        nii_data = nii.raw
                        idx = findall(nii_data .> 0)
                        if !isempty(idx)
                            k_min = minimum(ci[3] for ci in idx) - 1
                            k_max = maximum(ci[3] for ci in idx) - 1
                            dir_22 = Float32(direction[9])
                            z_min_p = Float32(origin[3]) + Float32(k_min) * Float32(spacing[3]) * dir_22
                            z_max_p = Float32(origin[3]) + Float32(k_max) * Float32(spacing[3]) * dir_22
                            z_lo = min(z_min_p, z_max_p); z_hi = max(z_min_p, z_max_p)
                            if eff_part in ["min", "inferior", "min_z", "absolute_min", "abs_min"]
                                z_lo
                            elseif eff_part in ["max", "superior", "max_z", "absolute_max", "abs_max"]
                                z_hi
                            else
                                (z_lo + z_hi) / 2.0f0
                            end
                        else
                            nothing
                        end
                    catch
                        nothing
                    end
                else
                    nothing
                end
                if z_val !== nothing
                    dir_22 = Float32(direction[9])
                    z_min_vol = Float32(origin[3])
                    z_max_vol = Float32(origin[3]) + Float32(dims[3] - 1) * Float32(spacing[3]) * dir_22
                    vol_lo = min(z_min_vol, z_max_vol) - 50.0f0
                    vol_hi = max(z_min_vol, z_max_vol) + 50.0f0
                    if z_val < vol_lo || z_val > vol_hi
                        vox_key = "$(cand)_vox"
                        if haskey(computed_landmarks, vox_key)
                            vox_idx = computed_landmarks[vox_key]
                            z_val = Float32(origin[3]) + (Float32(vox_idx) - 1.0f0) * Float32(spacing[3]) * dir_22
                        end
                    end
                    push!(found_z, z_val + item_off)
                    if selection_mode == "first"
                        return z_val + item_off
                    end
                    break
                end
            end
        end
    end
    
    if !isempty(found_z)
        if selection_mode == "min_physical"
            return minimum(found_z)
        elseif selection_mode == "max_physical"
            return maximum(found_z)
        else
            return found_z[1]
        end
    end
    return nothing
end

@kernel function gpu_apply_z_bounds_kernel!(
    mask,
    dims_x::Int32, dims_y::Int32, dims_z::Int32,
    k_min::Int32, k_max::Int32
)
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        if k < k_min || k > k_max
            mask[i, j, k] = UInt8(0)
        end
    end
end

@kernel function gpu_apply_box_bounds_kernel!(
    mask,
    dims_x::Int32, dims_y::Int32, dims_z::Int32,
    sp_x::Float32, sp_y::Float32, sp_z::Float32,
    orig_x::Float32, orig_y::Float32, orig_z::Float32,
    dir_00::Float32, dir_11::Float32, dir_22::Float32,
    min_x::Float32, max_x::Float32,
    min_y::Float32, max_y::Float32,
    min_z::Float32, max_z::Float32
)
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        if mask[i, j, k] > 0
            pos_x = orig_x + (Float32(i) - 1.0f0) * sp_x * dir_00
            pos_y = orig_y + (Float32(j) - 1.0f0) * sp_y * dir_11
            pos_z = orig_z + (Float32(k) - 1.0f0) * sp_z * dir_22
            
            if pos_x < min_x || pos_x > max_x ||
               pos_y < min_y || pos_y > max_y ||
               pos_z < min_z || pos_z > max_z
                mask[i, j, k] = 0
            end
        end
    end
end

function apply_z_plane_restriction(
    backend::KernelAbstractions.Backend,
    mask::AbstractArray,
    z_restr::Dict,
    get_mask_fn::Function,
    computed_landmarks::Dict,
    dims::Tuple{Int, Int, Int},
    spacing::Tuple{Float64, Float64, Float64},
    origin::Tuple{Float64, Float64, Float64},
    direction::NTuple{9, Float64};
    side::String=""
)
    mask_arr = (mask isa CuArray || mask isa SubArray) ? mask : (backend isa CPU ? UInt8.(mask .> 0) : adapt(backend, UInt8.(mask .> 0)))
    sup_spec = get(z_restr, "superior", nothing)
    sup_part = get(z_restr, "superior_part", "max")
    sup_off = get(z_restr, "superior_offset_mm", 0.0)
    sup_sel = get(z_restr, "superior_selection_mode", "first")
    
    inf_spec = get(z_restr, "inferior", nothing)
    inf_part = get(z_restr, "inferior_part", "min")
    inf_off = get(z_restr, "inferior_offset_mm", 0.0)
    inf_sel = get(z_restr, "inferior_selection_mode", "first")
    
    sup_z = resolve_landmark_z(sup_spec, sup_part, sup_off, get_mask_fn, computed_landmarks, dims, spacing, origin, direction; side=side, selection_mode=sup_sel)
    # Fix 14: For INFERIOR boundaries, "max_physical" semantics in Python are inverted —
    # Python picks the rib with MINIMUM physical Z (most inferior = least restrictive).
    # Empirically verified: Station_1_LowCervical z_min=224 (Python) = rib_right_1 centroid (min Z).
    # "max_physical" on inferior boundary should behave as "min_physical".
    eff_inf_sel = (inf_sel == "max_physical") ? "min_physical" : inf_sel
    inf_z = resolve_landmark_z(inf_spec, inf_part, inf_off, get_mask_fn, computed_landmarks, dims, spacing, origin, direction; side=side, selection_mode=eff_inf_sel)
    
    
    strict = get(z_restr, "strict", false)
    if strict
        if (sup_spec !== nothing && sup_z === nothing) || (inf_spec !== nothing && inf_z === nothing)
            return KernelAbstractions.zeros(backend, UInt8, dims)
        end
    end
    
    k_min = if inf_z !== nothing
        v = Int32(floor(Int, (Float64(inf_z) - origin[3]) / spacing[3]) + 1)
        v
    else
        Int32(1)
    end
    k_max = if sup_z !== nothing
        Int32(floor(Int, (Float64(sup_z) - origin[3]) / spacing[3]))
    else
        Int32(dims[3])
    end
    
    if k_min > 1 || k_max < dims[3]
        println("    [DEBUG z_restr] k_min=$k_min k_max=$k_max (sup_z=$sup_z inf_z=$inf_z) sup_spec=$sup_spec/$sup_part inf_spec=$inf_spec/$inf_part")
        if k_min > k_max
            println("    [WARNING z_restr] k_min > k_max — entire mask will be zeroed! Swapping may be needed.")
        end
        kernel! = gpu_apply_z_bounds_kernel!(backend)
        kernel!(
            mask_arr,
            Int32(dims[1]), Int32(dims[2]), Int32(dims[3]),
            k_min, k_max,
            ndrange=dims
        )
        KernelAbstractions.synchronize(backend)
    end
    return mask_arr
end

# ============================================================
# GPU Slice-wise Constraint Kernels
# ============================================================

@kernel function compute_slice_y_bounds_kernel!(
    @Const(mask),
    min_y_out,
    max_y_out,
    dims_x::Int32, dims_y::Int32, dims_z::Int32
)
    k = @index(Global, Linear)
    if k <= dims_z
        min_j = Int32(999999)
        max_j = Int32(-1)
        for j in Int32(1):dims_y
            has_fg = false
            for i in Int32(1):dims_x
                if mask[i, j, k] > 0
                    has_fg = true
                    break
                end
            end
            if has_fg
                if j < min_j
                    min_j = j
                end
                if j > max_j
                    max_j = j
                end
            end
        end
        if min_j < min_y_out[k]
            min_y_out[k] = min_j
        end
        if max_j > max_y_out[k]
            max_y_out[k] = max_j
        end
    end
end

@kernel function compute_slice_y_most_anterior_kernel!(
    @Const(mask),
    best_ant_y_out,
    best_limit_y_out,
    part_mode::Int32, # 1=center, 2=anterior, 3=posterior
    is_ras_y::Bool,
    dims_x::Int32, dims_y::Int32, dims_z::Int32
)
    k = @index(Global, Linear)
    if k <= dims_z
        min_j = Int32(999999)
        max_j = Int32(-1)
        for j in Int32(1):dims_y
            has_fg = false
            for i in Int32(1):dims_x
                if mask[i, j, k] > 0
                    has_fg = true
                    break
                end
            end
            if has_fg
                if j < min_j
                    min_j = j
                end
                if j > max_j
                    max_j = j
                end
            end
        end
        if min_j <= max_j
            # Landmark is present in slice k
            # In LPS (+Y Posterior): smaller j is anterior.
            # In RAS (+Y Anterior): larger j is anterior.
            ant_j = is_ras_y ? max_j : min_j
            is_better = is_ras_y ? (ant_j > best_ant_y_out[k]) : (ant_j < best_ant_y_out[k])
            if is_better
                best_ant_y_out[k] = ant_j
                if part_mode == Int32(1) # center
                    best_limit_y_out[k] = (min_j + max_j) ÷ Int32(2)
                elseif part_mode == Int32(2) # anterior
                    best_limit_y_out[k] = ant_j
                else # posterior
                    post_j = is_ras_y ? min_j : max_j
                    best_limit_y_out[k] = post_j
                end
            end
        end
    end
end

@kernel function compute_slice_x_bounds_kernel!(
    @Const(mask),
    min_x_out,
    max_x_out,
    dims_x::Int32, dims_y::Int32, dims_z::Int32,
    is_left::Bool, is_right::Bool, mid_x::Int32
)
    k = @index(Global, Linear)
    if k <= dims_z
        min_i = Int32(999999)
        max_i = Int32(-1)
        for i in Int32(1):dims_x
            if is_right && i > mid_x
                continue
            end
            if is_left && i <= mid_x
                continue
            end
            has_fg = false
            for j in Int32(1):dims_y
                if mask[i, j, k] > 0
                    has_fg = true
                    break
                end
            end
            if has_fg
                if i < min_i
                    min_i = i
                end
                if i > max_i
                    max_i = i
                end
            end
        end
        if min_i < min_x_out[k]
            min_x_out[k] = min_i
        end
        if max_i > max_x_out[k]
            max_x_out[k] = max_i
        end
    end
end

@kernel function apply_slicewise_y_constraint_kernel!(
    mask::AbstractArray{UInt8, 3},
    @Const(slice_limits),
    offset_vox::Int32,
    is_posterior::Bool,
    dims_x::Int32, dims_y::Int32, dims_z::Int32
)
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        lim = slice_limits[k]
        if lim != Int32(-1)
            if is_posterior
                threshold = lim + offset_vox
                if j <= threshold
                    mask[i, j, k] = UInt8(0)
                end
            else
                threshold = lim - offset_vox
                if j >= threshold
                    mask[i, j, k] = UInt8(0)
                end
            end
        end
    end
end

@kernel function apply_slicewise_x_constraint_kernel!(
    mask::AbstractArray{UInt8, 3},
    @Const(slice_limits),
    @Const(slice_center_x),
    offset_vox::Int32,
    mode::Int32, # 1=LeftOf, 2=RightOf, 3=LateralTo, 4=MedialTo
    midline_x::Int32,
    dims_x::Int32, dims_y::Int32, dims_z::Int32
)
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        lim = slice_limits[k]
        if lim != Int32(-1)
            if mode == Int32(1) # LeftOf in LPS (+X is Left): keep X >= lim + offset
                limit = lim + offset_vox
                if i < limit
                    mask[i, j, k] = UInt8(0)
                end
            elseif mode == Int32(2) # RightOf in LPS (-X is Right): keep X <= lim - offset
                limit = lim - offset_vox
                if i > limit
                    mask[i, j, k] = UInt8(0)
                end
            elseif mode == Int32(3) # LateralTo
                cx = slice_center_x[k]
                if cx < midline_x # Right side: Lateral is X <= limit
                    limit = lim - offset_vox
                    if i > limit
                        mask[i, j, k] = UInt8(0)
                    end
                else # Left side: Lateral is X >= limit
                    limit = lim + offset_vox
                    if i < limit
                        mask[i, j, k] = UInt8(0)
                    end
                end
            elseif mode == Int32(4) # MedialTo
                cx = slice_center_x[k]
                if cx < midline_x # Right side: Medial is X >= limit
                    limit = lim + offset_vox
                    if i < limit
                        mask[i, j, k] = UInt8(0)
                    end
                else # Left side: Medial is X <= limit
                    limit = lim - offset_vox
                    if i > limit
                        mask[i, j, k] = UInt8(0)
                    end
                end
            end
        end
    end
end

@kernel function apply_slicewise_between_x_kernel!(
    mask::AbstractArray{UInt8, 3},
    @Const(x_min_per_slice),
    @Const(x_max_per_slice),
    dims_x::Int32, dims_y::Int32, dims_z::Int32
)
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        xmin = x_min_per_slice[k]
        xmax = x_max_per_slice[k]
        if xmin != Int32(-1) && xmax != Int32(-1)
            if i < xmin || i > xmax
                mask[i, j, k] = UInt8(0)
            end
        end
    end
end

@kernel function apply_line_limit_kernel!(
    mask::AbstractArray{UInt8, 3},
    @Const(line_c_x),
    @Const(line_c_y),
    @Const(line_vx),
    @Const(line_vy),
    @Const(has_line),
    label_value::Int32,
    dims_x::Int32, dims_y::Int32, dims_z::Int32
)
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        if has_line[k]
            x_f = Float32(i - 1)
            y_f = Float32(j - 1)
            cx = line_c_x[k]
            cy = line_c_y[k]
            vx = line_vx[k]
            vy = line_vy[k]
            cross = (x_f - cx) * vy - (y_f - cy) * vx
            keep = (label_value == Int32(2)) ? (cross > 0.0f0) : (cross <= 0.0f0)
            if !keep
                mask[i, j, k] = UInt8(0)
            end
        end
    end
end

@kernel function apply_slicewise_lateral_constraint_kernel!(
    mask::AbstractArray{UInt8, 3},
    @Const(left_lat_limits),
    @Const(right_lat_limits),
    offset_vox::Int32,
    midline_x::Int32,
    dims_x::Int32, dims_y::Int32, dims_z::Int32
)
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        if i >= midline_x
            lim_l = left_lat_limits[k]
            if lim_l != Int32(-1) && i <= (lim_l + offset_vox)
                mask[i, j, k] = UInt8(0)
            end
        else
            lim_r = right_lat_limits[k]
            if lim_r != Int32(-1) && i >= (lim_r - offset_vox)
                mask[i, j, k] = UInt8(0)
            end
        end
    end
end

@kernel function apply_plane_limit_kernel!(
    mask::AbstractArray{UInt8, 3},
    dims_x::Int32, dims_y::Int32, dims_z::Int32,
    sp_x::Float32, sp_y::Float32, sp_z::Float32,
    orig_x::Float32, orig_y::Float32, orig_z::Float32,
    dir_00::Float32, dir_11::Float32, dir_22::Float32,
    pt_x::Float32, pt_y::Float32, pt_z::Float32,
    norm_x::Float32, norm_y::Float32, norm_z::Float32,
    keep_negative::Bool,
    restrict_left::Bool,
    restrict_right::Bool
)
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        if mask[i, j, k] > 0
            pos_x = orig_x + (Float32(i) - 1.0f0) * sp_x * dir_00
            pos_y = orig_y + (Float32(j) - 1.0f0) * sp_y * dir_11
            pos_z = orig_z + (Float32(k) - 1.0f0) * sp_z * dir_22
            
            dot_val = (pos_x - pt_x) * norm_x + (pos_y - pt_y) * norm_y + (pos_z - pt_z) * norm_z
            keep = keep_negative ? (dot_val <= 0.0f0) : (dot_val >= 0.0f0)
            if !keep
                mid_x_phys = orig_x + (Float32(dims_x) / 2.0f0) * sp_x * dir_00
                is_left_side = (pos_x >= mid_x_phys)
                should_clear = true
                if restrict_left
                    should_clear = is_left_side
                elseif restrict_right
                    should_clear = !is_left_side
                end
                if should_clear
                    mask[i, j, k] = UInt8(0)
                end
            end
        end
    end
end


@kernel function slice_bounds_kernel!(min_x_map, max_x_map, mask, dims_x::Int32, dims_y::Int32, dims_z::Int32)
    k = @index(Global, Linear)
    if k <= dims_z
        min_i = Int32(dims_x + 1)
        max_i = Int32(0)
        for j in Int32(1):dims_y
            for i in Int32(1):dims_x
                if mask[i, j, k] > 0
                    if min_i > dims_x
                        min_i = i
                    end
                    if i < min_i; min_i = i; end
                    if i > max_i; max_i = i; end
                end
            end
        end
        if max_i >= min_i
            min_x_map[k] = min_i
            max_x_map[k] = max_i
        else
            min_x_map[k] = Int32(0)
            max_x_map[k] = Int32(0)
        end
    end
end

function get_slice_bounds_gpu(backend, m, dims; return_gpu=false)
    dims_x, dims_y, dims_z = Int32(dims[1]), Int32(dims[2]), Int32(dims[3])
    min_x_gpu = KernelAbstractions.zeros(backend, Int32, dims_z)
    max_x_gpu = KernelAbstractions.zeros(backend, Int32, dims_z)
    
    k! = slice_bounds_kernel!(backend)
    k!(min_x_gpu, max_x_gpu, m, dims_x, dims_y, dims_z, ndrange=dims_z)
    KernelAbstractions.synchronize(backend)
    
    if return_gpu
        return min_x_gpu, max_x_gpu
    else
        return adapt(Array, min_x_gpu), adapt(Array, max_x_gpu)
    end
end

@kernel function slice_y_bounds_kernel!(min_y_map, max_y_map, mask, dims_x::Int32, dims_y::Int32, dims_z::Int32)
    k = @index(Global, Linear)
    if k <= dims_z
        min_j = Int32(dims_y + 1)
        max_j = Int32(0)
        for i in Int32(1):dims_x
            for j in Int32(1):dims_y
                if mask[i, j, k] > 0
                    if min_j > dims_y
                        min_j = j
                    end
                    if j < min_j; min_j = j; end
                    if j > max_j; max_j = j; end
                end
            end
        end
        if max_j >= min_j
            min_y_map[k] = min_j
            max_y_map[k] = max_j
        else
            min_y_map[k] = Int32(0)
            max_y_map[k] = Int32(0)
        end
    end
end

function get_slice_y_bounds_gpu(backend, m, dims; return_gpu::Bool=false)
    dims_x, dims_y, dims_z = Int32(dims[1]), Int32(dims[2]), Int32(dims[3])
    min_y_gpu = KernelAbstractions.zeros(backend, Int32, dims_z)
    max_y_gpu = KernelAbstractions.zeros(backend, Int32, dims_z)
    
    k! = slice_y_bounds_kernel!(backend)
    k!(min_y_gpu, max_y_gpu, m, dims_x, dims_y, dims_z, ndrange=dims_z)
    KernelAbstractions.synchronize(backend)
    
    if return_gpu
        return min_y_gpu, max_y_gpu
    else
        return adapt(Array, min_y_gpu), adapt(Array, max_y_gpu)
    end
end

@kernel function zero_anterior_to_y_kernel!(mask, min_y_arr, dims_x::Int32, dims_y::Int32, dims_z::Int32)
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        y_limit = min_y_arr[k]
        if y_limit > Int32(0) && j < y_limit
            mask[i, j, k] = UInt8(0)
        end
    end
end

function zero_anterior_to_y_gpu!(backend, mask, min_y_arr, dims)
    min_y_gpu = (min_y_arr isa AbstractArray && !(min_y_arr isa Array)) ? min_y_arr : adapt(backend, Int32.(min_y_arr))
    k! = zero_anterior_to_y_kernel!(backend)
    k!(mask, min_y_gpu, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=dims)
    KernelAbstractions.synchronize(backend)
end


function apply_constraints(
    backend::KernelAbstractions.Backend,
    mask::AbstractArray,
    constraints::Vector,
    get_mask_fn::Function,
    computed_landmarks::Dict,
    dims::Tuple{Int, Int, Int},
    spacing::Tuple{Float64, Float64, Float64},
    origin::Tuple{Float64, Float64, Float64},
    direction::NTuple{9, Float64};
    side::Union{String, Nothing}=nothing
)
    mask_arr = (mask isa CuArray || mask isa SubArray) ? mask : (backend isa CPU ? copy(mask) : adapt(backend, UInt8.(mask .> 0)))
    min_x = -100000.0f0; max_x = 100000.0f0
    min_y = -100000.0f0; max_y = 100000.0f0
    min_z = -100000.0f0; max_z = 100000.0f0
    
    dir_00 = Float32(direction[1])
    dir_11 = Float32(direction[5])
    dir_22 = Float32(direction[9])
    
    for c in constraints
        c isa Dict || continue
        c_type = get(c, "constraint_type", get(c, "type", ""))
        lm = get(c, "landmark", "")
        off = Float32(get(c, "offset_mm", 0.0))
        part = get(c, "boundary_part", "center")
        
        # SideLimit: Python converts this to LeftOf/RightOf based on side parameter
        if c_type == "SideLimit"
            effective_side_sl = get(c, "side_for_limit", get(c, "side", side !== nothing ? side : ""))
            if !isempty(effective_side_sl)
                c_type = lowercase(effective_side_sl) == "left" ? "LeftOf" : "RightOf"
                # Auto-select edge: LeftOf uses max (left edge in LPS), RightOf uses min (right edge in LPS)
                part = lowercase(effective_side_sl) == "left" ? "max" : "min"
                shift_right = Float32(get(c, "shift_right_mm", 0.0))
                if shift_right != 0.0f0
                    off = c_type == "LeftOf" ? off - shift_right : off + shift_right
                end
            else
                continue
            end
        end




        if c_type == "BetweenLandmarksXLimit"
            lm1 = get(c, "landmark_1", "")
            lm2 = get(c, "landmark_2", "")
            lm1_str = lm1 isa Dict ? get(lm1, "landmark", "") : string(lm1)
            lm2_str = lm2 isa Dict ? get(lm2, "landmark", "") : string(lm2)
            m1 = !isempty(lm1_str) ? get_mask_fn(lm1_str) : nothing
            m2 = !isempty(lm2_str) ? get_mask_fn(lm2_str) : nothing
            if (m1 === nothing || m2 === nothing) && occursin("submandibular", lowercase(lm1_str))
                gland_l = get_mask_fn("submandibular_gland_left")
                gland_r = get_mask_fn("submandibular_gland_right")
                if gland_l !== nothing && gland_r !== nothing
                    m1 = gland_l
                    m2 = gland_r
                end
            end

            if m1 !== nothing && m2 !== nothing
                l_min, l_max = get_slice_bounds_gpu(backend, m1, dims)
                r_min, r_max = get_slice_bounds_gpu(backend, m2, dims)
                
                x_min_arr = zeros(Int32, dims[3])
                x_max_arr = zeros(Int32, dims[3])
                
                valid_ks = Int[]
                for k in 1:dims[3]
                    if l_max[k] > 0 && r_max[k] > 0
                        push!(valid_ks, k)
                    end
                end
                
                if !isempty(valid_ks)
                    split_method = get(c, "split_method", "")
                    
                    for k in 1:dims[3]
                        use_k = k
                        if !(k in valid_ks)
                            use_k = valid_ks[argmin(abs.(valid_ks .- k))]
                        end
                        
                        xl_min, xl_max = l_min[use_k], l_max[use_k]
                        xr_min, xr_max = r_min[use_k], r_max[use_k]
                        
                        if split_method == "midline"
                            mid_x = div((xl_min + xl_max) + (xr_min + xr_max), 4)
                            side_to_keep = get(c, "side_to_keep", "left")
                            if lowercase(side_to_keep) == "left"
                                x_min_arr[k] = mid_x
                                x_max_arr[k] = dims[1]
                            else
                                x_min_arr[k] = 1
                                x_max_arr[k] = mid_x
                            end
                        else
                            xmin, xmax = if xl_min > xr_max
                                xr_max, xl_min
                            elseif xr_min > xl_max
                                xl_max, xr_min
                            else
                                min(xl_min, xr_min), max(xl_max, xr_max)
                            end
                            x_min_arr[k] = xmin
                            x_max_arr[k] = xmax
                        end
                    end
                    
                    xmin_gpu = adapt(backend, x_min_arr)
                    xmax_gpu = adapt(backend, x_max_arr)
                    k! = apply_slicewise_between_x_kernel!(backend)
                    k!(mask_arr, xmin_gpu, xmax_gpu, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=dims)
                    KernelAbstractions.synchronize(backend)
                end
            end


        elseif c_type == "LineLimitBetweenLandmarks"
            lm1 = get(c, "landmark_1", "aorta_ascending")
            lm2 = get(c, "landmark_2", "aorta_descending")
            lm1_str = lm1 isa Dict ? get(lm1, "landmark", "") : string(lm1)
            lm2_str = lm2 isa Dict ? get(lm2, "landmark", "") : string(lm2)
            m1 = !isempty(lm1_str) ? get_mask_fn(lm1_str) : nothing
            m2 = !isempty(lm2_str) ? get_mask_fn(lm2_str) : nothing
            if m1 !== nothing && m2 !== nothing
                sx1, sy1, c1 = get_centroid_per_slice_gpu(backend, m1, dims)
                sx2, sy2, c2 = get_centroid_per_slice_gpu(backend, m2, dims)
                
                label_val = Int32(get(c, "label_value", 2))
                
                line_c_x = zeros(Float32, dims[3])
                line_c_y = zeros(Float32, dims[3])
                line_vx = zeros(Float32, dims[3])
                line_vy = zeros(Float32, dims[3])
                has_line = falses(dims[3])
                
                last_line = nothing
                for k in 1:dims[3]
                    if c1[k] > 0 && c2[k] > 0
                        c_x1 = Float32(sx1[k] - c1[k]) / Float32(c1[k])
                        c_y1 = Float32(sy1[k] - c1[k]) / Float32(c1[k])
                        c_x2 = Float32(sx2[k] - c2[k]) / Float32(c2[k])
                        c_y2 = Float32(sy2[k] - c2[k]) / Float32(c2[k])
                        vx = c_x2 - c_x1
                        vy = c_y2 - c_y1
                        last_line = (c_x1, c_y1, vx, vy)
                    end
                    if last_line !== nothing
                        line_c_x[k] = last_line[1]
                        line_c_y[k] = last_line[2]
                        line_vx[k] = last_line[3]
                        line_vy[k] = last_line[4]
                        has_line[k] = true
                    end
                end
                
                if any(has_line)
                    k_ll! = apply_line_limit_kernel!(backend)
                    k_ll!(mask_arr, adapt(backend, line_c_x), adapt(backend, line_c_y), adapt(backend, line_vx), adapt(backend, line_vy), adapt(backend, has_line), label_val, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=dims)
                    KernelAbstractions.synchronize(backend)
                end
            end
        elseif c_type == "PlaneLimit"
            lm_key = get(c, "landmark", "")
            plane_def = nothing
            keys_to_try = [lm_key]
            if side !== nothing && !isempty(side)
                insert!(keys_to_try, 1, "$(lm_key)_$(lowercase(side))")
            end
            for k in keys_to_try
                if haskey(computed_landmarks, k)
                    val = computed_landmarks[k]
                    if val isa Tuple || val isa Vector
                        plane_def = val
                        break
                    elseif val isa Number
                        plane_def = [Float64(origin[1] + dims[1]*spacing[1]/2.0), Float64(origin[2] + dims[2]*spacing[2]/2.0), Float64(val)]
                        break
                    end
                end
            end
            
            ref_pt = nothing
            normal = get(c, "normal", nothing)
            if plane_def !== nothing && length(plane_def) == 2
                ref_pt = plane_def[1]
                normal = plane_def[2]
            elseif plane_def !== nothing && length(plane_def) >= 3
                ref_pt = [Float64(plane_def[1]), Float64(plane_def[2]), Float64(plane_def[3])]
                if normal === nothing
                    normal = [0.0, 0.0, 1.0]
                end
            end
            
            if ref_pt === nothing
                part_req = occursin("bottom", lowercase(lm_key)) || occursin("min", lowercase(lm_key)) || occursin("inf", lowercase(lm_key)) ? "min" : "max"
                z_res = resolve_landmark_z(lm_key, part_req, 0.0, get_mask_fn, computed_landmarks, dims, spacing, origin, direction; side=(side !== nothing ? side : ""))
                if z_res !== nothing
                    ref_pt = [Float64(origin[1] + dims[1]*spacing[1]/2.0), Float64(origin[2] + dims[2]*spacing[2]/2.0), Float64(z_res)]
                    if normal === nothing
                        normal = [0.0, 0.0, 1.0]
                    end
                end
            end
            
            if normal === nothing
                normal = [0.0, 1.0, 0.0]
            end
            
            if ref_pt !== nothing && length(ref_pt) >= 3 && length(normal) >= 3
                off_mm = Float64(get(c, "offset_mm", 0.0))
                norm_vec = [Float64(normal[1]), Float64(normal[2]), Float64(normal[3])]
                n_len = sqrt(sum(norm_vec .^ 2))
                if n_len > 0.0
                    norm_vec ./= n_len
                    pt_vec = [Float64(ref_pt[1]), Float64(ref_pt[2]), Float64(ref_pt[3])] + off_mm .* norm_vec
                    side_to_keep = get(c, "side_to_keep", "negative")
                    keep_neg = (side_to_keep == "negative")
                    
                    restrict_side = ""
                    if occursin("_left", lowercase(lm_key))
                        restrict_side = "left"
                    elseif occursin("_right", lowercase(lm_key))
                        restrict_side = "right"
                    elseif side !== nothing && side != ""
                        restrict_side = lowercase(side)
                    end
                    
                    dir_00 = Float32(direction[1]); dir_11 = Float32(direction[5]); dir_22 = Float32(direction[9])
                    k_pl! = apply_plane_limit_kernel!(backend)
                    k_pl!(
                        mask_arr,
                        Int32(dims[1]), Int32(dims[2]), Int32(dims[3]),
                        Float32(spacing[1]), Float32(spacing[2]), Float32(spacing[3]),
                        Float32(origin[1]), Float32(origin[2]), Float32(origin[3]),
                        dir_00, dir_11, dir_22,
                        Float32(pt_vec[1]), Float32(pt_vec[2]), Float32(pt_vec[3]),
                        Float32(norm_vec[1]), Float32(norm_vec[2]), Float32(norm_vec[3]),
                        keep_neg,
                        restrict_side == "left",
                        restrict_side == "right",
                        ndrange=dims
                    )
                    KernelAbstractions.synchronize(backend)
                end
            end
        elseif c_type in ["AnteriorTo", "PosteriorTo", "SuperiorTo", "InferiorTo", "MedialTo", "LateralTo", "LeftOf", "RightOf"]
            mode_str = lowercase(string(get(c, "mode", "")))
            is_slice_wise = get(c, "slice_wise", false) || (mode_str in ["per_layer", "per_slice", "slicewise", "slice_wise"])
            lm_raw_check = get(c, "landmark", get(c, "landmarks", get(c, "target", "")))
            lm_str_check = lowercase(string(lm_raw_check))
            is_spine_lm = occursin("spine", lm_str_check) || occursin("spinal", lm_str_check) || occursin("vertebrae", lm_str_check) || occursin("cord", lm_str_check)
            is_muscle_lm = occursin("sternocleidomastoid", lm_str_check) || occursin("trapezius", lm_str_check) || occursin("muscle", lm_str_check) || occursin("scapula", lm_str_check) || occursin("scalene", lm_str_check) || occursin("splenius", lm_str_check) || occursin("levator", lm_str_check) || occursin("pectoralis", lm_str_check)
            
            if (is_slice_wise || is_spine_lm || is_muscle_lm) && c_type in ["AnteriorTo", "PosteriorTo"]
                lm_raw = lm_raw_check
                parallel_to_lm = get(c, "parallel_to", "")
                
                if !isempty(parallel_to_lm)
                    # Parallel-to logic (e.g. parallel_to sternum)
                    p_mask = get_mask_fn(parallel_to_lm)
                    
                    # We need the union of all landmarks for the cut point
                    lm_names = if lm_raw isa Vector
                        [x isa Dict ? get(x, "landmark", "") : string(x) for x in lm_raw]
                    elseif lm_raw isa Dict
                        [get(lm_raw, "landmark", "")]
                    elseif lm_raw isa String && !isempty(lm_raw)
                        [lm_raw]
                    else
                        String[]
                    end
                    
                    if p_mask !== nothing
                        p_mask_gpu = p_mask isa CuArray ? p_mask : adapt(backend, UInt8.(p_mask .> 0))
                        
                        y_min_gpu = KernelAbstractions.allocate(backend, Int32, dims[1], dims[3])
                        y_max_gpu = KernelAbstractions.allocate(backend, Int32, dims[1], dims[3])
                        
                        k_prof! = extract_y_profile_kernel!(backend)
                        k_prof!(p_mask_gpu, y_min_gpu, y_max_gpu, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=(dims[1], dims[3]))
                        KernelAbstractions.synchronize(backend)
                        
                        p_y_min = adapt(Array, y_min_gpu)
                        p_y_max = adapt(Array, y_max_gpu)
                        
                        line_m = fill(Float32(-9999), dims[3])
                        line_b = fill(Float32(0), dims[3])
                        
                        y_increasing = direction[5] > 0
                        part = lowercase(string(get(c, "boundary_part", get(c, "part", "center"))))
                        use_min_fit = (part in ["min", "anterior"]) ? y_increasing : !y_increasing
                        
                        for k in 1:dims[3]
                            X_st = Float32[]
                            Y_st = Float32[]
                            for i in 1:dims[1]
                                val = use_min_fit ? p_y_min[i, k] : p_y_max[i, k]
                                if val != (use_min_fit ? 9999 : -9999)
                                    push!(X_st, Float32(i))
                                    push!(Y_st, Float32(val))
                                end
                            end
                            
                            if length(X_st) >= 2
                                mean_X = sum(X_st) / length(X_st)
                                mean_Y = sum(Y_st) / length(Y_st)
                                cov_XY = sum((X_st .- mean_X) .* (Y_st .- mean_Y))
                                var_X = sum((X_st .- mean_X) .^ 2)
                                if var_X > 0.0f0
                                    line_m[k] = cov_XY / var_X
                                end
                            end
                        end
                        
                        l_y_min_gpu = KernelAbstractions.allocate(backend, Int32, dims[1], dims[3])
                        l_y_max_gpu = KernelAbstractions.allocate(backend, Int32, dims[1], dims[3])
                        best_ant_b = fill(y_increasing ? Float32(1e9) : Float32(-1e9), dims[3])
                        has_any_b = fill(false, dims[3])
                        
                        for l in lm_names
                            m = nothing
                            if side !== nothing && !isempty(side)
                                m = get_mask_fn("$(l)_$side")
                                if m === nothing; m = get_mask_fn("$(l)_$(titlecase(side))"); end
                                if m === nothing; m = get_mask_fn("$(l)_$(lowercase(side))"); end
                            end
                            if m === nothing; m = get_mask_fn(l); end
                            if m !== nothing
                                m_gpu = m isa CuArray ? m : adapt(backend, UInt8.(m .> 0))
                                k_prof!(m_gpu, l_y_min_gpu, l_y_max_gpu, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=(dims[1], dims[3]))
                                KernelAbstractions.synchronize(backend)
                                l_y_min = adapt(Array, l_y_min_gpu)
                                l_y_max = adapt(Array, l_y_max_gpu)
                                
                                for k in 1:dims[3]
                                    m_fit = line_m[k]
                                    m_fit == Float32(-9999) && continue
                                    
                                    b_min_vals = Float32[]
                                    b_max_vals = Float32[]
                                    for i in 1:dims[1]
                                        ymin_v = l_y_min[i, k]
                                        if ymin_v != 9999
                                            push!(b_min_vals, Float32(ymin_v) - m_fit * Float32(i))
                                        end
                                        ymax_v = l_y_max[i, k]
                                        if ymax_v != -9999
                                            push!(b_max_vals, Float32(ymax_v) - m_fit * Float32(i))
                                        end
                                    end
                                    
                                    if !isempty(b_min_vals) && !isempty(b_max_vals)
                                        ant_b = minimum(b_min_vals)
                                        post_b = maximum(b_max_vals)
                                        is_better = y_increasing ? (ant_b < best_ant_b[k]) : (ant_b > best_ant_b[k])
                                        if is_better
                                            best_ant_b[k] = ant_b
                                            has_any_b[k] = true
                                            if part in ["mid", "center", "mean"]
                                                line_b[k] = (ant_b + post_b) / 2.0f0
                                            elseif part in ["min", "anterior", "min_y"]
                                                line_b[k] = ant_b
                                            else
                                                line_b[k] = post_b
                                            end
                                        end
                                    end
                                end
                            end
                        end
                        
                        line_m_gpu = adapt(backend, line_m)
                        line_b_gpu = adapt(backend, line_b)
                        offset_vox = Int32(round(Int, off / Float32(spacing[2])))
                        is_posterior = (c_type == "PosteriorTo")
                        
                        k_apply_line! = apply_slicewise_y_line_constraint_kernel!(backend)
                        k_apply_line!(mask_arr, line_m_gpu, line_b_gpu, offset_vox, is_posterior, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=dims)
                        KernelAbstractions.synchronize(backend)
                    end
                    continue
                end


                lm_names = if lm_raw isa Vector
                    [x isa Dict ? get(x, "landmark", "") : string(x) for x in lm_raw]
                elseif lm_raw isa Dict
                    [get(lm_raw, "landmark", "")]
                elseif lm_raw isa String && !isempty(lm_raw)
                    [lm_raw]
                else
                    String[]
                end
                
                is_ras_y = direction[5] < 0.0
                best_ant_y_gpu = KernelAbstractions.zeros(backend, Int32, dims[3])
                best_limit_y_gpu = KernelAbstractions.zeros(backend, Int32, dims[3])
                best_ant_y_gpu .= is_ras_y ? Int32(-1) : Int32(999999)
                best_limit_y_gpu .= Int32(-1)
                
                part_mode = if part in ["mid", "center", "mean"]
                    Int32(1)
                elseif part in ["min", "anterior", "min_y"]
                    Int32(2)
                else
                    Int32(3)
                end
                
                k_bounds! = compute_slice_y_most_anterior_kernel!(backend)
                found_any = false
                for lm_name in lm_names
                    isempty(lm_name) && continue
                    m_ref = nothing
                    if side !== nothing && !isempty(side)
                        m_ref = get_mask_fn("$(lm_name)_$side")
                        if m_ref === nothing; m_ref = get_mask_fn("$(lm_name)_$(titlecase(side))"); end
                        if m_ref === nothing; m_ref = get_mask_fn("$(lm_name)_$(lowercase(side))"); end
                    end
                    if m_ref === nothing; m_ref = get_mask_fn(lm_name); end
                    if m_ref !== nothing
                        m_ref_gpu = (m_ref isa CuArray) ? m_ref : adapt(backend, UInt8.(m_ref .> 0))
                        k_bounds!(m_ref_gpu, best_ant_y_gpu, best_limit_y_gpu, part_mode, is_ras_y, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=dims[3])
                        found_any = true
                    end
                    m_ref = nothing
                end
                if !found_any && haskey(c, "fallback_landmark")
                    fb_lm = string(c["fallback_landmark"])
                    m_ref = nothing
                    if side !== nothing && !isempty(side)
                        m_ref = get_mask_fn("$(fb_lm)_$side")
                        if m_ref === nothing; m_ref = get_mask_fn("$(fb_lm)_$(titlecase(side))"); end
                        if m_ref === nothing; m_ref = get_mask_fn("$(fb_lm)_$(lowercase(side))"); end
                    end
                    if m_ref === nothing; m_ref = get_mask_fn(fb_lm); end
                    if m_ref !== nothing
                        m_ref_gpu = (m_ref isa CuArray) ? m_ref : adapt(backend, UInt8.(m_ref .> 0))
                        k_bounds!(m_ref_gpu, best_ant_y_gpu, best_limit_y_gpu, part_mode, is_ras_y, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=dims[3])
                        found_any = true
                    end
                    m_ref = nothing
                end
                KernelAbstractions.synchronize(backend)
                
                if found_any
                    best_limit_cpu = adapt(Array, best_limit_y_gpu)
                    slice_y_limits = fill(Int32(-1), dims[3])
                    only_where_pres = Bool(get(c, "only_where_present", false))
                    valid_ks = findall(best_limit_cpu .!= -1)
                    
                    if !isempty(valid_ks)
                        for k in 1:dims[3]
                            src_k = -1
                            if best_limit_cpu[k] != -1
                                src_k = k
                            elseif !only_where_pres
                                src_k = valid_ks[argmin(abs.(valid_ks .- k))]
                            end
                            
                            if src_k != -1
                                slice_y_limits[k] = best_limit_cpu[src_k]
                            end
                        end
                    end
                    
                    slice_y_limits_gpu = adapt(backend, slice_y_limits)
                    offset_vox = Int32(round(Int, off / Float32(spacing[2])))
                    is_posterior = (c_type == "PosteriorTo")
                    is_ras_y = direction[5] < 0.0
                    is_posterior_eff = is_ras_y ? !is_posterior : is_posterior
                    
                    k_apply! = apply_slicewise_y_constraint_kernel!(backend)
                    k_apply!(mask_arr, slice_y_limits_gpu, offset_vox, is_posterior_eff, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=dims)
                    KernelAbstractions.synchronize(backend)
                end
                continue
            end
            
            if is_slice_wise && c_type == "LateralTo" && haskey(c, "landmarks")
                lm_list = c["landmarks"]
                m_l = nothing; m_r = nothing
                for lm_item in lm_list
                    lm_s = lm_item isa Dict ? get(lm_item, "landmark", "") : string(lm_item)
                    if occursin("left", lowercase(lm_s)); m_l = get_mask_fn(lm_s); end
                    if occursin("right", lowercase(lm_s)); m_r = get_mask_fn(lm_s); end
                end
                
                mid_x = div(dims[1], 2)
                l_limits = fill(Int32(-1), dims[3])
                r_limits = fill(Int32(-1), dims[3])
                
                if m_l !== nothing
                    max_l_gpu = KernelAbstractions.zeros(backend, Int32, dims[3]); max_l_gpu .= Int32(-1)
                    min_l_gpu = KernelAbstractions.zeros(backend, Int32, dims[3]); min_l_gpu .= Int32(999999)
                    k_b! = compute_slice_x_bounds_kernel!(backend)
                    k_b!(m_l, min_l_gpu, max_l_gpu, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), is_left_side, is_right_side, mid_x, ndrange=dims[3])
                    KernelAbstractions.synchronize(backend)
                    l_limits .= adapt(Array, max_l_gpu)
                end
                if m_r !== nothing
                    max_r_gpu = KernelAbstractions.zeros(backend, Int32, dims[3]); max_r_gpu .= Int32(-1)
                    min_r_gpu = KernelAbstractions.zeros(backend, Int32, dims[3]); min_r_gpu .= Int32(999999)
                    k_b! = compute_slice_x_bounds_kernel!(backend)
                    k_b!(m_r, min_r_gpu, max_r_gpu, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), is_left_side, is_right_side, mid_x, ndrange=dims[3])
                    KernelAbstractions.synchronize(backend)
                    r_limits .= adapt(Array, min_r_gpu)
                end
                
                l_gpu = adapt(backend, l_limits)
                r_gpu = adapt(backend, r_limits)
                k_lat! = apply_lateral_to_kernel!(backend)
                k_lat!(mask_arr, l_gpu, r_gpu, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=dims)
                KernelAbstractions.synchronize(backend)
                continue
            end
            
            if is_slice_wise && (c_type in ["LeftOf", "RightOf", "LateralTo", "MedialTo"])
                b_part = lowercase(string(get(c, "boundary_part", get(c, "part", part))))
                off_mm = off
                is_left_side = side !== nothing && lowercase(side) == "left"
                is_right_side = side !== nothing && lowercase(side) == "right"
                restrict_to_side = get(c, "restrict_to_side", false)
                filter_left = restrict_to_side && is_left_side
                filter_right = restrict_to_side && is_right_side
                
                lm_names = String[]
                if haskey(c, "landmark")
                    val = c["landmark"]
                    if val isa Vector
                        for x in val; push!(lm_names, x isa Dict ? get(x, "landmark", "") : string(x)); end
                    elseif val isa Dict
                        push!(lm_names, get(val, "landmark", ""))
                    elseif val isa String && !isempty(val)
                        push!(lm_names, val)
                    end
                elseif haskey(c, "landmarks")
                    for x in c["landmarks"]; push!(lm_names, x isa Dict ? get(x, "landmark", "") : string(x)); end
                end
                
                mid_x = Int32(div(dims[1], 2))
                min_x_gpu = KernelAbstractions.zeros(backend, Int32, dims[3])
                max_x_gpu = KernelAbstractions.zeros(backend, Int32, dims[3])
                min_x_gpu .= Int32(999999)
                max_x_gpu .= Int32(-1)
                
                k_x_bounds! = compute_slice_x_bounds_kernel!(backend)
                found_any = false
                for lm_name in lm_names
                    isempty(lm_name) && continue
                    m_ref = nothing
                    if occursin("spine", lowercase(lm_name)) || occursin("vertebrae", lowercase(lm_name))
                        m_ref = get_mask_fn("fused_spine")
                    end
                    # For side-qualified constraints, try the side-specific landmark first
                    # (e.g., "scapula" with side="right" → try "scapula_right" before bilateral union)
                    if m_ref === nothing && side !== nothing && !isempty(side)
                        m_ref = get_mask_fn("$(lm_name)_$(lowercase(side))")
                    end
                    if m_ref === nothing; m_ref = get_mask_fn(lm_name); end
                    if m_ref !== nothing
                        m_ref_gpu = (m_ref isa CuArray) ? m_ref : adapt(backend, UInt8.(m_ref .> 0))
                        k_x_bounds!(m_ref_gpu, min_x_gpu, max_x_gpu, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), filter_left, filter_right, mid_x, ndrange=dims[3])
                        found_any = true
                    end
                    m_ref = nothing
                end
                KernelAbstractions.synchronize(backend)

                if found_any
                    min_x_cpu = adapt(Array, min_x_gpu)
                    max_x_cpu = adapt(Array, max_x_gpu)
                    
                    slice_x_limits = fill(Int32(-1), dims[3])
                    slice_center_x = fill(Int32(div(dims[1], 2)), dims[3])
                    only_where_pres = Bool(get(c, "only_where_present", false))
                    valid_ks = findall(max_x_cpu .!= -1)
                    
                    if !isempty(valid_ks)
                        for k in 1:dims[3]
                            src_k = -1
                            if max_x_cpu[k] != -1
                                src_k = k
                            elseif !only_where_pres
                                src_k = valid_ks[argmin(abs.(valid_ks .- k))]
                            end
                            
                            if src_k != -1
                                slice_center_x[k] = if is_left_side
                                    Int32(dims[1])
                                elseif is_right_side
                                    Int32(1)
                                else
                                    Int32(div(min_x_cpu[src_k] + max_x_cpu[src_k], 2))
                                end
                                
                                if b_part in ["min", "left", "min_x"]
                                    slice_x_limits[k] = min_x_cpu[src_k]
                                elseif b_part in ["max", "right", "max_x"]
                                    slice_x_limits[k] = max_x_cpu[src_k]
                                elseif b_part in ["medial"]
                                    slice_x_limits[k] = is_left_side ? min_x_cpu[src_k] : max_x_cpu[src_k]
                                elseif b_part in ["lateral"]
                                    slice_x_limits[k] = is_left_side ? max_x_cpu[src_k] : min_x_cpu[src_k]
                                elseif b_part in ["mid", "center", "mean"]
                                    slice_x_limits[k] = Int32(div(min_x_cpu[src_k] + max_x_cpu[src_k], 2))
                                else
                                    slice_x_limits[k] = max_x_cpu[src_k]
                                end
                            end
                        end
                    end
                    
                    slice_x_limits_gpu = adapt(backend, slice_x_limits)
                    slice_center_x_gpu = adapt(backend, slice_center_x)
                    eff_off = off_mm
                    if c_type == "LeftOf" || (c_type == "LateralTo" && is_left_side) || (c_type == "MedialTo" && is_right_side)
                        eff_off -= 2.0f0
                    elseif c_type == "RightOf" || (c_type == "LateralTo" && is_right_side) || (c_type == "MedialTo" && is_left_side)
                        eff_off += 2.0f0
                    end
                    offset_vox = Int32(round(Int, eff_off / Float32(spacing[1])))
                    
                    mode_int = c_type == "LeftOf" ? Int32(1) :
                               c_type == "RightOf" ? Int32(2) :
                               (c_type == "LateralTo" && is_left_side) ? Int32(1) :
                               (c_type == "LateralTo" && is_right_side) ? Int32(2) :
                               (c_type == "MedialTo" && is_left_side) ? Int32(2) :
                               (c_type == "MedialTo" && is_right_side) ? Int32(1) :
                               c_type == "LateralTo" ? Int32(3) : Int32(4)
                    k_apply_x! = apply_slicewise_x_constraint_kernel!(backend)
                    k_apply_x!(mask_arr, slice_x_limits_gpu, slice_center_x_gpu, offset_vox, mode_int, Int32(div(dims[1], 2)), Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=dims)
                    KernelAbstractions.synchronize(backend)
                end
                continue
            end
            
            lm_raw = get(c, "landmark", get(c, "landmarks", ""))
            lm_names = if lm_raw isa Vector
                [x isa Dict ? get(x, "landmark", "") : string(x) for x in lm_raw]
            elseif lm_raw isa Dict
                [get(lm_raw, "landmark", "")]
            elseif lm_raw isa String && !isempty(lm_raw)
                [lm_raw]
            else
                String[]
            end
            
            if c_type in ["SuperiorToLowestOf", "InferiorToLowestOf", "SuperiorToHighestOf", "InferiorToHighestOf"]
                valid_zs = Float32[]
                parts = get(c, "boundary_parts", [])
                for (idx, lm_name) in enumerate(lm_names)
                    isempty(lm_name) && continue
                    part_elem = (parts isa Vector && idx <= length(parts)) ? parts[idx] : part
                    z_ref = resolve_landmark_z(lm_name, part_elem, 0.0, get_mask_fn, computed_landmarks, dims, spacing, origin, direction; side=(side !== nothing ? side : ""))
                    if z_ref !== nothing
                        push!(valid_zs, z_ref)
                    end
                end
                if !isempty(valid_zs)
                    target_z = occursin("LowestOf", c_type) ? minimum(valid_zs) : maximum(valid_zs)
                    if startswith(c_type, "SuperiorTo")
                        min_z = max(min_z, target_z + off)
                    else
                        max_z = min(max_z, target_z + off)
                    end
                end
                continue
            end
            
            for lm_name in lm_names
                isempty(lm_name) && continue
                
                if c_type == "SuperiorTo" || c_type == "InferiorTo"
                    z_ref = resolve_landmark_z(lm_name, part, 0.0, get_mask_fn, computed_landmarks, dims, spacing, origin, direction; side=(side !== nothing ? side : ""))
                    if z_ref === nothing && haskey(c, "fallback_landmark")
                        fb_part = get(c, "fallback_boundary_part", part)
                        z_ref = resolve_landmark_z(string(c["fallback_landmark"]), fb_part, 0.0, get_mask_fn, computed_landmarks, dims, spacing, origin, direction; side=(side !== nothing ? side : ""))
                    end
                    if z_ref !== nothing
                        if c_type == "SuperiorTo"
                            min_z = max(min_z, z_ref + off)
                        elseif c_type == "InferiorTo"
                            max_z = min(max_z, z_ref + off)
                        end
                    end
                    continue
                end
                
                m_ref = nothing
                if side !== nothing && !isempty(side)
                    m_ref = get_mask_fn("$(lm_name)_$(lowercase(side))")
                end
                if m_ref === nothing
                    m_ref = get_mask_fn(lm_name)
                end
                if m_ref !== nothing
                    bbox = gpu_bounding_box(backend, m_ref)
                    m_ref = nothing
                    
                    if bbox !== nothing
                        k_min = bbox[5] - 1
                        k_max = bbox[6] - 1
                        z_min_p = Float32(origin[3]) + Float32(k_min) * Float32(spacing[3]) * dir_22
                        z_max_p = Float32(origin[3]) + Float32(k_max) * Float32(spacing[3]) * dir_22
                        z_lower = min(z_min_p, z_max_p); z_upper = max(z_min_p, z_max_p)
                        z_ref_val = if part in ["min", "inferior", "min_z", "absolute_min", "abs_min"]
                            z_lower
                        elseif part in ["mid", "center", "mean"]
                            (z_lower + z_upper) / 2.0f0
                        elseif part in ["max", "superior", "max_z", "absolute_max", "abs_max"]
                            z_upper
                        else
                            if c_type == "SuperiorTo"
                                z_lower
                            elseif c_type == "InferiorTo"
                                z_upper
                            else
                                z_upper
                            end
                        end

                        j_min = bbox[3] - 1
                        j_max = bbox[4] - 1
                        y_min_p = Float32(origin[2]) + Float32(j_min) * Float32(spacing[2]) * dir_11
                        y_max_p = Float32(origin[2]) + Float32(j_max) * Float32(spacing[2]) * dir_11
                        y_lower = min(y_min_p, y_max_p); y_upper = max(y_min_p, y_max_p)
                        y_ref_val = if part in ["min", "anterior", "min_y"]
                            y_lower
                        elseif part in ["mid", "center", "mean"]
                            (y_lower + y_upper) / 2.0f0
                        elseif part in ["max", "posterior", "max_y"]
                            y_upper
                        else
                            if c_type == "AnteriorTo"
                                y_upper
                            elseif c_type == "PosteriorTo"
                                y_lower
                            else
                                y_upper
                            end
                        end

                        i_min = bbox[1] - 1
                        i_max = bbox[2] - 1
                        x_min_p = Float32(origin[1]) + Float32(i_min) * Float32(spacing[1]) * dir_00
                        x_max_p = Float32(origin[1]) + Float32(i_max) * Float32(spacing[1]) * dir_00
                        x_lower = min(x_min_p, x_max_p); x_upper = max(x_min_p, x_max_p)
                        x_ref_val = if part in ["min", "left", "min_x"]
                            x_lower
                        elseif part in ["mid", "center", "mean"]
                            (x_lower + x_upper) / 2.0f0
                        elseif part in ["max", "right", "max_x"]
                            x_upper
                        elseif part == "lateral"
                            (side !== nothing && lowercase(side) == "left") ? x_upper : x_lower
                        elseif part == "medial"
                            (side !== nothing && lowercase(side) == "left") ? x_lower : x_upper
                        else
                            if (c_type == "RightOf" || (c_type == "MedialTo" && (side === nothing || lowercase(side) == "left")))
                                x_lower
                            elseif (c_type == "LeftOf" || (c_type == "MedialTo" && (side !== nothing && lowercase(side) == "right")))
                                x_upper
                            elseif (c_type == "LateralTo" && (side === nothing || lowercase(side) == "left"))
                                x_upper
                            elseif (c_type == "LateralTo" && (side !== nothing && lowercase(side) == "right"))
                                x_lower
                            else
                                x_upper
                            end
                        end

                        if c_type == "AnteriorTo"
                            max_y = min(max_y, y_ref_val + off)
                        elseif c_type == "PosteriorTo"
                            min_y = max(min_y, y_ref_val + off)
                        elseif c_type == "SuperiorTo"
                            min_z = max(min_z, z_ref_val + off)
                        elseif c_type == "InferiorTo"
                            max_z = min(max_z, z_ref_val + off)
                        elseif c_type == "LeftOf"
                            eff_off = (off == 0.0f0 ? -2.0f0 : off)
                            min_x = max(min_x, x_ref_val + eff_off)
                        elseif c_type == "RightOf"
                            eff_off = (off == 0.0f0 ? 2.0f0 : off)
                            max_x = min(max_x, x_ref_val + eff_off)
                        elseif c_type == "MedialTo"
                            if side !== nothing && lowercase(side) == "left"
                                eff_off = (off == 0.0f0 ? 2.0f0 : off)
                                max_x = min(max_x, x_ref_val + eff_off)
                            elseif side !== nothing && lowercase(side) == "right"
                                eff_off = (off == 0.0f0 ? -2.0f0 : off)
                                min_x = max(min_x, x_ref_val + eff_off)
                            end
                        elseif c_type == "LateralTo"
                            if side !== nothing && lowercase(side) == "left"
                                eff_off = (off == 0.0f0 ? -2.0f0 : off)
                                min_x = max(min_x, x_ref_val + eff_off)
                            elseif side !== nothing && lowercase(side) == "right"
                                eff_off = (off == 0.0f0 ? 2.0f0 : off)
                                max_x = min(max_x, x_ref_val + eff_off)
                            end
                        end
                    end
                else
                    z_ref = resolve_landmark_z(lm_name, part, off, get_mask_fn, computed_landmarks, dims, spacing, origin, direction; side=(side !== nothing ? side : ""))
                    if z_ref !== nothing
                        if c_type == "SuperiorTo"
                            min_z = max(min_z, z_ref)
                        elseif c_type == "InferiorTo"
                            max_z = min(max_z, z_ref)
                        end
                    end
                end
            end
        end
    end
    
    if min_x > -99999.0f0 || max_x < 99999.0f0 ||
       min_y > -99999.0f0 || max_y < 99999.0f0 ||
       min_z > -99999.0f0 || max_z < 99999.0f0
        if min_z > -99999.0f0 && max_z < 99999.0f0 && min_z > max_z
            min_z, max_z = max_z, min_z
        end
        eff_min_x = min_x > -99999.0f0 ? Float32(floor((min_x - origin[1]) / (spacing[1] * dir_00)) * spacing[1] * dir_00 + origin[1]) : min_x
        eff_max_x = max_x < 99999.0f0 ? Float32(floor((max_x - origin[1]) / (spacing[1] * dir_00)) * spacing[1] * dir_00 + origin[1]) : max_x
        eff_min_y = min_y > -99999.0f0 ? Float32(floor((min_y - origin[2]) / (spacing[2] * dir_11)) * spacing[2] * dir_11 + origin[2]) : min_y
        eff_max_y = max_y < 99999.0f0 ? Float32(floor((max_y - origin[2]) / (spacing[2] * dir_11)) * spacing[2] * dir_11 + origin[2]) : max_y
        eff_min_z = min_z > -99999.0f0 ? Float32(floor((min_z - origin[3]) / (spacing[3] * dir_22)) * spacing[3] * dir_22 + origin[3]) : min_z
        eff_max_z = max_z < 99999.0f0 ? Float32(floor((max_z - origin[3]) / (spacing[3] * dir_22)) * spacing[3] * dir_22 + origin[3]) : max_z


        kernel! = gpu_apply_box_bounds_kernel!(backend)
        kernel!(
            mask_arr,
            Int32(dims[1]), Int32(dims[2]), Int32(dims[3]),
            Float32(spacing[1]), Float32(spacing[2]), Float32(spacing[3]),
            Float32(origin[1]), Float32(origin[2]), Float32(origin[3]),
            Float32(direction[1]), Float32(direction[5]), Float32(direction[9]),
            eff_min_x, eff_max_x,
            eff_min_y, eff_max_y,
            eff_min_z, eff_max_z,
            ndrange=dims
        )
        KernelAbstractions.synchronize(backend)
    end
    return mask_arr
end

# ============================================================
# Geometric Primitives & Splitting
# ============================================================

@kernel function cylinder_primitive_kernel!(
    out::AbstractArray{UInt8, 3},
    dims_x::Int32, dims_y::Int32, dims_z::Int32,
    sp_x::Float32, sp_y::Float32, sp_z::Float32,
    orig_x::Float32, orig_y::Float32, orig_z::Float32,
    dir_00::Float32, dir_11::Float32, dir_22::Float32,
    p1_x::Float32, p1_y::Float32, p1_z::Float32,
    p2_x::Float32, p2_y::Float32, p2_z::Float32,
    radius::Float32
)
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        px = orig_x + (Float32(i) - 1.0f0) * sp_x * dir_00
        py = orig_y + (Float32(j) - 1.0f0) * sp_y * dir_11
        pz = orig_z + (Float32(k) - 1.0f0) * sp_z * dir_22

        vx = p2_x - p1_x
        vy = p2_y - p1_y
        vz = p2_z - p1_z
        len_sq = vx*vx + vy*vy + vz*vz
        
        if len_sq > 0.0001f0
            wx = px - p1_x
            wy = py - p1_y
            wz = pz - p1_z
            
            t = (wx*vx + wy*vy + wz*vz) / len_sq
            t_clamped = clamp(t, 0.0f0, 1.0f0)
            
            qx = p1_x + t_clamped * vx
            qy = p1_y + t_clamped * vy
            qz = p1_z + t_clamped * vz
            
            dx = px - qx
            dy = py - qy
            dz = pz - qz
            dist_sq = dx*dx + dy*dy + dz*dz
            
            if dist_sq <= radius * radius
                out[i, j, k] = UInt8(1)
            end
        end
    end
end

function execute_cylinder_primitive(backend, p1, p2, radius_mm, dims, spacing, origin, direction)
    out = KernelAbstractions.zeros(backend, UInt8, dims)
    dir_00 = Float32(direction[1]); dir_11 = Float32(direction[5]); dir_22 = Float32(direction[9])
    kernel! = cylinder_primitive_kernel!(backend)
    kernel!(
        out,
        Int32(dims[1]), Int32(dims[2]), Int32(dims[3]),
        Float32(spacing[1]), Float32(spacing[2]), Float32(spacing[3]),
        Float32(origin[1]), Float32(origin[2]), Float32(origin[3]),
        dir_00, dir_11, dir_22,
        Float32(p1[1]), Float32(p1[2]), Float32(p1[3]),
        Float32(p2[1]), Float32(p2[2]), Float32(p2[3]),
        Float32(radius_mm),
        ndrange=dims
    )
    KernelAbstractions.synchronize(backend)
    return out
end

@kernel function ellipsoid_primitive_kernel!(
    out::AbstractArray{UInt8, 3},
    dims_x::Int32, dims_y::Int32, dims_z::Int32,
    sp_x::Float32, sp_y::Float32, sp_z::Float32,
    orig_x::Float32, orig_y::Float32, orig_z::Float32,
    dir_00::Float32, dir_11::Float32, dir_22::Float32,
    cx::Float32, cy::Float32, cz::Float32,
    rx::Float32, ry::Float32, rz::Float32
)
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        px = orig_x + (Float32(i) - 1.0f0) * sp_x * dir_00
        py = orig_y + (Float32(j) - 1.0f0) * sp_y * dir_11
        pz = orig_z + (Float32(k) - 1.0f0) * sp_z * dir_22

        dx = (px - cx) / (rx > 0.0001f0 ? rx : 1.0f0)
        dy = (py - cy) / (ry > 0.0001f0 ? ry : 1.0f0)
        dz = (pz - cz) / (rz > 0.0001f0 ? rz : 1.0f0)
        
        if (dx*dx + dy*dy + dz*dz) <= 1.0f0
            out[i, j, k] = UInt8(1)
        end
    end
end

function execute_ellipsoid_primitive(backend, c_pt::Tuple{Float32, Float32, Float32}, radii::Tuple{Float32, Float32, Float32}, dims, spacing, origin, direction)
    out = KernelAbstractions.zeros(backend, UInt8, dims)
    dir_00 = Float32(direction[1]); dir_11 = Float32(direction[5]); dir_22 = Float32(direction[9])
    kernel! = ellipsoid_primitive_kernel!(backend)
    kernel!(
        out,
        Int32(dims[1]), Int32(dims[2]), Int32(dims[3]),
        Float32(spacing[1]), Float32(spacing[2]), Float32(spacing[3]),
        Float32(origin[1]), Float32(origin[2]), Float32(origin[3]),
        dir_00, dir_11, dir_22,
        c_pt[1], c_pt[2], c_pt[3],
        radii[1], radii[2], radii[3],
        ndrange=dims
    )
    KernelAbstractions.synchronize(backend)
    return out
end

@kernel function primary_vector_kernel!(
    out::AbstractArray{UInt8, 3},
    base_arr::AbstractArray{UInt8, 3},
    full_arr::AbstractArray{UInt8, 3},
    dims_x::Int32, dims_y::Int32, dims_z::Int32,
    cp_x::Float32, cp_y::Float32, cp_z::Float32,
    vd_x::Float32, vd_y::Float32, vd_z::Float32
)
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        ix = Float32(i - 1)
        iy = Float32(j - 1)
        iz = Float32(k - 1)
        
        b_val = base_arr[i, j, k]
        f_val = full_arr[i, j, k]
        
        dot_prod = (ix - cp_x)*vd_x + (iy - cp_y)*vd_y + (iz - cp_z)*vd_z
        is_in = (b_val > 0) || ((f_val > 0) && (dot_prod > 0.0f0))
        out[i, j, k] = is_in ? UInt8(1) : UInt8(0)
    end
end

function execute_primary_vector(backend, prim_m::AbstractArray, vec_m::AbstractArray, dims, spacing, origin, direction, base_margin_mm::Float64, exp_mm::Float64)
    base_m = execute_anisotropic_expansion(backend, prim_m, dims, spacing, Dict("all" => base_margin_mm))
    
    sum_x_p, sum_y_p, count_p = get_centroid_per_slice_gpu(backend, prim_m, dims)
    sum_x_v, sum_y_v, count_v = get_centroid_per_slice_gpu(backend, vec_m, dims)
    
    total_count_p = sum(count_p)
    total_count_v = sum(count_v)
    if total_count_p == 0 || total_count_v == 0
        return base_m
    end
    
    cp_x = Float32(sum(sum_x_p) - total_count_p) / total_count_p
    cp_y = Float32(sum(sum_y_p) - total_count_p) / total_count_p
    cp_z = Float32(sum((k - 1) * count_p[k] for k in 1:dims[3])) / total_count_p
    
    cv_x = Float32(sum(sum_x_v) - total_count_v) / total_count_v
    cv_y = Float32(sum(sum_y_v) - total_count_v) / total_count_v
    cv_z = Float32(sum((k - 1) * count_v[k] for k in 1:dims[3])) / total_count_v

    
    vx = cv_x - cp_x; vy = cv_y - cp_y; vz = cv_z - cp_z
    v_norm = sqrt(vx*vx + vy*vy + vz*vz)
    if v_norm <= 0.0001f0
        return base_m
    end
    vd_x = vx / v_norm; vd_y = vy / v_norm; vd_z = vz / v_norm
    
    full_exp = execute_anisotropic_expansion(backend, prim_m, dims, spacing, Dict("all" => exp_mm))
    
    out = KernelAbstractions.zeros(backend, UInt8, dims)
    kernel! = primary_vector_kernel!(backend)
    kernel!(
        out, base_m, full_exp,
        Int32(dims[1]), Int32(dims[2]), Int32(dims[3]),
        cp_x, cp_y, cp_z,
        vd_x, vd_y, vd_z,
        ndrange=dims
    )
    KernelAbstractions.synchronize(backend)
    return out
end

@kernel function convex_hull_bridge_kernel!(
    out::AbstractArray{UInt8, 3},
    m1::AbstractArray{UInt8, 3},
    m2::AbstractArray{UInt8, 3},
    dims_x::Int32, dims_y::Int32, dims_z::Int32
)
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        if m1[i, j, k] > 0 || m2[i, j, k] > 0
            out[i, j, k] = UInt8(1)
        end
    end
end

function cross_2d(o::Tuple{Float64, Float64}, a::Tuple{Float64, Float64}, b::Tuple{Float64, Float64})
    return (a[1] - o[1]) * (b[2] - o[2]) - (a[2] - o[2]) * (b[1] - o[1])
end

function monotone_chain_2d(pts::Vector{Tuple{Float64, Float64}})
    n = length(pts)
    if n <= 2
        return pts
    end
    sorted_pts = sort(unique(pts))
    n_sorted = length(sorted_pts)
    if n_sorted <= 2
        return sorted_pts
    end
    
    # Lower hull
    lower = Tuple{Float64, Float64}[]
    for p in sorted_pts
        while length(lower) >= 2 && cross_2d(lower[end-1], lower[end], p) <= 0
            pop!(lower)
        end
        push!(lower, p)
    end
    
    # Upper hull
    upper = Tuple{Float64, Float64}[]
    for p in reverse(sorted_pts)
        while length(upper) >= 2 && cross_2d(upper[end-1], upper[end], p) <= 0
            pop!(upper)
        end
        push!(upper, p)
    end
    
    pop!(lower)
    pop!(upper)
    return vcat(lower, upper)
end

function point_in_convex_poly_2d(x::Float64, y::Float64, poly::Vector{Tuple{Float64, Float64}}, eps=1e-7)
    n = length(poly)
    for i in 1:n
        p1 = poly[i]
        p2 = poly[i == n ? 1 : i + 1]
        cp = (p2[1] - p1[1]) * (y - p1[2]) - (p2[2] - p1[2]) * (x - p1[1])
        if cp < -eps
            return false
        end
    end
    return true
end

function convex_hull_image_2d(slice_2d::AbstractMatrix{UInt8})
    dims = size(slice_2d)
    out_slice = zeros(UInt8, dims)
    
    idx = findall(slice_2d .> 0)
    if isempty(idx)
        return out_slice
    end
    if length(idx) < 3
        out_slice[idx] .= 1
        return out_slice
    end
    
    # Convert to 0-based coordinates with diamond offsets matching skimage
    pts = Tuple{Float64, Float64}[]
    sizehint!(pts, length(idx) * 4)
    for ci in idx
        r = Float64(ci[1] - 1)
        c = Float64(ci[2] - 1)
        push!(pts, (r - 0.5, c))
        push!(pts, (r + 0.5, c))
        push!(pts, (r, c - 0.5))
        push!(pts, (r, c + 0.5))
    end
    
    poly = monotone_chain_2d(pts)
    n_poly = length(poly)
    if n_poly < 3
        out_slice[idx] .= 1
        return out_slice
    end
    
    min_r = max(1, Int(floor(minimum(p[1] for p in poly))) + 1)
    max_r = min(dims[1], Int(ceil(maximum(p[1] for p in poly))) + 1)
    min_c = max(1, Int(floor(minimum(p[2] for p in poly))) + 1)
    max_c = min(dims[2], Int(ceil(maximum(p[2] for p in poly))) + 1)
    
    for c in min_c:max_c
        cf = Float64(c - 1)
        for r in min_r:max_r
            rf = Float64(r - 1)
            if point_in_convex_poly_2d(rf, cf, poly)
                out_slice[r, c] = 1
            end
        end
    end
    out_slice .|= slice_2d
    return out_slice
end



@kernel function extract_row_bounds_kernel!(min_x_map, max_x_map, m1, m2, dims_x::Int32, dims_y::Int32, dims_z::Int32)
    j, k = @index(Global, NTuple)
    if j <= dims_y && k <= dims_z
        min_i = Int32(dims_x + 1)
        max_i = Int32(0)
        for i in Int32(1):dims_x
            if m1[i, j, k] > 0 || m2[i, j, k] > 0
                if min_i > dims_x
                    min_i = i
                end
                max_i = i
            end
        end
        if max_i >= min_i
            min_x_map[j, k] = min_i
            max_x_map[j, k] = max_i
        else
            min_x_map[j, k] = Int32(0)
            max_x_map[j, k] = Int32(0)
        end
    end
end


@kernel function rasterize_hull_kernel!(out, edges_min_x, edges_max_x, dims_x::Int32, dims_y::Int32, dims_z::Int32)
    j, k = @index(Global, NTuple)
    if j <= dims_y && k <= dims_z
        min_x = edges_min_x[j, k]
        max_x = edges_max_x[j, k]
        if min_x > 0 && max_x > 0
            for i in min_x:max_x
                out[i, j, k] = UInt8(1)
            end
        end
    end
end


@kernel function gpu_monotone_chain_hull_kernel!(
    edges_min_x, edges_max_x,
    @Const(min_x_map), @Const(max_x_map),
    pts_x, pts_y,
    hull_x, hull_y,
    dims_y::Int32, dims_z::Int32
)
    k = @index(Global)
    if k <= dims_z
        num_pts = Int32(0)
        for j in Int32(1):dims_y
            mx1 = min_x_map[j, k]
            if mx1 > Int32(0)
                num_pts += Int32(1)
                pts_x[num_pts, k] = mx1
                pts_y[num_pts, k] = j
                
                mx2 = max_x_map[j, k]
                if mx2 > mx1
                    num_pts += Int32(1)
                    pts_x[num_pts, k] = mx2
                    pts_y[num_pts, k] = j
                end
            end
        end
        
        for j in Int32(1):dims_y
            edges_min_x[j, k] = Int32(0)
            edges_max_x[j, k] = Int32(0)
        end
        
        if num_pts >= Int32(3)
            # Andrew's Monotone Chain (already sorted lexicographically by (y, x))
            # Lower hull
            k_hull = Int32(1)
            for i in Int32(1):num_pts
                px = pts_x[i, k]
                py = pts_y[i, k]
                while k_hull >= Int32(3)
                    p1x = hull_x[k_hull - Int32(2), k]
                    p1y = hull_y[k_hull - Int32(2), k]
                    p2x = hull_x[k_hull - Int32(1), k]
                    p2y = hull_y[k_hull - Int32(1), k]
                    cp = (Int64(p2x) - Int64(p1x)) * (Int64(py) - Int64(p2y)) - (Int64(p2y) - Int64(p1y)) * (Int64(px) - Int64(p2x))
                    if cp <= Int64(0)
                        k_hull -= Int32(1)
                    else
                        break
                    end
                end
                hull_x[k_hull, k] = px
                hull_y[k_hull, k] = py
                k_hull += Int32(1)
            end
            
            # Upper hull
            t = k_hull + Int32(1)
            for i in (num_pts - Int32(1)):Int32(-1):Int32(1)
                px = pts_x[i, k]
                py = pts_y[i, k]
                while k_hull >= t
                    p1x = hull_x[k_hull - Int32(2), k]
                    p1y = hull_y[k_hull - Int32(2), k]
                    p2x = hull_x[k_hull - Int32(1), k]
                    p2y = hull_y[k_hull - Int32(1), k]
                    cp = (Int64(p2x) - Int64(p1x)) * (Int64(py) - Int64(p2y)) - (Int64(p2y) - Int64(p1y)) * (Int64(px) - Int64(p2x))
                    if cp <= Int64(0)
                        k_hull -= Int32(1)
                    else
                        break
                    end
                end
                hull_x[k_hull, k] = px
                hull_y[k_hull, k] = py
                k_hull += Int32(1)
            end
            hull_len = k_hull - Int32(1)
            
            for e in Int32(1):hull_len
                next_e = e < hull_len ? e + Int32(1) : Int32(1)
                p1x = hull_x[e, k]
                p1y = hull_y[e, k]
                p2x = hull_x[next_e, k]
                p2y = hull_y[next_e, k]
                
                if p1y != p2y
                    min_y = min(p1y, p2y)
                    max_y = max(p1y, p2y)
                    for j in min_y:max_y
                        t_ratio = Float32(j - p1y) / Float32(p2y - p1y)
                        x_interp = Int32(round(Float32(p1x) + t_ratio * Float32(p2x - p1x)))
                        
                        cur_min = edges_min_x[j, k]
                        if cur_min == Int32(0) || x_interp < cur_min
                            edges_min_x[j, k] = x_interp
                        end
                        cur_max = edges_max_x[j, k]
                        if cur_max == Int32(0) || x_interp > cur_max
                            edges_max_x[j, k] = x_interp
                        end
                    end
                else
                    j = p1y
                    min_hx = min(p1x, p2x)
                    max_hx = max(p1x, p2x)
                    cur_min = edges_min_x[j, k]
                    if cur_min == Int32(0) || min_hx < cur_min
                        edges_min_x[j, k] = min_hx
                    end
                    cur_max = edges_max_x[j, k]
                    if cur_max == Int32(0) || max_hx > cur_max
                        edges_max_x[j, k] = max_hx
                    end
                end
            end
        elseif num_pts > Int32(0)
            for j in Int32(1):dims_y
                edges_min_x[j, k] = min_x_map[j, k]
                edges_max_x[j, k] = max_x_map[j, k]
            end
        end
    end
end

function execute_convex_hull_bridge(backend, mask_a_gpu, mask_b_gpu, dims, spacing, origin, direction; plane="axial")
    # Handle multi-plane hull via permutation
    if plane == "coronal"
        # Coronal: XZ per Y-slice. Permute (X,Y,Z) -> (X,Z,Y) so Y becomes slice axis
        mask_a_gpu = permutedims(mask_a_gpu, (1, 3, 2))
        mask_b_gpu = permutedims(mask_b_gpu, (1, 3, 2))
        dims = (dims[1], dims[3], dims[2])
    elseif plane == "sagittal"
        # Sagittal: YZ per X-slice. Permute (X,Y,Z) -> (Y,Z,X) so X becomes slice axis  
        mask_a_gpu = permutedims(mask_a_gpu, (2, 3, 1))
        mask_b_gpu = permutedims(mask_b_gpu, (2, 3, 1))
        dims = (dims[2], dims[3], dims[1])
    end

    out = KernelAbstractions.zeros(backend, UInt8, dims)
    dims_x, dims_y, dims_z = Int32(dims[1]), Int32(dims[2]), Int32(dims[3])
    
    min_x_gpu = KernelAbstractions.zeros(backend, Int32, dims_y, dims_z)
    max_x_gpu = KernelAbstractions.zeros(backend, Int32, dims_y, dims_z)
    
    k_extract! = extract_row_bounds_kernel!(backend)
    k_extract!(min_x_gpu, max_x_gpu, mask_a_gpu, mask_b_gpu, dims_x, dims_y, dims_z, ndrange=(dims_y, dims_z))
    KernelAbstractions.synchronize(backend)
    
    edges_min_gpu = KernelAbstractions.zeros(backend, Int32, dims_y, dims_z)
    edges_max_gpu = KernelAbstractions.zeros(backend, Int32, dims_y, dims_z)
    scratch_pts_x = KernelAbstractions.zeros(backend, Int32, 2048, dims_z)
    scratch_pts_y = KernelAbstractions.zeros(backend, Int32, 2048, dims_z)
    scratch_hull_x = KernelAbstractions.zeros(backend, Int32, 2048, dims_z)
    scratch_hull_y = KernelAbstractions.zeros(backend, Int32, 2048, dims_z)
    
    k_hull! = gpu_monotone_chain_hull_kernel!(backend)
    k_hull!(edges_min_gpu, edges_max_gpu, min_x_gpu, max_x_gpu, scratch_pts_x, scratch_pts_y, scratch_hull_x, scratch_hull_y, dims_y, dims_z, ndrange=dims_z)
    KernelAbstractions.synchronize(backend)
    
    k_rast! = rasterize_hull_kernel!(backend)
    k_rast!(out, edges_min_gpu, edges_max_gpu, dims_x, dims_y, dims_z, ndrange=(dims_y, dims_z))
    KernelAbstractions.synchronize(backend)
    
    # Un-permute result back to original orientation
    if plane == "coronal"
        out = permutedims(out, (1, 3, 2))
    elseif plane == "sagittal"
        out = permutedims(out, (3, 1, 2))
    end

    return out
end

@kernel function fill_sideways_kernel!(
    out,
    @Const(medial), @Const(lateral),
    dims_x::Int32, dims_y::Int32, dims_z::Int32
)
    j, k = @index(Global, NTuple)
    if j <= dims_y && k <= dims_z
        m_min = Int32(dims_x + 1)
        m_max = Int32(0)
        for i in Int32(1):dims_x
            if medial[i, j, k] > 0
                if m_min > dims_x; m_min = i; end
                m_max = i
            end
        end
        
        l_min = Int32(dims_x + 1)
        l_max = Int32(0)
        for i in Int32(1):dims_x
            if lateral[i, j, k] > 0
                if l_min > dims_x; l_min = i; end
                l_max = i
            end
        end
        
        if m_max >= m_min && l_max >= l_min
            span_min = min(m_min, l_min)
            span_max = max(m_max, l_max)
            for i in span_min:span_max
                out[i, j, k] = UInt8(1)
            end
        end
    end
end

function execute_sideways_growth_gpu(backend, medial_arrs, lateral_arrs, obstacle_arrs, dims)
    out = KernelAbstractions.zeros(backend, UInt8, dims)
    med_union = KernelAbstractions.zeros(backend, UInt8, dims)
    lat_union = KernelAbstractions.zeros(backend, UInt8, dims)
    
    for arr in medial_arrs
        if arr !== nothing; med_union .|= arr; end
    end
    for arr in lateral_arrs
        if arr !== nothing; lat_union .|= arr; end
    end
    
    k! = fill_sideways_kernel!(backend)
    k!(out, med_union, lat_union, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=(dims[2], dims[3]))
    KernelAbstractions.synchronize(backend)
    
    for arr in obstacle_arrs
        if arr !== nothing
            out .&= .~(arr .> UInt8(0))
        end
    end
    return out
end


"""
    execute_dynamic_surface_split(backend, parent_mask, divider_mask, edge, keep_mode, dims, spacing, origin, direction)
GPU-native dynamic surface splitting.
"""
function execute_dynamic_surface_split(backend, parent_mask::AbstractArray, divider_mask::AbstractArray,
                                       edge::String, keep_mode::String,
                                       dims::Tuple, spacing, origin, direction)
    host_parent = parent_mask isa Array ? parent_mask : adapt(Array, parent_mask)
    host_divider = divider_mask isa Array ? divider_mask : adapt(Array, divider_mask)
    
    surface_arr = zeros(UInt8, dims)
    
    for z in 1:dims[3]
        slice = view(host_divider, :, :, z)
        if !any(slice .> 0)
            continue
        end
        
        if edge == "posterior"
            for x in 1:dims[1]
                col = view(slice, x, :)
                ys = findall(col .> 0)
                if !isempty(ys)
                    min_y = minimum(ys)
                    surface_arr[x, 1:min_y, z] .= 1
                end
            end
        elseif edge == "anterior"
            for x in 1:dims[1]
                col = view(slice, x, :)
                ys = findall(col .> 0)
                if !isempty(ys)
                    max_y = maximum(ys)
                    surface_arr[x, max_y:dims[2], z] .= 1
                end
            end
        elseif edge == "medial"
            xs = findall(vec(any(slice .> 0, dims=2)))
            if !isempty(xs)
                min_x = minimum(xs)
                surface_arr[1:min_x, :, z] .= 1
            end
        end
    end
    
    result = zeros(UInt8, dims)
    if keep_mode == "anterior"
        result .= UInt8.((host_parent .> 0) .& (surface_arr .== 0))
    else
        result .= UInt8.((host_parent .> 0) .& (surface_arr .> 0))
    end
    
    return adapt(typeof(parent_mask), result)
end

function execute_ellipsoid_from_bbox(
    backend::KernelAbstractions.Backend,
    base_mask::AbstractArray,
    side::String,
    dims::Tuple{Int, Int, Int},
    spacing::Tuple{Float64, Float64, Float64},
    origin::Tuple{Float64, Float64, Float64},
    direction::NTuple{9, Float64};
    femur_mask::Union{AbstractArray, Nothing}=nothing,
    foramen_mask::Union{AbstractArray, Nothing}=nothing,
    expand_medial_mm::Float64=30.0,
    radius_multiplier::Float64=1.88,
    center_x_mode::String="center"
)
    idx = adapt(Array, findall(base_mask .> UInt8(0)))
    
    if isempty(idx)
        return KernelAbstractions.zeros(backend, UInt8, dims)
    end
    
    xs = [ci[1] for ci in idx]
    ys = [ci[2] for ci in idx]
    zs = [ci[3] for ci in idx]
    
    min_x, max_x = minimum(xs), maximum(xs)
    min_y, max_y = minimum(ys), maximum(ys)
    min_z, max_z = minimum(zs), maximum(zs)
    
    c_x = (min_x + max_x) / 2.0
    c_y = (min_y + max_y) / 2.0
    c_z = (min_z + max_z) / 2.0
    
    baseline_height_mm = max(40.0, (max_z - min_z) * spacing[3])
    rz_mm = (baseline_height_mm / 2.0) * radius_multiplier
    
    if foramen_mask !== nothing
        f_idx = adapt(Array, findall(foramen_mask .> UInt8(0)))
        
        if length(f_idx) > 50
            f_zs = [ci[3] for ci in f_idx]
            comb_min_z = min(min_z, minimum(f_zs))
            comb_max_z = max(max_z, maximum(f_zs))
            c_z = (comb_min_z + comb_max_z) / 2.0
            rz_mm = ((comb_max_z - comb_min_z) * spacing[3] / 2.0) * radius_multiplier
            if center_x_mode != "center"
                f_xs = [ci[1] for ci in f_idx]
                f_ys = [ci[2] for ci in f_idx]
                c_x = (minimum(f_xs) + maximum(f_xs)) / 2.0
                c_y = (minimum(f_ys) + maximum(f_ys)) / 2.0
            end
        else
            c_z = max_z - (rz_mm / spacing[3])
        end
    end
    
    if center_x_mode == "medial_edge"
        c_x = (lowercase(side) == "left") ? min_x : max_x
    end
    
    rx_mm = expand_medial_mm * radius_multiplier
    ry_mm = ((max_y - min_y) * spacing[2] / 2.0) * radius_multiplier
    
    rx_vox = rx_mm / spacing[1]
    ry_vox = ry_mm / spacing[2]
    rz_vox = rz_mm / spacing[3]
    
    host_out = zeros(UInt8, dims)
    rx2 = rx_vox^2; ry2 = ry_vox^2; rz2 = rz_vox^2
    
    # Generate ellipsoid voxels and clip superior to max_z
    for k in 1:min(max_z, dims[3])
        dz2 = (k - c_z)^2 / rz2
        if dz2 <= 1.0
            rem_y = 1.0 - dz2
            for j in 1:dims[2]
                dy2 = (j - c_y)^2 / ry2
                if dy2 <= rem_y
                    rem_x = rem_y - dy2
                    dx_max = sqrt(rem_x * rx2)
                    i_start = max(1, ceil(Int, c_x - dx_max))
                    i_end = min(dims[1], floor(Int, c_x + dx_max))
                    if i_start <= i_end
                        host_out[i_start:i_end, j, k] .= UInt8(1)
                    end
                end
            end
        end
    end
    
    # Lateral Femur Filter (Per slice)
    if femur_mask !== nothing
        fe_idx = adapt(Array, findall(femur_mask .> UInt8(0)))
        for k in 1:dims[3]
            femur_slice = view(host_femur, :, :, k)
            f_idx_slice = findall(femur_slice .> 0)
            if !isempty(f_idx_slice)
                xf = [ci[1] for ci in f_idx_slice]
                if lowercase(side) == "left"
                    limit_x = minimum(xf)
                    host_out[limit_x:end, :, k] .= UInt8(0)
                else
                    limit_x = maximum(xf)
                    host_out[1:limit_x, :, k] .= UInt8(0)
                end
            end
        end
    end
    
    return adapt(typeof(base_mask), host_out)
end

function convex_hull_2d_local(pts::Vector{Tuple{Int, Int}})
    n = length(pts)
    if n <= 2 return pts end
    sort!(pts)
    cross_prod(o, a, b) = (a[1] - o[1]) * (b[2] - o[2]) - (a[2] - o[2]) * (b[1] - o[1])
    lower = Tuple{Int, Int}[]
    for p in pts
        while length(lower) >= 2 && cross_prod(lower[end-1], lower[end], p) <= 0
            pop!(lower)
        end
        push!(lower, p)
    end
    upper = Tuple{Int, Int}[]
    for i in n:-1:1
        p = pts[i]
        while length(upper) >= 2 && cross_prod(upper[end-1], upper[end], p) <= 0
            pop!(upper)
        end
        push!(upper, p)
    end
    pop!(lower)
    pop!(upper)
    return vcat(lower, upper)
end

function fill_2d_polygon_local!(slice_out::AbstractMatrix{UInt8}, hull::Vector{Tuple{Int, Int}})
    if length(hull) < 3
        for p in hull
            slice_out[p[1], p[2]] = UInt8(1)
        end
        return
    end
    min_y = minimum(p[2] for p in hull)
    max_y = maximum(p[2] for p in hull)
    n = length(hull)
    for y in min_y:max_y
        nodes = Int[]
        j = n
        for i in 1:n
            p_i = hull[i]
            p_j = hull[j]
            if (p_i[2] < y && p_j[2] >= y) || (p_j[2] < y && p_i[2] >= y)
                x = p_i[1] + (y - p_i[2]) / (p_j[2] - p_i[2]) * (p_j[1] - p_i[1])
                push!(nodes, round(Int, x))
            end
            j = i
        end
        sort!(nodes)
        for idx in 1:2:length(nodes)
            if idx + 1 <= length(nodes)
                for x in nodes[idx]:nodes[idx+1]
                    if 1 <= x <= size(slice_out, 1) && 1 <= y <= size(slice_out, 2)
                        slice_out[x, y] = UInt8(1)
                    end
                end
            end
        end
    end
end




@kernel function split_neck_2_kernel!(out_2a, out_2b, parent_m, max_y_arr, dims_y)
    i, j, k = @index(Global, NTuple)
    pv = parent_m[i, j, k]
    my = max_y_arr[k]
    if my > Int32(0)
        # IIa = anterior (Y <= max_y), IIb = posterior (Y > max_y)
        out_2a[i, j, k] = (j <= my) ? pv : UInt8(0)
        out_2b[i, j, k] = (j > my) ? pv : UInt8(0)
    else
        out_2a[i, j, k] = pv
        out_2b[i, j, k] = pv
    end
end

@kernel function compute_max_y_kernel!(max_y_arr, ijv_mask, dims_x, dims_y, dims_z)
    I, J, K = @index(Global, NTuple)
    if I <= dims_x && J <= dims_y && K <= dims_z
        if ijv_mask[I, J, K] > UInt8(0)
            KernelAbstractions.@atomic max_y_arr[K] max J
        end
    end
end

function split_neck_level_2(
    parent_mask::AbstractArray,
    ijv_mask::AbstractArray,
    dims::Tuple{Int, Int, Int}
)
    if parent_mask isa Array
        # CPU path (unchanged)
        out_2a = copy(parent_mask)
        out_2b = copy(parent_mask)
        for k in 1:dims[3]
            ijv_slice = view(ijv_mask, :, :, k)
            idx = findall(ijv_slice .> 0)
            if !isempty(idx)
                ijv_post_y = maximum(ci[2] for ci in idx)
                out_2a[:, (ijv_post_y+1):end, k] .= UInt8(0)
                out_2b[:, 1:ijv_post_y, k] .= UInt8(0)
            end
        end
        return (out_2a, out_2b)
    else
        # GPU path — no 131 MB CPU transfers
        backend = KernelAbstractions.get_backend(parent_mask)
        max_y_gpu = KernelAbstractions.zeros(backend, Int32, dims[3])
        
        k_max! = compute_max_y_kernel!(backend)
        k_max!(max_y_gpu, ijv_mask, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=dims)
        KernelAbstractions.synchronize(backend)
        
        out_2a = similar(parent_mask)
        out_2b = similar(parent_mask)
        
        k_split! = split_neck_2_kernel!(backend)
        k_split!(out_2a, out_2b, parent_mask, max_y_gpu, Int32(dims[2]), ndrange=dims)
        KernelAbstractions.synchronize(backend)
        
        return (out_2a, out_2b)
    end
end

function get_largest_connected_component(arr::AbstractArray{T, 3}; dest=nothing) where T
    # GPU-native CCL path — zero CPU-GPU transfers (except 1 scalar from findmax)
    if haskey(StaticArena.CCL_ARENA, :labels) && !(arr isa Array)
        labels_buf = StaticArena.CCL_ARENA[:labels]
        counts_buf = StaticArena.CCL_ARENA[:counts]
        output_buf = dest !== nothing ? dest : StaticArena.CCL_ARENA[:output]
        dims = size(arr)
        backend = KernelAbstractions.get_backend(arr)
        
        # Convert input to UInt8 if needed (zero allocation if already CuArray{UInt8, 3})
        mask_u8 = (arr isa CuArray{UInt8, 3}) ? arr : map(x -> x > zero(T) ? UInt8(1) : UInt8(0), arr)
        
        block_val = get(StaticArena.CCL_ARENA, :block_val, nothing)
        block_idx = get(StaticArena.CCL_ARENA, :block_idx, nothing)
        
        GpuCCL.gpu_largest_connected_component!(
            backend, output_buf, mask_u8, dims, labels_buf, counts_buf;
            block_val=block_val, block_idx=block_idx
        )
        if dest !== nothing
            return dest
        else
            return copy(output_buf)
        end
    end
    
    # CPU fallback (for non-GPU arrays)
    host_arr = arr isa Array ? arr : Array(arr)
    vox_count = count(host_arr .> 0)
    if vox_count <= 1
        return arr
    end
    dims = size(host_arr)
    visited = zeros(Bool, dims)
    best_component = Tuple{Int, Int, Int}[]
    curr_component = Tuple{Int, Int, Int}[]
    queue = Tuple{Int, Int, Int}[]
    sizehint!(queue, min(vox_count, 100000))
    sizehint!(curr_component, min(vox_count, 100000))
    
    for z in 1:dims[3], y in 1:dims[2], x in 1:dims[1]
        if host_arr[x, y, z] > 0 && !visited[x, y, z]
            empty!(curr_component)
            empty!(queue)
            
            visited[x, y, z] = true
            push!(queue, (x, y, z))
            push!(curr_component, (x, y, z))
            
            head = 1
            while head <= length(queue)
                cx, cy, cz = queue[head]
                head += 1
                
                # 26-connected neighbors (matching SimpleITK default)
                for dz in -1:1, dy in -1:1, dx in -1:1
                    (dx == 0 && dy == 0 && dz == 0) && continue
                    nx, ny, nz = cx + dx, cy + dy, cz + dz
                    if nx >= 1 && nx <= dims[1] && ny >= 1 && ny <= dims[2] && nz >= 1 && nz <= dims[3]
                        if host_arr[nx, ny, nz] > 0 && !visited[nx, ny, nz]
                            visited[nx, ny, nz] = true
                            push!(queue, (nx, ny, nz))
                            push!(curr_component, (nx, ny, nz))
                        end
                    end
                end
            end
            
            if length(curr_component) > length(best_component)
                best_component = copy(curr_component)
            end
        end
    end
    
    host_out = zeros(UInt8, dims)
    for (x, y, z) in best_component
        host_out[x, y, z] = 1
    end
    return adapt(typeof(arr), host_out)
end


@kernel function execute_z_prop_kernel!(out_m, in_m, dims_x::Int32, dims_y::Int32, dims_z::Int32, step::Int32, rng_start::Int32, rng_end::Int32, p_stop::Int32)
    i, j = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y
        has_active = false
        curr_k = rng_start
        while (step > 0 ? curr_k <= rng_end : curr_k >= rng_end)
            if in_m[i, j, curr_k] > 0
                has_active = true
            end
            if has_active
                out_m[i, j, curr_k] = UInt8(1)
            end
            if p_stop != -1 && curr_k == p_stop
                has_active = false # Stop propagating past the stop plane
            end
            curr_k += step
        end
    end
end

function execute_z_propagation(backend, seed_m::AbstractArray{UInt8, 3}, direction_str::String, stop_m::Union{AbstractArray{UInt8, 3}, Nothing}, direction_matrix, dims::Tuple{Int, Int, Int})
    dims_x, dims_y, dims_z = Int32(dims[1]), Int32(dims[2]), Int32(dims[3])
    z_vector_z = Float64(direction_matrix[9])
    
    inf_bound = Int32(-1)
    sup_bound = Int32(-1)
    if stop_m !== nothing
        bbox = gpu_bounding_box(backend, stop_m)
        if bbox !== nothing
            stop_z_min, stop_z_max = Int32(bbox[5]), Int32(bbox[6])
            inf_bound = z_vector_z > 0 ? stop_z_min : stop_z_max
            sup_bound = z_vector_z > 0 ? stop_z_max : stop_z_min
        end
    end
    
    function smear(dir_name::String)
        res = copy(seed_m)
        step = Int32(dir_name == "inferior" ? (z_vector_z > 0 ? -1 : 1) : (z_vector_z > 0 ? 1 : -1))
        rng_start = step == -1 ? dims_z : Int32(1)
        rng_end = step == -1 ? Int32(1) : dims_z
        p_stop = dir_name == "inferior" ? inf_bound : sup_bound
        
        kernel! = execute_z_prop_kernel!(backend)
        kernel!(res, seed_m, dims_x, dims_y, dims_z, step, rng_start, rng_end, p_stop, ndrange=(dims_x, dims_y))
        KernelAbstractions.synchronize(backend)
        return res
    end
    
    norm_dir = lowercase(direction_str)
    out_arr = if norm_dir == "both"
        smear("inferior") .| smear("superior")
    else
        smear(norm_dir)
    end
    
    return out_arr
end



@kernel function zero_out_split_kernel!(out_m, axis_idx::Int32, split_val::Int32, keep_min::Bool, dims_x::Int32, dims_y::Int32, dims_z::Int32)
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        val = (axis_idx == 1) ? i : ((axis_idx == 2) ? j : k)
        if keep_min
            if val >= split_val
                out_m[i, j, k] = UInt8(0)
            end
        else
            if val < split_val
                out_m[i, j, k] = UInt8(0)
            end
        end
    end
end

function execute_split_mask(backend, input_m::AbstractArray{UInt8, 3}, axis_str::String, method_str::String, keep_str::String, dims::Tuple{Int, Int, Int})
    axis_idx = Int32(lowercase(axis_str) == "z" ? 3 : (lowercase(axis_str) == "y" ? 2 : 1))
    keep_min = !(lowercase(keep_str) in ["positive", "max", "pos", "right"])
    
    bbox = gpu_bounding_box(backend, input_m)
    if bbox === nothing
        return copy(input_m)
    end
    
    c_min = axis_idx == 1 ? bbox[1] : (axis_idx == 2 ? bbox[3] : bbox[5])
    c_max = axis_idx == 1 ? bbox[2] : (axis_idx == 2 ? bbox[4] : bbox[6])
    
    split_val = if startswith(lowercase(method_str), "ratio_")
        ratio = parse(Float32, replace(lowercase(method_str), "ratio_" => ""))
        floor(Int32, c_min - 1 + (c_max - c_min) * ratio) + Int32(1)
    elseif lowercase(method_str) == "centroid"
        sum_x, sum_y, count = get_centroid_per_slice_gpu(backend, input_m, dims)
        total_count = sum(count)
        if total_count == 0
            c_min
        else
            if axis_idx == 1
                floor(Int32, Float32(sum(sum_x) - total_count) / total_count) + Int32(1)
            elseif axis_idx == 2
                floor(Int32, Float32(sum(sum_y) - total_count) / total_count) + Int32(1)
            else
                floor(Int32, Float32(sum((k - 1) * count[k] for k in 1:dims[3])) / total_count) + Int32(1)
            end
        end
    elseif lowercase(method_str) == "image_center"
        div(dims[axis_idx], 2)
    else # center
        div((c_min - 1) + (c_max - 1), 2) + Int32(1)
    end
    
    out_m = copy(input_m)
    kernel! = zero_out_split_kernel!(backend)
    kernel!(out_m, axis_idx, Int32(split_val), keep_min, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=dims)
    KernelAbstractions.synchronize(backend)
    
    return out_m
end


# ============================================================
# Iliac System Implementations
# ============================================================


@kernel function apply_packed_exclusions_kernel!(
    mask::AbstractArray{UInt8, 3},
    exclusion_dyn::AbstractArray{UInt8, 3},
    packed_data::AbstractArray{UInt8, 4},
    excl_ch::AbstractArray{Int32, 1},
    excl_id::AbstractArray{UInt8, 1},
    num_excl::Int32
)
    i, j, k = @index(Global, NTuple)
    
    if mask[i, j, k] > 0
        if exclusion_dyn[i, j, k] > 0
            mask[i, j, k] = UInt8(0)
        else
            for idx in 1:num_excl
                if packed_data[i, j, k, excl_ch[idx]] == excl_id[idx]
                    mask[i, j, k] = UInt8(0)
                    break
                end
            end
        end
    end
end

function apply_packed_exclusions!(backend, mask, exclusion_dyn, packed_tensor, excl_names, get_mask_into_fn)
    dims = size(mask)
    m_buf = nothing
    _mbuf_key = nothing
    
    excl_ch = Int32[]
    excl_id = UInt8[]
    
    for name in excl_names
        ch_info = get(packed_tensor.registry, name, nothing)
        if ch_info !== nothing
            push!(excl_ch, Int32(ch_info[1]))
            push!(excl_id, UInt8(ch_info[2]))
        else
            if m_buf === nothing
                m_buf, _mbuf_key = StaticArena.acquire_mask(backend)
            end
            if get_mask_into_fn(name, m_buf)
                exclusion_dyn .|= m_buf
            end
        end
    end
    if _mbuf_key !== nothing; StaticArena.release_mask(_mbuf_key); end
    
    num_excl = Int32(length(excl_ch))
    if num_excl > 0 || any(exclusion_dyn .> 0)
        excl_ch_gpu = adapt(backend, excl_ch)
        excl_id_gpu = adapt(backend, excl_id)
        
        kernel! = apply_packed_exclusions_kernel!(backend)
        kernel!(mask, exclusion_dyn, packed_tensor.data, excl_ch_gpu, excl_id_gpu, num_excl, ndrange=dims)
        KernelAbstractions.synchronize(backend)
    end
end


@kernel function split_oblique_kernel!(out_arr, exp_m, cx_l_map, cy_l_map, cx_r_map, cy_r_map, z_min, z_max, dims, is_left::Bool)
    i, j, k = @index(Global, NTuple)
    if i <= dims[1] && j <= dims[2] && k >= z_min && k <= z_max
        if exp_m[i, j, k] > 0
            cx_l = cx_l_map[k]
            cy_l = cy_l_map[k]
            cx_r = cx_r_map[k]
            cy_r = cy_r_map[k]
            if cx_l >= 0.0f0 && cx_r >= 0.0f0
                mid_x = (cx_l + cx_r) / 2.0f0
                mid_y = (cy_l + cy_r) / 2.0f0
                vec_x = cx_l - cx_r
                vec_y = cy_l - cy_r
                
                val = (Float32(i) - mid_x) * vec_x + (Float32(j) - mid_y) * vec_y
                if is_left
                    if val >= 0.0f0
                        out_arr[i, j, k] = UInt8(1)
                    end
                else
                    if val < 0.0f0
                        out_arr[i, j, k] = UInt8(1)
                    end
                end
            else
                out_arr[i, j, k] = exp_m[i, j, k]
            end
        end
    end
end

function execute_external_iliac(
    backend::KernelAbstractions.Backend,
    get_mask_fn::Function,
    get_mask_into_fn::Function,
    packed_tensor,
    side::String,
    dims::Tuple{Int, Int, Int},
    spacing::Tuple{Float64, Float64, Float64},
    origin::Tuple{Float64, Float64, Float64},
    direction::Tuple,
    computed_landmarks::Dict
)
    s_lower = lowercase(side)
    is_left = (s_lower == "left")
    is_ras = direction[1] > 0.0
    keep_larger = is_ras ? !is_left : is_left
    keep_side_str = keep_larger ? "max" : "min"

    
    art_l = get_mask_fn("iliac_artery_left")
    art_r = get_mask_fn("iliac_artery_right")
    ven_l = get_mask_fn("iliac_vena_left")
    ven_r = get_mask_fn("iliac_vena_right")
    
    art = is_left ? art_l : art_r
    ven = is_left ? ven_l : ven_r
    
    if art === nothing && ven === nothing
        return KernelAbstractions.zeros(backend, UInt8, dims)
    end
    
    comb = art !== nothing && ven !== nothing ? (art .| ven) : (art !== nothing ? art : ven)
    
    excl_names = String[]
    for ex in ["sacrum", "hip", "urinary_bladder", "rectum", "prostate", "uterus", "iliopsoas", "piriformis", "obturator_internus"]
        push!(excl_names, ex)
        push!(excl_names, "$(ex)_left")
        push!(excl_names, "$(ex)_right")
        push!(excl_names, "$(ex)_$s_lower")
    end
    
    exclusion_dyn, _edyn_key = StaticArena.acquire_mask(backend)
    
    exp_m = execute_anisotropic_expansion(backend, comb, dims, spacing, Dict("all" => 7.0))
    apply_packed_exclusions!(backend, exp_m, exclusion_dyn, packed_tensor, excl_names, get_mask_into_fn)
    StaticArena.release_mask(_edyn_key)
    exp_m .= ifelse.(comb .> UInt8(0), UInt8(0), exp_m)
    
    femur = get_mask_fn("femur")
    if femur === nothing; femur = get_mask_fn("femur_$s_lower"); end
    z_femur = 65
    if femur !== nothing
        bbox = gpu_bounding_box(backend, femur)
        if bbox !== nothing
            z_femur = bbox[6]
        end
    end
    
    z_bif = 96
    p1_key = "internal_iliac_p1_$s_lower"
    if haskey(computed_landmarks, p1_key)
        p1 = computed_landmarks[p1_key]
        z_bif = clamp(round(Int, (Float64(p1[3]) - origin[3]) / (spacing[3] * Float64(direction[9]))) + 1, 1, dims[3])
    end
    
    z_min = min(z_femur, z_bif - 3)
    z_max = max(z_femur, z_bif - 3)
    
    out_arr = KernelAbstractions.zeros(backend, UInt8, dims)
    
    sum_x_l, sum_y_l, count_l = get_centroid_per_slice_gpu(backend, art_l !== nothing ? art_l : KernelAbstractions.zeros(backend, UInt8, dims), dims)
    sum_x_r, sum_y_r, count_r = get_centroid_per_slice_gpu(backend, art_r !== nothing ? art_r : KernelAbstractions.zeros(backend, UInt8, dims), dims)
    
    cx_l_map = KernelAbstractions.allocate(backend, Float32, dims[3])
    cy_l_map = KernelAbstractions.allocate(backend, Float32, dims[3])
    cx_r_map = KernelAbstractions.allocate(backend, Float32, dims[3])
    cy_r_map = KernelAbstractions.allocate(backend, Float32, dims[3])
    
    cx_l_host = zeros(Float32, dims[3])
    cy_l_host = zeros(Float32, dims[3])
    cx_r_host = zeros(Float32, dims[3])
    cy_r_host = zeros(Float32, dims[3])
    
    for k in 1:dims[3]
        if count_l[k] > 0
            cx_l_host[k] = sum_x_l[k] / count_l[k]
            cy_l_host[k] = sum_y_l[k] / count_l[k]
        else
            cx_l_host[k] = -1.0f0
        end
        if count_r[k] > 0
            cx_r_host[k] = sum_x_r[k] / count_r[k]
            cy_r_host[k] = sum_y_r[k] / count_r[k]
        else
            cx_r_host[k] = -1.0f0
        end
    end
    
    copyto!(cx_l_map, cx_l_host)
    copyto!(cy_l_map, cy_l_host)
    copyto!(cx_r_map, cx_r_host)
    copyto!(cy_r_map, cy_r_host)
    
    k_split! = split_oblique_kernel!(backend)
    k_split!(out_arr, exp_m, cx_l_map, cy_l_map, cx_r_map, cy_r_map, Int(z_min), Int(z_max), dims, is_left, ndrange=dims)
    KernelAbstractions.synchronize(backend)
    
    return out_arr
end

function execute_common_iliac(
    backend::KernelAbstractions.Backend,
    get_mask_fn::Function,
    side::String,
    dims::Tuple{Int, Int, Int},
    spacing::Tuple{Float64, Float64, Float64},
    origin::Tuple{Float64, Float64, Float64},
    direction::Tuple,
    computed_landmarks::Dict
)
    s_lower = lowercase(side)
    is_left = (s_lower == "left")
    is_ras = direction[1] > 0.0
    keep_larger = is_ras ? !is_left : is_left
    keep_side_str = keep_larger ? "max" : "min"

    
    art_l = get_mask_fn("iliac_artery_left")
    art_r = get_mask_fn("iliac_artery_right")
    ven_l = get_mask_fn("iliac_vena_left")
    ven_r = get_mask_fn("iliac_vena_right")
    
    art = is_left ? art_l : art_r
    ven = is_left ? ven_l : ven_r
    
    if art === nothing && ven === nothing
        return KernelAbstractions.zeros(backend, UInt8, dims)
    end
    
    comb = art !== nothing && ven !== nothing ? (art .| ven) : (art !== nothing ? art : ven)
    
    excl_names = String[]
    for ex in ["sacrum", "vertebrae_L5", "psoas_major", "iliopsoas"]
        push!(excl_names, ex)
        push!(excl_names, "$(ex)_left")
        push!(excl_names, "$(ex)_right")
        push!(excl_names, "$(ex)_$s_lower")
    end
    
    exclusion_dyn, _edyn_key = StaticArena.acquire_mask(backend)
    
    exp_m = execute_anisotropic_expansion(backend, comb, dims, spacing, Dict("all" => 7.0))
    apply_packed_exclusions!(backend, exp_m, exclusion_dyn, packed_tensor, excl_names, get_mask_into_fn)
    StaticArena.release_mask(_edyn_key)
    exp_m .= ifelse.(comb .> UInt8(0), UInt8(0), exp_m)
    
    aorta = get_mask_fn("aorta")
    z_aorta = 109
    if aorta !== nothing
        bbox = gpu_bounding_box(backend, aorta)
        if bbox !== nothing
            z_aorta = bbox[5]
        end
    end
    
    z_bif = 96
    p1_key = "internal_iliac_p1_$s_lower"
    if haskey(computed_landmarks, p1_key)
        p1 = computed_landmarks[p1_key]
        z_bif = clamp(round(Int, (Float64(p1[3]) - origin[3]) / (spacing[3] * Float64(direction[9]))) + 1, 1, dims[3])
    end
    
    z_min = min(z_bif + 3, z_aorta)
    z_max = max(z_bif + 3, z_aorta)
    
    out_arr = KernelAbstractions.zeros(backend, UInt8, dims)
    
    sum_x_l, sum_y_l, count_l = get_centroid_per_slice_gpu(backend, art_l !== nothing ? art_l : KernelAbstractions.zeros(backend, UInt8, dims), dims)
    sum_x_r, sum_y_r, count_r = get_centroid_per_slice_gpu(backend, art_r !== nothing ? art_r : KernelAbstractions.zeros(backend, UInt8, dims), dims)
    
    cx_l_map = KernelAbstractions.allocate(backend, Float32, dims[3])
    cy_l_map = KernelAbstractions.allocate(backend, Float32, dims[3])
    cx_r_map = KernelAbstractions.allocate(backend, Float32, dims[3])
    cy_r_map = KernelAbstractions.allocate(backend, Float32, dims[3])
    
    cx_l_host = zeros(Float32, dims[3])
    cy_l_host = zeros(Float32, dims[3])
    cx_r_host = zeros(Float32, dims[3])
    cy_r_host = zeros(Float32, dims[3])
    
    for k in 1:dims[3]
        if count_l[k] > 0
            cx_l_host[k] = sum_x_l[k] / count_l[k]
            cy_l_host[k] = sum_y_l[k] / count_l[k]
        else
            cx_l_host[k] = -1.0f0
        end
        if count_r[k] > 0
            cx_r_host[k] = sum_x_r[k] / count_r[k]
            cy_r_host[k] = sum_y_r[k] / count_r[k]
        else
            cx_r_host[k] = -1.0f0
        end
    end
    
    copyto!(cx_l_map, cx_l_host)
    copyto!(cy_l_map, cy_l_host)
    copyto!(cx_r_map, cx_r_host)
    copyto!(cy_r_map, cy_r_host)
    
    k_split! = split_oblique_kernel!(backend)
    k_split!(out_arr, exp_m, cx_l_map, cy_l_map, cx_r_map, cy_r_map, Int(z_min), Int(z_max), dims, is_left, ndrange=dims)
    KernelAbstractions.synchronize(backend)
    
    return out_arr
end

function execute_iliac_bifurcation(
    backend::KernelAbstractions.Backend,
    get_mask_fn::Function,
    side::String,
    dims::Tuple{Int, Int, Int},
    spacing::Tuple{Float64, Float64, Float64},
    origin::Tuple{Float64, Float64, Float64},
    direction::Tuple,
    computed_landmarks::Dict
)
    s_lower = lowercase(side)
    is_left = (s_lower == "left")
    is_ras = direction[1] > 0.0
    keep_larger = is_ras ? !is_left : is_left
    keep_side_str = keep_larger ? "max" : "min"

    
    art_l = get_mask_fn("iliac_artery_left")
    art_r = get_mask_fn("iliac_artery_right")
    ven_l = get_mask_fn("iliac_vena_left")
    ven_r = get_mask_fn("iliac_vena_right")
    
    art = is_left ? art_l : art_r
    ven = is_left ? ven_l : ven_r
    
    if art === nothing && ven === nothing
        return KernelAbstractions.zeros(backend, UInt8, dims)
    end
    
    comb = art !== nothing && ven !== nothing ? (art .| ven) : (art !== nothing ? art : ven)
    
    excl_names = String[]
    for ex in ["sacrum", "hip", "urinary_bladder", "iliopsoas"]
        push!(excl_names, ex)
        push!(excl_names, "$(ex)_left")
        push!(excl_names, "$(ex)_right")
        push!(excl_names, "$(ex)_$s_lower")
    end
    
    exclusion_dyn, _edyn_key = StaticArena.acquire_mask(backend)
    
    exp_m = execute_anisotropic_expansion(backend, comb, dims, spacing, Dict("all" => 7.0))
    apply_packed_exclusions!(backend, exp_m, exclusion_dyn, packed_tensor, excl_names, get_mask_into_fn)
    StaticArena.release_mask(_edyn_key)
    exp_m .= ifelse.(comb .> UInt8(0), UInt8(0), exp_m)
    
    z_bif = 96
    p1_key = "internal_iliac_p1_$s_lower"
    if haskey(computed_landmarks, p1_key)
        p1 = computed_landmarks[p1_key]
        z_bif = clamp(round(Int, (Float64(p1[3]) - origin[3]) / (spacing[3] * Float64(direction[9]))) + 1, 1, dims[3])
    end
    
    z_offset = max(1, round(Int, 10.0 / spacing[3]))
    z_min = max(1, z_bif - z_offset)
    z_max = min(dims[3], z_bif + z_offset)
    
    h_exp = adapt(Array, exp_m)
    h_art_l = art_l !== nothing ? adapt(Array, art_l) : zeros(UInt8, dims)
    h_art_r = art_r !== nothing ? adapt(Array, art_r) : zeros(UInt8, dims)
    
    out_arr = zeros(UInt8, dims)
    for z in z_min:z_max
        sl = view(h_exp, :, :, z)
        if any(sl .> 0)
            idx_l = findall(view(h_art_l, :, :, z) .> 0)
            idx_r = findall(view(h_art_r, :, :, z) .> 0)
            if !isempty(idx_l) && !isempty(idx_r)
                mean_x_l = sum(ci[1] for ci in idx_l) / length(idx_l)
                mean_x_r = sum(ci[1] for ci in idx_r) / length(idx_r)
                mid_x = round(Int, (mean_x_l + mean_x_r) / 2.0)
                
                for y in 1:dims[2], x in 1:dims[1]
                    if sl[x, y] > 0
                        if is_left && x >= mid_x
                            out_arr[x, y, z] = UInt8(1)
                        elseif !is_left && x < mid_x
                            out_arr[x, y, z] = UInt8(1)
                        end
                    end
                end
            else
                out_arr[:, :, z] .= sl
            end
        end
    end
    
    return adapt(typeof(exp_m), out_arr)
end




@kernel function dilate_2d_slice_kernel!(out_slice, in_slice, dims_x::Int32, dims_y::Int32)
    i, j = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y
        if in_slice[i, j] > 0
            out_slice[i, j] = UInt8(1)
            if i > 1; out_slice[i-1, j] = UInt8(1); end
            if i < dims_x; out_slice[i+1, j] = UInt8(1); end
            if j > 1; out_slice[i, j-1] = UInt8(1); end
            if j < dims_y; out_slice[i, j+1] = UInt8(1); end
        end
    end
end

function run_layer_wise_propagation_gpu!(backend, mask, target, direction::String, step_dilation_px::Int, max_slices::Int, dims)
    dims_x, dims_y, dims_z = Int32(dims[1]), Int32(dims[2]), Int32(dims[3])
    
    # Extract min/max Z to find seed
    bbox = gpu_bounding_box(backend, mask)
    if bbox === nothing
        return
    end
    z_min, z_max = Int(bbox[5]), Int(bbox[6])
    
    # Dilate target by 5px
    target_dil = copy(target)
    for _ in 1:5
        tmp = copy(target_dil)
        
        # We reuse the 3D dilation, but we want 2D slices. 
        # Actually wait, we can just use our 2D slice kernel on each slice, or a 3D 6-connected kernel.
        # LayerWise propagation uses 2D dilation for target_m in Julia, but a 3D cross is fine.
        # Let's write a simple 2D dilation across all slices for the target.
        k_dil_all! = dilate_all_2d_kernel!(backend)
        k_dil_all!(tmp, target_dil, dims_x, dims_y, dims_z, ndrange=dims)
        KernelAbstractions.synchronize(backend)
        target_dil = tmp
    end
    
    step = direction == "inferior" ? -1 : 1
    curr_z = direction == "inferior" ? z_min : z_max
    
    slices_propagated = 0
    curr_slice_arr = KernelAbstractions.zeros(backend, UInt8, dims_x, dims_y)
    view_mask = view(mask, :, :, curr_z)
    copyto!(curr_slice_arr, view_mask)
    
    k_dil! = dilate_2d_slice_kernel!(backend)
    
    while (1 <= curr_z + step <= dims_z) && (slices_propagated < max_slices)
        curr_z += step
        slices_propagated += 1
        
        next_slice = copy(curr_slice_arr)
        if step_dilation_px > 0
            for _ in 1:step_dilation_px
                tmp = copy(next_slice)
                k_dil!(tmp, next_slice, dims_x, dims_y, ndrange=(dims_x, dims_y))
                KernelAbstractions.synchronize(backend)
                next_slice = tmp
            end
        end
        
        # Collision check
        target_slice = view(target_dil, :, :, curr_z)
        # Just use any(next_slice .& target_slice)
        collision = any(next_slice .> 0 .&& target_slice .> 0)
        
        view_mask_next = view(mask, :, :, curr_z)
        copyto!(view_mask_next, next_slice)
        
        if collision
            break
        end
        curr_slice_arr = next_slice
    end
end

@kernel function dilate_all_2d_kernel!(out_m, in_m, dims_x::Int32, dims_y::Int32, dims_z::Int32)
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        if in_m[i, j, k] > 0
            out_m[i, j, k] = UInt8(1)
            if i > 1; out_m[i-1, j, k] = UInt8(1); end
            if i < dims_x; out_m[i+1, j, k] = UInt8(1); end
            if j > 1; out_m[i, j-1, k] = UInt8(1); end
            if j < dims_y; out_m[i, j+1, k] = UInt8(1); end
        end
    end
end


@kernel function sum_coords_kernel!(sum_x, sum_y, count, mask, dims_x::Int32, dims_y::Int32, dims_z::Int32)
    k = @index(Global, Linear)
    if k <= dims_z
        sx = Int32(0)
        sy = Int32(0)
        cnt = Int32(0)
        for j in Int32(1):dims_y
            for i in Int32(1):dims_x
                if mask[i, j, k] > 0
                    sx += i
                    sy += j
                    cnt += Int32(1)
                end
            end
        end
        sum_x[k] = sx
        sum_y[k] = sy
        count[k] = cnt
    end
end

function get_centroid_per_slice_gpu(backend, m, dims)
    dims_z = Int32(dims[3])
    sum_x = KernelAbstractions.zeros(backend, Int32, dims_z)
    sum_y = KernelAbstractions.zeros(backend, Int32, dims_z)
    count = KernelAbstractions.zeros(backend, Int32, dims_z)
    
    k! = sum_coords_kernel!(backend)
    k!(sum_x, sum_y, count, m, Int32(dims[1]), Int32(dims[2]), dims_z, ndrange=dims_z)
    KernelAbstractions.synchronize(backend)
    
    return adapt(Array, sum_x), adapt(Array, sum_y), adapt(Array, count)
end






@kernel function extract_y_profile_kernel!(
    @Const(mask),
    y_min, y_max,
    dims_x::Int32, dims_y::Int32, dims_z::Int32
)
    i, k = @index(Global, NTuple)
    if i <= dims_x && k <= dims_z
        my_min = Int32(9999)
        my_max = Int32(-9999)
        for j in Int32(1):dims_y
            if mask[i, j, k] > 0
                my_min = min(my_min, j)
                my_max = max(my_max, j)
            end
        end
        y_min[i, k] = my_min
        y_max[i, k] = my_max
    end
end

@kernel function apply_slicewise_y_line_constraint_kernel!(
    mask::AbstractArray{UInt8, 3},
    @Const(line_m),
    @Const(line_b),
    offset_vox::Int32,
    is_posterior::Bool,
    dims_x::Int32, dims_y::Int32, dims_z::Int32
)
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        m = line_m[k]
        b = line_b[k]
        if m != -9999.0f0
            lim_y = m * Float32(i) + b
            # is_posterior means we want voxels POSTERIOR to the line
            # in LPS, posterior is larger Y
            # So if is_posterior, we want j > lim_y + offset
            # if !is_posterior (AnteriorTo), we want j < lim_y + offset
            # Actually, the python code: use_min_y = y_increasing_index ?
            
            # Let's check DagVm spacing direction:
            # y_increasing means larger Y is POSTERIOR.
            # AnteriorTo -> j < lim_y + offset_vox
            # PosteriorTo -> j > lim_y + offset_vox
            
            # Note: the input `is_posterior` maps to `c_type == "PosteriorTo"`
            # But wait, offset direction is tricky. Let's just do:
            if is_posterior
                if j < lim_y + Float32(offset_vox)
                    mask[i, j, k] = UInt8(0)
                end
            else
                if j > lim_y + Float32(offset_vox)
                    mask[i, j, k] = UInt8(0)
                end
            end
        end
    end
end

function execute_volumetric_boundary_2d_gpu(
    backend, dims, spacing,
    struct_arrs, medial_arrs, lateral_arrs, posterior_arrs, obstacle_arrs;
    posterior_offset_mm=5.0, growth_mode="convex_hull", keep_all_components=false
)
    out = KernelAbstractions.zeros(backend, UInt8, dims)
    
    if growth_mode == "sideways" && (!isempty(medial_arrs) || !isempty(lateral_arrs))
        hull_mask = execute_sideways_growth_gpu(backend, medial_arrs, lateral_arrs, obstacle_arrs, dims)
        if !keep_all_components
            hull_mask = get_largest_connected_component(hull_mask)
        end
        return hull_mask
    end
    
    # 1. Combine structures
    boundary_mask = KernelAbstractions.zeros(backend, UInt8, dims)
    for arr in struct_arrs
        if arr !== nothing; boundary_mask .|= arr; end
    end
    for arr in medial_arrs
        if arr !== nothing; boundary_mask .|= arr; end
    end
    for arr in lateral_arrs
        if arr !== nothing; boundary_mask .|= arr; end
    end
    
    if !any(boundary_mask .> UInt8(0))
        return out
    end
    
    # 2. Compute Convex Hull on GPU using our bridge logic
    hull_mask = execute_convex_hull_bridge(backend, boundary_mask, boundary_mask, dims, spacing, (0.0,0.0,0.0), (1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0))
    
    # 3. Posterior restriction
    post_mask = KernelAbstractions.zeros(backend, UInt8, dims)
    has_post = false
    for arr in posterior_arrs
        if arr !== nothing
            post_mask .|= arr
            has_post = true
        end
    end
    
    if has_post
        if posterior_offset_mm > 0.0
            dil_post = execute_anisotropic_expansion(backend, post_mask, dims, spacing, Dict("all" => Float64(posterior_offset_mm)))
            hull_mask .&= dil_post
        else
            hull_mask .&= post_mask
        end
    end
    
    # 4. Remove obstacles
    for arr in obstacle_arrs
        if arr !== nothing
            hull_mask .&= .~(arr .> UInt8(0))
        end
    end
    
    # 5. Extract largest connected component to act as flood fill
    if !keep_all_components
        lcc_mask = get_largest_connected_component(hull_mask)
        return lcc_mask
    else
        return hull_mask
    end
end


@kernel function compute_slice_x_bounds_kernel!(input, min_x, max_x, dims_x, dims_y, dims_z, is_left, is_right, mid_x)
    I, J, K = @index(Global, NTuple)
    if I <= dims_x && J <= dims_y && K <= dims_z
        if input[I, J, K] > 0
            if is_left
                KernelAbstractions.@atomic min_x[K] min I
            elseif is_right
                KernelAbstractions.@atomic max_x[K] max I
            end
        end
    end
end

function execute_subscapularis_band(backend::KernelAbstractions.Backend, subscap::AbstractArray{UInt8, 3}, is_left::Bool, shift_mm::Float64, spacing::NTuple{3, Float64}, dims::Tuple{Int, Int, Int})
    out_mask = KernelAbstractions.zeros(backend, UInt8, dims)
    shift_pixels = Int32(round(shift_mm / spacing[1]))
    
    if backend isa CPU
        subscap_cpu = subscap
        min_x_cpu = fill(Int32(999999), dims[3])
        max_x_cpu = fill(Int32(-1), dims[3])
        for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
            if subscap_cpu[i, j, k] > 0
                if i < min_x_cpu[k]; min_x_cpu[k] = i; end
                if i > max_x_cpu[k]; max_x_cpu[k] = i; end
            end
        end
    else
        # GPU path — use existing kernel (no 131 MB CPU transfer)
        min_x_gpu = KernelAbstractions.allocate(backend, Int32, dims[3])
        max_x_gpu = KernelAbstractions.allocate(backend, Int32, dims[3])
        fill!(min_x_gpu, Int32(999999))
        fill!(max_x_gpu, Int32(-1))
        mid_x = Int32(dims[1] ÷ 2)
        k_bounds! = compute_slice_x_bounds_kernel!(backend)
        k_bounds!(subscap, min_x_gpu, max_x_gpu, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), is_left, !is_left, mid_x, ndrange=dims)
        KernelAbstractions.synchronize(backend)
        # Transfer only the 1D bounds (~2 KB) to CPU for the sequential fill step
        min_x_cpu = Array(min_x_gpu)
        max_x_cpu = Array(max_x_gpu)
    end
    
    # Fill missing slices using lateral edge
    lat_arr = fill(Int32(-1), dims[3])
    last_valid = -1
    for k in dims[3]:-1:1
        val = is_left ? max_x_cpu[k] : min_x_cpu[k]
        if (is_left && val > 0) || (!is_left && val < 999999)
            last_valid = val
        end
        lat_arr[k] = last_valid
    end
    
    # Forward fill if top is missing
    last_valid = -1
    for k in 1:dims[3]
        if lat_arr[k] == -1 && last_valid != -1
            lat_arr[k] = last_valid
        elseif lat_arr[k] != -1
            last_valid = lat_arr[k]
        end
    end
    
    lat_gpu = adapt(backend, lat_arr)
    
    @kernel function apply_subscap_band!(out, lat_arr_gpu, dims_x, dims_y, dims_z, is_left)
        i, j, k = @index(Global, NTuple)
        if k <= dims_z
            l_x = lat_arr_gpu[k]
            if l_x != -1
                if is_left
                    if i > l_x
                        out[i, j, k] = UInt8(1)
                    end
                else
                    if i < l_x
                        out[i, j, k] = UInt8(1)
                    end
                end
            end
        end
    end
    
    kernel! = apply_subscap_band!(backend)
    kernel!(out_mask, lat_gpu, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), is_left, ndrange=dims)
    KernelAbstractions.synchronize(backend)
    
    return out_mask
end

@kernel function aorta_medial_bounds_kernel!(aorta, min_x, max_x, dims_x, dims_y, dims_z)
    I, J, K = @index(Global, NTuple)
    if I <= dims_x && J <= dims_y && K <= dims_z
        if aorta[I, J, K] > 0
            KernelAbstractions.@atomic min_x[K] min I
            KernelAbstractions.@atomic max_x[K] max I
        end
    end
end

@kernel function apply_aorta_medial_plane!(out, min_x, max_x, dims_x, dims_y, dims_z)
    i, j, k = @index(Global, NTuple)
    if k <= dims_z
        mn = min_x[k]
        mx = max_x[k]
        if mn < 999999 && mx > 0
            cx = (mn + mx) ÷ 2
            if i == cx
                out[i, j, k] = UInt8(1)
            end
        end
    end
end

function execute_aorta_medial_plane(backend::KernelAbstractions.Backend, aorta::AbstractArray{UInt8, 3}, dims::Tuple{Int, Int, Int})
    out_mask = KernelAbstractions.zeros(backend, UInt8, dims)
    
    if backend isa CPU
        # CPU path (unchanged)
        for k in 1:dims[3]
            min_x = 999999
            max_x = -1
            for j in 1:dims[2], i in 1:dims[1]
                if aorta[i, j, k] > 0
                    if i < min_x; min_x = i; end
                    if i > max_x; max_x = i; end
                end
            end
            if min_x < 999999 && max_x > 0
                cx = (min_x + max_x) ÷ 2
                for j in 1:dims[2]
                    out_mask[cx, j, k] = UInt8(1)
                end
            end
        end
    else
        # GPU path — use existing kernels (no CPU transfer)
        min_x_gpu = KernelAbstractions.allocate(backend, Int32, dims[3])
        max_x_gpu = KernelAbstractions.zeros(backend, Int32, dims[3])
        fill!(min_x_gpu, Int32(999999))
        
        k_bounds! = aorta_medial_bounds_kernel!(backend)
        k_bounds!(aorta, min_x_gpu, max_x_gpu, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=dims)
        KernelAbstractions.synchronize(backend)
        
        k_apply! = apply_aorta_medial_plane!(backend)
        k_apply!(out_mask, min_x_gpu, max_x_gpu, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=dims)
        KernelAbstractions.synchronize(backend)
    end
    
    return out_mask
end

end # module