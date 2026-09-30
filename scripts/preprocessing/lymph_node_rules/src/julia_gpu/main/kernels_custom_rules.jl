module CustomRules

using CUDA
using KernelAbstractions
using KernelAbstractions: @index, @synchronize
using Adapt
using ..RuleExecutors

export ExtractMainBronchi, InguinalAnteriorProxy, PancreasSplitHelper, PyloricSplitHelper, StomachLongAxisHelper, HelperSplenicPDCustom, HilarAnteriorHelper, InternalIliacCustom, AxillaryRTOGRelaxed, AnteriorGrowthMask, PosteriorGrowthMask, VolumetricBoundary2DLateralGrowth, LimitZByLandmark, limit_z_by_landmark_kernel!, AnteriorExtrusion, anterior_growth_kernel!, PropagateZ, propagate_z_kernel!, Station1LowCervical, IliacBifurcationCustom, AxillaryRTOG, ExternalIliacCustom, CommonIliacCustom, AxillaryHelperB, PleuralSpaceCustom, PresacralAnteriorCustom, split_mask_kernel!, cylinder_primitive_kernel!, VolumetricBoundary2D, compute_2d_convex_hull_per_slice!, ErosionHelper

# =====================================================================
# 0. SplitMask / SplitConnectedComponents (Pure GPU)
# =====================================================================
@kernel function split_mask_kernel!(output, input, dims, axis_idx::Int, split_val::Int, keep_min::Bool)
    I, J, K = @index(Global, NTuple)
    if I <= dims[1] && J <= dims[2] && K <= dims[3]
        if input[I, J, K] > 0
            coord = axis_idx == 1 ? I : (axis_idx == 2 ? J : K)
            if (keep_min && coord <= split_val) || (!keep_min && coord > split_val)
                output[I, J, K] = UInt8(1)
            else
                output[I, J, K] = UInt8(0)
            end
        else
            output[I, J, K] = UInt8(0)
        end
    end
end

@kernel function cylinder_primitive_kernel!(output, p1_x, p1_y, p1_z, p2_x, p2_y, p2_z, radius_mm, orig_x, orig_y, orig_z, sp_x, sp_y, sp_z)
    I, J, K = @index(Global, NTuple)
    dims = size(output)
    if I <= dims[1] && J <= dims[2] && K <= dims[3]
        phys_x = Float32(orig_x) + (Float32(I) - 1.0f0) * Float32(sp_x)
        phys_y = Float32(orig_y) + (Float32(J) - 1.0f0) * Float32(sp_y)
        phys_z = Float32(orig_z) + (Float32(K) - 1.0f0) * Float32(sp_z)
        
        dx = Float32(p2_x) - Float32(p1_x)
        dy = Float32(p2_y) - Float32(p1_y)
        dz = Float32(p2_z) - Float32(p1_z)
        len = sqrt(dx*dx + dy*dy + dz*dz)
        
        if len > 0.0f0
            ux = dx / len
            uy = dy / len
            uz = dz / len
            
            vx = phys_x - Float32(p1_x)
            vy = phys_y - Float32(p1_y)
            vz = phys_z - Float32(p1_z)
            
            proj = vx*ux + vy*uy + vz*uz
            if proj >= 0.0f0 && proj <= len
                dist2 = (vx*vx + vy*vy + vz*vz) - proj*proj
                r = Float32(radius_mm)
                if dist2 <= r * r
                    output[I, J, K] = UInt8(1)
                end
            end
        end
    end
end

# =====================================================================
# 1. HilarAnteriorHelper (Pure GPU)
# =====================================================================
@kernel function hilar_anterior_helper_kernel!(output, input, dims, voxel_offset_y, y_is_posterior)
    I, K = @index(Global, NTuple)
    if I <= dims[1] && K <= dims[3]
        if y_is_posterior
            y_front = 0
            for j in 1:dims[2]
                if input[I, j, K] > 0
                    y_front = j
                    break
                end
            end
            if y_front > 0
                y_start = max(1, y_front - voxel_offset_y)
                for j in y_start:y_front
                    output[I, j, K] = 1
                end
            end
        else
            y_front = 0
            for j in dims[2]:-1:1
                if input[I, j, K] > 0
                    y_front = j
                    break
                end
            end
            if y_front > 0
                y_end = min(dims[2], y_front + voxel_offset_y)
                for j in y_front:y_end
                    output[I, j, K] = 1
                end
            end
        end
    end
end

function HilarAnteriorHelper(tm, out_device::AbstractArray{UInt8, 3}, params::Dict, in_device::AbstractArray{UInt8, 3}, args...)
    dims = size(out_device)
    offset_mm = Float32(get(params, "offset_mm", 40.0))
    spacing_y = Float32(get(params, "spacing_y", 1.0))
    y_is_posterior = get(params, "y_is_posterior", true)
    voxel_offset_y = Int(round(offset_mm / spacing_y))
    
    fill!(out_device, 0)
    kernel! = hilar_anterior_helper_kernel!(tm.backend)
    kernel!(out_device, in_device, dims, voxel_offset_y, y_is_posterior, ndrange=(dims[1], dims[3]))
    KernelAbstractions.synchronize(tm.backend)
end

# =====================================================================
# 2. InternalIliacCustom (Pure GPU)
# =====================================================================
@kernel function internal_iliac_split_kernel!(output, input, dims, mid_limits, is_left)
    I, J, K = @index(Global, NTuple)
    if I <= dims[1] && J <= dims[2] && K <= dims[3]
        if input[I, J, K] > 0
            mid_x = mid_limits[K]
            if mid_x <= Int32(0)
                output[I, J, K] = input[I, J, K]
            else
                if (is_left && I >= mid_x) || (!is_left && I <= mid_x)
                    output[I, J, K] = input[I, J, K]
                else
                    output[I, J, K] = 0
                end
            end
        else
            output[I, J, K] = 0
        end
    end
end

function InternalIliacCustom(tm, out_device::AbstractArray{UInt8, 3}, params::Dict, prim_device::AbstractArray{UInt8, 3}, art_l::AbstractArray{UInt8, 3}, art_r::AbstractArray{UInt8, 3}, args...)
    side = get(params, "side", "Left")
    is_left = lowercase(side) == "left"
    dims = size(prim_device)
    
    mid_limits = KernelAbstractions.zeros(tm.backend, Int32, dims[3])
    k1! = compute_iliac_midline_kernel!(tm.backend)
    k1!(mid_limits, art_l, art_r, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=dims[3])
    KernelAbstractions.synchronize(tm.backend)
    
    k2! = internal_iliac_split_kernel!(tm.backend)
    k2!(out_device, prim_device, dims, mid_limits, is_left, ndrange=dims)
    KernelAbstractions.synchronize(tm.backend)
end

# =====================================================================
# 3. ExternalIliacCustom & IliacBifurcationCustom (Pure GPU)
# =====================================================================
@kernel function iliac_split_kernel!(output, input, dims, z_min, z_max, mid_x, is_left)
    I, J, K = @index(Global, NTuple)
    if I <= dims[1] && J <= dims[2] && K <= dims[3]
        if K >= z_min && K <= z_max && input[I, J, K] > 0
            if (is_left && I >= mid_x) || (!is_left && I <= mid_x)
                output[I, J, K] = 1
            else
                output[I, J, K] = 0
            end
        else
            output[I, J, K] = 0
        end
    end
end

@kernel function iliac_split_slicewise_kernel!(output, input, dims, z_min, z_max, mid_limits, is_left)
    I, J, K = @index(Global, NTuple)
    if I <= dims[1] && J <= dims[2] && K <= dims[3]
        if K >= z_min && K <= z_max && input[I, J, K] > 0
            mid_x = mid_limits[K]
            if mid_x <= Int32(0)
                mid_x = Int32(dims[1] ÷ 2)
            end
            if (is_left && I >= mid_x) || (!is_left && I <= mid_x)
                output[I, J, K] = 1
            else
                output[I, J, K] = 0
            end
        else
            output[I, J, K] = 0
        end
    end
end

@kernel function compute_iliac_midline_kernel!(
    mid_limits,
    @Const(art_l), @Const(art_r),
    dims_x::Int32, dims_y::Int32, dims_z::Int32
)
    k = @index(Global)
    if k <= dims_z
        sum_xl = Int64(0); count_l = Int64(0)
        sum_xr = Int64(0); count_r = Int64(0)
        for j in 1:dims_y, i in 1:dims_x
            if art_l[i, j, k] > UInt8(0)
                sum_xl += i
                count_l += 1
            end
            if art_r[i, j, k] > UInt8(0)
                sum_xr += i
                count_r += 1
            end
        end
        if count_l > 0 && count_r > 0
            mean_xl = Float64(sum_xl) / Float64(count_l)
            mean_xr = Float64(sum_xr) / Float64(count_r)
            mid_limits[k] = Int32(round((mean_xl + mean_xr) / 2.0))
        else
            mid_limits[k] = Int32(0)
        end
    end
end

function ExternalIliacCustom(tm, out_device::AbstractArray{UInt8, 3}, params::Dict, computed_landmarks::Dict, prim_device::AbstractArray{UInt8, 3}, femur_device::AbstractArray{UInt8, 3}, art_l_device::AbstractArray{UInt8, 3}, art_r_device::AbstractArray{UInt8, 3}, ven_l_device::AbstractArray{UInt8, 3}, ven_r_device::AbstractArray{UInt8, 3}, args...)
    side = get(params, "side", "Left")
    is_left = lowercase(side) == "left"
    padding_mm = Float32(get(params, "padding_mm", 7.0))
    sp_x = Float32(get(params, "spacing_x", 1.0))
    sp_y = Float32(get(params, "spacing_y", 1.0))
    sp_z = Float32(get(params, "spacing_z", 1.0))
    orig_x = Float32(get(params, "origin_x", 0.0))
    orig_y = Float32(get(params, "origin_y", 0.0))
    orig_z = Float32(get(params, "origin_z", 0.0))
    dims = size(out_device)
    
    # 1. Base vessel union on GPU (same side artery & vein)
    vessel_union = KernelAbstractions.zeros(tm.backend, UInt8, dims...)
    art_dev = is_left ? art_l_device : art_r_device
    ven_dev = is_left ? ven_l_device : ven_r_device
    vessel_union .= art_dev .| ven_dev
    
    # 2. Anisotropic dilation on GPU (7mm)
    dil_arr = KernelAbstractions.zeros(tm.backend, UInt8, dims...)
    dil_arr .= execute_anisotropic_expansion(tm.backend, vessel_union, size(vessel_union), (Float64(sp_x), Float64(sp_y), Float64(sp_z)), Dict("all" => padding_mm))
    
    # 3. Z-plane limits from computed landmarks
    p1_key = "internal_iliac_p1_" * lowercase(side)
    if !haskey(computed_landmarks, p1_key)
        println("    [INFO] Missing $p1_key for ExternalIliacCustom; skipping area calculation.")
        fill!(out_device, 0)
        return
    end
    p1 = computed_landmarks[p1_key]
    z_bif = round(Int, (Float64(p1[3]) - Float64(orig_z)) / Float64(sp_z)) + 1
    
    femur_bbox = RuleExecutors.gpu_bounding_box(tm.backend, femur_device)
    if femur_bbox === nothing
        println("    [INFO] Missing femur bounding box for ExternalIliacCustom; skipping area calculation.")
        fill!(out_device, 0)
        return
    end
    z_femur = femur_bbox[6]
    
    z_min = min(z_bif, z_femur)
    z_max = max(z_bif, z_femur)
    
    # Dynamic midline split per slice on GPU
    mid_limits_gpu = KernelAbstractions.zeros(tm.backend, Int32, dims[3])
    k_mid! = compute_iliac_midline_kernel!(tm.backend)
    k_mid!(mid_limits_gpu, art_l_device, art_r_device, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=dims[3])
    KernelAbstractions.synchronize(tm.backend)
    
    kernel! = iliac_split_slicewise_kernel!(tm.backend)
    kernel!(out_device, dil_arr, dims, z_min, z_max, mid_limits_gpu, is_left, ndrange=dims)
    KernelAbstractions.synchronize(tm.backend)
end

function CommonIliacCustom(tm, out_device::AbstractArray{UInt8, 3}, params::Dict, computed_landmarks::Dict, aorta_device::AbstractArray{UInt8, 3}, art_l_device::AbstractArray{UInt8, 3}, art_r_device::AbstractArray{UInt8, 3}, ven_l_device::AbstractArray{UInt8, 3}, ven_r_device::AbstractArray{UInt8, 3}, args...)
    side = get(params, "side", "Left")
    is_left = lowercase(side) == "left"
    padding_mm = Float32(get(params, "padding_mm", 7.0))
    sp_x = Float32(get(params, "spacing_x", 1.0))
    sp_y = Float32(get(params, "spacing_y", 1.0))
    sp_z = Float32(get(params, "spacing_z", 1.0))
    orig_x = Float32(get(params, "origin_x", 0.0))
    orig_y = Float32(get(params, "origin_y", 0.0))
    orig_z = Float32(get(params, "origin_z", 0.0))
    dims = size(out_device)
    
    # 1. Base vessel union on GPU (same side artery & vein)
    vessel_union = KernelAbstractions.zeros(tm.backend, UInt8, dims...)
    art_dev = is_left ? art_l_device : art_r_device
    ven_dev = is_left ? ven_l_device : ven_r_device
    vessel_union .= art_dev .| ven_dev
    
    # 2. Anisotropic dilation on GPU (7mm)
    dil_arr = KernelAbstractions.zeros(tm.backend, UInt8, dims...)
    dil_arr .= execute_anisotropic_expansion(tm.backend, vessel_union, size(vessel_union), (Float64(sp_x), Float64(sp_y), Float64(sp_z)), Dict("all" => padding_mm))
    
    # 3. Z-plane limits: use actual artery extent + 10mm superior margin
    art_bbox = RuleExecutors.gpu_bounding_box(tm.backend, art_dev)
    if art_bbox === nothing
        println("    [INFO] Missing artery bounding box for CommonIliacCustom; skipping area calculation.")
        fill!(out_device, 0)
        return
    end
    # Use the full z-extent of the iliac artery, plus 10mm margin superiorly
    z_margin = round(Int, 10.0 / sp_z)
    z_min = art_bbox[5]
    z_max = min(dims[3], art_bbox[6] + z_margin)
    
    # Dynamic midline split per slice on GPU
    mid_limits_gpu = KernelAbstractions.zeros(tm.backend, Int32, dims[3])
    k_mid! = compute_iliac_midline_kernel!(tm.backend)
    k_mid!(mid_limits_gpu, art_l_device, art_r_device, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=dims[3])
    KernelAbstractions.synchronize(tm.backend)
    
    kernel! = iliac_split_slicewise_kernel!(tm.backend)
    kernel!(out_device, dil_arr, dims, z_min, z_max, mid_limits_gpu, is_left, ndrange=dims)
    KernelAbstractions.synchronize(tm.backend)
end

function IliacBifurcationCustom(tm, out_device::AbstractArray{UInt8, 3}, params::Dict, computed_landmarks::Dict, art_l_device::AbstractArray{UInt8, 3}, art_r_device::AbstractArray{UInt8, 3}, ven_l_device::AbstractArray{UInt8, 3}, ven_r_device::AbstractArray{UInt8, 3}, args...)
    side = get(params, "side", "Left")
    is_left = lowercase(side) == "left"
    padding_mm = Float32(get(params, "padding_mm", 7.0))
    sp_x = Float32(get(params, "spacing_x", 1.0))
    sp_y = Float32(get(params, "spacing_y", 1.0))
    sp_z = Float32(get(params, "spacing_z", 1.0))
    orig_x = Float32(get(params, "origin_x", 0.0))
    orig_y = Float32(get(params, "origin_y", 0.0))
    orig_z = Float32(get(params, "origin_z", 0.0))
    dims = size(out_device)
    
    # 1. Base vessel union on GPU
    vessel_union = KernelAbstractions.zeros(tm.backend, UInt8, dims...)
    art_dev = is_left ? art_l_device : art_r_device
    ven_dev = is_left ? ven_l_device : ven_r_device
    vessel_union .= art_dev .| ven_dev
    
    # 2. Anisotropic dilation on GPU (7mm)
    dil_arr = KernelAbstractions.zeros(tm.backend, UInt8, dims...)
    dil_arr .= execute_anisotropic_expansion(tm.backend, vessel_union, size(vessel_union), (Float64(sp_x), Float64(sp_y), Float64(sp_z)), Dict("all" => padding_mm))
    
    # 3. Z-plane bounds (+/- 10mm around bifurcation)
    p1_key = "internal_iliac_p1_" * lowercase(side)
    if !haskey(computed_landmarks, p1_key)
        println("    [INFO] Missing $p1_key for IliacBifurcationCustom; skipping area calculation.")
        fill!(out_device, 0)
        return
    end
    p1 = computed_landmarks[p1_key]
    z_bif = round(Int, (Float64(p1[3]) - Float64(orig_z)) / Float64(sp_z)) + 1
    
    z_offset_idx = round(Int, 10.0 / sp_z)
    z_min = max(1, z_bif - z_offset_idx)
    z_max = min(dims[3], z_bif + z_offset_idx)
    
    # Dynamic midline split per slice on GPU
    mid_limits_gpu = KernelAbstractions.zeros(tm.backend, Int32, dims[3])
    k_mid! = compute_iliac_midline_kernel!(tm.backend)
    k_mid!(mid_limits_gpu, art_l_device, art_r_device, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=dims[3])
    KernelAbstractions.synchronize(tm.backend)
    
    kernel! = iliac_split_slicewise_kernel!(tm.backend)
    kernel!(out_device, dil_arr, dims, z_min, z_max, mid_limits_gpu, is_left, ndrange=dims)
    KernelAbstractions.synchronize(tm.backend)
end

# =====================================================================
# 4. AxillaryRTOG / AxillaryRTOGRelaxed / AxillaryHelperB (Pure GPU)
# =====================================================================
@kernel function axillary_rtog_plane_kernel!(output, flood, dims, orig_x, orig_y, orig_z, sp_x, sp_y, sp_z, 
    coracoid_x, coracoid_y, coracoid_z, 
    rib5_x, rib5_y, rib5_z,
    rib3_x, rib3_y, rib3_z,
    pm_nx, pm_ny, pm_nz,
    is_left, level, z_min_phys, z_max_phys)
    
    I, J, K = @index(Global, NTuple)
    if I <= dims[1] && J <= dims[2] && K <= dims[3]
        if flood[I, J, K] > 0
            px = Float32(orig_x) + (Float32(I) - 1.0f0) * Float32(sp_x)
            py = Float32(orig_y) + (Float32(J) - 1.0f0) * Float32(sp_y)
            pz = Float32(orig_z) + (Float32(K) - 1.0f0) * Float32(sp_z)
            
            if pz >= min(Float32(z_min_phys), Float32(z_max_phys)) && pz <= max(Float32(z_min_phys), Float32(z_max_phys))
                # Lateral boundary line from coracoid to rib 5
                dz_lat = Float32(coracoid_z - rib5_z)
                if abs(dz_lat) < 1f-6; dz_lat = 1f-6; end
                t_lat = (pz - Float32(rib5_z)) / dz_lat
                x_lat = Float32(rib5_x) + t_lat * Float32(coracoid_x - rib5_x)
                
                # Medial boundary line from coracoid to rib 3
                dz_med = Float32(coracoid_z - rib3_z)
                if abs(dz_med) < 1f-6; dz_med = 1f-6; end
                t_med = (pz - Float32(rib3_z)) / dz_med
                x_med = Float32(rib3_x) + t_med * Float32(coracoid_x - rib3_x)
                
                # Signed distance to PM plane
                plane_dist = Float32(pm_nx) * (px - Float32(coracoid_x)) + 
                             Float32(pm_ny) * (py - Float32(coracoid_y)) + 
                             Float32(pm_nz) * (pz - Float32(coracoid_z))
                
                post_to_pm = plane_dist >= -10.0f0
                ant_to_pm = plane_dist < 10.0f0
                
                valid = false
                if is_left
                    if level == 1
                        valid = (px >= x_lat) && post_to_pm
                    elseif level == 2
                        valid = (px < x_lat) && (px >= x_med) && post_to_pm
                    elseif level == 3
                        valid = (px < x_med) && (px >= x_med - 35.0f0) && (px >= 15.0f0) && post_to_pm
                    elseif level == 4  # Rotter
                        valid = (px < x_lat) && (px >= x_med) && ant_to_pm
                    end
                else
                    if level == 1
                        valid = (px <= x_lat) && post_to_pm
                    elseif level == 2
                        valid = (px > x_lat) && (px <= x_med) && post_to_pm
                    elseif level == 3
                        valid = (px > x_med) && (px <= x_med + 35.0f0) && (px <= -15.0f0) && post_to_pm
                    elseif level == 4  # Rotter
                        valid = (px > x_lat) && (px <= x_med) && ant_to_pm
                    end
                end
                
                output[I, J, K] = valid ? UInt8(1) : UInt8(0)
            else
                output[I, J, K] = UInt8(0)
            end
        else
            output[I, J, K] = UInt8(0)
        end
    end
end


function AxillaryRTOG(tm, out_device::AbstractArray{UInt8, 3}, params::Dict, bounds_device::AbstractArray{UInt8, 3}, computed_landmarks::Dict, pec_view::AbstractArray{UInt8, 3}, pm_view::AbstractArray{UInt8, 3}, sub_view::AbstractArray{UInt8, 3}, art_view::AbstractArray{UInt8, 3}, chestwall_view::AbstractArray{UInt8, 3})
    side = get(params, "side", "left")
    side_norm = lowercase(side)
    is_left = side_norm == "left"
    level_str = lowercase(get(params, "level", "i"))
    
    level = 1
    if level_str == "ii"; level = 2;
    elseif level_str == "iii"; level = 3;
    elseif level_str == "rotter"; level = 4;
    end
    
    geom_key = "axillary_geometry_$side_norm"
    if !haskey(computed_landmarks, geom_key)
        println("    [WARNING] Axillary geometry missing for $side_norm. Outputting empty.")
        fill!(out_device, 0)
        return
    end
    geom = computed_landmarks[geom_key]
    
    sp_x = Float32(get(params, "spacing_x", 1.0))
    sp_y = Float32(get(params, "spacing_y", 1.0))
    sp_z = Float32(get(params, "spacing_z", 1.0))
    orig_x = Float32(get(params, "origin_x", 0.0))
    orig_y = Float32(get(params, "origin_y", 0.0))
    orig_z = Float32(get(params, "origin_z", 0.0))
    dims = size(out_device)
    
    coracoid_x = Float64(geom["coracoid_x"])
    coracoid_y = Float64(geom["coracoid_y"])
    coracoid_z = Float64(geom["coracoid_z"])
    rib5_x = Float64(geom["rib5_x"])
    rib5_y = Float64(geom["rib5_y"])
    rib5_z = Float64(geom["rib5_z"])
    rib3_x = Float64(geom["rib3_x"])
    rib3_y = Float64(geom["rib3_y"])
    rib3_z = Float64(geom["rib3_z"])
    pm_nx = Float64(geom["pm_nx"])
    pm_ny = Float64(geom["pm_ny"])
    pm_nz = Float64(geom["pm_nz"])
    
    # Get Z bounds of pectoralis minor (pm_view) for levels 1 and 2
    pm_min_z = Float64(rib5_z)
    pm_max_z = Float64(coracoid_z) # fallback
    if length(pm_view) > 0
        bbox = RuleExecutors.gpu_bounding_box(tm.backend, pm_view)
        if bbox[6] > 0
            pm_min_z = orig_z + (bbox[5] - 1.0) * sp_z
            pm_max_z = orig_z + (bbox[6] - 1.0) * sp_z
        end
    end
    
    # Inferior and Superior limit per level
    if level == 1 || level == 2
        # Use precise pec minor Z bounds for Axillary I and II
    art_min_z = pm_max_z
    if length(art_view) > 0
        bbox_art = RuleExecutors.gpu_bounding_box(tm.backend, art_view)
        if bbox_art[6] > 0
            art_min_z = orig_z + (bbox_art[5] - 1.0) * sp_z
        end
    end

        z_min_phys = Float32(pm_min_z)
        z_max_phys = Float32(min(pm_max_z, art_min_z)) # Cap at the subclavian artery!
    elseif level == 3
        z_min_phys = Float32(coracoid_z - 55.0)
        z_max_phys = Float32(coracoid_z)
    else # Rotter
        z_min_phys = Float32(coracoid_z - 65.0)
        z_max_phys = Float32(pm_max_z)
    end
    
    flood = KernelAbstractions.zeros(tm.backend, UInt8, dims...)
    
    if level == 1
        # Level 1: fill area between anteriorly pec major, posteriorly subscapularis
        AxillaryHelperB(tm, flood, params, pec_view, sub_view)
    elseif level == 2
        # Level 2: fill area between anterior chest wall (chestwall_view) and pectoralis minor (pm_view)
        AxillaryHelperB(tm, flood, params, chestwall_view, pm_view)
    elseif level == 3
        art_sum = (art_view isa CuArray) ? Int(CUDA.sum(art_view .> 0)) : count(art_view .> 0)
        if art_sum > 0
            flood .= execute_anisotropic_expansion(tm.backend, art_view, size(art_view), (Float64(sp_x), Float64(sp_y), Float64(sp_z)), Dict("all" => 25.0))
        else
            flood .= execute_anisotropic_expansion(tm.backend, pec_view, size(pec_view), (Float64(sp_x), Float64(sp_y), Float64(sp_z)), Dict("all" => 30.0))
        end
    else # Rotter
        flood .= execute_anisotropic_expansion(tm.backend, pec_view, size(pec_view), (Float64(sp_x), Float64(sp_y), Float64(sp_z)), Dict("all" => 12.0))
    end
    
    kernel! = axillary_rtog_plane_kernel!(tm.backend)
    kernel!(out_device, flood, dims, orig_x, orig_y, orig_z, sp_x, sp_y, sp_z,
            Float32(coracoid_x), Float32(coracoid_y), Float32(coracoid_z),
            Float32(rib5_x), Float32(rib5_y), Float32(rib5_z),
            Float32(rib3_x), Float32(rib3_y), Float32(rib3_z),
            Float32(pm_nx), Float32(pm_ny), Float32(pm_nz),
            is_left, level, z_min_phys, z_max_phys, ndrange=dims)
    KernelAbstractions.synchronize(tm.backend)
end

function AxillaryRTOGRelaxed(tm, out_device::AbstractArray{UInt8, 3}, params::Dict, bounds_device::AbstractArray{UInt8, 3}, computed_landmarks::Dict, pec_view::AbstractArray{UInt8, 3}, pm_view::AbstractArray{UInt8, 3}, sub_view::AbstractArray{UInt8, 3}, art_view::AbstractArray{UInt8, 3}, chestwall_view::AbstractArray{UInt8, 3})
    AxillaryRTOG(tm, out_device, params, bounds_device, computed_landmarks, pec_view, pm_view, sub_view, art_view, chestwall_view)
end

@kernel function axillary_rectangle_kernel!(output, pec, sub, dims, is_left, medial_vox, lateral_vox)
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
            end
        end
        
        if pec_count > 0 && sub_count > 0 && sub_max_y > pec_min_y
            # We want to fill from the posterior of PEC to the posterior of SUB
            y_start = pec_max_y
            y_end = sub_max_y
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

function AxillaryHelperB(tm, out_device::AbstractArray{UInt8, 3}, params::Dict, pec_device::AbstractArray{UInt8, 3}, sub_device::AbstractArray{UInt8, 3}, args...)
    dims = size(out_device)
    side = lowercase(get(params, "side", "left"))
    is_left = side == "left"
    sp_x = Float32(get(params, "spacing_x", 1.0))
    medial_vox = Int32(round(40.0 / sp_x))
    lateral_vox = Int32(round(100.0 / sp_x))
    
    fill!(out_device, 0)
    kernel! = axillary_rectangle_kernel!(tm.backend)
    kernel!(out_device, pec_device, sub_device, Int32.(dims), is_left, medial_vox, lateral_vox, ndrange=dims[3])
    KernelAbstractions.synchronize(tm.backend)
end


# =====================================================================
# 5. AnteriorGrowthMask & PosteriorGrowthMask (Pure GPU)
# =====================================================================
@kernel function anterior_growth_kernel!(output, input, dims, voxel_dist)
    I, K = @index(Global, NTuple)
    if I <= dims[1] && K <= dims[3]
        for j in 1:dims[2]
            if input[I, j, K] > 0
                y_start = max(1, j - voxel_dist)
                for y in y_start:j
                    output[I, y, K] = 1
                end
            end
        end
    end
end

function AnteriorGrowthMask(tm, out_device::AbstractArray{UInt8, 3}, params::Dict, in_device::AbstractArray{UInt8, 3}, args...)
    dims = size(out_device)
    dist = Float32(get(params, "distance", get(params, "growth_mm", get(params, "distance_mm", 10.0))))
    sp_y = Float32(get(params, "spacing_y", 1.0))
    voxel_dist = Int(round(dist / sp_y))
    
    fill!(out_device, 0)
    kernel! = anterior_growth_kernel!(tm.backend)
    kernel!(out_device, in_device, dims, voxel_dist, ndrange=(dims[1], dims[3]))
    KernelAbstractions.synchronize(tm.backend)
end

@kernel function posterior_growth_kernel!(output, input, dims, voxel_dist)
    I, K = @index(Global, NTuple)
    if I <= dims[1] && K <= dims[3]
        for j in 1:dims[2]
            if input[I, j, K] > 0
                y_end = min(dims[2], j + voxel_dist)
                for y in j:y_end
                    output[I, y, K] = 1
                end
            end
        end
    end
end

function PosteriorGrowthMask(tm, out_device::AbstractArray{UInt8, 3}, params::Dict, in_device::AbstractArray{UInt8, 3}, args...)
    dims = size(out_device)
    dist = Float32(get(params, "distance", get(params, "growth_mm", get(params, "distance_mm", 10.0))))
    sp_y = Float32(get(params, "spacing_y", 1.0))
    voxel_dist = Int(round(dist / sp_y))
    
    fill!(out_device, 0)
    kernel! = posterior_growth_kernel!(tm.backend)
    kernel!(out_device, in_device, dims, voxel_dist, ndrange=(dims[1], dims[3]))
    KernelAbstractions.synchronize(tm.backend)
end

# =====================================================================
# 6. VolumetricBoundary2DLateralGrowth (Pure GPU - Fully Parameterized)
# =====================================================================
@kernel function extract_volumetric_growth_params_kernel!(
    start_coord_map, obs_min_growth_map, obs_max_growth_map,
    obs_min_ortho_map, obs_max_ortho_map,
    limit_ortho_map, split_ortho_map,
    @Const(start_m), @Const(obs_m), @Const(limit_m), @Const(split_m),
    has_limit::Bool, has_split::Bool,
    axis_growth::Int32, step_growth::Int32,
    dims_x::Int32, dims_y::Int32, dims_z::Int32
)
    k = @index(Global)
    if k <= dims_z
        # 1. Start coordinate along axis_growth
        min_start = Int32(dims_x + dims_y + 1)
        max_start = Int32(0)
        for j in 1:dims_y, i in 1:dims_x
            if start_m[i, j, k] > UInt8(0)
                g_coord = (axis_growth == Int32(1)) ? i : j
                if g_coord < min_start; min_start = g_coord; end
                if g_coord > max_start; max_start = g_coord; end
            end
        end
        if max_start >= min_start
            start_coord_map[k] = (step_growth > Int32(0)) ? min_start : max_start
        else
            start_coord_map[k] = Int32(0)
        end

        # 2. Obstacle bounds along axis_growth and axis_ortho
        min_obs_g = Int32(dims_x + dims_y + 1); max_obs_g = Int32(0)
        min_obs_o = Int32(dims_x + dims_y + 1); max_obs_o = Int32(0)
        for j in 1:dims_y, i in 1:dims_x
            if obs_m[i, j, k] > UInt8(0)
                g_coord = (axis_growth == Int32(1)) ? i : j
                o_coord = (axis_growth == Int32(1)) ? j : i
                if g_coord < min_obs_g; min_obs_g = g_coord; end
                if g_coord > max_obs_g; max_obs_g = g_coord; end
                if o_coord < min_obs_o; min_obs_o = o_coord; end
                if o_coord > max_obs_o; max_obs_o = o_coord; end
            end
        end
        if max_obs_g >= min_obs_g
            obs_min_growth_map[k] = min_obs_g
            obs_max_growth_map[k] = max_obs_g
            obs_min_ortho_map[k] = min_obs_o
            obs_max_ortho_map[k] = max_obs_o
        else
            obs_min_growth_map[k] = Int32(0)
            obs_max_growth_map[k] = Int32(0)
            obs_min_ortho_map[k] = Int32(0)
            obs_max_ortho_map[k] = Int32(0)
        end

        # 3. Limit coordinate along axis_ortho
        if has_limit
            max_lim_o = Int32(0)
            for j in 1:dims_y, i in 1:dims_x
                if limit_m[i, j, k] > UInt8(0)
                    o_coord = (axis_growth == Int32(1)) ? j : i
                    if o_coord > max_lim_o; max_lim_o = o_coord; end
                end
            end
            limit_ortho_map[k] = max_lim_o
        else
            limit_ortho_map[k] = Int32(0)
        end

        # 4. Split coordinate along axis_ortho
        if has_split
            max_s_o = Int32(0)
            for j in 1:dims_y, i in 1:dims_x
                if split_m[i, j, k] > UInt8(0)
                    o_coord = (axis_growth == Int32(1)) ? j : i
                    if o_coord > max_s_o; max_s_o = o_coord; end
                end
            end
            split_ortho_map[k] = max_s_o
        else
            split_ortho_map[k] = Int32(0)
        end
    end
end

@kernel function interpolate_volumetric_growth_params_kernel!(
    start_coord_map, obs_min_growth_map, obs_max_growth_map,
    obs_min_ortho_map, obs_max_ortho_map,
    limit_ortho_map, split_ortho_map,
    has_limit::Bool, has_split::Bool,
    dims_z::Int32
)
    k = @index(Global)
    if k <= dims_z
        # Interpolate start_coord_map
        if start_coord_map[k] == Int32(0)
            best_d = dims_z + Int32(1); best_z = Int32(0)
            for z in 1:dims_z
                if start_coord_map[z] > Int32(0)
                    d = abs(z - k)
                    if d < best_d; best_d = d; best_z = z; end
                end
            end
            if best_z > Int32(0)
                start_coord_map[k] = start_coord_map[best_z]
            end
        end

        # Interpolate obstacle maps
        if obs_max_growth_map[k] == Int32(0)
            best_d = dims_z + Int32(1); best_z = Int32(0)
            for z in 1:dims_z
                if obs_max_growth_map[z] > Int32(0)
                    d = abs(z - k)
                    if d < best_d; best_d = d; best_z = z; end
                end
            end
            if best_z > Int32(0)
                obs_min_growth_map[k] = obs_min_growth_map[best_z]
                obs_max_growth_map[k] = obs_max_growth_map[best_z]
                obs_min_ortho_map[k] = obs_min_ortho_map[best_z]
                obs_max_ortho_map[k] = obs_max_ortho_map[best_z]
            end
        end

        # Interpolate limit_ortho_map
        if has_limit && limit_ortho_map[k] == Int32(0)
            best_d = dims_z + Int32(1); best_z = Int32(0)
            for z in 1:dims_z
                if limit_ortho_map[z] > Int32(0)
                    d = abs(z - k)
                    if d < best_d; best_d = d; best_z = z; end
                end
            end
            if best_z > Int32(0)
                limit_ortho_map[k] = limit_ortho_map[best_z]
            end
        end

        # Interpolate split_ortho_map
        if has_split && split_ortho_map[k] == Int32(0)
            best_d = dims_z + Int32(1); best_z = Int32(0)
            for z in 1:dims_z
                if split_ortho_map[z] > Int32(0)
                    d = abs(z - k)
                    if d < best_d; best_d = d; best_z = z; end
                end
            end
            if best_z > Int32(0)
                split_ortho_map[k] = split_ortho_map[best_z]
            end
        end
    end
end

@kernel function volumetric_boundary_growth_kernel!(
    output,
    @Const(obs_m),
    @Const(start_coord_map),
    @Const(obs_min_growth_map), @Const(obs_max_growth_map),
    @Const(obs_min_ortho_map), @Const(obs_max_ortho_map),
    @Const(limit_ortho_map), @Const(split_ortho_map),
    z_min::Int32, z_max::Int32,
    axis_growth::Int32, step_growth::Int32,
    sublevel_code::Int32, max_dist_px::Int32,
    limit_to_obstacle_min::Bool,
    dims_x::Int32, dims_y::Int32, dims_z::Int32
)
    ortho_idx, k = @index(Global, NTuple)
    dims_growth = (axis_growth == Int32(1)) ? dims_x : dims_y
    dims_ortho = (axis_growth == Int32(1)) ? dims_y : dims_x
    
    if ortho_idx <= dims_ortho && k <= dims_z
        if k >= z_min && k <= z_max
            start_coord = start_coord_map[k]
            lim_ortho = limit_ortho_map[k]
            split_ortho = split_ortho_map[k]
            obs_max_o = obs_max_ortho_map[k]
            obs_min_g = obs_min_growth_map[k]
            obs_max_g = obs_max_growth_map[k]

            valid = true
            # 1. Orthogonal limit: e.g. skip anything anterior to limit (ortho < lim_ortho in LPS)
            if lim_ortho > Int32(0) && ortho_idx < lim_ortho
                valid = false
            end

            # 2. Sublevel partition:
            if valid
                if sublevel_code == Int32(1) # IIa: anterior/at split coordinate
                    if split_ortho > Int32(0) && ortho_idx > split_ortho
                        valid = false
                    end
                elseif sublevel_code == Int32(2) # IIb: posterior to split coordinate, anterior to posterior edge of obstacle
                    if split_ortho > Int32(0) && ortho_idx <= split_ortho
                        valid = false
                    end
                    if obs_max_o > Int32(0) && ortho_idx > obs_max_o
                        valid = false
                    end
                end
            end

            if valid && start_coord > Int32(0)
                # Check if this line has obstacle
                has_obs = false
                for g in 1:dims_growth
                    idx_x = (axis_growth == Int32(1)) ? g : ortho_idx
                    idx_y = (axis_growth == Int32(1)) ? ortho_idx : g
                    if obs_m[idx_x, idx_y, k] > UInt8(0)
                        has_obs = true
                        break
                    end
                end

                # Determine obstacle limit coordinate
                obs_lim_max = (step_growth > Int32(0)) ? obs_max_g : obs_min_g
                obs_lim_min = (step_growth > Int32(0)) ? obs_min_g : obs_max_g

                obs_lim = Int32(0)
                if limit_to_obstacle_min
                    if has_obs
                        obs_lim = obs_lim_max
                    elseif obs_max_o > Int32(0) && ortho_idx > obs_max_o
                        obs_lim = obs_lim_max
                    else
                        obs_lim = obs_lim_min
                    end
                else
                    obs_lim = obs_lim_max
                end

                curr_g = start_coord
                dist = Int32(0)
                while curr_g >= Int32(1) && curr_g <= dims_growth && dist < max_dist_px
                    if obs_lim > Int32(0)
                        if step_growth > Int32(0) && curr_g > obs_lim; break; end
                        if step_growth < Int32(0) && curr_g < obs_lim; break; end
                    end

                    idx_x = (axis_growth == Int32(1)) ? curr_g : ortho_idx
                    idx_y = (axis_growth == Int32(1)) ? ortho_idx : curr_g

                    if obs_m[idx_x, idx_y, k] > UInt8(0)
                        break
                    end

                    output[idx_x, idx_y, k] = UInt8(1)
                    curr_g += step_growth
                    dist += Int32(1)
                end
            end
        end
    end
end

function VolumetricBoundary2DLateralGrowth(
    tm,
    out_device::AbstractArray{UInt8, 3},
    params::Dict,
    start_device::AbstractArray{UInt8, 3},
    obs_device::AbstractArray{UInt8, 3},
    limit_device::Union{AbstractArray{UInt8, 3}, Nothing}=nothing,
    split_device::Union{AbstractArray{UInt8, 3}, Nothing}=nothing
)
    dims = size(out_device)
    dims_x, dims_y, dims_z = Int32(dims[1]), Int32(dims[2]), Int32(dims[3])

    growth_axis_str = lowercase(string(get(params, "growth_axis", "x")))
    axis_growth = (growth_axis_str == "y" || growth_axis_str == "2") ? Int32(2) : Int32(1)
    
    growth_target = lowercase(string(get(params, "growth_target", "lateral")))
    side = lowercase(string(get(params, "side", "")))

    step_growth = Int32(1)
    if axis_growth == Int32(1)
        if growth_target == "lateral"
            step_growth = (side == "right") ? Int32(-1) : Int32(1)
        elseif growth_target == "medial"
            step_growth = (side == "right") ? Int32(1) : Int32(-1)
        elseif growth_target in ["-x", "right"]
            step_growth = Int32(-1)
        else
            step_growth = Int32(1)
        end
    else
        if growth_target in ["anterior", "-y"]
            step_growth = Int32(-1)
        else
            step_growth = Int32(1)
        end
    end
    if haskey(params, "step_growth")
        step_growth = Int32(params["step_growth"])
    end

    sublevel_str = string(get(params, "sublevel", ""))
    sublevel_code = (sublevel_str == "IIa") ? Int32(1) : ((sublevel_str == "IIb") ? Int32(2) : Int32(0))
    if haskey(params, "sublevel_code")
        sublevel_code = Int32(params["sublevel_code"])
    end

    sp_axis = (axis_growth == Int32(1)) ? Float32(get(params, "spacing_x", 1.0)) : Float32(get(params, "spacing_y", 1.0))
    max_dist_mm = Float32(get(params, "max_dist_mm", 85.0))
    max_dist_px = round(Int32, max_dist_mm / sp_axis)
    if haskey(params, "max_dist_px")
        max_dist_px = Int32(params["max_dist_px"])
    end

    limit_to_obstacle_min = Bool(get(params, "limit_to_obstacle_min_if_no_hit", true))

    z_min = Int32(get(params, "z_min", 1))
    z_max = Int32(get(params, "z_max", dims_z))

    backend = tm.backend

    eff_limit_m = (limit_device !== nothing) ? limit_device : obs_device
    has_limit = (limit_device !== nothing) && Bool(get(params, "has_limit", true))

    eff_split_m = (split_device !== nothing) ? split_device : obs_device
    has_split = (split_device !== nothing) && Bool(get(params, "has_split", true))

    start_coord_map = KernelAbstractions.zeros(backend, Int32, dims_z)
    obs_min_growth_map = KernelAbstractions.zeros(backend, Int32, dims_z)
    obs_max_growth_map = KernelAbstractions.zeros(backend, Int32, dims_z)
    obs_min_ortho_map = KernelAbstractions.zeros(backend, Int32, dims_z)
    obs_max_ortho_map = KernelAbstractions.zeros(backend, Int32, dims_z)
    limit_ortho_map = KernelAbstractions.zeros(backend, Int32, dims_z)
    split_ortho_map = KernelAbstractions.zeros(backend, Int32, dims_z)

    k_ext! = extract_volumetric_growth_params_kernel!(backend)
    k_ext!(
        start_coord_map, obs_min_growth_map, obs_max_growth_map,
        obs_min_ortho_map, obs_max_ortho_map,
        limit_ortho_map, split_ortho_map,
        start_device, obs_device, eff_limit_m, eff_split_m,
        has_limit, has_split, axis_growth, step_growth,
        dims_x, dims_y, dims_z,
        ndrange=dims_z
    )
    KernelAbstractions.synchronize(backend)

    k_interp! = interpolate_volumetric_growth_params_kernel!(backend)
    k_interp!(
        start_coord_map, obs_min_growth_map, obs_max_growth_map,
        obs_min_ortho_map, obs_max_ortho_map,
        limit_ortho_map, split_ortho_map,
        has_limit, has_split, dims_z,
        ndrange=dims_z
    )
    KernelAbstractions.synchronize(backend)

    fill!(out_device, UInt8(0))
    dims_ortho = (axis_growth == Int32(1)) ? dims_y : dims_x
    k_growth! = volumetric_boundary_growth_kernel!(backend)
    k_growth!(
        out_device, obs_device,
        start_coord_map, obs_min_growth_map, obs_max_growth_map,
        obs_min_ortho_map, obs_max_ortho_map,
        limit_ortho_map, split_ortho_map,
        z_min, z_max,
        axis_growth, step_growth,
        sublevel_code, max_dist_px,
        limit_to_obstacle_min,
        dims_x, dims_y, dims_z,
        ndrange=(dims_ortho, dims_z)
    )
    KernelAbstractions.synchronize(backend)
    return out_device
end

# Backwards-compatible 3-mask signature
function VolumetricBoundary2DLateralGrowth(
    tm,
    out_device::AbstractArray{UInt8, 3},
    params::Dict,
    in_device::AbstractArray{UInt8, 3},
    obs_device::AbstractArray{UInt8, 3},
    ijv_device::AbstractArray{UInt8, 3}
)
    VolumetricBoundary2DLateralGrowth(tm, out_device, params, in_device, obs_device, nothing, ijv_device)
end

# =====================================================================
# 7. LimitZByLandmark (Pure GPU)
# =====================================================================
# 7. LimitZByLandmark (Pure GPU)
# =====================================================================
@kernel function limit_z_by_landmark_kernel!(output, input, dims, z_min::Int, z_max::Int)
    I, J, K = @index(Global, NTuple)
    if I <= dims[1] && J <= dims[2] && K <= dims[3]
        if K >= z_min && K <= z_max
            output[I, J, K] = input[I, J, K]
        else
            output[I, J, K] = UInt8(0)
        end
    end
end

function LimitZByLandmark(tm, out_device::AbstractArray{UInt8, 3}, params::Dict, in_device::AbstractArray{UInt8, 3}, args...)
    copyto!(out_device, in_device)
end

function AnteriorExtrusion(tm, out_device::AbstractArray{UInt8, 3}, params::Dict, in_device::AbstractArray{UInt8, 3}, args...)
    AnteriorGrowthMask(tm, out_device, params, in_device, args...)
end

@kernel function propagate_z_kernel!(output, input, dims, is_inferior::Bool, z_terminus::Int)
    I, J = @index(Global, NTuple)
    if I <= dims[1] && J <= dims[2]
        if is_inferior
            top_z = 0
            for k in 1:dims[3]
                if input[I, J, k] > 0
                    top_z = k
                    break
                end
            end
            if top_z > 0
                term_z = max(1, z_terminus)
                for k in top_z:-1:term_z
                    output[I, J, k] = 1
                end
            end
        else
            bot_z = 0
            for k in 1:dims[3]
                if input[I, J, k] > 0
                    bot_z = k
                    break
                end
            end
            if bot_z > 0
                term_z = min(dims[3], z_terminus)
                for k in bot_z:term_z
                    output[I, J, k] = 1
                end
            end
        end
    end
end

function PropagateZ(tm, out_device::AbstractArray{UInt8, 3}, params::Dict, in_device::AbstractArray{UInt8, 3}, args...)
    direction = get(params, "direction", "inferior")
    is_inferior = lowercase(direction) == "inferior"
    z_term = is_inferior ? 1 : size(out_device, 3)
    fill!(out_device, 0)
    kernel! = propagate_z_kernel!(tm.backend)
    kernel!(out_device, in_device, size(out_device), is_inferior, z_term, ndrange=(size(out_device, 1), size(out_device, 2)))
    KernelAbstractions.synchronize(tm.backend)
end

@kernel function station1_filter_kernel!(out, in_mask, trachea, esophagus, thyroid, lung_l, lung_r, scm, scalene, cricoid, manubrium, clavicle, dims, cricoid_z_min::Int, mid_x_z, clav_min_x_z, clav_max_x_z, side_is_left::Bool, side_is_right::Bool, floor_z_map)
    I, J, K = @index(Global, NTuple)
    if I <= dims[1] && J <= dims[2] && K <= dims[3]
        if in_mask[I, J, K] > 0 && K >= floor_z_map[I, J] && K <= cricoid_z_min
            mid_x = mid_x_z[K]
            is_valid_side = (!side_is_left && !side_is_right) || (side_is_left && I >= mid_x) || (side_is_right && I <= mid_x)
            
            if side_is_left && I > clav_max_x_z[K]
                is_valid_side = false
            elseif side_is_right && I < clav_min_x_z[K]
                is_valid_side = false
            end
            
            if is_valid_side
                is_excluded = (trachea[I, J, K] > 0) || (esophagus[I, J, K] > 0) || (thyroid[I, J, K] > 0) ||
                              (lung_l[I, J, K] > 0) || (lung_r[I, J, K] > 0) || (scm[I, J, K] > 0) || (scalene[I, J, K] > 0) ||
                              (cricoid[I, J, K] > 0) || (manubrium[I, J, K] > 0) || (clavicle[I, J, K] > 0)
                out[I, J, K] = is_excluded ? UInt8(0) : UInt8(1)
            else
                out[I, J, K] = UInt8(0)
            end
        else
            out[I, J, K] = UInt8(0)
        end
    end
end

@kernel function compute_trachea_mid_x_kernel!(
    mid_x,
    @Const(trachea),
    dims_x::Int32, dims_y::Int32, dims_z::Int32,
    default_x::Int32
)
    k = @index(Global)
    if k <= dims_z
        sum_x = Int64(0); count = Int64(0)
        for j in 1:dims_y, i in 1:dims_x
            if trachea[i, j, k] > UInt8(0)
                sum_x += i
                count += 1
            end
        end
        if count > 0
            mid_x[k] = Int32(round(Float64(sum_x) / Float64(count)))
        else
            mid_x[k] = default_x
        end
    end
end

@kernel function compute_clavicle_lateral_x_kernel!(
    clav_min_x, clav_max_x,
    @Const(clavicle),
    dims_x::Int32, dims_y::Int32, dims_z::Int32
)
    k = @index(Global)
    if k <= dims_z
        min_x = dims_x
        max_x = Int32(1)
        found = false
        for j in 1:dims_y, i in 1:dims_x
            if clavicle[i, j, k] > UInt8(0)
                if i < min_x
                    min_x = Int32(i)
                end
                if i > max_x
                    max_x = Int32(i)
                end
                found = true
            end
        end
        if found
            clav_min_x[k] = min_x
            clav_max_x[k] = max_x
        else
            clav_min_x[k] = Int32(1)
            clav_max_x[k] = dims_x
        end
    end
end

@kernel function bone_floor_extract_kernel!(
    floor_z,
    @Const(manubrium), @Const(clavicle),
    dims_x::Int32, dims_y::Int32, dims_z::Int32
)
    i, j = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y
        top_k = Int32(0)
        for k in dims_z:-1:1
            if manubrium[i, j, k] > UInt8(0) || clavicle[i, j, k] > UInt8(0)
                top_k = Int32(k)
                break
            end
        end
        floor_z[i, j] = top_k
    end
end

@kernel function jfa_propagate_floor_kernel!(
    floor_out, @Const(floor_in),
    step::Int32,
    dims_x::Int32, dims_y::Int32
)
    i, j = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y
        best_val = floor_in[i, j]
        if best_val == Int32(0)
            for dj in Int32(-1):Int32(1), di in Int32(-1):Int32(1)
                ni = i + di * step
                nj = j + dj * step
                if ni >= Int32(1) && ni <= dims_x && nj >= Int32(1) && nj <= dims_y
                    cand = floor_in[ni, nj]
                    if cand > Int32(0)
                        best_val = cand
                        break
                    end
                end
            end
        end
        floor_out[i, j] = best_val
    end
end

function Station1LowCervical(tm, out_device::AbstractArray{UInt8, 3}, params::Dict, in_device::AbstractArray{UInt8, 3},
                             cricoid_device::AbstractArray{UInt8, 3}, trachea_device::AbstractArray{UInt8, 3},
                             manubrium_device::AbstractArray{UInt8, 3}, clavicle_device::AbstractArray{UInt8, 3},
                             esophagus_device::AbstractArray{UInt8, 3}, thyroid_device::AbstractArray{UInt8, 3},
                             lung_l_device::AbstractArray{UInt8, 3}, lung_r_device::AbstractArray{UInt8, 3},
                             scm_device::AbstractArray{UInt8, 3}, scalene_device::AbstractArray{UInt8, 3}, args...)
    side_arg = length(args) >= 3 ? string(args[3]) : ""
    side = !isempty(side_arg) ? lowercase(side_arg) : lowercase(get(params, "side", ""))
    side_is_left = side == "left"
    side_is_right = side == "right"
    dims = size(out_device)
    sp_x = Float32(get(params, "spacing_x", 1.0))
    sp_y = Float32(get(params, "spacing_y", 1.0))
    
    # 1. Cricoid Superior Bound (GPU bounding boxes)
    c_bbox = RuleExecutors.gpu_bounding_box(tm.backend, cricoid_device)
    cricoid_z_min = dims[3]
    if c_bbox !== nothing
        cricoid_z_min = c_bbox[5]
    else
        tr_bbox = RuleExecutors.gpu_bounding_box(tm.backend, trachea_device)
        if tr_bbox !== nothing
            cricoid_z_min = tr_bbox[6]
        end
    end
    
    # 2. Trachea Midline per slice on GPU
    mid_x_device = KernelAbstractions.zeros(tm.backend, Int32, dims[3])
    k_mid! = compute_trachea_mid_x_kernel!(tm.backend)
    k_mid!(mid_x_device, trachea_device, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), Int32(dims[1] ÷ 2), ndrange=dims[3])
    KernelAbstractions.synchronize(tm.backend)
    
    # 2b. Clavicle lateral constraints per slice on GPU
    clav_min_x_device = KernelAbstractions.zeros(tm.backend, Int32, dims[3])
    clav_max_x_device = KernelAbstractions.zeros(tm.backend, Int32, dims[3])
    k_clav! = compute_clavicle_lateral_x_kernel!(tm.backend)
    k_clav!(clav_min_x_device, clav_max_x_device, clavicle_device, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=dims[3])
    KernelAbstractions.synchronize(tm.backend)
    
    # 3. Bone Floor Map on GPU (manubrium + clavicles)
    floor_z_a = KernelAbstractions.zeros(tm.backend, Int32, dims[1], dims[2])
    floor_z_b = KernelAbstractions.zeros(tm.backend, Int32, dims[1], dims[2])
    k_bf! = bone_floor_extract_kernel!(tm.backend)
    k_bf!(floor_z_a, manubrium_device, clavicle_device, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=(dims[1], dims[2]))
    KernelAbstractions.synchronize(tm.backend)
    
    # Jump Flooding propagation across XY on GPU (9 passes instead of 512 serial CPU passes)
    k_jfa! = jfa_propagate_floor_kernel!(tm.backend)
    cur_in = floor_z_a
    cur_out = floor_z_b
    for step in Int32[256, 128, 64, 32, 16, 8, 4, 2, 1]
        k_jfa!(cur_out, cur_in, step, Int32(dims[1]), Int32(dims[2]), ndrange=(dims[1], dims[2]))
        KernelAbstractions.synchronize(tm.backend)
        cur_in, cur_out = cur_out, cur_in
    end
    floor_z_device = cur_in
    
    fill!(out_device, 0)
    kernel! = station1_filter_kernel!(tm.backend)
    kernel!(out_device, in_device, trachea_device, esophagus_device, thyroid_device,
            lung_l_device, lung_r_device, scm_device, scalene_device,
            cricoid_device, manubrium_device, clavicle_device,
            dims, cricoid_z_min, mid_x_device, clav_min_x_device, clav_max_x_device, side_is_left, side_is_right, floor_z_device, ndrange=dims)
    KernelAbstractions.synchronize(tm.backend)
end

function compute_2d_convex_hull_per_slice!(out_device::AbstractArray{UInt8, 3}, in_device::AbstractArray{UInt8, 3})
    backend = KernelAbstractions.get_backend(in_device)
    hull = RuleExecutors.execute_convex_hull_bridge(backend, in_device, in_device, size(in_device), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    copyto!(out_device, hull)
end

function VolumetricBoundary2D(tm, out_device::AbstractArray{UInt8, 3}, params::Dict, in_device::AbstractArray{UInt8, 3}, args...)
    compute_2d_convex_hull_per_slice!(out_device, in_device)
end

@kernel function pleural_space_kernel!(
    output,
    @Const(lung),
    shift_medial::Int32,
    shift_lateral::Int32,
    is_left::Bool,
    dims_x::Int32, dims_y::Int32, dims_z::Int32
)
    j, k = @index(Global, NTuple)
    if j <= dims_y && k <= dims_z
        min_x = Int32(dims_x + 1)
        max_x = Int32(0)
        for i in Int32(1):dims_x
            if lung[i, j, k] > UInt8(0)
                if min_x > dims_x; min_x = i; end
                max_x = i
            end
        end
        if max_x >= min_x
            if is_left
                x_lateral = max_x
                x_start = max(Int32(1), x_lateral - shift_medial)
                x_end = min(dims_x, x_lateral + shift_lateral)
                for i in x_start:x_end
                    output[i, j, k] = UInt8(1)
                end
            else
                x_lateral = min_x
                x_start = max(Int32(1), x_lateral - shift_lateral)
                x_end = min(dims_x, x_lateral + shift_medial)
                for i in x_start:x_end
                    output[i, j, k] = UInt8(1)
                end
            end
        end
    end
end

function PleuralSpaceCustom(tm, out_device::AbstractArray{UInt8, 3}, params::Dict, in_device::AbstractArray{UInt8, 3}, args...)
    backend = tm.backend
    dims = size(out_device)
    dims_x, dims_y, dims_z = Int32(dims[1]), Int32(dims[2]), Int32(dims[3])
    
    sp_x = Float32(get(params, "spacing_x", 1.0))
    dist_lat = Float32(get(params, "distance_lateral_mm", get(params, "distance_mm", 35.0)))
    dist_med = Float32(get(params, "distance_medial_mm", 10.0))
    
    shift_lat = max(Int32(1), Int32(round(dist_lat / sp_x)))
    shift_med = max(Int32(1), Int32(round(dist_med / sp_x)))
    
    eff_side = lowercase(string(get(params, "side", "left")))
    is_left = eff_side == "left" || occursin("left", lowercase(string(get(params, "lung_landmark", ""))))
    
    fill!(out_device, UInt8(0))
    k! = pleural_space_kernel!(backend)
    k!(out_device, in_device, shift_med, shift_lat, is_left, dims_x, dims_y, dims_z, ndrange=(dims_y, dims_z))
    KernelAbstractions.synchronize(backend)
    return out_device
end

@kernel function presacral_anterior_kernel!(output, input, dims, voxel_dist::Int32, mode::Int32)
    I, K = @index(Global, NTuple)
    if I <= dims[1] && K <= dims[3]
        min_y = Int32(0)
        for j in Int32(1):dims[2]
            if input[I, j, K] > UInt8(0)
                min_y = j
                break
            end
        end
        if min_y > Int32(0)
            start_y = max(Int32(1), min_y - voxel_dist)
            if mode == Int32(1) # front_line
                output[I, min_y, K] = UInt8(1)
            elseif mode == Int32(2) # moved_line
                output[I, start_y, K] = UInt8(1)
            else # zone (3)
                for j in (start_y + Int32(1)):(min_y - Int32(1))
                    output[I, j, K] = UInt8(1)
                end
            end
        end
    end
end

function PresacralAnteriorCustom(tm, out_device::AbstractArray{UInt8, 3}, params::Dict, in_device::AbstractArray{UInt8, 3}, args...)
    out_type = lowercase(get(params, "output_type", "zone"))
    dims = size(out_device)
    dist = Float32(get(params, "distance", get(params, "growth_mm", get(params, "distance_mm", 10.0))))
    sp_y = Float32(get(params, "spacing_y", 1.0))
    voxel_dist = Int32(round(dist / sp_y))
    
    mode = Int32(3) # default: zone
    if out_type == "front_line"
        mode = Int32(1)
    elseif out_type == "moved_line" || out_type == "shifted"
        mode = Int32(2)
    end
    
    fill!(out_device, UInt8(0))
    kernel! = presacral_anterior_kernel!(tm.backend)
    kernel!(out_device, in_device, (Int32(dims[1]), Int32(dims[2]), Int32(dims[3])), voxel_dist, mode, ndrange=(dims[1], dims[3]))
    KernelAbstractions.synchronize(tm.backend)
end

@kernel function split_components_x_kernel!(output, input, dims, is_left)
    I, J, K = @index(Global, NTuple)
    if I <= dims[1] && J <= dims[2] && K <= dims[3]
        mid_x = dims[1] ÷ 2
        if input[I, J, K] > 0
            if is_left && I >= mid_x
                output[I, J, K] = UInt8(1)
            elseif !is_left && I < mid_x
                output[I, J, K] = UInt8(1)
            else
                output[I, J, K] = UInt8(0)
            end
        else
            output[I, J, K] = UInt8(0)
        end
    end
end

function SplitConnectedComponents(tm, out_device::AbstractArray{UInt8, 3}, params::Dict, in_device::AbstractArray{UInt8, 3}, args...)
    dims = size(out_device)
    side = lowercase(get(params, "side", "left"))
    is_left = side == "left"
    
    kernel! = split_components_x_kernel!(tm.backend)
    kernel!(out_device, in_device, dims, is_left, ndrange=dims)
    KernelAbstractions.synchronize(tm.backend)
end


@kernel function posterior_slice_max_y_kernel!(max_y_slice, @Const(scm), dims_x::Int32, dims_y::Int32, dims_z::Int32)
    k = @index(Global, Linear)
    if k <= dims_z
        my_max = Int32(0)
        for j in 1:dims_y
            for i in 1:dims_x
                if scm[i, j, k] > UInt8(0)
                    my_max = max(my_max, Int32(j))
                end
            end
        end
        max_y_slice[k] = my_max
    end
end

@kernel function apply_posterior_mask_kernel!(output, @Const(level5), @Const(max_y_slice), dims_x::Int32, dims_y::Int32, dims_z::Int32)
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        my_max = max_y_slice[k]
        if my_max > 0 && j >= my_max && level5[i, j, k] > UInt8(0)
            output[i, j, k] = UInt8(1)
        else
            output[i, j, k] = UInt8(0)
        end
    end
end

function apply_posterior_triangle_support!(backend, level5_gpu, scm_gpu, neck2b_gpu, dims, spacing, origin, direction)
    dims_x, dims_y, dims_z = Int32(dims[1]), Int32(dims[2]), Int32(dims[3])
    
    max_y_slice = KernelAbstractions.zeros(backend, Int32, dims_z)
    k1! = posterior_slice_max_y_kernel!(backend)
    k1!(max_y_slice, scm_gpu, dims_x, dims_y, dims_z, ndrange=dims_z)
    KernelAbstractions.synchronize(backend)
    
    post_scm = KernelAbstractions.zeros(backend, UInt8, dims)
    k2! = apply_posterior_mask_kernel!(backend)
    k2!(post_scm, level5_gpu, max_y_slice, dims_x, dims_y, dims_z, ndrange=dims)
    KernelAbstractions.synchronize(backend)
    
    # Needs RuleExecutors.execute_convex_hull_bridge
    # We will inject this directly into kernels_custom_rules.jl
    bridged = RuleExecutors.execute_convex_hull_bridge(backend, post_scm, neck2b_gpu, dims, spacing, origin, direction)
    level5_gpu .|= bridged
    return nothing
end


# =====================================================================
# Spleen PD Point Custom Kernel (PCA per slice)
# =====================================================================
function HelperSplenicPDCustom(tm, out_device::AbstractArray{UInt8, 3}, params::Dict, spleen_device::AbstractArray{UInt8, 3}, args...)
    # We download the spleen mask to CPU because per-slice PCA is much simpler sequentially.
    spleen = Array(spleen_device)
    dims = size(spleen)
    out_cpu = zeros(UInt8, dims)

    for z in 1:dims[3]
        # Find all points in spleen on this slice
        slice_mask = spleen[:, :, z]
        pts_x = Float64[]
        pts_y = Float64[]
        for j in 1:dims[2]
            for i in 1:dims[1]
                if slice_mask[i, j] > 0
                    push!(pts_x, Float64(i))
                    push!(pts_y, Float64(j))
                end
            end
        end
        
        N = length(pts_x)
        if N < 5
            continue
        end

        cx = sum(pts_x) / N
        cy = sum(pts_y) / N

        cxx = 0.0
        cyy = 0.0
        cxy = 0.0
        for k in 1:N
            dx = pts_x[k] - cx
            dy = pts_y[k] - cy
            cxx += dx * dx
            cyy += dy * dy
            cxy += dx * dy
        end
        cxx /= N
        cyy /= N
        cxy /= N

        # Eigenvalues of 2x2 covariance matrix
        tr = cxx + cyy
        det = cxx * cyy - cxy * cxy
        # largest eigenvalue
        lambda1 = tr / 2.0 + sqrt(max(0.0, (tr / 2.0)^2 - det))
        
        # Eigenvector v1 (long axis)
        if abs(cxy) > 1e-6
            v1_x = lambda1 - cyy
            v1_y = cxy
        else
            if cxx > cyy
                v1_x = 1.0
                v1_y = 0.0
            else
                v1_x = 0.0
                v1_y = 1.0
            end
        end
        
        norm_v1 = sqrt(v1_x^2 + v1_y^2)
        if norm_v1 > 1e-6
            v1_x /= norm_v1
            v1_y /= norm_v1
        end

        # Short axis v2 (perpendicular)
        v2_x = -v1_y
        v2_y = v1_x

        # Find the medial-most point of the spleen (minimum X = toward aorta/midline)
        min_x_val = Inf
        min_x_y = cy
        for k in 1:N
            if pts_x[k] < min_x_val
                min_x_val = pts_x[k]
                min_x_y = pts_y[k]
            end
        end
        
        pd_x = round(Int, min_x_val)
        pd_y = round(Int, min_x_y)

        # Draw a 5x5 square around pd_x, pd_y to ensure it forms a robust hull
        for dj in -2:2
            for di in -2:2
                px = pd_x + di
                py = pd_y + dj
                if px >= 1 && px <= dims[1] && py >= 1 && py <= dims[2]
                    out_cpu[px, py, z] = 1
                end
            end
        end
    end

    # Upload back to GPU
    copyto!(out_device, out_cpu)
    KernelAbstractions.synchronize(tm.backend)
end

# =====================================================================
# Stomach Long Axis Split Custom Kernel (PCA per slice)
# =====================================================================
function StomachLongAxisHelper(tm, out_device::AbstractArray{UInt8, 3}, params::Dict, organ_device::AbstractArray{UInt8, 3}, args...)
    # Download organ mask to CPU for per-slice PCA
    organ = Array(organ_device)
    dims = size(organ)
    out_cpu = zeros(UInt8, dims)
    
    # Get the side parameter: "left" or "right" of the long axis
    side = lowercase(get(params, "side", "right"))

    for z in 1:dims[3]
        slice_mask = organ[:, :, z]
        pts_x = Float64[]
        pts_y = Float64[]
        for j in 1:dims[2]
            for i in 1:dims[1]
                if slice_mask[i, j] > 0
                    push!(pts_x, Float64(i))
                    push!(pts_y, Float64(j))
                end
            end
        end
        
        N = length(pts_x)
        if N < 5
            continue
        end

        cx = sum(pts_x) / N
        cy = sum(pts_y) / N

        # Compute 2x2 covariance matrix
        cxx = 0.0; cyy = 0.0; cxy = 0.0
        for k in 1:N
            dx = pts_x[k] - cx
            dy = pts_y[k] - cy
            cxx += dx * dx
            cyy += dy * dy
            cxy += dx * dy
        end
        cxx /= N; cyy /= N; cxy /= N

        # Eigenvector for largest eigenvalue (long axis)
        tr = cxx + cyy
        det = cxx * cyy - cxy * cxy
        lambda1 = tr / 2.0 + sqrt(max(0.0, (tr / 2.0)^2 - det))
        
        if abs(cxy) > 1e-6
            v1_x = lambda1 - cyy
            v1_y = cxy
        else
            if cxx > cyy
                v1_x = 1.0; v1_y = 0.0
            else
                v1_x = 0.0; v1_y = 1.0
            end
        end
        
        norm_v1 = sqrt(v1_x^2 + v1_y^2)
        if norm_v1 > 1e-6
            v1_x /= norm_v1; v1_y /= norm_v1
        end

        # Short axis (perpendicular to long axis) defines the split direction
        v2_x = -v1_y
        v2_y = v1_x

        # For each voxel on this slice, determine which side of the long axis it is on
        # side == "right" means keep voxels where projection on short axis < 0 (lower X = medial)
        # side == "left" means keep voxels where projection on short axis >= 0 (higher X = lateral)
        for j in 1:dims[2]
            for i in 1:dims[1]
                dx = Float64(i) - cx
                dy = Float64(j) - cy
                proj = dx * v2_x + dy * v2_y
                if side == "right"
                    if proj <= 0.0
                        out_cpu[i, j, z] = 1
                    end
                else  # left
                    if proj >= 0.0
                        out_cpu[i, j, z] = 1
                    end
                end
            end
        end
    end

    copyto!(out_device, out_cpu)
    KernelAbstractions.synchronize(tm.backend)
end

# =====================================================================
# Pancreas Head/Tail Split via Per-Slice Connected Component Analysis
# =====================================================================
function PancreasSplitHelper(tm, out_device::AbstractArray{UInt8, 3}, params::Dict,
                              pancreas_device::AbstractArray{UInt8, 3}, args...)
    pancreas = Array(pancreas_device)
    dims = size(pancreas)
    out_cpu = zeros(UInt8, dims)
    side = lowercase(get(params, "side", "head"))  # "head" or "tail"

    # Get duodenum from args for Z reference
    duodenum = nothing
    if length(args) >= 1 && args[1] !== nothing
        duodenum = Array(args[1])
    end

    # Find top Z of duodenum
    start_z = dims[3]
    if duodenum !== nothing
        for z in dims[3]:-1:1
            if any(@view(duodenum[:,:,z]) .> UInt8(0))
                start_z = z; break
            end
        end
    end

    # Iterate from start_z downward to find first slice where pancreas has 2+ CCs
    split_x = nothing
    for z in start_z:-1:1
        slice = @view(pancreas[:,:,z])
        if !any(slice .> UInt8(0)); continue; end

        # Simple 2D CCL via BFS flood fill
        labeled = zeros(Int, dims[1], dims[2])
        label_count = 0
        for j in 1:dims[2], i in 1:dims[1]
            if slice[i,j] > UInt8(0) && labeled[i,j] == 0
                label_count += 1
                queue = Tuple{Int,Int}[(i,j)]
                labeled[i,j] = label_count
                while !isempty(queue)
                    ci, cj = popfirst!(queue)
                    for (di,dj) in ((-1,0),(1,0),(0,-1),(0,1))
                        ni, nj = ci+di, cj+dj
                        if 1<=ni<=dims[1] && 1<=nj<=dims[2] &&
                           slice[ni,nj] > UInt8(0) && labeled[ni,nj] == 0
                            labeled[ni,nj] = label_count
                            push!(queue, (ni,nj))
                        end
                    end
                end
            end
        end

        if label_count >= 2
            # Find centroids of each component
            comp_cx = Dict{Int,Float64}()
            comp_count = Dict{Int,Int}()
            for j in 1:dims[2], i in 1:dims[1]
                l = labeled[i,j]
                if l > 0
                    comp_cx[l] = get(comp_cx, l, 0.0) + Float64(i)
                    comp_count[l] = get(comp_count, l, 0) + 1
                end
            end
            for l in keys(comp_cx)
                comp_cx[l] /= comp_count[l]
            end

            # Find the more-leftward component (lower centroid X = anatomical right in LPS, but here we want the tail side)
            # Actually: in our array coords, lower X = patient right. The pancreas tail is on patient left (higher X).
            # The leftward component (lower X centroid) is the head side.
            # The user says: "look at the connected component that is more to the left" (patient left = higher X)
            # "get a point on it that is most to the right" (most to patient right = lowest X on that component)
            # "all that is to the right of p1 on pancreas is pancreatic head"
            # So: leftward in patient space = higher X in array. Find that component. Its rightmost point (lowest X) = split_x.
            # Everything with X <= split_x = head. X > split_x = tail.
            sorted_labels = sort(collect(keys(comp_cx)), by=l->comp_cx[l], rev=true)  # highest X first = most patient-left
            left_patient_label = sorted_labels[1]

            # Find the most patient-right point (min X) of this patient-left component
            min_x = dims[1] + 1
            for j in 1:dims[2], i in 1:dims[1]
                if labeled[i,j] == left_patient_label && i < min_x
                    min_x = i
                end
            end
            split_x = min_x
            break
        end
    end

    if split_x === nothing
        # Fallback: use center X of pancreas
        px_sum = 0.0; px_count = 0
        for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
            if pancreas[i,j,k] > UInt8(0)
                px_sum += Float64(i); px_count += 1
            end
        end
        split_x = px_count > 0 ? round(Int, px_sum / px_count) : dims[1] ÷ 2
    end

    # Apply split: head = X <= split_x (patient right), tail = X > split_x (patient left)
    for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
        if pancreas[i,j,k] > UInt8(0)
            if side == "head"
                if i <= split_x
                    out_cpu[i,j,k] = UInt8(1)
                end
            else  # tail
                if i > split_x
                    out_cpu[i,j,k] = UInt8(1)
                end
            end
        end
    end

    copyto!(out_device, out_cpu)
    KernelAbstractions.synchronize(tm.backend)
end

# =====================================================================
# Pyloric Area Detection via Per-Slice Connected Component Analysis on Stomach
# =====================================================================
function PyloricSplitHelper(tm, out_device::AbstractArray{UInt8, 3}, params::Dict,
                             stomach_device::AbstractArray{UInt8, 3}, args...)
    stomach_orig = Array(stomach_device)
    dims = size(stomach_orig)
    out_cpu = zeros(UInt8, dims)

    # Try with original, then with increasing erosion
    for erosion_step in 0:5
        stomach = copy(stomach_orig)
        # Apply morphological erosion (shrink by 1 voxel per step, 6-connected)
        for _ in 1:erosion_step
            eroded = zeros(UInt8, dims)
            for k in 2:dims[3]-1, j in 2:dims[2]-1, i in 2:dims[1]-1
                if stomach[i,j,k] > UInt8(0) &&
                   stomach[i-1,j,k] > UInt8(0) && stomach[i+1,j,k] > UInt8(0) &&
                   stomach[i,j-1,k] > UInt8(0) && stomach[i,j+1,k] > UInt8(0) &&
                   stomach[i,j,k-1] > UInt8(0) && stomach[i,j,k+1] > UInt8(0)
                    eroded[i,j,k] = UInt8(1)
                end
            end
            stomach = eroded
        end

        # Find LOWEST Z (most inferior = near pylorus) where stomach has 2+ CCs
        # Iterate from bottom up — the pyloric narrowing is at the inferior end
        for z in 1:dims[3]
            slice = @view(stomach[:,:,z])
            if !any(slice .> UInt8(0)); continue; end

            # 2D CCL via BFS
            labeled = zeros(Int, dims[1], dims[2])
            label_count = 0
            for jj in 1:dims[2], ii in 1:dims[1]
                if slice[ii,jj] > UInt8(0) && labeled[ii,jj] == 0
                    label_count += 1
                    queue = Tuple{Int,Int}[(ii,jj)]
                    labeled[ii,jj] = label_count
                    while !isempty(queue)
                        ci, cj = popfirst!(queue)
                        for (di,dj) in ((-1,0),(1,0),(0,-1),(0,1))
                            ni, nj = ci+di, cj+dj
                            if 1<=ni<=dims[1] && 1<=nj<=dims[2] &&
                               slice[ni,nj] > UInt8(0) && labeled[ni,nj] == 0
                                labeled[ni,nj] = label_count
                                push!(queue, (ni,nj))
                            end
                        end
                    end
                end
            end

            if label_count >= 2
                # Found the highest slice with 2+ CCs
                # Now re-run CCL on the ORIGINAL (un-eroded) stomach at this Z
                orig_slice = @view(stomach_orig[:,:,z])
                if !any(orig_slice .> UInt8(0)); continue; end
                
                labeled_orig = zeros(Int, dims[1], dims[2])
                label_count_orig = 0
                for jj in 1:dims[2], ii in 1:dims[1]
                    if orig_slice[ii,jj] > UInt8(0) && labeled_orig[ii,jj] == 0
                        label_count_orig += 1
                        queue = Tuple{Int,Int}[(ii,jj)]
                        labeled_orig[ii,jj] = label_count_orig
                        while !isempty(queue)
                            ci, cj = popfirst!(queue)
                            for (di,dj) in ((-1,0),(1,0),(0,-1),(0,1))
                                ni, nj = ci+di, cj+dj
                                if 1<=ni<=dims[1] && 1<=nj<=dims[2] &&
                                   orig_slice[ni,nj] > UInt8(0) && labeled_orig[ni,nj] == 0
                                    labeled_orig[ni,nj] = label_count_orig
                                    push!(queue, (ni,nj))
                                end
                            end
                        end
                    end
                end

                # Find rightward component on original (lowest X centroid = patient right in LPS)
                comp_cx = Dict{Int,Float64}()
                comp_count = Dict{Int,Int}()
                for jj in 1:dims[2], ii in 1:dims[1]
                    l = labeled_orig[ii,jj]
                    if l > 0
                        comp_cx[l] = get(comp_cx, l, 0.0) + Float64(ii)
                        comp_count[l] = get(comp_count, l, 0) + 1
                    end
                end
                for l in keys(comp_cx)
                    comp_cx[l] /= comp_count[l]
                end
                
                # If original has only 1 CC at this slice, just use it
                if label_count_orig < 2
                    for jj in 1:dims[2], ii in 1:dims[1]
                        if labeled_orig[ii,jj] > 0
                            out_cpu[ii,jj,z] = UInt8(1)
                        end
                    end
                else
                    # Rightward = lowest X centroid (patient right in LPS)
                    sorted_labels = sort(collect(keys(comp_cx)), by=l->comp_cx[l])
                    right_label = sorted_labels[1]

                    # Find max X of the rightward CC — this defines the pyloric boundary
                    pyloric_max_x = 0
                    for jj in 1:dims[2], ii in 1:dims[1]
                        if labeled_orig[ii,jj] == right_label && ii > pyloric_max_x
                            pyloric_max_x = ii
                        end
                    end

                    # Mark ALL original stomach voxels at or below split Z that are within pyloric X range
                    for kk in 1:z, jj in 1:dims[2], ii in 1:dims[1]
                        if stomach_orig[ii,jj,kk] > UInt8(0) && ii <= pyloric_max_x
                            out_cpu[ii,jj,kk] = UInt8(1)
                        end
                    end
                end

                println("    [PyloricSplit] Found 2-CC split at Z=$z (erosion_step=$erosion_step), output $(sum(out_cpu)) voxels")
                copyto!(out_device, out_cpu)
                KernelAbstractions.synchronize(tm.backend)
                return
            end
        end
    end

    # Fallback: use bottom third of stomach
    min_z = dims[3]; max_z = 1
    for k in 1:dims[3]
        if any(@view(stomach_orig[:,:,k]) .> UInt8(0))
            min_z = min(min_z, k); max_z = max(max_z, k)
        end
    end
    cutoff_z = min_z + (max_z - min_z) ÷ 3
    for k in min_z:cutoff_z, j in 1:dims[2], i in 1:dims[1]
        if stomach_orig[i,j,k] > UInt8(0)
            out_cpu[i,j,k] = UInt8(1)
        end
    end
    println("    [PyloricSplit] No 2-CC slice found, using bottom-third fallback ($(sum(out_cpu)) voxels)")

    copyto!(out_device, out_cpu)
    KernelAbstractions.synchronize(tm.backend)
end

# =====================================================================
# Morphological Erosion Helper (single step, 6-connected)
# =====================================================================
@kernel function erosion_6conn_kernel!(out, inp, dims_x::Int32, dims_y::Int32, dims_z::Int32)
    i, j, k = @index(Global, NTuple)
    if i >= Int32(2) && i <= dims_x - Int32(1) && j >= Int32(2) && j <= dims_y - Int32(1) && k >= Int32(2) && k <= dims_z - Int32(1)
        if inp[i,j,k] > UInt8(0) &&
           inp[i-1,j,k] > UInt8(0) && inp[i+1,j,k] > UInt8(0) &&
           inp[i,j-1,k] > UInt8(0) && inp[i,j+1,k] > UInt8(0) &&
           inp[i,j,k-1] > UInt8(0) && inp[i,j,k+1] > UInt8(0)
            out[i,j,k] = UInt8(1)
        end
    end
end

function ErosionHelper(tm, out_device::AbstractArray{UInt8, 3}, params::Dict,
                        in_device::AbstractArray{UInt8, 3}, args...)
    dims = size(in_device)
    fill!(out_device, UInt8(0))
    kernel! = erosion_6conn_kernel!(tm.backend)
    kernel!(out_device, in_device, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=dims)
    KernelAbstractions.synchronize(tm.backend)
end

# =====================================================================
# InguinalAnteriorProxy (Pure GPU)
# =====================================================================
@kernel function inguinal_anterior_filter_kernel!(output, input, dims, med_x, med_y, lat_x, lat_y, is_lps)
    I, J, K = @index(Global, NTuple)
    if I <= dims[1] && J <= dims[2] && K <= dims[3]
        if input[I, J, K] > 0
            dx = Float32(lat_x - med_x)
            dy = Float32(lat_y - med_y)
            if abs(dx) > 1e-5
                slope = dy / dx
                y_line = Float32(med_y) + slope * (Float32(I) - Float32(med_x))
                if is_lps
                    if Float32(J) <= y_line
                        output[I, J, K] = input[I, J, K]
                    else
                        output[I, J, K] = 0
                    end
                else
                    if Float32(J) >= y_line
                        output[I, J, K] = input[I, J, K]
                    else
                        output[I, J, K] = 0
                    end
                end
            else
                output[I, J, K] = input[I, J, K]
            end
        else
            output[I, J, K] = 0
        end
    end
end

function InguinalAnteriorProxy(tm, out_device::AbstractArray{UInt8, 3}, params::Dict, prim_device::AbstractArray{UInt8, 3}, hip_device::AbstractArray{UInt8, 3}, args...)
    side = get(params, "side", "Left")
    spacing = get(params, "spacing", (1.0, 1.0, 1.0))
    direction = get(params, "direction", (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    # Pipeline data is always in LPS (converted during H5 creation via T_lps_ras).
    # In LPS: Y increases posteriorly, so anterior = lower Y values.
    is_lps = true
    
    dims = size(prim_device)
    # 2D projection: max over Z axis, stays on GPU until we need extreme points
    hip_2d_gpu = dropdims(any(hip_device .> UInt8(0), dims=3), dims=3)
    hip_2d = Array(hip_2d_gpu)  # Only transfer the 2D projection (~512 KB vs 131 MB)
    
    xs, ys = [], []
    for j in 1:dims[2], i in 1:dims[1]
        if hip_2d[i, j]
            push!(xs, i)
            push!(ys, j)
        end
    end
    
    if isempty(xs)
        out_device .= prim_device
        return
    end
    
    x_min, x_max = minimum(xs), maximum(xs)
    x_mid = (x_min + x_max) / 2.0
    
    med_xs, med_ys = Int[], Int[]
    lat_xs, lat_ys = Int[], Int[]
    
    is_left = lowercase(side) == "left"
    for k in 1:length(xs)
        x = xs[k]
        y = ys[k]
        if is_left
            if x <= x_mid
                push!(med_xs, x); push!(med_ys, y)
            else
                push!(lat_xs, x); push!(lat_ys, y)
            end
        else
            if x >= x_mid
                push!(med_xs, x); push!(med_ys, y)
            else
                push!(lat_xs, x); push!(lat_ys, y)
            end
        end
    end
    
    if isempty(med_xs) || isempty(lat_xs)
        out_device .= prim_device
        return
    end
    
    dy_pixels = 5.0 / spacing[2]
    med_y = 0; med_x = 0; lat_y = 0; lat_x = 0
    if is_lps
        med_idx = argmin(med_ys)
        lat_idx = argmin(lat_ys)
        med_y = med_ys[med_idx] + dy_pixels
        lat_y = lat_ys[lat_idx] + dy_pixels
        med_x = med_xs[med_idx]
        lat_x = lat_xs[lat_idx]
    else
        med_idx = argmax(med_ys)
        lat_idx = argmax(lat_ys)
        med_y = med_ys[med_idx] - dy_pixels
        lat_y = lat_ys[lat_idx] - dy_pixels
        med_x = med_xs[med_idx]
        lat_x = lat_xs[lat_idx]
    end
    
    k! = inguinal_anterior_filter_kernel!(tm.backend)
    k!(out_device, prim_device, dims, Float32(med_x), Float32(med_y), Float32(lat_x), Float32(lat_y), is_lps, ndrange=dims)
    KernelAbstractions.synchronize(tm.backend)
end



# =====================================================================
# ExtractMainBronchi (Task 2: Sagittal Iteration)
# =====================================================================
function count_2d_cc(slice::AbstractMatrix{UInt8})
    visited = zeros(Bool, size(slice))
    count = 0
    for i in 1:size(slice, 1)
        for j in 1:size(slice, 2)
            if slice[i, j] > 0 && !visited[i, j]
                count += 1
                queue = [(i, j)]
                visited[i, j] = true
                while !isempty(queue)
                    curr = popfirst!(queue)
                    for d in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
                        ni, nj = curr[1]+d[1], curr[2]+d[2]
                        if 1 <= ni <= size(slice, 1) && 1 <= nj <= size(slice, 2)
                            if slice[ni, nj] > 0 && !visited[ni, nj]
                                visited[ni, nj] = true
                                push!(queue, (ni, nj))
                            end
                        end
                    end
                end
            end
        end
    end
    return count
end

@kernel function extract_main_bronchi_kernel!(output, input, dims, split_x, is_left)
    I, J, K = @index(Global, NTuple)
    if I <= dims[1] && J <= dims[2] && K <= dims[3]
        if input[I, J, K] > 0
            if is_left
                # Left bronchus: medial is right (larger X). Lateral is left (smaller X).
                # Wait: LPS means -X is Right, +X is Left.
                # So X max is Leftmost (lateral for left). X min is Rightmost (medial for left).
                # Let's keep the exact condition the python script used or just use the logic passed.
                if I > split_x
                    output[I, J, K] = 1
                end
            else
                if I < split_x
                    output[I, J, K] = 1
                end
            end
        end
    end
end

function ExtractMainBronchi(tm, out_device::AbstractArray{UInt8, 3}, params::Dict, in_device::AbstractArray{UInt8, 3}, args...)
    side = get(params, "side", "left")
    dims = size(out_device)
    in_cpu = Array(in_device)
    
    nz = findall(in_cpu .> 0)
    if isempty(nz)
        return
    end
    xs = [p[1] for p in nz]
    x_min, x_max = minimum(xs), maximum(xs)
    
    split_x = side == "left" ? x_min : x_max
    
    # In Slicer/DICOM LPS: 
    # Left Lung is at +X (larger X). Right Lung is at -X (smaller X).
    # So for Left side, medial is smaller X, lateral is larger X.
    # Python logic for Left: scan from Medial to Lateral (x_min to x_max).
    if side == "left"
        for x in x_min:x_max
            cc_count = count_2d_cc(in_cpu[x, :, :])
            if cc_count > 1
                split_x = x
                break
            end
        end
        # Keep everything medial to the split (X < split_x)
        @kernel function left_kernel!(out, in, dims, sp_x)
            I, J, K = @index(Global, NTuple)
            if in[I, J, K] > 0 && I < sp_x
                out[I, J, K] = 1
            end
        end
        kernel = left_kernel!(tm.backend)
        kernel(out_device, in_device, dims, split_x, ndrange=dims)
    else
        # For Right side, medial is larger X, lateral is smaller X.
        # Scan from Medial to Lateral (x_max down to x_min).
        for x in x_max:-1:x_min
            cc_count = count_2d_cc(in_cpu[x, :, :])
            if cc_count > 1
                split_x = x
                break
            end
        end
        # Keep everything medial to the split (X > split_x)
        @kernel function right_kernel!(out, in, dims, sp_x)
            I, J, K = @index(Global, NTuple)
            if in[I, J, K] > 0 && I > sp_x
                out[I, J, K] = 1
            end
        end
        kernel = right_kernel!(tm.backend)
        kernel(out_device, in_device, dims, split_x, ndrange=dims)
    end
end

end # module
