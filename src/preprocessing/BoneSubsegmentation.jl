module BoneSubsegmentation

using KernelAbstractions
using LinearAlgebra
using Statistics
using CUDA

export generate_bone_subsegments, extract_bone_layers!

"""
Kernel to evaluate Euclidean distance from lesion surface and apply bone surface constraint.
"""
@kernel function kernel_bone_surface!(
    surface_out,
    @Const(bone_cortical),
    @Const(lesion_mask),
    @Const(lesion_coords_x),
    @Const(lesion_coords_y),
    @Const(lesion_coords_z),
    num_lesion_pts::Int,
    sp_x::Float32,
    sp_y::Float32,
    sp_z::Float32,
    max_dist_mm::Float32
)
    I = @index(Global, Cartesian)
    ix, iy, iz = I[1], I[2], I[3]
    
    # Only process cortical bone voxels that are strictly within bone and NOT inside the lesion
    if bone_cortical[ix, iy, iz] && !lesion_mask[ix, iy, iz]
        min_d2 = Float32(1e9)
        max_d2 = max_dist_mm * max_dist_mm
        
        # Check distance to lesion points
        for k in 1:num_lesion_pts
            lx = lesion_coords_x[k]
            ly = lesion_coords_y[k]
            lz = lesion_coords_z[k]
            
            dx = Float32(ix - lx) * sp_x
            dy = Float32(iy - ly) * sp_y
            dz = Float32(iz - lz) * sp_z
            d2 = dx*dx + dy*dy + dz*dz
            if d2 < min_d2
                min_d2 = d2
            end
        end
        
        if min_d2 <= max_d2
            surface_out[ix, iy, iz] = true
        else
            surface_out[ix, iy, iz] = false
        end
    else
        surface_out[ix, iy, iz] = false
    end
end

"""
Kernel to extract marrow voxels within spherical radius R_L around marrow centroid.
"""
@kernel function kernel_bone_marrow!(
    marrow_out,
    @Const(bone_marrow),
    @Const(lesion_mask),
    cx::Float32,
    cy::Float32,
    cz::Float32,
    sp_x::Float32,
    sp_y::Float32,
    sp_z::Float32,
    radius_mm::Float32
)
    I = @index(Global, Cartesian)
    ix, iy, iz = I[1], I[2], I[3]
    
    if bone_marrow[ix, iy, iz] && !lesion_mask[ix, iy, iz]
        dx = (Float32(ix) - cx) * sp_x
        dy = (Float32(iy) - cy) * sp_y
        dz = (Float32(iz) - cz) * sp_z
        d2 = dx*dx + dy*dy + dz*dz
        if d2 <= radius_mm * radius_mm
            marrow_out[ix, iy, iz] = true
        else
            marrow_out[ix, iy, iz] = false
        end
    else
        marrow_out[ix, iy, iz] = false
    end
end

"""
Morphological Bone Surface and Bone Marrow Subsegmentation.
- `mask_vol`: 3D Float32 labelmap volume containing lesions (values >= 1).
- `bone_vol`: 3D Float32 labelmap volume with cortical (2) and cancellous/marrow (1), or binary bone.
- `spacing`: (sx, sy, sz) voxel spacing in mm.
- `lesion_id`: integer ID of lesion to subsegment.

Returns:
`(surface_mask::Array{Bool, 3}, marrow_mask::Array{Bool, 3})`
"""
function generate_bone_subsegments(
    mask_vol::AbstractArray{Float32, 3},
    bone_vol::AbstractArray{Float32, 3},
    spacing::Tuple{Float64, Float64, Float64},
    lesion_id::Int;
    max_surface_dist_mm::Float64 = 25.0
)
    nx, ny, nz = size(mask_vol)
    surface_mask = zeros(Bool, nx, ny, nz)
    marrow_mask = zeros(Bool, nx, ny, nz)
    
    # 1. Find lesion coordinates
    lesion_indices = findall(mask_vol .== Float32(lesion_id))
    if isempty(lesion_indices)
        return surface_mask, marrow_mask
    end
    
    # Compute bounding box with 30mm margin
    sp_x, sp_y, sp_z = Float32(spacing[1]), Float32(spacing[2]), Float32(spacing[3])
    margin_x = ceil(Int, 30.0 / sp_x)
    margin_y = ceil(Int, 30.0 / sp_y)
    margin_z = ceil(Int, 30.0 / sp_z)
    
    xs = [idx[1] for idx in lesion_indices]
    ys = [idx[2] for idx in lesion_indices]
    zs = [idx[3] for idx in lesion_indices]
    
    x_min = max(1, minimum(xs) - margin_x)
    x_max = min(nx, maximum(xs) + margin_x)
    y_min = max(1, minimum(ys) - margin_y)
    y_max = min(ny, maximum(ys) + margin_y)
    z_min = max(1, minimum(zs) - margin_z)
    z_max = min(nz, maximum(zs) + margin_z)
    
    crop_lesion = Array{Bool}(mask_vol[x_min:x_max, y_min:y_max, z_min:z_max] .== Float32(lesion_id))
    crop_bone = bone_vol[x_min:x_max, y_min:y_max, z_min:z_max]
    
    crop_dims = size(crop_lesion)
    
    crop_bone_bool = Array{Bool}(crop_bone .> 0.0f0)
    crop_cortical = copy(crop_bone_bool)
    crop_marrow = copy(crop_bone_bool)
    
    # Extract sub-sampled lesion border coordinates for KernelAbstractions kernel
    crop_lesion_pts = findall(crop_lesion)
    if isempty(crop_lesion_pts)
        return surface_mask, marrow_mask
    end
    
    # Subsample if too dense to keep kernel fast (< 500 boundary points)
    step = max(1, length(crop_lesion_pts) ÷ 500)
    sampled_pts = crop_lesion_pts[1:step:end]
    num_pts = length(sampled_pts)
    
    lx = Int32[p[1] for p in sampled_pts]
    ly = Int32[p[2] for p in sampled_pts]
    lz = Int32[p[3] for p in sampled_pts]

    # Convert to GPU arrays if CUDA is available, on device 1
    has_cuda = false
    try
        has_cuda = CUDA.functional()
    catch
    end

    if has_cuda
        try
            CUDA.device!(1)
        catch e
            @warn "Failed to set CUDA device to 1: $e"
        end
        crop_surface = CUDA.zeros(Bool, crop_dims)
        crop_cortical_gpu = CUDA.CuArray(crop_cortical)
        crop_lesion_gpu = CUDA.CuArray(crop_lesion)
        lx_gpu = CUDA.CuArray(lx)
        ly_gpu = CUDA.CuArray(ly)
        lz_gpu = CUDA.CuArray(lz)
    else
        crop_surface = zeros(Bool, crop_dims)
        crop_cortical_gpu = crop_cortical
        crop_lesion_gpu = crop_lesion
        lx_gpu = lx
        ly_gpu = ly
        lz_gpu = lz
    end
    
    backend = KernelAbstractions.get_backend(crop_surface)
    kernel_surface = kernel_bone_surface!(backend)
    
    kernel_surface(
        crop_surface,
        crop_cortical_gpu,
        crop_lesion_gpu,
        lx_gpu, ly_gpu, lz_gpu,
        num_pts,
        sp_x, sp_y, sp_z,
        Float32(max_surface_dist_mm),
        ndrange=crop_dims
    )
    KernelAbstractions.synchronize(backend)
    
    # 3. Calculate Marrow sphere radius R_L and centroid
    voxel_vol = sp_x * sp_y * sp_z
    lesion_vol_mm3 = length(lesion_indices) * voxel_vol
    R_L = max(4.0f0, Float32((3.0 * lesion_vol_mm3 / (4.0 * pi))^(1.0 / 3.0)))
    
    # Find closest marrow point to lesion centroid
    lesion_cx = mean(xs) - x_min + 1
    lesion_cy = mean(ys) - y_min + 1
    lesion_cz = mean(zs) - z_min + 1
    
    marrow_pts = findall(crop_marrow .& .!crop_lesion)
    
    if has_cuda
        crop_marrow_res = CUDA.zeros(Bool, crop_dims)
        crop_marrow_gpu = CUDA.CuArray(crop_marrow)
    else
        crop_marrow_res = zeros(Bool, crop_dims)
        crop_marrow_gpu = crop_marrow
    end
    
    if !isempty(marrow_pts)
        min_d = 1e9
        best_pt = marrow_pts[1]
        for p in marrow_pts
            dx = (Float32(p[1]) - lesion_cx) * sp_x
            dy = (Float32(p[2]) - lesion_cy) * sp_y
            dz = (Float32(p[3]) - lesion_cz) * sp_z
            d = dx*dx + dy*dy + dz*dz
            if d < min_d
                min_d = d
                best_pt = p
            end
        end
        
        m_cx = Float32(best_pt[1])
        m_cy = Float32(best_pt[2])
        m_cz = Float32(best_pt[3])
        
        kernel_marrow = kernel_bone_marrow!(backend)
        kernel_marrow(
            crop_marrow_res,
            crop_marrow_gpu,
            crop_lesion_gpu,
            m_cx, m_cy, m_cz,
            sp_x, sp_y, sp_z,
            R_L,
            ndrange=crop_dims
        )
        KernelAbstractions.synchronize(backend)
    end
    
    # Move back to CPU if needed
    crop_surface_cpu = has_cuda ? Array(crop_surface) : crop_surface
    crop_marrow_res_cpu = has_cuda ? Array(crop_marrow_res) : crop_marrow_res
    
    # STRICT BONE CONSTRAINT: Enforce bone boundary intersection so no voxels leak outside bone
    surface_mask[x_min:x_max, y_min:y_max, z_min:z_max] .= crop_surface_cpu .& crop_bone_bool
    marrow_mask[x_min:x_max, y_min:y_max, z_min:z_max] .= crop_marrow_res_cpu .& crop_bone_bool
    
    # Global safety check
    surface_mask[bone_vol .<= 0] .= false
    marrow_mask[bone_vol .<= 0] .= false
    
    return surface_mask, marrow_mask
end

end # module
