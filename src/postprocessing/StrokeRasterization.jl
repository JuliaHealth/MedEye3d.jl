"""
Module for continuous thick-line (swept capsule / disk) rasterization
using KernelAbstractions.jl on CPU and GPU (CUDA).
Interpolates between consecutive sampled mouse drag coordinates to guarantee
gapless strokes at any drawing speed and brush thickness.
"""
module StrokeRasterization

using KernelAbstractions
using CUDA

export rasterize_thick_line!, rasterize_polyline!

"""
    rasterize_thick_segment_kernel!(out_mask, p1_x, p1_y, p2_x, p2_y, radius_sq, val, min_x, min_y, max_x, max_y)

KernelAbstractions 2D kernel that evaluates the Euclidean distance from each pixel (x, y)
within the local segment bounding box to the line segment [P1, P2].
If dist^2 <= radius_sq, sets out_mask[x, y] = val.
"""
@kernel function rasterize_thick_segment_kernel!(
    out_mask,
    @Const(p1_x), @Const(p1_y),
    @Const(p2_x), @Const(p2_y),
    @Const(radius_sq),
    @Const(val),
    @Const(min_x), @Const(min_y),
    @Const(max_x), @Const(max_y)
)
    i, j = @index(Global, NTuple)
    x = min_x + i - 1
    y = min_y + j - 1

    if x <= max_x && y <= max_y && x >= 1 && x <= size(out_mask, 1) && y >= 1 && y <= size(out_mask, 2)
        dx = Float32(p2_x - p1_x)
        dy = Float32(p2_y - p1_y)
        l2 = dx * dx + dy * dy

        ux = Float32(x - p1_x)
        uy = Float32(y - p1_y)

        # Projection factor along segment
        t = l2 > 0.0f0 ? clamp((ux * dx + uy * dy) / l2, 0.0f0, 1.0f0) : 0.0f0
        proj_x = Float32(p1_x) + t * dx
        proj_y = Float32(p1_y) + t * dy

        dist_sq = (Float32(x) - proj_x)^2 + (Float32(y) - proj_y)^2
        if dist_sq <= radius_sq
            @inbounds out_mask[x, y] = val
        end
    end
end

"""
    rasterize_thick_line!(mask, p1, p2, radius, val; use_gpu=false)

Rasterizes a continuous thick line segment from `p1` to `p2` with radius `radius`
setting overlapping pixels to `val`.
Works on 2D Arrays and SubArrays (views).
"""
function rasterize_thick_line!(
    mask::AbstractMatrix{T},
    p1::Tuple{Int,Int},
    p2::Tuple{Int,Int},
    radius::Int,
    val::T;
    use_gpu::Bool = false
) where T
    rad = max(0, radius)
    min_x = max(1, min(p1[1], p2[1]) - rad)
    max_x = min(size(mask, 1), max(p1[1], p2[1]) + rad)
    min_y = max(1, min(p1[2], p2[2]) - rad)
    max_y = min(size(mask, 2), max(p1[2], p2[2]) + rad)

    w = max_x - min_x + 1
    h = max_y - min_y + 1
    if w <= 0 || h <= 0
        return mask
    end

    backend = (use_gpu && CUDA.functional() && (mask isa CuArray)) ? CUDABackend() : CPU()
    kernel! = rasterize_thick_segment_kernel!(backend)

    radius_sq = Float32(rad * rad)
    kernel!(mask, p1[1], p1[2], p2[1], p2[2], radius_sq, val, min_x, min_y, max_x, max_y, ndrange=(w, h))
    KernelAbstractions.synchronize(backend)
    return mask
end

"""
    rasterize_polyline!(mask, points, radius, val; use_gpu=false)

Rasterizes a continuous polyline connecting all consecutive coordinates in `points`
with thick line segments of radius `radius`. If only 1 point is given, rasterizes a circle.
"""
function rasterize_polyline!(
    mask::AbstractMatrix{T},
    points::Vector{Tuple{Int,Int}},
    radius::Int,
    val::T;
    use_gpu::Bool = false
) where T
    if isempty(points)
        return mask
    elseif length(points) == 1
        return rasterize_thick_line!(mask, points[1], points[1], radius, val; use_gpu=use_gpu)
    end

    for i in 1:length(points)-1
        rasterize_thick_line!(mask, points[i], points[i+1], radius, val; use_gpu=use_gpu)
    end
    return mask
end

end # module StrokeRasterization
