"""
Bilateral splitting kernel for the GPU DAG pipeline.
Splits a mask into left/right halves based on a midline reference (spine).
"""

using KernelAbstractions
using KernelAbstractions: @index

@kernel function bilateral_split_kernel!(left_out, right_out, input, midline_x::Int32)
    I, J, K = @index(Global, NTuple)
    val = input[I, J, K]
    # In LPS: increasing X = Left
    # Left keeps X >= midline, Right keeps X < midline
    if I >= midline_x
        left_out[I, J, K] = val
        right_out[I, J, K] = UInt8(0)
    else
        left_out[I, J, K] = UInt8(0)
        right_out[I, J, K] = val
    end
end

@kernel function bilateral_split_per_slice_kernel!(left_out, right_out, input, midline_per_z)
    I, J, K = @index(Global, NTuple)
    val = input[I, J, K]
    mid = midline_per_z[K]
    if I >= mid
        left_out[I, J, K] = val
        right_out[I, J, K] = UInt8(0)
    else
        left_out[I, J, K] = UInt8(0)
        right_out[I, J, K] = val
    end
end

"""
    find_midline_x(spine_arr)

Find the median X index of a spine mask to use as the bilateral midline.
Returns a single integer index.
"""
function find_midline_x(spine_arr::AbstractArray{UInt8,3})
    dims = size(spine_arr)
    sum_x = 0
    count = 0
    for K in 1:dims[3], J in 1:dims[2], I in 1:dims[1]
        if spine_arr[I, J, K] > 0
            sum_x += I
            count += 1
        end
    end
    if count > 0
        return round(Int, sum_x / count)
    else
        return dims[1] ÷ 2  # fallback to image center
    end
end

"""
    find_midline_per_slice(landmark_arr, default_midline)

For per-slice bilateral splitting (e.g., using trachea, esophagus, sternum).
Returns a Vector{Int32} of per-Z-slice midline X indices.
"""
function find_midline_per_slice(landmark_arr::AbstractArray{UInt8,3}, default_midline::Int)
    dims = size(landmark_arr)
    midlines = fill(Int32(default_midline), dims[3])
    
    for K in 1:dims[3]
        sum_x = 0
        count = 0
        for J in 1:dims[2], I in 1:dims[1]
            if landmark_arr[I, J, K] > 0
                sum_x += I
                count += 1
            end
        end
        if count > 0
            midlines[K] = Int32(round(Int, sum_x / count))
        end
    end
    
    return midlines
end

"""
    split_bilateral!(backend, tm, in_ch, spine_ch, out_left_ch, out_right_ch; 
                     split_landmark_ch=0, split_type="spine")

Split a bilateral mask into left and right channels.
- spine_ch: channel with the spine/fused_spine mask for default midline
- split_landmark_ch: optional channel for per-slice splitting (e.g., trachea for Retrotracheal)
- split_type: "spine" (global midline) or "per_slice" (per-Z-slice midline from split_landmark)
"""
function split_bilateral!(backend, tm, in_ch::Int, spine_ch::Int, 
                          out_left_ch::Int, out_right_ch::Int;
                          split_landmark_ch::Int=0, split_type::String="spine")
    in_arr = get_channel_view(tm, in_ch)
    dims = size(in_arr)
    
    # Get or allocate output channels
    left_arr = get_channel_view(tm, out_left_ch)
    right_arr = get_channel_view(tm, out_right_ch)
    
    spine_arr = get_channel_view(tm, spine_ch)
    default_midline = find_midline_x(spine_arr)
    
    # CPU-based bilateral splitting (TensorManager stores CPU arrays)
    if split_type == "per_slice" && split_landmark_ch > 0
        lm_arr = get_channel_view(tm, split_landmark_ch)
        midlines = find_midline_per_slice(lm_arr, default_midline)
        
        Threads.@threads for K in 1:dims[3]
            mid = midlines[K]
            for J in 1:dims[2], I in 1:dims[1]
                val = in_arr[I, J, K]
                if I >= mid
                    left_arr[I, J, K] = val
                    right_arr[I, J, K] = UInt8(0)
                else
                    left_arr[I, J, K] = UInt8(0)
                    right_arr[I, J, K] = val
                end
            end
        end
    else
        mid = Int(default_midline)
        Threads.@threads for K in 1:dims[3]
            for J in 1:dims[2], I in 1:dims[1]
                val = in_arr[I, J, K]
                if I >= mid
                    left_arr[I, J, K] = val
                    right_arr[I, J, K] = UInt8(0)
                else
                    left_arr[I, J, K] = UInt8(0)
                    right_arr[I, J, K] = val
                end
            end
        end
    end
    
    println("  -> Bilateral split done: midline_x=$default_midline")
end
