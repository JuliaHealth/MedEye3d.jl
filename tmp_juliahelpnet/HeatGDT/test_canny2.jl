using Images, ImageEdgeDetection
using BenchmarkTools

function canny3d(vol)
    out = zeros(Float32, size(vol))
    for z in 1:size(vol, 3)
        slice = vol[:, :, z]
        # normalize slice to 0-1
        mi, ma = minimum(slice), maximum(slice)
        if ma > mi
            slice = (slice .- mi) ./ (ma - mi)
        end
        c = canny(slice, (Percentile(80), Percentile(20)), 2)
        out[:, :, z] .= c
    end
    return out
end

vol = rand(Float32, 64, 64, 64)
@btime canny3d($vol)
