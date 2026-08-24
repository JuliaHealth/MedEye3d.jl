import HDF5
h5 = HDF5.h5open("data/pat_6_files/preprocessed_volumes.h5", "r")
ct0 = read(h5["BASELINE/SPECT_CT_Volume_0.nii.gz"])
ct1 = read(h5["TFM_Transform_SPECT_to_Baseline_1.tfm/SPECT_CT_Volume_1.nii.gz"])
println("Size ct0: ", size(ct0))
println("Size ct1: ", size(ct1))

# Let's extract a 1D profile along the Z axis (spine) or Y axis (body)
# Find the center of mass or simply a line through the middle
cx, cy = div(size(ct0, 1), 2), div(size(ct0, 2), 2)
profile0 = ct0[cx, cy, :]
profile1 = ct1[cx, cy, :]

# Calculate cross-correlation to find the shift
using Statistics
function xcorr(a, b)
    n = length(a)
    lags = -50:50
    corrs = Float64[]
    for lag in lags
        sum_ab = 0.0
        count = 0
        for i in 1:n
            j = i + lag
            if 1 <= j <= n
                sum_ab += a[i] * b[j]
                count += 1
            end
        end
        push!(corrs, count > 0 ? sum_ab / count : 0.0)
    end
    return lags[argmax(corrs)]
end

shift_z = xcorr(profile0, profile1)
println("Estimated Z shift (in voxels): ", shift_z)

# Let's do Y shift
cz = div(size(ct0, 3), 2)
profile0_y = ct0[cx, :, cz]
profile1_y = ct1[cx, :, cz]
shift_y = xcorr(profile0_y, profile1_y)
println("Estimated Y shift (in voxels): ", shift_y)

close(h5)
