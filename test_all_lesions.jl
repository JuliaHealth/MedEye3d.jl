using Pkg
Pkg.activate(".")
using HDF5
h5_path = "data/pat_6_files/preprocessed_volumes.h5"
h5_read = h5open(h5_path, "r")
mask_vol = read(h5_read["TFM_Transform_SPECT_to_Baseline_0.tfm/SPECT_Lesions_0.nii.gz"])
for lid in unique(mask_vol)
    lid == 0 && continue
    pts = findall(mask_vol .== lid)
    min_z, max_z = minimum(p[3] for p in pts), maximum(p[3] for p in pts)
    println("Lesion $lid Z=($min_z, $max_z)")
end
