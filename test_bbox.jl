using Pkg
Pkg.activate(".")
using MedImages
using HDF5

data_dir = "data/pat_6_files"
baseline_ct = MedImages.load_image(joinpath(data_dir, "Fixed_CT_Volume_0.nii.gz"), "CT")

h5_path = joinpath(data_dir, "preprocessed_volumes.h5")
h5_read = h5open(h5_path, "r")
mask_vol = read(h5_read["TFM_Transform_SPECT_to_Baseline_0.tfm/SPECT_Lesions_0.nii.gz"])

skelly_img = MedImages.load_image(joinpath(data_dir, "anatomy_out_spect_ct_0/SPECT_CT_Volume_0_medium_postprocessed_subseg_postprocessed.nii.gz"), "CT")

include("scripts/lib/SceneHierarchy.jl")
using .SceneHierarchy
T_ITK = parse_tfm(joinpath(data_dir, "Transform_SPECT_to_Baseline_0.tfm"))
skelly_img = apply_transform_to_medimage(skelly_img, T_ITK)
skelly_resampled = MedImages.resample_to_image(baseline_ct, skelly_img, MedImages.Nearest_neighbour_en)
skelly_vox = Float32.(skelly_resampled.voxel_data)

# Bounding box of lesion 2 in mask_vol
pts = findall(mask_vol .== 2)
if !isempty(pts)
    min_x, max_x = minimum(p[1] for p in pts), maximum(p[1] for p in pts)
    min_y, max_y = minimum(p[2] for p in pts), maximum(p[2] for p in pts)
    min_z, max_z = minimum(p[3] for p in pts), maximum(p[3] for p in pts)
    println("Lesion 2 bbox: X=($min_x, $max_x), Y=($min_y, $max_y), Z=($min_z, $max_z)")
else
    println("Lesion 2 empty")
end

# Check skelly near that box
skelly_near = count(skelly_vox[max(1, min_x-10):min(512, max_x+10), max(1, min_y-10):min(512, max_y+10), max(1, min_z-10):min(326, max_z+10)] .> 0)
println("Skelly voxels near lesion 2 (±10): ", skelly_near)

# Skelly global bbox
s_pts = findall(skelly_vox .> 0)
if !isempty(s_pts)
    smin_x, smax_x = minimum(p[1] for p in s_pts), maximum(p[1] for p in s_pts)
    smin_y, smax_y = minimum(p[2] for p in s_pts), maximum(p[2] for p in s_pts)
    smin_z, smax_z = minimum(p[3] for p in s_pts), maximum(p[3] for p in s_pts)
    println("Skelly global bbox: X=($smin_x, $smax_x), Y=($smin_y, $smax_y), Z=($smin_z, $smax_z)")
end
