using Pkg
Pkg.activate(".")
using MedImages
using HDF5

data_dir = "data/pat_6_files"
baseline_ct = MedImages.load_image(joinpath(data_dir, "Fixed_CT_Volume_0.nii.gz"), "CT")

h5_path = joinpath(data_dir, "preprocessed_volumes.h5")
h5_read = h5open(h5_path, "r")
mask_vol = read(h5_read["TFM_Transform_SPECT_to_Baseline_0.tfm/SPECT_Lesions_0.nii.gz"])
println("mask_vol shape: ", size(mask_vol), " sum>0: ", count(mask_vol .> 0))

skelly_img = MedImages.load_image(joinpath(data_dir, "anatomy_out_spect_ct_0/SPECT_CT_Volume_0_medium_postprocessed_subseg_postprocessed.nii.gz"), "CT")
println("skelly_img orig shape: ", size(skelly_img.voxel_data))

include("scripts/lib/SceneHierarchy.jl")
using .SceneHierarchy
T_ITK = parse_tfm(joinpath(data_dir, "Transform_SPECT_to_Baseline_0.tfm"))
skelly_img = apply_transform_to_medimage(skelly_img, T_ITK)

# Resample using MedImages CPU resampling just for testing
skelly_resampled = MedImages.resample_to_image(baseline_ct, skelly_img, MedImages.Nearest_neighbour_en)
skelly_vox = Float32.(skelly_resampled.voxel_data)
println("skelly_vox shape: ", size(skelly_vox), " sum>0: ", count(skelly_vox .> 0))

# Overlap for lesion 2
bin_mask = (mask_vol .== 2)
skelly_overlap = count(bin_mask .& (skelly_vox .> 0))
println("Lesion 2 overlap: ", skelly_overlap, " / ", count(bin_mask))
