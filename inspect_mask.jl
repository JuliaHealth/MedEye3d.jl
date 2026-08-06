using MedImages
nifti_dir = joinpath(@__DIR__, "data", "pat6_pet_only_debug", "nifti_extracted")
seg_1_raw = MedImages.load_image(joinpath(nifti_dir, "NNUNET_OR_Lesions_CT_0_artficials_added.nii.gz"), "CT")
ct_1 = MedImages.load_image(joinpath(nifti_dir, "Fixed_CT_Volume_0.nii.gz"), "CT")
seg_1 = MedImages.resample_to_image(ct_1, seg_1_raw, MedImages.Nearest_neighbour_en)
mask_vol = Float32.(seg_1.voxel_data)
println(unique(mask_vol))
