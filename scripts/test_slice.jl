using Pkg
Pkg.instantiate()
using MedImages

img_path = "/root/data_decathlon/Task09_Spleen/imagesTr/spleen_10.nii.gz"
mask_path = "/root/data_decathlon/Task09_Spleen/labelsTr/spleen_10.nii.gz"

med_img = MedImages.Load_and_save.load_image(img_path, "")
vol_img = Float32.(med_img.voxel_data)
med_mask = MedImages.Load_and_save.load_image(mask_path, "")
vol_mask = Float32.(med_mask.voxel_data)

z_center = round(Int, size(vol_img, 3) / 2)
println("Z center is: ", z_center)
println("Max img at Z center: ", maximum(vol_img[:, :, z_center]))
println("Max mask at Z center: ", maximum(vol_mask[:, :, z_center]))

x_center = round(Int, size(vol_img, 1) / 2)
println("X center is: ", x_center)
println("Max img at X center: ", maximum(vol_img[x_center, :, :]))
println("Max mask at X center: ", maximum(vol_mask[x_center, :, :]))
