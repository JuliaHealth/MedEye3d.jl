using NIfTI
using MedImages

img_path = "/root/data_decathlon/Task09_Spleen/imagesTr/spleen_10.nii.gz"
mask_path = "/root/data_decathlon/Task09_Spleen/labelsTr/spleen_10.nii.gz"

med_mask = MedImages.Load_and_save.load_image(mask_path, "")
vol_mask = Float32.(med_mask.voxel_data)

println("Max overall: ", maximum(vol_mask))
println("Min overall: ", minimum(vol_mask))

z_center = size(vol_mask, 3) ÷ 2
y_center = size(vol_mask, 2) ÷ 2
x_center = size(vol_mask, 1) ÷ 2

println("Max at z_center ($z_center): ", maximum(vol_mask[:, :, z_center]))
println("Max at y_center ($y_center): ", maximum(vol_mask[:, y_center, :]))
println("Max at x_center ($x_center): ", maximum(vol_mask[x_center, :, :]))

count_z = sum(vol_mask[:, :, z_center] .> 0)
count_y = sum(vol_mask[:, y_center, :] .> 0)
count_x = sum(vol_mask[x_center, :, :] .> 0)

println("Count at z: ", count_z)
println("Count at y: ", count_y)
println("Count at x: ", count_x)

