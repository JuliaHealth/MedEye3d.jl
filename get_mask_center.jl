using Pkg
Pkg.activate(".")
using MedImages

mask_path = "/root/data_decathlon/Task09_Spleen/labelsTr/spleen_10.nii.gz"
med_mask = MedImages.Load_and_save.load_image(mask_path, "")
vol_mask = Float32.(med_mask.voxel_data)

indices = findall(vol_mask .> 0)
x_sum = sum(getindex.(indices, 1))
y_sum = sum(getindex.(indices, 2))
z_sum = sum(getindex.(indices, 3))
count = length(indices)

println("Mask center X: ", round(Int, x_sum / count))
println("Mask center Y: ", round(Int, y_sum / count))
println("Mask center Z: ", round(Int, z_sum / count))

