using Pkg
Pkg.instantiate()
using MedImages
using Statistics

function main()
    med_img = MedImages.Load_and_save.load_image("/root/data_decathlon/Task09_Spleen/imagesTr/spleen_10.nii.gz", "")
    vol_img = Float32.(med_img.voxel_data)
    
    axial_slice = vol_img[:, :, 27]
    println("Axial size: ", size(axial_slice))
    println("Axial Min: ", minimum(axial_slice))
    println("Axial Max: ", maximum(axial_slice))
    println("Axial Mean: ", mean(axial_slice))
    
    coronal_vol = permutedims(vol_img, (1, 3, 2))
    coronal_slice = coronal_vol[:, :, 256]
    println("Coronal size: ", size(coronal_slice))
    println("Coronal Min: ", minimum(coronal_slice))
    println("Coronal Max: ", maximum(coronal_slice))
    println("Coronal Mean: ", mean(coronal_slice))
end

main()
