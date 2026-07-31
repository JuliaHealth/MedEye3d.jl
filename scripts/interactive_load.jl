using Pkg
Pkg.activate(".")
using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using GLMakie
using MedImages
using ColorTypes

# Set up paths
img_path = "/root/data_decathlon/Task09_Spleen/imagesTr/spleen_10.nii.gz"
mask_path = "/root/data_decathlon/Task09_Spleen/labelsTr/spleen_10.nii.gz"

println("Loading image from $img_path")
med_img = MedImages.Load_and_save.load_image(img_path, "")
vol_img = Float32.(med_img.voxel_data)
println("Image loaded! Shape: ", size(vol_img), " Max value: ", maximum(vol_img), " Min value: ", minimum(vol_img))

println("Loading mask from $mask_path")
med_mask = MedImages.Load_and_save.load_image(mask_path, "")
vol_mask = Float32.(med_mask.voxel_data)
println("Mask loaded! Shape: ", size(vol_mask))

# We want Transverse (Axial) view. We will scroll along Z (the 3rd dimension).
textureSpec_img = TextureSpec{Float32}(
    name="MainImage",
    isMainImage=true,
    color=RGB(1.0, 1.0, 1.0),
    minAndMaxValue=Float32.([-100, 200]) # Windowing
)

textureSpec_mask = TextureSpec{Float32}(
    name="Mask",
    isMainImage=false,
    color=RGB(1.0, 0.0, 0.0),
    minAndMaxValue=Float32.([0, 1])
)

voxelDataTupleVector = Any[
    ("MainImage", vol_img),
    ("Mask", vol_mask)
]

println("Starting visualizer...")
dummyStudySrc = Tuple{String,String}()

mainMedEye3dInstance = MedEye3d.SegmentationDisplay.displayImage(
    dummyStudySrc;
    voxelDataTupleVector=voxelDataTupleVector,
    textureSpecArray = Vector{TextureSpec}([textureSpec_img, textureSpec_mask]),
    origins = Vector{Tuple{Float64,Float64,Float64}}([(0.0, 0.0, 0.0)]),
    windowWidth=1000,
    fractionOfMainImage=Float32(1.0),
    quadView=false,
    dimensionsToScroll=3
)

println("Visualizer is running! Press Ctrl+C in terminal or close window to exit.")

import GLFW
while !GLFW.WindowShouldClose(mainMedEye3dInstance.window)
    GLFW.PollEvents()
    sleep(0.1)
end
