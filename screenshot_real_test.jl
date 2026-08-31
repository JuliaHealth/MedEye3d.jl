using Pkg
Pkg.activate("/workspaces/MedEye3d.jl")

using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using ColorTypes
using GLFW
using NIfTI

println("Loading real medical data...")
image_path = "/root/data_decathlon/Task09_Spleen/imagesTr/spleen_10.nii.gz"
mask_path = "/root/data_decathlon/Task09_Spleen/labelsTr/spleen_10.nii.gz"

nii_img = niread(image_path)
nii_mask = niread(mask_path)
vol_img = Float32.(nii_img.raw)
vol_mask = Float32.(nii_mask.raw)

textureSpec_img = TextureSpec{Float32}(
    name="MainImage",
    isMainImage=true,
    color=RGB(1.0, 1.0, 1.0),
    minAndMaxValue=Float32.([-150, 250])
)
textureSpec_mask = TextureSpec{Float32}(
    name="Mask",
    isMainImage=false,
    color=RGB(1.0, 0.0, 0.0),
    maskContribution=Float32(0.6)
)
textureSpecArray = Vector{Vector{TextureSpec}}([
    TextureSpec[deepcopy(textureSpec_img), deepcopy(textureSpec_mask)]
])

spacing = (1.0, 1.0, 1.0)
origin = (0.0, 0.0, 0.0)

dataToScroll = Vector{DataToScrollDims}()
push!(dataToScroll, DataToScrollDims(imageSize=size(vol_img), voxelSize=spacing, dimensionToScroll=3))

println("Starting display...")
mainMedEye3dInstance = MedEye3d.SegmentationDisplay.coordinateDisplay(
    textureSpecArray, 
    Float32(1.0), 
    dataToScroll, 
    [spacing], 
    [origin],
    Dict{String,Vector}("supervoxel_vertices" => [], "supervoxel_indices" => []),
    Dict{Int64, Dict{Int64, Dict{String, Any}}}()
)

dataToScrollList = [
    ThreeDimRawDat{Float32}(type=Float32, name="MainImage", dat=vol_img),
    ThreeDimRawDat{Float32}(type=Float32, name="Mask", dat=vol_mask)
]
ScrollDat = FullScrollableDat(dataToScroll=dataToScrollList, dimensionToScroll=3)

MedEye3d.SegmentationDisplay.passDataForScrolling(mainMedEye3dInstance, ScrollDat)

sleep(1)
put!(mainMedEye3dInstance.channel, 5)

println("Waiting for render loop...")
sleep(4)

println("Capturing screenshot...")
ctx = mainMedEye3dInstance.states[1].mainForDisplayObjects.vulkanCtx
MedEye3d.VulkanBackend.VulkanScreenshot.capture_screenshot(ctx, "/workspaces/MedEye3d.jl/screenshot_real_0.ppm", image_index=0)
MedEye3d.VulkanBackend.VulkanScreenshot.capture_screenshot(ctx, "/workspaces/MedEye3d.jl/screenshot_real_1.ppm", image_index=1)
MedEye3d.VulkanBackend.VulkanScreenshot.capture_screenshot(ctx, "/workspaces/MedEye3d.jl/screenshot_real_2.ppm", image_index=2)

println("Done.")
