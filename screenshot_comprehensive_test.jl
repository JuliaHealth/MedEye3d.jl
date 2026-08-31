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

println("Volume size: ", size(vol_img))

# Find slices with mask
for z in 1:size(vol_mask, 3)
    if any(vol_mask[:,:,z] .> 0)
        println("Mask present at slice $z, count: ", count(vol_mask[:,:,z] .> 0))
    end
end

textureSpec_img = TextureSpec{Float32}(
    name="MainImage",
    isMainImage=true,
    color=RGB(1.0, 1.0, 1.0),
    minAndMaxValue=Float32.([-150, 250])
)
textureSpec_mask = TextureSpec{Float32}(
    name="Mask",
    isMainImage=false,
    isNuclearMask=false,
    color=RGB(1.0, 0.0, 0.0),
    maskContribution=Float32(0.8),
    minAndMaxValue=Float32.([0.5, 1.5])
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
put!(mainMedEye3dInstance.channel, 45)

println("Waiting for render at spleen slice ~46...")
sleep(4)

println("Capturing spleen slice screenshot...")
ctx = mainMedEye3dInstance.states[1].mainForDisplayObjects.vulkanCtx
MedEye3d.VulkanBackend.VulkanScreenshot.capture_screenshot(ctx, "/workspaces/MedEye3d.jl/screenshot_spleen.ppm", image_index=0)

# Scroll to upper slice
put!(mainMedEye3dInstance.channel, 25)
sleep(2)

println("Capturing upper abdomen screenshot...")
MedEye3d.VulkanBackend.VulkanScreenshot.capture_screenshot(ctx, "/workspaces/MedEye3d.jl/screenshot_upper.ppm", image_index=0)

# Scroll to lower slice
put!(mainMedEye3dInstance.channel, -60)
sleep(2)

println("Capturing lower slice screenshot...")
MedEye3d.VulkanBackend.VulkanScreenshot.capture_screenshot(ctx, "/workspaces/MedEye3d.jl/screenshot_lower.ppm", image_index=0)

println("All screenshots captured!")
