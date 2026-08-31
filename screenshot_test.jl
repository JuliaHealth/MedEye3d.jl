using Pkg
Pkg.activate("/workspaces/MedEye3d.jl")

using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using ColorTypes
using GLFW

println("Creating synthetic data...")
vol_img = Float32.(zeros(100, 100, 100))
vol_mask = Float32.(zeros(100, 100, 100))

# Draw a sphere in the middle
for i in 1:100, j in 1:100, k in 1:100
    if (i-50)^2 + (j-50)^2 + (k-50)^2 < 400
        vol_img[i, j, k] = 100.0
        if i > 50
            vol_mask[i, j, k] = 1.0
        end
    end
end

textureSpec_img = TextureSpec{Float32}(
    name="MainImage",
    isMainImage=true,
    color=RGB(1.0, 1.0, 1.0),
    minAndMaxValue=Float32.([-100, 200])
)
textureSpec_mask = TextureSpec{Float32}(
    name="Mask",
    isMainImage=false,
    color=RGB(1.0, 0.0, 0.0),
    maskContribution=Float32(0.5)
)
textureSpecArray = Vector{Vector{TextureSpec}}([
    TextureSpec[deepcopy(textureSpec_img), deepcopy(textureSpec_mask)]
])
voxelDataTupleVector = Vector{Vector{Any}}([
    Any[("MainImage", vol_img), ("Mask", vol_mask)]
])

spacing = (1.0, 1.0, 1.0)
origin = (0.0, 0.0, 0.0)

dataToScroll = Vector{DataToScrollDims}()
push!(dataToScroll, DataToScrollDims(imageSize=size(vol_img), voxelSize=spacing, dimensionToScroll=3))

println("Starting display...")
mainMedEye3dInstance = MedEye3d.SegmentationDisplay.coordinateDisplay(textureSpecArray, 1.0, dataToScroll, spacing, origin)

dataToScrollList = [
    ThreeDimRawDat{Float32}(type=Float32, name="MainImage", dat=vol_img),
    ThreeDimRawDat{Float32}(type=Float32, name="Mask", dat=vol_mask)
]
ScrollDat = FullScrollableDat(dataToScroll=dataToScrollList, dimensionToScroll=3)

mainMedEye3dInstance.states[1].onScrollData = ScrollDat
mainMedEye3dInstance.states[1].currentDisplayedSlice = 50

# trigger initial render
MedEye3d.ReactToScroll.reactToScroll(0, mainMedEye3dInstance.states, false)

println("Waiting for render loop...")
sleep(2)

println("Capturing screenshot...")
# trigger Vulkan screenshot directly
ctx = mainMedEye3dInstance.states[1].mainForDisplayObjects.vulkanCtx
MedEye3d.VulkanBackend.VulkanScreenshot.capture_screenshot(ctx, "/workspaces/MedEye3d.jl/screenshot1.png")

println("Done.")
