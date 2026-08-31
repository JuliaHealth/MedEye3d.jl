using Pkg
Pkg.activate("/workspaces/MedEye3d.jl")

using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using MedEye3d.MakieEvents  # Import event types
using Accessors
using ColorTypes
using GLFW
using NIfTI

println("=== COMPREHENSIVE FUNCTIONALITY TEST ===")
println("Loading real medical data...")

image_path = "/root/data_decathlon/Task09_Spleen/imagesTr/spleen_10.nii.gz"
mask_path = "/root/data_decathlon/Task09_Spleen/labelsTr/spleen_10.nii.gz"

nii_img = niread(image_path)
nii_mask = niread(mask_path)
vol_img = Float32.(nii_img.raw)
vol_mask = Float32.(nii_mask.raw)

println("Volume size: ", size(vol_img))

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
mainInst = MedEye3d.SegmentationDisplay.coordinateDisplay(
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
MedEye3d.SegmentationDisplay.passDataForScrolling(mainInst, ScrollDat)

sleep(1)

ctx = mainInst.states[1].mainForDisplayObjects.vulkanCtx
state = mainInst.states[1]

# ========================================================
# TEST 1: Normal scroll to spleen slice
# ========================================================
println("\n--- TEST 1: Scroll to spleen slice 31 ---")
put!(mainInst.channel, 30)
sleep(3)

# Check what data was uploaded
sdd = state.currentlyDispDat
println("Current slice: ", sdd.sliceNumber)
println("Textures in currentlyDispDat: ", [d.name for d in sdd.listOfDataAndImageNames])
for d in sdd.listOfDataAndImageNames
    dat = d.dat
    println("  $(d.name): nonzero=$(count(dat .!= 0)), min=$(minimum(dat)), max=$(maximum(dat))")
end

MedEye3d.VulkanBackend.VulkanScreenshot.capture_screenshot(ctx, "/workspaces/MedEye3d.jl/t1_spleen.ppm", image_index=0)

# ========================================================
# TEST 2: Zoom (ScrollZoomEvent)
# ========================================================
println("\n--- TEST 2: Zoom In ---")
for i in 1:7
    put!(mainInst.channel, ScrollZoomEvent(2.0))
    sleep(0.3)
end
put!(mainInst.channel, 0)  # trigger re-render
sleep(2)

println("Zoom level: ", state.calcDimsStruct.zoom)
MedEye3d.VulkanBackend.VulkanScreenshot.capture_screenshot(ctx, "/workspaces/MedEye3d.jl/t2_zoomed.ppm", image_index=0)

# ========================================================
# TEST 3: Pan
# ========================================================
println("\n--- TEST 3: Pan ---")
state.calcDimsStruct.panX = 0.15f0
state.calcDimsStruct.panY = -0.1f0
put!(mainInst.channel, 0)
sleep(2)

println("Pan: ($(state.calcDimsStruct.panX), $(state.calcDimsStruct.panY))")
MedEye3d.VulkanBackend.VulkanScreenshot.capture_screenshot(ctx, "/workspaces/MedEye3d.jl/t3_panned.ppm", image_index=0)

# Reset
state.calcDimsStruct.zoom = 1.0f0
state.calcDimsStruct.panX = 0.0f0
state.calcDimsStruct.panY = 0.0f0

# ========================================================
# TEST 4: Multiple slices (verify anatomy changes)
# ========================================================
println("\n--- TEST 4: Different slices ---")
put!(mainInst.channel, 15)
sleep(2)
println("Slice 4a: ", state.currentDisplayedSlice)
MedEye3d.VulkanBackend.VulkanScreenshot.capture_screenshot(ctx, "/workspaces/MedEye3d.jl/t4a_upper.ppm", image_index=0)

put!(mainInst.channel, -30)
sleep(2)
println("Slice 4b: ", state.currentDisplayedSlice)
MedEye3d.VulkanBackend.VulkanScreenshot.capture_screenshot(ctx, "/workspaces/MedEye3d.jl/t4b_lower.ppm", image_index=0)

# ========================================================
# TEST 5: Bone window
# ========================================================
println("\n--- TEST 5: Bone window ---")
# Switch to bone window
specs = state.mainForDisplayObjects.listOfTextSpecifications
specs[1] = setproperties(specs[1], (minAndMaxValue=Float32.([-700, 1300]),))
state.mainForDisplayObjects.listOfTextSpecifications[1] = specs[1]

put!(mainInst.channel, 20)
sleep(2)
MedEye3d.VulkanBackend.VulkanScreenshot.capture_screenshot(ctx, "/workspaces/MedEye3d.jl/t5_bone_window.ppm", image_index=0)

# Reset to soft tissue window
specs[1] = setproperties(specs[1], (minAndMaxValue=Float32.([-150, 250]),))
state.mainForDisplayObjects.listOfTextSpecifications[1] = specs[1]

# ========================================================
# TEST 6: Ctrl+scroll blend
# ========================================================
println("\n--- TEST 6: PET/Mask Blend ---")
put!(mainInst.channel, -10)
sleep(1)
put!(mainInst.channel, PetBlendEvent(0.9f0))
sleep(1)
put!(mainInst.channel, 0)
sleep(2)
MedEye3d.VulkanBackend.VulkanScreenshot.capture_screenshot(ctx, "/workspaces/MedEye3d.jl/t6_blend.ppm", image_index=0)

println("\n=== ALL TESTS COMPLETE ===")
