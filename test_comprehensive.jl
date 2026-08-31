using Pkg
Pkg.activate("/workspaces/MedEye3d.jl")

using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
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

# Create a synthetic multi-label mask from the spleen mask
# Label 1 = spleen (original), Label 2 = synthetic bone-like region (high HU), Label 3 = another region
vol_multilabel = copy(vol_mask)
# Add label 2 where HU > 200 (bone-like)
bone_mask = vol_img .> 200.0f0
vol_multilabel[bone_mask] .= 2.0f0
# Keep spleen as label 1 (override bone in spleen area)
vol_multilabel[vol_mask .> 0] .= 1.0f0

println("Volume size: ", size(vol_img))
println("Multi-label unique: ", sort(unique(vol_multilabel))[1:min(end,10)])

# ========================================================
# TEST 1: Single panel with mask overlay + scroll to spleen
# ========================================================
println("\n--- TEST 1: Single Panel + Mask Overlay ---")

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

# Scroll to spleen slice (slice 31 has max mask coverage)
put!(mainInst.channel, 30)  # scroll from 1 to ~31
sleep(3)

ctx = mainInst.states[1].mainForDisplayObjects.vulkanCtx

# Screenshot 1: Spleen slice with mask overlay
println("Screenshot 1: Spleen slice with mask at slice ~31")
MedEye3d.VulkanBackend.VulkanScreenshot.capture_screenshot(ctx, "/workspaces/MedEye3d.jl/test1_spleen_mask.ppm", image_index=0)

# ========================================================
# TEST 2: Zoom (Shift+Scroll simulated)
# ========================================================
println("\n--- TEST 2: Zoom In (2x) ---")

# Programmatically set zoom
zoom_event = ScrollZoomEvent(2.0)  # positive = zoom in
put!(mainInst.channel, zoom_event)
sleep(0.5)
put!(mainInst.channel, zoom_event)
sleep(0.5)
put!(mainInst.channel, zoom_event)
sleep(0.5)
put!(mainInst.channel, zoom_event)
sleep(0.5)
put!(mainInst.channel, zoom_event)
sleep(0.5)
put!(mainInst.channel, zoom_event)
sleep(0.5)
put!(mainInst.channel, zoom_event)
sleep(1)

# Need to trigger a re-render by scrolling
put!(mainInst.channel, 0)
sleep(2)

println("Screenshot 2: Zoomed in view")
MedEye3d.VulkanBackend.VulkanScreenshot.capture_screenshot(ctx, "/workspaces/MedEye3d.jl/test2_zoomed.ppm", image_index=0)

# ========================================================
# TEST 3: Pan (after zoom, shift the view)
# ========================================================
println("\n--- TEST 3: Pan ---")

# Programmatically set pan via the calcDimsStruct
state = mainInst.states[1]
state.calcDimsStruct.panX = 0.15f0
state.calcDimsStruct.panY = -0.1f0
# Trigger re-render
put!(mainInst.channel, 0)
sleep(2)

println("Screenshot 3: Panned view")
MedEye3d.VulkanBackend.VulkanScreenshot.capture_screenshot(ctx, "/workspaces/MedEye3d.jl/test3_panned.ppm", image_index=0)

# Reset zoom/pan
state.calcDimsStruct.zoom = 1.0f0
state.calcDimsStruct.panX = 0.0f0
state.calcDimsStruct.panY = 0.0f0

# ========================================================
# TEST 4: Different slices (scroll up and down)
# ========================================================
println("\n--- TEST 4: Scroll to different slices ---")

# Scroll to slice near top
put!(mainInst.channel, 20)  # from ~31 to ~51
sleep(2)
println("Screenshot 4a: Upper thorax slice")
MedEye3d.VulkanBackend.VulkanScreenshot.capture_screenshot(ctx, "/workspaces/MedEye3d.jl/test4a_upper.ppm", image_index=0)

# Scroll to lower slice
put!(mainInst.channel, -40)  # from ~51 to ~11
sleep(2)
println("Screenshot 4b: Lower abdomen slice")
MedEye3d.VulkanBackend.VulkanScreenshot.capture_screenshot(ctx, "/workspaces/MedEye3d.jl/test4b_lower.ppm", image_index=0)

# ========================================================
# TEST 5: CT Windowing (bone window)
# ========================================================
println("\n--- TEST 5: Bone window ---")

# Change windowing to bone window: center 300, width 2000 → min=-700, max=1300
state.mainForDisplayObjects.listOfTextSpecifications[1] = setproperties(
    state.mainForDisplayObjects.listOfTextSpecifications[1],
    (minAndMaxValue=Float32.([-700, 1300]),)
)
# Scroll to a slice with visible bone (spine)
put!(mainInst.channel, 20)  # go up to see more vertebrae
sleep(2)
println("Screenshot 5: Bone window")
MedEye3d.VulkanBackend.VulkanScreenshot.capture_screenshot(ctx, "/workspaces/MedEye3d.jl/test5_bone_window.ppm", image_index=0)

# Reset to soft tissue window
state.mainForDisplayObjects.listOfTextSpecifications[1] = setproperties(
    state.mainForDisplayObjects.listOfTextSpecifications[1],
    (minAndMaxValue=Float32.([-150, 250]),)
)

# ========================================================
# TEST 6: PET/Mask Blend (Ctrl+scroll simulated)
# ========================================================
println("\n--- TEST 6: PET/Mask Blend ---")

# Scroll back to spleen
put!(mainInst.channel, -10)
sleep(1)

# Increase mask contribution
pet_event = PetBlendEvent(0.9f0)
put!(mainInst.channel, pet_event)
sleep(1)
put!(mainInst.channel, 0) # trigger render
sleep(2)
println("Screenshot 6: High blend")
MedEye3d.VulkanBackend.VulkanScreenshot.capture_screenshot(ctx, "/workspaces/MedEye3d.jl/test6_blend.ppm", image_index=0)

println("\n=== ALL TESTS COMPLETE ===")
