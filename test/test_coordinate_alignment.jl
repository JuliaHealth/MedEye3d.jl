using Test
using MedEye3d
using MedEye3d.DataStructs
using MedEye3d.ForDisplayStructs
using MedEye3d.StructsManag
using MedEye3d.MakieEvents
using MedEye3d.ReactOnMouseClickAndDrag
using MedEye3d.SegmentationDisplay
using ColorTypes
using FileIO

# 1. Create a 512x512 test image with a grid and 4 asymmetric corner markers
w, h, d = 512, 512, 10
ct = zeros(Float32, w, h, d)
for x in 1:w, y in 1:h, z in 1:d
    ct[x, y, z] = Float32(x + y) / Float32(w + h) * 100.0f0
end

mask = zeros(Int16, w, h, d)
# Marker A (small X=80, small Y=80): Label 1 (Red)
mask[75:85, 75:85, :] .= Int16(1)

# Marker B (large X=430, small Y=80): Label 2 (Green)
mask[425:435, 75:85, :] .= Int16(2)

# Marker C (small X=80, large Y=430): Label 3 (Blue)
mask[75:85, 425:435, :] .= Int16(3)

# Marker D (large X=430, large Y=430): Label 4 (Yellow)
mask[425:435, 425:435, :] .= Int16(4)

# Colors: 1=Red, 2=Green, 3=Blue, 4=Yellow, 5=White (for test paint)
colors = [RGB(1.0, 0.0, 0.0), RGB(0.0, 1.0, 0.0), RGB(0.0, 0.0, 1.0), RGB(1.0, 1.0, 0.0), RGB(1.0, 1.0, 1.0)]

textSpec_ct = TextureSpec{Float32}(name="CT", isMainImage=true, minAndMaxValue=Float32.([0, 100]))
textSpec_mask = TextureSpec{Int16}(
    name="Mask", isMainImage=false, isMultiDiscreteMask=true, isIntegerTexture=true,
    colorSet=colors, minAndMaxValue=Int16.([0, 5]), isEditable=true, maskContribution=1.0f0, strokeWidth=Int32(5)
)

texSpecs = [
    TextureSpec[deepcopy(textSpec_ct), deepcopy(textSpec_mask)],
    TextureSpec[deepcopy(textSpec_ct)],
    TextureSpec[deepcopy(textSpec_ct), deepcopy(textSpec_mask)],
    TextureSpec[deepcopy(textSpec_ct), deepcopy(textSpec_mask)],
    TextureSpec[deepcopy(textSpec_ct), deepcopy(textSpec_mask)]
]

vdt = [
    Any[("CT", ct), ("Mask", mask)],
    Any[("CT", ct)],
    Any[("CT", ct), ("Mask", mask)],
    Any[("CT", ct), ("Mask", mask)],
    Any[("CT", ct), ("Mask", mask)]
]

sp = (1.0, 1.0, 1.0)
op = (0.0, 0.0, 0.0)
spacings = [[sp, sp], [sp], [sp, sp], [sp, sp], [sp, sp]]
origins = [[op, op], [op], [op, op], [op, op], [op, op]]

dummyStudySrc = Vector{Vector{Tuple{String,String}}}()
dispInstance = SegmentationDisplay.displayImage(
    dummyStudySrc;
    textureSpecArray=texSpecs,
    voxelDataTupleVector=vdt,
    spacings=spacings,
    origins=origins,
    fractionOfMainImage=Float32(1.0),
    windowWidth=1400,
    quadView=true
)

channel = dispInstance.channel

# Initialize slices
put!(channel, Int64(0))
sleep(1.5)

# Capture initial screenshot
scr_path = "/workspaces/MedEye3d.jl/test/test_data/screenshots/align_01_initial.png"
done = Channel{Bool}(1)
put!(channel, ScreenshotEvent(scr_path, done))
take!(done)
println("Saved align_01_initial.png")

# Now let's paint directly at Marker A screen position (195, 70) with Label 5 (White)
println("Simulating paint at Marker A screen position (195, 70) with Label 5 (White)...")
put!(channel, PaintValEvent(5, true))
put!(channel, [MouseStruct(isLeftButtonDown=true, lastCoordinates=[CartesianIndex(195, 70)], actualWindowWidth=1400, actualWindowHeight=900)])
sleep(0.5)

# Also let's paint at Marker B screen position (503, 70) with Label 5 (White)
println("Simulating paint at Marker B screen position (503, 70) with Label 5 (White)...")
put!(channel, [MouseStruct(isLeftButtonDown=true, lastCoordinates=[CartesianIndex(503, 70)], actualWindowWidth=1400, actualWindowHeight=900)])
sleep(0.5)

scr_path2 = "/workspaces/MedEye3d.jl/test/test_data/screenshots/align_02_painted.png"
done2 = Channel{Bool}(1)
put!(channel, ScreenshotEvent(scr_path2, done2))
take!(done2)
println("Saved align_02_painted.png")

# Check where voxels were set in mask across all slices
painted_voxels = findall(mask .== Int16(5))
println("Painted voxels with Label 5 across mask: $(length(painted_voxels))")
for p in painted_voxels[1:min(8, length(painted_voxels))]
    println("  painted at voxel: ($(p[1]), $(p[2]), $(p[3]))")
end

# Clean shutdown
put!(channel, CloseWindowEvent())
sleep(0.5)
println("Test done.")
exit(0)
