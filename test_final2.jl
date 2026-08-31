using Pkg
Pkg.activate("/workspaces/MedEye3d.jl")

using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using MedEye3d.MakieEvents
using ColorTypes
using GLFW
using NIfTI
using FileIO

println("=== FINAL COMPREHENSIVE TEST ===")

image_path = "/root/data_decathlon/Task09_Spleen/imagesTr/spleen_10.nii.gz"
mask_path = "/root/data_decathlon/Task09_Spleen/labelsTr/spleen_10.nii.gz"
nii_img = niread(image_path)
nii_mask = niread(mask_path)
vol_img = Float32.(nii_img.raw)
vol_mask = Float32.(nii_mask.raw)
println("Volume: ", size(vol_img))

ts_img = TextureSpec{Float32}(name="MainImage", isMainImage=true, color=RGB(1.,1.,1.), minAndMaxValue=Float32.([-150,250]))
ts_mask = TextureSpec{Float32}(name="Mask", isMainImage=false, color=RGB(1.,0.,0.), maskContribution=Float32(0.8), minAndMaxValue=Float32.([0.5,1.5]))
tsa = Vector{Vector{TextureSpec}}([TextureSpec[deepcopy(ts_img), deepcopy(ts_mask)]])

spacing = (1.,1.,1.); origin = (0.,0.,0.)
dts = Vector{DataToScrollDims}()
push!(dts, DataToScrollDims(imageSize=size(vol_img), voxelSize=spacing, dimensionToScroll=3))

inst = MedEye3d.SegmentationDisplay.coordinateDisplay(
    tsa, Float32(1.0), dts, [spacing], [origin],
    Dict{String,Vector}("supervoxel_vertices"=>[],"supervoxel_indices"=>[]),
    Dict{Int64,Dict{Int64,Dict{String,Any}}}()
)

SD = FullScrollableDat(dataToScroll=[
    ThreeDimRawDat{Float32}(type=Float32, name="MainImage", dat=vol_img),
    ThreeDimRawDat{Float32}(type=Float32, name="Mask", dat=vol_mask)
], dimensionToScroll=3)
MedEye3d.SegmentationDisplay.passDataForScrolling(inst, SD)
sleep(1)

state = inst.states[1]
ch = inst.channel

function capture_synced(ch, path)
    done = Channel{Bool}(1)
    put!(ch, ScreenshotEvent(path, done))
    take!(done)
end

# ========================================================
# TEST 1: Scroll to spleen slice 31
# ========================================================
println("\n--- TEST 1: Spleen slice 31 + mask ---")
put!(ch, 30); sleep(3)
println("Slice: $(state.currentDisplayedSlice)")
capture_synced(ch, "/workspaces/MedEye3d.jl/final_t1.png")
println("Captured t1")

# ========================================================
# TEST 2: Zoom 2x
# ========================================================
println("\n--- TEST 2: Zoom 2x ---")
for i in 1:7; put!(ch, ScrollZoomEvent(2.0)); sleep(0.2); end
sleep(0.5)
put!(ch, 0); sleep(1)
println("Zoom: $(state.calcDimsStruct.zoom)")
capture_synced(ch, "/workspaces/MedEye3d.jl/final_t2.png")
println("Captured t2")

# ========================================================
# TEST 3: Pan while zoomed
# ========================================================
println("\n--- TEST 3: Pan ---")
state.calcDimsStruct.panX = 0.15f0
state.calcDimsStruct.panY = -0.1f0
put!(ch, 0); sleep(1)
capture_synced(ch, "/workspaces/MedEye3d.jl/final_t3.png")
println("Captured t3")

# Reset
state.calcDimsStruct.zoom = 1.0f0
state.calcDimsStruct.panX = 0.0f0
state.calcDimsStruct.panY = 0.0f0

# ========================================================
# TEST 4: Different slices
# ========================================================
println("\n--- TEST 4: Different slices ---")
put!(ch, 15); sleep(2)
println("Slice 4a: $(state.currentDisplayedSlice)")
capture_synced(ch, "/workspaces/MedEye3d.jl/final_t4a.png")

put!(ch, -30); sleep(2)
println("Slice 4b: $(state.currentDisplayedSlice)")
capture_synced(ch, "/workspaces/MedEye3d.jl/final_t4b.png")

# ========================================================
# TEST 5: Bone window
# ========================================================
println("\n--- TEST 5: Bone window ---")
state.mainForDisplayObjects.listOfTextSpecifications[1].minAndMaxValue .= Float32[-700, 1300]
put!(ch, 10); sleep(2)
capture_synced(ch, "/workspaces/MedEye3d.jl/final_t5.png")
println("Captured t5")

state.mainForDisplayObjects.listOfTextSpecifications[1].minAndMaxValue .= Float32[-150, 250]

println("\n=== ALL FINAL TESTS COMPLETE ===")
