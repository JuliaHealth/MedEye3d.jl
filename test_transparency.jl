using Pkg
Pkg.activate("/workspaces/MedEye3d.jl")

using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using MedEye3d.MakieEvents
using ColorTypes
using GLFW
using NIfTI

println("=== TRANSPARENCY TEST ===")

image_path = "/root/data_decathlon/Task09_Spleen/imagesTr/spleen_10.nii.gz"
mask_path = "/root/data_decathlon/Task09_Spleen/labelsTr/spleen_10.nii.gz"
nii_img = niread(image_path)
nii_mask = niread(mask_path)
vol_img = Float32.(nii_img.raw)
vol_mask = Float32.(nii_mask.raw)

# Use semi-transparent mask with maskContribution=0.5
ts_img = TextureSpec{Float32}(name="MainImage", isMainImage=true, color=RGB(1.,1.,1.), minAndMaxValue=Float32.([-150,250]))
ts_mask = TextureSpec{Float32}(
    name="Mask", isMainImage=false,
    color=RGB(1.,0.,0.),
    maskContribution=Float32(0.5),
    minAndMaxValue=Float32.([0.5,1.5])
)
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

ch = inst.channel

function capture_synced(ch, path)
    done = Channel{Bool}(1)
    put!(ch, ScreenshotEvent(path, done))
    take!(done)
end

# Scroll to spleen
put!(ch, 30); sleep(3)
capture_synced(ch, "/workspaces/MedEye3d.jl/trans_test.png")
println("Captured transparency test")

println("=== DONE ===")
