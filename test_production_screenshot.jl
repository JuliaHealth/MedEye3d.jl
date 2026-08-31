using Pkg
Pkg.activate("/workspaces/MedEye3d.jl")

using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using MedEye3d.MakieEvents
using MedEye3d.SegmentationDisplay
using ColorTypes
using GLFW
using NIfTI

println("=== PRODUCTION SCREENSHOT TEST ===")

image_path = "/root/data_decathlon/Task09_Spleen/imagesTr/spleen_10.nii.gz"
mask_path = "/root/data_decathlon/Task09_Spleen/labelsTr/spleen_10.nii.gz"
vol_img = Float32.(niread(image_path).raw)
vol_mask = Float32.(niread(mask_path).raw)

colors_mapped = [RGB(1.0, 0.0, 0.0)]
ts_img = TextureSpec{Float32}(name="CT", isMainImage=true, color=RGB(1.,1.,1.), minAndMaxValue=Float32.([-150,250]))
ts_mask = TextureSpec{Float32}(name="Mask", isMainImage=false, isMultiDiscreteMask=true, colorSet=colors_mapped, minAndMaxValue=Float32.([0, 1]))

# Test single panel first (should be full screen)
tsa1 = Vector{Vector{TextureSpec}}([TextureSpec[deepcopy(ts_img), deepcopy(ts_mask)]])
spacing = (1.,1.,1.); origin = (0.,0.,0.)
dts1 = [DataToScrollDims(imageSize=size(vol_img), voxelSize=spacing, dimensionToScroll=3)]

inst = SegmentationDisplay.coordinateDisplay(
    tsa1, Float32(1.0), dts1, [spacing], [origin],
    Dict{String,Vector}("supervoxel_vertices"=>[],"supervoxel_indices"=>[]),
    Dict{Int64,Dict{Int64,Dict{String,Any}}}()
)

SD = FullScrollableDat(dataToScroll=[
    ThreeDimRawDat{Float32}(type=Float32, name="CT", dat=vol_img),
    ThreeDimRawDat{Float32}(type=Float32, name="Mask", dat=vol_mask)
], dimensionToScroll=3)
SegmentationDisplay.passDataForScrolling(inst, SD)
sleep(2)

ch = inst.channel
function capture_synced(ch, path)
    done = Channel{Bool}(1)
    put!(ch, ScreenshotEvent(path, done))
    take!(done)
    println("Saved: $path")
end

# Scroll to spleen (approx slice 30 of 63)
put!(ch, 30); sleep(2)

# Debug: print panel viewport info
for (i, st) in enumerate(inst.states)
    verts = st.calcDimsStruct.mainImageQuadVert
    if length(verts) >= 32
        xs = Float32[verts[1], verts[9], verts[17], verts[25]]
        ys = Float32[verts[2], verts[10], verts[18], verts[26]]
        println("Panel $i: NDC x=[$(minimum(xs)), $(maximum(xs))], y=[$(minimum(ys)), $(maximum(ys))]")
    else
        println("Panel $i: EMPTY verts")
    end
end

capture_synced(ch, "/workspaces/MedEye3d.jl/data/scr/single_panel.png")

println("=== DONE ===")
