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

println("=== QUAD VIEW TEST ===")

image_path = "/root/data_decathlon/Task09_Spleen/imagesTr/spleen_10.nii.gz"
mask_path = "/root/data_decathlon/Task09_Spleen/labelsTr/spleen_10.nii.gz"
vol_img = Float32.(niread(image_path).raw)
vol_mask = Float32.(niread(mask_path).raw)

# Prepare multi-panel texture specs for quad view
colors_mapped = [RGB(1.0, 0.0, 0.0)]
ts_img = TextureSpec{Float32}(name="CT", isMainImage=true, color=RGB(1.,1.,1.), minAndMaxValue=Float32.([-150,250]))
ts_mask = TextureSpec{Float32}(name="Mask", isMainImage=false, isMultiDiscreteMask=true, colorSet=colors_mapped, minAndMaxValue=Float32.([0, 1]))

# 4 panels for quad view
tsa = Vector{Vector{TextureSpec}}([
    TextureSpec[deepcopy(ts_img), deepcopy(ts_mask)],  # Panel 1: Axial (top-left)
    TextureSpec[deepcopy(ts_img)],                      # Panel 2: PET-only (top-right)
    TextureSpec[deepcopy(ts_img), deepcopy(ts_mask)],  # Panel 3: Sagittal (bottom-left)
    TextureSpec[deepcopy(ts_img), deepcopy(ts_mask)]   # Panel 4: Coronal (bottom-right)
])

spacing = (1.,1.,1.); origin = (0.,0.,0.)
dts = Vector{DataToScrollDims}()
push!(dts, DataToScrollDims(imageSize=size(vol_img), voxelSize=spacing, dimensionToScroll=3))  # Axial
push!(dts, DataToScrollDims(imageSize=size(vol_img), voxelSize=spacing, dimensionToScroll=3))  # PET
# Sagittal: scroll dimension 1
sag_size = (size(vol_img,2), size(vol_img,3), size(vol_img,1))
push!(dts, DataToScrollDims(imageSize=sag_size, voxelSize=(spacing[2],spacing[3],spacing[1]), dimensionToScroll=3))
# Coronal: scroll dimension 2
cor_size = (size(vol_img,1), size(vol_img,3), size(vol_img,2))
push!(dts, DataToScrollDims(imageSize=cor_size, voxelSize=(spacing[1],spacing[3],spacing[2]), dimensionToScroll=3))

inst = SegmentationDisplay.coordinateDisplay(
    tsa, Float32(1.0), dts,
    [spacing, spacing, (spacing[2],spacing[3],spacing[1]), (spacing[1],spacing[3],spacing[2])],
    [origin, origin, origin, origin],
    Dict{String,Vector}("supervoxel_vertices"=>[],"supervoxel_indices"=>[]),
    Dict{Int64,Dict{Int64,Dict{String,Any}}}()
)

# Prepare volume data
vol_sag = permutedims(vol_img, (2,3,1))
vol_cor = permutedims(vol_img, (1,3,2))
mask_sag = permutedims(vol_mask, (2,3,1))
mask_cor = permutedims(vol_mask, (1,3,2))

SD_list = [
    FullScrollableDat(dataToScroll=[
        ThreeDimRawDat{Float32}(type=Float32, name="CT", dat=vol_img),
        ThreeDimRawDat{Float32}(type=Float32, name="Mask", dat=vol_mask)
    ], dimensionToScroll=3, imagePos=1),
    FullScrollableDat(dataToScroll=[
        ThreeDimRawDat{Float32}(type=Float32, name="CT", dat=vol_img)
    ], dimensionToScroll=3, imagePos=2),
    FullScrollableDat(dataToScroll=[
        ThreeDimRawDat{Float32}(type=Float32, name="CT", dat=vol_sag),
        ThreeDimRawDat{Float32}(type=Float32, name="Mask", dat=mask_sag)
    ], dimensionToScroll=3, imagePos=3),
    FullScrollableDat(dataToScroll=[
        ThreeDimRawDat{Float32}(type=Float32, name="CT", dat=vol_cor),
        ThreeDimRawDat{Float32}(type=Float32, name="Mask", dat=mask_cor)
    ], dimensionToScroll=3, imagePos=4)
]
SegmentationDisplay.passDataForScrolling(inst, SD_list)
sleep(3)

ch = inst.channel
function capture_synced(ch, path)
    done = Channel{Bool}(1)
    put!(ch, ScreenshotEvent(path, done))
    take!(done)
end

# Navigate to spleen slice
put!(ch, 30); sleep(3)

# Print panel info
for (i, st) in enumerate(inst.states)
    verts = st.calcDimsStruct.mainImageQuadVert
    if length(verts) >= 32
        println("Panel $i: NDC x=[$(verts[1]),$(verts[9])], y=[$(verts[2]),$(verts[10])]")
    else
        println("Panel $i: no verts")
    end
end

capture_synced(ch, "/workspaces/MedEye3d.jl/data/scr/quadview_test.png")
println("Quad view screenshot saved!")

println("=== DONE ===")
