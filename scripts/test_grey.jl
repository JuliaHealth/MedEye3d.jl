using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using MedEye3d.StructsManag
using MedImages
using ColorTypes
using GLFW

image_path = "/root/data_decathlon/Task09_Spleen/imagesTr/spleen_10.nii.gz"
mask_path = "/root/data_decathlon/Task09_Spleen/labelsTr/spleen_10.nii.gz"

nii_img = MedEye3d.NIfTI.niread(image_path)
nii_mask = MedEye3d.NIfTI.niread(mask_path)
vol_img = Float32.(nii_img.raw)
vol_mask = Float32.(nii_mask.raw)

# FORCE THE WHOLE IMAGE TO BE 150.0 (WHITE/GREY) SO IT CANNOT BE BLACK!
vol_img .= 150.0

vol_img_axial = vol_img
vol_mask_axial = vol_mask

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
    maskContribution=Float32(1.0)
)
textureSpecArray = Vector{Vector{TextureSpec}}([
    TextureSpec[deepcopy(textureSpec_img), deepcopy(textureSpec_mask)]
])
voxelDataTupleVector = Vector{Vector{Any}}([
    Any[("MainImage", vol_img_axial), ("Mask", vol_mask_axial)]
])

spacing_axial = (1.0, 1.0, 1.0)
origin_axial = (0.0, 0.0, 0.0)

voxelDataForUniforms = Vector{Vector{Any}}()
dataToScroll = Vector{DataToScrollDims}()

voxelDataForUniforms_current = map(x -> x[2], voxelDataTupleVector[1])
push!(voxelDataForUniforms, voxelDataForUniforms_current)
push!(dataToScroll, DataToScrollDims(imageSize=size(voxelDataForUniforms_current[1]), voxelSize=spacing_axial, dimensionToScroll=3))

mainMedEye3dInstance = MedEye3d.SegmentationDisplay.coordinateDisplay(textureSpecArray, 1.0, dataToScroll, spacing_axial, origin_axial)

tuplesForScroll = map(x -> (x[1][1], x[1][2], x[2]), zip(voxelDataTupleVector[1], dataToScroll))
dataToScrollList = map(x -> ThreeDimRawDat{Float32}(type=Float32, name=x[1], dat=x[2]), voxelDataTupleVector[1])
ScrollDat = FullScrollableDat(dataToScroll=dataToScrollList, dimensionToScroll=3)

mainMedEye3dInstance.states[1].onScrollData = ScrollDat
mainMedEye3dInstance.states[1].currentDisplayedSlice = 50

MedEye3d.ReactToScroll.reactToScroll(0, mainMedEye3dInstance.states, false)

sleep(1)
run(`import -window root test_grey.png`)
sleep(1)
