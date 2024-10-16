using MedImages
#I use Simple ITK as most robust
using MedEye3d, Conda, PyCall, Pkg
import MedEye3d
import MedEye3d.ForDisplayStructs
import MedEye3d.ForDisplayStructs.TextureSpec
using ColorTypes
import MedEye3d.SegmentationDisplay

import MedEye3d.DataStructs.ThreeDimRawDat
import MedEye3d.DataStructs.DataToScrollDims
import MedEye3d.DataStructs.FullScrollableDat
import MedEye3d.ForDisplayStructs.KeyboardStruct
import MedEye3d.ForDisplayStructs.MouseStruct
import MedEye3d.OpenGLDisplayUtils
import MedEye3d.DisplayWords.textLinesFromStrings
import MedEye3d.StructsManag.getThreeDims

function getImageFromDirectory(dirString, isMHD::Bool, isDicomList::Bool)
    #simpleITK object used to read from disk
    medimage_object = MedImages.load_image(dirString)
    return medimage_object
end
#getPixelDataAndSpacing

#reorienting to permuted dimensions 1,2,3 so the image is not inverted or sideways
function permuteAndReverse(pixels)
    pixels = permutedims(pixels, (1, 2, 3))
    sizz = size(pixels)
    for i in 1:sizz[1]
        for j in 1:sizz[3]
            pixels[i, :, j] = reverse(pixels[i, :, j])
        end#
    end#
    return pixels
end#permuteAndReverse


function getPixelsAndSpacing(image)
    pixelsArr = image.voxel_data #we need numpy in order for pycall to automatically change it into julia array
    spacings = image.spacing
    # spacings = (spacings[3], spacings[2], spacings[1])
    return (permuteAndReverse(pixelsArr), spacings)
end#getPixelsAndSpacing

niftiImage = "D:/mingw_installation/home/hurtbadly/Downloads/volume-0.nii.gz"
dirOfExample = "D:/mingw_installation/home/hurtbadly/Downloads/ct_soft_pat_3_sudy_0.nii.gz"
dirOfExamplePET = "D:/mingw_installation/home/hurtbadly/Downloads/pet_orig_pat_3_sudy_0.nii.gz"
resample_image = "D:/mingw_installation/home/hurtbadly/Downloads/Output Volume.nii.gz"
resample_image_extreme_spacing = "D:/mingw_installation/home/hurtbadly/Downloads/Output Volume_1.nii.gz"
extreme_test_one = "D:/mingw_installation/home/hurtbadly/Downloads/extreme_test_one.nii.gz"

ctImage = getImageFromDirectory(extreme_test_one, false, true)

ctPixels, ctSpacing = getPixelsAndSpacing(ctImage)
# ctSpacing = (ctSpacing[3], ctSpacing[2], ctSpacing[1]) # @info ctSpacing

@info ctSpacing
ctPixels = Float32.(ctPixels)
# ctPixels = ctPixels .- minimum(ctPixels)
# ctPixels = ctPixels ./ maximum(ctPixels)
# ctPixels = rand(Float32, size(ctPixels))


#IF WE PUT THE imageSize = size(ctPixels) then we seem to be having a segmentation fault error
datToScrollDimsB = MedEye3d.ForDisplayStructs.DataToScrollDims(imageSize=size(ctPixels), voxelSize=ctSpacing, dimensionToScroll=3);
# example of texture specification used - we need to describe all arrays we want to display, to see all possible configurations look into TextureSpec struct docs .
textureSpecificationsPETCT::Vector{TextureSpec} = [
    TextureSpec{Float32}(
        name="manualModif",
        numb=Int32(2),
        color=RGB(0.0, 1.0, 0.0), minAndMaxValue=Float32.([0, 1]), isEditable=true
    ), TextureSpec{Float32}(
        name="CTIm",
        numb=Int32(3),
        studyType="CT",
        # isMainImage=true,
        color=RGB(1.0, 1.0, 1.0),
        minAndMaxValue=Float32.([0, 100]))
]
fractionOfMainIm = Float32(0.8);

import MedEye3d.DisplayWords.textLinesFromStrings

mainLines = textLinesFromStrings(["main Line1", "main Line 2"]);
supplLines = map(x -> textLinesFromStrings(["sub  Line 1 in $(x)", "sub  Line 2 in $(x)"]), 1:size(ctPixels)[3]);

import MedEye3d.StructsManag.getThreeDims

manualModifPixels = zeros(Float32, size(ctPixels))
manualModifPixels[50:100, 50:100, 50:100] .= 1.0
tupleVect = [("CTIm", ctPixels), ("manualModif", manualModifPixels)]
slicesDat = getThreeDims(tupleVect)
"""
Holds data necessary to display scrollable data
"""
mainScrollDat = FullScrollableDat(dataToScrollDims=datToScrollDimsB, dimensionToScroll=1, dataToScroll=slicesDat, mainTextToDisp=mainLines, sliceTextToDisp=supplLines);

# SegmentationDisplay.coordinateDisplay(textureSpecificationsPETCT ,fractionOfMainIm ,datToScrollDimsB ,1000);
mainMedEye3dObject = SegmentationDisplay.coordinateDisplay(textureSpecificationsPETCT, fractionOfMainIm, datToScrollDimsB, 1000);

Main.SegmentationDisplay.passDataForScrolling(mainMedEye3dObject, mainScrollDat);
