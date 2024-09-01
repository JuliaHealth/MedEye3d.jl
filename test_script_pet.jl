using MedImages
#I use Simple ITK as most robust
using MedEye3d, Conda, PyCall, Pkg
import MedEye3d
import MedEye3d.ForDisplayStructs
import MedEye3d.ForDisplayStructs.TextureSpec
using ColorTypes
import MedEye3d.SegmentationDisplay
using Statistics
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
    return (permuteAndReverse(pixelsArr), spacings)
end#getPixelsAndSpacing

dirOfExample = "D:/mingw_installation/home/hurtbadly/Downloads/ct_soft_pat_3_sudy_0.nii.gz"
dirOfExamplePET = "D:/mingw_installation/home/hurtbadly/Downloads/pet_orig_pat_3_sudy_0.nii.gz"


imagePET = getImageFromDirectory(dirOfExamplePET, false, true)


# In my case PET data holds 64 bit floats what is unsupported by Opengl
purePetPixels, purePetSpacing = getPixelsAndSpacing(imagePET)
# imagePET.voxel_data = Float32.(purePetPixels)
purePetPixels = Float32.(purePetPixels)

"""
        medImageDataInstance = MedImages.load_image(studySrcPath)
        #permuting the voxelData to some default orientation, such that the image is not inverted or sideways
        medImageDataInstance.voxel_data = permutedims(medImageDataInstance.voxel_data, (1, 2, 3)) #previously in the test script the default was (3, 2, 1)
        sizeInfo = size(medImageDataInstance.voxel_data)
        for outerNum in 1:sizeInfo[1]
            for innerNum in 1:sizeInfo[3]
                medImageDataInstance.voxel_data[outerNum, :, innerNum] = reverse(medImageDataInstance.voxel_data[outerNum, :, innerNum])
            end
        end
        #Float conversion happens here for voxelData, currently only Floats are supported to keep it simple
        medImageDataInstance.voxel_data = Float32.(medImageDataInstance.voxel_data) #conversion of the voxel data in the original Medimage struct to Float32. for openGL compatibility


"""

#IF WE PUT THE imageSize = size(ctPixels) then we seem to be having a segmentation fault error
datToScrollDimsB = MedEye3d.ForDisplayStructs.DataToScrollDims(imageSize=size(purePetPixels), voxelSize=purePetSpacing, dimensionToScroll=3);
# example of texture specification used - we need to describe all arrays we want to display, to see all possible configurations look into TextureSpec struct docs .

"""
        return TextureSpec{Float32}(name="PET", studyType="PET", isContinuusMask=true, numb=Int32(1), colorSet=[RGB(0.0, 0.0, 0.0), RGB(1.0, 1.0, 0.0), RGB(1.0, 0.5, 0.0), RGB(1.0, 0.0, 0.0), RGB(1.0, 0.0, 0.0)], minAndMaxValue=Float32.([200, 8000]))

"""

purePetMedian = median(purePetPixels)
purePetStd = std(purePetPixels)
textureSpecificationsPETCT::Vector{TextureSpec} = [
    TextureSpec{Float32}(
        name="PET",
        # isNuclearMask=true,
        # we point out that we will supply multiple colors
        isContinuusMask=true,
        studyType="PET",
        #by the number 1 we will reference this data by for example making it visible or not
        numb=Int32(1),
        colorSet=[RGB(0.0, 0.0, 0.0), RGB(1.0, 1.0, 0.0), RGB(1.0, 0.5, 0.0), RGB(1.0, 0.0, 0.0)]
        #display cutoff all values below 200 will be set 2000 and above 8000 to 8000 but only in display - source array will not be modified
        # , minAndMaxValue=Float32.([minimum(purePetPixels), maximum(purePetPixels)])),

        , minAndMaxValue=Float32.([(purePetMedian - purePetStd / 2), (purePetMedian + purePetStd * 2)])), TextureSpec{Float32}(
        name="manualModif",
        numb=Int32(2),
        color=RGB(0.0, 1.0, 0.0), minAndMaxValue=Float32.([0, 1]), isEditable=true
    )
]
fractionOfMainIm = Float32(0.8);

import MedEye3d.DisplayWords.textLinesFromStrings

mainLines = textLinesFromStrings(["main Line1", "main Line 2"]);
supplLines = map(x -> textLinesFromStrings(["sub  Line 1 in $(x)", "sub  Line 2 in $(x)"]), 1:size(purePetPixels)[3]);

import MedEye3d.StructsManag.getThreeDims

tupleVect = [("PET", purePetPixels), ("manualModif", zeros(Float32, size(purePetPixels)))]
slicesDat = getThreeDims(tupleVect)
"""
Holds data necessary to display scrollable data
"""
mainScrollDat = FullScrollableDat(dataToScrollDims=datToScrollDimsB, dimensionToScroll=1, dataToScroll=slicesDat, mainTextToDisp=mainLines, sliceTextToDisp=supplLines);

# SegmentationDisplay.coordinateDisplay(textureSpecificationsPETCT ,fractionOfMainIm ,datToScrollDimsB ,1000);
mainMedEye3dObject = SegmentationDisplay.coordinateDisplay(textureSpecificationsPETCT, fractionOfMainIm, datToScrollDimsB, 1000);

Main.SegmentationDisplay.passDataForScrolling(mainMedEye3dObject, mainScrollDat);
