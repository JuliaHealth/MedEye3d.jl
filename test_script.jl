
"""

DUAL IMAGES DISPLAY OVERLAYED ON TOP OF EACH OTHER ALLOWING FOR BETTER STUDY ANALYSIS

ISSUE, THE MAIN IMAGE DATA ARGUMENT TO THE DataToScrolLDims struct for imageSize results in segmentation fault, if we pass other pixel data it works as intended.
"""




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
using Statistics

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

dirOfExample = "/media/jm/hddData/projects/MedEye3d.jl/docs/src/data/Example_ct.nii.gz"
dirOfExamplePET = "/media/jm/hddData/projects/MedEye3d.jl/docs/src/data/petb.nii.gz"


imagePET = getImageFromDirectory(dirOfExamplePET, false, true)
ctImage = getImageFromDirectory(dirOfExample, false, true)
pet_image_resampled = MedImages.Resample_to_target.resample_to_image(imagePET, ctImage, MedImages.MedImage_data_struct.B_spline_en)


ctPixels, ctSpacing = getPixelsAndSpacing(ctImage)
# In my case PET data holds 64 bit floats what is unsupported by Opengl
petPixels, petSpacing = getPixelsAndSpacing(pet_image_resampled)
purePetPixels, PurePetSpacing = getPixelsAndSpacing(imagePET)

petPixels = Float32.(petPixels)
ctPixels = Float32.(ctPixels)



#IF WE PUT THE imageSize = size(ctPixels) then we seem to be having a segmentation fault error
datToScrollDimsB = MedEye3d.ForDisplayStructs.DataToScrollDims(imageSize=size(ctPixels), voxelSize=PurePetSpacing, dimensionToScroll=3);

purePetMedian = Statistics.median(petPixels)
purePetStd = Statistics.std(petPixels)

# example of texture specification used - we need to describe all arrays we want to display, to see all possible configurations look into TextureSpec struct docs .
textureSpecificationsPETCT::Vector{TextureSpec} = [
    TextureSpec{Float32}(
        name="PET",
        # we point out that we will supply multiple colors
        isContinuusMask=true,
        #by the number 1 we will reference this data by for example making it visible or not
        studyType="PET",
        numb=Int32(1),
        colorSet=[RGB(0.0, 0.0, 0.0), RGB(1.0, 1.0, 0.0), RGB(1.0, 0.5, 0.0), RGB(1.0, 0.0, 0.0), RGB(1.0, 0.0, 0.0)]
        #display cutoff all values below 200 will be set 2000 and above 8000 to 8000 but only in display - source array will not be modified
        , minAndMaxValue=Float32.([(purePetMedian - purePetStd / 2), (purePetMedian + purePetStd * 2)])
    ),
    TextureSpec{Float32}(
        name="manualModif",
        numb=Int32(2),
        color=RGB(0.0, 1.0, 0.0), minAndMaxValue=Float32.([0, 1]), isEditable=true
    ), TextureSpec{Float32}(
        name="CTIm",
        studyType="CT",
        numb=Int32(3),
        color=RGB(1.0, 1.0, 1.0),
        minAndMaxValue=Float32.([0, 100]))
]
fractionOfMainIm = Float32(0.8);

import MedEye3d.DisplayWords.textLinesFromStrings

mainLines = textLinesFromStrings(["main Line1", "main Line 2"]);
supplLines = map(x -> textLinesFromStrings(["sub  Line 1 in $(x)", "sub  Line 2 in $(x)"]), 1:size(ctPixels)[3]);

import MedEye3d.StructsManag.getThreeDims

tupleVect = [("PET", petPixels), ("CTIm", ctPixels), ("manualModif", zeros(Float32, size(petPixels)))]
slicesDat = getThreeDims(tupleVect)
"""
Holds data necessary to display scrollable data
"""
mainScrollDat = FullScrollableDat(dataToScrollDims=datToScrollDimsB, dimensionToScroll=1, dataToScroll=slicesDat, mainTextToDisp=mainLines, sliceTextToDisp=supplLines);

# SegmentationDisplay.coordinateDisplay(textureSpecificationsPETCT ,fractionOfMainIm ,datToScrollDimsB ,1000);
mainMedEye3dObject = SegmentationDisplay.coordinateDisplay(textureSpecificationsPETCT, fractionOfMainIm, datToScrollDimsB, 1000);

Main.SegmentationDisplay.passDataForScrolling(mainMedEye3dObject, mainScrollDat);
