#I use Simple ITK as most robust
using MedEye3d, Conda,PyCall,Pkg

Conda.pip_interop(true)
Conda.pip("install", "SimpleITK")
Conda.pip("install", "h5py")
sitk = pyimport("SimpleITK")
np= pyimport("numpy")

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
import MedEye3d.ForDisplayStructs.ActorWithOpenGlObjects
import MedEye3d.OpenGLDisplayUtils
import MedEye3d.DisplayWords.textLinesFromStrings
import MedEye3d.StructsManag.getThreeDims


function getImageFromDirectory(dirString,isMHD::Bool, isDicomList::Bool)
    #simpleITK object used to read from disk 
    reader = sitk.ImageSeriesReader()
    if(isDicomList)# data in form of folder with dicom files
        dicom_names = reader.GetGDCMSeriesFileNames(dirString)
        reader.SetFileNames(dicom_names)
        return reader.Execute()
    elseif(isMHD) #mhd file
        return sitk.ReadImage(dirString)
    end
end#getPixelDataAndSpacing


function permuteAndReverse(pixels)
    pixels=  permutedims(pixels, (3,2,1))
    sizz=size(pixels)
    for i in 1:sizz[1]
        for j in 1:sizz[3]
            pixels[i,:,j] =  reverse(pixels[i,:,j])
        end# 
    end# 
    return pixels
  end#permuteAndReverse


function getPixelsAndSpacing(image)
    pixelsArr = np.array(sitk.GetArrayViewFromImage(image))# we need numpy in order for pycall to automatically change it into julia array
    spacings = image.GetSpacing()
    return ( permuteAndReverse(pixelsArr), spacings  )
end#getPixelsAndSpacing

# directories to adapt
dirOfExample ="D:/mingw_installation/home/hurtbadly/Downloads/ct_soft_pat_3_sudy_0.nii.gz"
dirOfExamplePET ="D:/mingw_installation/home/hurtbadly/Downloads/pet_orig_pat_3_sudy_0.nii.gz"

# in most cases dimensions of PET and CT data arrays will be diffrent in order to make possible to display them we need to resample and make dimensions equal
imagePET= getImageFromDirectory(dirOfExamplePET,true,false)
ctImage= getImageFromDirectory(dirOfExample,true,false)
pet_image_resampled = sitk.Resample(imagePET, ctImage)

ctPixels, ctSpacing = getPixelsAndSpacing(ctImage)
# In my case PET data holds 64 bit floats what is unsupported by Opengl
petPixels, petSpacing =getPixelsAndSpacing(pet_image_resampled)
purePetPixels, PurePetSpacing = getPixelsAndSpacing(imagePET)

petPixels = Float32.(petPixels)

# we need to pass some metadata about image array size and voxel dimensions to enable proper display
datToScrollDimsB= MedEye3d.ForDisplayStructs.DataToScrollDims(imageSize=  size(ctPixels) ,voxelSize=PurePetSpacing, dimensionToScroll = 3 );
# example of texture specification used - we need to describe all arrays we want to display, to see all possible configurations look into TextureSpec struct docs .
textureSpecificationsPETCT = [
  TextureSpec{Float32}(
      name = "PET",
      isNuclearMask=true,
      # we point out that we will supply multiple colors
      isContinuusMask=true,
      #by the number 1 we will reference this data by for example making it visible or not
      numb= Int32(1),
      colorSet = [RGB(0.0,0.0,0.0),RGB(1.0,1.0,0.0),RGB(1.0,0.5,0.0),RGB(1.0,0.0,0.0) ,RGB(1.0,0.0,0.0)]
      #display cutoff all values below 200 will be set 2000 and above 8000 to 8000 but only in display - source array will not be modified
      ,minAndMaxValue= Float32.([200,8000])
     ),
  TextureSpec{UInt8}(
      name = "manualModif",
      numb= Int32(2),
      color = RGB(0.0,1.0,0.0)
      ,minAndMaxValue= UInt8.([0,1])
      ,isEditable = true
     ),

     TextureSpec{Int16}(
      name= "CTIm",
      numb= Int32(3),
      isMainImage = true,
      minAndMaxValue= Int16.([0,100]))  
];
# We need also to specify how big part of the screen should be occupied by the main image and how much by text fractionOfMainIm= Float32(0.8);
fractionOfMainIm= Float32(0.8);

import MedEye3d.DisplayWords.textLinesFromStrings

mainLines= textLinesFromStrings(["main Line1", "main Line 2"]);
supplLines=map(x->  textLinesFromStrings(["sub  Line 1 in $(x)", "sub  Line 2 in $(x)"]), 1:size(ctPixels)[3] );


import MedEye3d.StructsManag.getThreeDims

tupleVect = [("PET",petPixels) ,("CTIm",ctPixels),("manualModif",zeros(UInt8,size(petPixels)) ) ]
slicesDat= getThreeDims(tupleVect )

mainScrollDat = FullScrollableDat(dataToScrollDims =datToScrollDimsB
                                 ,dimensionToScroll=1 # what is the dimension of plane we will look into at the beginning for example transverse, coronal ...
                                 ,dataToScroll= slicesDat
                                 ,mainTextToDisp= mainLines
                                 ,sliceTextToDisp=supplLines );

# function prepares all for display; 1000 in the end is responsible for setting window width for more look into SegmentationDisplay.coordinateDisplay

SegmentationDisplay.coordinateDisplay(textureSpecificationsPETCT ,fractionOfMainIm ,datToScrollDimsB ,1000);

# As all is ready we can finally display image 

Main.SegmentationDisplay.passDataForScrolling(mainScrollDat);
