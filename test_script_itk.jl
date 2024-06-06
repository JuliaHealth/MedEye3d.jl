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
```

Helper functions used to upload data - those will be enclosed (with many more) in a package that Is currently in development - 3dMedPipe 

```
"""
given directory (dirString) to file/files it will return the simple ITK image for futher processing
isMHD when true - data in form of folder with dicom files
isMHD  when true - we deal with MHD data
"""
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

"""
becouse Julia arrays is column wise contiguus in memory and open GL expects row wise we need to rotate and flip images 
pixels - 3 dimensional array of pixel data 
"""
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

"""
given simple ITK image it reads associated pixel data - and transforms it by permuteAndReverse functions
it will also return voxel spacing associated with the image
"""
function getPixelsAndSpacing(image)
    pixelsArr = np.array(sitk.GetArrayViewFromImage(image))# we need numpy in order for pycall to automatically change it into julia array
    spacings = image.GetSpacing()
    return ( permuteAndReverse(pixelsArr), spacings  )
end#getPixelsAndSpacing
```

directories of PET/CT Data - from https://wiki.cancerimagingarchive.net/display/Public/Head-Neck-PET-CT


```
# directories to adapt
dirOfExample ="/home/hurtbadly/Downloads/ct_soft_pat_3_sudy_0.nii.gz"
dirOfExamplePET ="/home/hurtbadly/Downloads/pet_orig_pat_3_sudy_0.nii.gz"

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
"""
If we want to display some text we need to pass it as a vector of SimpleLineTextStructs - utility function to achieve this is 
textLinesFromStrings() where we pass resies of strings, if we want we can also edit those structs to controll size of text and space to next line look into SimpleLineTextStruct doc
mainLines - will be displayed over all slices
supplLines - will be displayed over each slice where is defined - below just dummy data
"""
import MedEye3d.DisplayWords.textLinesFromStrings

mainLines= textLinesFromStrings(["main Line1", "main Line 2"]);
supplLines=map(x->  textLinesFromStrings(["sub  Line 1 in $(x)", "sub  Line 2 in $(x)"]), 1:size(ctPixels)[3] );
```


If we want to pass 3 dimensional array of scrollable data we need to supply it via vector ThreeDimRawDat's struct
utility function to make creation of those easier is getThreeDims which creates series of ThreeDimRawDat from list of tuples where
    first entry is String and second entry is 3 dimensional array with data 
    strings needs to be the same as we  defined in texture specifications at the bagining
    data arrays needs to be o the same size and be of the same type we specified in texture specification
```
import MedEye3d.StructsManag.getThreeDims

tupleVect = [("PET",petPixels) ,("CTIm",ctPixels),("manualModif",zeros(UInt8,size(petPixels)) ) ]
slicesDat= getThreeDims(tupleVect )
"""
Holds data necessary to display scrollable data
"""
mainScrollDat = FullScrollableDat(dataToScrollDims =datToScrollDimsB
                                 ,dimensionToScroll=1 # what is the dimension of plane we will look into at the beginning for example transverse, coronal ...
                                 ,dataToScroll= slicesDat
                                 ,mainTextToDisp= mainLines
                                 ,sliceTextToDisp=supplLines );

```
This function prepares all for display; 1000 in the end is responsible for setting window width for more look into SegmentationDisplay.coordinateDisplay
```
SegmentationDisplay.coordinateDisplay(textureSpecificationsPETCT ,fractionOfMainIm ,datToScrollDimsB ,1000);
```
As all is ready we can finally display image 
```
Main.SegmentationDisplay.passDataForScrolling(mainScrollDat);
