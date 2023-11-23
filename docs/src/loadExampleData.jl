######### Defining Helper Functions and imports

#]add ColorTypes,DataTypesBasic,Dictionaries,FreeType,FreeTypeAbstraction,GLFW,GeometryTypes,
using Pkg
Pkg.add(url="https://github.com/jakubMitura14/MedPipe3D.jl.git")

Pkg.add("CUDA")

Pkg.add("Rocket")

Pkg.add(Pkg.PackageSpec(;name="ColorTypes", version="0.11.4"))
Pkg.add(Pkg.PackageSpec(;name="DataTypesBasic", version="2.0.3"))
Pkg.add(Pkg.PackageSpec(;name="Dictionaries", version="0.3.25"))
Pkg.add(Pkg.PackageSpec(;name="FreeType", version="4.1.0"))
Pkg.add(Pkg.PackageSpec(;name="FreeTypeAbstraction", version="0.10.0"))
Pkg.add(Pkg.PackageSpec(;name="GLFW", version="3.4.1"))
Pkg.add(Pkg.PackageSpec(;name="GeometryTypes", version="0.8.5"))
Pkg.add(Pkg.PackageSpec(;name="Match", version="1.2.0"))
Pkg.add(Pkg.PackageSpec(;name="ModernGL", version="1.1.5"))
Pkg.add(Pkg.PackageSpec(;name="Observables", version="0.5.4"))
Pkg.add(Pkg.PackageSpec(;name="Parameters", version="0.12.3"))
Pkg.add(Pkg.PackageSpec(;name="Setfield", version="1.1.1"))
Pkg.add(Pkg.PackageSpec(;name="HDF5", version="0.17.1"))
Pkg.add(Pkg.PackageSpec(;name="StaticArrays", version="1.5.0"))
Pkg.add(Pkg.PackageSpec(;name="MedEval3D", version="0.2.0"))
Pkg.add("Conda")
Pkg.add("PythonCall")
Pkg.add("CondaPkg")
Pkg.add("Libz")
using Pkg

Pkg.add(["ProgressMeter","StaticArrays","BSON","Distributed","Flux","Hyperopt","Colors"
,"Plots","Distributions","Clustering","IrrationalConstants","ParallelStencil"
,"HDF5","CoordinateTransformations","NIfTI","Images"])


# CondaPkg.add_pip("SimpleITK", version="")

using Pkg
# Pkg.add(url="https://github.com/jakubMitura14/Rocket.jl.git")
# Pkg.add("Rocket")
# directories of PET/CT Data - from not published (yet) dataset - single example available from https://wwsi365-my.sharepoint.com/:f:/g/personal/s9956jm_ms_wwsi_edu_pl/Eq3cL7Md5bhPvnUlFLAMKZAB3nsbl6Q18fG96iVajvnNqA?e=bzX68X
dirOfExample ="/workspaces/MedEye3d.jl/docs/src/data/Example_ct.nii.gz"
dirOfExamplePET ="/workspaces/MedEye3d.jl/docs/src/data/petb.nii.gz"

# @spawn :interactive
#I use Simple ITK as most robust
using MedEye3d,CondaPkg,Pkg,PythonCall, Base.Threads

# Conda.pip_interop(true)
# Conda.pip("install", "SimpleITK")
# Conda.pip("install", "h5py")
# CondaPkg.add("SimpleITK")
# CondaPkg.add("numpy")
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
using Rocket
import  Rocket.AsyncSchedulerInstance
import  Rocket.AsyncSchedulerDataMessage


using MedPipe3D
using Distributions
using Clustering
using IrrationalConstants
using ParallelStencil
using MedPipe3D.LoadFromMonai, MedPipe3D.HDF5saveUtils,MedEye3d.visualizationFromHdf5, MedEye3d.distinctColorsSaved
using CUDA
using HDF5,Colors,Rocket
using CoordinateTransformations,Images

using NIfTI,LinearAlgebra


function Rocket.__process_channeled_message(instance::AsyncSchedulerInstance{D}, message::AsyncSchedulerDataMessage{D}, actor) where D
   # print("oooooooooooo")
   on_next!(actor, message.data)
end


Pkg.instantiate()

#add ProgressMeter StaticArrays BSON Distributed Flux Hyperopt Colors Plots Distributions Clustering IrrationalConstants ParallelStencil HDF5

#directory where we want to store our HDF5 that we will use
pathToHDF5="/root/data/smallDataSet.hdf5"
# data_dir = "/home/data/Task09_Spleen/"
data_dir = "/root/data_decathlon/Task09_Spleen"
fid = h5open(pathToHDF5, "w")

# Channel{T=Any}(func::Function, size=0; taskref=nothing, spawn=false)
# Base.Channel(size=0; taskref=nothing, spawn=false)


# Base.Channel{Char}(1, spawn=true)





#representing number that is the patient id in this dataset
patentNum = 3
patienGroupName=string(patentNum)
z=7# how big is the area from which we collect data to construct probability distributions
klusterNumb = 5# number of clusters - number of probability distributions we will use
#directory of folder with files in this directory all of the image files should be in subfolder volumes 0-49 and labels labels if one ill use lines below

train_labels = map(fileEntry-> joinpath(data_dir,"labelsTr",fileEntry),readdir(joinpath(data_dir,"labelsTr"); sort=true))
train_images = map(fileEntry-> joinpath(data_dir,"imagesTr",fileEntry),readdir(joinpath(data_dir,"imagesTr"); sort=true))


#zipping so we will have tuples with image and label names
zipped= collect(zip(train_images,train_labels))
zipped=map(tupl -> (replace(tupl[1], "._" => ""), replace(tupl[2], "._" => "")),zipped)
tupl=zipped[patentNum]






#proper loading
loaded = MedPipe3D.LoadFromMonai.loadBySitkromImageAndLabelPaths(tupl[1],tupl[2])
#!!!!!!!!!! important if you are just creating the hdf5 file  do it with "w" option otherwise do it with "r+"
#fid = h5open(pathToHDF5, "r+") 
gr= getGroupOrCreate(fid, patienGroupName)
#for this particular example we are intrested only in liver so we will keep only this label
labelArr=Float32.(map(entry-> UInt32(entry==1),loaded[2]))
#we save loaded and trnsformed data into HDF5 to avoid doing preprocessing every time
saveMaskBeforeVisualization(fid,patienGroupName,loaded[1],"image", "CT" )
saveMaskBeforeVisualization(fid,patienGroupName,labelArr,"labelSet", "boolLabel" )



# here we did default transformations so voxel dimension is set to 1,1,1 in any other case one need to set spacing attribute manually to proper value
# spacing can be found in metadata dictionary that is third entry in loadByMonaiFromImageAndLabelPaths output
# here metadata = loaded[3]
writeGroupAttribute(fid,patienGroupName, "spacing", [1,1,1])

#******************for display
#just needed so we will not have 2 same colors for two diffrent informations
listOfColorUsed= falses(18)

##below we define additional arrays that are not present in original data but will be needed for annotations and storing algorithm output 

#manual Modification array
manualModif = MedEye3d.ForDisplayStructs.TextureSpec{Float32}(# choosing number type manually to reduce memory usage
    name = "manualModif",
    isContinuusMask=true,
    colorSet = [getSomeColor(listOfColorUsed),getSomeColor(listOfColorUsed) ]

    # color = RGB(0.2,0.5,0.2) #getSomeColor(listOfColorUsed)# automatically choosing some contrasting color
    ,minAndMaxValue= Float32.([0,1]) #important to keep the same number type as chosen at the bagining
    ,isEditable = true ) # we will be able to manually modify this array in a viewer

manualModifB = MedEye3d.ForDisplayStructs.TextureSpec{Float32}(# choosing number type manually to reduce memory usage
    name = "manualModifB",
    isContinuusMask=true,
    colorSet = [getSomeColor(listOfColorUsed),getSomeColor(listOfColorUsed) ]

    # color = RGB(0.2,0.5,0.2) #getSomeColor(listOfColorUsed)# automatically choosing some contrasting color
    ,minAndMaxValue= Float32.([0,1]) #important to keep the same number type as chosen at the bagining
    ,isEditable = true ) # we will be able to manually modify this array in a viewer

algoVisualization = MedEye3d.ForDisplayStructs.TextureSpec{Float32}(
    name = "algoOutput",
    # we point out that we will supply multiple colors
    isContinuusMask=true,
    colorSet = [getSomeColor(listOfColorUsed),getSomeColor(listOfColorUsed) ]
    ,minAndMaxValue= Float32.([0,1])# values between 0 and 1 as this represent probabilities
   )

    addTextSpecs=Vector{MedEye3d.ForDisplayStructs.TextureSpec}(undef,3)
    addTextSpecs[1]=manualModif
    addTextSpecs[2]=manualModifB
    addTextSpecs[3]=algoVisualization


#2) primary display of chosen image 
mainScrollDat= loadFromHdf5Prim(fid,patienGroupName,addTextSpecs,listOfColorUsed)



# function getSimpleItkObject()
#     return sitk
# end

# """
# given directory (dirString) to file/files it will return the simple ITK image for futher processing
# isMHD when true - data in form of folder with dicom files
# isMHD  when true - we deal with MHD data
# """
# function getImageFromDirectory(dirString,isMHD::Bool, isDicomList::Bool)
#     #simpleITK object used to read from disk 
#     reader = sitk.ImageSeriesReader()
#     if(isDicomList)# data in form of folder with dicom files
#         dicom_names = reader.GetGDCMSeriesFileNames(dirString)
#         reader.SetFileNames(dicom_names)
#         return reader.Execute()
#     elseif(isMHD) #mhd file
#         return sitk.ReadImage(dirString)
#     end
# end#getPixelDataAndSpacing

# function permuteAndReverseFromSitk(pixels)
#     pixels = permutedims(pixels, (3, 2, 1))
#     sizz = size(pixels)
#     for i in 1:sizz[2]
#         for j in 1:sizz[3]
#             pixels[:, i, j] = reverse(pixels[:, i, j])
#         end# 
#     end# 
#     return pixels
# end#permuteAndReverse
# """
# resample to given size using sitk
# """

# function resamplesitkImageTosize(image, targetSpac, sitk, interpolator)

#     orig_spacing = pyconvert(Array, image.GetSpacing())
#     origSize = pyconvert(Array, image.GetSize())

#     new_size = (Int(round(origSize[1] * (orig_spacing[1] / targetSpac[1]))),
#         Int(round(origSize[2] * (orig_spacing[2] / targetSpac[2]))),
#         Int(round(origSize[3] * (orig_spacing[3] / targetSpac[3]))))

#     resample = sitk.ResampleImageFilter()
#     resample.SetOutputSpacing(targetSpac)
#     resample.SetOutputDirection(image.GetDirection())
#     resample.SetOutputOrigin(image.GetOrigin())
#     resample.SetTransform(sitk.Transform())
#     resample.SetDefaultPixelValue(image.GetPixelIDValue())
#     resample.SetInterpolator(interpolator)
#     resample.SetSize(new_size)
#     return resample.Execute(image)

# end

# """
# given file paths it loads 
# imagePath - path to main image
# labelPath - path to label
# also it make the spacing equal to target spacing and the orientation as RAS
# """
# function loadBySitkromImageAndLabelPaths(
#     imagePath, pet_path, targetSpacing=(1, 1, 1))

#     sitk = getSimpleItkObject()

#     image = sitk.ReadImage(imagePath)
#     pet = sitk.ReadImage(pet_path)

#     image = sitk.DICOMOrient(image, "RAS")
#     pet = sitk.DICOMOrient(pet, "RAS")

#     image = resamplesitkImageTosize(image, targetSpacing, sitk, sitk.sitkBSpline)
#     pet = resamplesitkImageTosize(pet, targetSpacing, sitk, sitk.sitkBSpline)

#     imageArr = permuteAndReverseFromSitk(pyconvert(Array, sitk.GetArrayFromImage(image)))
#     petArr = permuteAndReverseFromSitk(pyconvert(Array, sitk.GetArrayFromImage(pet)))

#     imageSize = image.GetSize()
#     labelSize = pet.GetSize()


#     return (imageArr, petArr, imageSize, imageSize, labelSize)

# end

# """
# padd with given value symmetrically to get the predifined target size and return padded image
# """
# function padToSize(image1, targetSize, paddValue, sitk)
#     currentSize = pyconvert(Array, image1.GetSize())
#     sizediffs = (targetSize[1] - currentSize[1], targetSize[2] - currentSize[2], targetSize[3] - currentSize[3])
#     halfDiffSize = (Int(floor(sizediffs[1] / 2)), Int(floor(sizediffs[2] / 2)), Int(floor(sizediffs[3] / 2)))
#     rest = (sizediffs[1] - halfDiffSize[1], sizediffs[2] - halfDiffSize[2], sizediffs[3] - halfDiffSize[3])
#     #print(f" currentSize {currentSize} targetSize {targetSize} halfDiffSize {halfDiffSize}  rest {rest} paddValue {paddValue} sizediffs {type(sizediffs)}")

#     # halfDiffSize=()
#     # rest=zeros(Int,rest)

#     return sitk.ConstantPad(image1, halfDiffSize, rest, paddValue)
#     #return sitk.ConstantPad(image1, (1,1,1), (1,1,1), paddValue)
# end #padToSize

# ctPixels,petPixels,  imageSize, labelSize =loadBySitkromImageAndLabelPaths(dirOfExample,dirOfExamplePET)

# # # in most cases dimensions of PET and CT data arrays will be diffrent in order to make possible to display them we need to resample and make dimensions equal
# # imagePET= getImageFromDirectory(dirOfExamplePET,true,false)
# # ctImage= getImageFromDirectory(dirOfExample,true,false)
# # pet_image_resampled = sitk.Resample(imagePET, ctImage)

# # ctPixels, ctSpacing = getPixelsAndSpacing(ctImage)
# # # In my case PET data holds 64 bit floats what is unsupported by Opengl
# # petPixels, petSpacing =getPixelsAndSpacing(pet_image_resampled)
# # purePetPixels, PurePetSpacing = getPixelsAndSpacing(imagePET)

# petPixels = Float32.(petPixels)
# ctPixels = ctPixels
# PurePetSpacing=(1, 1, 1)
# imageSize=pyconvert(Array,imageSize)
# imageSize=(imageSize[1],imageSize[2],imageSize[3])
# # we need to pass some metadata about image array size and voxel dimensions to enable proper display
# datToScrollDimsB= MedEye3d.ForDisplayStructs.DataToScrollDims(imageSize=  imageSize ,voxelSize=PurePetSpacing, dimensionToScroll = 3 );
# # example of texture specification used - we need to describe all arrays we want to display, to see all possible configurations look into TextureSpec struct docs .
# textureSpecificationsPETCT = [
# #   TextureSpec{Float32}(
# #       name = "PET",
# #       isNuclearMask=true,
# #       # we point out that we will supply multiple colors
# #       isContinuusMask=true,
# #       #by the number 1 we will reference this data by for example making it visible or not
# #       numb= Int32(1),
# #       colorSet = [RGB(0.0,0.0,0.0),RGB(1.0,1.0,0.0),RGB(1.0,0.5,0.0),RGB(1.0,0.0,0.0) ,RGB(1.0,0.0,0.0)]
# #       #display cutoff all values below 200 will be set 2000 and above 8000 to 8000 but only in display - source array will not be modified
# #       ,minAndMaxValue= Float32.([200,8000])
# #      ),
#   TextureSpec{Float32}(
#       name = "manualModif",
#       numb= Int32(2),
#       color = RGB(0.0,1.0,0.0)
#       ,minAndMaxValue= Float32.([0,1])
#       ,isEditable = true
#      ),

#      TextureSpec{Int16}(
#       name= "CTIm",
#       numb= Int32(3),
#       isMainImage = true,
#       minAndMaxValue= Int16.([0,100]))  
# ];
# # We need also to specify how big part of the screen should be occupied by the main image and how much by text fractionOfMainIm= Float32(0.8);
# fractionOfMainIm= Float32(0.8);

# import MedEye3d.DisplayWords.textLinesFromStrings

# mainLines= textLinesFromStrings(["main Line1", "main Line 2"]);
# supplLines=map(x->  textLinesFromStrings(["sub  Line 1 in $(x)", "sub  Line 2 in $(x)"]), 1:size(ctPixels)[3] );


# import MedEye3d.StructsManag.getThreeDims

# # tupleVect = [("PET",petPixels) ,("CTIm",ctPixels),("manualModif",zeros(UInt8,size(petPixels)) ) ]
# tupleVect = [("CTIm",ctPixels),("manualModif",zeros(Float32,imageSize)) ]
# slicesDat= getThreeDims(tupleVect )
# """
# Holds data necessary to display scrollable data
# """
# mainScrollDat = FullScrollableDat(dataToScrollDims =datToScrollDimsB
#                                  ,dimensionToScroll=1 # what is the dimension of plane we will look into at the beginning for example transverse, coronal ...
#                                  ,dataToScroll= slicesDat
#                                  ,mainTextToDisp= mainLines
#                                  ,sliceTextToDisp=supplLines );

# SegmentationDisplay.coordinateDisplay(textureSpecificationsPETCT ,fractionOfMainIm ,datToScrollDimsB ,1000);

# Main.SegmentationDisplay.passDataForScrolling(mainScrollDat);










# SegmentationDisplay

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

exampleLabel = "C:\\GitHub\\JuliaMedPipe\\data\\liverPrimData\\training-labels\\label\\liver-seg002.mhd"
exampleCTscann = "C:\\GitHub\\JuliaMedPipe\\data\\liverPrimData\\training-scans\\scan\\liver-orig002.mhd"



imagePureCT= getImageFromDirectory(exampleCTscann,true,false)
imageMask= getImageFromDirectory(exampleLabel,true,false)

ctPixelsPure, ctSpacingPure = getPixelsAndSpacing(imagePureCT)
maskPixels, maskSpacing =getPixelsAndSpacing(imageMask)

datToScrollDimsB= MedEye3d.ForDisplayStructs.DataToScrollDims(imageSize=  size(ctPixelsPure) ,voxelSize=ctSpacingPure, dimensionToScroll = 3 );
# example of texture specification used - we need to describe all arrays we want to display
listOfTexturesSpec = [
    TextureSpec{UInt8}(
        name = "goldStandardLiver",
        numb= Int32(1),
        color = RGB(1.0,0.0,0.0)
        ,minAndMaxValue= Int8.([0,1])
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

 fractionOfMainIm= Float32(0.8);
 """
 If we want to display some text we need to pass it as a vector of SimpleLineTextStructs 
 """
 import MedEye3d.DisplayWords.textLinesFromStrings
 
 mainLines= textLinesFromStrings(["main Line1", "main Line 2"]);
 supplLines=map(x->  textLinesFromStrings(["sub  Line 1 in $(x)", "sub  Line 2 in $(x)"]), 1:size(ctPixelsPure)[3] );
 
 """
 If we want to pass 3 dimensional array of scrollable data"""
 import MedEye3d.StructsManag.getThreeDims
 
 tupleVect = [("goldStandardLiver",maskPixels) ,("CTIm",ctPixelsPure),("manualModif",zeros(UInt8,size(ctPixelsPure)) ) ]
 slicesDat= getThreeDims(tupleVect )
 
 """
 Holds data necessary to display scrollable data
 """
 mainScrollDat = FullScrollableDat(dataToScrollDims =datToScrollDimsB
                                  ,dimensionToScroll=1 # what is the dimension of plane we will look into at the beginning for example transverse, coronal ...
                                  ,dataToScroll= slicesDat
                                  ,mainTextToDisp= mainLines
                                  ,sliceTextToDisp=supplLines );
 
 
 SegmentationDisplay.coordinateDisplay(listOfTexturesSpec ,fractionOfMainIm ,datToScrollDimsB ,1000);

 Main.SegmentationDisplay.passDataForScrolling(mainScrollDat);

using Base.Threads                               

function sum_multi_bad(a)
           s = 0
           Threads.@threads for i in a
               s += i
           end
           s
end

sum_multi_bad(1:10000000)
70140554652


sum_multi_bad(1:1_000_000000)

function f()
    a=a+1
end    

for i in 1:200
    @spawn :interactive f()
end
a


 using GLFW
 GLFW.PollEvents()
 using Base.Threads
 nthreads(:interactive)
 nthreads(:default)


#  apt install -y ubuntu-release-upgrader-core
#  do-release-upgrade -d