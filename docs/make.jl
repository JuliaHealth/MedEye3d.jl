using Documenter
using NuclearMedEye

makedocs(
    sitename = "NuclearMedEye",
    format = Documenter.HTML(),
    modules = [NuclearMedEye,NuclearMedEye.SegmentationDisplay
    ,NuclearMedEye.ReactingToInput
    ,NuclearMedEye.ReactOnKeyboard
    ,NuclearMedEye.ReactOnMouseClickAndDrag
    ,NuclearMedEye.ReactToScroll
    ,NuclearMedEye.PrepareWindow
    ,NuclearMedEye.TextureManag
    ,NuclearMedEye.DisplayWords
    ,NuclearMedEye.Uniforms
    ,NuclearMedEye.ShadersAndVerticiesForText
    ,NuclearMedEye.ShadersAndVerticies
    ,NuclearMedEye.OpenGLDisplayUtils
    ,NuclearMedEye.CustomFragShad
    ,NuclearMedEye.PrepareWindowHelpers
    ,NuclearMedEye.StructsManag
    ,NuclearMedEye.ForDisplayStructs
    ,NuclearMedEye.DataStructs
    ,NuclearMedEye.BasicStructs
    ,NuclearMedEye.ModernGlUtil]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#




using NuclearMedEye

using Base: String


using Conda
using PyCall
using Pkg
Pkg.build("HDF5")
using HDF5
Conda.pip_interop(true)
Conda.pip("install", "SimpleITK")
Conda.pip("install", "h5py")

sitk = pyimport("SimpleITK")
np= pyimport("numpy")


########
# interpolation adapted from https://discourse.itk.org/t/compose-image-from-different-modality-with-different-number-of-slices/2286/8

#dirOfExample = DrWatson.datadir("PETCT","manifest-1608669183333" ,"Lung-PET-CT-Dx","Lung_Dx-G0045","05-08-2011-NA-PET01PTheadlung Adult-09984","10.000000-Thorax  1.0  B70f-66628" )
#dirOfExample = DrWatson.datadir("data","PETphd","slicerExp" ,"CT","ScalarVolume_17")
# dirOfExample ="C:\\GitHub\\Probabilistic-medical-segmentation\\data\\PETphd\\slicerExp\\PETB\\ScalarVolume_17"
# dirOfExamplePET ="C:\\GitHub\\Probabilistic-medical-segmentation\\data\\PETphd\\slicerExp\\PETCT\\ScalarVolume_11"
dirOfExample ="C:\\GitHub\\JuliaMedPipe\\data\\PETphd\\slicerExp\\all17\\bad17NL-bad17NL\\20150518-PET^1_PET_CT_WholeBody_140_70_Recon (Adult)\\4-CT AC WB  1.5  B30f"
dirOfExamplePET ="C:\\GitHub\\JuliaMedPipe\\data\\PETphd\\slicerExp\\all17\\bad17NL-bad17NL\\20150518-PET^1_PET_CT_WholeBody_140_70_Recon (Adult)\\3-PET WB"






# data\\PETphd\\slicerExp\\CT\\ScalarVolume_17

# C:\\Users\\1\\Documents\\GitHub\\Probabilistic-medical-segmentation\\data\\PETphd\\slicerExp\\CT\\ScalarVolume_17

reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(dirOfExample)
reader.SetFileNames(dicom_names)

ctImage = reader.Execute()
pixelss = np.array(sitk.GetArrayViewFromImage(ctImage))

spacingsCt = ctImage.GetSpacing()
spacingListCT = [spacingsCt[3],spacingsCt[2],spacingsCt[1]]



dicom_namesPET = reader.GetGDCMSeriesFileNames(dirOfExamplePET)
reader = sitk.ImageSeriesReader()

reader.SetFileNames(dicom_namesPET)

imagePET = reader.Execute()
pixelsPET = np.array(sitk.GetArrayViewFromImage(imagePET))

spacingsPET = imagePET.GetSpacing()
spacingListPET = [spacingsPET[1],spacingsPET[2],spacingsPET[3]]




# Resample PET onto CT grid using default interpolator and identity transformation.

function permuteAndReverse(pixels)
  pixels=  permutedims(pixels, (3,2,1))
  sizz=size(pixels)
  for i in 1:sizz[1]
      pixels[i,:,:] =  reverse(pixels[i,:,:])
  end# 

  for i in 1:sizz[2]
  pixels[:,i,:] =  reverse(pixels[:,i,:])
  end# 
  return pixels
end#permuteAndReverse

pet_image_resampled = sitk.Resample(imagePET, ctImage)

resampledPet = pet_image_resampled
pixelsResampled = np.array(sitk.GetArrayViewFromImage(resampledPet))

pixelsResampled = permuteAndReverse(pixelsResampled)
pixelssB = permuteAndReverse(pixelss)

typeof(pixelssB)
typeof(pixelsResampled)


import NuclearMedEye
import NuclearMedEye.ForDisplayStructs
import NuclearMedEye.ForDisplayStructs.TextureSpec
using ColorTypes
datToScrollDimsB= NuclearMedEye.ForDisplayStructs.DataToScrollDims(imageSize=  size(pixelsResampled) ,voxelSize= (spacingListPET[1],spacingListPET[2],spacingListPET[3]) , dimensionToScroll = 3 );




listOfTexturesToCreateB = [
  TextureSpec{Float32}(
      name = "PET",
      isNuclearMask=true,
      isContinuusMask=true,
      numb= Int32(1),
      colorSet = [RGB(0.0,0.0,0.0),RGB(1.0,1.0,0.0),RGB(1.0,0.5,0.0),RGB(1.0,0.0,0.0) ,RGB(1.0,0.0,0.0)]
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
#   
fractionOfMainIm= Float32(0.8);

import NuclearMedEye.SegmentationDisplay

SegmentationDisplay.coordinateDisplay(listOfTexturesToCreateB ,fractionOfMainIm ,datToScrollDimsB ,1000);

SegmentationDisplay.mainActor.actor.currentDisplayedSlice=40
import NuclearMedEye.DisplayWords.textLinesFromStrings
import NuclearMedEye.DataStructs.ThreeDimRawDat
import NuclearMedEye.DataStructs.FullScrollableDat
import NuclearMedEye.ForDisplayStructs.KeyboardStruct
import NuclearMedEye.ForDisplayStructs.MouseStruct
import NuclearMedEye.ForDisplayStructs.ActorWithOpenGlObjects

mainLines= textLinesFromStrings(["main Line1", "main Line 2"]);
supplLines=map(x->  textLinesFromStrings(["sub  Line 1 in $(x)", "sub  Line 2 in $(x)"]), 1:size(pixelsResampled)[3] );






# slicesDatB=  [ThreeDimRawDat{Int16}(Int16,"goldStandardLiver",pixelsResampledB)
# ,ThreeDimRawDat{Int16}(Int16,"CTIm",pixelssB)
# ,ThreeDimRawDat{UInt8}(UInt8,"manualModif",zeros(UInt8,size(pixelssB)  ))
#  ];
slicesDatB=  [ThreeDimRawDat{Float32}(Float32,"PET",pixelsResampled)
,ThreeDimRawDat{Int16}(Int16,"CTIm",pixelssB)
,ThreeDimRawDat{UInt8}(UInt8,"manualModif",zeros(UInt8,size(pixelsResampled)))
 ];


mainScrollDatB = FullScrollableDat(dataToScrollDims =datToScrollDimsB
                                 ,dimensionToScroll=1
                                 ,dataToScroll= slicesDatB
                                 ,mainTextToDisp= mainLines
                                 ,sliceTextToDisp=supplLines );


Main.SegmentationDisplay.passDataForScrolling(mainScrollDatB);

window = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.window
syncActor = Main.SegmentationDisplay.mainActor
# mouseCallbackSubscribable = Main.SegmentationDisplay.mainActor.actor.mouseCallbackSubscribable
# keyboardCallbackSubscribable = Main.SegmentationDisplay.mainActor.actor.keyboardCallbackSubscribable
# scrollCallbackSubscribable = Main.SegmentationDisplay.mainActor.actor.scrollCallbackSubscribable

using GLFW,DataTypesBasic


GLFW.PollEvents()


using BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.seconds =5000
BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true

#Cross Section plane translation


#will need to divide by 2 
function toBenchmarkScroll(toSc) 
    NuclearMedEye.ReactToScroll.reactToScroll(toSc ,syncActor, false)
end

  rand(1:10,10,10)


#Response to mouse interaction
carts = [CartesianIndex(12,13),CartesianIndex(12,15),CartesianIndex(12,18),CartesianIndex(2,10),CartesianIndex(2,14)]


function toBenchmarkPaint()
    NuclearMedEye.ReactOnMouseClickAndDrag.reactToMouseDrag(MouseStruct(true,false, carts),syncActor )
end


#Response to section plane change
function toBenchmarkPaint()
    NuclearMedEye.ReactOnKeyboard.processKeysInfo(Option(DataToScrollDims(dimensionToScroll=1)),syncActor,KeyboardStruct(),false    )
    NuclearMedEye.ReactOnKeyboard.processKeysInfo(Option(DataToScrollDims(dimensionToScroll=2)),syncActor,KeyboardStruct(),false    )
end



dragHandl = NuclearMedEye.ReactOnMouseClickAndDrag.mouseClickHandler

@btime 1+1

GLFW.SetCursorPos(window,100,100)















