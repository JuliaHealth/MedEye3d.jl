######### Defining Helper Functions and imports

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
    sizz=size(pixels)

1#
2# all upside down
3
21# on 3 dim correct
12
13
23
32# liver on the left
32
23
123
321# 3 upside down
312
213# 1,2 good ; 3 upside down
231

321 ; 2#3 upside down

321 ; 1#3 ok 1 upside down
2;12 # 3 ok 1 and 2 upside down

2; 13 # 3 upside down
2; 3# 3 ok 1 and 2 upside down
2;2# 1,2, ok 3 upside down
2;23# liver on the left evrywhere

32;1 # all upside down
32;21 # 3 left to right
32;213 # 3 left to right
32;13# 3 is ok 1 and 2 upside down
32;1 # 3 upside down




32; 3#- upside down
21 ; 1 #- upside down

# for i in 1:sizz[3]
#     pixels[:,:,i] =  reverse(pixels[:,:,i])
#     end# 

#  for i in 1:sizz[2]
#     pixels[:,i,:] =  reverse(pixels[:,i,:])
#     end# 

#     for i in 1:sizz[1]
#         pixels[i,:,:] =  reverse(pixels[i,:,:])
#     end# 


  
    pixels=  permutedims(pixels, (3,2,1))
    sizz=size(pixels)


    for i in 1:sizz[1]
        for j in 1:sizz[3]
            pixels[i,:,j] =  reverse(pixels[i,:,j])
        end# 
    end# 



    # for i in 1:sizz[1]
    #     pixels[i,:,:] =  reverse(pixels[i,:,:])
    # end# 



    # for i in 1:sizz[1]
    #     pixels[i,:,:] =  reverse(pixels[i,:,:])
    # end# 
  
    # for i in 1:sizz[2]
    # pixels[:,i,:] =  reverse(pixels[:,i,:])
    # end# 

    # for i in 1:sizz[1]
    #     pixels[i,:,:] =  reverse(pixels[i,:,:])
    # end# 



            

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



# directories of PET/CT Data - from not published (yet) dataset - single example available from https://wwsi365-my.sharepoint.com/:f:/g/personal/s9956jm_ms_wwsi_edu_pl/Eq3cL7Md5bhPvnUlFLAMKZAB3nsbl6Q18fG96iVajvnNqA?e=bzX68X
dirOfExample ="C:\\GitHub\\JuliaMedPipe\\data\\PETphd\\slicerExp\\all17\\bad17NL-bad17NL\\20150518-PET^1_PET_CT_WholeBody_140_70_Recon (Adult)\\4-CT AC WB  1.5  B30f"
dirOfExamplePET ="C:\\GitHub\\JuliaMedPipe\\data\\PETphd\\slicerExp\\all17\\bad17NL-bad17NL\\20150518-PET^1_PET_CT_WholeBody_140_70_Recon (Adult)\\3-PET WB"



# in most cases dimensions of PET and CT data arrays will be diffrent in order to make possible to display them we need to resample and make dimensions equal
imagePET= getImageFromDirectory(dirOfExamplePET,false,true)
ctImage= getImageFromDirectory(dirOfExample,false,true)
pet_image_resampled = sitk.Resample(imagePET, ctImage)
purePetPixels, PurePetSpacing = getPixelsAndSpacing(imagePET)
ctPixels, ctSpacing = getPixelsAndSpacing(ctImage)
# In my case PET data holds 64 bit floats what is unsupported by Opengl
petPixels, petSpacing =getPixelsAndSpacing(pet_image_resampled)
petPixels = Float32.(petPixels)

# we need to pass some metadata about image array size and voxel dimensions to enable proper display
datToScrollDimsB= MedEye3d.ForDisplayStructs.DataToScrollDims(imageSize=  size(ctPixels) ,voxelSize=PurePetSpacing, dimensionToScroll = 3 );
# example of texture specification used - we need to describe all arrays we want to display
textureSpecificationsPETCT = [
  TextureSpec{Float32}(
      name = "PET",
      isNuclearMask=true,
      # we point out that we will supply multiple colors
      isContinuusMask=true,
      #by the number 1 we will reference this data by for example making it visible or not
      numb= Int32(1),
      colorSet = [RGB(0.0,0.0,0.0),RGB(1.0,1.0,0.0),RGB(1.0,0.5,0.0),RGB(1.0,0.0,0.0)]
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

"""
If we want to pass 3 dimensional array of scrollable data we need to supply it via vector ThreeDimRawDat's struct
utility function to make creation of those easier is getThreeDims which creates series of ThreeDimRawDat from list of tuples where
    first entry is String and second entry is 3 dimensional array with data 
    strings needs to be the same as we  defined in texture specifications at the bagining
    data arrays needs to be o the same size and be of the same type we specified in texture specification
"""
import MedEye3d.StructsManag
tupleVect = [("PET",petPixels) ,("CTIm",ctPixels),("manualModif",zeros(UInt8,size(petPixels)) ) ]
slicesDat= MedEye3d.StructsManag.getThreeDims(tupleVect )
"""
Holds data necessary to display scrollable data
"""
mainScrollDat = FullScrollableDat(dataToScrollDims =datToScrollDimsB
                                 ,dimensionToScroll=1 # what is the dimension of plane we will look into at the beginning for example transverse, coronal ...
                                 ,dataToScroll= slicesDat
                                 ,mainTextToDisp= mainLines
                                 ,sliceTextToDisp=supplLines );


tupleVect = [("PET",petPixels) ,("CTIm",ctPixels),("manualModif",zeros(UInt8,size(petPixels)) ) ]
slicesDat=  MedEye3d.StructsManag.getThreeDims(tupleVect )
"""
Holds data necessary to display scrollable data
"""
mainScrollDat = FullScrollableDat(dataToScrollDims =datToScrollDimsB
                                 ,dimensionToScroll=1 # what is the dimension of plane we will look into at the beginning for example transverse, coronal ...
                                 ,dataToScroll= slicesDat
                                 ,mainTextToDisp= mainLines
                                 ,sliceTextToDisp=supplLines );


"""
This function prepares all for display; 1000 in the end is responsible for setting window width for more look into SegmentationDisplay.coordinateDisplay
"""
SegmentationDisplay.coordinateDisplay(textureSpecificationsPETCT ,fractionOfMainIm ,datToScrollDimsB ,1200);
"""
as all is ready we can finally display image 
"""
Main.SegmentationDisplay.passDataForScrolling(mainScrollDat);
Main.SegmentationDisplay.mainActor.actor.currentDisplayedSlice = 800

#########Interactions


"""
Next all Interactions are done either by mouse or by keyboard shortcuts
left click and drag - will mark active texture (look below - set with alt ...) 
    if it is set to be modifiable in the texture specifications, to the set value and size (by tab...)
right click and drag - sets remembered position - when we will change plane of crossection
     for example from tranverse to coonal this point will be also visible on new plane
all keyboard shortcuts will be activated on RELEASE of keys or by pressing enter while still pressing
shift + number - make mask associated with given number visible
ctrl + number -  make mask associated with given number invisible 
alt + number -  make mask associated with given number active for mouse interaction 
tab + number - sets the number that will be  used as an input to masks modified by mouse
    when tab plus (and then no number) will be pressed it will increase stroke width
    when tab minus (and then no number) will be pressed it will increase stroke width
shift + numberA + "-"(minus sign) +numberB  - display diffrence between masks associated with numberA and numberB - also it makes automaticall mask A and B invisible
ctrl + numberA + "-"(minus sign) +numberB  - stops displaying diffrence between masks associated with numberA and numberB - also it makes automaticall mask A and B visible
space + 1 or 2 or 3 - change the plane of view (transverse, coronal, sagittal)
ctrl + z - undo last action
tab +/- increase or decrease stroke width
F1 - will display wide window for bone Int32(1000),Int32(-1000)
F2 - will display window for soft tissues Int32(400),Int32(-200)
F3 - will display wide window for lung viewing  Int32(0),Int32(-1000)
KEY_F4,  KEY_F5 -
    sets minimum (F4) and maximum (KEY_F5) value for display (with combination of + and minus signs - to increase or decrease given treshold) - 
        in case of continuus colors it will clamp values - so all above max will be equaled to max ; and min if smallert than min
        in case of main CT mask - it will controll min shown white and max shown black
        in case of maks with single color associated we will step data so if data is outside the rande it will return 0 - so will not affect display
"""




# ######### Benchmark PET/CT  
# #For transparency I include Below code used to benchark  PET/CT data

# window = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.window
# syncActor = Main.SegmentationDisplay.mainActor

# using GLFW,DataTypesBasic, ModernGL,Setfield
# using BenchmarkTools

# BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
# BenchmarkTools.DEFAULT_PARAMETERS.seconds =5000
# BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true


# function toBenchmarkScroll(toSc) 
#     MedEye3d.ReactToScroll.reactToScroll(toSc ,syncActor, false)
# end


# function toBenchmarkPaint(carts)
#     MedEye3d.ReactOnMouseClickAndDrag.reactToMouseDrag(MouseStruct(true,false, carts),syncActor )
# end


# function toBenchmarkPlaneTranslation(toScroll)
#     MedEye3d.ReactOnKeyboard.processKeysInfo(Option(toScroll),syncActor,KeyboardStruct(),false    )
#     OpenGLDisplayUtils.basicRender(syncActor.actor.mainForDisplayObjects.window)
#     glFinish()
# end


# function prepareRAndomCart(randInt) 
#     return [CartesianIndex(12+randInt,13+randInt),CartesianIndex(12+randInt,15+randInt),CartesianIndex(12+randInt,18+randInt),CartesianIndex(2+randInt,10+randInt),CartesianIndex(2+randInt,14+randInt)]
# end

# #we want some integers but not 0
# sc = @benchmarkable toBenchmarkScroll(y) setup=(y = filter(it->it!=0, rand(-5:5,20))[1]  )

# paint =  @benchmarkable toBenchmarkPaint(y) setup=(y =  prepareRAndomCart(rand(1:40,1)[1]  ))  

# translations =  @benchmarkable toBenchmarkPlaneTranslation(y) setup=(y = setproperties(syncActor.actor.onScrollData.dataToScrollDims,  (dimensionToScroll=rand(1:3,2)[1])) )  

# using BenchmarkPlots, StatsPlots
# # Define a parent BenchmarkGroup to contain our suite

# scrollingPETCT = run(sc)
# mouseInteractionPETCT = run(paint)
# translationsPETCT = run(translations)


# plot(scrollingPETCT)


# #### PURE CT image exaple , MHD files  taken from https://sliver07.grand-challenge.org/

# exampleLabel = "/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/data/liverPrimData/training-labels/label/liver-seg002.mhd"
# exampleCTscann = "/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/data/liverPrimData/training-scans/scan/liver-orig002.mhd"

# # loading data
# imagePureCT= getImageFromDirectory(exampleCTscann,true,false)
# imageMask= getImageFromDirectory(exampleLabel,true,false)


# ctPixelsPure, ctSpacingPure = getPixelsAndSpacing(imagePureCT)
# maskPixels, maskSpacing =getPixelsAndSpacing(imageMask)



# # we need to pass some metadata about image array size and voxel dimensions to enable proper display
# datToScrollDimsB= MedEye3d.ForDisplayStructs.DataToScrollDims(imageSize=  size(ctPixelsPure) ,voxelSize=ctSpacingPure, dimensionToScroll = 3 );
# # example of texture specification used - we need to describe all arrays we want to display
# listOfTexturesSpec = [
#     TextureSpec{UInt8}(
#         name = "goldStandardLiver",
#         numb= Int32(1),
#         color = RGB(1.0,0.0,0.0)
#         ,minAndMaxValue= Int8.([0,1])
#        ),
#     TextureSpec{UInt8}(
#         name = "manualModif",
#         numb= Int32(2),
#         color = RGB(0.0,1.0,0.0)
#         ,minAndMaxValue= UInt8.([0,1])
#         ,isEditable = true
#        ),

#        TextureSpec{Int16}(
#         name= "CTIm",
#         numb= Int32(3),
#         isMainImage = true,
#         minAndMaxValue= Int16.([0,100]))  
#  ];
# # We need also to specify how big part of the screen should be occupied by the main image and how much by text fractionOfMainIm= Float32(0.8);
# fractionOfMainIm= Float32(0.8);
# """
# If we want to display some text we need to pass it as a vector of SimpleLineTextStructs 
# """
# import MedEye3d.DisplayWords.textLinesFromStrings

# mainLines= textLinesFromStrings(["main Line1", "main Line 2"]);
# supplLines=map(x->  textLinesFromStrings(["sub  Line 1 in $(x)", "sub  Line 2 in $(x)"]), 1:size(ctPixelsPure)[3] );

# """
# If we want to pass 3 dimensional array of scrollable data"""
# import MedEye3d.StructsManag.getThreeDims

# tupleVect = [("goldStandardLiver",maskPixels) ,("CTIm",ctPixelsPure),("manualModif",zeros(UInt8,size(ctPixelsPure)) ) ]
# slicesDat= MedEye3d.StructsManag.getThreeDims(tupleVect )


# """
# Holds data necessary to display scrollable data
# """
# mainScrollDat = FullScrollableDat(dataToScrollDims =datToScrollDimsB
#                                  ,dimensionToScroll=1 # what is the dimension of plane we will look into at the beginning for example transverse, coronal ...
#                                  ,dataToScroll= slicesDat
#                                  ,mainTextToDisp= mainLines
#                                  ,sliceTextToDisp=supplLines );


# """
# This function prepares all for display; 1000 in the end is responsible for setting window width for more look into SegmentationDisplay.coordinateDisplay
# """
# SegmentationDisplay.coordinateDisplay(listOfTexturesSpec ,fractionOfMainIm ,datToScrollDimsB ,1000);
# """
# as all is ready we can finally display image 
# """
# Main.SegmentationDisplay.passDataForScrolling(mainScrollDat);


# ###########next part of benchmark

# #we want some integers but not 0
# scPureCt = @benchmarkable toBenchmarkScroll(y) setup=(y = filter(it->it!=0, rand(-5:5,20))[1]  )

# paintPureCt =  @benchmarkable toBenchmarkPaint(y) setup=(y =  prepareRAndomCart(rand(1:40,1)[1]  ))  

# translationsPureCt =  @benchmarkable toBenchmarkPlaneTranslation(y) setup=(y = setproperties(syncActor.actor.onScrollData.dataToScrollDims,  (dimensionToScroll=rand(1:3,2)[1])) )  

# using BenchmarkPlots, StatsPlots



# scB = @benchmarkable toBenchmarkScroll(y) setup=(y = filter(it->it!=0, rand(-5:5,20))[1]  )

# paintB =  @benchmarkable toBenchmarkPaint(y) setup=(y =  prepareRAndomCart(rand(1:40,1)[1]  ))  

# translationsB =  @benchmarkable toBenchmarkPlaneTranslation(y) setup=(y = setproperties(syncActor.actor.onScrollData.dataToScrollDims,  (dimensionToScroll=rand(1:3,2)[1])) )  

# using BenchmarkPlots, StatsPlots
# # Define a parent BenchmarkGroup to contain our suite

# scrollingPureCT = run(scB)
# mouseInteractionPureCT = run(paintB)
# translationsPureCT = run(translationsB)





#  using GLFW
#  GLFW.PollEvents()



