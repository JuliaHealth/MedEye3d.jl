# MedEye3d
Main goal of the package is conviniently visualize 3d medical imaging to make segmentation simpler


## Important
newer simpler example together with some extras in https://github.com/jakubMitura14/MedPipe3DTutorial



Detailed description can be found in published article 
http://zeszyty-naukowe.wwsi.edu.pl/zeszyty/zeszyt25/Jakub_Mitura_Beata_E.Chrapko_WWSI_nr_25.pdf

In order to use package just type in Repl

```
]add MedEye3d
```

the tool is part of bigger framework - example in link below
https://github.com/jakubMitura14/MedPipe3DTutorial



Image below just represents limitless possibilities of color ranges, and that thanks to OpenGl even theorethically complex data to display will render nearly instantenously. 


![image](https://user-images.githubusercontent.com/53857487/131262103-4662bf13-11ca-42a7-836e-a89eb6d17c82.png)

##
Below the functionality of the package will be described on the basis of some examples
In case of any questions, feature requests, or propositions of cooperation  post them here on Github or contact me via LinkedIn linkedin.com/in/jakub-mitura-7b2013151

You can also look into my **article** that is currently in review describing this package https://www.overleaf.com/read/dwzpdwrgspts - If you have any thoughts, comments or propositions of improvemens  about the article please let me know for example via linkedin.com/in/jakub-mitura-7b2013151 or here on github.

## Defining Helper Functions and imports
```
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

Directories - obviously you need to provide path to place where it is stored on your disk. You can download PET/CT data from. You can download example data from https://wwsi365-my.sharepoint.com/:f:/g/personal/s9956jm_ms_wwsi_edu_pl/Eq3cL7Md5bhPvnUlFLAMKZAB3nsbl6Q18fG96iVajvnNqA?e=bzX68X

```
# directories of PET/CT Data - from https://wiki.cancerimagingarchive.net/display/Public/Head-Neck-PET-CT
dirOfExample ="C:\\GitHub\\JuliaMedPipe\\data\\PETphd\\slicerExp\\all17\\bad17NL-bad17NL\\20150518-PET^1_PET_CT_WholeBody_140_70_Recon (Adult)\\4-CT AC WB  1.5  B30f"
dirOfExamplePET ="C:\\GitHub\\JuliaMedPipe\\data\\PETphd\\slicerExp\\all17\\bad17NL-bad17NL\\20150518-PET^1_PET_CT_WholeBody_140_70_Recon (Adult)\\3-PET WB"

# in most cases dimensions of PET and CT data arrays will be diffrent in order to make possible to display them we need to resample and make dimensions equal
imagePET= getImageFromDirectory(dirOfExamplePET,false,true)
ctImage= getImageFromDirectory(dirOfExample,false,true)
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
```
So after invoking this function one should see image sth like below

![image](https://user-images.githubusercontent.com/53857487/131359926-56d2ac89-1754-4b05-9c38-3c00d990c404.png)

## Interactions

Interactions are summarized in video https://youtu.be/tv7-nGiik-w

Next all Interactions are done either by mouse or by keyboard shortcuts

left click and drag - will mark active texture (look below - set with alt ...) 
    if it is set to be modifiable in the texture specifications, to the set value and size (by tab...)
right click and drag - sets remembered position - when we will change plane of crossection
     for example from tranverse to coonal this point will be also visible on new plane

all keyboard shortcuts will be activated on RELEASE of keys or by pressing enter while still pressing other; +,- and z keys acts also like enter 

f key - fast scrolling

s key - slow scrolling

shift + number - make mask associated with given number visible


ctrl + number -  make mask associated with given number invisible 


alt + number -  make mask associated with given number active for mouse interaction 


tab + number - sets the number that will be  used as an input to masks modified by mouse
  
  when tab plus (and then no number) will be pressed it will increase stroke width
  
  when tab minus (and then no number) will be pressed it will increase stroke width
    
    
shift + numberA + "m"(m letter) +numberB  - display diffrence between masks associated with numberA and numberB - also it makes automaticall mask A and B invisible


ctrl + numberA + "-"(minus sign) +numberB  - stops displaying diffrence between masks associated with numberA and numberB - also it makes automaticall mask A and B visible


space + 1 or 2 or 3 - change the plane of view (transverse, coronal, sagittal)


ctrl + z - undo last action


tab +/- increase or decrease stroke width


F1 - will display wide window for bone Int32(1000),Int32(-1000)


F2 - will display window for soft tissues Int32(400),Int32(-200)


F3 - will display wide window for lung viewing  Int32(0),Int32(-1000)

F4,  F5 sets minimum (F4) and maximum (KEY_F5) value for display (with combination of + and minus signs - to increase or decrease given treshold) - 

In case of continuus colors it will clamp values - so all above max will be equaled to max ; and min if smallert than min

In case of main CT mask - it will controll min shown white and max shown black

In case of maks with single color associated we will step data so if data is outside the rande it will return 0 - so will not affect display
F6 - controlls contribution  of given mask to the overall image - maximum value is 1 minimum 0 if we have 3 masks and all control contribution is set to 1 and all are visible their corresponding influence to pixel color is 33%  if plus is pressed it will increse contribution by 0.1   if minus is pressed it will decrease contribution by 0.1  



## Benchmark PET/CT  

For transparency I include Below code used to benchark  PET/CT data
```
window = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.window
syncActor = Main.SegmentationDisplay.mainActor

using GLFW,DataTypesBasic, ModernGL,Setfield
using BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.seconds =5000
BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true


function toBenchmarkScroll(toSc) 
    MedEye3d.ReactToScroll.reactToScroll(toSc ,syncActor, false)
end


function toBenchmarkPaint(carts)
    MedEye3d.ReactOnMouseClickAndDrag.reactToMouseDrag(MouseStruct(true,false, carts),syncActor )
end


function toBenchmarkPlaneTranslation(toScroll)
    MedEye3d.ReactOnKeyboard.processKeysInfo(Option(toScroll),syncActor,KeyboardStruct(),false    )
    OpenGLDisplayUtils.basicRender(syncActor.actor.mainForDisplayObjects.window)
    glFinish()
end


function prepareRAndomCart(randInt) 
    return [CartesianIndex(12+randInt,13+randInt),CartesianIndex(12+randInt,15+randInt),CartesianIndex(12+randInt,18+randInt),CartesianIndex(2+randInt,10+randInt),CartesianIndex(2+randInt,14+randInt)]
end

#we want some integers but not 0
sc = @benchmarkable toBenchmarkScroll(y) setup=(y = filter(it->it!=0, rand(-5:5,20))[1]  )

paint =  @benchmarkable toBenchmarkPaint(y) setup=(y =  prepareRAndomCart(rand(1:40,1)[1]  ))  

translations =  @benchmarkable toBenchmarkPlaneTranslation(y) setup=(y = setproperties(syncActor.actor.onScrollData.dataToScrollDims,  (dimensionToScroll=rand(1:3,2)[1])) )  

using BenchmarkPlots, StatsPlots
# Define a parent BenchmarkGroup to contain our suite

scrollingPETCT = run(sc)
mouseInteractionPETCT = run(paint)
translationsPETCT = run(translations)


plot(scrollingPETCT)
```
If all will be ready you should see sth like on the image below


##  PURE CT image exaple , MHD file
Files  taken from https://sliver07.grand-challenge.org/
As previosly adjust path to your case
```
exampleLabel = "C:\\GitHub\\JuliaMedPipe\\data\\liverPrimData\\training-labels\\label\\liver-seg002.mhd"
exampleCTscann = "C:\\GitHub\\JuliaMedPipe\\data\\liverPrimData\\training-scans\\scan\\liver-orig002.mhd"
```
Loading data
``` 
imagePureCT= getImageFromDirectory(exampleCTscann,true,false)
imageMask= getImageFromDirectory(exampleLabel,true,false)

ctPixelsPure, ctSpacingPure = getPixelsAndSpacing(imagePureCT)
maskPixels, maskSpacing =getPixelsAndSpacing(imageMask)
```
We need to pass some metadata about image array size and voxel dimensions to enable proper display

```
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
``` 
We need also to specify how big part of the screen should be occupied by the main image and how much by text fractionOfMainIm= Float32(0.8);
```
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


```
This function prepares all for display; 1000 in the end is responsible for setting window width for more look into SegmentationDisplay.coordinateDisplay
```
SegmentationDisplay.coordinateDisplay(listOfTexturesSpec ,fractionOfMainIm ,datToScrollDimsB ,1000);
```
As all is ready we can finally display image 
```
Main.SegmentationDisplay.passDataForScrolling(mainScrollDat);

```
## Next part of benchmark for pure CT
```
#we want some integers but not 0
scPureCt = @benchmarkable toBenchmarkScroll(y) setup=(y = filter(it->it!=0, rand(-5:5,20))[1]  )

paintPureCt =  @benchmarkable toBenchmarkPaint(y) setup=(y =  prepareRAndomCart(rand(1:40,1)[1]  ))  

translationsPureCt =  @benchmarkable toBenchmarkPlaneTranslation(y) setup=(y = setproperties(syncActor.actor.onScrollData.dataToScrollDims,  (dimensionToScroll=rand(1:3,2)[1])) )  

using BenchmarkPlots, StatsPlots



scB = @benchmarkable toBenchmarkScroll(y) setup=(y = filter(it->it!=0, rand(-5:5,20))[1]  )

paintB =  @benchmarkable toBenchmarkPaint(y) setup=(y =  prepareRAndomCart(rand(1:40,1)[1]  ))  

translationsB =  @benchmarkable toBenchmarkPlaneTranslation(y) setup=(y = setproperties(syncActor.actor.onScrollData.dataToScrollDims,  (dimensionToScroll=rand(1:3,2)[1])) )  

using BenchmarkPlots, StatsPlots
# Define a parent BenchmarkGroup to contain our suite

scrollingPureCT = run(scB)
mouseInteractionPureCT = run(paintB)
translationsPureCT = run(translationsB)
```

If You will find usefull my work please cite it 
```
@Article{Mitura2021,
  author   = {Mitura, Jakub and Chrapko, Beata E.},
  journal  = {Zeszyty Naukowe WWSI},
  title    = {{3D Medical Segmentation Visualization in Julia with MedEye3d}},
  year     = {2021},
  number   = {25},
  pages    = {57--67},
  volume   = {15},
  doi      = {10.26348/znwwsi.25.57},
  keywords = {OpenGl, Computer Tomagraphy, PET/CT, medical image annotation, medical image visualization},
}

```



References (numbers are assoiated with the article that is in review - here just to state  them)

[1]    Jeff  Bezanson  et  al.  “Julia:  A  fresh  approach  to  numerical  computing”.  In:SIAM  Review59.1(2017),  pp.  65–98.doi:10.1137/141000671.url:https://epubs.siam.org/doi/10.1137/141000671.

[2]    George Datseris et al. “DrWatson: the perfect sidekick for your scientific inquiries”. In:Journal ofOpen Source Software5.54 (2020), p. 2673.doi:10.21105/joss.02673.url:https://doi.org/10.21105/joss.02673.7


[3]    Bagaev Dmitry.Rocket.jl. 2021.url:https://github.com/biaslab/Rocket.jl.

[4]    Andy Ferris.Dictionaries.jl. 2021.url:https://github.com/andyferris/Dictionaries.jl.

[5]    jorge-brito.Glutils.jl. 2021.url:https://github.com/jorge-brito/Glutils.jl.

[6]    Ron Kikinis, Steve Pieper, and Kirby Vosburgh.3D Slicer: A Platform for Subject-Specific ImageAnalysis,  Visualization,  and  Clinical  Support.  Vol.  3.  Jan.  2014,  pp.  277–289.isbn:  978-1-4614-7656-6.doi:10.1007/978-1-4614-7657-3_19.

[7]    Andreas  Markus  Loening  and  Sanjiv  Sam  Gambhir.  “AMIDE:  A  Free  Software  Tool  for  Multi-modality Medical Image Analysis”. In:Molecular Imaging2.3 (2003), p. 15353500200303133.doi:10.1162/15353500200303133.

[8]    Mauro.Parameters.jl. 2021.url:https://github.com/mauro3/Parameters.jl.

[9]    Jarrett Revels.BenchmarkTools.jl. 2021.url:https://github.com/JuliaCI/BenchmarkTools.jl.

[10]    aaalexandrov SimonDanisch.FreeTypeAbstraction.jl. 2021.url:https://github.com/JuliaGraphics/FreeTypeAbstraction.jl.

[11]    o-jasper  SimonDanisch  rennis250.ModernGL.jl.  2021.url:https : / / github . com / JuliaGL /ModernGL.jl.

[12]    Physycians SPSK4.Sample PET/CT Data. 2021.url:https://wwsi365-my.sharepoint.com/:f:/g/personal/s9956jm_ms_wwsi_edu_pl/Eq3cL7Md5bhPvnUlFLAMKZAB3nsbl6Q18fG96iVajvnNqA?e=bzX68X.

[13]    Kevin Squire.Match.jl. 2021.url:https://github.com/kmsquire/Match.jl.

[14]    Martin Styner Tobias Heimann Bram van Ginneken.SILVER07. 2021.url:http://www.sliver07.org/.

[15]    Jan Weidner.Setfield.jl. 2021.url:https://github.com/jw3126/Setfield.jl.

[16]    Mason Woo et al.OpenGL programming guide: the official guide to learning OpenGL, version 1.2.Addison-Wesley Longman Publishing Co., Inc., 1999.



