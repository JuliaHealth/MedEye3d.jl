
######### Defining Helper Functions and imports
--

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


######### Benchmark PET/CT  
#For transparency I include Below code used to benchark  PET/CT data

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


#### PURE CT image exaple , MHD files  taken from https://sliver07.grand-challenge.org/

exampleLabel = "C:\\GitHub\\JuliaMedPipe\\data\\liverPrimData\\training-labels\\label\\liver-seg002.mhd"
exampleCTscann = "C:\\GitHub\\JuliaMedPipe\\data\\liverPrimData\\training-scans\\scan\\liver-orig002.mhd"

# loading data
imagePureCT= getImageFromDirectory(exampleCTscann,true,false)
imageMask= getImageFromDirectory(exampleLabel,true,false)


ctPixelsPure, ctSpacingPure = getPixelsAndSpacing(imagePureCT)
maskPixels, maskSpacing =getPixelsAndSpacing(imageMask)



# we need to pass some metadata about image array size and voxel dimensions to enable proper display
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
# We need also to specify how big part of the screen should be occupied by the main image and how much by text fractionOfMainIm= Float32(0.8);
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


"""
This function prepares all for display; 1000 in the end is responsible for setting window width for more look into SegmentationDisplay.coordinateDisplay
"""
SegmentationDisplay.coordinateDisplay(listOfTexturesSpec ,fractionOfMainIm ,datToScrollDimsB ,1000);
"""
as all is ready we can finally display image 
"""
Main.SegmentationDisplay.passDataForScrolling(mainScrollDat);


###########next part of benchmark

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





 using GLFW
 GLFW.PollEvents()



