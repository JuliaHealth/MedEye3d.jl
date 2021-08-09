using Base: Int16

     using DrWatson
     @quickactivate "Probabilistic medical segmentation"
     
     using Setfield
     using GLFW
     using ModernGL
     using ColorTypes
     using Glutils

     include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/structs/forDisplayStructs.jl")
     include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/loadData/manageH5File.jl")
     using Main.h5manag
     include(DrWatson.scriptsdir("display","GLFW","shadersEtc","CustomFragShad.jl"))

     include(DrWatson.scriptsdir("display","GLFW","startModules","PrepareWindowHelpers.jl"))
     include(DrWatson.scriptsdir("display","GLFW","modernGL","OpenGLDisplayUtils.jl"))
     include(DrWatson.scriptsdir("display","GLFW","shadersEtc","ShadersAndVerticies.jl"))
     include(DrWatson.scriptsdir("display","GLFW","shadersEtc","ShadersAndVerticiesForText.jl"))



     include(DrWatson.scriptsdir("display","GLFW","shadersEtc","Uniforms.jl"))
     
     include(DrWatson.scriptsdir("display","GLFW","modernGL","TextureManag.jl") )
     include(DrWatson.scriptsdir("display","GLFW","startModules","PrepareWindow.jl"))
      

     include(DrWatson.scriptsdir("display","reactingToMouseKeyboard","ReactToScroll.jl") )
     include(DrWatson.scriptsdir("display","reactingToMouseKeyboard","ReactOnMouseClickAndDrag.jl") )

     include(DrWatson.scriptsdir("display","reactingToMouseKeyboard","ReactingToInput.jl") )
     include(DrWatson.scriptsdir("display","GLFW","SegmentationDisplay.jl"))




using Revise 
using Main.SegmentationDisplay
#data about textures we want to create
using  Main.ForDisplayStructs
using Parameters

# list of texture specifications, important is that main texture - main image should be specified first
#Order is important !
listOfTexturesToCreate = [
Main.ForDisplayStructs.TextureSpec(
    name = "mainLab",
    dataType= UInt8,
    strokeWidth = 5,
    color = RGB(1.0,0.0,0.0)
   ),
Main.ForDisplayStructs.TextureSpec(
    name = "testLab1",
    numb= Int32(1),
    dataType= UInt8,
    color = RGB(0.0,1.0,0.0)
   ),
    Main.ForDisplayStructs.TextureSpec(
    name = "testLab2",
    numb= Int32(2),
    dataType= UInt8,
    color = RGB(0.0,0.0,1.0)
     ),
     Main.ForDisplayStructs.TextureSpec(
      name = "textText",
      isTextTexture = true,
      dataType= UInt8,
      color = RGB(0.0,0.0,1.0)
    ),
    Main.ForDisplayStructs.TextureSpec(
    name= "CTIm",
    numb= Int32(3),
    isMainImage = true,
    dataType= Int16)  
      ]



      
    include(DrWatson.scriptsdir("display","reactingToMouseKeyboard","ReactingToInput.jl"))
    using Main.ReactingToInput

    segmPath = DrWatson.scriptsdir("display","GLFW","SegmentationDisplay.jl")
    include(segmPath)
    

    using Main.ForDisplayStructs
    using Main.ReactToScroll
    using Main.SegmentationDisplay
    using Rocket
    
    
#data source
# exampleDat = Int16.(Main.h5manag.getExample())
# exampleLabels = UInt8.(Main.h5manag.getExampleLabels())
# dims = size(exampleDat)
# widthh=dims[2]
# heightt=dims[3]
# slicesNumb= dims[1]

#  slice = 200
#   listOfDataAndImageNamesSlice = [("mainLab",exampleLabels[slice,:,:]),("CTIm",exampleDat[slice,:,:] )]


#     listOfDataAndImageNames = [("mainLab",exampleLabels),("CTIm",exampleDat)]
  
    
#     imagedims=dims
#     imageWidth = dims[2]
#     imageHeight = dims[3]
#configuring
   # Main.SegmentationDisplay.coordinateDisplay(listOfTexturesToCreate,512,512, 1000,800)

 
    # Main.SegmentationDisplay.passDataForScrolling(listOfDataAndImageNames)

    # Main.SegmentationDisplay.updateSingleImagesDisplayed(listOfDataAndImageNamesSlice,200 )


#  window = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.window
    # stopListening = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.stopListening
    
  #  textSpec = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[1]
    # textSpecB = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[2]



    Main.SegmentationDisplay.coordinateDisplay(listOfTexturesToCreate,40,40, 1000,800)


   ### playing with uniforms
   program = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.shader_program
   shader_program =program
    



###### main data ...





testLab1Dat =UInt8.(map(xx-> (xx >0 ? 1 : 0), rand(Int8,10,40,40)))
testLab2Dat= UInt8.(map(xx-> (xx >0 ? 1 : 0), rand(Int8,10,40,40)))

mainMaskDummy = UInt8.(map(xx-> (xx >0 ? 1 : 0), rand(Int8,10,40,40)))
# testLab1Dat =rand(UInt8,10,40,40)
# testLab2Dat= rand(UInt8,10,40,40)

# mainMaskDummy = rand(UInt8,10,40,40)


ctDummy =  Int16.(map(xx-> (xx >0 ? 1 : 0), rand(Int16,10,40,40)))# will give white background for testing 
    listOfDataAndImageNames = [("mainLab",mainMaskDummy)
    ,("CTIm",ctDummy )  
    ,("testLab2",testLab2Dat) 
    ,("testLab1",testLab1Dat) ]
    #,("testLab2",zeros(Int8,10,512,512))

    Main.SegmentationDisplay.passDataForScrolling(listOfDataAndImageNames)

    slicee = 3
    listOfDataAndImageNamesSlice = [ ("mainLab",mainMaskDummy[slicee,:,:]) ,  ("testLab2",testLab2Dat[slicee,:,:]) 
      ,("testLab1",testLab1Dat[slicee,:,:]) 
    ,("CTIm",ctDummy[slicee,:,:] )]


aa=zeros(40,40)
bb=zeros(40,40)
cc=zeros(40,40)
aa[1,1]=1
bb[1,1]=1
cc[1,1]=1

    listOfDataAndImageNamesSlice = [ ("mainLab",aa) ,
      ("testLab2",bb) 
      ,("testLab1",cc)     ,("CTIm",ctDummy[slicee,:,:] )]


       Main.SegmentationDisplay.updateSingleImagesDisplayed(listOfDataAndImageNamesSlice,3 )



    window = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.window
    

    textLiverMain = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[1]
    textureB = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[2]
    textureC = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[3]
    textTexture = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[4]
    textureCt = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[5]
    
    Main.SegmentationDisplay.mainActor.actor.textureToModifyVec= [textLiverMain]
    dattt = Main.SegmentationDisplay.mainActor.actor.onScrollData[4][2]
maximum(dattt)



updateTexture(dattt,textureB )

basicRender(window)

using Main.Uniforms, Main.OpenGLDisplayUtils

    
textLiverMain.uniforms.colorsMaskRef
textTexture.uniforms.colorsMaskRef


  setTextureVisibility(true ,textLiverMain.uniforms)
  setMaskColor(RGB(1.0,1.0,0.0) ,textLiverMain.uniforms)

  setMaskColor(RGB(0.0,1.0,0.0) ,textTexture.uniforms)
  setTextureVisibility(true ,textTexture.uniforms)

    setMaskColor(RGB(1.0,0.0,0.5) ,textureC.uniforms)
    setTextureVisibility(true ,textureC.uniforms)


  setMaskColor(RGB(0.5,0.5,0.0) ,textureB.uniforms)
  setTextureVisibility(true ,textureB.uniforms)
  setTextureVisibility(true,textureCt.uniforms)


  #   basicRender(window)

 setCTWindow(Int32(0), Int32(0),Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[5].uniforms)




using Main.CustomFragShad
    strr= Main.CustomFragShad.createCustomFramgentShader(listOfTexturesToCreate)
    for st in split(strr, "\n")
    @info st
    end
    using Main.ShadersAndVerticies


using Main.ForDisplayStructs
    
  ks =KeyboardStruct(false,false,false, false, [],Int32(3),"", GLFW.PRESS)
  ks2=setproperties(ks, (isCtrlPressed=true))

    
    scCode = @match ks begin
      KeyboardStruct(false,false,false, false, [],Int32(3),"", GLFW.PRESS) => print("ctrl false")
      KeyboardStruct(true,false,false, false, [],Int32(3),"", GLFW.PRESS) => print("ctrl true")
      _ => "notImp" # not Important
   end
    



x = Option("hi new")

bar(Option())


   GLFW.PollEvents()



