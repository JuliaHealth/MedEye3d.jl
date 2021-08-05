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
    name = "grandTruthLiverLabel",
    dataType= UInt8,
    strokeWidth = 5,
    color = RGB(1.0,0.0,0.0)
   ),
Main.ForDisplayStructs.TextureSpec(
    name = "mainForModificationsTexture1",
    dataType= UInt8,
    color = RGB(0.0,1.0,0.0)
   ),
    Main.ForDisplayStructs.TextureSpec(
    name = "mainForModificationsTexture2",
    dataType= UInt8,
    color = RGB(0.0,0.0,1.0)
     )      
    ,Main.ForDisplayStructs.TextureSpec(
    name= "mainCTImage",
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
#   listOfDataAndImageNamesSlice = [("grandTruthLiverLabel",exampleLabels[slice,:,:]),("mainCTImage",exampleDat[slice,:,:] )]


#     listOfDataAndImageNames = [("grandTruthLiverLabel",exampleLabels),("mainCTImage",exampleDat)]
  
    
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

# mainForModificationsTexture1Dat =UInt8.(round.(rand(10,40,40)))
# mainForModificationsTexture2Dat= UInt8.(round.(rand(10,40,40)))


mainForModificationsTexture1Dat =zeros(UInt8,10,40,40)
mainForModificationsTexture2Dat= zeros(UInt8,10,40,40)

mainMaskDummy = zeros(UInt8,10,40,40)

# mainForModificationsTexture1Dat =rand(UInt8,10,40,40)
# mainForModificationsTexture2Dat= rand(UInt8,10,40,40)

# mainMaskDummy = rand(UInt8,10,40,40)

ctDummy = ones(Int16,10,40,40)# will give white background for testing 
    listOfDataAndImageNames = [("grandTruthLiverLabel",mainMaskDummy)
    ,("mainCTImage",ctDummy )  
    ,("mainForModificationsTexture2",mainForModificationsTexture2Dat) 
    ,("mainForModificationsTexture1",mainForModificationsTexture1Dat) ]
    #,("mainForModificationsTexture2",zeros(Int8,10,512,512))

    Main.SegmentationDisplay.passDataForScrolling(listOfDataAndImageNames)
    slicee = 3
    listOfDataAndImageNamesSlice = [ ("grandTruthLiverLabel",mainMaskDummy[slicee,:,:]) ,  ("mainForModificationsTexture2",mainForModificationsTexture2Dat[slicee,:,:]) 
      ,("mainForModificationsTexture1",mainForModificationsTexture1Dat[slicee,:,:]) 
    ,("mainCTImage",ctDummy[slicee,:,:] )]


    # listOfDataAndImageNamesSlice = [ ("grandTruthLiverLabel",a) ,      ("mainForModificationsTexture2",UInt16.(a)) 
    #   ,("mainForModificationsTexture1",a)   ,("mainCTImage",a)]




    window = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.window
    
    Main.SegmentationDisplay.updateSingleImagesDisplayed(listOfDataAndImageNamesSlice,3 )

    textSpec = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[3]
    
    Main.SegmentationDisplay.mainActor.actor.textureToModifyVec= [textSpec]

    using Main.Uniforms, Main.OpenGLDisplayUtils

    glClearColor(0.0, 0.0, 0.1 , 1.0)
    glActiveTexture(GL_TEXTURE0 +2); # active proper texture unit before binding
    glBindTexture(GL_TEXTURE_2D, textSpec.ID[]); 
    setMaskColor(RGB(1.0,0.0,0.0) ,textSpec.uniforms)
    setTextureVisibility(true ,textSpec.uniforms)
  
    basicRender(window)

    setCTWindow(Int32(-1), Int32(-1),Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[4].uniforms)




# using Main.CustomFragShad
#     strr= Main.CustomFragShad.createCustomFramgentShader(listOfTexturesToCreate)
#     for st in split(strr, "\n")
#     @info st
#     end
#     using Main.ShadersAndVerticies


 #  Main.ShadersAndVerticies.createFragmentShader("", listOfTexturesToCreate)

  #   textSpec

  #   @uniforms  (min_shown_white, max_shown_black, displayRange,
  #   iTexture0, uTexture0, fTexture0, typeOfMainSampler, isVisibleTexture0
  #   ,nuclearMask,isVisibleNuclearMask,
  #   uImask0,          uImask1,
  #         uImask2,          uImask3,          uImask4,          uImask5,          uImask6,          uImask7,          uImask8,
  #         imask0,          imask1,          imask2,          imask3,          imask4,          imask5,          imask6,          imask7,          imask8,          fmask0,
  #         fmask1,          fmask2,          fmask3,          fmask4,          fmask5,          fmask6,          fmask7,          fmask8,         uIcolorMask0,         uIcolorMask1,
  #        uIcolorMask2,         uIcolorMask3,         uIcolorMask4,         uIcolorMask5,         uIcolorMask6,         uIcolorMask7,         uIcolorMask8,         icolorMask0,
  #        icolorMask1,         icolorMask2,         icolorMask3,         icolorMask4,         icolorMask5,         icolorMask6,         icolorMask7,         icolorMask8,
  #        fcolorMask0,         fcolorMask1,         fcolorMask2,         fcolorMask3,         fcolorMask4,         fcolorMask5,         fcolorMask6,         fcolorMask7,
  #        fcolorMask8,        isVisibleTexture0,        isVisibleNuclearMask,                  uIisVisk0,       uIisVisk1,       uIisVisk2,       uIisVisk3,       uIisVisk4,
  #      uIisVisk5,       uIisVisk6,       uIisVisk7,       uIisVisk8,       iisVisk0,       iisVisk1,       iisVisk2,       iisVisk3,       iisVisk4,       iisVisk5,       iisVisk6,
  #      iisVisk7,       iisVisk8,       fisVisk0,       fisVisk1,       fisVisk2,       fisVisk3,       fisVisk4,       fisVisk5,       fisVisk6,       fisVisk7,       fisVisk8) = program
    


  #      glGetUniformLocation(program, "uImask0")
  #      uIisVisk0
  #   ##
  #   @uniforms! begin
  #   uIisVisk0:=false
  #   uIisVisk1:=false
  #   uIisVisk2:=false
  #   uIisVisk3:=false
  #   uIisVisk4:=false
  #   uIisVisk5:=false
  #   uIisVisk6:=false
  #   uIisVisk7:=false
  #   uIisVisk8:=false
  # end



    ##
  #  glUniform1i(textSpec.uniforms.isVisibleRef,false);

   # Main.SegmentationDisplay.mainActor.actor.textureToModifyVec= [listOfTexturesToCreate[1]]
  
  
  


   GLFW.PollEvents()
