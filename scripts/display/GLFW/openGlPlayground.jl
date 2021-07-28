using Rocket: isempty
##

     using DrWatson
     @quickactivate "Probabilistic medical segmentation"
     
     using Setfield
     using GLFW
     using ColorTypes

     include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/structs/forDisplayStructs.jl")
     include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/loadData/manageH5File.jl")
     using Main.h5manag

     include(DrWatson.scriptsdir("display","GLFW","startModules","PrepareWindowHelpers.jl"))
     include(DrWatson.scriptsdir("display","GLFW","modernGL","OpenGLDisplayUtils.jl"))
     include(DrWatson.scriptsdir("display","GLFW","startModules","ShadersAndVerticies.jl"))
     include(DrWatson.scriptsdir("display","GLFW","modernGL","TextureManag.jl") )
     include(DrWatson.scriptsdir("display","GLFW","startModules","PrepareWindow.jl"))


     include(DrWatson.scriptsdir("display","reactingToMouseKeyboard","ReactToScroll.jl") )
     include(DrWatson.scriptsdir("display","reactingToMouseKeyboard","ReactOnMouseClickAndDrag.jl") )

     include(DrWatson.scriptsdir("display","reactingToMouseKeyboard","ReactingToInput.jl") )
     include(DrWatson.scriptsdir("display","GLFW","SegmentationDisplay.jl"))


#data source
exampleDat = Int16.(Main.h5manag.getExample())
exampleLabels = UInt8.(Main.h5manag.getExampleLabels())
dims = size(exampleDat)
widthh=dims[2]
heightt=dims[3]
slicesNumb= dims[1]

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
    colors = [RGB(1.0,0.0,0.0)],
    GL_Rtype=  GL_R8UI,
    OpGlType = GL_UNSIGNED_BYTE,
    samplName = "msk0" ),
Main.ForDisplayStructs.TextureSpec(
    name = "mainForModificationsTexture",
    colors = [RGB(0.0,1.0,0.0)],
    GL_Rtype=  GL_R8UI,
    OpGlType = GL_UNSIGNED_BYTE,
    samplName = "mask1" ),    
Main.ForDisplayStructs.TextureSpec(
    name= "mainCTImage",
    GL_Rtype =  GL_R16I ,
    OpGlType =  GL_SHORT,
    samplName = "Texture0")  
      ]
   
   
   
    include(DrWatson.scriptsdir("display","reactingToMouseKeyboard","ReactingToInput.jl"))
    using Main.ReactingToInput

    segmPath = DrWatson.scriptsdir("display","GLFW","SegmentationDisplay.jl")
    include(segmPath)
    

    using Main.ForDisplayStructs
    using Main.ReactToScroll
    using Main.SegmentationDisplay
    using Rocket
    
    


 slice = 200
  listOfDataAndImageNamesSlice = [("grandTruthLiverLabel",exampleLabels[slice,:,:]),("mainCTImage",exampleDat[slice,:,:] )]


    listOfDataAndImageNames = [("grandTruthLiverLabel",exampleLabels),("mainCTImage",exampleDat)]
  
    
    imagedims=dims
    imageWidth = dims[2]
    imageHeight = dims[3]
#configuring
    Main.SegmentationDisplay.coordinateDisplay(listOfTexturesToCreate,512,512, 1000,800)

 
    Main.SegmentationDisplay.passDataForScrolling(listOfDataAndImageNames)

    Main.SegmentationDisplay.updateSingleImagesDisplayed(listOfDataAndImageNamesSlice,200 )


    window = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.window
    stopListening = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.stopListening
    
    textSpec = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[1]
    textSpecB = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[2]

    Main.SegmentationDisplay.mainActor.actor.onScrollData
    push!(Main.SegmentationDisplay.mainActor.actor.textureToModifyVec, textSpec)
    GLFW.PollEvents()


#     stopListening[]=false

#     tt= listOfTexturesToCreate[1]
#     res = Vector{TextureSpec}()

#     push!(res,setproperties(tt, (ID=Ref(UInt32(0)))) )


#     isempty([])










#     textSpec.ID
    
 

#     windowDims =     GLFW.GetWindowSize(window)
#     windowWidth = windowDims[1]
#     windowHeight = windowDims[2]

#     quadmaxX = Int32(floor(windowWidth*0.8))
#     quadMaxY = windowHeight # but we need to remember that maximum values are in bottom right corner and beginning is upper left corner
  

#     using Main.OpenGLDisplayUtils

#     #working
#     currX = 443
#     currY = 586
#     updateTexture(rand(10,10), textSpecB,
#     Int64(floor( ((currX)/(windowWidth*0.9))*imageWidth)  )
#     ,
#     Int64(floor(  ((windowHeight-currY)/windowHeight)*imageHeight)  )
#     ,5,5 )
#     basicRender(window)


#     using Parameters




#     mutable struct ParaB
#         a::Float64
#         b::Int
#         c::Int
#         d::Int
#     end
    
#     function f!(var, pa::Para)
#         @unpack a, b = pa # equivalent to: a,b = pa.a,pa.b
#         out = var + a + b
#         b = 77
#         @pack! pa = b # equivalent to: pa.b = b
#         return out, pa
#     end
    
#     out, pa = f!(7, Para(1,2)) # -> 10.0, Para(1.0, 77)



#     using Setfield
#     pp = ParaB(1.0,2,3,4)

# setproperties(pp, (a=9.0, b=99))
# pp

