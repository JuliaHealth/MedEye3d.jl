

     using DrWatson
     @quickactivate "Probabilistic medical segmentation"

     using GLFW

     include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/structs/forDisplayStructs.jl")
     include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/loadData/manageH5File.jl")
     using Main.h5manag

     include(DrWatson.scriptsdir("display","GLFW","startModules","PrepareWindowHelpers.jl"))
     include(DrWatson.scriptsdir("display","GLFW","modernGL","OpenGLDisplayUtils.jl"))
     include(DrWatson.scriptsdir("display","GLFW","startModules","ShadersAndVerticies.jl"))
     include(DrWatson.scriptsdir("display","GLFW","modernGL","TextureManag.jl") )
     include(DrWatson.scriptsdir("display","GLFW","startModules","PrepareWindow.jl"))


     include(DrWatson.scriptsdir("display","reactingToMouseKeyboard","ReactToScroll.jl") )
     include(DrWatson.scriptsdir("display","reactingToMouseKeyboard","ReactingToInput.jl") )
     include(DrWatson.scriptsdir("display","reactingToMouseKeyboard","ReactOnMouseClickAndDrag.jl") )



#data source
exampleDat = Int16.(Main.h5manag.getExample())
exampleLabels = UInt8.(Main.h5manag.getExampleLabels())
dims = size(exampleDat)
widthh=dims[2]
heightt=dims[3]
slicesNumb= dims[1]

using Revise 
segmPath = DrWatson.scriptsdir("display","GLFW","SegmentationDisplay.jl")
include(segmPath)
using Main.SegmentationDisplay
#data about textures we want to create
using  Main.ForDisplayStructs
using Parameters

# list of texture specifications, important is that main texture - main image should be specified first
#Order is important !
listOfTexturesToCreate = [
Main.ForDisplayStructs.TextureSpec("grandTruthLiverLabel",
                 widthh,
                heightt,
                slicesNumb,
                GL_R8UI,
                GL_UNSIGNED_BYTE,
                "msk0" 
                ,0),
                Main.ForDisplayStructs.TextureSpec("mainCTImage",
                widthh,
                heightt,
                slicesNumb,
                GL_R16I,
                GL_SHORT,
                "Texture0" 
                ,0) 
                     
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
  
    
    
#############configuring
    Main.SegmentationDisplay.coordinateDisplay(listOfTexturesToCreate,1000,800)

    Main.SegmentationDisplay.passDataForScrolling(listOfDataAndImageNames)

    Main.SegmentationDisplay.updateSingleImagesDisplayed(listOfDataAndImageNamesSlice,200 )


    window = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.window
    stopListening = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.stopListening
    
    textSpec = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[1]
    textSpecB = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[2]

    
    imagedims=dims
    imageWidth = dims[2]
    imageHeight = dims[3]

    windowDims =     GLFW.GetWindowSize(window)
    windowWidth = windowDims[1]
    windowHeight = windowDims[2]
    
    quadmaxX = Int32(floor(windowWidth*0.8))
    quadMaxY = windowHeight # but we need to remember that maximum values are in bottom right corner and beginning is upper left corner
  

    using Main.OpenGLDisplayUtils

    #working
    currX = 443
    currY = 586
    updateTexture(rand(10,10), textSpecB,
    Int64(floor( ((currX)/(windowWidth*0.9))*imageWidth)  )
    ,
    Int64(floor(  ((windowHeight-currY)/windowHeight)*imageHeight)  )
    ,5,5 )
    basicRender(window)




    360*2

720/800

    800 = 512

    (10/800)*






    Main.SegmentationDisplay.cleanUp()

    x= sync( ActorWithOpenGlObjects())  SyncActor{Any, ActorWithOpenGlObjects}
x.actor.currentDisplayedSlice =0 



#############



# struct StoreActor{D} <: Rocket.Actor{D}
#     values :: Vector{D}

#     StoreActor{D}() where D = new(Vector{D}())
# end


# Rocket.on_next!(::StoreActor, data::Int)     = println("Int: $data")
# Rocket.on_next!(::StoreActor, data::Float64) = println("Float64: $data")
# Rocket.on_next!(::StoreActor, data)          = println("Something else: $data")



segmPath = DrWatson.scriptsdir("display","GLFW","SegmentationDisplay.jl")
include(segmPath)

using Main.ForDisplayStructs
using Main.ReactToScroll
using Main.SegmentationDisplay
using Rocket





const scrollback = Main.ReactToScroll.ScrollCallbackSubscribable()

forDisplayConstants = Main.SegmentationDisplay.coordinateDisplay(listOfTexturesToCreate)

Main.ReactToScroll.registerMouseScrollFunctions(forDisplayConstants.window,forDisplayConstants.stopListening)

GLFW.SetScrollCallback(forDisplayConstants.window, (a, xoff, yoff) -> scrollback(a, xoff, yoff))

# Than later in your application you can do smth like

subscription = subscribe!(scrollback, (direction) -> println(direction))








#wrapping the Open Gl and GLFW objects into an observable
forDisplayConstantObesrvable = of(Main.SegmentationDisplay.coordinateDisplay(listOfTexturesToCreate))

keep_actor = ActorWithOpenGlObjects()
subscribe!(forDisplayConstantObesrvable, keep_actor) # configuring



source = from([false,false,false,false,false,false])
subscribe!(source, keep_actor) # imitasting scroll

keep_actor.mainForDisplayObjects

# Logs
# Completed!

println(keep_actor.currentDisplayedSlice)

