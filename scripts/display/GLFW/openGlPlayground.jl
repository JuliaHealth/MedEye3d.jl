

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
    
    


 slice = 215
  listOfDataAndImageNamesSlice = [("grandTruthLiverLabel",exampleLabels[slice,:,:]),("mainCTImage",exampleDat[slice,:,:] )]


    listOfDataAndImageNames = [("grandTruthLiverLabel",exampleLabels),("mainCTImage",exampleDat)]
  
    
    
#############configuring
    Main.SegmentationDisplay.coordinateDisplay(listOfTexturesToCreate)


    Main.SegmentationDisplay.mainActor

    Main.SegmentationDisplay.passDataForScrolling(listOfDataAndImageNames)


    Main.SegmentationDisplay.updateSingleImagesDisplayed(listOfDataAndImageNamesSlice )


    includet(DrWatson.scriptsdir("display","reactingToMouseKeyboard","ReactingToInput.jl"))
    using Main.ReactingToInput
    Main.ReactingToInput.registerMouseScrollFunctions(forDisplayConstants.window)



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

