using DrWatson
@quickactivate "Probabilistic medical segmentation"

using GLFW: Window

include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/structs/forDisplayStructs.jl")
include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/loadData/manageH5File.jl")
using Main.h5manag

include(DrWatson.scriptsdir("display","GLFW","startModules","PrepareWindowHelpers.jl"))
include(DrWatson.scriptsdir("display","GLFW","modernGL","OpenGLDisplayUtils.jl"))
include(DrWatson.scriptsdir("display","GLFW","startModules","ShadersAndVerticies.jl"))
include(DrWatson.scriptsdir("display","GLFW","modernGL","TextureManag.jl") )
pathPrepareWindow = DrWatson.scriptsdir("display","GLFW","startModules","PrepareWindow.jl")
include(pathPrepareWindow)


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

    
    forDisplayConstants = Main.SegmentationDisplay.coordinateDisplay(listOfTexturesToCreate)

    slice = 215
    listOfDataAndImageNames = [("grandTruthLiverLabel",exampleLabels[slice,:,:]),("mainCTImage",exampleDat[slice,:,:] )]
    Main.SegmentationDisplay.updateImagesDisplayed(listOfDataAndImageNames
         ,forDisplayConstants )


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