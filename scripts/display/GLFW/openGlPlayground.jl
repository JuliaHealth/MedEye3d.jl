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
                GL_R8UI,
                GL_UNSIGNED_BYTE,
                "msk0" 
                ,0),
                Main.ForDisplayStructs.TextureSpec("mainCTImage",
                widthh,
                heightt,
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





# ##################
# #clear color buffer
# glClearColor(0.0, 0.0, 0.1 , 1.0)
# #true labels
# glActiveTexture(GL_TEXTURE0 + 1); # active proper texture unit before binding
# glUniform1i(glGetUniformLocation(shader_program, "msk0"), 1);# we first look for uniform sampler in shader - here 
# trueLabels= createTexture(1,exampleLabels[210,:,:],widthh,heightt,GL_R8UI,GL_UNSIGNED_BYTE)#binding texture and populating with data
# #main image
# glActiveTexture(GL_TEXTURE0); # active proper texture unit before binding
# glUniform1i(glGetUniformLocation(shader_program, "Texture0"), 0);# we first look for uniform sampler in shader - here 
# mainTexture= createTexture(0,exampleDat[210,:,:],widthh,heightt,GL_R16I,GL_SHORT)#binding texture and populating with data
# #render
# basicRender()


# #############
# #order of texture uploads  is important and texture 0 should be last binded as far as I get it 
# stopListening[]=true
# glClearColor(0.0, 0.0, 0.1 , 1.0)
# #update labels
# updateTexture(Int16,widthh,heightt,exampleLabels[200,:,:], trueLabels,stopListening,pboId, DATA_SIZE,GL_UNSIGNED_BYTE)
# #update main image
# updateTexture(Int16,widthh,heightt,exampleDat[200,:,:], mainTexture,stopListening,pboId, DATA_SIZE, GL_SHORT)
# basicRender()
# stopListening[]= false


