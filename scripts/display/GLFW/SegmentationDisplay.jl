
using DrWatson
@quickactivate "Probabilistic medical segmentation"


# SegmentationDisplayStr = """
# Main module controlling displaying segmentations image and data

# Description of nthe file structure
# ModernGlUtil.jl 
#     - copied utility functions from ModernGl.jl  
#     - depends on external : ModernGL and  GLFW
# PreperWindowHelpers.jl 
#     - futher abstractions used in PreperWindow
#     - depends on ModernGlUtil.jl 
# shadersAndVerticies.jl 
#     -store constant values of shader code and constant needed to render shapes
#     -depends on external: ModernGL, GeometryTypes, GLFW
#     -needs to be invoked only after initializeWindow() is invoked from PreperWindowHelpers module - becouse GLFW context mus be ready
# PrepareWindow.jl 
#     - collects functions and data and creates configured window with shapes needed to display textures and configures listening to mouse and keybourd inputs
#     - depends on internal:ModernGlUtil,PreperWindowHelpers,shadersAndVerticies,OpenGLDisplayUtils
#     - depends on external:ModernGL, GeometryTypes, GLFW
# TextureManag.jl 
#     - as image + masks are connacted by shaders into single texture this module  by controlling textures controlls image display
#     - depends on external: ModernGL
#     -depends on internal :OpenGLDisplayUtils
# ReactingToInput.jl - using Rocket.jl (reactivate functional programming ) enables reacting to user input
#     -depends on external: Rocket, GLFW
# OpenGLDisplayUtils.jl - some utility functions used in diffrent parts of program
#     - depends on external :GLFW, ModernGL 
# """
# @doc SegmentationDisplayStr
module SegmentationDisplay

using DrWatson
@quickactivate "Probabilistic medical segmentation"
export coordinateDisplay
using ModernGL
using GLFW
using Main.PrepareWindow
using Main.TextureManag
using Main.OpenGLDisplayUtils
using Main.ForDisplayStructs

#holds the list of texture specifications with ids
modulelistOfTextSpecs::Vector{TextureSpec} =[]

window,vertex_shader,fragment_shader ,shader_program,stopListening = Main.PrepareWindow.displayAll()

```@doc
coordinating displaying , 
listOfTextSpecs - holds required data needed to initialize textures
```
function coordinateDisplay(listOfTextSpecs::Vector{TextureSpec})
#We store here needed variables from window and shaders initializations
#stopListening - atomic boolean that marks to event listening loop to halt in order to free GLFW context

#clear color buffer

modulelistOfTextSpecs= initializeTextures(shader_program,listOfTextSpecs)
end #coordinateDisplay
```@doc
coordinating updating all of the images, masks... 
listOfTextSpecs - holds required data needed to initialize textures - 
tuples where first entry is name of image that we given in configuration; 
and second entry is data that we want to pass

```
function updateImagesDisplayed(listOfDataAndImageNames:: Vector{Tuple{String, Array}})
    #clearing color buffer
    glClearColor(0.0, 0.0, 0.1 , 1.0)
    for updateDat in listOfDataAndImageNames
        find( (texSpec)-> , modulelistOfTextSpecs)
        updateTexture()
    end #for 
    #render onto the screen
    Main.OpenGLDisplayUtils.basicRender(window)
end

[("", rand())]

#pboId, DATA_SIZE = preparePixelBuffer(Int16,widthh,heightt,0)

# ##################
#clear color buffer
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



############clean up

#remember to unsubscribe; remove textures; clear buffers and close window


end #SegmentationDisplay
