
using DrWatson
@quickactivate "Probabilistic medical segmentation"


SegmentationDisplayStr = """
Main module controlling displaying segmentations image and data

Description of nthe file structure
ModernGlUtil.jl 
    - copied utility functions from ModernGl.jl  
    - depends on external : ModernGL and  GLFW
PreperWindowHelpers.jl 
    - futher abstractions used in PreperWindow
    - depends on ModernGlUtil.jl 
shadersAndVerticies.jl 
    -store constant values of shader code and constant needed to render shapes
    -depends on external: ModernGL, GeometryTypes, GLFW
    -needs to be invoked only after initializeWindow() is invoked from PreperWindowHelpers module - becouse GLFW context mus be ready
PrepareWindow.jl 
    - collects functions and data and creates configured window with shapes needed to display textures and configures listening to mouse and keybourd inputs
    - depends on internal:ModernGlUtil,PreperWindowHelpers,shadersAndVerticies,OpenGLDisplayUtils
    - depends on external:ModernGL, GeometryTypes, GLFW
TextureManag.jl 
    - as image + masks are connacted by shaders into single texture this module  by controlling textures controlls image display
    - depends on external: ModernGL
    - depends on internal :OpenGLDisplayUtils
ReactToScroll.jl
    - functions needed to react to scrolling
    - depends on external: Rocket, GLFW
    - depends on internal : ForDisplayStructs.jl
ReactingToInput.jl - using Rocket.jl (reactivate functional programming ) enables reacting to user input
    - depends on external: Rocket, GLFW
    - depends on internal : ReactToScroll.jl, ForDisplayStructs.jl
OpenGLDisplayUtils.jl - some utility functions used in diffrent parts of program
    - depends on external :GLFW, ModernGL 
"""
# @doc SegmentationDisplayStr
module SegmentationDisplay

using DrWatson
@quickactivate "Probabilistic medical segmentation"
export coordinateDisplay
export passDataForScrolling

using ModernGL
using GLFW
using Main.PrepareWindow
using Main.TextureManag
using Main.OpenGLDisplayUtils
using Main.ForDisplayStructs
using Main.ReactingToInput
using Rocket

#holds actor that is main structure that process inputs from GLFW and reacts to it
mainActor = sync(ActorWithOpenGlObjects())



coordinateDisplayStr = """
coordinating displaying - sets needed constants that are storeds in  forDisplayConstants; and configures interactions from GLFW events
listOfTextSpecs - holds required data needed to initialize textures
"""
@doc coordinateDisplayStr
function coordinateDisplay(listOfTextSpecs::Vector{TextureSpec})

window,vertex_shader,fragment_shader ,shader_program,stopListening = Main.PrepareWindow.displayAll()
forDispObj =  forDisplayObjects(
    initializeTextures(shader_program,listOfTextSpecs)
    ,window,vertex_shader,fragment_shader ,shader_program,stopListening, Threads.Atomic{Bool}(0)
)

#wrapping the Open Gl and GLFW objects into an observable and passing it to the actor
forDisplayConstantObesrvable = of(forDispObj)
subscribe!(forDisplayConstantObesrvable, mainActor) # configuring

registerInteractions()#passing needed subscriptions from GLFW

end #coordinateDisplay


passDataForScrollingStr =    """
is used to pass into the actor data that will be used for scrolling
onScrollData - list of tuples where first is the name of the texture that we provided and second is associated data (3 dimensional array of appropriate type)
"""
@doc passDataForScrollingStr
function passDataForScrolling(onScrollData::Vector{Tuple{String, Array{T, 3} where T}})
#wrapping the data into an observable and passing it to the actor
forScrollData = of(onScrollData)
subscribe!(forScrollData, mainActor) 
end


updateSingleImagesDisplayedStr =    """
enables updating just a single slice that is displayed - do not change what will happen after scrolling
one need to pass data to actor in vector of tuples whee first entry in tuple is name of texture given in the setup and second is 2 dimensional aray of appropriate type with image data

"""
@doc updateSingleImagesDisplayedStr
function updateSingleImagesDisplayed(listOfDataAndImageNames::Vector{Tuple{String, Array{T, 2} where T}})
    forDispData = of(listOfDataAndImageNames)
    subscribe!(forDispData, mainActor) 

end #updateSingleImagesDisplayed



registerInteractionsStr =    """
is using the actor that is instantiated in this module and connects it to GLFW context
by invoking appropriate registering functions and passing to it to the main Actor controlling input
"""
@doc registerInteractionsStr
function registerInteractions()
    subscribeGLFWtoActor(mainActor)
end


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
