"""
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
ReactOnMouseClickAndDrag.jl   
    - functions needed to enable mouse interactions 
    - internal depentdencies: ForDisplayStructs, TextureManag,OpenGLDisplayUtils
    -external dependencies: Rocket, GLFW
ReactingToInput.jl - using Rocket.jl (reactivate functional programming ) enables reacting to user input
    - depends on external: Rocket, GLFW
    - depends on internal : ReactToScroll.jl, ForDisplayStructs.jl 
OpenGLDisplayUtils.jl - some utility functions used in diffrent parts of program
    - depends on external :GLFW, ModernGL 
"""
module SegmentationDisplay

export coordinateDisplay
export passDataForScrolling

using ModernGL, GLFW,  PrepareWindow,  TextureManag, OpenGLDisplayUtils,  ForDisplayStructs, Uniforms,  DisplayWords, Dictionaries
using  ReactingToInput, Rocket, Setfield, Logging,  ShadersAndVerticiesForText,FreeTypeAbstraction, DisplayWords,  DataStructs,  StructsManag

using DrWatson
@quickactivate "NuclearMedEye"

#holds actor that is main structure that process inputs from GLFW and reacts to it
mainActor = sync(ActorWithOpenGlObjects())
#collecting all subsciptions  to be able to clean all later
subscriptions = []


"""
coordinating displaying - sets needed constants that are storeds in  forDisplayConstants; and configures interactions from GLFW events
listOfTextSpecs - holds required data needed to initialize textures
keeps also references to needed uniforms etc.
windowWidth::Int,windowHeight::Int - GLFW window dimensions
fractionOfMainIm - how much of width should be taken by the main image
heightToWithRatio - needed for proper display of main texture - so it would not be stretched ...
"""
function coordinateDisplay(listOfTextSpecsPrim::Vector{TextureSpec}
                        ,fractionOfMainIm::Float32
                        ,dataToScrollDims::DataToScrollDims=DataToScrollDims()#stores additional data about full dimensions of scrollable dat - this is necessery for switching slicing plane orientation efficiently
                        ,windowWidth::Int=1200
                        ,windowHeight::Int= Int(round(windowWidth*fractionOfMainIm))
                        ,textTexturewidthh::Int32=Int32(2000)
                        ,textTextureheightt::Int32= Int32( round((windowHeight/(windowWidth*(1-fractionOfMainIm)) ))*textTexturewidthh)
                        ,windowControlStruct::WindowControlStruct=WindowControlStruct()) 
   #setting number to texture that will be needed in shader configuration
   listOfTextSpecs= map(x->setproperties(x[2],(whichCreated=x[1])),enumerate(listOfTextSpecsPrim))
    #calculations of necessary constants needed to calculate window size , mouse position ...
   calcDimStruct= CalcDimsStruct(windowWidth=windowWidth 
                  ,windowHeight=windowHeight 
                  ,fractionOfMainIm=fractionOfMainIm
                  ,wordsImageQuadVert= ShadersAndVerticiesForText.getWordsVerticies(fractionOfMainIm)  
                  ,wordsQuadVertSize= sizeof( ShadersAndVerticiesForText.getWordsVerticies(fractionOfMainIm))
                  ,textTexturewidthh=textTexturewidthh
                  ,textTextureheightt=textTextureheightt ) |>
   (calcDim)-> getHeightToWidthRatio(calcDim,dataToScrollDims )|>
   (calcDim)-> getMainVerticies(calcDim)

   subscribe!(of(calcDimStruct),mainActor )

 #creating window and event listening loop
    window,vertex_shader,fragment_shader ,shader_program,stopListening,vbo,ebo,fragment_shader_words,vbo_words,shader_program_words,gslsStr =  PrepareWindow.displayAll(listOfTextSpecs,calcDimStruct )

    # than we set those uniforms, open gl types and using data from arguments  to fill texture specifications
    mainImageUnifs,listOfTextSpecsMapped= assignUniformsAndTypesToMasks(listOfTextSpecs,shader_program,windowControlStruct) 

    @info "listOfTextSpecsMapped" listOfTextSpecsMapped
    #initializing object that holds data reqired for interacting with opengl 
    initializedTextures =  initializeTextures(listOfTextSpecsMapped,calcDimStruct)
   
    numbDict = filter(x-> x.numb>=0,initializedTextures) |>
    (filtered)-> Dictionary(map(it->it.numb,filtered),collect(eachindex(filtered))) # a way for fast query using assigned numbers

    forDispObj =  forDisplayObjects(
            listOfTextSpecifications=initializedTextures
            ,window= window
            ,vertex_shader= vertex_shader
            ,fragment_shader= fragment_shader
            ,shader_program= shader_program
            ,stopListening= stopListening
            ,vbo= vbo[]
            ,ebo= ebo[]
            ,mainImageUniforms= mainImageUnifs
            ,TextureIndexes= Dictionary(map(it->it.name,initializedTextures),collect(eachindex(initializedTextures)))
            ,numIndexes= numbDict 
            ,gslsStr=gslsStr
            ,windowControlStruct=windowControlStruct
   )




    #in order to clean up all resources while closing
    GLFW.SetWindowCloseCallback(window, (_) -> cleanUp())

    #wrapping the Open Gl and GLFW objects into an observable and passing it to the actor
    forDisplayConstantObesrvable = of(forDispObj)
    subscribe!(forDisplayConstantObesrvable, mainActor) # configuring
    #passing for text display object 
    forTextDispStruct = prepareForDispStruct(length(initializedTextures)
                                            ,fragment_shader_words
                                            ,vbo_words
                                            ,shader_program_words
                                            ,window
                                            ,textTexturewidthh
                                            ,textTextureheightt
                                            ,forDispObj)

    subscribe!(of(forTextDispStruct),mainActor )
                                    
    registerInteractions()#passing needed subscriptions from GLFW


end #coordinateDisplay

"""
is used to pass into the actor data that will be used for scrolling
onScrollData - struct holding between others list of tuples where first is the name of the texture that we provided and second is associated data (3 dimensional array of appropriate type)
"""
function passDataForScrolling(onScrollData::FullScrollableDat)
    #wrapping the data into an observable and passing it to the actor
    forScrollData = of(onScrollData)
    subscribe!(forScrollData, mainActor) 
end


"""
enables updating just a single slice that is displayed - do not change what will happen after scrolling
one need to pass data to actor in 
listOfDataAndImageNames - struct holding  tuples where first entry in tuple is name of texture given in the setup and second is 2 dimensional aray of appropriate type with image data
sliceNumber - the number to which we set slice in order to later start scrolling the scroll data from this point
"""
function updateSingleImagesDisplayed( listOfDataAndImageNames::SingleSliceDat)
    forDispData = of(listOfDataAndImageNames)
    subscribe!(forDispData, mainActor) 

end #updateSingleImagesDisplayed


 """
is using the actor that is instantiated in this module and connects it to GLFW context
by invoking appropriate registering functions and passing to it to the main Actor controlling input
"""
function registerInteractions()
    subscriptionsInner = subscribeGLFWtoActor(mainActor)
    for el in subscriptionsInner
        push!(subscriptions,el)
    end #for


end

"""
Preparing ForWordsDispStruct that will be needed for proper displaying of texts
    numberOfActiveTextUnits - number of textures already used - so we we will know what is still free 
    fragment_shader_words - reference to fragment shader used to display text
    vbo_words - vertex buffer object used to display words
    shader_program_words - shader program associated with displaying text
    widthh, heightt - size of the texture - the bigger the higher resolution, but higher computation cost

return prepared for displayStruct    
"""
function prepareForDispStruct(numberOfActiveTextUnits::Int
                            ,fragment_shader_words::UInt32
                            ,vbo_words::Base.RefValue{UInt32}
                            ,shader_program_words::UInt32
                            ,window
                            ,widthh::Int32 =Int32(1)
                            ,heightt::Int32=Int32(1)
                            ,forDispObj::forDisplayObjects=forDisplayObjects()
                            ) ::ForWordsDispStruct

      res =  ForWordsDispStruct(
            fontFace = FreeTypeAbstraction.findfont("hack";  additional_fonts= datadir("fonts"))
            ,textureSpec = createTextureForWords(numberOfActiveTextUnits
                                                 ,widthh
                                                 ,heightt 
                                                 ,getProperGL_TEXTURE(numberOfActiveTextUnits+1) )
            ,fragment_shader_words= fragment_shader_words
            ,vbo_words=vbo_words
            ,shader_program_words=shader_program_words
         )

    return res
end#prepereForDispStruct


"""
In order to properly close displayer we need to :
 remove buffers that wer use 
 remove shaders 
 remove all textures
 unsubscibe all of the subscriptions to the mainActor
 finalize main actor and reinstantiate it
 close GLFW window
"""
function cleanUp()
    obj = mainActor.actor.mainForDisplayObjects
    glFlush()
    GLFW.DestroyWindow(obj.window)

    # glClearColor(0.0, 0.0, 0.1 , 1.0) # for a good begining
    # #first we unsubscribe and give couple seconds for processes to stop
    # for sub in subscriptions
    #     unsubscribe!(sub)
    # end # for
    # sleep(5)
    # obj = mainActor.actor.mainForDisplayObjects
    # #deleting textures
    # glDeleteTextures(length(obj.listOfTextSpecifications), map(text->text.ID,obj.listOfTextSpecifications));
    # #destroying buffers
    # glDeleteBuffers(2,[obj.vbo,obj.ebo])
    # #detaching shaders
    # glDeleteShader(obj.fragment_shader);
    # glDeleteShader(obj.vertex_shader);
    # #destroying program
    # glDeleteProgram(obj.shader_program)
    # #finalizing and recreating main actor
end #cleanUp    



end #SegmentationDisplay
