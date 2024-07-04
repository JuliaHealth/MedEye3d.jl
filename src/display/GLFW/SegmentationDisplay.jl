"""
Main module controlling displaying segmentations image and data

"""
module SegmentationDisplay
export coordinateDisplay
export passDataForScrolling

using ModernGL, GLFW, ..PrepareWindow, ..TextureManag, ..OpenGLDisplayUtils, ..ForDisplayStructs, ..Uniforms, ..DisplayWords, Dictionaries
using ..ReactingToInput, ..ReactToScroll, Setfield, Logging, ..ShadersAndVerticiesForText, FreeTypeAbstraction, ..DisplayWords, ..DataStructs, ..StructsManag
using ..ReactOnKeyboard, ..ReactOnMouseClickAndDrag

#  do not copy it into the consumer function
"""
configuring consumer function on_next! function using multiple dispatch mechanism in order to connect input to proper functions
"""
on_next!(stateObject::StateDataFields, data::Int64) = reactToScroll(data, stateObject)
on_next!(stateObject::StateDataFields, data::forDisplayObjects) = setUpMainDisplay(data, stateObject)
on_next!(stateObject::StateDataFields, data::ForWordsDispStruct) = setUpWordsDisplay(data, stateObject)
on_next!(stateObject::StateDataFields, data::CalcDimsStruct) = setUpCalcDimsStruct(data, stateObject)
on_next!(stateObject::StateDataFields, data::valueForMasToSetStruct) = setUpvalueForMasToSet(data, stateObject)
on_next!(stateObject::StateDataFields, data::FullScrollableDat) = setUpForScrollData(data, stateObject)
on_next!(stateObject::StateDataFields, data::SingleSliceDat) = updateSingleImagesDisplayedSetUp(data, stateObject)
on_next!(stateObject::StateDataFields, data::Vector{MouseStruct}) = react_to_draw(data, stateObject)
on_next!(stateObject::StateDataFields, data::MouseStruct) = reactToMouseDrag(data, stateObject) #needs modification , with the react_to_draw, data of vectorStruct (MoustStruct)
on_next!(stateObject::StateDataFields, data::KeyInputFields) = reactToKeyInput(data, stateObject)
on_error!(stateObject::StateDataFields, err) = error(err)
on_complete!(stateObject::StateDataFields) = ""


"""
is used to pass into the actor data that will be used for scrolling
onScrollData - struct holding between others list of tuples where first is the name of the texture that we provided and second is associated data (3 dimensional array of appropriate type)
"""

function passDataForScrolling(mainMedEye3dInstance::MainMedEye3d, onScrollData::FullScrollableDat)
    """
    put data onto the channel, matching types with on_next.
    """
    #modify here the data for passing onto the channel
    put!(mainMedEye3dInstance.channel, onScrollData)
end



"""
is using the actor that is instantiated in this module and connects it to GLFW context
by invoking appropriate registering functions and passing to it to the main Actor controlling input
"""
function registerInteractions(window::GLFW.Window, mainMedEye3dInstance::MainMedEye3d, calcDimStruct::CalcDimsStruct)
    subscribeGLFWtoActor(window, mainMedEye3dInstance, calcDimStruct)
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
function prepareForDispStruct(numberOfActiveTextUnits::Int, fragment_shader_words::UInt32, vbo_words::Base.RefValue{UInt32}, shader_program_words::UInt32, window, widthh::Int32=Int32(1), heightt::Int32=Int32(1), forDispObj::forDisplayObjects=forDisplayObjects()
)::ForWordsDispStruct

    res = ForWordsDispStruct(
        fontFace=FTFont(joinpath(dirname(dirname(pathof(FreeTypeAbstraction))), "test", "hack_regular.ttf")), textureSpec=createTextureForWords(numberOfActiveTextUnits, widthh, heightt, getProperGL_TEXTURE(numberOfActiveTextUnits + 1)), fragment_shader_words=fragment_shader_words, vbo_words=vbo_words, shader_program_words=shader_program_words
    )

    return res
end#prepereForDispStruct


"""
coordinating displaying - sets needed constants that are storeds in  forDisplayConstants; and configures interactions from GLFW events
listOfTextSpecs - holds required data needed to initialize textures
keeps also references to needed ..Uniforms etc.
windowWidth::Int,windowHeight::Int - GLFW window dimensions
fractionOfMainIm - how much of width should be taken by the main image
heightToWithRatio - needed for proper display of main texture - so it would not be stretched ...
"""
function coordinateDisplay(listOfTextSpecsPrim::Vector{TextureSpec}, fractionOfMainIm::Float32, dataToScrollDims::DataToScrollDims=DataToScrollDims(), windowWidth::Int=1200, windowHeight::Int=Int(round(windowWidth * fractionOfMainIm)), textTexturewidthh::Int32=Int32(2000), textTextureheightt::Int32=Int32(round((windowHeight / (windowWidth * (1 - fractionOfMainIm)))) * textTexturewidthh), windowControlStruct::WindowControlStruct=WindowControlStruct())

    #setting number to texture that will be needed in shader configuration
    listOfTextSpecs::Vector{TextureSpec} = map(x -> setproperties(x[2], (whichCreated = x[1])), enumerate(listOfTextSpecsPrim))

    #calculations of necessary constants needed to calculate window size , mouse position ...
    calcDimStruct = CalcDimsStruct(windowWidth=windowWidth, windowHeight=windowHeight, fractionOfMainIm=fractionOfMainIm, wordsImageQuadVert=ShadersAndVerticiesForText.getWordsVerticies(fractionOfMainIm), wordsQuadVertSize=sizeof(ShadersAndVerticiesForText.getWordsVerticies(fractionOfMainIm)), textTexturewidthh=textTexturewidthh, textTextureheightt=textTextureheightt) |>
                    (calcDim) -> getHeightToWidthRatio(calcDim, dataToScrollDims) |>
                                 (calcDim) -> getMainVerticies(calcDim)
    #    put!(mainMedEye3dInstance.channel, calcDimStruct)


    #creating window and event listening loop
    window, vertex_shader, fragment_shader, shader_program, vbo, ebo, fragment_shader_words, vbo_words, shader_program_words, gslsStr = PrepareWindow.displayAll(listOfTextSpecs, calcDimStruct)

    GLFW.MakeContextCurrent(window)
    # than we set those ..Uniforms, open gl types and using data from arguments  to fill texture specifications
    mainImageUnifs, listOfTextSpecsMapped = assignUniformsAndTypesToMasks(listOfTextSpecs, shader_program, windowControlStruct)

    #@info "listOfTextSpecsMapped" listOfTextSpecsMapped
    #initializing object that holds data reqired for interacting with opengl
    initializedTextures = initializeTextures(listOfTextSpecsMapped, calcDimStruct)

    numbDict = filter(x -> x.numb >= 0, initializedTextures) |>
               (filtered) -> Dictionary(map(it -> it.numb, filtered), collect(eachindex(filtered))) # a way for fast query using assigned numbers

    forDispObj = forDisplayObjects(
        listOfTextSpecifications=initializedTextures, window=window, vertex_shader=vertex_shader, fragment_shader=fragment_shader, shader_program=shader_program, vbo=vbo[], ebo=ebo[], mainImageUniforms=mainImageUnifs, TextureIndexes=Dictionary(map(it -> it.name, initializedTextures), collect(eachindex(initializedTextures))), numIndexes=numbDict, gslsStr=gslsStr, windowControlStruct=windowControlStruct
    )
    #finding some texture that can be modifid and set as one active for modifications
    # put!(mainMedEye3dInstance.channel, forDispObj)
    #in order to clean up all resources while closing



    #passing for text display object
    forTextDispStruct = prepareForDispStruct(length(initializedTextures), fragment_shader_words, vbo_words, shader_program_words, window, textTexturewidthh, textTextureheightt, forDispObj)


    # put!(mainMedEye3dInstance.channel, forTextDispStruct)
    function consumer(mainChannel::Base.Channel{Any})
        shouldStop = [false]
        stateInstance = StateDataFields()
        stateInstance.textureToModifyVec = filter(it -> it.isEditable, initializedTextures)
        #    in case we are recreating all we need to destroy old textures ... generally simplest is destroy window
        function cleanUp()
            obj = stateInstance.mainForDisplayObjects
            glDeleteTextures(length(obj.listOfTextSpecifications), map(text -> text.ID, obj.listOfTextSpecifications))
            glFlush()
            GLFW.DestroyWindow(obj.window)
            shouldStop[1] = true
            GLFW.Terminate()
        end #cleanUp

        if (typeof(stateInstance.mainForDisplayObjects.window) == GLFW.Window)
            cleanUp()
        end#
        GLFW.SetWindowCloseCallback(window, (_) -> cleanUp())

        while !shouldStop[1]
            channelData = take!(mainChannel)
            # get the aggregation here, only when the type is mouseStruct.
            if typeof(channelData) == MouseStruct
                mouseStructAggregationArray::Vector{MouseStruct} = [channelData]
                while !isempty(mainChannel) && typeof(fetch(mainChannel)) == MouseStruct
                    push!(mouseStructAggregationArray, take!(mainChannel))
                end
                channelData = mouseStructAggregationArray
            end
            on_next!(stateInstance, channelData)

        end
    end #end of consumer

    mainMedEye3dInstance = MainMedEye3d(channel=Base.Channel{Any}(consumer, 1000; spawn=false, threadpool=:interactive))
    # mainMedEye3dInstance = MainMedEye3d(channel = Base.Channel{Any}(1000))
    put!(mainMedEye3dInstance.channel, calcDimStruct)
    put!(mainMedEye3dInstance.channel, forDispObj)
    put!(mainMedEye3dInstance.channel, forTextDispStruct)


    registerInteractions(window, mainMedEye3dInstance, calcDimStruct)#passing needed subscriptions from GLFW
    # errormonitor(@async consumer(mainMedEye3dInstance.channel))
    return mainMedEye3dInstance
end #coordinateDisplay

end #SegmentationDisplay
