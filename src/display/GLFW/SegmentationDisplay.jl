"""
Main module controlling displaying segmentations image and data

"""
module SegmentationDisplay
export loadRegisteredImages, displayImage, coordinateDisplay, passDataForScrolling

using ColorTypes, MedImages, ModernGL, GLFW, Dictionaries, Logging, Setfield, FreeTypeAbstraction, Statistics, Observables
using ..PrepareWindow, ..PrepareWindowHelpers, ..TextureManag, ..OpenGLDisplayUtils, ..ForDisplayStructs, ..Uniforms, ..DisplayWords
using ..ReactingToInput, ..ReactToScroll, ..ShadersAndVerticiesForText, ..ShadersAndVerticiesForLine, ..ShadersAndVerticiesForSupervoxels, ..DisplayWords, ..DataStructs, ..StructsManag
using ..ReactOnKeyboard, ..ReactOnMouseClickAndDrag, ..DisplayDataManag

#  do not copy it into the consumer function
"""
configuring consumer function on_next! function using multiple dispatch mechanism in order to connect input to proper functions
"""
on_next!(stateObjects::Vector{StateDataFields}, data::Int64) = reactToScroll(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::forDisplayObjects) = setUpMainDisplay(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::ForWordsDispStruct) = setUpWordsDisplay(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::CalcDimsStruct) = setUpCalcDimsStruct(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::valueForMasToSetStruct) = setUpvalueForMasToSet(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::FullScrollableDat) = setUpForScrollData(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::SingleSliceDat) = updateSingleImagesDisplayedSetUp(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::Vector{MouseStruct}) = react_to_draw(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::MouseStruct) = reactToMouseDrag(data, stateObjects) #needs modification , with the react_to_draw, data of vectorStruct (MoustStruct)
on_next!(stateObjects::Vector{StateDataFields}, data::KeyInputFields) = reactToKeyInput(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::DisplayedVoxels) = retrieveVoxelArray(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::CustomDisplayedVoxels) = depositVoxelArray(data, stateObjects)
on_error!(stateObjects::Vector{StateDataFields}, err) = error(err)
on_complete!(stateObjects::Vector{StateDataFields}) = ""






"""
is used to pass into the actor data that will be used for scrolling
onScrollData - struct holding between others list of tuples where first is the name of the texture that we provided and second is associated data (3 dimensional array of appropriate type)
"""
function passDataForScrolling(
    mainMedEye3dInstance::MainMedEye3d,
    onScrollData::Union{FullScrollableDat,Vector{FullScrollableDat}}
)
    # """
    # put data onto the channel, matching types with on_next.

    # put the data in the onScrollData which screen need to be
    # """

    #modify here the data for passing onto the channel
    if typeof(onScrollData) == FullScrollableDat
        put!(mainMedEye3dInstance.channel, onScrollData)
    elseif typeof(onScrollData) == Vector{FullScrollableDat}
        foreach(enumerate(onScrollData)) do (index, onScrollDataInstance)
            onScrollDataInstance.imagePos = index
            put!(mainMedEye3dInstance.channel, onScrollDataInstance)
        end
    end

    #we can put the vector of onScrollData into the channel.
end



"""
is using the actor that is instantiated in this module and connects it to GLFW context
by invoking appropriate registering functions and passing to it to the main Actor controlling input
"""
function registerInteractions(
    window::GLFW.Window,
    mainMedEye3dInstance::MainMedEye3d,
    calcDimStruct::Union{CalcDimsStruct,Vector{CalcDimsStruct}}
)
    if typeof(calcDimStruct) == CalcDimsStruct
        subscribeGLFWtoActor(window, mainMedEye3dInstance, calcDimStruct)
    elseif typeof(calcDimStruct) == Vector{CalcDimsStruct}
        foreach(calcDimStruct) do currentCalcDim
            subscribeGLFWtoActor(window, mainMedEye3dInstance, currentCalcDim)
        end
    end
    # subscribeGLFWtoActor(window, mainMedEye3dInstance, calcDimStruct)
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
function prepareForDispStruct(
    numberOfActiveTextUnits::Int,
    fragment_shader_words::UInt32,
    vbo_words::Base.RefValue{UInt32},
    shader_program_words::UInt32,
    window,
    widthh::Int32=Int32(1),
    heightt::Int32=Int32(1),
    forDispObj::forDisplayObjects=forDisplayObjects()
)::ForWordsDispStruct

    res = ForWordsDispStruct(
        fontFace=FTFont(joinpath(dirname(dirname(pathof(FreeTypeAbstraction))), "test", "hack_regular.ttf")), textureSpec=createTextureForWords(numberOfActiveTextUnits, widthh, heightt, getProperGL_TEXTURE(numberOfActiveTextUnits + 1)), fragment_shader_words=fragment_shader_words, vbo_words=vbo_words, shader_program_words=shader_program_words
    )

    return res
end#prepereForDispStruct



"""
Returns the display mode of the visualizer
"""
function getDisplayMode(listOfTextSpecs::Union{Vector{TextureSpec},Vector{Vector{TextureSpec}}})::DisplayMode
    if typeof(listOfTextSpecs) == Vector{TextureSpec}
        return SingleImage
    elseif typeof(listOfTextSpecs) == Vector{Vector{TextureSpec}}
        return MultiImage
    end
end


"""
Carries out the initialization of shader and buffers for
SuperVoxels
"""
function initializeSupervoxels(vertex_shader, vao, ebo, vboVector, svVertAndInd)
    vbo = vboVector[1] #Sinlge single image mode only rect vertex buffer
    fragment_shader_supervoxel, shader_program_supervoxel = ShadersAndVerticiesForSupervoxels.createAndInitSupervoxelLineShaderProgram(vertex_shader)
    vao_supervoxel = PrepareWindowHelpers.createVertexBuffer()
    vbo_supervoxel = PrepareWindowHelpers.createDynamicDAtaBuffer(svVertAndInd["supervoxel_vertices"])
    ebo_supervoxel = PrepareWindowHelpers.createElementBuffer(svVertAndInd["supervoxel_indices"])

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Float32), Ptr{Nothing}(0))
    glEnableVertexAttribArray(0)

    glBindVertexArray(vao[])

    superVoxels::GlShaderAndBufferFields = GlShaderAndBufferFields(
        shaderProgram=shader_program_supervoxel,
        fragmentShader=fragment_shader_supervoxel,
        vao=vao_supervoxel,
        vbo=vbo_supervoxel,
        ebo=ebo_supervoxel
    )


    mainRect::GlShaderAndBufferFields = GlShaderAndBufferFields(
        vao=vao,
        vbo=vbo,
        ebo=ebo
    )

    return (superVoxels, mainRect)
end


"""
Carries out the initialization of shader and buffers for
crosshair
"""
function initializeCrosshair(vertex_shader, vao, ebo, vboVector, fragment_shader_words, vbo_words, shader_program_words)

    # glBindVertexArray(0) #unbinding vao for the main rect
    fragment_shader_line, shader_program_line = ShadersAndVerticiesForLine.createAndInitLineShaderProgram(vertex_shader)
    vao_line = PrepareWindowHelpers.createVertexBuffer()
    vbo_line = PrepareWindowHelpers.createDynamicDAtaBuffer(ShadersAndVerticiesForLine.line_vertices)
    ebo_line = PrepareWindowHelpers.createElementBuffer(ShadersAndVerticiesForLine.line_indices)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Float32), Ptr{Nothing}(0))
    glEnableVertexAttribArray(0)

    # glBindVertexArray(0) #unbinding vao for the crosshair
    glBindVertexArray(vao[]) #binding vao for the main rect

    crosshair::GlShaderAndBufferFields = GlShaderAndBufferFields(
        shaderProgram=shader_program_line,
        fragmentShader=fragment_shader_line,
        vao=vao_line,
        vbo=vbo_line,
        ebo=ebo_line
    )

    #we are not initializing shaderProgram in mainRects
    mainRects::Vector{GlShaderAndBufferFields} = []
    foreach(enumerate(vboVector)) do (index, vbo)
        push!(mainRects, GlShaderAndBufferFields(
            vao=vao,
            vbo=vbo,
            ebo=ebo
        ))
    end


    textFields = GlShaderAndBufferFields(
        shaderProgram=shader_program_words,
        fragmentShader=fragment_shader_words,
        vbo=vbo_words
    )

    return (crosshair, mainRects, textFields)
end


"""
coordinating displaying - sets needed constants that are storeds in  forDisplayConstants; and configures interactions from GLFW events
listOfTextSpecs - holds required data needed to initialize textures
keeps also references to needed ..Uniforms etc.
windowWidth::Int,windowHeight::Int - GLFW window dimensions
fractionOfMainIm - how much of width should be taken by the main image
heightToWithRatio - needed for proper display of main texture - so it would not be stretched ...
"""
function coordinateDisplay(
    listOfTextSpecsPrim::Union{Vector{TextureSpec},Vector{Vector{TextureSpec}}},
    fractionOfMainIm::Float32,
    dataToScrollDims::Union{DataToScrollDims,Vector{DataToScrollDims}}=DataToScrollDims(),
    spacing::Union{Vector{Tuple{Float64,Float64,Float64}},Vector{Vector{Tuple{Float64,Float64,Float64}}}}=Vector{Tuple{Float64,Float64,Float64}}(),
    origin::Union{Vector{Tuple{Float64,Float64,Float64}},Vector{Vector{Tuple{Float64,Float64,Float64}}}}=Vector{Tuple{Float64,Float64,Float64}}(),
    svVertAndInd::Dict{String,Vector}=Dict{String,Vector}("supervoxel_vertices" => [], "supervoxel_indices" => []),
    windowWidth::Int=1200,
    windowHeight::Int=Int(round(windowWidth * fractionOfMainIm)),
    textTexturewidthh::Int32=Int32(2000),
    textTextureheightt::Int32=Int32(round((windowHeight / (windowWidth * (1 - fractionOfMainIm)))) * textTexturewidthh),
    windowControlStruct::WindowControlStruct=WindowControlStruct()
)


    displayMode = getDisplayMode(listOfTextSpecsPrim)
    #setting number to texture that will be needed in shader configuration
    #enumerate function returns index,value pair of each item in an array, here for the TextureSpecStruct, setting the whichCreated field to the current index
    listOfTextSpecs::Union{Vector{TextureSpec{Float32}},Vector{Vector{TextureSpec{Float32}}}} = (typeof(listOfTextSpecsPrim) == Vector{TextureSpec}) ? map(x -> setproperties(x[2], (whichCreated = x[1])), enumerate(listOfTextSpecsPrim)) : map(innerVector -> map(x -> setproperties(x[2], (whichCreated = x[1])), enumerate(innerVector)), listOfTextSpecsPrim)

    #calculations of necessary constants needed to calculate window size , mouse position ...

    #we need multiple calcDims if the current display mode is multi Image display, evident from the inner vectors
    calcDimStructs::Vector{CalcDimsStruct} = Vector{CalcDimsStruct}()


    if typeof(dataToScrollDims) == DataToScrollDims
        push!(calcDimStructs, CalcDimsStruct(
            windowWidth=windowWidth,
            windowHeight=windowHeight,
            fractionOfMainIm=fractionOfMainIm,
            wordsImageQuadVert=ShadersAndVerticiesForText.getWordsVerticies(fractionOfMainIm),
            wordsQuadVertSize=sizeof(ShadersAndVerticiesForText.getWordsVerticies(fractionOfMainIm)),
            textTexturewidthh=textTexturewidthh,
            textTextureheightt=textTextureheightt) |>
                              (calcDim) -> getHeightToWidthRatio(calcDim, dataToScrollDims) |>
                                           (calcDim) -> getMainVerticies(calcDim, displayMode, 1)) #passing Image index as 1 for Single Image display mode
    elseif typeof(dataToScrollDims) == Vector{DataToScrollDims}
        foreach(enumerate(dataToScrollDims)) do (imageIndex, scrollDims)
            push!(calcDimStructs, CalcDimsStruct(
                windowWidth=windowWidth,
                windowHeight=windowHeight,
                fractionOfMainIm=fractionOfMainIm,
                wordsImageQuadVert=ShadersAndVerticiesForText.getWordsVerticies(fractionOfMainIm),
                wordsQuadVertSize=sizeof(ShadersAndVerticiesForText.getWordsVerticies(fractionOfMainIm)),
                textTexturewidthh=textTexturewidthh,
                textTextureheightt=textTextureheightt,
                imagePos=imageIndex) |>
                                  (calcDim) -> getHeightToWidthRatio(calcDim, scrollDims) |>
                                               (calcDim) -> getMainVerticies(calcDim, displayMode, imageIndex))
        end
    end

    #    put!(mainMedEye3dInstance.channel, calcDimStruct)


    #creating window and event listening loop

    #we can pass the first calcDimStruct here, since we need to get the window height and width which is same across all the calcDims
    window, vertex_shader, vao, ebo, fragment_shader_words, vbo_words, shader_program_words, gslsStr = PrepareWindow.displayAll(calcDimStructs[1])



    fragmentShaderVector = []
    shaderProgramVector = []
    vboVector = []


    listOfTextSpecsMapped::Vector{Vector{TextureSpec}} = []

    #here for fragShaderA we have fragmentShaderVector[1], shaderProgramVector[1], vboVector[1]
    foreach(enumerate(calcDimStructs)) do (index, calcDimStruct)
        fragShad, shadProg, vbo = PrepareWindow.createAndInitShaderProgram(vertex_shader, typeof(listOfTextSpecs) == Vector{TextureSpec{Float32}} ? listOfTextSpecs : listOfTextSpecs[index], gslsStr, calcDimStruct, index)
        push!(fragmentShaderVector, fragShad)
        push!(shaderProgramVector, shadProg)
        push!(vboVector, vbo)
        push!(listOfTextSpecsMapped, assignUniformsAndTypesToMasks(typeof(listOfTextSpecs) == Vector{TextureSpec{Float32}} ? listOfTextSpecs : listOfTextSpecs[index], shaderProgramVector[index]))
        # @info "look here" index calcDimStruct.mainImageQuadVert
    end



    """
    begin crosshair line rendering statements

    Crosshair constraints :
    1. Only in multi-image display mode
    2. Share the same crosshair with both the states
    3. Reveal slices in the same spatial area as the image from our mouse cursor.
    4. Reduce the size of the crosshair
    5. Make sure the correct vbo in multi-image are modified , initialize them properly in the relevanat states
    """

    #Crosshair display only in multi-image mode
    crosshair, mainRects, textFields = (Nothing, Nothing, Nothing)

    supervoxel, mainRect = (Nothing, Nothing)
    if displayMode == MultiImage
        crosshair, mainRects, textFields = initializeCrosshair(vertex_shader, vao, ebo, vboVector, fragment_shader_words, vbo_words, shader_program_words)
    elseif displayMode == SingleImage
        supervoxel, mainRect = initializeSupervoxels(vertex_shader, vao, ebo, vboVector, svVertAndInd)
    end

    """
    end crosshair
    """




    GLFW.MakeContextCurrent(window)
    # than we set those ..Uniforms, open gl types and using data from arguments  to fill texture specifications
    # listOfTextSpecsMapped::Vector{Vector{TextureSpec}} = []
    # foreach(enumerate(listOfTextSpecs)) do (index, textSpecVector)
    # push!(listOfTextSpecsMapped, assignUniformsAndTypesToMasks(textSpecVector, shaderProgramVector[index]))
    # end

    #@info "listOfTextSpecsMapped" listOfTextSpecsMapped
    #initializing object that holds data reqired for interacting with opengl
    initializedTextures::Vector{Vector{TextureSpec}} = []
    foreach(enumerate(listOfTextSpecsMapped)) do (index, mappedTextSpecVector)
        push!(initializedTextures, initializeTextures(mappedTextSpecVector, calcDimStructs[index]))
    end


    numbDicts = []
    foreach(initializedTextures) do initializedTexture
        push!(numbDicts, filter(x -> x.numb >= 0, initializedTexture) |>
                         (filtered) -> Dictionary(map(it -> it.numb, filtered), collect(eachindex(filtered)))) # a way for fast query using assigned numbers
    end


    forDispObjs::Vector{forDisplayObjects} = Vector{forDisplayObjects}()
    foreach(enumerate(initializedTextures)) do (index, initTextureVector)
        push!(forDispObjs, forDisplayObjects(
            listOfTextSpecifications=initTextureVector,
            window=window,
            vertex_shader=vertex_shader,
            fragment_shader=fragmentShaderVector[index],
            shader_program=shaderProgramVector[index],
            vbo=vboVector[index][],
            ebo=ebo[],
            imageUniforms=listOfTextSpecsMapped[index][1].uniforms,
            TextureIndexes=Dictionary(map(it -> it.name, initTextureVector),
                collect(eachindex(initTextureVector))),
            numIndexes=numbDicts[index],
            gslsStr=gslsStr,
            windowControlStruct=windowControlStruct,
            imagePos=index
        ))
    end
    #finding some texture that can be modifid and set as one active for modifications
    # put!(mainMedEye3dInstance.channel, forDispObj)
    #in order to clean up all resources while closing



    #passing for text display object
    #hardcoding text for now for the left image display
    # forTextDispStruct = prepareForDispStruct(length(initializedTextures[1]), fragment_shader_words, vbo_words, shader_program_words, window, textTexturewidthh, textTextureheightt, forDispObjs[1])
    forTextDispStructs::Vector{ForWordsDispStruct} = Vector{ForWordsDispStruct}()
    foreach(enumerate(initializedTextures)) do (index, initTextureVector)
        push!(forTextDispStructs, prepareForDispStruct(
            length(initTextureVector),
            fragment_shader_words,
            vbo_words,
            shader_program_words,
            window,
            textTexturewidthh,
            textTextureheightt,
            forDispObjs[index]
        ))
    end





    # put!(mainMedEye3dInstance.channel, forTextDispStruct)
    function consumer(mainChannel::Base.Channel{Any})
        shouldStop = [false]
        stateInstances::Vector{StateDataFields} = Vector{StateDataFields}()
        if displayMode == MultiImage
            # Initialization of GlShaderAndBufferFields for crosshair so different StateDataFields in multi-image mode
            stateInstances = [StateDataFields(displayMode=displayMode, imagePosition=index, switchIndex=index, crosshairFields=crosshair, mainRectFields=mainRects[index], textFields=textFields, spacingsValue=spacing[index], originValue=origin[index]) for (index, _) in enumerate(initializedTextures)]
        else
            stateInstances = [StateDataFields(displayMode=displayMode, imagePosition=index, switchIndex=index, spacingsValue=spacing[index], originValue=origin[index], supervoxelFields=supervoxel, mainRectFields=mainRect, supervoxelVertAndInd=svVertAndInd) for (index, _) in enumerate(initializedTextures)]
        end
        #Setting second state information to be 0, because we need to access information from the first state only
        if length(stateInstances) > 1 && displayMode == MultiImage
            stateInstances[2].switchIndex = 0
        end


        foreach(enumerate(stateInstances)) do (index, stateInstance)
            stateInstance.textureToModifyVec = filter(it -> it.isEditable, initializedTextures[index])
        end
        #    in case we are recreating all we need to destroy old textures ... generally simplest is destroy window

        function cleanUp()
            objs = []
            foreach(stateInstances) do stateInstance
                push!(objs, stateInstance.mainForDisplayObjects)
            end
            foreach(objs) do obj
                glDeleteTextures(length(obj.listOfTextSpecifications), map(text -> text.ID, obj.listOfTextSpecifications))
                glFlush()
                GLFW.DestroyWindow(obj.window)
            end
            shouldStop[1] = true
            GLFW.Terminate()
        end #cleanUp

        if (typeof(stateInstances[1].mainForDisplayObjects.window) == GLFW.Window) #harcoded check to get only the first stateInstance since the window is same for all
            cleanUp()
        end#
        GLFW.SetWindowCloseCallback(window, (_) -> cleanUp())


        while !shouldStop[1]
            channelData = take!(mainChannel)
            # get the aggregation here, only when the type is mouseStruct.
            if typeof(channelData) == MouseStruct
                if (channelData.isLeftButtonDown)
                    mouseStructAggregationArray::Vector{MouseStruct} = [channelData]
                    while !isempty(mainChannel) && typeof(fetch(mainChannel)) == MouseStruct
                        push!(mouseStructAggregationArray, take!(mainChannel))
                    end
                    channelData = mouseStructAggregationArray
                end

            elseif typeof(channelData) == CalcDimsStruct || typeof(channelData) == forDisplayObjects || typeof(channelData) == FullScrollableDat
                stateInstances[1].switchIndex = channelData.imagePos > 1 ? 2 : 1  #Setting the current State to modify to be for the left or the right image
            end

            on_next!(stateInstances, channelData)

        end
    end #end of consumer

    mainMedEye3dInstance = MainMedEye3d(channel=Base.Channel{Any}(consumer, 1000; spawn=false, threadpool=:interactive), textDispObj=forTextDispStructs[1], displayMode=displayMode)
    # mainMedEye3dInstance = MainMedEye3d(channel = Base.Channel{Any}(1000))


    foreach(calcDimStructs) do currentCalcDim
        put!(mainMedEye3dInstance.channel, currentCalcDim)
    end

    foreach(forDispObjs) do currentDispObj
        put!(mainMedEye3dInstance.channel, currentDispObj)
    end




    put!(mainMedEye3dInstance.channel, forTextDispStructs[1])



    registerInteractions(window, mainMedEye3dInstance, calcDimStructs)#passing needed subscriptions from GLFW
    # errormonitor(@async consumer(mainMedEye3dInstance.channel))

    return mainMedEye3dInstance
end #coordinateDisplay


"""
Defining some default textures for PET, CT and ManualModif, subject to change
"""
function getDefaultTexture(
    studyType::Union{MedImages.MedImage_data_struct.Image_type,String},
    numbIndex::Int32
)
    if studyType == MedImages.MedImage_data_struct.PET_type
        return TextureSpec{Float32}(
            name="PET",
            studyType="PET",
            isContinuusMask=true,
            numb=numbIndex,
            colorSet=[RGB(0.0, 0.0, 0.0), RGB(1.0, 1.0, 0.0), RGB(1.0, 0.5, 0.0), RGB(1.0, 0.0, 0.0), RGB(1.0, 0.0, 0.0)],
            minAndMaxValue=Float32.([200, 8000])
        )
    elseif studyType == MedImages.MedImage_data_struct.CT_type
        return TextureSpec{Float32}(
            name="CTIm",
            studyType="CT",
            numb=numbIndex,
            color=RGB(1.0, 1.0, 1.0),
            minAndMaxValue=Float32.([0, 100])
        )
    elseif studyType == "ManualModif"
        return TextureSpec{Float32}(
            name="manualModif",
            numb=Int32(2),
            color=RGB(0.0, 1.0, 0.0),
            minAndMaxValue=Float32.([0, 1]),
            isEditable=true
        )

    else
        return TextureSpec{Float32}(
            name="default",
            studyType="CT",
            numb=Int32(4),
            colorSet=[RGB(0.0, 0.0, 0.0), RGB(1.0, 1.0, 1.0)]
        )
    end
end

"""
Loading Nifti volumes or Dicom Series with MedImages.jl package.
Single Image or Multi-Image display supported.
"""
function loadRegisteredImages(
    studySrc::Union{Vector{String},String,Vector{Vector{String}}}
)

    medImageDataInstances::Union{Vector{MedImages.MedImage},Vector{Vector{MedImages.MedImage}}} = typeof(studySrc) == Vector{Vector{String}} ? Vector{Vector{MedImages.MedImage}}() : Vector{MedImages.MedImage}()

    if typeof(studySrc) == String
        push!(medImageDataInstances, MedImages.load_image(studySrc))
    elseif typeof(studySrc) == Vector{String}
        for studySrcPath in studySrc
            push!(medImageDataInstances, MedImages.load_image(studySrcPath))
        end

    elseif typeof(studySrc) == Vector{Vector{String}}
        for studySrcVector in studySrc
            medImageInnerVector::Vector{MedImages.MedImage} = Vector{MedImages.MedImage}()
            for studySrcPath in studySrcVector
                push!(medImageInnerVector, MedImages.load_image(studySrcPath))
            end
            push!(medImageDataInstances, medImageInnerVector)
        end
    end



    if typeof(medImageDataInstances) == Vector{MedImages.MedImage}

        for medImageDataInstance in medImageDataInstances
            #permuting the voxelData to some default orientation, such that the image is not inverted or sideways
            medImageDataInstance.voxel_data = permutedims(medImageDataInstance.voxel_data, (3, 2, 1)) #previously in the test script the default was (3, 2, 1)
            sizeInfo = size(medImageDataInstance.voxel_data)
            for outerNum in 1:sizeInfo[1]
                for innerNum in 1:sizeInfo[3]
                    medImageDataInstance.voxel_data[outerNum, :, innerNum] = reverse(medImageDataInstance.voxel_data[outerNum, :, innerNum])
                end
            end
            #Float conversion happens here for voxelData, currently only Floats are supported to keep it simple
            medImageDataInstance.voxel_data = Float32.(medImageDataInstance.voxel_data)
        end

    elseif typeof(medImageDataInstances) == Vector{Vector{MedImages.MedImage}}
        for medImageInnerVector in medImageDataInstances
            for medImageDataInstance in medImageInnerVector
                medImageDataInstance.voxel_data = permutedims(medImageDataInstance.voxel_data, (3, 2, 1)) #previously in the test script the default was (3, 2, 1)
                sizeInfo = size(medImageDataInstance.voxel_data)
                for outerNum in 1:sizeInfo[1]
                    for innerNum in 1:sizeInfo[3]
                        medImageDataInstance.voxel_data[outerNum, :, innerNum] = reverse(medImageDataInstance.voxel_data[outerNum, :, innerNum])
                    end
                end
                #Float conversion happens here for voxelData, currently only Floats are supported to keep it simple
                medImageDataInstance.voxel_data = Float32.(medImageDataInstance.voxel_data)
            end
        end
    end

    return medImageDataInstances #returns the vector of MedImages or a Vector of Vector of MedImages
end




"""
High Level Initialisation function for the visualizer
"""
function displayImage(
    studySrc::Union{Vector{String},String,Vector{Vector{String}}}
    ; textureSpecArray::Union{Vector{TextureSpec},Vector{Vector{TextureSpec}}}=Vector{TextureSpec}(),
    voxelDataTupleVector::Union{Vector{Any},Vector{Vector{Any}}}=[],
    spacings::Union{Vector{Tuple{Float64,Float64,Float64}},Vector{Vector{Tuple{Float64,Float64,Float64}}}}=Vector{Tuple{Float64,Float64,Float64}}(),
    origins::Union{Vector{Tuple{Float64,Float64,Float64}},Vector{Vector{Tuple{Float64,Float64,Float64}}}}=Vector{Tuple{Float64,Float64,Float64}}(),
    fractionOfMainImage::Float32=Float32(0.8),
    windowWidth::Int=1000,
    svVertAndInd::Dict{String,Vector}=Dict{String,Vector}("supervoxel_vertices" => [], "supervoxel_indices" => [])
)


    #asserting that the length of the studySrc is 2, if it is a multi-dimensions vector
    if typeof(studySrc) == Vector{Vector{String}}
        try
            @assert length(studySrc) == 2
        catch assertionError
            @error "MedEye3d.jl currently do not support more than 2 images for comparison." assertionError
        end
    end

    medImageData::Union{Vector{MedImages.MedImage},Vector{Vector{MedImages.MedImage}}} = loadRegisteredImages(studySrc)
    #NOTE : for overlaid images, they need to be resampled first
    #NOIE : Dicom is currently not supported, due to the lack of support for Dicom in MedImages.jl

    if isempty(textureSpecArray) && isempty(voxelDataTupleVector) && isempty(spacings) && isempty(origins)
        #Reassigning textureSpecArray, voxelDataTupleVector, spacings  depending upong the typeof studySrc
        textureSpecArray = typeof(studySrc) == Vector{Vector{String}} ? Vector{Vector{TextureSpec}}() : Vector{TextureSpec}()
        voxelDataTupleVector = typeof(studySrc) == Vector{Vector{String}} ? Vector{Vector{Any}}() : Vector{Any}()
        spacings = typeof(studySrc) == Vector{Vector{String}} ? Vector{Vector{Tuple{Float64,Float64,Float64}}}() : Vector{Tuple{Float64,Float64,Float64}}()
        origins = typeof(studySrc) == Vector{Vector{String}} ? Vector{Vector{Tuple{Float64,Float64,Float64}}}() : Vector{Tuple{Float64,Float64,Float64}}()

        if typeof(medImageData) == Vector{MedImages.MedImage}
            for (index, medImage) in enumerate(medImageData)
                if medImage.image_type == MedImages.MedImage_data_struct.PET_type
                    push!(textureSpecArray, getDefaultTexture(MedImages.MedImage_data_struct.PET_type, Int32(index)))
                    push!(voxelDataTupleVector, ("PET", medImage.voxel_data))
                    push!(spacings, medImage.spacing)
                    push!(origins, medImage.origin)
                elseif medImage.image_type == MedImages.MedImage_data_struct.CT_type
                    push!(textureSpecArray, getDefaultTexture(MedImages.MedImage_data_struct.CT_type, Int32(index)))
                    push!(voxelDataTupleVector, ("CTIm", medImage.voxel_data))
                    push!(spacings, medImage.spacing)
                    push!(origins, medImage.origin)
                end

            end

        elseif typeof(medImageData) == Vector{Vector{MedImages.MedImage}}
            for medImageInnerVector in medImageData
                for (innerIndex, medImage) in enumerate(medImageInnerVector)


                    #==
                     check to ensure No texture has number 2, since it is reserved for ManualModif
                    ==#
                    innerIndex = innerIndex == 2 ? innerIndex + 1 : innerIndex

                    innerTextureSpecArray::Vector{TextureSpec} = Vector{TextureSpec}()
                    innerVoxelDataTupleVector::Vector{Any} = Vector{Any}()
                    innerSpacings::Vector{Tuple{Float64,Float64,Float64}} = Vector{Tuple{Float64,Float64,Float64}}()
                    innerOrigins::Vector{Tuple{Float64,Float64,Float64}} = Vector{Tuple{Float64,Float64,Float64}}()

                    if medImage.image_type == MedImages.MedImage_data_struct.PET_type
                        push!(innerTextureSpecArray, getDefaultTexture(MedImages.MedImage_data_struct.PET_type, Int32(innerIndex)))
                        push!(innerVoxelDataTupleVector, ("PET", medImage.voxel_data))
                        push!(innerSpacings, medImage.spacing)
                        push!(innerOrigins, medImage.origin)
                    elseif medImage.image_type == MedImages.MedImage_data_struct.CT_type
                        push!(innerTextureSpecArray, getDefaultTexture(MedImages.MedImage_data_struct.CT_type, Int32(innerIndex)))
                        push!(innerVoxelDataTupleVector, ("CTIm", medImage.voxel_data))
                        push!(innerSpacings, medImage.spacing)
                        push!(innerOrigins, medImage.origin)
                    end

                    push!(textureSpecArray, innerTextureSpecArray)
                    push!(voxelDataTupleVector, innerVoxelDataTupleVector)
                    push!(spacings, innerSpacings)
                    push!(origins, innerOrigins)
                end
            end

        end
    end

    #for correct display and windowing for PET we do (median -std /2) for min and (median + std * 2) for max

    if typeof(textureSpecArray) == Vector{TextureSpec}
        for textur in textureSpecArray
            if textur.studyType == "PET"
                textur.minAndMaxValue = Float32.([median(voxelDataTupleVector[1][2]) - std(voxelDataTupleVector[1][2]) / 2, median(voxelDataTupleVector[1][2]) + std(voxelDataTupleVector[1][2]) * 2])
            end
        end

    elseif typeof(textureSpecArray) == Vector{Vector{TextureSpec}}

        for (index, texturVector) in enumerate(textureSpecArray)
            for textur in texturVector
                if textur.studyType == "PET"
                    textur.minAndMaxValue = Float32.([median(voxelDataTupleVector[index][1][2]) - std(voxelDataTupleVector[index][1][2]) / 2, median(voxelDataTupleVector[index][1][2]) + std(voxelDataTupleVector[index][1][2]) * 2])
                end
            end
        end

    end


    #Texture specification for manual modification Mask
    if typeof(textureSpecArray) == Vector{TextureSpec}
        insert!(textureSpecArray, 2, getDefaultTexture("ManualModif", Int32(2)))
    elseif typeof(textureSpecArray) == Vector{Vector{TextureSpec}}
        for texturVector in textureSpecArray
            insert!(texturVector, 2, getDefaultTexture("ManualModif", Int32(2)))
        end
    end


    # @info "look here" typeof(voxelDataTupleVector) typeof(voxelDataTupleVector[1]) voxelDataTupleVector[1]
    # voxelDataForUniforms::Union{Vector{Array{Float32,3}},Vector{Vector{Array{Float32,3}}}} = map(x -> map(tup -> tup[2], x), voxelDataTupleVector)
    voxelDataForUniforms::Union{Vector{Array{Float32,3}},Vector{Vector{Array{Float32,3}}}} = typeof(voxelDataTupleVector) == Vector{Any} ? map(tuple -> tuple[2], voxelDataTupleVector) : map(innerVector -> map(tuple -> tuple[2], innerVector), voxelDataTupleVector)

    if typeof(voxelDataTupleVector) == Vector{Any}
        voxelDataForUniforms = map(tuple -> tuple[2], voxelDataTupleVector)
    elseif typeof(voxelDataTupleVector) == Vector{Vector{Any}}
        voxelDataForUniforms = map(innerVector -> map(tuple -> tuple[2], innerVector), voxelDataTupleVector)
    end

    #Since there are repeating Tuples for Manual Modif, we need to ensure only a unique ones exist based on the first loaded image

    if typeof(voxelDataTupleVector) == Vector{Any}
        insert!(voxelDataTupleVector, 2, ("manualModif", zeros(Float32, size(voxelDataForUniforms[1]))))
    elseif typeof(voxelDataTupleVector) == Vector{Vector{Any}}
        for (vectorIndex, innerVector) in enumerate(voxelDataTupleVector)
            insert!(innerVector, 2, ("manualModif", zeros(Float32, size(voxelDataForUniforms[vectorIndex][1]))))
        end
    end


    datToScrollDimsB::Union{DataToScrollDims,Vector{DataToScrollDims}} = typeof(voxelDataTupleVector) == Vector{Vector{Any}} ? Vector{DataToScrollDims}() : DataToScrollDims()
    mainLines::Union{Vector{Vector{SimpleLineTextStruct}},Vector{SimpleLineTextStruct}} = Vector{SimpleLineTextStruct}() #Subject to change
    supplLines::Union{Vector{Vector{Vector{SimpleLineTextStruct}}},Vector{Vector{SimpleLineTextStruct}}} = Vector{Vector{SimpleLineTextStruct}}() #Subject to change


    if typeof(voxelDataForUniforms) == Vector{Array{Float32,3}} #Our data is in Float32 format in 3 dimensions
        datToScrollDimsB = DataToScrollDims(imageSize=size(voxelDataForUniforms[1]), voxelSize=spacings[1], dimensionToScroll=3)
        mainLines = textLinesFromStrings(["main line 1", "main line 2"])
        supplLines = map(x -> textLinesFromStrings(["sub line 1 in $(x)", "sub line 2 in $(x)"]), 1:size(voxelDataForUniforms[1])[3])

    elseif typeof(voxelDataForUniforms) == Vector{Vector{Array{Float32,3}}}
        for (index, innerVector) in enumerate(voxelDataForUniforms)
            push!(datToScrollDimsB, DataToScrollDims(imageSize=size(innerVector[1]), voxelSize=spacings[index][1], dimensionToScroll=3))
        end
        mainLines = textLinesFromStrings(["main line 1", "main line 2"])
        supplLines = map(x -> textLinesFromStrings(["sub line 1 in $(x)", "sub line 2 in $(x)"]), 1:size(voxelDataForUniforms[1][1])[3]) #change this added [1] bcuz to get the first vector

    end



    sliceData::Union{Vector{ThreeDimRawDat{Float32}},Vector{Vector{ThreeDimRawDat{Float32}}}} = typeof(voxelDataTupleVector) == Vector{Vector{Any}} ? Vector{Vector{ThreeDimRawDat{Float32}}}() : Vector{ThreeDimRawDat{Float32}}()
    mainScrollData::Union{FullScrollableDat,Vector{FullScrollableDat}} = typeof(voxelDataTupleVector) == Vector{Vector{Any}} ? Vector{FullScrollableDat}() : FullScrollableDat()
    if typeof(voxelDataTupleVector) == Vector{Any}
        sliceDatad = getThreeDims(voxelDataTupleVector)
        # @info typeof(sliceDatad)
        sliceData = sliceDatad
        mainScrollData = FullScrollableDat(dataToScrollDims=datToScrollDimsB, dimensionToScroll=1, dataToScroll=sliceData, mainTextToDisp=mainLines, sliceTextToDisp=supplLines)

    elseif typeof(voxelDataTupleVector) == Vector{Vector{Any}}
        for (index, innerVector) in enumerate(voxelDataTupleVector)
            push!(sliceData, getThreeDims(innerVector))
            push!(mainScrollData, FullScrollableDat(dataToScrollDims=datToScrollDimsB[index], dimensionToScroll=1, dataToScroll=sliceData[index], mainTextToDisp=mainLines, sliceTextToDisp=supplLines))
        end
    end


    # Few assertions to ensure correct types between the textureSpecification type and the voxel data type

    if typeof(textureSpecArray) == Vector{TextureSpec}

        for (textureSpec, tupleVector) in zip(textureSpecArray, voxelDataTupleVector)
            @assert typeof(textureSpec) == TextureSpec{Float32}
            # @info typeof(voxelData)
            @assert typeof(tupleVector[2]) == Array{Float32,3}

            @assert textureSpec.name == tupleVector[1]
            # @info typeof(voxelData)
        end

    elseif typeof(textureSpecArray) == Vector{Vector{TextureSpec}}
        for (textureSpecVector, tupleVector) in zip(textureSpecArray, voxelDataTupleVector)
            for (textureSpec, tuple) in zip(textureSpecVector, tupleVector)
                @assert typeof(textureSpec) == TextureSpec{Float32}
                @assert typeof(tuple[2]) == Array{Float32,3}
                @assert textureSpec.name == tuple[1]
            end
        end
    end


    medEye3dChannelInstance = coordinateDisplay(textureSpecArray, fractionOfMainImage, datToScrollDimsB, spacings, origins, svVertAndInd, windowWidth)



    # Populating the fields for mainMedEye3dInstance
    # try
    displayMode = getDisplayMode(textureSpecArray)

    if displayMode == SingleImage
        medEye3dChannelInstance.voxelArrayShapes = map(x -> size(x[2]), voxelDataTupleVector)
        medEye3dChannelInstance.voxelArrayTypes = map(x -> typeof(x[2][1, 1, 1]), voxelDataTupleVector) #getting the type of the first element

        @info "!! Crosshair rendering is currently only supported in Multi image display mode !!"
    else
        @info "!! On Screen Voxel modification is currently only supported in Single image display mode !!"
    end



    passDataForScrolling(medEye3dChannelInstance, mainScrollData)
    return medEye3dChannelInstance
end


end #SegmentationDisplay


"""
DOCS::
Usage of interactive thread
In mutli-image only one image modality at a time can be visualized simultaneously. Either pet or ct.
During the initilization of states in consumer not all the fields of GlShaderAndBufferFields are populated. (for eg mainRectFields.shaderProgram)
Annotations are not saved and are cannot be undone.
Crosshair rendering is only supported in multi-image display mode.
Annotations are only supported in single-image display mode.
Disabling the concept of overlaid images in multi-image display mode. Thought manual-modification masks are working.
Advise Users to restart their Julia REPL session once they are done with the visualization
Advise Users to only change the plane of the left image in multi-image display for crosshair display.
ADvise users when willing to display hdf5 data first convert into nifti with the function and then display normally

NOTS:
return stuff similar to words_display for each calcDimStruct in the vector of calcDims
make changes to put forTextDispStruct in the mainMedEye3dInstance
allow user access to voxel modification in the case of single Image display  [DONE]
fix text rendering when in multi image
Add support for dynamic crosshair rendering on passive image
add support for supervoxel line rendering and sobel filter
Dynamic allocation of texture number no matter if the images are overlaid or not
Correct windowing for ct images f1, f2,f3
Test overlaid images in single Image and multi-image
With Crosshair rendering added, the keymaps for setting visiblity does not work
With Crosshair rendering added, the keymaps for changing windowing does not work
Allow People to load and visualize custom annotations masks [manual modifications]
In shader and vertices for supervoxels, during the calculation of vertices for supervoxels, make sure to use Float32 for calculation.
Add a sample nifti file in the supervoxel directory, since the function seem to be modifying the original nifti input image
 """
