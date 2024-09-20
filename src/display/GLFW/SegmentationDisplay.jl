"""
Main module controlling displaying segmentations image and data

"""
module SegmentationDisplay
export loadRegisteredImages, displayImage, coordinateDisplay, passDataForScrolling

using ColorTypes, MedImages, ModernGL, GLFW, Dictionaries, Logging, Setfield, FreeTypeAbstraction, Statistics, Observables
using ..PrepareWindow, ..TextureManag, ..OpenGLDisplayUtils, ..ForDisplayStructs, ..Uniforms, ..DisplayWords
using ..ReactingToInput, ..ReactToScroll, ..ShadersAndVerticiesForText, ..DisplayWords, ..DataStructs, ..StructsManag
using ..ReactOnKeyboard, ..ReactOnMouseClickAndDrag, ..DisplayDataManag
using ..sv_shaders_etc


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
on_next!(stateObject::StateDataFields, data::DisplayedVoxels) = retrieveVoxelArray(data, stateObject)
on_next!(stateObject::StateDataFields, data::CustomDisplayedVoxels) = depositVoxelArray(data, stateObject)
on_error!(stateObject::StateDataFields, err) = error(err)
on_complete!(stateObject::StateDataFields) = ""


"""
is used to pass into the actor data that will be used for scrolling
onScrollData - struct holding between others list of tuples where first is the name of the texture that we provided and second is associated data (3 dimensional array of appropriate type)
"""

function passDataForScrolling(
    mainMedEye3dInstance::MainMedEye3d,
    onScrollData::FullScrollableDat
)
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
function registerInteractions(
    window::GLFW.Window,
    mainMedEye3dInstance::MainMedEye3d,
    calcDimStruct::CalcDimsStruct
)
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
coordinating displaying - sets needed constants that are storeds in  forDisplayConstants; and configures interactions from GLFW events
listOfTextSpecs - holds required data needed to initialize textures
keeps also references to needed ..Uniforms etc.
windowWidth::Int,windowHeight::Int - GLFW window dimensions
fractionOfMainIm - how much of width should be taken by the main image
heightToWithRatio - needed for proper display of main texture - so it would not be stretched ...
"""
function coordinateDisplay(
    listOfTextSpecsPrim::Vector{TextureSpec},
    fractionOfMainIm::Float32,
    dataToScrollDims::DataToScrollDims=DataToScrollDims(),
    windowWidth::Int=1200,
    windowHeight::Int=Int(round(windowWidth * fractionOfMainIm)),
    textTexturewidthh::Int32=Int32(2000),
    textTextureheightt::Int32=Int32(round((windowHeight / (windowWidth * (1 - fractionOfMainIm)))) * textTexturewidthh),
    windowControlStruct::WindowControlStruct=WindowControlStruct())

    #setting number to texture that will be needed in shader configuration
    #enumerate function returns index,value pair of each item in an array, here for the TextureSpecStruct, setting the whichCreated field to the current index
    listOfTextSpecs::Vector{TextureSpec} = map(x -> setproperties(x[2], (whichCreated = x[1])), enumerate(listOfTextSpecsPrim))

    #calculations of necessary constants needed to calculate window size , mouse position ...
    calcDimStruct = CalcDimsStruct(
        windowWidth=windowWidth,
        windowHeight=windowHeight,
        fractionOfMainIm=fractionOfMainIm,
        wordsImageQuadVert=ShadersAndVerticiesForText.getWordsVerticies(fractionOfMainIm),
        wordsQuadVertSize=sizeof(ShadersAndVerticiesForText.getWordsVerticies(fractionOfMainIm)),
        textTexturewidthh=textTexturewidthh,
        textTextureheightt=textTextureheightt) |>
                    (calcDim) -> getHeightToWidthRatio(calcDim, dataToScrollDims) |>
                                 (calcDim) -> getMainVerticies(calcDim)

    #    put!(mainMedEye3dInstance.channel, calcDimStruct)


    #creating window and event listening loop

    #### supervoxel added VAO in line below
    window, vertex_shader, fragment_shader, shader_program, vbo, ebo, fragment_shader_words, vbo_words, shader_program_words, gslsStr,VAO = PrepareWindow.displayAll(listOfTextSpecs, calcDimStruct)

    GLFW.MakeContextCurrent(window)
    # than we set those ..Uniforms, open gl types and using data from arguments  to fill texture specifications
    listOfTextSpecsMapped = assignUniformsAndTypesToMasks(listOfTextSpecs, shader_program, windowControlStruct)

    #@info "listOfTextSpecsMapped" listOfTextSpecsMapped
    #initializing object that holds data reqired for interacting with opengl
    initializedTextures = initializeTextures(listOfTextSpecsMapped, calcDimStruct)

    numbDict = filter(x -> x.numb >= 0, initializedTextures) |>
               (filtered) -> Dictionary(map(it -> it.numb, filtered), collect(eachindex(filtered))) # a way for fast query using assigned numbers

    forDispObj = forDisplayObjects(
        listOfTextSpecifications=initializedTextures,
        window=window,
        vertex_shader=vertex_shader,
        fragment_shader=fragment_shader,
        shader_program=shader_program,
        vbo=vbo[],
        ebo=ebo[],
        imageUniforms=listOfTextSpecsMapped[1].uniforms,
        TextureIndexes=Dictionary(map(it -> it.name, initializedTextures),
            collect(eachindex(initializedTextures))),
        numIndexes=numbDict,
        gslsStr=gslsStr,
        windowControlStruct=windowControlStruct
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
Loading Nifti volumes or Dicom Series with MedImages.jl package
Currently, there is only support for 1 image load and visualization at a time, Subjected to change
"""
function loadRegisteredImages(
    studySrc::Union{Vector{String},String}
)

    medImageDataInstances::Vector{MedImages.MedImage} = []

    if typeof(studySrc) == String
        push!(medImageDataInstances, MedImages.load_image(studySrc))
    elseif typeof(studySrc) == Vector{String}
        for studySrcPath in studySrc
            push!(medImageDataInstances, MedImages.load_image(studySrcPath))
        end
    end

    for medImageDataInstance in medImageDataInstances
        #permuting the voxelData to some default orientation, such that the image is not inverted or sideways
        medImageDataInstance.voxel_data = permutedims(medImageDataInstance.voxel_data, (1, 2, 3)) #previously in the test script the default was (3, 2, 1)
        sizeInfo = size(medImageDataInstance.voxel_data)
        for outerNum in 1:sizeInfo[1]
            for innerNum in 1:sizeInfo[3]
                medImageDataInstance.voxel_data[outerNum, :, innerNum] = reverse(medImageDataInstance.voxel_data[outerNum, :, innerNum])
            end
        end
        #Float conversion happens here for voxelData, currently only Floats are supported to keep it simple
        medImageDataInstance.voxel_data = Float32.(medImageDataInstance.voxel_data)
    end


    return medImageDataInstances
end




"""
High Level Initialisation function for the visualizer
"""
function displayImage(
    studySrc::Union{Vector{String},String},
    textureSpecArray::Vector{TextureSpec}=Vector{TextureSpec}(),
    voxelDataTupleVector::Vector{Any}=[],
    spacings::Vector{Tuple{Float64,Float64,Float64}}=Vector{Tuple{Float64,Float64,Float64}}(),
    fractionOfMainImage::Float32=Float32(0.8)
)

    medImageData::Vector{MedImages.MedImage} = loadRegisteredImages(studySrc)
    #NOTE : for overlaid images, they need to be resampled first
    #NOIE : Dicom is currently not supported, due to the lack of support for Dicom in MedImages.jl

    if isempty(textureSpecArray) && isempty(voxelDataTupleVector) && isempty(spacings)
        for (index, medImage) in enumerate(medImageData)
            if medImage.image_type == MedImages.MedImage_data_struct.PET_type
                push!(textureSpecArray, getDefaultTexture(MedImages.MedImage_data_struct.PET_type, Int32(index)))
                push!(voxelDataTupleVector, ("PET", medImage.voxel_data))
                push!(spacings, medImage.spacing)
            elseif medImage.image_type == MedImages.MedImage_data_struct.CT_type
                push!(textureSpecArray, getDefaultTexture(MedImages.MedImage_data_struct.CT_type, Int32(index)))
                push!(voxelDataTupleVector, ("CTIm", medImage.voxel_data))
                push!(spacings, medImage.spacing)
            end

        end
    end

    #for correct display and windowing for PET we do (median -std /2) for min and (median + std * 2) for max

    for textur in textureSpecArray
        if textur.studyType == "PET"
            textur.minAndMaxValue = Float32.([median(voxelDataTupleVector[1][2]) - std(voxelDataTupleVector[1][2]) / 2, median(voxelDataTupleVector[1][2]) + std(voxelDataTupleVector[1][2]) * 2])
        end
    end

    #Texture specification for manual modification Mask
    insert!(textureSpecArray, 2, getDefaultTexture("ManualModif", Int32(2)))


    voxelDataForUniforms::Vector{Array{Float32}} = map(x -> x[2], voxelDataTupleVector)

    #Since there are repeating Tuples for Manual Modif, we need to ensure only a unique ones exist based on the first loaded image
    insert!(voxelDataTupleVector, 2, ("manualModif", zeros(Float32, size(voxelDataForUniforms[1]))))

    datToScrollDimsB = DataToScrollDims(imageSize=size(voxelDataForUniforms[1]), voxelSize=spacings[1], dimensionToScroll=3)
    mainLines = textLinesFromStrings(["main line 1", "main line 2"])
    supplLines = map(x -> textLinesFromStrings(["sub line 1 in $(x)", "sub line 2 in $(x)"]), 1:size(voxelDataForUniforms[1])[3])

    sliceData = getThreeDims(voxelDataTupleVector)
    mainScrollData = FullScrollableDat(dataToScrollDims=datToScrollDimsB, dimensionToScroll=1, dataToScroll=sliceData, mainTextToDisp=mainLines, sliceTextToDisp=supplLines)

    # Few assertions to ensure correct types between the textureSpecification type and the voxel data type
    for (textureSpec, tupleVector) in zip(textureSpecArray, voxelDataTupleVector)
        @assert typeof(textureSpec) == TextureSpec{Float32}
        # @info typeof(voxelData)
        @assert typeof(tupleVector[2]) == Array{Float32,3}

        @assert textureSpec.name == tupleVector[1]
        # @info typeof(voxelData)
    end

    medEye3dChannelInstance = coordinateDisplay(textureSpecArray, fractionOfMainImage, datToScrollDimsB, 1000)



    #Populating the fields for mainMedEye3dInstance
    medEye3dChannelInstance.voxelArrayShapes = map(x -> size(x[2]), voxelDataTupleVector)
    medEye3dChannelInstance.voxelArrayTypes = map(x -> typeof(x[2][1, 1, 1]), voxelDataTupleVector) #getting the type of the first element


    passDataForScrolling(medEye3dChannelInstance, mainScrollData)
    return medEye3dChannelInstance
end


end #SegmentationDisplay
