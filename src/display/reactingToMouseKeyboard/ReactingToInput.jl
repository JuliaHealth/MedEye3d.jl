
module ReactingToInput
using GLFW, ModernGL, Setfield, DataTypesBasic, Base.Threads
using ..ReactToScroll, ..ForDisplayStructs, ..ReactOnKeyboard
using ..TextureManag, ..ReactOnMouseClickAndDrag, ..ReactOnKeyboard, ..DataStructs, ..StructsManag, ..DisplayWords
# using ..MaskDiffrence
using ..KeyboardVisibility, ..OtherKeyboardActions, ..WindowControll, ..ChangePlane
export subscribeGLFWtoActor, setUpForScrollData, setUpCalcDimsStruct, setUpWordsDisplay, setUpMainDisplay, setUpvalueForMasToSet, updateSingleImagesDisplayedSetUp


"""
adding the data into about openGL and GLFW context to enable proper display of main image and masks
"""
function setUpMainDisplay(mainForDisplayObjects::forDisplayObjects, mainState::StateDataFields)
    mainState.mainForDisplayObjects = mainForDisplayObjects
end#setUpMainDisplay

"""
adding the data needed for text display; also activates appropriate quad for the display
    it also configures texture that is build for text display
"""
function setUpWordsDisplay(textDispObject::ForWordsDispStruct, mainState::StateDataFields)

    bindAndActivateForText(textDispObject.shader_program_words, textDispObject.fragment_shader_words, mainState.mainForDisplayObjects.vertex_shader, textDispObject.vbo_words, mainState.calcDimsStruct)

    texId = createTexture(UInt8, mainState.calcDimsStruct.textTexturewidthh, mainState.calcDimsStruct.textTextureheightt, GL_R8UI, GL_UNSIGNED_BYTE)

    textSpec = setproperties(textDispObject.textureSpec, (ID = texId))

    samplerRef = glGetUniformLocation(textDispObject.shader_program_words, "TextTexture1")

    glUniform1i(samplerRef, length(mainState.mainForDisplayObjects.listOfTextSpecifications) + 1)
    textDispObjectiNITIALIZED = setproperties(textDispObject, (textureSpec = textSpec))

    mainState.textDispObj = textDispObjectiNITIALIZED
    # now reactivating the main vbo and shader program


    reactivateMainObj(mainState.mainForDisplayObjects.shader_program, mainState.mainForDisplayObjects.vbo, mainState.calcDimsStruct)
end#setUpWordsDisplay


"""
adding the data about 3 dimensional arrays that will be source of data used for scrolling behaviour
onScroll Data - list of tuples where first is the name of the texture that we provided and second is associated data (3 dimensional array of appropriate type)

"""
function setUpForScrollData(onScrollData::FullScrollableDat, mainState::StateDataFields)



    onScrollData.slicesNumber = getSlicesNumber(onScrollData)
    mainState.onScrollData = onScrollData
    #In order to refresh all in case we would change the texture dimensions ...
    ChangePlane.processKeysInfo(Option(onScrollData.dataToScrollDims), mainState, KeyboardStruct())
    #so  It will precalculate some data and later mouse modification will be swift
    oldd = mainState.valueForMasToSet

    mainState.valueForMasToSet = valueForMasToSetStruct(value=0)
    ReactOnMouseClickAndDrag.reactToMouseDrag(MouseStruct(true, false, [CartesianIndex(5, 5)]), mainState)
    mainState.valueForMasToSet = oldd




end#setUpMainDisplay


"""
add data needed for proper calculations of mouse, verticies positions ... etc
"""
function setUpCalcDimsStruct(calcDim::CalcDimsStruct, mainState::StateDataFields)
    # @info calcDim
    mainState.calcDimsStruct = calcDim

end#setUpCalcDimsStruct



"""
sets value we are setting to the  active mask vie mause interaction, in case mask is modifiable
"""
function setUpvalueForMasToSet(valueForMasToSett::valueForMasToSetStruct, mainState::StateDataFields)

    mainState.valueForMasToSet = valueForMasToSett

    updateImagesDisplayed(mainState.currentlyDispDat, mainState.mainForDisplayObjects, mainState.textDispObj, mainState.calcDimsStruct, valueForMasToSett)

end#setUpvalueForMasToSet


"""
enables updating just a single slice that is displayed - do not change what will happen after scrolling
one need to pass data to actor in
struct that holds tuple where first entry is
-vector of tuples whee first entry in tuple is name of texture given in the setup and second is 2 dimensional aray of appropriate type with image data
- Int - second is Int64 - that is marking the screen number to which we wan to set the actor state
"""
function updateSingleImagesDisplayedSetUp(singleSliceDat::SingleSliceDat, mainState::StateDataFields)
    updateImagesDisplayed(singleSliceDat, mainState.mainForDisplayObjects, mainState.textDispObj, mainState.calcDimsStruct, mainState.valueForMasToSet)


    mainState.currentlyDispDat = singleSliceDat
    mainState.currentDisplayedSlice = singleSliceDat.sliceNumber
    mainState.isSliceChanged = true # mark for mouse interaction that we changed slice
end #updateSingleImagesDisplayed





# @spawn :interactive

"""
when GLFW context is ready we need to use this  function in order to register GLFW events to Rocket actor - we use subscription for this
    actor - Roctet actor that holds objects needed for display like window etc...
    return list of subscriptions so if we will need it we can unsubscribe
"""
function subscribeGLFWtoActor(window::GLFW.Window, mainMedEye3dObject::MainMedEye3d, calcDim::CalcDimsStruct)


    ReactToScroll.registerMouseScrollFunctions(window, mainMedEye3dObject.channel)
    GLFW.SetScrollCallback(window, (a, xoff, yoff) -> begin
        put!(mainMedEye3dObject.channel, Int64(yoff))
    end)

    ReactOnKeyboard.registerKeyboardFunctions(window, mainMedEye3dObject.channel)
    ReactOnMouseClickAndDrag.registerMouseClickFunctions(window, calcDim, mainMedEye3dObject.channel)

end






end #ReactToGLFWInpuut
