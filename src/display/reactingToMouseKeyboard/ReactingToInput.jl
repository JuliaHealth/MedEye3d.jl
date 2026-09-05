
module ReactingToInput
using GLFW, Setfield, DataTypesBasic, Base.Threads
using ..ReactToScroll, ..ForDisplayStructs, ..ReactOnKeyboard
using ..TextureManag, ..ReactOnMouseClickAndDrag, ..ReactOnKeyboard, ..DataStructs, ..StructsManag, ..DisplayWords
using ..KeyboardVisibility, ..OtherKeyboardActions, ..WindowControll, ..ChangePlane, ..MakieEvents
export subscribeGLFWtoActor, setUpForScrollData, setUpCalcDimsStruct, setUpWordsDisplay, setUpMainDisplay, setUpvalueForMasToSet, updateSingleImagesDisplayedSetUp


"""
adding the data into about display context to enable proper display of main image and masks
"""
function setUpMainDisplay(mainForDisplayObjects::forDisplayObjects, mainStates::Vector{StateDataFields})
    mainState = mainStates[mainStates[1].switchIndex]
    mainState.mainForDisplayObjects = mainForDisplayObjects
end#setUpMainDisplay

"""
adding the data needed for text display — no-op, text rendering removed for Vulkan backend.
"""
function setUpWordsDisplay(textDispObject::ForWordsDispStruct, mainStates::Vector{StateDataFields})
    # No-op: text rendering removed
end#setUpWordsDisplay


"""
adding the data about 3 dimensional arrays that will be source of data used for scrolling behaviour
onScroll Data - list of tuples where first is the name of the texture that we provided and second is associated data (3 dimensional array of appropriate type)

"""
function setUpForScrollData(onScrollData::FullScrollableDat, mainStates::Vector{StateDataFields})

    mainState = mainStates[mainStates[1].switchIndex]


    onScrollData.slicesNumber = getSlicesNumber(onScrollData)
    mainState.onScrollData = onScrollData
    mainState.currentDisplayedSlice = max(1, onScrollData.slicesNumber ÷ 2)
    #In order to refresh all in case we would change the texture dimensions ...
    ChangePlane.processKeysInfo(Option(onScrollData.dataToScrollDims), mainState, KeyboardStruct())
    #so  It will precalculate some data and later mouse modification will be swift
    oldd = mainState.valueForMasToSet

    mainState.valueForMasToSet = valueForMasToSetStruct(value=0)
    ReactOnMouseClickAndDrag.reactToMouseDrag(MouseStruct(isLeftButtonDown=true, isRightButtonDown=false, lastCoordinates=[CartesianIndex(5, 5)]), mainStates)
    mainState.valueForMasToSet = oldd




end#setUpMainDisplay


"""
add data needed for proper calculations of mouse, verticies positions ... etc
"""
function setUpCalcDimsStruct(calcDim::CalcDimsStruct, mainStates::Vector{StateDataFields})
    # @info calcDim
    mainState = mainStates[mainStates[1].switchIndex]
    mainState.calcDimsStruct = calcDim

end#setUpCalcDimsStruct



"""
sets value we are setting to the  active mask vie mause interaction, in case mask is modifiable
"""
function setUpvalueForMasToSet(valueForMasToSett::valueForMasToSetStruct, mainStates::Vector{StateDataFields})
    mainState = mainStates[mainStates[1].switchIndex]

    mainState.valueForMasToSet = valueForMasToSett

    updateImagesDisplayed(mainState.currentlyDispDat, mainState.mainForDisplayObjects, mainState.textDispObj, mainState.calcDimsStruct, valueForMasToSett, mainState.crosshairFields, mainState.mainRectFields, mainState.displayMode)

end#setUpvalueForMasToSet


"""
enables updating just a single slice that is displayed - do not change what will happen after scrolling
one need to pass data to actor in
struct that holds tuple where first entry is
-vector of tuples whee first entry in tuple is name of texture given in the setup and second is 2 dimensional aray of appropriate type with image data
- Int - second is Int64 - that is marking the screen number to which we wan to set the actor state
"""
function updateSingleImagesDisplayedSetUp(singleSliceDat::SingleSliceDat, mainStates::Vector{StateDataFields})
    mainState = mainStates[mainStates[1].switchIndex]
    updateImagesDisplayed(singleSliceDat, mainState.mainForDisplayObjects, mainState.textDispObj, mainState.calcDimsStruct, mainState.valueForMasToSet, mainState.crosshairFields, mainState.mainRectFields, mainState.displayMode)


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

    # registerMouseScrollFunctions sets the scroll callback WITH shift-zoom detection.
    # Do NOT override it with a plain Int64-only callback afterward!
    ReactToScroll.registerMouseScrollFunctions(window, mainMedEye3dObject.channel)

    ReactOnKeyboard.registerKeyboardFunctions(window, mainMedEye3dObject.channel)
    ReactOnMouseClickAndDrag.registerMouseClickFunctions(window, calcDim, mainMedEye3dObject.channel)

    # Window management events routed directly through the channel
    GLFW.SetWindowCloseCallback(window, (_) -> put!(mainMedEye3dObject.channel, CloseWindowEvent()))
    GLFW.SetFramebufferSizeCallback(window, (win, fb_w, fb_h) -> begin
        # GLFW cursor coords are in WINDOW space, not framebuffer space.
        # Use window size for CalcDimsStruct (quad vertices / coordinate mapping),
        # framebuffer size for Vulkan swapchain (actual GPU pixel dimensions).
        win_w, win_h = GLFW.GetWindowSize(win)
        put!(mainMedEye3dObject.channel, ResizeWindowEvent(Int(win_w), Int(win_h), Int(fb_w), Int(fb_h)))
    end)
end






end #ReactToGLFWInpuut
