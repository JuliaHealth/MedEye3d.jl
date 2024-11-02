
"""
module that holds functions needed to  react to scrolling
Generally first we need to pass the GLFW callback to the Rocket obeservable
code adapted from https://discourse.julialang.org/t/custom-subject-in-rocket-jl-for-mouse-events-from-glfw/65133/3
"""
module ReactToScroll
using ModernGL, GLFW, Logging
using ..DisplayWords, ..ForDisplayStructs, ..TextureManag, ..DataStructs, ..StructsManag, ..ShadersAndVerticiesForSupervoxels

export reactToScroll
export registerMouseScrollFunctions




"""
uploading data to given texture; of given types associated
returns subscription in order to enable unsubscribing in the end
window - GLFW window
return scrollback - that holds boolean subject (observable) to which we can react by subscribing appropriate actor
"""
function registerMouseScrollFunctions(window::GLFW.Window, mainChannel::Base.Channel{Any})
    GLFW.SetScrollCallback(window, (a, xoff, yoff) -> begin
        put!(mainChannel, Int64(yoff)) #if there is type distortion in the channel, we can implement custom struct types
    end)

end #registerMouseScrollFunctions


"""
in case of the scroll p true will be send in case of down - false
in response to it it sets new screen int variable and changes displayed screen
toBeSavedForBack - just marks weather we wat to save the info how to undo latest action
 - false if we invoke it from undoing
"""
function reactToScroll(scrollNumb::Int64, mainStates::Vector{StateDataFields}, toBeSavedForBack::Bool=true)
    mainState = mainStates[mainStates[1].switchIndex] #getting information from the first state

    current = mainState.currentDisplayedSlice
    old = current
    #when shift is pressed scrolling is 10 times faster

    if (!mainState.mainForDisplayObjects.isFastScroll)
        current += scrollNumb
    else
        current += scrollNumb * 10
    end


    #isScrollUp ? current+=1 : current-=1

    # we do not want to move outside of possible range of slices
    lastSlice = mainState.onScrollData.slicesNumber
    if (lastSlice > 1)

        mainState.isSliceChanged = true
        if (current < 1)
            current = 1
        end
        if (lastSlice < 1)
            lastSlice = 1
        end
        if (current >= lastSlice)
            current = lastSlice
        end
        #logic to change displayed screen
        #we select slice that we are intrested in
        singleSlDat = mainState.onScrollData.dataToScroll |>
                      (scrDat) -> map(threeDimDat -> threeToTwoDimm(threeDimDat.type, Int64(current), mainState.onScrollData.dimensionToScroll, threeDimDat), scrDat) |>
                                  (twoDimList) -> SingleSliceDat(listOfDataAndImageNames=twoDimList, sliceNumber=current, textToDisp=getTextForCurrentSlice(mainState.onScrollData, Int32(current)))


        updateImagesDisplayed(singleSlDat, mainState.mainForDisplayObjects, mainState.textDispObj, mainState.calcDimsStruct, mainState.valueForMasToSet, mainState.crosshairFields, mainState.mainRectFields, mainState.displayMode)




        """
        Added by me recently for testing
        Add a check here to only invoke this in singelImage display mode
        """

        if mainState.displayMode == SingleImage
            ShadersAndVerticiesForSupervoxels.renderSupervoxelLines(mainState.mainForDisplayObjects, mainState.supervoxelFields, mainState.mainRectFields, mainState.supervoxelVertAndInd)
        end
        """
        END
        """










        mainState.currentlyDispDat = singleSlDat
        # updating the last mouse position so when we will change plane it will better show actual position
        currentDim = Int64(mainState.onScrollData.dataToScrollDims.dimensionToScroll)
        lastMouse = mainState.lastRecordedMousePosition
        locArr = [lastMouse[1], lastMouse[2], lastMouse[3]]
        locArr[currentDim] = current
        mainState.lastRecordedMousePosition = CartesianIndex(locArr[1], locArr[2], locArr[3])
        #saving information about current slice for future reference
        mainState.currentDisplayedSlice = current
        #enable undoing the action
        # if (toBeSavedForBack)
        #     func = () -> reactToScroll(old -= scrollNumb, mainState, false)
        #     addToforUndoVector(mainState, func)
        # end

    end#if

end#reactToScroll





end #ReactToScroll
