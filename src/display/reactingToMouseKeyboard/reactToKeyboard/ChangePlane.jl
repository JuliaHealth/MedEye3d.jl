"""
controls changing plane for example from transverse to saggital ...
"""
module ChangePlane
using GLFW, Dictionaries, Parameters, DataTypesBasic, Setfield
using ..StructsManag, ..DataStructs, ..ForDisplayStructs, ..ReactToScroll

"""
In case we want to change the dimansion of scrolling so for example from transverse to coronal ...
    toBeSavedForBack - just marks weather we wat to save the info how to undo latest action
    - false if we invoke it from undoing
"""

function processKeysInfo(toScrollDatPrim::Identity{DataToScrollDims}, stateObject::StateDataFields, keyInfo::KeyboardStruct, toBeSavedForBack::Bool=true)

    toScrollDat = toScrollDatPrim.value

    old = stateObject.onScrollData.dimensionToScroll

    ratioSetcalcDim = getHeightToWidthRatio(stateObject.calcDimsStruct, toScrollDat)
    newCalcDim = getMainVerticies(ratioSetcalcDim, stateObject.displayMode, stateObject.imagePosition)

    stateObject.calcDimsStruct = newCalcDim

    # Update scroll dimension and slice metadata
    stateObject.onScrollData.dimensionToScroll = toScrollDat.dimensionToScroll
    stateObject.onScrollData.dataToScrollDims = toScrollDat
    stateObject.onScrollData.slicesNumber = getSlicesNumber(stateObject.onScrollData)

    # Get the slice of interest based on last recorded mouse position or middle slice
    mousePos = stateObject.lastRecordedMousePosition
    current = if mousePos == CartesianIndex(1, 1, 1) && stateObject.currentDisplayedSlice > 1
        stateObject.currentDisplayedSlice
    elseif mousePos == CartesianIndex(1, 1, 1)
        max(1, stateObject.onScrollData.slicesNumber ÷ 2)
    else
        clamp(mousePos[toScrollDat.dimensionToScroll], 1, stateObject.onScrollData.slicesNumber)
    end

    # Generate 2D slice data for display
    singleSlDat = stateObject.onScrollData.dataToScroll |>
                  (scrDat) -> map(threeDimDat -> threeToTwoDimm(threeDimDat.type, Int64(current), toScrollDat.dimensionToScroll, threeDimDat), scrDat) |>
                              (twoDimList) -> SingleSliceDat(listOfDataAndImageNames=twoDimList, sliceNumber=current)

    stateObject.currentlyDispDat = singleSlDat

    # Save information about current slice for future reference
    stateObject.currentDisplayedSlice = current

end#processKeysInfo
end#ChangePlane
