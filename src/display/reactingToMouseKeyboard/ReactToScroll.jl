
"""
module that holds functions needed to  react to scrolling
Generally first we need to pass the GLFW callback to the Rocket obeservable
code adapted from https://discourse.julialang.org/t/custom-subject-in-rocket-jl-for-mouse-events-from-glfw/65133/3
"""
module ReactToScroll
using ModernGL, GLFW, Logging
using ..DisplayWords, ..ForDisplayStructs, ..TextureManag, ..DataStructs, ..StructsManag, ..ShadersAndVerticiesForSupervoxels, ..ShadersAndVerticiesForLine, ..MakieEvents

export reactToScroll, reactToScrollZoom
export registerMouseScrollFunctions




"""
uploading data to given texture; of given types associated
returns subscription in order to enable unsubscribing in the end
window - GLFW window
return scrollback - that holds boolean subject (observable) to which we can react by subscribing appropriate actor
"""
function registerMouseScrollFunctions(window::GLFW.Window, mainChannel::Base.Channel{Any})
    GLFW.SetScrollCallback(window, (a, xoff, yoff) -> begin
        if GLFW.GetKey(window, GLFW.KEY_LEFT_SHIFT) == GLFW.PRESS || GLFW.GetKey(window, GLFW.KEY_RIGHT_SHIFT) == GLFW.PRESS
            put!(mainChannel, ScrollZoomEvent(Float64(yoff)))
        else
            put!(mainChannel, Int64(yoff))
        end
    end)
end #registerMouseScrollFunctions

function reactToScrollZoom(data::ScrollZoomEvent, mainStates::Vector{StateDataFields})
    mainState = mainStates[mainStates[1].switchIndex]
    
    # Zoom factor: 1.1 for each scroll tick
    zoomFactor = data.zoom_delta > 0 ? 1.1f0 : (1.0f0 / 1.1f0)
    
    # Increase zoom, clamp between 1.0 (no zoom) and 20.0
    newZoom = clamp(mainState.calcDimsStruct.zoom * zoomFactor, 1.0f0, 20.0f0)
    
    # If zoom hits 1.0, reset panning as well to keep it clean
    if newZoom == 1.0f0
        mainState.calcDimsStruct.panX = 0.0f0
        mainState.calcDimsStruct.panY = 0.0f0
    end
    
    mainState.calcDimsStruct.zoom = newZoom
    @info "Zoom: $(newZoom)x (panel=$(mainStates[1].switchIndex))"
    
    # Re-render the current slice — reactToScroll with 0 scrollNumb will
    # re-extract the same slice and call updateImagesDisplayed, which 
    # applies applyZoomPan internally.
    reactToScroll(0, mainStates, false)
end



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
        # Inside the reactToScroll function, find this section:
if mainState.displayMode == SingleImage && !isempty(mainState.allSupervoxels)
    # Change this line:
    # current = mainState.lastRecordedMousePosition[toScrollDat.dimensionToScroll]
    ShadersAndVerticiesForSupervoxels.renderSupervoxelLines(mainState.mainForDisplayObjects, mainState.supervoxelFields, mainState.mainRectFields,
    mainState.allSupervoxels, mainState.onScrollData.dimensionToScroll, current)
end
        # if mainState.displayMode == SingleImage && !isempty(mainState.allSupervoxels)
        #     current_slice_sv = getSvCurrentSlice(mainState.allSupervoxels, current)
        #     ShadersAndVerticiesForSupervoxels.renderSupervoxelLines(mainState.mainForDisplayObjects, mainState.supervoxelFields, mainState.mainRectFields, current_slice_sv)
        # end

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
        
        # Multiview synchronization
        if length(mainStates) > 1 && mainState.mainForDisplayObjects.isSyncScrollOn
            if length(mainStates) >= 4
                clickedPanel = mainStates[1].switchIndex
                
                # Use the current crosshair position (where the user last clicked or synced)
                activePos = mainState.lastRecordedMousePosition
                if clickedPanel == 1 || clickedPanel == 2 # Axial
                    origX, origY = activePos[1], activePos[2]
                    origZ = current
                elseif clickedPanel == 3  # Sagittal (permuted 2,3,1)
                    origY, origZ = activePos[1], activePos[2]
                    origX = current
                else # Bottom-Right (4) (Coronal) (permuted 1,3,2)
                    origX, origZ = activePos[1], activePos[2]
                    origY = current
                end
                
                # Panel 1, 2, 5 scroll Z (origZ), Panel 3 scrolls origX, Panel 4 scrolls origY
                targets = [(1, origZ), (2, origZ), (3, origX), (4, origY)]
                if length(mainStates) >= 5
                    push!(targets, (5, origZ))
                end
                
                # Update lastRecordedMousePosition for all panels to ensure crosshairs synchronize!
                for i in 1:length(mainStates)
                    if i == 1 || i == 2 || i == 5
                        mainStates[i].lastRecordedMousePosition = CartesianIndex(origX, origY, origZ)
                    elseif i == 3
                        mainStates[i].lastRecordedMousePosition = CartesianIndex(origY, origZ, origX)
                    else
                        mainStates[i].lastRecordedMousePosition = CartesianIndex(origX, origZ, origY)
                    end
                end
                
                for (panelIdx, targetSlice) in targets
                    if panelIdx != clickedPanel
                        panelState = mainStates[panelIdx]
                        currSlice = panelState.currentDisplayedSlice
                        
                        if targetSlice != currSlice || panelState.currentlyDispDat === nothing
                            lastSlice = panelState.onScrollData.slicesNumber
                            newSlice = clamp(targetSlice, 1, lastSlice)
                            
                            singleSlDatSync = panelState.onScrollData.dataToScroll |>
                                (scrDat) -> map(threeDimDat -> threeToTwoDimm(threeDimDat.type, Int64(newSlice), panelState.onScrollData.dimensionToScroll, threeDimDat), scrDat) |>
                                (twoDimList) -> SingleSliceDat(listOfDataAndImageNames=twoDimList, sliceNumber=newSlice, textToDisp=getTextForCurrentSlice(panelState.onScrollData, Int32(newSlice)))
                            
                            panelState.currentlyDispDat = singleSlDatSync
                            panelState.currentDisplayedSlice = newSlice
                            panelState.isSliceChanged = true
                            
                            for updateDat in singleSlDatSync.listOfDataAndImageNames
                                findList = findall((texSpec) -> texSpec.name == updateDat.name, panelState.mainForDisplayObjects.listOfTextSpecifications)
                                if !isempty(findList)
                                    texSpec = panelState.mainForDisplayObjects.listOfTextSpecifications[findList[1]]
                                    transformedDat = applyZoomPan(updateDat.dat, panelState.calcDimsStruct.zoom, panelState.calcDimsStruct.panX, panelState.calcDimsStruct.panY)
                                    updateTexture(updateDat.type, transformedDat, texSpec, 0, 0, panelState.calcDimsStruct.imageTextureWidth, panelState.calcDimsStruct.imageTextureHeight)
                                end
                            end
                        end
                        
                        # UPDATE lastRecordedMousePosition for the synced panels
                        if panelIdx == 1 || panelIdx == 2
                            panelState.lastRecordedMousePosition = CartesianIndex(origX, origY, origZ)
                        elseif panelIdx == 3
                            panelState.lastRecordedMousePosition = CartesianIndex(origY, origZ, origX)
                        else
                            panelState.lastRecordedMousePosition = CartesianIndex(origX, origZ, origY)
                        end
                    end
                end
                

            else
                for panelIdx in 1:length(mainStates)
                    if panelIdx != mainStates[1].switchIndex
                        panelState = mainStates[panelIdx]
                        targetDim = panelState.onScrollData.dataToScrollDims.dimensionToScroll
                        targetSlice = locArr[targetDim]
                        
                        if targetSlice != panelState.currentDisplayedSlice || panelState.currentlyDispDat === nothing
                            # clamp
                            lastSlice = panelState.onScrollData.slicesNumber
                            newSlice = clamp(targetSlice, 1, lastSlice)
                            
                            # update texture data (singleSlDat logic duplicated manually)
                            singleSlDatSync = panelState.onScrollData.dataToScroll |>
                                (scrDat) -> map(threeDimDat -> threeToTwoDimm(threeDimDat.type, Int64(newSlice), panelState.onScrollData.dimensionToScroll, threeDimDat), scrDat) |>
                                (twoDimList) -> SingleSliceDat(listOfDataAndImageNames=twoDimList, sliceNumber=newSlice, textToDisp=getTextForCurrentSlice(panelState.onScrollData, Int32(newSlice)))
                            
                            panelState.currentlyDispDat = singleSlDatSync
                            panelState.currentDisplayedSlice = newSlice
                            panelState.isSliceChanged = true
                            
                            # upload textures to GPU without SwapBuffers
                            for updateDat in singleSlDatSync.listOfDataAndImageNames
                                findList = findall((texSpec) -> texSpec.name == updateDat.name, panelState.mainForDisplayObjects.listOfTextSpecifications)
                                if !isempty(findList)
                                    texSpec = panelState.mainForDisplayObjects.listOfTextSpecifications[findList[1]]
                                    transformedDat = applyZoomPan(updateDat.dat, panelState.calcDimsStruct.zoom, panelState.calcDimsStruct.panX, panelState.calcDimsStruct.panY)
                                    updateTexture(updateDat.type, transformedDat, texSpec, 0, 0, panelState.calcDimsStruct.imageTextureWidth, panelState.calcDimsStruct.imageTextureHeight)
                                end
                            end
                            
                            # also update panelState.lastRecordedMousePosition to keep them in sync
                            panelState.lastRecordedMousePosition = CartesianIndex(locArr[1], locArr[2], locArr[3])
                        end
                    end
                end
            end
        end

    end#if

end#reactToScroll
function getSvCurrentSlice(all_supervoxels::Dict{Int, Dict{Int, Dict{String,Any}}}, slice_number, mainState=nothing)
    # Get the current axis
    current_axis = mainState !== nothing ? mainState.onScrollData.dimensionToScroll : 3

    # Check if we have supervoxels for this axis
    if !haskey(all_supervoxels, current_axis)
        return Dict{String,Any}(
            "supervoxel_vertices" => Float32[],
            "supervoxel_indices" => UInt32[],
            "slice_position" => Float64(slice_number)
        )
    end

    # Get the supervoxels for the current axis
    axis_supervoxels = all_supervoxels[current_axis]

    # Find the closest slice
    slice_positions = [sv["slice_position"] for (_, sv) in axis_supervoxels]
    if isempty(slice_positions)
        return Dict{String,Any}(
            "supervoxel_vertices" => Float32[],
            "supervoxel_indices" => UInt32[],
            "slice_position" => Float64(slice_number)
        )
    end

    closest_slice_key = argmin(abs.(collect(keys(axis_supervoxels)) .- slice_number))
    slice_key = collect(keys(axis_supervoxels))[closest_slice_key]

    return axis_supervoxels[slice_key]
end

# function getSvCurrentSlice(all_supervoxels::Dict{Int, Dict{String,Any}}, slice_number)
#     if haskey(all_supervoxels, slice_number)
#         return all_supervoxels[slice_number]
#     else
#         return Dict{String,Any}(
#             "supervoxel_vertices" => Float32[],
#             "supervoxel_indices" => UInt32[],
#             "slice_position" => Float64(slice_number)
#             )

#     end
# end

end #ReactToScroll
