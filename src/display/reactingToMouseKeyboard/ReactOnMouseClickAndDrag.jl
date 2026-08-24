

"""
module
code adapted from https://discourse.julialang.org/t/custom-subject-in-rocket-jl-for-mouse-events-from-glfw/65133/3
it is design to help processing data from
    -GLFW.SetCursorPosCallback(window, (_, x, y) -> println("cursor: x, y")) and  for example : cursor: 29.0, 469.0  types   Float64  Float64
    -GLFW.SetMouseButtonCallback(window, (_, button, action, mods) -> println("button action"))  for example types MOUSE_BUTTON_1 PRESS   GLFW.MouseButton  GLFW.Action
The main function is to mark the interaction of the mouse to be saved in appropriate mask and be rendered onto the screen
so we modify the data that is the basis of the mouse interaction mask  and we pass the data on so appropriate part of the texture would be modified to be displayed on screen

"""
module ReactOnMouseClickAndDrag
using Logging, Parameters, Setfield, GLFW, ModernGL, Dates, Parameters, Logging, Base.Threads
using ..ForDisplayStructs, ..TextureManag, ..OpenGLDisplayUtils
using ..DataStructs, ..StructsManag, ..ShadersAndVerticiesForLine, ..ReactToScroll, ..DisplayWords, ..StrokeRasterization
import Logging, Base.Threads
export registerMouseClickFunctions
export reactToMouseDrag
export react_to_draw
export reactToDoubleClick
export DoubleClickEvent

"""
Calculates OpenGl coordinate system values for
    left edge of the text area,
    middle point of the image region,
    range of values for the image
"""
#careful fraction of mainImage should not be grather than 1
function openGlSystemVals(fractionOfMainImage::Float32, windowWidth::Int)
    #Working in normal coordinate system 0 -1 for calculating mid point

    windowWidthLowerBound = 0
    windowWidthUpperBound = windowWidth
    openGlLowerBound = -1
    openGlUpperBound = 1

    textAreaBegin = fractionOfMainImage * windowWidth
    imageRange = windowWidthUpperBound - windowWidthLowerBound
    imageMidPoint = (imageRange * fractionOfMainImage) / 2

    #if ratio of mainIMage is 0.8 , gives us (0.8/1 *2 ) -1 = 0.6
    textBeginningOpenGl = ((textAreaBegin / imageRange) * 2) - 1
    #gives us -0.2
    imageMidPointOpenGl = (imageMidPoint / imageRange) * 2 - 1
    openGlImageRange = textBeginningOpenGl - openGlLowerBound

    return (textBeginningOpenGl, imageMidPointOpenGl, openGlImageRange)
end

"""
we pass coordinate of cursor only when isLeftButtonDown is true and we make it true
if left button is presed down - we make it true if the left button is pressed over image and false if mouse get out of the window or we get information about button release
imageWidth adn imageHeight are the dimensions of textures that we use to display
"""
# Module-level timestamp for double-click detection (avoids GLFW.GetTime which doesn't exist in Julia GLFW.jl)
const lastLeftClickTimestamp = Ref{Float64}(0.0)

function registerMouseClickFunctions(window::GLFW.Window, calcD::CalcDimsStruct, mainChannel::Base.Channel{Any})
    xmin = Int32(calcD.windowWidthCorr)
    xmax = Int32(calcD.avWindWidtForMain - calcD.windowWidthCorr)

    ymin = Int32(calcD.windowHeightCorr)
    ymax = Int32(calcD.avWindHeightForMain - calcD.windowHeightCorr)
    # calculating dimensions of quad becouse it do not occupy whole window, and we want to react only to those mouse positions that are on main image quad
    mouseStructInstance = MouseStruct()
    
    # Query actual GLFW window size (may differ from requested size due to WM resize)
    actualW, actualH = GLFW.GetWindowSize(window)
    mouseStructInstance.actualWindowWidth = Int(actualW)
    mouseStructInstance.actualWindowHeight = Int(actualH)
    @info "GLFW actual window size: $(actualW)x$(actualH) vs stored: $(calcD.windowWidth)x$(calcD.windowHeight)"

    GLFW.SetCursorPosCallback(window, (a, x, y) -> begin
        actualW, actualH = GLFW.GetWindowSize(window)
        mouseStructInstance.actualWindowWidth = Int(actualW)
        mouseStructInstance.actualWindowHeight = Int(actualH)
        
        if (x >= 0 && x <= actualW && y >= 0 && y <= actualH)
            point = CartesianIndex(Int(x), Int(y))
            mouseStructInstance.lastCoordinates = [point]
            # Snapshot into a new struct so later callbacks cannot overwrite this message
            put!(mainChannel, MouseStruct(
                isLeftButtonDown  = mouseStructInstance.isLeftButtonDown,
                isRightButtonDown = mouseStructInstance.isRightButtonDown,
                lastCoordinates   = [point],
                actualWindowWidth  = mouseStructInstance.actualWindowWidth,
                actualWindowHeight = mouseStructInstance.actualWindowHeight,
            ))
        end
    end)# and  for example : cursor: 29.0, 469.0  types   Float64  Float64
    GLFW.SetMouseButtonCallback(window, (a, button, action, mods) -> begin
        # Only update the flag for the button that actually changed
        if button == GLFW.MOUSE_BUTTON_1
            mouseStructInstance.isLeftButtonDown = (action == GLFW.PRESS)
        elseif button == GLFW.MOUSE_BUTTON_2
            mouseStructInstance.isRightButtonDown = (action == GLFW.PRESS)
        end
        
        leftMouseButtonDownResult = (button == GLFW.MOUSE_BUTTON_1 && action == GLFW.PRESS)

        # Double-click detection: fire a dedicated DoubleClickEvent into the channel.
        # Uses time() (Base Julia) — GLFW.GetTime() does not exist in Julia's GLFW.jl.
        if leftMouseButtonDownResult
            now = time()
            if (now - lastLeftClickTimestamp[]) < 0.35  # 350ms threshold
                # Fire DoubleClickEvent — its own dispatch type, not embedded in MouseStruct
                coords = mouseStructInstance.lastCoordinates
                put!(mainChannel, DoubleClickEvent(
                    x = isempty(coords) ? 0 : coords[1][1],
                    y = isempty(coords) ? 0 : coords[1][2],
                    actualWindowWidth  = mouseStructInstance.actualWindowWidth,
                    actualWindowHeight = mouseStructInstance.actualWindowHeight,
                ))
            end
            lastLeftClickTimestamp[] = now
        end

        # Snapshot regular mouse event (for right-click and position tracking)
        put!(mainChannel, MouseStruct(
            isLeftButtonDown  = mouseStructInstance.isLeftButtonDown,
            isRightButtonDown = mouseStructInstance.isRightButtonDown,
            lastCoordinates   = mouseStructInstance.lastCoordinates,
            actualWindowWidth  = mouseStructInstance.actualWindowWidth,
            actualWindowHeight = mouseStructInstance.actualWindowHeight,
        ))
    end) # for example types MOUSE_BUTTON_1 PRESS   GLFW.MouseButton  GLFW.Action

end #registerMouseScrollFunctions



mouseCoords_channel = Base.Channel{MouseStruct}(100)
# we can fetch! on the channel, what is the next thing line, if the mouseStruct, check previous one by fetch. If it mouseStruct, aggregate those 2 and fetch the next one
#fetch in while loop, until no more mouseStructs, then we have the last one, and we can react to it

# Double-click zoom state for QuadImage mode
mutable struct QuadZoomState
    isZoomed::Bool
    zoomedPanel::Int
    savedVerts::Vector{Vector{Float32}}
    savedVertSizes::Vector{Int64}
end
const quadZoomState = QuadZoomState(false, 0, Vector{Float32}[], Int64[])


"""
used when we want to save some manual modifications
"""
function react_to_draw(mouseStructArray::Vector{MouseStruct}, mainStates::Vector{StateDataFields})
    if isempty(mouseStructArray)
        return
    end

    # First, detect the active panel from the first sampled point
    first_mouse = mouseStructArray[1]
    if !isempty(first_mouse.lastCoordinates)
        x, y = first_mouse.lastCoordinates[1][1], first_mouse.lastCoordinates[1][2]
        viewportW = Float64(mainStates[1].calcDimsStruct.windowWidth)
        viewportH = Float64(mainStates[1].calcDimsStruct.windowHeight)
        actualW = first_mouse.actualWindowWidth > 0 ? Float64(first_mouse.actualWindowWidth) : viewportW
        actualH = first_mouse.actualWindowHeight > 0 ? Float64(first_mouse.actualWindowHeight) : viewportH
        
        is_compare = false
        if length(mainStates) >= 5
            botVerts = mainStates[3].calcDimsStruct.mainImageQuadVert
            if !isempty(botVerts) && all(v -> v == 0.0f0, botVerts[1:2])
                is_compare = true
            end
        end
        
        if is_compare
            mainStates[1].switchIndex = x < actualW / 2.0 ? 1 : 5
        elseif length(mainStates) >= 4
            if x < actualW / 2.0 && y < actualH / 2.0
                mainStates[1].switchIndex = 1
            elseif x >= actualW / 2.0 && y < actualH / 2.0
                mainStates[1].switchIndex = 2
            elseif x < actualW / 2.0 && y >= actualH / 2.0
                mainStates[1].switchIndex = 3
            else
                mainStates[1].switchIndex = 4
            end
        elseif length(mainStates) > 1
            textBeginning, midPoint, imageRange = openGlSystemVals(mainStates[1].calcDimsStruct.fractionOfMainIm, mainStates[1].calcDimsStruct.windowWidth)
            cursorXPosOpenGl = (x / mainStates[1].calcDimsStruct.windowWidth) * 2 - 1
            if cursorXPosOpenGl > midPoint
                mainStates[1].switchIndex = 2
            else
                mainStates[1].switchIndex = 1
            end
        end
    end

    stateObject = mainStates[mainStates[1].switchIndex]
    if !stateObject.valueForMasToSet.is_painting_active || isempty(stateObject.textureToModifyVec)
        return
    end
    texture = stateObject.textureToModifyVec[1]
    calcDim = stateObject.calcDimsStruct

    # Extract all sampled mouse coordinates in texture pixel space
    sampledPoints = Tuple{Int,Int}[]
    for mouseStruct in mouseStructArray
        mouseCoords = mouseStruct.lastCoordinates
        actualW = Float64(mouseStruct.actualWindowWidth)
        actualH = Float64(mouseStruct.actualWindowHeight)
        for c in mouseCoords
            texX, texY = StructsManag.getTextureCoordinatesFromScreen(c[1], c[2], calcDim, actualW, actualH)
            ix, iy = Int(round(texX)), Int(round(texY))
            if ix >= 1 && ix <= calcDim.imageTextureWidth && iy >= 1 && iy <= calcDim.imageTextureHeight
                push!(sampledPoints, (ix, iy))
            end
        end
    end

    if isempty(sampledPoints)
        return
    end

    # Build polyline: connect from last point of previous frame if on the same slice
    pointsToRasterize = Tuple{Int,Int}[]
    if !stateObject.isSliceChanged && !isempty(stateObject.lastPaintCoords)
        push!(pointsToRasterize, (stateObject.lastPaintCoords[1][1], stateObject.lastPaintCoords[1][2]))
    end
    append!(pointsToRasterize, sampledPoints)
    stateObject.isSliceChanged = false

    # Store last point for next frame
    stateObject.lastPaintCoords = [CartesianIndex(sampledPoints[end][1], sampledPoints[end][2])]

    # Access current slice data
    twoDimDat = stateObject.currentlyDispDat |>
                (singSl) -> singSl.listOfDataAndImageNames[singSl.nameIndexes[texture.name]]

    toSet = convert(twoDimDat.type, convert(parameter_type(texture), stateObject.valueForMasToSet.value))
    strokeW = Int(texture.strokeWidth)

    # In-place continuous thick-line interpolation using KernelAbstractions
    StrokeRasterization.rasterize_polyline!(twoDimDat.dat, pointsToRasterize, strokeW, toSet)

    singleSliceDat = setproperties(stateObject.currentlyDispDat, (listOfDataAndImageNames = [twoDimDat]))
    updateImagesDisplayed(singleSliceDat, stateObject.mainForDisplayObjects, stateObject.textDispObj, stateObject.calcDimsStruct, stateObject.valueForMasToSet, stateObject.crosshairFields, stateObject.mainRectFields, stateObject.displayMode)
end#react_to_draw

"""
we use mouse coordinate to modify the texture that is currently active for modifications
    - we take information about texture currently active for modifications from variables stored in actor
    from texture specification we take also its id and its properties ...
"""
function reactToMouseDrag(mousestr::MouseStruct, mainStates::Vector{StateDataFields})
    mainState = mainStates[1] #first State for holding switch index information
    # obj = mainState.mainForDisplayObjects
    # textureList = mainState.textureToModifyVec
    mouseCoords = mousestr.lastCoordinates
    
    # 1. Update switchIndex based on mouse position — needs coords
    if !isempty(mouseCoords)
        if length(mainStates) >= 4 # QuadImage mode
            if quadZoomState.isZoomed
                # When zoomed, always target the zoomed panel
                mainState.switchIndex = quadZoomState.zoomedPanel
            else
                # viewportW/H = requested window size used in glViewport (defines NDC→pixel mapping)
                viewportW = Float64(mainStates[1].calcDimsStruct.windowWidth)
                viewportH = Float64(mainStates[1].calcDimsStruct.windowHeight)
                # actualW/H = GLFW content area (may be smaller if WM resized window to fit screen)
                actualW = mousestr.actualWindowWidth > 0 ? Float64(mousestr.actualWindowWidth) : viewportW
                actualH = mousestr.actualWindowHeight > 0 ? Float64(mousestr.actualWindowHeight) : viewportH
                x, y = (mouseCoords[1][1], mouseCoords[1][2])
                
                # Detect compare mode: panels 3+4 are hidden (vertices zeroed out)
                # In compare mode, left half = panel 1, right half = panel 5
                is_compare = false
                if length(mainStates) >= 5
                    botVerts = mainStates[3].calcDimsStruct.mainImageQuadVert
                    if !isempty(botVerts) && all(v -> v == 0.0f0, botVerts[1:2])  # first X,Y coords are 0 = hidden
                        is_compare = true
                    end
                end
                
                if is_compare
                    # Compare mode: left = panel 1, right = panel 5
                    mainState.switchIndex = x < actualW / 2.0 ? 1 : 5
                else
                    # Quad view mode: standard 4-panel layout
                    if x < actualW / 2.0 && y < actualH / 2.0
                        mainStates[1].switchIndex = 1  # Top-Left (Axial CT/PET)
                    elseif x >= actualW / 2.0 && y < actualH / 2.0
                        mainStates[1].switchIndex = 2  # Top-Right (Pure PET Axial)
                    elseif x < actualW / 2.0 && y >= actualH / 2.0
                        mainStates[1].switchIndex = 3  # Bottom-Left (Sagittal)
                    else
                        mainStates[1].switchIndex = 4  # Bottom-Right (Coronal)
                    end
                end
            end
        elseif length(mainStates) > 1
            textBeginning, midPoint, imageRange = openGlSystemVals(mainState.calcDimsStruct.fractionOfMainIm, mainState.calcDimsStruct.windowWidth)
            cursorXPosOpenGl = (mouseCoords[1][1] / mainState.calcDimsStruct.windowWidth) * 2 - 1
            if cursorXPosOpenGl > midPoint
                mainState.switchIndex = 2
            elseif cursorXPosOpenGl < midPoint
                mainState.switchIndex = 1
            end
        end
    end # end !isempty(mouseCoords) for panel detection
    
    # If the right mouse button is released, clear the pan drag state
    if !mousestr.isRightButtonDown
        for state in mainStates
            empty!(state.lastPanDragCoords)
        end
    end

    # If the left mouse button is released, clear the paint stroke tail
    if !mousestr.isLeftButtonDown
        for state in mainStates
            empty!(state.lastPaintCoords)
        end
    end

    # 2. Right-click cross-plane jumping or panning
    if !isempty(mouseCoords) && mousestr.isRightButtonDown && length(mainStates) >= 4 # QuadImage mode
        viewportW = Float64(mainStates[1].calcDimsStruct.windowWidth)
        viewportH = Float64(mainStates[1].calcDimsStruct.windowHeight)
        actualW = mousestr.actualWindowWidth > 0 ? Float64(mousestr.actualWindowWidth) : viewportW
        actualH = mousestr.actualWindowHeight > 0 ? Float64(mousestr.actualWindowHeight) : viewportH
        x, y = (mouseCoords[1][1], mouseCoords[1][2])
        clickedPanel = mainState.switchIndex
        
        panelState = mainStates[clickedPanel]
        
        if isempty(panelState.lastPanDragCoords)
            # Read actual rendered vertex positions from the panel's calcDimsStruct
            texX, texY = getTextureCoordinatesFromScreen(x, y, panelState.calcDimsStruct, actualW, actualH)
            
            if panelState.moveLesionMode
                target_id = 0
                for ts in panelState.mainForDisplayObjects.listOfTextSpecifications
                    if (ts.isMultiDiscreteMask || ts.name == "Mask" || ts.name == "manualModif") && !isempty(ts.minAndMaxValue) && ts.minAndMaxValue[1] > 0
                        target_id = Int(round(ts.minAndMaxValue[1]))
                        break
                    end
                end
                if target_id <= 0
                    target_id = panelState.valueForMasToSet.value
                end
                if target_id <= 0
                    for ts in mainStates[1].mainForDisplayObjects.listOfTextSpecifications
                        if (ts.isMultiDiscreteMask || ts.name == "Mask" || ts.name == "manualModif") && !isempty(ts.minAndMaxValue) && ts.minAndMaxValue[1] > 0
                            target_id = Int(round(ts.minAndMaxValue[1]))
                            break
                        end
                    end
                end
                if target_id <= 0
                    target_id = 1
                end
                
                panelState.movingLesionID = target_id
                panelState.movingLesionStartTex = (texX, texY)
                panelState.movingLesionLastDelta = CartesianIndex(0,0,0)
                seg_vol = nothing
                # In compare mode, search the right panel's data for the lesion
                MEH = parentmodule(parentmodule(@__MODULE__)).SegmentationDisplay.MakieEventHandlers
                search_panel = (MEH.compare_mode[] && clickedPanel == 5) ? mainStates[5] : mainStates[1]
                for dat in search_panel.onScrollData.dataToScroll
                    if dat.name == "Mask" || dat.name == "segmentation"
                        if panelState.movingLesionID > 0 && any(dat.dat .== Float32(panelState.movingLesionID))
                            seg_vol = dat.dat
                            break
                        elseif seg_vol === nothing
                            seg_vol = dat.dat
                        end
                    elseif dat.name == "manualModif" && seg_vol === nothing
                        seg_vol = dat.dat
                    end
                end
                
                if seg_vol !== nothing && panelState.movingLesionID > 0
                    panelState.movingLesionOriginalCoords = findall(seg_vol .== Float32(panelState.movingLesionID))
                    panelState.movingLesionOriginalBGs = zeros(Float32, length(panelState.movingLesionOriginalCoords))
                    @info "Move Lesion START: lesion=$(panelState.movingLesionID) found $(length(panelState.movingLesionOriginalCoords)) voxels"
                else
                    panelState.movingLesionOriginalCoords = CartesianIndex{3}[]
                    panelState.movingLesionOriginalBGs = Float32[]
                    @warn "Move Lesion START: No voxels found for lesion $(panelState.movingLesionID)"
                end
                panelState.lastPanDragCoords = [CartesianIndex(Int(round(x)), Int(round(y)))]
                return
            end
            
            # Initial press: Do the jump!
            panelState.lastPanDragCoords = [CartesianIndex(Int(round(x)), Int(round(y)))]
            
            @info "RIGHT-CLICK: panel=$clickedPanel windowXY=($x,$y) viewport=$(Int(viewportW))x$(Int(viewportH)) actual=$(Int(actualW))x$(Int(actualH))"
            @info "  texX=$texX texY=$texY"
            
            currentSlice = panelState.currentDisplayedSlice
            
            if clickedPanel == 1 || clickedPanel == 2 || clickedPanel == 5
                origX, origY, origZ = texX, texY, currentSlice
            elseif clickedPanel == 3  # Sagittal
                origY, origZ, origX = texX, texY, currentSlice
            else # Bottom-Right (4) (Coronal)
                origX, origZ, origY = texX, texY, currentSlice
            end
            
            # Ensure lastRecordedMousePosition is updated for ALL panels so scroll sync knows the intersection!
            for i in 1:length(mainStates)
                if i == 1 || i == 2 || i == 5
                    mainStates[i].lastRecordedMousePosition = CartesianIndex(origX, origY, origZ)
                elseif i == 3
                    mainStates[i].lastRecordedMousePosition = CartesianIndex(origY, origZ, origX)
                else
                    mainStates[i].lastRecordedMousePosition = CartesianIndex(origX, origZ, origY)
                end
            end
            
            @info "  origX=$origX origY=$origY origZ=$origZ currentSlice=$currentSlice"
            @info "  Axial scrolls Z(1-$(mainStates[1].onScrollData.slicesNumber)) Sag scrolls origX(1-$(mainStates[3].onScrollData.slicesNumber)) Cor scrolls origY(1-$(mainStates[4].onScrollData.slicesNumber))"
            
            # Jump other panels to the corresponding slices
            # Panel 1, 2 & 5 scroll through Z (origZ), Panel 3 scrolls through origX, Panel 4 scrolls through origY
            targets = [(1, origZ), (2, origZ), (3, origX), (4, origY)]
            if length(mainStates) >= 5
                push!(targets, (5, origZ))
            end
            
            for (p_idx, targetSlice) in targets
                if p_idx != clickedPanel && p_idx <= length(mainStates)
                    otherState = mainStates[p_idx]
                    
                    # Read max slices for clamp
                    lastSlice = otherState.onScrollData.slicesNumber
                    newSlice = clamp(targetSlice, 1, lastSlice)
                    
                    # Always synchronize data so that crosshairs stay locked
                    singleSlDat = otherState.onScrollData.dataToScroll |>
                        (scrDat) -> map(threeDimDat -> threeToTwoDimm(threeDimDat.type, Int64(newSlice), otherState.onScrollData.dimensionToScroll, threeDimDat), scrDat) |>
                        (twoDimList) -> SingleSliceDat(listOfDataAndImageNames=twoDimList, sliceNumber=newSlice, textToDisp=getTextForCurrentSlice(otherState.onScrollData, Int32(newSlice)))
                    
                    # Upload new texture data to GPU (without rendering/SwapBuffers)
                    for updateDat in singleSlDat.listOfDataAndImageNames
                        findList = findall((texSpec) -> texSpec.name == updateDat.name, otherState.mainForDisplayObjects.listOfTextSpecifications)
                        if !isempty(findList)
                            texSpec = otherState.mainForDisplayObjects.listOfTextSpecifications[findList[1]]
                            transformedDat = applyZoomPan(updateDat.dat, otherState.calcDimsStruct.zoom, otherState.calcDimsStruct.panX, otherState.calcDimsStruct.panY)
                            updateTexture(updateDat.type, transformedDat, texSpec, 0, 0, otherState.calcDimsStruct.imageTextureWidth, otherState.calcDimsStruct.imageTextureHeight)
                        end
                    end
                    
                    # Update state (slice number and display data)
                    otherState.currentlyDispDat = singleSlDat
                    otherState.currentDisplayedSlice = newSlice
                    otherState.isSliceChanged = true
                end
            end
        else
            texX, texY = getTextureCoordinatesFromScreen(x, y, panelState.calcDimsStruct, actualW, actualH)
            
            if panelState.moveLesionMode
                if !isempty(panelState.movingLesionOriginalCoords) && panelState.movingLesionID > 0
                    startTexX, startTexY = panelState.movingLesionStartTex
                    dx_tex = round(Int, texX - startTexX)
                    dy_tex = round(Int, texY - startTexY)
                    
                    dx_vox, dy_vox, dz_vox = 0, 0, 0
                    if clickedPanel == 1 || clickedPanel == 2 || clickedPanel == 5
                        dx_vox, dy_vox = dx_tex, dy_tex
                    elseif clickedPanel == 3 # Sagittal (2,3,1) -> x is slice, y is texX, z is texY
                        dy_vox, dz_vox = dx_tex, texY - startTexY # keeping logic aligned
                        # Simplified delta mapping for sagittal
                        dx_vox, dy_vox, dz_vox = 0, dx_tex, dy_tex
                    else # Coronal (1,3,2) -> y is slice, x is texX, z is texY
                        dx_vox, dy_vox, dz_vox = dx_tex, 0, dy_tex
                    end
                    
                    new_delta = CartesianIndex(dx_vox, dy_vox, dz_vox)
                    
                    if new_delta != panelState.movingLesionLastDelta
                        target_id = panelState.movingLesionID
                        orig_coords = panelState.movingLesionOriginalCoords
                        orig_bgs = panelState.movingLesionOriginalBGs
                        old_d = panelState.movingLesionLastDelta
                        
                        dx, dy, dz = new_delta[1], new_delta[2], new_delta[3]
                        old_dx, old_dy, old_dz = old_d[1], old_d[2], old_d[3]
                        
                        MEH = parentmodule(parentmodule(@__MODULE__)).SegmentationDisplay.MakieEventHandlers
                        panels_to_update = (MEH.compare_mode[] && clickedPanel == 5) ? [5] : collect(1:length(mainStates))
                        
                        for p_idx in panels_to_update
                            st = mainStates[p_idx]
                            p_delta = (p_idx == 3) ? CartesianIndex(dy, dz, dx) : ((p_idx == 4) ? CartesianIndex(dx, dz, dy) : CartesianIndex(dx, dy, dz))
                            p_old_d = (p_idx == 3) ? CartesianIndex(old_dy, old_dz, old_dx) : ((p_idx == 4) ? CartesianIndex(old_dx, old_dz, old_dy) : CartesianIndex(old_dx, old_dy, old_dz))
                            
                            p_orig_coords = map(orig_coords) do c
                                (p_idx == 3) ? CartesianIndex(c[2], c[3], c[1]) : ((p_idx == 4) ? CartesianIndex(c[1], c[3], c[2]) : c)
                            end
                            
                            for dat in st.onScrollData.dataToScroll
                                if dat.name == "Mask" || dat.name == "segmentation" || dat.name == "manualModif"
                                    seg_v = dat.dat
                                    # 1. Restore old
                                    last_coords = [c + p_old_d for c in p_orig_coords]
                                    for (j, c) in enumerate(last_coords)
                                        if checkbounds(Bool, seg_v, c) && j <= length(orig_bgs)
                                            seg_v[c] = orig_bgs[j]
                                        end
                                    end
                                    
                                    # 2. Write new
                                    new_coords = [c + p_delta for c in p_orig_coords]
                                    for c in new_coords
                                        if checkbounds(Bool, seg_v, c)
                                            seg_v[c] = Float32(target_id)
                                        end
                                    end
                                end
                            end
                        end
                        
                        panelState.movingLesionLastDelta = new_delta
                        
                        # Synchronize tp_data_cache
                        try
                            MEH = parentmodule(parentmodule(@__MODULE__)).SegmentationDisplay.MakieEventHandlers
                            tp_idx = (MEH.compare_mode[] && clickedPanel == 5) ? MEH.compare_right_tp[] : MEH.current_tp_index[]
                            if haskey(MEH.tp_data_cache, tp_idx)
                                tp_voxels = MEH.tp_data_cache[tp_idx]
                                for (p_idx, panel_data) in enumerate(tp_voxels)
                                    for entry in panel_data
                                        if entry[1] == "Mask" || entry[1] == "manualModif" || entry[1] == "segmentation"
                                            for st_dat in mainStates[p_idx].onScrollData.dataToScroll
                                                if st_dat.name == entry[1]
                                                    entry[2] .= st_dat.dat
                                                    break
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        catch err
                            @warn "Failed to sync tp_data_cache on move lesion: $err"
                        end
                        
                        # Re-render all panels to show moved lesion immediately
                        old_sw = mainStates[1].switchIndex
                        for p in 1:length(mainStates)
                            if sum(abs.(mainStates[p].calcDimsStruct.mainImageQuadVert)) > 0.01f0
                                mainStates[1].switchIndex = p
                                ReactToScroll.reactToScroll(0, mainStates, false)
                            end
                        end
                        mainStates[1].switchIndex = old_sw
                    end
                end
                return
            end
            
            # Dragging: Do the pan!
            lastX, lastY = panelState.lastPanDragCoords[1][1], panelState.lastPanDragCoords[1][2]
            
            dx = x - lastX
            dy = y - lastY  
            
            # Screen dx is horizontal (maps to panX in data matrix), screen dy is vertical (maps to panY in data matrix)
            # Both scaled by the current zoom level so panning speed matches cursor motion on screen
            panSpeedX = Float32(dx / actualW) / max(0.1f0, panelState.calcDimsStruct.zoom)
            panSpeedY = Float32(dy / actualH) / max(0.1f0, panelState.calcDimsStruct.zoom)
            
            panelState.calcDimsStruct.panX = clamp(panelState.calcDimsStruct.panX - panSpeedX, -1.0f0, 1.0f0)
            panelState.calcDimsStruct.panY = clamp(panelState.calcDimsStruct.panY + panSpeedY, -1.0f0, 1.0f0)
            
            panelState.lastPanDragCoords = [CartesianIndex(Int(round(x)), Int(round(y)))]
            
            # Re-render via reactToScroll for all visible panels to prevent freezing
            oldSwitch = mainStates[1].switchIndex
            for i in 1:length(mainStates)
                if sum(abs.(mainStates[i].calcDimsStruct.mainImageQuadVert)) > 0.01f0 # Not hidden
                    mainStates[1].switchIndex = i
                    reactToScroll(0, mainStates, false)
                end
            end
            mainStates[1].switchIndex = oldSwitch
        end
    end # end right-click handler

    # ─── Cursor Info Readout (runs on OpenGL thread via on_next! dispatch) ───────
    # Updates the cursor_info_text Observable and GLFW window title with:
    # study name, HU, SUV, lesion name, view orientation, and slice number.
    # Coordinate mapping is orientation-aware (axial/sagittal/coronal).
    try
        MEH = parentmodule(parentmodule(@__MODULE__)).SegmentationDisplay.MakieEventHandlers
        clickedPanel = mainState.switchIndex
        if clickedPanel >= 1 && clickedPanel <= length(mainStates) && !isempty(mouseCoords)
            panelState = mainStates[clickedPanel]
            viewportW = Float64(mainStates[1].calcDimsStruct.windowWidth)
            viewportH = Float64(mainStates[1].calcDimsStruct.windowHeight)
            actualW = mousestr.actualWindowWidth > 0 ? Float64(mousestr.actualWindowWidth) : viewportW
            actualH = mousestr.actualWindowHeight > 0 ? Float64(mousestr.actualWindowHeight) : viewportH
            x, y = mouseCoords[1][1], mouseCoords[1][2]
            texX, texY = getTextureCoordinatesFromScreen(x, y, panelState.calcDimsStruct, actualW, actualH)
            ix, iy = Int(round(texX)), Int(round(texY))
            currentSlice = panelState.currentDisplayedSlice

            # Build info parts
            parts = String[]

            # Study label (compare mode aware)
            tp_idx = MEH.current_tp_index[]
            tp_label = get(MEH.tp_labels, tp_idx, "TP $tp_idx")
            if MEH.compare_mode[]
                r_idx = MEH.compare_right_tp[]
                r_label = get(MEH.tp_labels, r_idx, "TP $r_idx")
                study_str = "L: $tp_label | R: $r_label"
            else
                study_str = tp_label
            end

            # Read values from 3D volumes using panel-local coordinates
            # (panels 3,4 have reoriented data, so ix,iy,currentSlice are already correct)
            for dat in panelState.onScrollData.dataToScroll
                if dat.name == "CT" && checkbounds(Bool, dat.dat, ix, iy, currentSlice)
                    hu = dat.dat[ix, iy, currentSlice]
                    push!(parts, "HU: $(round(Int, hu))")
                elseif dat.name == "PET" && checkbounds(Bool, dat.dat, ix, iy, currentSlice)
                    suv = dat.dat[ix, iy, currentSlice]
                    push!(parts, "SUV: $(round(suv, digits=2))")
                elseif (dat.name == "Mask" || dat.name == "segmentation") && checkbounds(Bool, dat.dat, ix, iy, currentSlice)
                    mask_val = dat.dat[ix, iy, currentSlice]
                    if mask_val > 0
                        lid = Int(round(mask_val))
                        # Look up lesion name from organ mapping if available
                        organ = try
                            get(MEH.global_organ_mapping[], lid, "")
                        catch
                            ""
                        end
                        label = isempty(organ) ? "Lesion $lid" : "$organ (L$lid)"
                        push!(parts, label)
                    end
                end
            end

            # Panel orientation indicator
            view_name = if clickedPanel == 1 || clickedPanel == 2 || clickedPanel == 5
                "Ax"
            elseif clickedPanel == 3
                "Sag"
            else
                "Cor"
            end
            push!(parts, "[$view_name] Sl:$currentSlice")

            info_str = join(parts, " | ")
            MEH.cursor_info_text[] = info_str
            MEH.cursor_study_text[] = study_str

            # Update GLFW window title (already on OpenGL thread inside on_next!, safe to call directly)
            GLFW.SetWindowTitle(panelState.mainForDisplayObjects.window,
                "MedEye3d - $study_str | $info_str")
        end
    catch
        # Never let info readout crash the mouse handler
    end

end#..ReactToScroll


"""
Handles double-click panel zoom toggle in QuadImage mode.
Dispatched via on_next!(states, data::DoubleClickEvent) — same pattern as all other event types.
"""
function reactToDoubleClick(event::DoubleClickEvent, mainStates::Vector{StateDataFields})
    if length(mainStates) < 4
        return
    end

    # Guard: Disable double click zoom while painting or erasing is active
    if !isempty(mainStates) && isdefined(mainStates[1], :valueForMasToSet) && mainStates[1].valueForMasToSet.is_painting_active
        @info "Double-click zoom ignored: painting/erasing is currently active"
        return
    end

    # Determine which panel was clicked from cursor position
    viewportW = Float64(mainStates[1].calcDimsStruct.windowWidth)
    viewportH = Float64(mainStates[1].calcDimsStruct.windowHeight)
    actualW = event.actualWindowWidth > 0 ? Float64(event.actualWindowWidth) : viewportW
    actualH = event.actualWindowHeight > 0 ? Float64(event.actualWindowHeight) : viewportH

    # Detect compare mode: panels 3+4 are hidden (vertices zeroed out)
    # In compare mode, left half = panel 1, right half = panel 5
    is_compare = false
    if length(mainStates) >= 5
        botVerts = mainStates[3].calcDimsStruct.mainImageQuadVert
        if !isempty(botVerts) && all(v -> v == 0.0f0, botVerts[1:2])
            is_compare = true
        end
    end

    clickedPanel = if quadZoomState.isZoomed
        quadZoomState.zoomedPanel  # when zoomed, always target the zoomed panel
    elseif is_compare
        # In compare mode: left half = panel 1, right half = panel 5
        event.x < actualW / 2.0 ? 1 : 5
    else
        glY = ((actualH - event.y) * 2.0 / viewportH) - 1.0
        topVerts = mainStates[1].calcDimsStruct.mainImageQuadVert
        botVerts = mainStates[3].calcDimsStruct.mainImageQuadVert
        glMidY = (Float64(min(topVerts[10], topVerts[18])) + Float64(max(botVerts[2], botVerts[26]))) / 2.0
        if event.x < actualW / 2.0 && glY > glMidY; 1
        elseif event.x >= actualW / 2.0 && glY > glMidY; 2
        elseif event.x < actualW / 2.0 && glY <= glMidY; 3
        else; 4
        end
    end

    if !quadZoomState.isZoomed
        @info "DOUBLE-CLICK ZOOM IN: panel=$clickedPanel (is_compare=$is_compare)"
        quadZoomState.savedVerts = [copy(s.calcDimsStruct.mainImageQuadVert) for s in mainStates]
        quadZoomState.savedVertSizes = [s.calcDimsStruct.mainQuadVertSize for s in mainStates]
        quadZoomState.zoomedPanel = clickedPanel
        quadZoomState.isZoomed = true

        zoomedCalcDim = getMainVerticies(mainStates[clickedPanel].calcDimsStruct, SingleImage, 1)
        mainStates[clickedPanel].calcDimsStruct = setproperties(
            mainStates[clickedPanel].calcDimsStruct,
            (mainImageQuadVert = zoomedCalcDim.mainImageQuadVert,
             mainQuadVertSize  = zoomedCalcDim.mainQuadVertSize))

        for i in 1:length(mainStates)
            if i != clickedPanel
                mainStates[i].calcDimsStruct = setproperties(
                    mainStates[i].calcDimsStruct,
                    (mainImageQuadVert = Float32[0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0],
                     mainQuadVertSize = 32 * sizeof(Float32)))
            end
        end
    else
        @info "DOUBLE-CLICK ZOOM OUT: restoring layout"
        for i in 1:min(length(mainStates), length(quadZoomState.savedVerts))
            mainStates[i].calcDimsStruct = setproperties(
                mainStates[i].calcDimsStruct,
                (mainImageQuadVert = quadZoomState.savedVerts[i],
                 mainQuadVertSize  = quadZoomState.savedVertSizes[i]))
        end
        quadZoomState.isZoomed = false
    end
end#reactToDoubleClick


"""
given list of cartesian coordinates and some window/ image characteristics - it translates mouse positions
to cartesian coordinates of the texture
strokeWidth - the property connected to the texture marking how thick should be the brush
mouseCoords - list of coordinates of mouse positions while left button remains pressed
calcDims - set of values usefull for calculating mouse position
return vector of translated cartesian coordinates
"""
function translateMouseToTexture(strokeWidth::Int32, mouseCoords::Vector{CartesianIndex{2}}, calcD::CalcDimsStruct, actualW::Int, actualH::Int)::Vector{CartesianIndex{2}}
    filteredList = Vector{CartesianIndex{2}}()
    for c in mouseCoords
        texX, texY = StructsManag.getTextureCoordinatesFromScreen(c[1], c[2], calcD, Float64(actualW), Float64(actualH))
        ix, iy = Int(round(texX)), Int(round(texY))
        if ix > 0 && iy > 0 && ix <= calcD.imageTextureWidth && iy <= calcD.imageTextureHeight
            push!(filteredList, CartesianIndex(ix, iy))
        end
    end

    if (!isempty(filteredList))
        return map(point -> addStrokeWidth(point, Int64(strokeWidth)), filteredList) |>  # adding some points around the point of choice so will be better visible
               (matrix) -> reduce(vcat, matrix) |># when we added some oints around we got list of lists so now we need to flatten it out
                           unique |> # we want only unique elements
                           uniq -> filter(it -> it[1] > 0 && it[1] <= calcD.imageTextureWidth && it[2] > 0 && it[2] <= calcD.imageTextureHeight, uniq)     # as we add new points they may end up getting outside the texture; we need to filter those out
    end #if
    #if we are here we do not have anything meaningfull else to return
    return Vector{CartesianIndex{2}}()
end #translateMouseToTexture

"""
adding the width to the stroke so we will be able to controll how thickly we are painting ...
"""
function addStrokeWidth(point::CartesianIndex{2}, strokeW::Int64)
    return CartesianIndices((-strokeW:strokeW, -strokeW:strokeW)) |> # set of cartesian indices that we will filter ot later
           list -> list .+ point |> # making coordinates around point of intrest
                   added -> filter(x -> (abs(point[1] - x[1]) + abs(x[2] - point[2])) < strokeW, added)# filtering to distant points
end#addStrokeWidth

# xx = CartesianIndex(2, 2)
# xx[1]
"""
helper function for translateMouseToTexture
"""
function getNewX(x::Int, calcD::CalcDimsStruct)::Int
    # first we subtract windowWidthCorr as in window the image do not need to start at the begining  of the window


    # 1) subtract from x the offset that is corrected width times widthCorr/2
    # 2) divide it by total width of the image taking into account offset from both sides
    # 3) now we have coordinate in range between 0 and 1 - relative coorde
    # 4) we get it to texture coordinates by multiplying by texture width
    # 5) we take min of texture width and result to clip it to max value
    # 6) we get max of 1 and result to avoid numbers less then 1

    return max(1, min(Int64(round(((x - (calcD.widthCorr * (calcD.corrected_width / 2))) / (calcD.corrected_width * (1 - calcD.widthCorr))) * calcD.imageTextureWidth)), calcD.imageTextureWidth))

    # In multi - image annotations for
    # LEFT IMAGE
    # return max(1,min(Int64(round(((x - calcD.widthCorr/2) / (calcD.corrected_width/2 + calcD.widthCorr)) * calcD.imageTextureWidth)), calcD.imageTextureWidth))
    # RIGHT IMAGE
    # return max(1, min(Int64(round(((x - (calcD.widthCorr / 2 + (calcD.corrected_width / 2))) / (calcD.corrected_width / 2 + calcD.widthCorr)) * calcD.imageTextureWidth)), calcD.imageTextureWidth))

end#getNewX

"""
helper function for translateMouseToTexture
"""
function getNewY(y::Int, calcD::CalcDimsStruct)::Int
    rounded_value = calcD.imageTextureHeight - round(((y - (calcD.heightCorr * (calcD.windowHeight / 2))) / (calcD.windowHeight * (1 - calcD.heightCorr))) * calcD.imageTextureHeight)

    clamped_value = clamp(rounded_value, 1, calcD.imageTextureHeight)
    return Int64(clamped_value)

end#getNewY

end #ReactOnMouseClickAndDrag



"""
left image

return max(1,min(Int64(round(((x - calcD.widthCorr/2) / (calcD.corrected_width/2 + calcD.widthCorr)) * calcD.imageTextureWidth)), calcD.imageTextureWidth))


right image
return max(1, min(Int64(round(((x - (calcD.widthCorr / 2 + (calcD.corrected_width / 2))) / (calcD.corrected_width / 2 + calcD.widthCorr)) * calcD.imageTextureWidth)), calcD.imageTextureWidth))


"""
