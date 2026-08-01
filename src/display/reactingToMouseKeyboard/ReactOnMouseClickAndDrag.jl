

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
using ..DataStructs, ..StructsManag, ..ShadersAndVerticiesForLine, ..ReactToScroll, ..DisplayWords
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
        # if (mouseStructInstance.isLeftButtonDown && x >= xmin && x <= xmax && y >= ymin && y <= ymax)
        if (x >= xmin && x <= xmax && y >= ymin && y <= ymax)
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
        leftMouseButtonDownResult = (button == GLFW.MOUSE_BUTTON_1 && action == GLFW.PRESS)
        mouseStructInstance.isLeftButtonDown = leftMouseButtonDownResult

        rightMouseButtonDownResult = (button == GLFW.MOUSE_BUTTON_2 && action == GLFW.PRESS)
        mouseStructInstance.isRightButtonDown = rightMouseButtonDownResult

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
# function react_to_draw(textureList,actor,mouseCoords_channel)
function react_to_draw(mouseStructArray::Vector{MouseStruct}, stateObjects::Vector{StateDataFields})
    stateObject = stateObjects[stateObjects[1].switchIndex]
    # sleep(0.1);
    # @info "react_to_draw after sleep" isready(mouseCoords_channel)
    texture = stateObject.textureToModifyVec[1]
    calcDim = stateObject.calcDimsStruct


    # mouseCoords=take!(mouseCoords_channel).lastCoordinates
    # mappedCoords=translateMouseToTexture(texture.strokeWidth, mouseCoords, actor.actor.calcDimsStruct)
    # # two dimensional coordinates on plane of intrest (current slice)

    """
    get a list of MouseStruct from fetch and take!
    map each MouseStruct using translateMouseToTexture
    result will be mappedCoords, in react_to_draw
    check whether the length of the mappedCoords is greater than 0
    """
    mappedCoords = Vector{CartesianIndex{2}}()
    for mouseStruct in mouseStructArray
        mouseCoords = mouseStruct.lastCoordinates
        append!(mappedCoords, translateMouseToTexture(texture.strokeWidth, mouseCoords, stateObject.calcDimsStruct))
    end
    # append!(mappedCoords, translateMouseToTexture(texture.strokeWidth, mouseCoords, stateObject.calcDimsStruct))

    # @info "react_to_draw after channel" mappedCoords

    # is_sth_in=true

    twoDimDat = stateObject.currentlyDispDat |> # accessing currently displayed data
                (singSl) -> singSl.listOfDataAndImageNames[singSl.nameIndexes[texture.name]] #accessing the texture data we want to modify



    toSet = convert(parameter_type(texture), stateObject.valueForMasToSet.value)
    sliceDat = modSlice!(twoDimDat, mappedCoords, convert(twoDimDat.type, toSet)) # modifying data associated with texture


    #  updateTexture(twoDimDat.type,sliceDat, texture,0,0,calcDim.imageTextureWidth,calcDim.imageTextureHeight  )

    singleSliceDat = setproperties(stateObject.currentlyDispDat, (listOfDataAndImageNames = [sliceDat]))



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
        if length(mainStates) == 4 # QuadImage mode
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
                
                # Convert mouse to NDC accounting for viewport vs content area mismatch
                mouseGlX = (x * 2.0 / viewportW) - 1.0
                mouseGlY = ((actualH - y) * 2.0 / viewportH) - 1.0
                
                # Compute panel split dynamically from actual vertex positions
                topVerts = mainStates[1].calcDimsStruct.mainImageQuadVert
                botVerts = mainStates[3].calcDimsStruct.mainImageQuadVert
                topPanelBottom = Float64(min(topVerts[10], topVerts[18]))
                botPanelTop = Float64(max(botVerts[2], botVerts[26]))
                glMidY = (topPanelBottom + botPanelTop) / 2.0
                
                if x < actualW / 2.0 && mouseGlY > glMidY
                    mainState.switchIndex = 1
                elseif x >= actualW / 2.0 && mouseGlY > glMidY
                    mainState.switchIndex = 2
                elseif x < actualW / 2.0 && mouseGlY <= glMidY
                    mainState.switchIndex = 3
                else
                    mainState.switchIndex = 4
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
    
    # 2. Right-click cross-plane jumping — needs coords
    if !isempty(mouseCoords) && mousestr.isRightButtonDown && length(mainStates) == 4 # QuadImage mode
        viewportW = Float64(mainStates[1].calcDimsStruct.windowWidth)
        viewportH = Float64(mainStates[1].calcDimsStruct.windowHeight)
        actualW = mousestr.actualWindowWidth > 0 ? Float64(mousestr.actualWindowWidth) : viewportW
        actualH = mousestr.actualWindowHeight > 0 ? Float64(mousestr.actualWindowHeight) : viewportH
        x, y = (mouseCoords[1][1], mouseCoords[1][2])
        clickedPanel = mainState.switchIndex

        # Read actual rendered vertex positions from the panel's calcDimsStruct
        verts = mainStates[clickedPanel].calcDimsStruct.mainImageQuadVert
        glLeft   = Float64(min(verts[17], verts[25]))
        glRight  = Float64(max(verts[1], verts[9]))
        glBottom = Float64(min(verts[10], verts[18]))
        glTop    = Float64(max(verts[2], verts[26]))
        
        # Convert click to OpenGL NDC accounting for viewport vs content area mismatch
        glX = (x * 2.0 / viewportW) - 1.0
        glY = ((actualH - y) * 2.0 / viewportH) - 1.0
        
        # Map click to OpenGL texture coordinates (s, t) within vertex bounds
        # s: 0=left edge, 1=right edge
        # t: 0=bottom of quad (array row 1), 1=top of quad (array row texH)
        # Clicks in padding zone clamp to nearest image edge
        s = clamp((glX - glLeft) / (glRight - glLeft), 0.0, 1.0)
        t = clamp((glY - glBottom) / (glTop - glBottom), 0.0, 1.0)

        texW = Float64(mainStates[clickedPanel].calcDimsStruct.imageTextureWidth)
        texH = Float64(mainStates[clickedPanel].calcDimsStruct.imageTextureHeight)
        
        # s=0 → col 1, s=1 → col texW
        texX = clamp(round(Int, s * texW), 1, Int(texW))
        
        # texY mapping (same for ALL panels):
        # t=1 (top of quad) → texY=texH (last row) — matches getNewY behavior
        # t=0 (bottom)      → texY=1 (first row)
        texY = clamp(round(Int, t * texH), 1, Int(texH))
        
        # Detect if click is in padding zone
        inPadding = (glX < glLeft || glX > glRight || glY < glBottom || glY > glTop)
        # Convert vertex bounds to mouse pixel coords for debug display
        imgTopPx = round(Int, actualH - (glTop + 1.0) / 2.0 * viewportH)
        imgBotPx = round(Int, actualH - (glBottom + 1.0) / 2.0 * viewportH)
        imgLeftPx = round(Int, (glLeft + 1.0) / 2.0 * viewportW)
        imgRightPx = round(Int, (glRight + 1.0) / 2.0 * viewportW)
        
        @info "RIGHT-CLICK: panel=$clickedPanel windowXY=($x,$y) inPadding=$inPadding viewport=$(Int(viewportW))x$(Int(viewportH)) actual=$(Int(actualW))x$(Int(actualH))"
        @info "  Image mouse-pixel bounds: top=$imgTopPx bot=$imgBotPx left=$imgLeftPx right=$imgRightPx"
        @info "  glXY=($glX,$glY) bounds=(L=$glLeft,R=$glRight,B=$glBottom,T=$glTop)"
        @info "  s=$s t=$t texX=$texX texY=$texY texW=$texW texH=$texH"
        
        currentSlice = mainStates[clickedPanel].currentDisplayedSlice
        
        # Map texture coordinates back to original volume coordinates
        # Panel 1 & 2 (Axial): data = (origX, origY, origZ), texture shows (origX, origY)
        # Panel 3 (Sagittal): data = permutedims(iso, (2,3,1)) = (origY, origZ, origX)
        #   texture shows (origY, origZ), slice = origX
        # Panel 4 (Coronal): data = permutedims(iso, (1,3,2)) = (origX, origZ, origY)
        #   texture shows (origX, origZ), slice = origY
        if clickedPanel == 1 || clickedPanel == 2
            origX, origY, origZ = texX, texY, currentSlice
        elseif clickedPanel == 3  # Sagittal
            origY, origZ, origX = texX, texY, currentSlice
        else # Bottom-Right (4) (Coronal)
            origX, origZ, origY = texX, texY, currentSlice
        end
        
        @info "  origX=$origX origY=$origY origZ=$origZ currentSlice=$currentSlice"
        @info "  Axial scrolls Z(1-$(mainStates[1].onScrollData.slicesNumber)) Sag scrolls origX(1-$(mainStates[3].onScrollData.slicesNumber)) Cor scrolls origY(1-$(mainStates[4].onScrollData.slicesNumber))"
        
        # Jump other panels to the corresponding slices
        # Panel 1 & 2 scroll through Z (origZ), Panel 3 scrolls through origX, Panel 4 scrolls through origY
        targets = [(1, origZ), (2, origZ), (3, origX), (4, origY)]
        
        for (panelIdx, targetSlice) in targets
            if panelIdx != clickedPanel
                panelState = mainStates[panelIdx]
                currSlice = panelState.currentDisplayedSlice
                @info "  Panel $panelIdx: current=$currSlice target=$targetSlice"
                if targetSlice != currSlice
                    # Clamp target to valid range
                    lastSlice = panelState.onScrollData.slicesNumber
                    newSlice = clamp(targetSlice, 1, lastSlice)
                    
                    # Extract new 2D slice from 3D data
                    singleSlDat = panelState.onScrollData.dataToScroll |>
                        (scrDat) -> map(threeDimDat -> threeToTwoDimm(threeDimDat.type, Int64(newSlice), panelState.onScrollData.dimensionToScroll, threeDimDat), scrDat) |>
                        (twoDimList) -> SingleSliceDat(listOfDataAndImageNames=twoDimList, sliceNumber=newSlice, textToDisp=getTextForCurrentSlice(panelState.onScrollData, Int32(newSlice)))
                    
                    # Upload new texture data to GPU (without rendering/SwapBuffers)
                    for updateDat in singleSlDat.listOfDataAndImageNames
                        findList = findall((texSpec) -> texSpec.name == updateDat.name, panelState.mainForDisplayObjects.listOfTextSpecifications)
                        if !isempty(findList)
                            texSpec = panelState.mainForDisplayObjects.listOfTextSpecifications[findList[1]]
                            updateTexture(updateDat.type, updateDat.dat, texSpec, 0, 0, panelState.calcDimsStruct.imageTextureWidth, panelState.calcDimsStruct.imageTextureHeight)
                        end
                    end
                    
                    # Update state (slice number and display data)
                    panelState.currentlyDispDat = singleSlDat
                    panelState.currentDisplayedSlice = newSlice
                    panelState.isSliceChanged = true
                end
            end
        end
    end # end right-click handler

    # 3. Crosshair rendering for multi-image modes (needs coords, MultiImage only)
    if !isempty(mouseCoords) && length(mainStates) > 1 && length(mainStates) != 4 #only in multiImage mode (but NOT QuadImage)
        textBeginning2, midPoint2, imageRange2 = openGlSystemVals(mainState.calcDimsStruct.fractionOfMainIm, mainState.calcDimsStruct.windowWidth)
        cursorXPosOpenGl2 = (mouseCoords[1][1] / mainState.calcDimsStruct.windowWidth) * 2 - 1
        if cursorXPosOpenGl2 > midPoint2
            # @info cursorXPosOpenGl2
            mainState.switchIndex = 2
        elseif cursorXPosOpenGl2 < midPoint2
            # @info cursorXPosOpenGl2
            mainState.switchIndex = 1
        end

        #switching the index here to log real space point from the left or right images for crosshair rendering
        mainState = mainStates[mainState.switchIndex]
        passiveState = mainState == mainStates[1] ? mainStates[2] : mainStates[1]

        #Conversion of mouse coordinates to multi-image specific texture coordinates
        #LEFT IMAGE
        texPosX = Nothing
        texPosY = Nothing
        calcD = mainState.calcDimsStruct
        x, y = (mouseCoords[1][1], mouseCoords[1][2])
        if mainState.imagePosition == 1
            texPosX = max(1, min(Float64(round(((x - calcD.widthCorr / 2) / (calcD.corrected_width / 2 + calcD.widthCorr)) * calcD.imageTextureWidth)), calcD.imageTextureWidth))
            #RIGHT IMAGE
        elseif mainState.imagePosition == 2
            texPosX = max(1, min(Float64(round(((x - (calcD.widthCorr / 2 + (calcD.corrected_width / 2))) / (calcD.corrected_width / 2 + calcD.widthCorr)) * calcD.imageTextureWidth)), calcD.imageTextureWidth))
        end
        texPosY = Float64(getNewY(y, calcD))

        ShadersAndVerticiesForLine.updateCrosshairPosition(texPosX, texPosY, mainState.crosshairFields, mainState.mainRectFields, mainState.mainForDisplayObjects, mainState.currentDisplayedSlice, passiveState.currentDisplayedSlice, mainState.onScrollData.dataToScrollDims.dimensionToScroll, passiveState.onScrollData.dataToScrollDims.dimensionToScroll, mainState.spacingsValue, passiveState.spacingsValue, mainState.originValue, passiveState.originValue, mainState.imagePosition, mainState.calcDimsStruct, passiveState.calcDimsStruct, passiveState)
    end

    #Dynamically moving crosshair on the screen based on mouse position, only if in multiImage mode, so simply shove it in above if block
    # if (mousestr.isLeftButtonDown)
    #     mappedCoords = translateMouseToTexture(Int32(1), mouseCoords, mainState.calcDimsStruct)
    #     mappedCorrd = mappedCoords
    #     if (!isempty(mappedCorrd))
    #         cartMapped = cartTwoToThree(mainState.onScrollData.dataToScrollDims, mainState.currentDisplayedSlice, mappedCoords[1])

    #         mainState.lastRecordedMousePosition = cartMapped
    #     end#if
    # end
    # @info "Mouse drag coordinates  : " mappedCoords

end#..ReactToScroll


"""
Handles double-click panel zoom toggle in QuadImage mode.
Dispatched via on_next!(states, data::DoubleClickEvent) — same pattern as all other event types.
"""
function reactToDoubleClick(event::DoubleClickEvent, mainStates::Vector{StateDataFields})
    length(mainStates) == 4 || return  # QuadImage only

    # Determine which panel was clicked from cursor position
    viewportW = Float64(mainStates[1].calcDimsStruct.windowWidth)
    viewportH = Float64(mainStates[1].calcDimsStruct.windowHeight)
    actualW = event.actualWindowWidth > 0 ? Float64(event.actualWindowWidth) : viewportW
    actualH = event.actualWindowHeight > 0 ? Float64(event.actualWindowHeight) : viewportH

    clickedPanel = if quadZoomState.isZoomed
        quadZoomState.zoomedPanel  # when zoomed, always target the zoomed panel
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
        @info "DOUBLE-CLICK ZOOM IN: panel=$clickedPanel"
        quadZoomState.savedVerts = [copy(s.calcDimsStruct.mainImageQuadVert) for s in mainStates]
        quadZoomState.savedVertSizes = [s.calcDimsStruct.mainQuadVertSize for s in mainStates]
        quadZoomState.zoomedPanel = clickedPanel
        quadZoomState.isZoomed = true

        zoomedCalcDim = getMainVerticies(mainStates[clickedPanel].calcDimsStruct, SingleImage, 1)
        mainStates[clickedPanel].calcDimsStruct = setproperties(
            mainStates[clickedPanel].calcDimsStruct,
            (mainImageQuadVert = zoomedCalcDim.mainImageQuadVert,
             mainQuadVertSize  = zoomedCalcDim.mainQuadVertSize))

        offscreen = Float32.([-10,-10,0,0,0,0,0,0, -10,-10,0,0,0,0,0,0,
                               -10,-10,0,0,0,0,0,0, -10,-10,0,0,0,0,0,0])
        for i in 1:4
            if i != clickedPanel
                mainStates[i].calcDimsStruct = setproperties(
                    mainStates[i].calcDimsStruct,
                    (mainImageQuadVert = offscreen,
                     mainQuadVertSize  = sizeof(offscreen)))
            end
        end
    else
        @info "DOUBLE-CLICK ZOOM OUT: restoring 4-pane"
        for i in 1:4
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
function translateMouseToTexture(strokeWidth::Int32, mouseCoords::Vector{CartesianIndex{2}}, calcD::CalcDimsStruct)::Vector{CartesianIndex{2}}


    filteredList = map(c -> CartesianIndex(getNewX(c[1], calcD), getNewY(c[2], calcD)), mouseCoords) |>
                   (x) -> filter(it -> it[1] > 0 && it[2] > 0, x)       # we do not want to try access it in point 0 as julia is 1 indexed
    if (!isempty(filteredList))
        return map(point -> addStrokeWidth(point, Int64(strokeWidth)), filteredList) |>  # adding some points around the point of choice so will be better visible
               (matrix) -> reduce(vcat, matrix) |># when we added some oints around we got list of lists so now we need to flatten it out
                           unique |> # we want only unique elements
                           uniq -> filter(it -> it[1] > 0 && it[1] < calcD.imageTextureWidth && it[2] > 0 && it[2] < calcD.imageTextureHeight, uniq)     # as we add new points they may end up getting outside the texture; we need to filter those out
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
