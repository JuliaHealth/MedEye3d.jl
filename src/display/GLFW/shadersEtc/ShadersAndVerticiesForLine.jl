module ShadersAndVerticiesForLine
using ModernGL, GeometryTypes, GLFW
using ..ForDisplayStructs, ..CustomFragShad, ..ModernGlUtil, ..PrepareWindowHelpers, ..DataStructs, ..TextureManag, ..BasicStructs, ..StructsManag, ..DisplayWords

export createAndInitLineShaderProgram, updateCrosshairPosition, calculateCrosshairVertices, updateCrosshairBuffer


#These are the openGl vertex coordinates for crosshair in 3d space
line_vertices = Float32[
    0.1, 0.0, 0.0,  # top right
    -0.1, 0.0, 0.0,  # bottom right
    0.0, -0.1, 0.0,  # bottom left
    0.0, 0.1, 0.0   # top left
]

# Indices for drawing lines
line_indices = UInt32[
    0, 1,  # Line from top right to bottom right
    2, 3   # Line from bottom left to top left
]

function fragShaderLineSrc()
    return """
    #version 330 core
    out vec4 FragColor;




    void main()
    {
        FragColor = vec4(1.0, 1.0, 0.0, 1.0); // Yellow color
    }
    """
end

function createAndInitLineShaderProgram(vertexShader::UInt32)
    fragmentShaderSourceLine = fragShaderLineSrc()
    fsh = """
    $(fragmentShaderSourceLine)
    """
    lineFragmentShader = createShader(fsh, GL_FRAGMENT_SHADER)
    lineShaderProgram = glCreateProgram()
    glAttachShader(lineShaderProgram, lineFragmentShader)
    glAttachShader(lineShaderProgram, vertexShader)
    glLinkProgram(lineShaderProgram)

    return (lineFragmentShader, lineShaderProgram)
end



function updateCrosshairBuffer(vertices, crosshair)

    glBindBuffer(GL_ARRAY_BUFFER, 0) #unbind the buffer for mainRect
    # Update the VBO with new vertex data
    glBindBuffer(GL_ARRAY_BUFFER, crosshair.vbo[]) #bind the buffer for crosshair vbo
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices)
    glBindBuffer(GL_ARRAY_BUFFER, 0) #unbind the buffer for crosshair

    # glBindBuffer(GL_ARRAY_BUFFER, textFields.vbo[])
end

function renderLines(forDisplayConstants, crosshair, mainRect)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, C_NULL)

    # Switch to crosshair shader and render crosshair
    glUseProgram(crosshair.shaderProgram)
    glBindVertexArray(crosshair.vao[])
    glDrawElements(GL_LINES, 4, GL_UNSIGNED_INT, C_NULL)

    # Switch back to main shader program
    # using the shader program from the mainRect causes the image render to disappear, so better use the one from forDisplayConstants !!
    glUseProgram(forDisplayConstants.shader_program)
    glBindVertexArray(mainRect.vao[])

    GLFW.SwapBuffers(forDisplayConstants.window)
end


function realSpacePoint(x, y, currentSlice, scrollDimension, spacing, origin)
    # scrollNumb = activeState.currentDisplayedSlice
    # currentDim = Int64(activeState.onScrollData.dataToScrollDims.dimensionToScroll)
    pointInRealSpace = [x, y, scrollDimension]
    #==
    Based on the dimensions we are scrolling in :
    3 : (x,y, scrollNum)
    2 : (x, scrollNumb, y)
    1 : (scrollNumb, x,y)
    ==#
    pointInRealSpace[scrollDimension] = currentSlice
    #using spacing[1] to access the only tuple of floats in the vector array
    #Adding this features allows us to disable the concept of overlaid images in multi-image
    foreach(enumerate(spacing[1])) do (index, val)
        pointInRealSpace[index] *= val
        pointInRealSpace[index] += origin[1][index]
    end
    return pointInRealSpace
end

function passiveTexPoint(activeRealPoint, passiveSpacing, passiveOrigin)
    foreach(enumerate(passiveSpacing[1])) do (index, val)
        activeRealPoint[index] -= passiveOrigin[1][index]
        activeRealPoint[index] /= val
    end
    return activeRealPoint
end



function passiveTexToWindRightX(tex_x, calcD::CalcDimsStruct)
    window_x = (tex_x / calcD.imageTextureWidth) * (calcD.corrected_width / 2 + calcD.widthCorr) +
               (calcD.widthCorr / 2 + (calcD.corrected_width / 2))
    return round(Int64, window_x)
end
function passiveTexToWindLeftX(tex_x, calcD::CalcDimsStruct)
    window_x = (tex_x / calcD.imageTextureWidth) * (calcD.corrected_width / 2 + calcD.widthCorr) + calcD.widthCorr / 2
    return round(Int64, window_x)
end
function passiveTexToWindY(tex_y, calcD::CalcDimsStruct)
    window_y = calcD.windowHeight - (
        (tex_y / calcD.imageTextureHeight) * (calcD.windowHeight * (1 - calcD.heightCorr)) +
        (calcD.heightCorr * (calcD.windowHeight / 2))
    )
    return round(Int64, window_y)
end


function skipSlice(mainState, scrollNumb)
    current = mainState.currentDisplayedSlice
    old = current
    #when shift is pressed scrolling is 10 times faster

    current = scrollNumb


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


        # updateImagesDisplayed(singleSlDat, mainState.mainForDisplayObjects, mainState.textDispObj, mainState.calcDimsStruct, mainState.valueForMasToSet, mainState.crosshairFields, mainState.mainRectFields, mainState.displayMode)



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
end


"""
Updating the values of the crosshair verticies to get dynamic
crosshair display
"""
function updateCrosshairPosition(x, y, crosshair, mainRect, forDisplayConstants, currentSlice, passiveCurrentSlice, scrollDimension, passiveScrollDimension, spacing, passiveSpacing, origin, passiveOrigin, activeImagePosition, activeCalcD, passiveCalcD, passiveState)

    activeRealPoint = realSpacePoint(x, y, currentSlice, scrollDimension, spacing, origin)
    passiveTexturePoint = passiveTexPoint(activeRealPoint, passiveSpacing, passiveOrigin)
    passiveX, passiveY, passiveScrollNumb = (Nothing, Nothing, Nothing)
    if passiveScrollDimension == 1
        passiveScrollNumb, passiveX, passiveY = passiveTexturePoint
    elseif passiveScrollDimension == 2
        passiveX, passiveScrollNumb, passiveY = passiveTexturePoint
    elseif passiveScrollDimension == 3
        passiveX, passiveY, passiveScrollNumb = passiveTexturePoint
    end
    passiveWindowX = activeImagePosition == 1 ? passiveTexToWindRightX(passiveX, passiveCalcD) : passiveTexToWindLeftX(passiveX, passiveCalcD)
    passiveWindowY = passiveTexToWindY(passiveY, passiveCalcD)

    passiveWindowPoint = [Nothing, Nothing, Nothing]
    if passiveScrollDimension == 1
        passiveWindowPoint = [passiveScrollNumb, passiveWindowX, passiveWindowY]
    elseif passiveScrollDimension == 2
        passiveWindowPoint = [passiveWindowX, passiveScrollNumb, passiveWindowY]
    elseif passiveScrollDimension == 3
        passiveWindowPoint = [passiveWindowX, passiveWindowY, passiveScrollNumb]
    end
    #passive spacing and origin
    # @info passiveWindowPoint
    passiveOpenGlX, passiveOpenGlY = [Nothing, Nothing]
    if passiveScrollDimension == 1
        passiveOpenGlX, passiveOpenGlY = (passiveWindowPoint[2], passiveWindowPoint[3])
    elseif passiveScrollDimension == 2
        passiveOpenGlX, passiveOpenGlY = (passiveWindowPoint[1], passiveWindowPoint[3])

    elseif passiveScrollDimension == 3
        passiveOpenGlX, passiveOpenGlY = (passiveWindowPoint[1], passiveWindowPoint[2])
    end

    passiveOpenGlX = (passiveOpenGlX / passiveCalcD.windowWidth) * 2 - 1
    passiveOpenGlY = ((passiveOpenGlY / passiveCalcD.windowHeight) * 2 - 1) * -1

    # @info passiveScrollNumb
    # @info Int(passiveScrollNumb)
    # Update crosshair vertices
    new_vertices = calculateCrosshairVertices(passiveX, passiveY, passiveScrollNumb, passiveScrollDimension, passiveCalcD, activeImagePosition)
    updateCrosshairBuffer(new_vertices, crosshair)

    skipSlice(passiveState, round(Int64, passiveScrollNumb)) #Adding round fixed Inexact error during conversion from float to Int
    updateImagesDisplayed(passiveState.currentlyDispDat, passiveState.mainForDisplayObjects, passiveState.textDispObj, passiveState.calcDimsStruct, passiveState.valueForMasToSet, passiveState.crosshairFields, passiveState.mainRectFields, passiveState.displayMode)

    renderLines(forDisplayConstants, crosshair, mainRect)
end

"""
Calculates the NDC vertices for the crosshair given the texture coordinates
"""
function calculateCrosshairVertices(texX, texY, texZ, scrollDimension, calcD::CalcDimsStruct, activeImagePosition::Int=1)
    passiveWindowX = activeImagePosition == 1 ? passiveTexToWindRightX(texX, calcD) : passiveTexToWindLeftX(texX, calcD)
    passiveWindowY = passiveTexToWindY(texY, calcD)

    passiveWindowPoint = [Nothing, Nothing, Nothing]
    if scrollDimension == 1
        passiveWindowPoint = [texZ, passiveWindowX, passiveWindowY]
    elseif scrollDimension == 2
        passiveWindowPoint = [passiveWindowX, texZ, passiveWindowY]
    elseif scrollDimension == 3
        passiveWindowPoint = [passiveWindowX, passiveWindowY, texZ]
    end

    passiveOpenGlX, passiveOpenGlY = [Nothing, Nothing]
    if scrollDimension == 1
        passiveOpenGlX, passiveOpenGlY = (passiveWindowPoint[2], passiveWindowPoint[3])
    elseif scrollDimension == 2
        passiveOpenGlX, passiveOpenGlY = (passiveWindowPoint[1], passiveWindowPoint[3])
    elseif scrollDimension == 3
        passiveOpenGlX, passiveOpenGlY = (passiveWindowPoint[1], passiveWindowPoint[2])
    end

    passiveOpenGlX = (passiveOpenGlX / calcD.windowWidth) * 2 - 1
    passiveOpenGlY = ((passiveOpenGlY / calcD.windowHeight) * 2 - 1) * -1

    return Float32[
        passiveOpenGlX-0.05, passiveOpenGlY, 0.0,
        passiveOpenGlX+0.05, passiveOpenGlY, 0.0,
        passiveOpenGlX, passiveOpenGlY-0.05, 0.0,
        passiveOpenGlX, passiveOpenGlY+0.05, 0.0
    ]
end

"""
Helper function specifically for QuadImage to compute vertices given CartesianIndex
"""
function calculateCrosshairVertices(activePos::CartesianIndex{3}, panelIdx::Int, calcD::CalcDimsStruct)
    # Mapping activePos (X,Y,Z) to panel's texture coordinates
    # Panel 1 & 2 (Axial): shows X,Y
    # Panel 3 (Sagittal): shows Y,Z (permuted 2,3,1 => dimToScroll=3)
    # Panel 4 (Coronal): shows X,Z (permuted 1,3,2 => dimToScroll=3)
    
    texX, texY, texZ = (0, 0, 0)
    if panelIdx == 1 || panelIdx == 2
        texX = activePos[1]
        texY = activePos[2]
        texZ = activePos[3]
    elseif panelIdx == 3
        texX = activePos[2]
        texY = activePos[3]
        texZ = activePos[1]
    elseif panelIdx == 4
        texX = activePos[1]
        texY = activePos[3]
        texZ = activePos[2]
    end
    
    # ActiveImagePosition is basically used to check if it's the right or left side of the window
    # In QuadImage: panel 1 (top-left) => Left, panel 2 (top-right) => Right
    # panel 3 (bottom-left) => Left, panel 4 (bottom-right) => Right
    imgPos = (panelIdx == 1 || panelIdx == 3) ? 2 : 1
    
    # In QuadImage, scrollDimension is always 3 for all panels
    return calculateCrosshairVertices(texX, texY, texZ, 3, calcD, imgPos)
end

end




"""
Next steps :
Provide the rendering of passive image with mouse move [DONE]
skip a certain number of slices   [DONE]
Make sure the shape of both the images loaded are same, if not make them same with the additional zeros [IN PROGRESS]
By the way remember to save crosshair position in state In mouse change position and in on scroll ![REDUNDANT]!
Scroll on the images and decide whether for the images to scroll simultaneously or update the position of the crosshair on scroll [DONE]
Reset the image or render a black color in the window in both the single image and multi-image mode to remove and rendered artifacts.
Precalculate verticies for supervoxel from main_supervoxel folder in sv_try branch
Render dynamic lines on test slice data on email, and display lines with the passed verticies.
"""


