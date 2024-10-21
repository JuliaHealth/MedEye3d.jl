module ShadersAndVerticiesForLine
using ModernGL, GeometryTypes, GLFW
using ..ForDisplayStructs, ..CustomFragShad, ..ModernGlUtil, ..PrepareWindowHelpers, ..DataStructs, ..TextureManag

export createAndInitLineShaderProgram, updateCrosshairPosition


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



function passiveTexToWindRightX(tex_x::Float64, calcD::CalcDimsStruct)
    window_x = (tex_x / calcD.imageTextureWidth) * (calcD.corrected_width / 2 + calcD.widthCorr) +
               (calcD.widthCorr / 2 + (calcD.corrected_width / 2))
    return round(Int64, window_x)
end
function passiveTexToWindLeftX(tex_x::Float64, calcD::CalcDimsStruct)
    window_x = (tex_x / calcD.imageTextureWidth) * (calcD.corrected_width / 2 + calcD.widthCorr) + calcD.widthCorr / 2
    return round(Int64, window_x)
end
function passiveTexToWindY(tex_y::Float64, calcD::CalcDimsStruct)
    window_y = calcD.windowHeight - (
        (tex_y / calcD.imageTextureHeight) * (calcD.windowHeight * (1 - calcD.heightCorr)) +
        (calcD.heightCorr * (calcD.windowHeight / 2))
    )
    return round(Int64, window_y)
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
    # Update crosshair vertices
    new_vertices = Float32[
        passiveOpenGlX-0.05, passiveOpenGlY, 0.0,
        passiveOpenGlX+0.05, passiveOpenGlY, 0.0,
        passiveOpenGlX, passiveOpenGlY-0.05, 0.0,
        passiveOpenGlX, passiveOpenGlY+0.05, 0.0
    ]
    updateCrosshairBuffer(new_vertices, crosshair)

    updateImagesDisplayed(passiveState.currentlyDispDat, passiveState.mainForDisplayObjects, passiveState.textDispObj, passiveState.calcDimsStruct, passiveState.valueForMasToSet, passiveState.crosshairFields, passiveState.mainRectFields, passiveState.displayMode)

    renderLines(forDisplayConstants, crosshair, mainRect)
end

end




"""
Next steps :
Provide the rendering of passive image with mouse move [DONE]
skip a certain number of slices [IN PROGRESS]
Make sure the shape of both the images loaded are same, if not make them same with the additional zeros [IN PROGRESS]
By the way remember to save crosshair position in state In mouse change position and in on scroll
"""


