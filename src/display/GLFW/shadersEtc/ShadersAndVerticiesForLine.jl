module ShadersAndVerticiesForLine
using ModernGL, GeometryTypes, GLFW
using ..ForDisplayStructs, ..CustomFragShad, ..ModernGlUtil, ..PrepareWindowHelpers

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



function updateCrosshairBuffer(vertices, crosshair, mainRect)

    glBindBuffer(GL_ARRAY_BUFFER, 0) #unbind the buffer for mainRect
    # Update the VBO with new vertex data
    glBindBuffer(GL_ARRAY_BUFFER, crosshair.vbo[]) #bind the buffer for crosshair vbo
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices)
    glBindBuffer(GL_ARRAY_BUFFER, 0) #unbind the buffer for crosshair

    glBindBuffer(GL_ARRAY_BUFFER, mainRect.vbo[])
end

"""
Updating the values of the crosshair verticies to get dynamic
crosshair display
"""
function updateCrosshairPosition(x, y, crosshair, mainRect)
    # Update crosshair vertices
    new_vertices = Float32[
        x-0.05, y, 0.0,
        x+0.05, y, 0.0,
        x, y-0.05, 0.0,
        x, y+0.05, 0.0
    ]
    updateCrosshairBuffer(new_vertices, crosshair, mainRect)
end

end




