using ModernGL, GLFW

function initializeWindow(windowWidth::Int, windowHeight::Int)
    GLFW.Init()
    # Create a windowed mode window and its OpenGL context
    window = GLFW.CreateWindow(windowWidth, windowHeight, "Segmentation Visualization")
    # Make the window's context current
    GLFW.MakeContextCurrent(window)
    GLFW.ShowWindow(window)
    GLFW.SetWindowSize(window, windowWidth, windowHeight) # Seems to be necessary to guarantee that window > 0
    glViewport(0, 0, windowWidth, windowHeight)
    glDisable(GL_LIGHTING)
    glEnable(GL_TEXTURE_2D)
    return window
end #initializeWindow

window = initializeWindow(800, 600)

# Vertex data for a line
vertices = Float32[
    0.5, 0.5, 0.0,  # top right
    0.5, -0.5, 0.0,  # bottom right
    -0.5, -0.5, 0.0,  # bottom left
    -0.5, 0.5, 0.0   # top left
]

# Indices for drawing lines
indices = UInt32[
    0, 1,  # Line from top right to bottom right
    2, 3   # Line from bottom left to top left
]

# Vertex shader source code
vertex_shader_source = """
#version 330 core
layout (location = 0) in vec3 aPos;
void main()
{
    gl_Position = vec4(aPos, 1.0);
}
"""

# Fragment shader source code
fragment_shader_source = """
#version 330 core
out vec4 FragColor;
void main()
{
    FragColor = vec4(1.0, 1.0, 0.0, 1.0); // Yellow color
}
"""

# Compile vertex shader
vertex_shader = glCreateShader(GL_VERTEX_SHADER)
function createShader(source, typ)
    # Create the shader
    shader = glCreateShader(typ)::GLuint
    if shader == 0
        error("Error creating shader: ", glErrorMessage())
    end
    # Compile the shader
    glShaderSource(shader, 1, convert(Ptr{UInt8}, pointer([convert(Ptr{GLchar}, pointer(source))])), C_NULL)
    glCompileShader(shader)
    # Check for errors
    # !validateShader(shader) && error("Shader creation error: ", getInfoLog(shader))
    return shader
end

vertex_shader = createShader(vertex_shader_source, GL_VERTEX_SHADER)
fragment_shader= createShader(fragment_shader_source, GL_FRAGMENT_SHADER)

# Link shaders into a shader program
shader_program = glCreateProgram()
glAttachShader(shader_program, vertex_shader)
glAttachShader(shader_program, fragment_shader)
glLinkProgram(shader_program)

# Check for linking errors
# glGetProgramiv(shader_program, GL_LINK_STATUS, success)
# if success[] == GL_FALSE
#     error("ERROR::SHADER::PROGRAM::LINKING_FAILED")
# end

# Delete the shaders as they're linked into our program now and no longer necessary
glDeleteShader(vertex_shader)
glDeleteShader(fragment_shader)

# Generate and bind a Vertex Array Object
vao = Ref(GLuint(0))
glGenVertexArrays(1, vao)
glBindVertexArray(0)
glBindVertexArray(vao[])

# Generate and bind a Vertex Buffer Object
vbo = Ref(GLuint(0))
glGenBuffers(1, vbo)
glBindBuffer(GL_ARRAY_BUFFER, 0)
glBindBuffer(GL_ARRAY_BUFFER, vbo[])
glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW)

# Generate and bind an Element Buffer Object
ebo = Ref(GLuint(0))
glGenBuffers(1, ebo)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo[])
glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW)

# Set vertex attribute pointers
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Float32), Ptr{Nothing}(0))
glEnableVertexAttribArray(0)

# Unbind the VBO (the VAO will remember the settings)
glBindBuffer(GL_ARRAY_BUFFER, 0)
glBindVertexArray(0)
glBindVertexArray(vao[])
# Function to render the scene
function render()
    glClear(GL_COLOR_BUFFER_BIT)

    # Use the shader program
    glUseProgram(shader_program)

    # Use the VAO
    glBindVertexArray(vao[])

    # Draw the lines
    glDrawElements(GL_LINES, 4, GL_UNSIGNED_INT, C_NULL)

    # Unbind the VAO
    glBindVertexArray(0)
end

# Main loop
render()
GLFW.SwapBuffers(window)