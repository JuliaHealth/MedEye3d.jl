using ModernGL
using GLFW
using Random

# Initialize GLFW and create a window
function init_glfw()
    if !GLFW.Init()
        error("Failed to initialize GLFW")
    end

    window = GLFW.CreateWindow(800, 600, "OpenGL Rectangle with Texture")
    if window == C_NULL
        error("Failed to create GLFW window")
    end

    GLFW.MakeContextCurrent(window)

    return window
end

# Define the vertex and fragment shaders
const vertex_shader_source = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
void main()
{
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
"""

    # FragColor = FragColor = vec4(texture(texture1, TexCoord).r,1.0,0.0,1.0);

const fragment_shader_source = """
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
uniform sampler2D texture1;
void main()
{
    FragColor = FragColor = vec4(1.0,0.0,0.0,1.0);
}
"""

# Compile the shaders and create a shader program
function compile_shader(source::String, shader_type::GLenum)
    shader = glCreateShader(shader_type)
    glShaderSource(shader, 1, [source], C_NULL)
    glCompileShader(shader)
    
    # Check for compilation errors
    success = Ref{GLint}(0)
    glGetShaderiv(shader, GL_COMPILE_STATUS, success)
    if success[] == GL_FALSE
        info_log = Array{GLchar}(undef, 512)
        glGetShaderInfoLog(shader, 512, C_NULL, info_log)
        error("Shader compilation failed: ", String(info_log))
    end
    
    return shader
end

function create_shader_program(vertex_source::String, fragment_source::String)
    vertex_shader = compile_shader(vertex_source, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_source, GL_FRAGMENT_SHADER)
    
    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)
    
    # Check for linking errors
    success = Ref{GLint}(0)
    glGetProgramiv(shader_program, GL_LINK_STATUS, success)
    if success[] == GL_FALSE
        info_log = Array{GLchar}(undef, 512)
        glGetProgramInfoLog(shader_program, 512, C_NULL, info_log)
        error("Shader program linking failed: ", String(info_log))
    end
    
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    
    return shader_program
end

# Define the vertex data for a rectangle
function create_rectangle_data()
    vertices = [
        # positions       # texture coords
        0.5f0,  0.5f0, 0.0f0,  1.0f0, 1.0f0,  # top right
        0.5f0, -0.5f0, 0.0f0,  1.0f0, 0.0f0,  # bottom right
       -0.5f0, -0.5f0, 0.0f0,  0.0f0, 0.0f0,  # bottom left
       -0.5f0,  0.5f0, 0.0f0,  0.0f0, 1.0f0   # top left 
    ]

    indices = [
        0, 1, 3,  # first triangle
        1, 2, 3   # second triangle
    ]

    return vertices, indices
end

# Create and bind the VAO and VBO
function setup_buffers(vertices, indices)
    VAO = Ref{GLuint}(0)
    VBO = Ref{GLuint}(0)
    EBO = Ref{GLuint}(0)
    glGenVertexArrays(1, VAO)
    glGenBuffers(1, VBO)
    glGenBuffers(1, EBO)

    glBindVertexArray(VAO[])

    glBindBuffer(GL_ARRAY_BUFFER, VBO[])
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[])
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), C_NULL)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), Ptr{Nothing}(3 * sizeof(GLfloat)))
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    return VAO, VBO, EBO
end

# Load a random texture
function load_random_texture()
    texture = Ref{GLuint}(0)
    glGenTextures(1, texture)
    glBindTexture(GL_TEXTURE_2D, texture[])

    # Generate random texture data
    width, height = 256, 256
    data = rand(Float32, width, height)


    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RED, GL_FLOAT, collect(data))


    # glGenerateMipmap(GL_TEXTURE_2D)

    # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    return texture
end

# Render the rectangle with the texture
function render(window, shader_program, VAO, texture)
    glClearColor(0.2f0, 0.3f0, 0.3f0, 1.0f0)  # Set clear color
    while !GLFW.WindowShouldClose(window)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(shader_program)
        glBindTexture(GL_TEXTURE_2D, texture[])
        glBindVertexArray(VAO[])
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, C_NULL)
        glBindVertexArray(0)

        GLFW.SwapBuffers(window)
        GLFW.PollEvents()
    end
end

# Main function
function main()
    window = init_glfw()
    shader_program = create_shader_program(vertex_shader_source, fragment_shader_source)
    vertices, indices = create_rectangle_data()
    VAO, VBO, EBO = setup_buffers(vertices, indices)
    texture = load_random_texture()

    render(window, shader_program, VAO, texture)

    glDeleteVertexArrays(1, VAO)
    glDeleteBuffers(1, VBO)
    glDeleteBuffers(1, EBO)
    GLFW.Terminate()
end

main()