using ModernGL
using GLFW,HDF5
using GeometryTypes
function initializeWindow(windowWidth::Int, windowHeight::Int)
    GLFW.Init()
    window = GLFW.CreateWindow(windowWidth, windowHeight, "Segmentation Visualization")
    GLFW.MakeContextCurrent(window)
    GLFW.ShowWindow(window)
    GLFW.SetWindowSize(window, windowWidth, windowHeight)
    glViewport(0, 0, windowWidth, windowHeight)
    glDisable(GL_LIGHTING)
    glEnable(GL_TEXTURE_2D)
    return window
end


function createShader(source, typ)
    shader = glCreateShader(typ)::GLuint
    glShaderSource(shader, 1, convert(Ptr{UInt8}, pointer([convert(Ptr{GLchar}, pointer(source))])), C_NULL)
    glCompileShader(shader)
    return shader
end

function createAndInitShaderProgram(vertex_shader::UInt32, fragment_shader_source)::Tuple{UInt32,UInt32}
    fragment_shader = createShader(fragment_shader_source, GL_FRAGMENT_SHADER)
    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)
    return fragment_shader, shader_program
end



function createTexture(juliaDataType::Type{juliaDataTyp}, width::Int32, height::Int32, GL_RType::UInt32=GL_R32F, OpGlType=GL_FLOAT) where {juliaDataTyp}
    texture = Ref(GLuint(0))
    glGenTextures(1, texture)
    glBindTexture(GL_TEXTURE_2D, texture[])
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0)
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RType, width, height)
    return texture
end


function getProperGL_TEXTURE(index::Int)::UInt32
    return eval(Meta.parse("GL_TEXTURE$(index)"))
end




function initialize_window_etc(window_width,window_height, texture_width, texture_height)
    window = initializeWindow(window_width, window_height)

    # Vertex data for a rectangle
    rectangle_vertices = Float32[
        1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,   # top right
        1.0, -1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,   # bottom right
        -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  # bottom left
        -1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0    # top left
    ]

    # Indices for the rectangle
    rectangle_indices = UInt32[
        0, 1, 3,  # First triangle
        1, 2, 3   # Second triangle
    ]

    # # Vertex data for lines
    # line_vertices = Float32[
    #     0.1, 0.0, 0.0,  # top right
    #     -0.1, 0.0, 0.0,  # bottom right
    #     0.0, -0.1, 0.0,  # bottom left
    #     0.0, 0.1, 0.0   # top left
    # ]

    # # Indices for drawing lines
    # line_indices = UInt32[
    #     0, 1,  # Line from top right to bottom right
    #     2, 3   # Line from bottom left to top left
    # ]

    # Vertex shader source code
    vertex_shader_source = """
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aColor;
    layout (location = 2) in vec2 aTexCoord;
    out vec3 ourColor;
    out vec2 TexCoord;
    void main()
    {
        gl_Position = vec4(aPos, 1.0);
        ourColor = aColor;
        TexCoord = aTexCoord;
    }
    """

    # Fragment shader source code
    fragment_shader_source = """
    #version 330 core
    out vec4 FragColor;
    in vec3 ourColor;
    in vec2 TexCoord;
    uniform sampler2D CTIm;

    uniform vec4 CTImColorMask = vec4(1.0, 1.0, 1.0, 1.0); // controlling colors
    uniform int CTImisVisible = 1; // controlling visibility

    uniform float CTImminValue = 0.0; // minimum possible value set in configuration
    uniform float CTImmaxValue = 100.0; // maximum possible value set in configuration
    uniform float CTImValueRange = 100.0; // range of possible values calculated from above
    uniform float CTImmaskContribution = 1.0; // controls contribution of mask to output color

    float changeClip(float min, float max, float value, float color, float range) {
        if (value < min) {
            return color * (min / range);
        } else if (value > max) {
            return color * (max / range);
        } else {
            return color * (value / range);
        }
    }

    void main() {
        vec4 texColor = texture(CTIm, TexCoord);
        float CTImRes = texColor.r; // Assuming the texture is in red channel
    FragColor = vec4((  changeClip(CTImminValue,CTImmaxValue,CTImRes,CTImColorMask.r,CTImValueRange)  + 0.0) ,
                        (  changeClip(CTImminValue,CTImmaxValue,CTImRes,CTImColorMask.g,CTImValueRange)  + 0.0),
                        (  changeClip(CTImminValue,CTImmaxValue,CTImRes,CTImColorMask.b,CTImValueRange)  + 0.0) ,
                        1.0);
    }
    """
    #    FragColor = texture(CTIm, TexCoord);

    fragment_shader_source_line = """
    #version 330 core
    out vec4 FragColor;




    void main()
    {
        FragColor = vec4(1.0, 1.0, 0.0, 1.0); // Yellow color
    }
    """


    vertex_shader = createShader(vertex_shader_source, GL_VERTEX_SHADER)
    fragment_shader_main, rectangle_shader_program = createAndInitShaderProgram(vertex_shader, fragment_shader_source)
    n = "CTIm"
    samplerName = n
    samplerRef=glGetUniformLocation(rectangle_shader_program,samplerName)
    glUniform1i(samplerRef, 0)


    fragment_shader_line, line_shader_program = createAndInitShaderProgram(vertex_shader, fragment_shader_source_line)

    # Generate and bind VAO for rectangle
    rectangle_vao = Ref(GLuint(0))
    glGenVertexArrays(1, rectangle_vao)
    glBindVertexArray(rectangle_vao[])

    # Generate and bind VBO for rectangle
    rectangle_vbo = Ref(GLuint(0))
    glGenBuffers(1, rectangle_vbo)
    glBindBuffer(GL_ARRAY_BUFFER, rectangle_vbo[])
    glBufferData(GL_ARRAY_BUFFER, sizeof(rectangle_vertices), rectangle_vertices, GL_STATIC_DRAW)

    # Generate and bind EBO for rectangle
    rectangle_ebo = Ref(GLuint(0))
    glGenBuffers(1, rectangle_ebo)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, rectangle_ebo[])
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(rectangle_indices), rectangle_indices, GL_STATIC_DRAW)

    # Set vertex attribute pointers for rectangle
    typee = Float32

    # position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(typee), C_NULL)
    glEnableVertexAttribArray(0)
    # color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(typee), Ptr{Nothing}(3 * sizeof(typee)))
    glEnableVertexAttribArray(1)
    # texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(typee), Ptr{Nothing}(6 * sizeof(typee)))
    glEnableVertexAttribArray(2)

    # Unbind the VAO for rectangle
    glBindVertexArray(0)

    textUreId = createTexture(Float32, Int32(texture_width), Int32(texture_height), GL_R32F, GL_FLOAT)
    return window, rectangle_vao, rectangle_vbo, rectangle_ebo, rectangle_shader_program, line_shader_program, textUreId
end





function initialize_lines(line_vertices, line_indices)
    # Generate and bind VAO for lines
    line_vao = Ref(GLuint(0))
    glGenVertexArrays(1, line_vao)
    glBindVertexArray(line_vao[])

    # Generate and bind VBO for lines
    line_vbo = Ref(GLuint(0))
    glGenBuffers(1, line_vbo)
    glBindBuffer(GL_ARRAY_BUFFER, line_vbo[])
    glBufferData(GL_ARRAY_BUFFER, sizeof(line_vertices), line_vertices, GL_STATIC_DRAW)

    # Generate and bind EBO for lines
    line_ebo = Ref(GLuint(0))
    glGenBuffers(1, line_ebo)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, line_ebo[])
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(line_indices), line_indices, GL_STATIC_DRAW)

    # Set vertex attribute pointers for lines
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Float32), Ptr{Nothing}(0))
    glEnableVertexAttribArray(0)

    # Unbind the VAO for lines
    glBindVertexArray(0)
    return line_vao, line_vbo, line_ebo
end    