
module sv_shaders_etc
using ModernGL, GeometryTypes, GLFW
export for_sv_shader
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


function for_sv_shader(VAO_old)
  
    line_positions = Float32.([
        -0.5, -0.5, 0.0,     1.0, 1.0, 0.0,      0.0, 0.0,   # start point (yellow)
        0.5, 0.5, 0.0,       1.0, 1.0, 0.0,      0.0, 0.0    # end point (yellow)
    ])

    # line_VAO = Ref{GLuint}(0)
    line_VBO = Ref{GLuint}(0)
    # glGenVertexArrays(1, line_VAO)
    glGenBuffers(1, line_VBO)

    # glBindVertexArray(line_VAO[])

    glBindBuffer(GL_ARRAY_BUFFER, line_VBO[])
    glBufferData(GL_ARRAY_BUFFER, sizeof(line_positions), line_positions, GL_STATIC_DRAW)

    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # glBindVertexArray(line_VAO[])
    # glDrawArrays(GL_LINES, 0, 2)
    # glBindVertexArray(0)


    return 1,2,3
  end
  
end  