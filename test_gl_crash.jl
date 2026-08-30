using ModernGL, GLFW

GLFW.Init()
window = GLFW.CreateWindow(640, 480, "Test")
GLFW.MakeContextCurrent(window)

vao = Ref(GLuint(0))
glGenVertexArrays(1, vao)
glBindVertexArray(vao[])

vbo = Ref(GLuint(0))
glGenBuffers(1, vbo)
glBindBuffer(GL_ARRAY_BUFFER, vbo[])
w_res = zeros(Float32, 32)
glBufferData(GL_ARRAY_BUFFER, sizeof(w_res), w_res, GL_STATIC_DRAW)

ebo = Ref(GLuint(0))
glGenBuffers(1, ebo)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo[])
elements = UInt32[0, 1, 2, 2, 3, 0]
glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW)

glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(Float32), C_NULL)
glEnableVertexAttribArray(0)

# Simulate Shader
vertex_shader_source = """
#version 330 core
layout (location = 0) in vec3 aPos;
void main()
{
    gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
}
"""
fragment_shader_source = """
#version 330 core
out vec4 FragColor;
void main()
{
    FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
}
"""
vertex_shader = glCreateShader(GL_VERTEX_SHADER)
glShaderSource(vertex_shader, 1, Ptr{UInt8}[pointer(vertex_shader_source)], C_NULL)
glCompileShader(vertex_shader)

fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
glShaderSource(fragment_shader, 1, Ptr{UInt8}[pointer(fragment_shader_source)], C_NULL)
glCompileShader(fragment_shader)

shader_program = glCreateProgram()
glAttachShader(shader_program, vertex_shader)
glAttachShader(shader_program, fragment_shader)
glLinkProgram(shader_program)
glUseProgram(shader_program)

glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, C_NULL)
GLFW.SwapBuffers(window)
println("Success!")
