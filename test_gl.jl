using ModernGL, GLFW

GLFW.Init()
GLFW.WindowHint(GLFW.CONTEXT_VERSION_MAJOR, 3)
GLFW.WindowHint(GLFW.CONTEXT_VERSION_MINOR, 3)
GLFW.WindowHint(GLFW.OPENGL_PROFILE, GLFW.OPENGL_CORE_PROFILE)

window = GLFW.CreateWindow(100, 100, "Test", GLFW.Monitor(C_NULL), GLFW.Window(C_NULL))
GLFW.MakeContextCurrent(window)

texture = Ref(GLuint(0))
glGenTextures(1, texture)
glBindTexture(GL_TEXTURE_2D, texture[])

glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

println("Before glTexImage2D: error=", glGetError())

# Try allocating with C_NULL
glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 128, 128, 0, GL_RED, GL_FLOAT, C_NULL)
err = glGetError()
println("After C_NULL glTexImage2D: error=", err)

tex_w = Ref{GLint}(0)
glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, tex_w)
println("Width queried: ", tex_w[])

# Try allocating with an array
data = zeros(Float32, 128, 128)
glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 128, 128, 0, GL_RED, GL_FLOAT, data)
err = glGetError()
println("After Array glTexImage2D: error=", err)

glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, tex_w)
println("Width queried: ", tex_w[])

# Try glTexStorage2D
texture2 = Ref(GLuint(0))
glGenTextures(1, texture2)
glBindTexture(GL_TEXTURE_2D, texture2[])
glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, 128, 128)
err = glGetError()
println("After glTexStorage2D: error=", err)

glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, tex_w)
println("Width queried: ", tex_w[])

GLFW.DestroyWindow(window)
