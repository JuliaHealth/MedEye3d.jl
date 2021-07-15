
# Here, we illustrate a pure ModernGL implementation of some polygon drawing
using ModernGL, GeometryTypes, GLFW
include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/display/GLFW/modernGL/ModernGlUtil.jl")
include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/display/GLFW/modernGL/basicFunctions.jl")
include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/display/GLFW/modernGL/shaders.jl")
include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/display/GLFW/modernGL/squarePoints.jl")



# Create the window. This sets all the hints and makes the context current.
window = initializeWindow()

# The shaders 
vertex_shader = createVertexShader()
fragment_shader = createFragmentShader()




# Connect the shaders by combining them into a program
shader_program = glCreateProgram()
glAttachShader(shader_program, vertex_shader)
glAttachShader(shader_program, fragment_shader)
#glBindFragDataLocation(shader_program, 0, "outColor") # optional

glLinkProgram(shader_program)
glUseProgram(shader_program)



###########buffers

#create vertex buffer
createVertexBuffer()

# Create the Vertex Buffer Objects (VBO)

vbo = createDAtaBuffer(vertices)

# Create the Element Buffer Object (EBO)
ebo = createElementBuffer(elements)


############ how data should be read from data buffer



typee = Float32

# position attribute
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(typee), C_NULL);
glEnableVertexAttribArray(0);
# color attribute
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(typee),  Ptr{Nothing}(3 * sizeof(typee)));
glEnableVertexAttribArray(1);
# texture coord attribute
glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(typee),  Ptr{Nothing}(6 * sizeof(typee)));
glEnableVertexAttribArray(2);



# glVertexAttribPointer(2, 2,
#                       GL_FLOAT, GL_FALSE, 0, C_NULL)

# glEnableVertexAttribArray(2)

# width = 5;
# height = 5;

# texture= createTexture(createData(width,height),width,height)


# glBindTexture(GL_TEXTURE_2D, texture[]);









# Draw while waiting for a close event
mainRenderingLoop(window)