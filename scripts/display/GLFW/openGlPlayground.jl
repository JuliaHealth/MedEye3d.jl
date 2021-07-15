
# Here, we illustrate a pure ModernGL implementation of some polygon drawing
using ModernGL, GeometryTypes, GLFW
include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/display/GLFW/modernGL/ModernGlUtil.jl")
include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/display/GLFW/modernGL/basicFunctions.jl")



# Create the window. This sets all the hints and makes the context current.
window = initializeWindow()

# The shaders. Here we do everything manually, but life will get
# easier with GLAbstraction. See drawing_polygons5.jl for such an
# implementation.


# Create and initialize shaders
const vsh = """
$(get_glsl_version_string())
layout (location = 0) in vec3 aPos;
in vec2 position;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
"""



const fsh = """
$(get_glsl_version_string())
out vec4 outColor;
void main() {
    outColor = vec4(1.0, 0.5, 1.0, 1.0);
}
"""

vertex_shader = createShader(vsh, GL_VERTEX_SHADER)
fragment_shader = createShader(fsh, GL_FRAGMENT_SHADER)


# Connect the shaders by combining them into a program
shader_program = glCreateProgram()
glAttachShader(shader_program, vertex_shader)
glAttachShader(shader_program, fragment_shader)
glBindFragDataLocation(shader_program, 0, "outColor") # optional

glLinkProgram(shader_program)
glUseProgram(shader_program)




# Now we define another geometry that we will render, a rectangle, this one with an index buffer
# The positions of the vertices in our rectangle
positions = Point{2,Float32}[(-0.5,  0.5),     # top-left
                             ( 0.5,  0.5),     # top-right
                             ( 0.5, -0.5),     # bottom-right
                             (-0.5, -0.5)]     # bottom-left

# Specify how vertices are arranged into faces
# Face{N,T} type specifies a face with N vertices, with index type
# T (you should choose UInt32), and index-offset O. If you're
# specifying faces in terms of julia's 1-based indexing, you should set
# O=0. (If you instead number the vertices starting with 0, set
# O=-1.)
elements = Face{3,UInt32}[(0,1,2),          # the first triangle
                          (2,3,0)]          # the second triangle


createVertexBuffer()

# Create the Vertex Buffer Objects (VBO)

createDAtaBuffer(positions)


# Link vertex data to attributes
pos_attribute = glGetAttribLocation(shader_program, "position")
glEnableVertexAttribArray(pos_attribute)
glVertexAttribPointer(pos_attribute, 2,
                      GL_FLOAT, GL_FALSE, 0, C_NULL)


# Create the Element Buffer Object (EBO)
ebo = createElementBuffer(elements)

# Draw while waiting for a close event
mainRenderingLoop(window)