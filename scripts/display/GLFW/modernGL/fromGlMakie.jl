
using DrWatson
@quickactivate "Probabilistic medical segmentation"

import GLFW
using ModernGL
# using SharedArrays
include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/display/GLFW/modernGL/ModernGlUtil.jl")
include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/display/GLFW/modernGL/basicFunctions.jl")

window = initializeWindow()

# The data for our rectangle
data = Point{2,Float32}[(-0.5,  0.5),     # top-left
( 0.5,  0.5),     # top-right
( 0.5, -0.5),     # bottom-right
(-0.5, -0.5)]  


indicies = Face{3,UInt32}[(0,1,2),          # the first triangle
(2,3,0)]          # the second triangle


# a way to pass data into GPU
vbo = createDAtaBuffer(data)


#controlling elements in order to draw multiple elements - in this case GL_TRIANGLES

ebo = createElementBuffer(indicies)


# Generate a vertex array and array buffer for our data
createVertexBuffer()



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


vertexShader = createShader(vsh, GL_VERTEX_SHADER)
fragmentShader = createShader(fsh, GL_FRAGMENT_SHADER)
#connecting shaders to create Open Gl program and using it 
program = createShaderProgram(vertexShader, fragmentShader)
glUseProgram(program)
positionAttribute = glGetAttribLocation(program, "position");
glEnableVertexAttribArray(positionAttribute)



#showing how openGL should read data from buffer in GPU
#glVertexAttribSetting(positionAttribute)
glVertexAttribPointer(positionAttribute, 2, GL_FLOAT, false, 0, C_NULL)



mainRenderingLoop(window)