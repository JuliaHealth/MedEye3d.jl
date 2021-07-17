using GeometryTypes: maximum, minimum
using BenchmarkTools
# Here, we illustrate a pure ModernGL implementation of some polygon drawing
using ModernGL, GeometryTypes, GLFW



include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/display/GLFW/modernGL/ModernGlUtil.jl")
include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/display/GLFW/modernGL/basicFunctions.jl")
include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/display/GLFW/modernGL/shaders.jl")
include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/display/GLFW/modernGL/squarePoints.jl")


werePreviousTexture = false

exampleDat = getExample()

#exampleDat = getExampleLabels()

exampleSlice = exampleDat[40,:,:]
exampleSliceReduced = reduce(vcat,exampleSlice)
minn = abs(minimum(exampleSliceReduced))

exampleSliceReduced = Float32.(reduce(vcat,exampleSlice))./(maximum(exampleSlice) - minimum(exampleSlice)).+1
# exampleSliceReduced= exampleSliceReduced./2
width = size(exampleSlice)[1]
height = size(exampleSlice)[2]






# Create the window. This sets all the hints and makes the context current.
window = initializeWindow()

# The shaders 
vertex_shader = createVertexShader()
fragment_shader = createFragmentShader()

#GLFW.DestroyWindow(window)



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

encodeDataFromDataBuffer()


# Draw while waiting for a close event
#mainRenderingLoop(window, width, height)

glClear()
# Pulse the background blue
glClearColor(0.0, 0.0, 0.1 , 1.0)
#glClear(GL_COLOR_BUFFER_BIT)
# Draw our triangle

if(werePreviousTexture)
    glDeleteTextures(1,previousTexture)
end
previousTexture= createTexture(exampleSliceReduced,width,height)



glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, C_NULL)

# Swap front and back buffers
GLFW.SwapBuffers(window)



try
	while !GLFW.WindowShouldClose(window)
	
        GLFW.PollEvents()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  	end
finally
	GLFW.DestroyWindow(window)
end