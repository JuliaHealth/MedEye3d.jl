using DrWatson
@quickactivate "Probabilistic medical segmentation"

using GeometryTypes: maximum, minimum
using BenchmarkTools
# Here, we illustrate a pure ModernGL implementation of some polygon drawing
using ModernGL, GeometryTypes, GLFW

dirToWorkerNumbs = DrWatson.scriptsdir("mainPipeline","processesDefinitions","workerNumbers.jl")
include(dirToWorkerNumbs)
using Main.workerNumbers

include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/display/GLFW/modernGL/ModernGlUtil.jl")
include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/display/GLFW/modernGL/basicFunctions.jl")
include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/display/GLFW/modernGL/shaders.jl")
include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/display/GLFW/modernGL/squarePoints.jl")
include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/display/GLFW/modernGL/textureManag.jl")



"""
preparing all for displaying the images and responding to mouse and keyboard input
"""
function displayAll(stopListening)
	if(Threads.nthreads()==1) 
		println("increase number of available threads look into https://docs.julialang.org/en/v1/manual/multi-threading/  or modify for example in vs code extension")
    end
    
    # Create the window. This sets all the hints and makes the context current.
	window = initializeWindow()
    
   	# The shaders 
	vertex_shader = createVertexShader()
	fragment_shader = createFragmentShader()


	# Connect the shaders by combining them into a program
	shader_program = glCreateProgram()
	glAttachShader(shader_program, vertex_shader)
	glAttachShader(shader_program, fragment_shader)

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

#capturing The data from GLFW
	controllWindowInput(window)

#loop that enables reacting to mouse and keyboards inputs  so every 0.1 seconds it will check GLFW weather any new events happened	
	t = @task begin;
		while(!GLFW.WindowShouldClose(window))
		sleep(0.1);
		if(!stopListening[])
		 glClear()             
		 # Poll for and process events
		  GLFW.PollEvents()
			end
		end
		end
	schedule(t)



	#GLFW.DestroyWindow(window)
return (window,vertex_shader,fragment_shader ,shader_program)

end# displayAll



basicRenderDoc = """
As most functions will deal with just addind the quad to the screen 
and swapping buffers
"""
@doc basicRenderDoc
function basicRender()
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, C_NULL)
	# Swap front and back buffers
	GLFW.SwapBuffers(window)

end


"""
uploading data to given texture; activating appropriate 
"""
function updateTexture(juliaDataTyp::Type{juliaDataType},width,height,data, textureId,stopListening,pboId, DATA_SIZE,GlNumbType )where{juliaDataType}

	glBindTexture(GL_TEXTURE_2D, textureId[]); 
	glTexSubImage2D(GL_TEXTURE_2D,0,0,0, width, height, GL_RED_INTEGER, GlNumbType, data);

end


