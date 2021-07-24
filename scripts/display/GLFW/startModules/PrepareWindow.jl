using DrWatson
@quickactivate "Probabilistic medical segmentation"

module PrepareWindow

export basicRender
export updateTexture
export displayAll


using ModernGL, GeometryTypes, GLFW

dirToWorkerNumbs = DrWatson.scriptsdir("mainPipeline","processesDefinitions","workerNumbers.jl")
include(dirToWorkerNumbs)

include(DrWatson.scriptsdir("display","GLFW","modernGL","ModernGlUtil.jl"))
include(DrWatson.scriptsdir("display","GLFW","modernGL","basicFunctions.jl"))
include(DrWatson.scriptsdir("display","GLFW","modernGL","shaders.jl"))
include(DrWatson.scriptsdir("display","GLFW","modernGL","squarePoints.jl"))
include(DrWatson.scriptsdir("display","GLFW","modernGL","textureManag.jl"))

using Main.BasicOpenGlConfigure
using Main.workerNumbers



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
		 # Poll for and process events
		  GLFW.PollEvents()
			end
		end
		end
	schedule(t)


return (window,vertex_shader,fragment_shader ,shader_program)

end# displayAll






end #PreperWindow