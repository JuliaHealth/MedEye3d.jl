using DrWatson
@quickactivate "Probabilistic medical segmentation"

include(DrWatson.scriptsdir("display","GLFW","startModules","PrepareWindowHelpers.jl"))
include(DrWatson.scriptsdir("display","GLFW","modernGL","OpenGLDisplayUtils.jl"))
include(DrWatson.scriptsdir("display","GLFW","startModules","ShadersAndVerticies.jl"))



module PrepareWindow
using DrWatson
@quickactivate "Probabilistic medical segmentation"

export displayAll

using ModernGL, GeometryTypes, GLFW
using Main.PrepareWindowHelpers
include(DrWatson.scriptsdir("display","GLFW","startModules","ModernGlUtil.jl"))
using  Main.OpenGLDisplayUtils
using Main.ShadersAndVerticies


# atomic variable that is enabling stopping async loop of event listening in order to enable othe actions with GLFW context
stopListening = Threads.Atomic{Bool}(0)
stopListening[]=false



displayAllStr="""
preparing all for displaying the images and responding to mouse and keyboard input
"""
@doc displayAllStr
function displayAll()
	if(Threads.nthreads()==1) 
		println("increase number of available threads look into https://docs.julialang.org/en/v1/manual/multi-threading/  or modify for example in vs code extension")
    end
    
    # Create the window. This sets all the hints and makes the context current.
	window = initializeWindow()
    
   	# The shaders 
	println(createcontextinfo())
	gslsStr = get_glsl_version_string()
	vertex_shader = createVertexShader(gslsStr)
	fragment_shader = createFragmentShader(gslsStr)


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

	vbo = createDAtaBuffer(Main.ShadersAndVerticies.vertices)

	# Create the Element Buffer Object (EBO)
	ebo = createElementBuffer(Main.ShadersAndVerticies.elements)


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