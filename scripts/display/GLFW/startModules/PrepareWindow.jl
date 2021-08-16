using DrWatson
@quickactivate "Probabilistic medical segmentation"


module PrepareWindow
using DrWatson
@quickactivate "Probabilistic medical segmentation"

export displayAll

using ModernGL, GeometryTypes, GLFW
using Main.PrepareWindowHelpers
include(DrWatson.scriptsdir("display","GLFW","startModules","ModernGlUtil.jl"))
using  Main.OpenGLDisplayUtils, Main.DataStructs
using Main.ShadersAndVerticies, Main.ForDisplayStructs,Main.ShadersAndVerticiesForText






displayAllStr="""
preparing all for displaying the images and responding to mouse and keyboard input
	windowWidth, windowHeight - initial dimensiona of GLFW window
	fractionOfMainIm - how much of width should be taken by the main image
"""
@doc displayAllStr
function displayAll(listOfTexturesToCreate::Vector{TextureSpec}
					,calcDimsStruct::CalcDimsStruct)
	# atomic variable that is enabling stopping async loop of event listening in order to enable othe actions with GLFW context
	stopListening = Threads.Atomic{Bool}(0)
	stopListening[]=false

	if(Threads.nthreads()==1) 
		println("increase number of available threads look into https://docs.julialang.org/en/v1/manual/multi-threading/  or modify for example in vs code extension")
    end
    # Create the window. This sets all the hints and makes the context current.
	window = initializeWindow(calcDimsStruct.windowWidth,calcDimsStruct.windowHeight)
    
   	# The shaders 
	println(createcontextinfo())
	gslsStr = get_glsl_version_string()


	vertex_shader = createVertexShader(gslsStr)
	fragment_shader_main = createFragmentShader(gslsStr,listOfTexturesToCreate)
	
##for control of text display
	fragment_shader_words = ShadersAndVerticiesForText.createFragmentShader(gslsStr)
	shader_program_words = glCreateProgram()
	glAttachShader(shader_program_words, fragment_shader_words)
	glAttachShader(shader_program_words, vertex_shader)

	
	vbo_words = Ref(GLuint(1))   # initial value is irrelevant, just allocate space
    glGenBuffers(1, vbo_words)
##for control of text display
  

	# Connect the shaders by combining them into a program
	shader_program = glCreateProgram()

	
	glAttachShader(shader_program, vertex_shader)
	glAttachShader(shader_program, fragment_shader_main)
	
	glLinkProgram(shader_program)
	glUseProgram(shader_program)
	
	###########buffers
	#create vertex buffer
	createVertexBuffer()
	# Create the Vertex Buffer Objects (VBO)
	vbo = createDAtaBuffer(calcDimsStruct.mainImageQuadVert)

	# Create the Element Buffer Object (EBO)
	ebo = createElementBuffer(Main.ShadersAndVerticies.elements)
	############ how data should be read from data buffer
	encodeDataFromDataBuffer()
	#capturing The data from GLFW
	controllWindowInput(window)

#loop that enables reacting to mouse and keyboards inputs  so every 0.1 seconds it will check GLFW weather any new events happened	
	t = @task begin;
		while(!GLFW.WindowShouldClose(window))
		sleep(0.001);
		if(!stopListening[])
		 # Poll for and process events
		  GLFW.PollEvents()
			end
		end
		end
	schedule(t)


return (window,vertex_shader,fragment_shader_main ,shader_program,stopListening,vbo,ebo,fragment_shader_words,vbo_words,shader_program_words)

end# displayAll






end #PreperWindow