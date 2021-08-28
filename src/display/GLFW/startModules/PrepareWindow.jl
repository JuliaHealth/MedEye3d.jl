module PrepareWindow

export displayAll,createAndInitShaderProgram

using ModernGL, GeometryTypes, GLFW
using  ..PrepareWindowHelpers
using   ..OpenGLDisplayUtils,  ..DataStructs,Logging
using  ..ShadersAndVerticies,  ..ForDisplayStructs, ..ShadersAndVerticiesForText,  ..ModernGlUtil






"""
preparing all for displaying the images and responding to mouse and keyboard input
	listOfTexturesToCreate- list of texture specifications needed to for example create optimal shader
	calcDimsStruct - holds important data about verticies, textures dimensions etc.
"""
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
	
	masks = filter(textSpec-> !textSpec.isMainImage ,listOfTexturesToCreate)
	someExampleMask = masks[1]
	someExampleMaskB = masks[2]
	@info "masks set for subtraction $(someExampleMask.name)" someExampleMaskB.name
	fragment_shader_main,shader_program= createAndInitShaderProgram(vertex_shader,listOfTexturesToCreate,someExampleMask,someExampleMaskB,gslsStr  )
	
	glUseProgram(shader_program)
	
##for control of text display
	fragment_shader_words = ShadersAndVerticiesForText.createFragmentShader(gslsStr)
	shader_program_words = glCreateProgram()
	glAttachShader(shader_program_words, fragment_shader_words)
	glAttachShader(shader_program_words, vertex_shader)

	
	vbo_words = Ref(GLuint(1))   # initial value is irrelevant, just allocate space
    glGenBuffers(1, vbo_words)
##for control of text display
  

###########buffers
	#create vertex buffer
	createVertexBuffer()
	# Create the Vertex Buffer Objects (VBO)
	vbo = createDAtaBuffer(calcDimsStruct.mainImageQuadVert)

	# Create the Element Buffer Object (EBO)
	ebo = createElementBuffer( ShadersAndVerticies.elements)
	############ how data should be read from data buffer
	encodeDataFromDataBuffer()
	#capturing The data from GLFW
	controllWindowInput(window)

#loop that enables reacting to mouse and keyboards inputs  so every 0.1 seconds it will check GLFW weather any new events happened	
	t = @task begin;
		while(!GLFW.WindowShouldClose(window))
		sleep(0.005);
		if(!stopListening[])
		 # Poll for and process events
		  GLFW.PollEvents()
			end
		end
		end
	schedule(t)


return (window,vertex_shader,fragment_shader_main ,shader_program,stopListening,vbo,ebo,fragment_shader_words,vbo_words,shader_program_words,gslsStr)

end# displayAll


"""
On the basis of information from listOfTexturesToCreate it creates specialized shader program
"""
function createAndInitShaderProgram(vertex_shader::UInt32
	,listOfTexturesToCreate::Vector{TextureSpec}
	,maskToSubtractFrom::TextureSpec
	,maskWeAreSubtracting ::TextureSpec
	,gslsStr::String)::Tuple{UInt32, UInt32}
			fragment_shader = createFragmentShader(gslsStr,listOfTexturesToCreate,maskToSubtractFrom,maskWeAreSubtracting)
			shader_program = glCreateProgram()
			glAttachShader(shader_program, fragment_shader)
			glAttachShader(shader_program, vertex_shader)
			glLinkProgram(shader_program)
return fragment_shader,shader_program

end#createShaderProgram



end #PreperWindow