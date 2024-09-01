module PrepareWindow

using Base.Threads, ModernGL, GeometryTypes, GLFW, Logging
using ..PrepareWindowHelpers, ..OpenGLDisplayUtils, ..DataStructs, ..ShadersAndVerticies, ..ForDisplayStructs, ..ShadersAndVerticiesForText, ..ModernGlUtil

export displayAll, createAndInitShaderProgram





"""
preparing all for displaying the images and responding to mouse and keyboard input
	listOfTexturesToCreate- list of texture specifications needed to for example create optimal shader
	calcDimsStruct - holds important data about verticies, textures dimensions etc.
"""
function displayAll(listOfTexturesToCreate::Vector{TextureSpec}, calcDimsStruct::CalcDimsStruct)

    if (nthreads(:interactive) == 0)
        @error " MedEye3D above version 0.5.6 requires setting of the interactive Thread (feature available from Julia 1.9 ) one can set it in linux by enviromental variable export JULIA_NUM_THREADS=3,1 where 1 after the coma is the interactive thread and 3 is the number of the other threads available on your machine; or start julia like this julia --threads 3,1; you can also use the docker container prepared by the author from  https://github.com/jakubMitura14/MedPipe3DTutorial. . More about interactive THreads on https://docs.julialang.org/en/v1/manual/multi-threading/"
        throw(error())

    end #if


    if (Threads.nthreads() == 1)
        println("increase number of available threads look into https://docs.julialang.org/en/v1/manual/multi-threading/  or modify for example in vs code extension")
    end
    # Create the window. This sets all the hints and makes the context current.
    window = initializeWindow(calcDimsStruct.windowWidth, calcDimsStruct.windowHeight)

    # The shaders
    println(createcontextinfo())
    gslsStr = get_glsl_version_string()


    vertex_shader = createVertexShader(gslsStr)

    # masks = filter(textSpec -> !textSpec.isMainImage, listOfTexturesToCreate)
    # someExampleMask = masks[begin]
    # someExampleMaskB = masks[end]
    # @info "masks set for subtraction $(someExampleMask.name)" someExampleMaskB.name
    # fragment_shader_main, shader_program = createAndInitShaderProgram(vertex_shader, listOfTexturesToCreate, someExampleMask, someExampleMaskB, gslsStr)
    fragment_shader_main, shader_program = createAndInitShaderProgram(vertex_shader, listOfTexturesToCreate, gslsStr)

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
    ebo = createElementBuffer(ShadersAndVerticies.elements)
    ############ how data should be read from data buffer
    encodeDataFromDataBuffer()
    #capturing The data from GLFW
    controllWindowInput(window)

    #loop that enables reacting to mouse and keyboards inputs  so every 0.1 seconds it will check GLFW weather any new events happened
    t = @task begin
        while (!GLFW.WindowShouldClose(window))
            sleep(0.001)
            # Poll for and process events
            GLFW.PollEvents()
        end
    end
    schedule(t)


    return (window, vertex_shader, fragment_shader_main, shader_program, vbo, ebo, fragment_shader_words, vbo_words, shader_program_words, gslsStr)

end# displayAll


"""
On the basis of information from listOfTexturesToCreate it creates specialized shader program
"""
function createAndInitShaderProgram(vertex_shader::UInt32, listOfTexturesToCreate::Vector{TextureSpec}, gslsStr::String)::Tuple{UInt32,UInt32}
    fragment_shader = ShadersAndVerticies.createFragmentShader(gslsStr, listOfTexturesToCreate)
    shader_program = glCreateProgram()
    glAttachShader(shader_program, fragment_shader)
    glAttachShader(shader_program, vertex_shader)
    glLinkProgram(shader_program)
    return fragment_shader, shader_program

end#createShaderProgram



end #PreperWindow
