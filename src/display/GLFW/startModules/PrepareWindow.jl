module PrepareWindow

using Base.Threads, ModernGL, GeometryTypes, GLFW, Logging
using ..PrepareWindowHelpers, ..OpenGLDisplayUtils, ..DataStructs, ..ShadersAndVerticies, ..ForDisplayStructs, ..ShadersAndVerticiesForText, ..ModernGlUtil

export displayAll, createAndInitShaderProgram


"""
preparing all for displaying the images and responding to mouse and keyboard input
	listOfTexturesToCreate- list of texture specifications needed to for example create optimal shader
	calcDimsStruct - holds important data about verticies, textures dimensions etc.
"""
function displayAll(calcDimsStruct::CalcDimsStruct)

    if (nthreads(:interactive) == 0)
        @warn "MedEye3D is running with standard threading (interactive thread pool not allocated). For optimal asynchronous responsiveness, consider starting with JULIA_NUM_THREADS=auto,1."
    end #if

    if (Threads.nthreads() == 1)
        @info "Running on single Julia thread. Multithreading can be enabled via JULIA_NUM_THREADS."
    end
    # Create the window. This sets all the hints and makes the context current.


    window = initializeWindow(calcDimsStruct.windowWidth, calcDimsStruct.windowHeight)

    # The shaders
    createcontextinfo()
    gslsStr = get_glsl_version_string()


    vertex_shader = createVertexShader(gslsStr)

    # masks = filter(textSpec -> !textSpec.isMainImage, listOfTexturesToCreate)
    # someExampleMask = masks[begin]
    # someExampleMaskB = masks[end]
    # @info "masks set for subtraction $(someExampleMask.name)" someExampleMaskB.name
    # fragment_shader_main, shader_program = createAndInitShaderProgram(vertex_shader, listOfTexturesToCreate, someExampleMask, someExampleMaskB, gslsStr)
    # fragment_shader_main, shader_program = createAndInitShaderProgram(vertex_shader, listOfTexturesToCreate, gslsStr)


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
    vao = createVertexBuffer()
    # Create the Vertex Buffer Objects (VBO)
    # vbo = createDAtaBuffer(calcDimsStruct.mainImageQuadVert)

    # Create the Element Buffer Object (EBO)
    ebo = createElementBuffer(ShadersAndVerticies.elements)
    ############ how data should be read from data buffer
    encodeDataFromDataBuffer()
    #capturing The data from GLFW
    controllWindowInput(window)

    #loop that enables reacting to mouse and keyboards inputs  so every 0.1 seconds it will check GLFW weather any new events happened
    pollingTask, stopChannel = createPollingTask(window)
    schedule(pollingTask)

    return (window, vertex_shader, vao, ebo, fragment_shader_words, vbo_words, shader_program_words, gslsStr, stopChannel)

end# displayAll


"""
On the basis of information from listOfTexturesToCreate it creates specialized shader program
"""
function createAndInitShaderProgram(vertex_shader::UInt32, listOfTexturesToCreate::Vector{TextureSpec{Float32}}, gslsStr::String, calcDimsStruct::CalcDimsStruct, num)

    fragment_shader = nothing
    if num == 1
        fragment_shader = ShadersAndVerticies.createFragmentShader(gslsStr, listOfTexturesToCreate, "green")
    else
        fragment_shader = ShadersAndVerticies.createFragmentShader(gslsStr, listOfTexturesToCreate, "red")

    end

    shader_program = glCreateProgram()
    glAttachShader(shader_program, fragment_shader)
    glAttachShader(shader_program, vertex_shader)
    glLinkProgram(shader_program)
    vbo = createDAtaBuffer(calcDimsStruct.mainImageQuadVert)
    glUseProgram(shader_program)

    return (fragment_shader, shader_program, vbo)

end#createShaderProgram


"""
Polling task creation — monitors window close state only.
GLFW.PollEvents() is NOT called here; it is called exclusively by the
Makie renderloop's pollevents() to ensure proper event ordering for
button click detection (mouseposition must fire before mousebutton).
"""
function createPollingTask(window::GLFW.Window)
    stopChannel = Channel{Bool}(1)
    
    t = @task begin
        try
            while true
                # Check for stop signal first
                if isready(stopChannel)
                    take!(stopChannel)
                    break
                end
                
                # Check if window should close
                if GLFW.WindowShouldClose(window)
                    break
                end
                
                sleep(0.008)
            end
        catch e
            @warn "GLFW polling task error: $e" exception=(e, catch_backtrace())
        finally
            @info "GLFW polling task ended"
        end
    end
    
    return (t, stopChannel)
end



end #PreperWindow
