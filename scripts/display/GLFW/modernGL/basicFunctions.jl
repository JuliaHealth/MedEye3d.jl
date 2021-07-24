using DrWatson
@quickactivate "Probabilistic medical segmentation"


using GLFW
using ModernGL
```@doc
data is loaded into a buffer which passes it into thw GPU for futher processing 
    - here the data is just passing the positions of verticies
    GL_STREAM_DRAW the data is set only once and used by the GPU at most a few times. 
    GL_STATIC_DRAW the data is set only once and used many times. 
    GL_DYNAMIC_DRAW the data is changed a lot and used many times.
    ```
function createDAtaBuffer(positions)
    vbo = Ref(GLuint(0))   # initial value is irrelevant, just allocate space
    glGenBuffers(1, vbo)
    glBindBuffer(GL_ARRAY_BUFFER, vbo[])
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW)
return vbo
end


```@doc
Similar to the VBO we bind the EBO and copy the indices into the buffer with glBufferData. 
    ```
function createElementBuffer(elements)
    ebo = Ref(GLuint(0))
    glGenBuffers(1, ebo)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo[])
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW)
return ebo
end



```@doc
vertex buffer keeping things simpler
    ```
function createVertexBuffer()
    vao = Ref(GLuint(0))
    glGenVertexArrays(1, vao)
    glBindVertexArray(vao[])
end


glVertexAttribSettingStr= """
showing how openGL should read data from buffer in GPU
in case of code like below it would mean:

first parameter specifies which vertex attribute we want to configure Remember that we specified the location of the position vertex attribute in the vertex shader
next argument specifies the size of the vertex attribute. The vertex attribute is a vec2 so it is composed of 2 values. 
The third argument specifies the type of the data which is GL_FLOAT
next argument specifies if we want the data to be normalized. If we’re inputting integer data types like int, byte and we’ve set this to GL_TRUE
    The fifth argument is known as the stride and tells us the space between consecutive vertex attributes. Since the next set of position data is
     located exactly 2 times the size of a float we could’ve also specified the stride as 0 to let OpenGL determine the stride 
    he last parameter is of type void* and thus requires that weird cast. This is the offset of where the position data begins in the buffer.

glVertexAttribPointer positionAttribute, 2, GL_FLOAT, false, 0, C_NULL

    The position data is stored as 32-bit  so 4 byte floating point values. 
    Each position is composed of 2 of those values. 
    """
    @doc glVertexAttribSettingStr
function glVertexAttribSetting(positionAttribute)
    glVertexAttribPointer(positionAttribute, 2, GL_FLOAT, false, 0, C_NULL)

end



```@doc
how data should be read from data buffer
    ```
function encodeDataFromDataBuffer()
    typee = Float32

    # position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(typee), C_NULL);
    glEnableVertexAttribArray(0);
    # color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(typee),  Ptr{Nothing}(3 * sizeof(typee)));
    glEnableVertexAttribArray(1);
    # texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(typee),  Ptr{Nothing}(6 * sizeof(typee)));
    glEnableVertexAttribArray(2);

end

```@doc
loop that collects any events from openGL in case 
of animations it can be  rendering loop
    ```
function sipmpleeventLoop(window)
        try
            while !GLFW.WindowShouldClose(window)
                glClear()
              
               # Poll for and process events
                GLFW.PollEvents()
        
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              end
        finally
            GLFW.DestroyWindow(window)
        end


end


controllWindowInputDoc = """
it will generally be invoked on GLFW.PollEvents()  in event loop and now depending on 
what will be pressed or clicked it will lead to diffrent actions
"""
@doc controllWindowInputDoc
function controllWindowInput(window)
	GLFW.SetWindowCloseCallback(window, (_) -> GLFW.DestroyWindow(window))
	GLFW.SetMouseButtonCallback(window, (_, button, action, mods) -> println("$button $action"))

# Input callbacks
GLFW.SetKeyCallback(window, (_, key, scancode, action, mods) -> begin
	name = GLFW.GetKeyName(key, scancode)
	if name == nothing
		println("scancode $scancode ", action)
	else
		println("key $name ", action)
	end
end)


end
controllScrollingDoc = """
controll swithing the slices while scrolling
"""
@doc controllScrollingDoc
function controllScrolling( )

pass
end





