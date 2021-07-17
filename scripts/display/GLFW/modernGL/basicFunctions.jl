using DrWatson
@quickactivate "Probabilistic medical segmentation"

using GLFW


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




```@doc
main rendering loop of open gl    ```
function mainRenderingLoop(window, textureWidth, textureHeihght)
# Loop until the user closes the window
try
	while !GLFW.WindowShouldClose(window)
		glClear()
	    # Pulse the background blue
        glClearColor(0.0, 0.0, 0.1 , 1.0)
        #glClear(GL_COLOR_BUFFER_BIT)
        # Draw our triangle

        if(werePreviousTexture)
            glDeleteTextures(1,previousTexture)
        end
        previousTexture= createTexture(createData(textureWidth,textureHeihght),textureWidth,textureHeihght)
        
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, C_NULL)
        
        # Swap front and back buffers
        GLFW.SwapBuffers(window)
        # Poll for and process events
        GLFW.PollEvents()

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  	end
finally
	GLFW.DestroyWindow(window)
end

end

# ```@doc
# showing how openGL should read data from buffer in GPU
# in case of code like below it would mean:

# first parameter specifies which vertex attribute we want to configure Remember that we specified the location of the position vertex attribute in the vertex shader
# next argument specifies the size of the vertex attribute. The vertex attribute is a vec2 so it is composed of 2 values. 
# The third argument specifies the type of the data which is GL_FLOAT
# next argument specifies if we want the data to be normalized. If we’re inputting integer data types like int, byte and we’ve set this to GL_TRUE
#     The fifth argument is known as the stride and tells us the space between consecutive vertex attributes. Since the next set of position data is
#      located exactly 2 times the size of a float we could’ve also specified the stride as 0 to let OpenGL determine the stride 
#     he last parameter is of type void* and thus requires that weird cast. This is the offset of where the position data begins in the buffer.

# glVertexAttribPointer positionAttribute, 2, GL_FLOAT, false, 0, C_NULL

#     The position data is stored as 32-bit  so 4 byte floating point values. 
#     Each position is composed of 2 of those values. 
#     ```
function glVertexAttribSetting(positionAttribute)
    glVertexAttribPointer(positionAttribute, 2, GL_FLOAT, false, 0, C_NULL)

end

# ```@doc
# based on http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-14-render-to-texture/
# return reference to framebuffer
# ```
# function creatFrameBuffer()
#     FramebufferName = Ref(GLuint(0))
#     glGenFramebuffers(1, FramebufferName);
#     glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName[]);
#     return FramebufferName
# end

# ```@doc
# based on http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-14-render-to-texture/
# return reference to framebuffer
# ```
# function createTexture()
#     # The texture we're going to render to
#     renderedTexture= Ref(GLuint(0));
#     glGenTextures(1, renderedTexture);
    
#     # "Bind" the newly created texture : all future texture functions will modify this texture
#     glBindTexture(GL_TEXTURE_2D, renderedTexture[]);
    
#     # Give an empty image to OpenGL ( the last "0" )
#     glTexImage2D(GL_TEXTURE_2D, 0,GL_RGB, 1024, 768, 0,GL_RGB, GL_UNSIGNED_BYTE, 0);
    
#     # Poor filtering. Needed !
#     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
#     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
#     return renderedTexture
# end

```@doc
creating texture that is storing integer values representing attenuation values in case of CT scan
```
function createTexture(data, width, height)

#The texture we're going to render to
    texture= Ref(GLuint(0));
     glGenTextures(1, texture);
     glBindTexture(GL_TEXTURE_2D, texture[]); 



     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);


    glTexImage2D(GL_TEXTURE_2D, 0, GL_R16I,
     width, height, 0, GL_RED_INTEGER, GL_SHORT, data);




return texture
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
loop that collects any events from openGL in case of animations it is rendering loop
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


"""
it will generally be invoked on GLFW.PollEvents()  in event loop and now depending on 
what will be pressed or clicked it will lead to diffrent actions
"""
function controllWindowInput(window)
	GLFW.SetWindowCloseCallback(window, (_) -> GLFW.DestroyWindow(window))
	#GLFW.SetMouseButtonCallback(window, (_, button, action, mods) -> println("$button $action"))

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

"""
will change display window  so we will be able to see better for example bones ...
    min_shown_white - value of cut off  - all values above will be shown as white 
    max_shown_black - value cut off - all values below will be shown as black
    https://radiopaedia.org/articles/windowing-ct
    soft tissues: W:350–400 L:20–60 4
    minimum and maximum possible values of hounsfield units ...
    int minn = -1024 ;
    int maxx  = 3071;
    and we need to pass data to shaders using https://community.khronos.org/t/const-data-from-vertex-shader-to-fragment-shader/66544
"""
function changeWindow(min_shown_white,max_shown_black)
return map(x->(x+ 1024)/(3071+1024),[min_shown_white,max_shown_black])

end
changeWindow(50,360)
changeWindow(50,360)[2] - changeWindow(50,360)[1]
