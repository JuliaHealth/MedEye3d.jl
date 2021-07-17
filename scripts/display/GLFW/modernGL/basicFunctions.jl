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
creating GL_LUMINANCE texture \(black and white\)
```
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


function createTexture(data, width, height)

#The texture we're going to render to
    texture= Ref(GLuint(0));
     glGenTextures(1, texture);
     glBindTexture(GL_TEXTURE_2D, texture[]); 


     borderColor = [ 1.0, 1.0, 0.0, 1.0];
     glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);  

     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    

    # int width, height, nrChannels;
    # unsigned char *data = stbi_load("container.jpg", &width, &height, &nrChannels, 0); 
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED,
     width, height, 0, GL_LUMINANCE, GL_FLOAT, data);

return texture
end




function createData(width,height)

 return  rand(Float32, width*height)

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