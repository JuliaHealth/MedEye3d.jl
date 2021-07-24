using DrWatson

```@doc
stores functions needed to create bind and update OpenGl textues 
```
module TextureManag
using  ModernGL
using DrWatson
using  Main.OpenGLDisplayUtils

@quickactivate "Probabilistic medical segmentation"



updateTextureString = """
uploading data to given texture; of given types associated
"""
@doc updateTextureString
function updateTexture(juliaDataTyp::Type{juliaDataType},width,height,data, textureId,stopListening,pboId, DATA_SIZE,GlNumbType )where{juliaDataType}

	glBindTexture(GL_TEXTURE_2D, textureId[]); 
	glTexSubImage2D(GL_TEXTURE_2D,0,0,0, width, height, GL_RED_INTEGER, GlNumbType, data);

end




```@doc
creating texture that is storing integer values representing attenuation values in case of CT scan
numb - which texture it is - basically important only that diffrent textures would have diffrent numbers

```
function createTexture(numb::Int,data, width::Int, height::Int,GL_RType =GL_R16I, GlNumbType = GL_SHORT  )

#The texture we're going to render to
    
texture= Ref(GLuint(numb));
    glGenTextures(1, texture);
    glBindTexture(GL_TEXTURE_2D, texture[]); 

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RType, width, height);
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0, width, height, GL_RED_INTEGER, GlNumbType, data);


return texture
end


```@doc
initializing and drawing the textures on the screen

```

# ##################
#clear color buffer
# glClearColor(0.0, 0.0, 0.1 , 1.0)
# #true labels
# glActiveTexture(GL_TEXTURE0 + 1); # active proper texture unit before binding
# glUniform1i(glGetUniformLocation(shader_program, "msk0"), 1);# we first look for uniform sampler in shader - here 
# trueLabels= createTexture(1,exampleLabels[210,:,:],widthh,heightt,GL_R8UI,GL_UNSIGNED_BYTE)#binding texture and populating with data
# #main image
# glActiveTexture(GL_TEXTURE0); # active proper texture unit before binding
# glUniform1i(glGetUniformLocation(shader_program, "Texture0"), 0);# we first look for uniform sampler in shader - here 
# mainTexture= createTexture(0,exampleDat[210,:,:],widthh,heightt,GL_R16I,GL_SHORT)#binding texture and populating with data
# #render
# basicRender()









########## puts bytes of image into PBO as fas as I get it  copy an image data to texture buffer


"""
width -width of the image in  number of pixels 
height - height of the image in  number of pixels 
pboNumber - just states which PBO it is
return reference to the pixel buffer object that we use to upload this texture and data size calculated for this texture

"""
function preparePixelBuffer(juliaDataTyp::Type{juliaDataType},width,height,pboNumber)where{juliaDataType}
    DATA_SIZE = 8 * sizeof(juliaDataTyp) *width * height  # number of bytes our image will have so in 2D it will be width times height times number of bytes needed for used datatype we need to multiply by 8 becouse sizeof() return bytes instead of bits
    pbo = Ref(GLuint(pboNumber))  
    glGenBuffers(1, pbo)
    return (pbo,DATA_SIZE)
end









usePixelBuferAndUploadDataStr = """
adapted from http://www.songho.ca/opengl/gl_pbo.html
creates single pixel buffer of given type
pboID - id of the pixel buffer object that was prepared for some particular texture
textureId - reference to id of a texture that we want to bind to this PBO
juliaDataType -julia type that is representing datatype in 2 dimensional array representing ima
width -width of the image in  number of pixels 
height - height of the image in  number of pixels 
subImageDataType - variable used in glTexSubImage2D to tell open Glo what type of data is in texture
data one dimensional array o julia type and width*height length
DATA_SIZE - size of texture in bytes
"""
@doc usePixelBuferAndUploadDataStr
function usePixelBuferAndUploadData(
    juliaDataTyp::Type{juliaDataType}
                    ,pboID 
                    ,width
                    ,height
                    ,data
                    ,textureId
                    ,DATA_SIZE
                    ,subImageDataType = GL_SHORT
                
                    )where{juliaDataType}

    glBindTexture(GL_TEXTURE_2D,textureId[]); 
    # copy pixels from PBO to texture object
    # Use offset instead of pointer.
   # glTexSubImage2D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, GLsizei(width), GLsizei(height),  GL_RED_INTEGER, GL_SHORT, Ptr{juliaDataTyp}());
   
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0, width, height, GL_RED_INTEGER, subImageDataType, Ptr{juliaDataType}());

  
    # bind the PBO
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID[]);


    # Note that glMapBuffer() causes sync issue.
    # If GPU is working with this buffer, glMapBuffer() will wait(stall)
    # until GPU to finish its job. To avoid waiting (idle), you can call
    # first glBufferData() with NULL pointer before glMapBuffer().
    # If you do that, the previous data in PBO will be discarded and
    # glMapBuffer() returns a new allocated pointer immediately
    # even if GPU is still working with the previous data.
    glBufferData(GL_PIXEL_UNPACK_BUFFER, DATA_SIZE, Ptr{juliaDataType}(), GL_STREAM_DRAW);
    
    # map the buffer object into client's memory
    glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY)
    
     
    ptr = Ptr{juliaDataType}(glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY))
    # update data directly on the mapped buffer - this is internal function implemented below
    
    updatePixels(ptr,data,length(data));

    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER); # release the mapped buffer
    
    # it is good idea to release PBOs with ID 0 after use.
    # Once bound with 0, all pixel operations are back to normal ways.
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);



end

updatePixelsStr = """
adapted from https://github.com/JuliaPlots/GLMakie.jl/blob/2717d812fdc66b283f63d5d97237e8d69e2c1f25/src/GLAbstraction/GLBuffer.jl from unsafe copy
"""
@doc updatePixelsStr
function updatePixels(ptr, data,length)
    for i=1:length
        unsafe_store!(ptr,data[i], i)
    end
end


end #TextureManag