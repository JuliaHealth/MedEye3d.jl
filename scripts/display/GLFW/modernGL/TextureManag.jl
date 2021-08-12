using DrWatson
@quickactivate "Probabilistic medical segmentation"

```@doc
stores functions needed to create bind and update OpenGl textues 
```
module TextureManag
using Base: Float16
using  ModernGL ,DrWatson,  Main.OpenGLDisplayUtils, Main.ForDisplayStructs
using  Main.Uniforms, Logging,Setfield, Glutils,Logging, Main.CustomFragShad
export initializeTextures,createTexture, updateImagesDisplayed, updateTexture, assignUniformsAndTypesToMasks

updateTextureString = """
uploading data to given texture; of given types associated - specified in TextureSpec
if we want to update only part of the texture we need to specify  what 
Just for reference openGL function definition
    void glTextureSubImage2D(	
        GLuint texture,
         GLint level,
         GLint xoffset,
         GLint yoffset,
         GLsizei width,
         GLsizei height,
         GLenum format,
         GLenum type,
         const void *pixels);  
"""
@doc updateTextureString
function updateTexture(data, textSpec::TextureSpec,xoffset=0,yoffset=0,widthh=textSpec.widthh,heightt =textSpec.heightt  )

    glActiveTexture(textSpec.actTextrureNumb); # active proper texture unit before binding
    glBindTexture(GL_TEXTURE_2D, textSpec.ID[]); 
	glTexSubImage2D(GL_TEXTURE_2D,0,xoffset,yoffset, widthh, heightt, GL_RED_INTEGER, textSpec.OpGlType, data)
end




```@doc
creating texture that is storing integer values representing attenuation values in case of CT scan
numb - which texture it is - basically important only that diffrent textures would have diffrent numbers

```
function createTexture(numb::Int, width::Int32, height::Int32,GL_RType::UInt32 =GL_R16I)
#The texture we're going to render to
    texture= Ref(GLuint(0));
    glGenTextures(1, texture);
    glBindTexture(GL_TEXTURE_2D, texture[]); 

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RType, width, height);
    #glTexSubImage2D(GL_TEXTURE_2D,0,0,0, width, height, GL_RED_INTEGER, GlNumbType, data);


return texture
end



```@doc
initializing textures 
shader_program- regference to OpenGL program so we will be able to initialize textures
listOfTextSpecs - list of TextureSpec structs that  holds data needed to 
it creates textrures as specified, renders them and return the list from  argument augmented by texture Id

```
function initializeTextures(listOfTextSpecs::Vector{Main.ForDisplayStructs.TextureSpec})::Vector{Main.ForDisplayStructs.TextureSpec}

    res = Vector{TextureSpec}()

    for (ind, textSpec ) in enumerate(listOfTextSpecs)
        index=ind-1
        textUreId= createTexture(index,textSpec.widthh,textSpec.heightt,textSpec.GL_Rtype)#binding texture and populating with data
        @info "textUreId in initializeTextures"  textUreId
       
        actTextrureNumb = getProperGL_TEXTURE(index)
        glActiveTexture(actTextrureNumb)
        glUniform1i(textSpec.uniforms.samplerRef,index);# we first look for uniform sampler in shader  

        push!(res,setproperties(textSpec, (ID=textUreId, actTextrureNumb=actTextrureNumb)))


    end # for

    return res
end #initializeAndDrawTextures


```@doc
associates GL_TEXTURE UInt32 to given index 
```
function getProperGL_TEXTURE(index::Int)::UInt32
    return eval(Meta.parse("GL_TEXTURE$(index)"))
end#getProperGL_TEXTURE

updateImagesDisplayedStr =    """
coordinating updating all of the images, masks... 
listOfTextSpecs - holds required data needed to initialize textures - 
tuples where first entry is name of image that we given in configuration; 
and second entry is data that we want to pass
forDisplayObjects - stores all needed constants that holds reference to GLFW and OpenGL
"""
@doc updateImagesDisplayedStr
function updateImagesDisplayed(listOfDataAndImageNames, forDisplayConstants::forDisplayObjects)
   
        forDisplayConstants.stopListening[]=true
             modulelistOfTextSpecs=forDisplayConstants.listOfTextSpecifications
            #clearing color buffer
            glClearColor(0.0, 0.0, 0.1 , 1.0)
            for updateDat in listOfDataAndImageNames
                findList= findall( (texSpec)-> texSpec.name == updateDat[1], modulelistOfTextSpecs)
                texSpec = !isempty(findList) ? modulelistOfTextSpecs[findList[1]] : throw(DomainError(findList, "no such name specified in start configuration - $( updateDat[1])")) 
                Main.TextureManag.updateTexture(updateDat[2],texSpec)
            end #for 
            #render onto the screen
            Main.OpenGLDisplayUtils.basicRender(forDisplayConstants.window)
            forDisplayConstants.stopListening[]=false
end



assignUniformsAndTypesToMaskstr =    """
on the basis of the type supplied in texture characteristic
it supplies given set of uniforms to it 
It would also assign proper openGl types to given julia data type
uniformsStructs - list of pairs where first entry  is a type of sampler like Int/Uint/Float and second is struct with references to uniforms 
textSpecs - list of texture specificaton that we want to enrich by adding information about uniforms
return list of texture specifications enriched by information about uniforms 
"""
@doc assignUniformsAndTypesToMaskstr
function assignUniformsAndTypesToMasks(textSpecs::Vector{Main.ForDisplayStructs.TextureSpec},shader_program::UInt32)
    mainTexture,notMainTextures=   divideTexteuresToMainAndRest(textSpecs)
#main texture uniforms
n= mainTexture.name
mainUnifs =MainImageUniforms(
    samplerName= n
    ,samplerRef = glGetUniformLocation(shader_program, n)
    ,isVisibleRef = glGetUniformLocation(shader_program, "$(n)isVisible")
    ,min_shown_white= glGetUniformLocation(shader_program, "min_shown_white")
   , max_shown_black= glGetUniformLocation(shader_program, "max_shown_black")
    ,displayRange=glGetUniformLocation(shader_program, "displayRange")
)
maintext = setproperties(mainTexture, (uniforms= mainUnifs ))
# joining main and not main textures data 
mapped= [ map(x-> setUniforms(x,shader_program) , notMainTextures)..., maintext]

return  (mainUnifs, map(x->setProperOpenGlTypes(x),mapped))


end#assignUniformsToMasks


setUniformsStr =    """
helper for assignUniformsToMasks 
On the basis of the name of the Texture it will assign the informs referencs to it 
- uniforms for main image will be set separately

"""
@doc setUniformsStr
function setUniforms(textSpec::Main.ForDisplayStructs.TextureSpec,shader_program::UInt32)::Main.ForDisplayStructs.TextureSpec
    
    n= textSpec.name
    unifs = MaskTextureUniforms(samplerName= n
                        ,samplerRef= glGetUniformLocation(shader_program, n)
                        ,colorsMaskRef=glGetUniformLocation(shader_program, "$(n)ColorMask") 
                        ,isVisibleRef=glGetUniformLocation(shader_program, "$(n)isVisible"))

    return setproperties(textSpec, (uniforms= unifs ))

end#assignUniformsToMasks



setProperOpenGlTypesStr = """
On the basis of the type associated to texture we set proper open Gl types associated 
based on https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml
and https://www.khronos.org/opengl/wiki/OpenGL_Type
"""
@doc setProperOpenGlTypesStr
function setProperOpenGlTypes(textSpec::Main.ForDisplayStructs.TextureSpec)::Main.ForDisplayStructs.TextureSpec
    if(textSpec.dataType== Float16 ) return  setproperties(textSpec, (GL_Rtype= GL_R16F ,OpGlType= GL_HALF_FLOAT ))     end 
    if(textSpec.dataType== Float32 ) return  setproperties(textSpec, (GL_Rtype= GL_R32F ,OpGlType= GL_FLOAT))     end 
    if(textSpec.dataType== Int8 ) return  setproperties(textSpec, (GL_Rtype= GL_R8I ,OpGlType= GL_BYTE ))     end 
    if(textSpec.dataType== UInt8 ) return  setproperties(textSpec, (GL_Rtype= GL_R8UI,OpGlType= GL_UNSIGNED_BYTE ))     end 
    if(textSpec.dataType== Int16 ) return  setproperties(textSpec, (GL_Rtype= GL_R16I,OpGlType= GL_SHORT ))     end 
    if(textSpec.dataType== UInt16) return  setproperties(textSpec, (GL_Rtype=GL_R16UI ,OpGlType=GL_UNSIGNED_SHORT))     end 
    if(textSpec.dataType== Int32 ) return  setproperties(textSpec, (GL_Rtype= GL_R32I,OpGlType= GL_INT ))     end 
    if(textSpec.dataType== UInt32) return  setproperties(textSpec, (GL_Rtype=GL_R32UI ,OpGlType= GL_UNSIGNED_INT))     end 

    throw(DomainError(textSpec, "type  of texture is not supported - supported types - Int8,16,32 UInt 8,16,32 float16,32")) 
end#






########## puts bytes of image into PBO as fas as I get it  copy an image data to texture buffer


preparePixelBufferStr="""
width -width of the image in  number of pixels 
height - height of the image in  number of pixels 
pboNumber - just states which PBO it is
return reference to the pixel buffer object that we use to upload this texture and data size calculated for this texture
"""
@doc preparePixelBufferStr
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