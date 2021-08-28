"""
stores functions needed to create bind and update OpenGl textues 
"""
module TextureManag
using Base: Float16
using  ModernGL ,DrWatson,   ..OpenGLDisplayUtils,  ..ForDisplayStructs
using   ..Uniforms, Logging,Setfield, Glutils,Logging,  ..CustomFragShad,  ..DataStructs,  ..DisplayWords
export activateTextures,addTextToTexture,initializeTextures,createTexture, getProperGL_TEXTURE,updateImagesDisplayed, updateTexture, assignUniformsAndTypesToMasks

"""
uploading data to given texture; of given types associated - specified in TextureSpec
if we want to update only part of the texture we need to specify  offset and size of texture we use 
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
function updateTexture(::Type{Tt}
                    ,data::AbstractArray
                    ,textSpec::TextureSpec
                    ,xoffset::Int
                    ,yoffset::Int
                    ,widthh::Int32
                    ,heightt::Int32) where{Tt}



    glActiveTexture(textSpec.actTextrureNumb); # active proper texture unit before binding
    glBindTexture(GL_TEXTURE_2D, textSpec.ID[]); 
   
    if((parameter_type(textSpec)== Float16) || (parameter_type(textSpec)== Float32))
        glTexSubImage2D(GL_TEXTURE_2D,0,xoffset,yoffset, widthh, heightt, GL_RED, textSpec.OpGlType, collect(data))
    else
	    glTexSubImage2D(GL_TEXTURE_2D,0,xoffset,yoffset, widthh, heightt, GL_RED_INTEGER, textSpec.OpGlType, collect(data))
	   # glTexSubImage2D(GL_TEXTURE_2D,0,xoffset,yoffset, widthh, heightt, GL_RED_INTEGER, textSpec.OpGlType, reduce(vcat,data))

    end  
    

   
end




"""
creating texture that is storing values like integer, uint, float values that are representing 
main image or  mask data and which will be used by a shader to draw appropriate colors
juliaDataType- data type defined as a Julia datatype of data we are dealing with 
width,height - dimensions of the texture that we need
GL_RType,OpGlType - Open Gl types needed to properly specify the  texture they need to be compatible with juliaDataType
"""
function createTexture(juliaDataType::Type{juliaDataTyp}
                        , width::Int32
                        , height::Int32
                        ,GL_RType::UInt32 =GL_R8UI
                        ,OpGlType= GL_UNSIGNED_BYTE) where {juliaDataTyp}


#The texture we're going to render to
    texture= Ref(GLuint(0));
    glGenTextures(1, texture);
    glBindTexture(GL_TEXTURE_2D, texture[]); 

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
#we just assign storage 
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RType, width, height);


return texture
end



"""
initializing textures  - so we basically execute specification for configuration and bind all to OpenGL context
listOfTextSpecs - list of TextureSpec structs that  holds data needed to 
calcDimStruct - struct holding necessery data about taxture dimensions, quad propertiess etc.
"""
function initializeTextures(listOfTextSpecs::Vector{TextureSpec}
                            ,calcDimStruct ::CalcDimsStruct)::Vector{TextureSpec}

    res = Vector{TextureSpec}()
       
  
    for (ind, textSpec ) in enumerate(listOfTextSpecs)
        index=ind-1
        textUreId= createTexture(parameter_type(textSpec),calcDimStruct.imageTextureWidth,calcDimStruct.imageTextureHeight,textSpec.GL_Rtype,textSpec.OpGlType )#binding texture and populating with data
        @info "textUreId in initializeTextures"  textUreId
       
        actTextrureNumb = getProperGL_TEXTURE(index)
        glActiveTexture(actTextrureNumb)
        glUniform1i(textSpec.uniforms.samplerRef,index);# we first look for uniform sampler in shader  
        # we set ..Uniforms of visibility and colors according to specified in configuration
        if(!textSpec.isMainImage) setMaskColor(textSpec.color ,textSpec.uniforms)   end
        
        
        setTextureVisibility(textSpec.isVisible ,textSpec.uniforms)


        push!(res,setproperties(textSpec, (ID=textUreId, actTextrureNumb=actTextrureNumb,associatedActiveNumer=index )))


    end # for

    return res
end #initializeAndDrawTextures

"""
activating textures that were already initialized in order to be able to use them with diffrent shader program 
shader_program- regference to OpenGL program so we will be able to activate  textures
listOfTextSpecs - list of TextureSpec structs that  holds data needed to bind textures to shader program (Hovewer this new shader program have to keep the same ..Uniforms)
return unmodified textures
"""
function activateTextures(listOfTextSpecs::Vector{TextureSpec} )::Vector{TextureSpec}
      
    for (ind, textSpec ) in enumerate(listOfTextSpecs)
        glBindTexture(GL_TEXTURE_2D, textSpec.ID[]); 
        glUniform1i(textSpec.uniforms.samplerRef,textSpec.associatedActiveNumer);# we first look for uniform sampler in shader  
        # we set ..Uniforms of visibility and colors according to specified in configuration
        if(!textSpec.isMainImage) setMaskColor(textSpec.color ,textSpec.uniforms)   end      
        setTextureVisibility(textSpec.isVisible ,textSpec.uniforms)
    end # for

    return listOfTextSpecs

end#activateTextures


"""
associates GL_TEXTURE UInt32 to given index 
"""
function getProperGL_TEXTURE(index::Int)::UInt32
    return eval(Meta.parse("GL_TEXTURE$(index)"))
end#getProperGL_TEXTURE

"""
coordinating updating all of the images, masks... 
singleSliceDat - holds data we want to use for update
forDisplayObjects - stores all needed constants that holds reference to GLFW and OpenGL
"""
function updateImagesDisplayed(singleSliceDat::SingleSliceDat
                            ,forDisplayConstants::forDisplayObjects
                            ,wordsDispObj::ForWordsDispStruct
                            ,calcDimStruct::CalcDimsStruct
                            ,valueForMaskToSett::valueForMasToSetStruct )

        forDisplayConstants.stopListening[]=true# freeing GLFW from listening for event
             modulelistOfTextSpecs=forDisplayConstants.listOfTextSpecifications

            for updateDat in singleSliceDat.listOfDataAndImageNames
                findList= findall( (texSpec)-> texSpec.name == updateDat.name, modulelistOfTextSpecs)
                texSpec = !isempty(findList) ? modulelistOfTextSpecs[findList[1]] : throw(DomainError(findList, "no such name specified in start configuration - $( updateDat[1])")) 
                updateTexture(updateDat.type,updateDat.dat,texSpec,0,0,calcDimStruct.imageTextureWidth,calcDimStruct.imageTextureHeight )
            end #for 
            #render text associated with this slice
            activateForTextDisp(
                wordsDispObj.shader_program_words
                ,wordsDispObj.vbo_words
                ,calcDimStruct)

            matr= addTextToTexture(wordsDispObj
                            ,[singleSliceDat.textToDisp...,valueForMaskToSett.text]
                            ,calcDimStruct )

            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, C_NULL)
           
            reactivateMainObj(forDisplayConstants.shader_program
                            ,forDisplayConstants.vbo
                            ,calcDimStruct )


            #render onto the screen
             OpenGLDisplayUtils.basicRender(forDisplayConstants.window)
            glFinish()
            forDisplayConstants.stopListening[]=false
end



"""
on the basis of the type supplied in texture characteristic
it supplies given set of ..Uniforms to it 
It would also assign proper openGl types to given julia data type, and pass data from texture specification to opengl context
textSpecs - list of texture specificaton that we want to enrich by adding information about ..Uniforms
return list of texture specifications enriched by information about ..Uniforms 
"""
function assignUniformsAndTypesToMasks(textSpecs::Vector{TextureSpec}
                ,shader_program::UInt32
                ,windowControlStruct::WindowControlStruct)
    mainTexture,notMainTextures=   divideTexteuresToMainAndRest(textSpecs)
#main texture ..Uniforms
n= mainTexture.name

mainUnifs =MainImageUniforms(
    samplerName= n
    ,samplerRef = glGetUniformLocation(shader_program, n)
    ,isVisibleRef = glGetUniformLocation(shader_program, "$(n)isVisible")
    ,min_shown_white= glGetUniformLocation(shader_program, "min_shown_white")
   , max_shown_black= glGetUniformLocation(shader_program, "max_shown_black")
    ,displayRange=glGetUniformLocation(shader_program, "displayRange")
    ,isMaskDiffrenceVis=glGetUniformLocation(shader_program, "isMaskDiffrenceVis")
    ,maskAIndex=glGetUniformLocation(shader_program, "maskAIndex")
    ,maskBIndex=glGetUniformLocation(shader_program, "maskBIndex")
    ,minNuclearMaskVal=glGetUniformLocation(shader_program, "minNuclearMaskVal")
    ,maxNuclearMaskVal=glGetUniformLocation(shader_program, "maxNuclearMaskVal")
    ,rangeOfNuclearMaskVal=glGetUniformLocation(shader_program, "rangeOfNuclearMaskVal")
    ,nuclearMaskSampler=glGetUniformLocation(shader_program, "nuclearMaskSampler")
    ,isNuclearMaskVis=glGetUniformLocation(shader_program, "isNuclearMaskVis")
)
setCTWindow(windowControlStruct.min_shown_white,windowControlStruct.max_shown_black,mainUnifs)

maintext = setproperties(mainTexture, (uniforms= mainUnifs ))
# joining main and not main textures data 
mapped= [ map(x-> setuniforms(x,shader_program) , notMainTextures)..., maintext]

return  (mainUnifs, map(x->setProperOpenGlTypes(x),mapped))


end#assign..UniformsToMasks


 """
helper for assign..UniformsToMasks 
On the basis of the name of the Texture it will assign the informs referencs to it 
- ..Uniforms for main image will be set separately

"""
function setuniforms(textSpec:: TextureSpec,shader_program::UInt32):: TextureSpec
    
    n= textSpec.name
    unifs = MaskTextureUniforms(samplerName= n
                        ,samplerRef= glGetUniformLocation(shader_program, n)
                        ,colorsMaskRef=glGetUniformLocation(shader_program, "$(n)ColorMask") 
                        ,isVisibleRef=glGetUniformLocation(shader_program, "$(n)isVisible"))

    return setproperties(textSpec, (uniforms= unifs ))

end#assign..UniformsToMasks



 """
On the basis of the type associated to texture we set proper open Gl types associated 
based on https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml
and https://www.khronos.org/opengl/wiki/OpenGL_Type
"""
function setProperOpenGlTypes(textSpec:: TextureSpec):: TextureSpec
    if(parameter_type(textSpec)== Float16 ) return  setproperties(textSpec, (GL_Rtype= GL_R16F ,OpGlType= GL_HALF_FLOAT ))     end 
    if(parameter_type(textSpec)== Float32 ) return  setproperties(textSpec, (GL_Rtype= GL_R32F ,OpGlType= GL_FLOAT))     end 
    if(parameter_type(textSpec)== Int8 ) return  setproperties(textSpec, (GL_Rtype= GL_R8I ,OpGlType= GL_BYTE ))     end 
    if(parameter_type(textSpec)== UInt8 ) return  setproperties(textSpec, (GL_Rtype= GL_R8UI,OpGlType= GL_UNSIGNED_BYTE ))     end 
    if(parameter_type(textSpec)== Int16 ) return  setproperties(textSpec, (GL_Rtype= GL_R16I,OpGlType= GL_SHORT ))     end 
    if(parameter_type(textSpec)== UInt16) return  setproperties(textSpec, (GL_Rtype=GL_R16UI ,OpGlType=GL_UNSIGNED_SHORT))     end 
    if(parameter_type(textSpec)== Int32 ) return  setproperties(textSpec, (GL_Rtype= GL_R32I,OpGlType= GL_INT ))     end 
    if(parameter_type(textSpec)== UInt32) return  setproperties(textSpec, (GL_Rtype=GL_R32UI ,OpGlType= GL_UNSIGNED_INT))     end 

    throw(DomainError(textSpec, "type  of texture is not supported - supported types - Int8,16,32 UInt 8,16,32 float16,32")) 
end#



"""
Given  vector of SimpleLineTextStructs it will return matrix of data that will be used 
to display text 
lines - data about text to be displayed
calcDimStruct - struct holding important data about size of textures etc.
wordsDispObj - object wit needed constants to display text
"""
function addTextToTexture(wordsDispObj::ForWordsDispStruct
                          ,lines::Vector{SimpleLineTextStruct}
                          ,calcDimStruct::CalcDimsStruct)
    textureWidth = calcDimStruct.textTexturewidthh
    fontFace= wordsDispObj.fontFace
    
    matrPrim=  map(x-> renderSingleLineOfText(x,textureWidth,fontFace) ,reverse(lines)) |>
    (xl)-> reduce( hcat  ,xl)
    
    sz= size(matrPrim)
    # below just to clear any data from texture uploaded before
    matr= hcat(calcDimStruct.textTextureZeros[:,sz[2]:size(calcDimStruct.textTextureZeros)[2]-1] ,matrPrim  )

    updateTexture(UInt8
                ,matr
                ,wordsDispObj.textureSpec
                ,0
                ,0
                ,calcDimStruct.textTexturewidthh
                ,calcDimStruct.textTextureheightt) #  ,Int32(10000),Int32(1000)
    return matr
end #addTextToTexture




########## puts bytes of image into PBO as fas as I get it  copy an image data to texture buffer


# preparePixelBufferStr="""
# width -width of the image in  number of pixels 
# height - height of the image in  number of pixels 
# pboNumber - just states which PBO it is
# return reference to the pixel buffer object that we use to upload this texture and data size calculated for this texture
# """
# @doc preparePixelBufferStr
# function preparePixelBuffer(juliaDataTyp::Type{juliaDataType},width,height,pboNumber)where{juliaDataType}
#     DATA_SIZE = 8 * sizeof(juliaDataTyp) *width * height  # number of bytes our image will have so in 2D it will be width times height times number of bytes needed for used datatype we need to multiply by 8 becouse sizeof() return bytes instead of bits
#     pbo = Ref(GLuint(pboNumber))  
#     glGenBuffers(1, pbo)
#     return (pbo,DATA_SIZE)
# end






# usePixelBuferAndUploadDataStr = """
# adapted from http://www.songho.ca/opengl/gl_pbo.html
# creates single pixel buffer of given type
# pboID - id of the pixel buffer object that was prepared for some particular texture
# textureId - reference to id of a texture that we want to bind to this PBO
# juliaDataType -julia type that is representing datatype in 2 dimensional array representing ima
# width -width of the image in  number of pixels 
# height - height of the image in  number of pixels 
# subImageDataType - variable used in glTexSubImage2D to tell open Glo what type of data is in texture
# data one dimensional array o julia type and width*height length
# DATA_SIZE - size of texture in bytes
# """
# @doc usePixelBuferAndUploadDataStr
# function usePixelBuferAndUploadData(
#     juliaDataTyp::Type{juliaDataType}
#                     ,pboID 
#                     ,width
#                     ,height
#                     ,data
#                     ,textureId
#                     ,DATA_SIZE
#                     ,subImageDataType = GL_SHORT
                
#                     )where{juliaDataType}

#     glBindTexture(GL_TEXTURE_2D,textureId[]); 
#     # copy pixels from PBO to texture object
#     # Use offset instead of pointer.
#    # glTexSubImage2D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, GLsizei(width), GLsizei(height),  GL_RED_INTEGER, GL_SHORT, Ptr{juliaDataTyp}());
   
#     glTexSubImage2D(GL_TEXTURE_2D,0,0,0, width, height, GL_RED_INTEGER, subImageDataType, Ptr{juliaDataType}());

  
#     # bind the PBO
#     glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID[]);


#     # Note that glMapBuffer() causes sync issue.
#     # If GPU is working with this buffer, glMapBuffer() will wait(stall)
#     # until GPU to finish its job. To avoid waiting (idle), you can call
#     # first glBufferData() with NULL pointer before glMapBuffer().
#     # If you do that, the previous data in PBO will be discarded and
#     # glMapBuffer() returns a new allocated pointer immediately
#     # even if GPU is still working with the previous data.
#     glBufferData(GL_PIXEL_UNPACK_BUFFER, DATA_SIZE, Ptr{juliaDataType}(), GL_STREAM_DRAW);
    
#     # map the buffer object into client's memory
#     glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY)
    
     
#     ptr = Ptr{juliaDataType}(glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY))
#     # update data directly on the mapped buffer - this is internal function implemented below
    
#     updatePixels(ptr,data,length(data));

#     glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER); # release the mapped buffer
    
#     # it is good idea to release PBOs with ID 0 after use.
#     # Once bound with 0, all pixel operations are back to normal ways.
#     glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);



# end

# updatePixelsStr = """
# adapted from https://github.com/JuliaPlots/GLMakie.jl/blob/2717d812fdc66b283f63d5d97237e8d69e2c1f25/src/GLAbstraction/GLBuffer.jl from unsafe copy
# """
# @doc updatePixelsStr
# function updatePixels(ptr, data,length)
#     for i=1:length
#         unsafe_store!(ptr,data[i], i)
#     end
# end


end #..TextureManag