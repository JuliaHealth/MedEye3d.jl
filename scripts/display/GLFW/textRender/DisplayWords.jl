using DrWatson
@quickactivate "Probabilistic medical segmentation"

"""
Module controlling displaying of the text associated with the segmentation 
- either text releted to all slices or just a single one currently displayed or both
"""
module DisplayWords
using FreeTypeAbstraction,Main.ForDisplayStructs,Main.DataStructs , ModernGL, ColorTypes,Main.PrepareWindowHelpers, Main.OpenGLDisplayUtils. Main.TextureManag, Main.ShadersAndVerticies, Main.ShadersAndVerticiesForText, Glutils, DrWatson
@quickactivate "Probabilistic medical segmentation"

include(DrWatson.scriptsdir("display","GLFW","startModules","ModernGlUtil.jl"))

export addTextToTexture,renderSingleLineOfText,activateForTextDisp,bindAndActivateForText,reactivateMainObj, createTextureForWords,bindAndActivateForText, bindAndDisplayTexture


```@doc
First We need to bind fragment shader created to deal with text and supply the vertex shader with data for quad where this text needs to be displayed
    shader_program- reference to shader program
    fragment_shader_words - reference to shader associated with text displaying
```
function bindAndActivateForText(shader_program_words::UInt32
                                ,fragment_shader_words::UInt32 
                                ,vertex_shader::UInt32 
                                ,vbo_words::Base.RefValue{UInt32}
                                ,calcDim::CalcDimsStruct)
    
    glLinkProgram(shader_program_words)
    glUseProgram(shader_program_words)
    
	glAttachShader(shader_program_words, fragment_shader_words)
	glAttachShader(shader_program_words, vertex_shader)

    glBindBuffer(GL_ARRAY_BUFFER, vbo_words[])
    glBufferData(GL_ARRAY_BUFFER,calcDim.mainQuadVertSize , calcDim.mainImageQuadVert, GL_STATIC_DRAW)

	encodeDataFromDataBuffer()
end #bindAndActivateForText

```@doc
In order to be able to display texture with text we need to activate main shader program and vbo
    shader_program- reference to shader program
    fragment_shader_words - reference to shader associated with text displaying
```
function activateForTextDisp(shader_program_words::UInt32
                            ,vbo_words::Base.RefValue{UInt32} 
                            ,calcDim::CalcDimsStruct)
    glUseProgram(shader_program_words)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_words[])
    glBufferData(GL_ARRAY_BUFFER, calcDim.wordsQuadVertSize  ,calcDim.wordsImageQuadVert , GL_STATIC_DRAW)

	encodeDataFromDataBuffer()

end#activateForTextDisp

```@doc
Given  vector of SimpleLineTextStructs it will return matrix of data that will be used 
to display text 
wordsDispObj - object wit needed constants to display text
```
function addTextToTexture(wordsDispObj::ForWordsDispStruct
                          ,lines::Vector{SimpleLineTextStruct}
                          ,calcDimStruct::CalcDimsStruct)
    textureWidth = calcDimStruct.textTexturewidthh
    fontFace= wordsDispObj.fontFace
    
    matr=  map(x-> renderSingleLineOfText(x,textureWidth,fontFace) ,lines) |>
    (xl)-> reduce( hcat  ,xl)
    
    sz= size(matr)
@info "Int32(sz[1])" Int32(sz[1])
@info "Int32(sz[2])" Int32(sz[2])
    updateTexture(UInt8
                ,matr
                ,wordsDispObj.textureSpec
                ,0
                ,calcDimStruct.textTextureheightt-sz[2]
                ,Int32(sz[1])
                ,Int32(sz[2])) #  ,Int32(10000),Int32(1000)
    return matr
end #addTextToTexture

```@doc
Given  single SimpleLineTextStruct it will return matrix of data that will be used  by addTextToTexture function
    to display text 
    texLine - source od data
    textureWidth - available width for a line
    fontFace - font we use
```
function renderSingleLineOfText(texLine::SimpleLineTextStruct
                                ,textureWidth::Int32
                                ,fontFace::FTFont)
   return  renderstring!(zeros(UInt8,textureWidth,textureWidth) #::Matrix{UInt8}
                ,texLine.text
                ,fontFace
                ,texLine.fontSize
                ,texLine.fontSize
                ,texLine.fontSize
                ,valign = :vtop
                ,halign = :hleft) |>
    (matr)-> matr[1: Int(round(texLine.fontSize*2*texLine.extraLineSpace)),: ] |>
    (smallerMatr)->collect(transpose(reverse(smallerMatr; dims=(1)))) # getting proper text alignment

end #renderSingleLineOfText    

```@doc
utility function that enables creating list of  text line structs from list of strings
```
function  textLinesFromStrings(strs::Vector{String} ) ::Vector{SimpleLineTextStruct}
    return map(x->  SimpleLineTextStruct(text=x,fontSize= 120,extraLineSpace=1  ) ,strs)
end




```@doc
Finally in order to enable later proper display of the images we need to reactivate main quad and shaders
shader_program- reference to shader program
fragment_shader_main- reference to shader associated with main images
```
function reactivateMainObj(shader_program::UInt32
                            ,vbo_main::UInt32
                            ,calcDim::CalcDimsStruct)
  
    glUseProgram(shader_program)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_main[])
    glBufferData(GL_ARRAY_BUFFER,calcDim.mainQuadVertSize, calcDim.mainImageQuadVert, GL_STATIC_DRAW)
    encodeDataFromDataBuffer()

end #reactivateMainObj


createTextureForWordsStr= """
Creates and initialize texture that will be used for displaying text
   !!!! important we need to first bind shader program for text display before we will  invoke this function
    numberOfActiveTextUnits - number of textures already used - so we we will know what is still free 
    widthh, heightt - size of the texture - the bigger the higher resolution, but higher computation cost
    shader_program_words- reference to shader used to  display text
    return fully initialized texture; also it assigne texture to appropriate sampler
    """
@doc createTextureForWordsStr
function createTextureForWords(numberOfActiveTextUnits::Int
                                ,widthh::Int32 =Int32(100)
                                ,heightt::Int32=Int32(1000)
                                ,shader_program_words::UInt32=UInt32(0) )::TextureSpec
@info "numberOfActiveTextUnits+1" numberOfActiveTextUnits+1
#    texId=createTexture(0,widthh, heightt,GL_R8UI)
#    glBindTexture(GL_TEXTURE_2D, texId[]); 
#    samplerRef= glGetUniformLocation(shader_program_words, "TextTexture1")
#    glUniform1i(samplerRef,numberOfActiveTextUnits+1);
    return Main.ForDisplayStructs.TextureSpec(
            name = "textText"
            ,isTextTexture = true
            ,dataType= UInt8
            ,color = RGB(0.0,0.0,1.0)
            ,widthh=widthh
            ,heightt=heightt
            #,ID=texId
            ,actTextrureNumb =getProperGL_TEXTURE(numberOfActiveTextUnits+1)
            ,OpGlType =GL_UNSIGNED_BYTE
        )
end#createTextureForWords





end#DisplayWords