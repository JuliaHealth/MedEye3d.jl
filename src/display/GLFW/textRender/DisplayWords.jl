"""
Module controlling displaying of the text associated with the segmentation 
- either text releted to all slices or just a single one currently displayed or both
"""
module DisplayWords
using FreeTypeAbstraction, ..ForDisplayStructs, ..DataStructs, ..ModernGlUtil , ModernGL, ColorTypes, ..PrepareWindowHelpers,   ..ShadersAndVerticies,  ..ShadersAndVerticiesForText


export getTextForCurrentSlice,textLinesFromStrings,renderSingleLineOfText,activateForTextDisp,bindAndActivateForText,reactivateMainObj, createTextureForWords,bindAndActivateForText, bindAndDisplayTexture


"""
First We need to bind fragment shader created to deal with text and supply the vertex shader with data for quad where this text needs to be displayed
    shader_program- reference to shader program
this function is intended to be invoked only once
    """
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

"""
In order to be able to display texture with text we need to activate main shader program and vbo
    shader_program- reference to shader program
    fragment_shader_words - reference to shader associated with text displaying
    calcDim - holds necessery constants holding for example window dimensions, texture sizes etc.
"""
function activateForTextDisp(shader_program_words::UInt32
                            ,vbo_words::Base.RefValue{UInt32} 
                            ,calcDim::CalcDimsStruct)
   glUseProgram(shader_program_words)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_words[])
    glBufferData(GL_ARRAY_BUFFER, calcDim.wordsQuadVertSize  ,calcDim.wordsImageQuadVert , GL_STATIC_DRAW)

	encodeDataFromDataBuffer()

end#activateForTextDisp



"""
Given  single SimpleLineTextStruct it will return matrix of data that will be used  by addTextToTexture function
    to display text 
    texLine - source od data
    textureWidth - available width for a line
    fontFace - font we use
"""
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

"""
utility function that enables creating list of  text line structs from list of strings
"""
function  textLinesFromStrings(strs::Vector{String} ) ::Vector{SimpleLineTextStruct}
    return map(x->  SimpleLineTextStruct(text=x,fontSize= 120,extraLineSpace=1  ) ,strs)
end




"""
Finally in order to enable later proper display of the images we need to reactivate main quad and shaders
shader_program- reference to shader program
fragment_shader_main- reference to shader associated with main images
"""
function reactivateMainObj(shader_program::UInt32
                            ,vbo_main::UInt32
                            ,calcDim::CalcDimsStruct)
  
    glUseProgram(shader_program)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_main[])
    glBufferData(GL_ARRAY_BUFFER,calcDim.mainQuadVertSize, calcDim.mainImageQuadVert, GL_STATIC_DRAW)
    encodeDataFromDataBuffer()

end #reactivateMainObj


"""
Creates and initialize texture that will be used for displaying text
   !!!! important we need to first bind shader program for text display before we will  invoke this function
    numberOfActiveTextUnits - number of textures already used - so we we will know what is still free 
    widthh, heightt - size of the texture - the bigger the higher resolution, but higher computation cost
    actTextrureNumb -proper OpenGL active texture 
    return fully initialized texture; also it assigne texture to appropriate sampler
    """
function createTextureForWords(numberOfActiveTextUnits::Int
                                ,widthh::Int32 =Int32(100)
                                ,heightt::Int32=Int32(1000)
                                ,actTextrureNumb::UInt32=UInt32(0) )::TextureSpec
    return  TextureSpec{UInt8}(
            name = "textText"
            ,color = RGB(0.0,0.0,1.0)
            #,ID=texId
            ,actTextrureNumb =actTextrureNumb
            ,OpGlType =GL_UNSIGNED_BYTE
        )
end#createTextureForWords

"""
we need to check wether scrolling dat contains some text that can be used for this particular slice display if not we will return only mainTextToDisp
"""
function getTextForCurrentSlice(scrollDat::FullScrollableDat, sliceNumb::Int32)::Vector{SimpleLineTextStruct}
    if( length(scrollDat.sliceTextToDisp)>=sliceNumb  ) 
        return  copy(vcat(scrollDat.mainTextToDisp ,  scrollDat.sliceTextToDisp[sliceNumb] ))
    end#if    
        return copy(scrollDat.mainTextToDisp )
end#getTextForCurrentSlice



end#DisplayWords