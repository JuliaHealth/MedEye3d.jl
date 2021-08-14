using DrWatson
@quickactivate "Probabilistic medical segmentation"

"""
Module controlling displaying of the text associated with the segmentation 
- either text releted to all slices or just a single one currently displayed or both
"""
module DisplayWords
using FreeTypeAbstraction,Main.ForDisplayStructs, ModernGL, ColorTypes,Main.PrepareWindowHelpers, Main.OpenGLDisplayUtils. Main.TextureManag,  Main.ShadersAndVerticiesForText, Glutils, DrWatson
@quickactivate "Probabilistic medical segmentation"

include(DrWatson.scriptsdir("display","GLFW","startModules","ModernGlUtil.jl"))

export activateForTextDisp,bindAndActivateForText,reactivateMainObj, createTextureForWords,bindAndActivateForText, bindAndDisplayTexture


```@doc
First We need to bind fragment shader created to deal with text and supply the vertex shader with data for quad where this text needs to be displayed
    shader_program- reference to shader program
    fragment_shader_words - reference to shader associated with text displaying
```
function bindAndActivateForText(shader_program_words::UInt32
                                ,fragment_shader_words::UInt32 
                                ,vbo_words::Base.RefValue{UInt32}
                                ,vertex_shader::UInt32 )
    
    glLinkProgram(shader_program_words)
    glUseProgram(shader_program_words)
    
	glAttachShader(shader_program_words, fragment_shader_words)
	glAttachShader(shader_program_words, vertex_shader)

    glBindBuffer(GL_ARRAY_BUFFER, vbo_words[])
    glBufferData(GL_ARRAY_BUFFER, sizeof(Main.ShadersAndVerticiesForText.verticesB), Main.ShadersAndVerticiesForText.verticesB, GL_STATIC_DRAW)

	encodeDataFromDataBuffer()
end #bindAndActivateForText

```@doc
In order to be able to display texture with text we need to activate main shader program and vbo
    shader_program- reference to shader program
    fragment_shader_words - reference to shader associated with text displaying
```
function activateForTextDisp(shader_program_words::UInt32,vbo_words::Base.RefValue{UInt32} )
    glUseProgram(shader_program_words)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_words[])
    glBufferData(GL_ARRAY_BUFFER, sizeof(Main.ShadersAndVerticiesForText.verticesB), Main.ShadersAndVerticiesForText.verticesB, GL_STATIC_DRAW)

	encodeDataFromDataBuffer()

end#activateForTextDisp

```@doc
one need to bind and activate appropriate objects like VBO and use the appropriate shader program to make text display possible
```
function bindAndDisplayTexture(textSpec::TextureSpec)

end #bindAndDisplayTExture


```@doc
Third we need to populate bound texture with data associated with text  - in order to render The text into  texture we will use 
FreeTypeAbstraction library
```
function addTextToTexture()

    face = FreeTypeAbstraction.findfont("hack";  additional_fonts= datadir("fonts"))
    typeof(face)
    img, extent = renderface(face, 'C', 64)
    
    
    
    # render a string into an existing matrix
    a = renderstring!(
        zeros(UInt8, 40, 40),
        "ilililililil",
        face,
        5,
        5,
        5,
        valign = :vbottom,
    )


end #bindAndDisplayTExture


```@doc
Finally in order to enable later proper display of the images we need to reactivate main quad and shaders
shader_program- reference to shader program
fragment_shader_main- reference to shader associated with main images
```
function reactivateMainObj(shader_program::UInt32
                            ,vbo_main::UInt32 )
  
   # glLinkProgram(shader_program)

    # glAttachShader(shader_program, fragment_shader_main)
    # glAttachShader(shader_program, vertex_shader)
    glUseProgram(shader_program)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_main[])
    glBufferData(GL_ARRAY_BUFFER, sizeof(Main.ShadersAndVerticies.vertices), Main.ShadersAndVerticies.vertices, GL_STATIC_DRAW)
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