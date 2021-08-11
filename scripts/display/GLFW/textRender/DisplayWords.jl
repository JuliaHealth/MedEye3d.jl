using DrWatson
@quickactivate "Probabilistic medical segmentation"
"""
Module controlling displaying of the text associated with the segmentation 
- either text releted to all slices or just a single one currently displayed or both
"""
module DisplayWords
using FreeTypeAbstraction, ModernGL, Main.PrepareWindowHelpers, Main.OpenGLDisplayUtils. Main.TextureManag,  Main.ShadersAndVerticiesForText, Glutils

include(DrWatson.scriptsdir("display","GLFW","startModules","ModernGlUtil.jl"))

export coordDisplayWords,createTextureForWords

```@doc
coordinating displaying of given text - function will be invoked by main actor holding all of the necessery data to enable displaying
```
function coordDisplayWords(dispObj::forDisplayObjects, wordsDispObj::ForWordsDispStruct)
    bindAndActivateForText(dispObj.shader_program, wordsDispObj.fragment_shader_words,wordsDispObj.vbo_words )
    bindAndDisplayTexture()
    addTextToTexture()
    reactivateMainObj(dispObj.shader_program, dispObj.fragment_shader, dispObj.vbo)

end#coordDisplayWords    

```@doc
First We need to bind fragment shader created to deal with text and supply the vertex shader with data for quad where this text needs to be displayed
    shader_program- reference to shader program
    fragment_shader_words - reference to shader associated with text displaying
```
function bindAndActivateForText(shader_program::Int32,fragment_shader_words::Int32 ,vbo_words::Int32 )
    glAttachShader(shader_program, fragment_shader_words)
    glLinkProgram(shader_program)
    glUseProgram(shader_program)
    
    glBindBuffer(GL_ARRAY_BUFFER, vbo_words[])
    glBufferData(GL_ARRAY_BUFFER, sizeof(Main.ShadersAndVerticiesForText.verticesB), Main.ShadersAndVerticiesForText.verticesB, GL_STATIC_DRAW)

	encodeDataFromDataBuffer()
end #bindAndActivateForText


```@doc
Secondly one need to bind and activate the texture that we will use for displaying the text 
```
function bindAndDisplayTexture()

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
function reactivateMainObj(shader_program::Int32,fragment_shader_main::Int32, vbo_main::Int32 )
    glAttachShader(shader_program, fragment_shader_words)
    glLinkProgram(shader_program)
    glUseProgram(shader_program)

    glBindBuffer(GL_ARRAY_BUFFER, vbo_words[])
    glBufferData(GL_ARRAY_BUFFER, sizeof(Main.ShadersAndVerticies.vertices), Main.ShadersAndVerticies.vertices, GL_STATIC_DRAW)
    
    encodeDataFromDataBuffer()
end #reactivateMainObj



```@doc
Creates and initialize texture that will be used for displaying text
```
function createTextureForWords()::TextureSpec

end#createTextureForWords


end#DisplayWords