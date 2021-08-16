using DrWatson
@quickactivate "Probabilistic medical segmentation"

using ModernGL, GeometryTypes, GLFW
include(DrWatson.scriptsdir("display","GLFW","startModules","ModernGlUtil.jl"))



#Create and initialize shaders
module ShadersAndVerticies
using ModernGL, GeometryTypes, GLFW,Main.ForDisplayStructs, Main.CustomFragShad


export createFragmentShader
export positions
export elements
export getMainVerticies
export createVertexShader



using DrWatson
@quickactivate "Probabilistic medical segmentation"
include(DrWatson.scriptsdir("display","GLFW","startModules","ModernGlUtil.jl"))

#in case one want to put shader gsls file into separate file
#fragmentShaderFileDir = DrWatson.scriptsdir("display","GLFW","shadersEtc","mainShader.frag")



```@doc
creating VertexShader  so controlling structures like verticies, quads
gslString so version of GSLS we are using currently
  ```
function createVertexShader(gslString::String)
vsh = """
$(gslString)
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec2 aTexCoord;
out vec3 ourColor;
smooth out vec2 TexCoord0;
void main()
{
    gl_Position = vec4(aPos, 1.0);
    ourColor = aColor;
  // TexCoord0 = vec2(aTexCoord.y, aTexCoord.x);
   TexCoord0 = aTexCoord;

}
"""
return createShader(vsh, GL_VERTEX_SHADER)
end


```@doc
loading th shader from file- so we have better experience writing shader in separate file
```
function getShaderFileText(path::String)
f = open(path)
return  join(readlines(f), "\n") 
end #getShaderFileText

#getShaderFileText(fragmentShaderFileDir)

```@doc
creating fragment Shader  so controlling colors and textures  
gslString so version of GSLS we are using currently
  ```
function createFragmentShader(gslString::String,listOfTexturesToCreate::Vector{TextureSpec})
    fsh = """
    $(gslString)
    $(Main.CustomFragShad.createCustomFramgentShader(listOfTexturesToCreate))  
    """
    return createShader(fsh, GL_FRAGMENT_SHADER)
    end
    


################### data to display verticies

# Specify how vertices are arranged into faces
# Face{N,T} type specifies a face with N vertices, with index type
# T (you should choose UInt32), and index-offset O. If you're
# specifying faces in terms of julia's 1-based indexing, you should set
# O=0. (If you instead number the vertices starting with 0, set
# O=-1.)
elements = Face{3,UInt32}[(0,1,2),          # the first triangle
(2,3,0)]          # the second triangle




  end #ShadersAndVerticies