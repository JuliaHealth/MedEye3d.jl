using DrWatson
@quickactivate "Probabilistic medical segmentation"

using ModernGL, GeometryTypes, GLFW
include(DrWatson.scriptsdir("display","GLFW","startModules","ModernGlUtil.jl"))



#Create and initialize shaders
module ShadersAndVerticies
using ModernGL, GeometryTypes, GLFW
export createVertexShader
export createFragmentShader
export positions
export elements
export vertices



using DrWatson
@quickactivate "Probabilistic medical segmentation"
include(DrWatson.scriptsdir("display","GLFW","startModules","ModernGlUtil.jl"))

fragmentShaderFileDir = DrWatson.scriptsdir("display","GLFW","startModules","mainShader.frag")


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

getShaderFileText(fragmentShaderFileDir)

```@doc
creating fragment Shader  so controlling colors and textures  
gslString so version of GSLS we are using currently
  ```
function createFragmentShader(gslString::String)
    fsh = """
    $(gslString)
    $(getShaderFileText(fragmentShaderFileDir))  
    """
    return createShader(fsh, GL_FRAGMENT_SHADER)
    end
    


################### data to display verticies


# Now we define another geometry that we will render, a rectangle, this one with an index buffer
# The positions of the vertices in our rectangle
positions = Point{2,Float32}[(-0.5,  0.5),     # top-left
( 0.5,  0.5),     # top-right
( 0.5, -0.5),     # bottom-right
(-0.5, -0.5)]     # bottom-left

# Specify how vertices are arranged into faces
# Face{N,T} type specifies a face with N vertices, with index type
# T (you should choose UInt32), and index-offset O. If you're
# specifying faces in terms of julia's 1-based indexing, you should set
# O=0. (If you instead number the vertices starting with 0, set
# O=-1.)
elements = Face{3,UInt32}[(0,1,2),          # the first triangle
(2,3,0)]          # the second triangle



vertices = Float32.([
  # positions          // colors           // texture coords
   0.8,  1.0, 0.0,   1.0, 0.0, 0.0,   1.0, 1.0,   # top right
   0.8, -1.0, 0.0,   0.0, 1.0, 0.0,   1.0, 0.0,   # bottom right
  -1.0, -1.0, 0.0,   0.0, 0.0, 1.0,   0.0, 0.0,   # bottom left
  -1.0,  1.0, 0.0,   1.0, 1.0, 0.0,   0.0, 1.0    # top left 
])


  end #ShadersAndVerticies