using DrWatson
@quickactivate "Probabilistic medical segmentation"

using ModernGL, GeometryTypes, GLFW
include(DrWatson.scriptsdir("display","GLFW","startModules","ModernGlUtil.jl"))



#Create and initialize shaders
module ShadersAndVerticiesForText
using ModernGL, GeometryTypes, GLFW,Main.ForDisplayStructs, Main.CustomFragShad






using DrWatson
@quickactivate "Probabilistic medical segmentation"
include(DrWatson.scriptsdir("display","GLFW","startModules","ModernGlUtil.jl"))


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
creating fragment Shader  so controlling colors and textures  
gslString so version of GSLS we are using currently
  ```
function createFragmentShader(gslString::String)
    fsh = """
    $(gslString)

    #extension GL_EXT_gpu_shader4 : enable    //Include support for this extension, which defines usampler2D

    out vec4 FragColor;    
    in vec3 ourColor;
    smooth in vec2 TexCoord0;

    uniform usampler2D TextTexture1;
    void main() {

    uint text1Texel = texture2D(TextTexture1, TexCoord0).r ;

     if(text1Texel > 0){
      FragColor = vec4(1.0,0.0,1.0,1.0);  }
       else {
    FragColor = vec4(0.0,1.0,0.0,1.0);

    }
    }

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



verticesB = Float32.([
  # positions          // colors           // texture coords
   1.0,  1.0, 0.0,   1.0, 0.0, 0.0,   1.0, 1.0,   # top right
   1.0, -1.0, 0.0,   0.0, 1.0, 0.0,   1.0, 0.0,   # bottom right
   0.8, -1.0, 0.0,   0.0, 0.0, 1.0,   0.0, 0.0,   # bottom left
   0.8,  1.0, 0.0,   1.0, 1.0, 0.0,   0.0, 1.0    # top left 
   ])





  end #ShadersAndVerticies