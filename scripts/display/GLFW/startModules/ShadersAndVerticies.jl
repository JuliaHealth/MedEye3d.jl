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

    //values needed to change integer values of computer tomography attenuation to floats representing colors
    //uniform int minn = -1024 ;
    //uniform int maxx  = 3071;
    //uniform int rang = 4095;

    //soft tissues: W:350–400 L:20–60 4
    uniform int  min_shown_white = 400 ;// value of cut off  - all values above will be shown as white 
    uniform int  max_shown_black = -200;//value cut off - all values below will be shown as black
    uniform float displayRange = 600.0;



    uniform isampler2D Texture0;
    uniform isampler2D nuclearMask;
    uniform isampler2D msk0;
    uniform isampler2D mask1;
    uniform isampler2D mask2;
    



    smooth in vec2 TexCoord0;
  
    void main()
    {
        int texel = texture2D(Texture0, TexCoord0).r;
        int mask0Texel = texture2D(msk0, TexCoord0).r;
        vec4 FragColorMask0 = vec4(mask0Texel, 0.0, 0.0, 0.5);
  
    if(texel >min_shown_white){
        FragColor = vec4(1.0, 1.0, 1.0, 1.0);
        }
    else if (texel< max_shown_black){
        FragColor = vec4(0.0, 0.0, 0.0, 1.0)*FragColorMask0;

    }
    else{
      float fla = float(texel-max_shown_black) ;
      float fl = fla/displayRange ;
      if(mask0Texel>0) {
        FragColor = vec4(fl, fl, fl, 1.0)*FragColorMask0;
      }else{
        FragColor = vec4(fl, fl, fl, 1.0);
      }
    }

    }


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