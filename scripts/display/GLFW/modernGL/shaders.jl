using ModernGL, GeometryTypes, GLFW



#Create and initialize shaders

#VERTEX Shader
function createVertexShader()
vsh = """
$(get_glsl_version_string())
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec2 aTexCoord;
out vec3 ourColor;
smooth out vec2 TexCoord0;
void main()
{
    gl_Position = vec4(aPos, 1.0);
    ourColor = aColor;
    TexCoord0 = aTexCoord;
}
"""
return createShader(vsh, GL_VERTEX_SHADER)
end



#FRAGMENT shader
function createFragmentShader()
    fsh = """
    $(get_glsl_version_string())
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

    smooth in vec2 TexCoord0;
  
    void main()
    {
        int texel = texture2D(Texture0, TexCoord0).r;

    if(texel >min_shown_white){
        FragColor = vec4(1.0, 1.0, 1.0, 1.0);
    }
    else if (texel< max_shown_black){
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);

    }
    else{
      float fla = float(texel-max_shown_black) ;
      float fl = fla/displayRange ;
      FragColor = vec4(fl, fl, fl, 1.0);
    }

    }


    """
    return createShader(fsh, GL_FRAGMENT_SHADER)
    end
    


# #FRAGMENT shader
# function createFragmentShader()
# fsh = """
# $(get_glsl_version_string())
# out vec4 FragColor;
  
# in vec3 ourColor;
# in vec2 TexCoord;
# uniform sampler2D ourTexture;
# uniform int minn = -1024 ;
# uniform int maxx  = 3071;
# uniform int rang = 4095;
# void main()
# {
#     float col=texture(ourTexture, TexCoord).r ;   // input color
#     FragColor = vec4(col,col,col,1.0f);
# }
# """
# return createShader(fsh, GL_FRAGMENT_SHADER)
# end


