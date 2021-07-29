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

/////////// samplers
    uniform isampler2D Texture0;
    uniform isampler2D nuclearMask;
    uniform isampler2D msk0;
    uniform isampler2D mask1;
    uniform isampler2D mask2;
    smooth in vec2 TexCoord0;
/////////////// color ranges
    vec3 newVec3Array[3];



  
// controlling color display of main image we keep all above some value as white and all below some value as black
void mainColor(in int texel, out vec4 FragColorMain)
{
    if(texel >min_shown_white){
        FragColorMain = vec4(1.0, 1.0, 1.0, 1.0);
        }
    else if (texel< max_shown_black){
        FragColorMain = vec4(0.0, 0.0, 0.0, 1.0);
   }
    else{
      float fla = float(texel-max_shown_black) ;
      float fl = fla/displayRange ;
      FragColorMain = vec4(fl, fl, fl, 1.0);
    }
}
//data types adapted from https://www.shaderific.com/glsl-types
//controlling color output modification by the
//the textures with values that are always between 0 and 1 
//(including binary masks) - if value is greater than 0 and flag of visibility is set to true 
//the color will affect main color otherwise it would be just passed through the function without any modifications
void mainColor(in int maskTexel,in vec4 FragColorMain,  out vec4 FragColorMask)
{
TODo() pass boolean that controlls visibility and pass input color/ colors as mask color
  if(maskTexel>0) {
       vec4 FragColorMask0 = vec4(mask0Texel, 0.0, 0.0, 0.5);
       FragColorMain = vec4(fl, fl, fl, 1.0)*FragColorMask0;
    }
}




    void main()
    {
        int texel = texture2D(Texture0, TexCoord0).r;
        int mask0Texel = texture2D(msk0, TexCoord0).r;
        int mask1Texel = texture2D(mask1, TexCoord0).r;
        
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

