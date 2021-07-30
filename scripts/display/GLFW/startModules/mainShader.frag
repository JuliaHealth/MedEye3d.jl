    #extension GL_EXT_gpu_shader4 : enable    //Include support for this extension, which defines usampler2D

    out vec4 FragColor;    
    in vec3 ourColor;
    smooth in vec2 TexCoord0;

    //values needed to change integer values of computer tomography attenuation to floats representing colors
    //uniform int minn = -1024 ;
    //uniform int maxx  = 3071;
    //uniform int rang = 4095;

    //soft tissues: W:350–400 L:20–60 4

///////// window control - those uniforms control window of main CT image 

    uniform int  min_shown_white = 0;//400 ;// value of cut off  - all values above will be shown as white 
    uniform int  max_shown_black = -200;//value cut off - all values below will be shown as black
    uniform float displayRange = 600.0;

/////////// samplers
    uniform isampler2D Texture0;
    uniform isampler2D nuclearMask;// nuclear medicine
    uniform isampler2D msk0;
    uniform isampler2D mask1;
    uniform isampler2D mask2;
/////////////// color ranges    --  7 is completely arbitrary if one would want more colrs - it is easy to change
   uniform vec3[7] colorsMask0;
   uniform vec3[7] colorsMask1;
   uniform vec3[7] colorsMask2;
///////////////visibility controll   --  set of booleans that controll weather the texture will affect final image (visible) or not
   uniform bool isVisibleTexture0 = true;
   uniform bool isVisibleNuclearMask = true;
   uniform bool isVisibleMask0 = true;
   uniform bool isVisibleMask1=true;
   uniform bool isVisibleMask2=true;

  
// controlling color display of main image we keep all above some value as white and all below some value as black
vec4 mainColor(in int texel)
{
    if(!isVisibleTexture0){
      return vec4(1.0, 1.0, 1.0, 1.0);    
    }
    else if(texel >min_shown_white){
        return vec4(1.0, 1.0, 1.0, 1.0);
        }
    else if (texel< max_shown_black){
        return vec4(0.0, 0.0, 0.0, 1.0);
   }
    else{
      float fla = float(texel-max_shown_black) ;
      float fl = fla/displayRange ;
     return vec4(fl, fl, fl, 1.0);
    }
}
//data types adapted from https://www.shaderific.com/glsl-types
//controlling color output modification by the
//the textures with values that are always between 0 and 1 
//(including binary masks) - if value is greater than 0 and flag of visibility is set to true 
//the color will affect main color otherwise it would be just passed through the function without any modifications
vec4 maskColor(in int maskTexel,in vec4 FragColorMain ,in bool isVisible ,in vec3[7] colorSet )
{
  if(maskTexel>0 && isVisible) {
       vec4 FragColorMask0 = vec4(maskTexel, 0.0, 0.0, 0.5);
       return FragColorMask0*FragColorMain;
    }
    return FragColorMain;
}




    void main()
    {
        vec4 FragColorA;    

        int texel = texture2D(Texture0, TexCoord0).r;
        int mask0Texel = texture2D(msk0, TexCoord0).r;
        int mask1Texel = texture2D(mask1, TexCoord0).r;
        int mask2Texel = texture2D(mask1, TexCoord0).r;

       FragColorA = mainColor(texel);
       FragColorA =  maskColor(mask0Texel,FragColorA,isVisibleMask0, colorsMask0);
       FragColorA =  maskColor(mask1Texel,FragColorA,isVisibleMask1, colorsMask1);
       FragColorA =  maskColor(mask2Texel,FragColorA,isVisibleMask2, colorsMask2);
    
    FragColor = FragColorA;
    // if(texel >min_shown_white){
    //     FragColor = vec4(1.0, 1.0, 1.0, 1.0);
    //     }
    // else if (texel< max_shown_black){
    //     FragColor = vec4(0.0, 0.0, 0.0, 1.0)*FragColorMask0;

    // }
    // else{
    //   float fla = float(texel-max_shown_black) ;
    //   float fl = fla/displayRange ;
    //   if(mask0Texel>0) {
    //     FragColor = vec4(fl, fl, fl, 1.0)*FragColorMask0;
    //   }else{
    //     FragColor = vec4(fl, fl, fl, 1.0);
    //   }
    // }

    }

