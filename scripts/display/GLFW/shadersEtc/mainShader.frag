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

    uniform int  min_shown_white = ;//400 ;// value of cut off  - all values above will be shown as white 
    uniform int  max_shown_black = -200;//value cut off - all values below will be shown as black
    uniform float displayRange = 600.0;

/////////// samplers
//samplers for main texture - one to choose
    uniform isampler2D iTexture0;
    uniform usampler2D uTexture0;
    uniform sampler2D fTexture0;
    uniform int typeOfMainSampler;// 1 will mean integer, 2 will mean uint and 3 float

//samplers for nuclear medicine  texture - one to choose

    uniform isampler2D inuclearMask;   // nuclear medicine
    uniform isampler2D unuclearMask;   // nuclear medicine
    uniform isampler2D fnuclearMask;   // nuclear medicine
///uint masks
    uniform usampler2D uImask0;
    uniform usampler2D uImask1;
    uniform usampler2D uImask2;
    uniform usampler2D uImask3;
    uniform usampler2D uImask4;
    uniform usampler2D uImask5;
    uniform usampler2D uImask6;
    uniform usampler2D uImask7;
    uniform usampler2D uImask8;
//int masks
    uniform isampler2D imask0;
    uniform isampler2D imask1;
    uniform isampler2D imask2;
    uniform isampler2D imask3;
    uniform isampler2D imask4;
    uniform isampler2D imask5;
    uniform isampler2D imask6;
    uniform isampler2D imask7;
    uniform isampler2D imask8;
//float masks
    uniform sampler2D fmask0;
    uniform sampler2D fmask1;
    uniform sampler2D fmask2;
    uniform sampler2D fmask3;
    uniform sampler2D fmask4;
    uniform sampler2D fmask5;
    uniform sampler2D fmask6;
    uniform sampler2D fmask7;
    uniform sampler2D fmask8;
/////////////// colors for masks

///uint colorMasks
    uniform vec4  uIcolorMask0;
    uniform vec4  uIcolorMask1;
    uniform vec4  uIcolorMask2;
    uniform vec4  uIcolorMask3;
    uniform vec4  uIcolorMask4;
    uniform vec4  uIcolorMask5;
    uniform vec4  uIcolorMask6;
    uniform vec4  uIcolorMask7;
    uniform vec4  uIcolorMask8;
//int colorMasks
    uniform vec4  icolorMask0;
    uniform vec4  icolorMask1;
    uniform vec4  icolorMask2;
    uniform vec4  icolorMask3;
    uniform vec4  icolorMask4;
    uniform vec4  icolorMask5;
    uniform vec4  icolorMask6;
    uniform vec4  icolorMask7;
    uniform vec4  icolorMask8;
//float colorMasks
    uniform vec4  fcolorMask0;
    uniform vec4  fcolorMask1;
    uniform vec4  fcolorMask2;
    uniform vec4  fcolorMask3;
    uniform vec4  fcolorMask4;
    uniform vec4  fcolorMask5;
    uniform vec4  fcolorMask6;
    uniform vec4  fcolorMask7;
    uniform vec4  fcolorMask8;
///////////////visibility controll   --  set of booleans that controll weather the texture will affect final image (visible) or not
   uniform bool isVisibleTexture0 = true;
   uniform bool isVisibleNuclearMask = false;
   

   ///uint is visible
  uniform bool uIisVisk0= false;
  uniform bool uIisVisk1= false;
  uniform bool uIisVisk2= false;
  uniform bool uIisVisk3= false;
  uniform bool uIisVisk4= false;
  uniform bool uIisVisk5= false;
  uniform bool uIisVisk6= false;
  uniform bool uIisVisk7= false;
  uniform bool uIisVisk8= false;
//int isVisks
  uniform bool iisVisk0= false;
  uniform bool iisVisk1= false;
  uniform bool iisVisk2= false;
  uniform bool iisVisk3= false;
  uniform bool iisVisk4= false;
  uniform bool iisVisk5= false;
  uniform bool iisVisk6= false;
  uniform bool iisVisk7= false;
  uniform bool iisVisk8= false;
//float isVisks
  uniform bool fisVisk0= false;
  uniform bool fisVisk1= false;
  uniform bool fisVisk2= false;
  uniform bool fisVisk3= false;
  uniform bool fisVisk4= false;
  uniform bool fisVisk5= false;
  uniform bool fisVisk6= false;
  uniform bool fisVisk7= false;
  uniform bool fisVisk8= false;
   
   
   
     
//in case of integer texture  controlling color display of main image we keep all above some value as white and all below some value as black
vec4 imainColor(in int texel)
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

//in case of unsignef integer texture  controlling color display of main image we keep all above some value as white and all below some value as black
vec4 umainColor(in uint texel)
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

//in case of float texture  controlling color display of main image we keep all above some value as white and all below some value as black
vec4 fmainColor(in float texel)
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
      float fla = texel-max_shown_black ;
      float fl = fla/displayRange ;
     return vec4(fl, fl, fl, 1.0);
    }
}




//data types adapted from https://www.shaderific.com/glsl-types
//controlling color output modification by the
//the textures with values that are always between 0 and 1 
//(including binary masks) - if value is greater than 0 and flag of visibility is set to true 
//the color will affect main color otherwise it would be just passed through the function without any modifications
vec4 umaskColor(in uint maskTexel,in vec4 FragColorMain ,in bool isVisible ,in vec4 color  )
{
  if(maskTexel>0 && isVisible) {
       return   color*FragColorMain ;//vec4(maskTexel, 0.0, 0.0, 0.5);
    }
    return FragColorMain;
}

vec4 imaskColor(in int maskTexel,in vec4 FragColorMain ,in bool isVisible ,in vec4 color  )
{
  if(maskTexel>0 && isVisible) {
       return   color*FragColorMain ;//vec4(maskTexel, 0.0, 0.0, 0.5);
    }
    return FragColorMain;
}

vec4 fmaskColor(in float maskTexel,in vec4 FragColorMain ,in bool isVisible ,in vec4 color  )
{
  if(maskTexel>0 && isVisible) {
       return   color*FragColorMain ;//vec4(maskTexel, 0.0, 0.0, 0.5);
    }
    return FragColorMain;
}




    void main()
    {
        vec4 FragColorA; 
        //depending on texture type we need to use diffrent samplers
           if(typeOfMainSampler==1){
                FragColorA = imainColor(texture2D(iTexture0, TexCoord0).r) ; }
            else if(typeOfMainSampler ==2){ 
                FragColorA = umainColor(texture2D(uTexture0, TexCoord0).r); }
            else{
                FragColorA = fmainColor(texture2D(fTexture0, TexCoord0).r); }            
            
    //we pass all masks so all would be available in case the mask would not be used it would not affect the image  
    //uints
    FragColorA=umaskColor(texture2D(uImask0, TexCoord0).r,FragColorA,uIisVisk0, uIcolorMask0);
    FragColorA=umaskColor(texture2D(uImask1, TexCoord0).r,FragColorA,uIisVisk1, uIcolorMask1);
    FragColorA=umaskColor(texture2D(uImask2, TexCoord0).r,FragColorA,uIisVisk2, uIcolorMask2);
    FragColorA=umaskColor(texture2D(uImask3, TexCoord0).r,FragColorA,uIisVisk3, uIcolorMask3);
    FragColorA=umaskColor(texture2D(uImask4, TexCoord0).r,FragColorA,uIisVisk4, uIcolorMask4);
    FragColorA=umaskColor(texture2D(uImask5, TexCoord0).r,FragColorA,uIisVisk5, uIcolorMask5);
    FragColorA=umaskColor(texture2D(uImask6, TexCoord0).r,FragColorA,uIisVisk6, uIcolorMask6);
    FragColorA=umaskColor(texture2D(uImask7, TexCoord0).r,FragColorA,uIisVisk7, uIcolorMask7);
    FragColorA=umaskColor(texture2D(uImask8, TexCoord0).r,FragColorA,uIisVisk8, uIcolorMask8);
  
    //int
    FragColorA=imaskColor(texture2D(imask0, TexCoord0).r,FragColorA,iisVisk0, icolorMask0);
    FragColorA=imaskColor(texture2D(imask1, TexCoord0).r,FragColorA,iisVisk1, icolorMask1);
    FragColorA=imaskColor(texture2D(imask2, TexCoord0).r,FragColorA,iisVisk2, icolorMask2);
    FragColorA=imaskColor(texture2D(imask3, TexCoord0).r,FragColorA,iisVisk3, icolorMask3);
    FragColorA=imaskColor(texture2D(imask4, TexCoord0).r,FragColorA,iisVisk4, icolorMask4);
    FragColorA=imaskColor(texture2D(imask5, TexCoord0).r,FragColorA,iisVisk5, icolorMask5);
    FragColorA=imaskColor(texture2D(imask6, TexCoord0).r,FragColorA,iisVisk6, icolorMask6);
    FragColorA=imaskColor(texture2D(imask7, TexCoord0).r,FragColorA,iisVisk7, icolorMask7);
    FragColorA=imaskColor(texture2D(imask8, TexCoord0).r,FragColorA,iisVisk8, icolorMask8);

    //float
    FragColorA=fmaskColor(texture2D(fmask0, TexCoord0).r,FragColorA,fisVisk0, fcolorMask0);
    FragColorA=fmaskColor(texture2D(fmask1, TexCoord0).r,FragColorA,fisVisk1, fcolorMask1);
    FragColorA=fmaskColor(texture2D(fmask2, TexCoord0).r,FragColorA,fisVisk2, fcolorMask2);
    FragColorA=fmaskColor(texture2D(fmask3, TexCoord0).r,FragColorA,fisVisk3, fcolorMask3);
    FragColorA=fmaskColor(texture2D(fmask4, TexCoord0).r,FragColorA,fisVisk4, fcolorMask4);
    FragColorA=fmaskColor(texture2D(fmask5, TexCoord0).r,FragColorA,fisVisk5, fcolorMask5);
    FragColorA=fmaskColor(texture2D(fmask6, TexCoord0).r,FragColorA,fisVisk6, fcolorMask6);
    FragColorA=fmaskColor(texture2D(fmask7, TexCoord0).r,FragColorA,fisVisk7, fcolorMask7);
    FragColorA=maskColor(texture2D(fmask8, TexCoord0).r,FragColorA,fisVisk8, fcolorMask8);


    FragColor = FragColorA;


    }


  out vec4 FragColor;    
  in vec3 ourColor;
  smooth in vec2 TexCoord0;
  
  uniform int  min_shown_white = 400;//400 ;// value of cut off  - all values above will be shown as white 
  uniform int  max_shown_black = -200;//value cut off - all values below will be shown as black
  uniform float displayRange = 600.0;
  
  uniform isampler2D mainCTImage; // main image sampler
  uniform bool isVisiblemainCTImage = true; // controllin main texture visibility
  
  
  //in case of int texture  controlling color display of main image we keep all above some value as white and all below some value as black
  vec4 mainColor(in int texel)
  {
      if(!isVisiblemainCTImage){
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
  
























[ Info: #extension GL_EXT_gpu_shader4 : enable    //Include support for this extension, which defines usampler2D
[ Info: out vec4 FragColor;
[ Info: in vec3 ourColor;
[ Info: smooth in vec2 TexCoord0;
[ Info: uniform int  min_shown_white = 400;//400 ;// value of cut off  - all values above will be shown as white
[ Info: uniform int  max_shown_black = -200;//value cut off - all values below will be shown as black
[ Info: uniform float displayRange = 600.0;
[ Info: 
[ Info: 
[ Info: 
[ Info: // for mask difference display
[ Info: uniform int isMaskDiffrenceVis=0 ;//1 if we want to display mask difference
[ Info: uniform int maskAIndex=0 ;//marks index of first mask we want to get diffrence visualized
[ Info: uniform int maskBIndex=0 ;//marks index of second mask we want to get diffrence visualized
[ Info: 
[ Info: 
[ Info: uniform isampler2D CTIm; // main image sampler
[ Info: uniform int CTImisVisible = 1; // controllin main texture visibility
[ Info: 
[ Info: 
[ Info: //in case of int texture  controlling color display of main image we keep all above some value as white and all below some value as black
[ Info: vec4 mainColor(in int texel)
[ Info: {
[ Info:     if(CTImisVisible==0){
[ Info:         return vec4(0.0, 0.0, 0.0, 1.0);
[ Info:     }
[ Info:     else if(texel >min_shown_white){
[ Info:         return vec4(1.0, 1.0, 1.0, 1.0);
[ Info:         }
[ Info:     else if (texel< max_shown_black){
[ Info:         return vec4(0.0, 0.0, 0.0, 1.0);
[ Info:    }
[ Info:     else{
[ Info:       float fl = float(texel-max_shown_black)/displayRange ;
[ Info:      return vec4(fl, fl, fl, 1.0);
[ Info:     }
[ Info: }
[ Info: 
[ Info: 
[ Info: 
[ Info: 
[ Info: 
[ Info: uniform usampler2D mainLab; // mask image sampler
[ Info: uniform vec4 mainLabColorMask= vec4(0.0,0.0,0.0,0.0); //controlling colors
[ Info: uniform int mainLabisVisible= 0; // controlling visibility
[ Info: 
[ Info: uniform uint  mainLabminValue= 0; // minimum possible value set in configuration
[ Info: uniform uint  mainLabmaxValue= 1; // maximum possible value set in configuration
[ Info: uniform uint  mainLabValueRange= 1; // range of possible values calculated from above
[ Info: 
[ Info: 
[ Info: uniform usampler2D testLab1; // mask image sampler
[ Info: uniform vec4 testLab1ColorMask= vec4(0.0,0.0,0.0,0.0); //controlling colors
[ Info: uniform int testLab1isVisible= 0; // controlling visibility
[ Info: 
[ Info: uniform uint  testLab1minValue= 0; // minimum possible value set in configuration
[ Info: uniform uint  testLab1maxValue= 1; // maximum possible value set in configuration
[ Info: uniform uint  testLab1ValueRange= 1; // range of possible values calculated from above
[ Info: 
[ Info: 
[ Info: uniform usampler2D testLab2; // mask image sampler
[ Info: uniform vec4 testLab2ColorMask= vec4(0.0,0.0,0.0,0.0); //controlling colors
[ Info: uniform int testLab2isVisible= 0; // controlling visibility
[ Info: 
[ Info: uniform uint  testLab2minValue= 0; // minimum possible value set in configuration
[ Info: uniform uint  testLab2maxValue= 1; // maximum possible value set in configuration
[ Info: uniform uint  testLab2ValueRange= 1; // range of possible values calculated from above
[ Info: 
[ Info: 
[ Info: uniform sampler2D nuclearMask; // mask image sampler
[ Info: uniform vec4[3] nuclearMaskColorMask  = vec4[3](vec4(0.0,0.0,0.0,1.0),vec4(0.0,0.0,1.0 ,1.0),vec4(1.0,0.0,0.0 ,1.0));// we add one so later function operating on this will make easier
[ Info: uniform int nuclearMaskisVisible= 0; // controlling visibility
[ Info: 
[ Info: uniform float  nuclearMaskminValue= 0.0; // minimum possible value set in configuration
[ Info: uniform float  nuclearMaskmaxValue= 2.0; // maximum possible value set in configuration
[ Info: uniform float  nuclearMaskValueRange= 2.0; // range of possible values calculated from above
[ Info: 
[ Info: 
[ Info: 
[ Info: 
[ Info: float rnuclearMaskgetColorForMultiColor(float texelRes) {
[ Info:         float normalized = (texelRes/float(nuclearMaskValueRange))*3;
[ Info:         uint index = uint(floor(normalized)) ;// so we normalize floor  in order to get index of color from color list
[ Info:         float[3] colorFloats = float[3](0.0,0.0,1.0   )  ;
[ Info:         float normalizedColorPercent= float(index)-normalized ;
[ Info:         return texelRes;//colorFloats[index +1]*normalizedColorPercent+colorFloats[index]*(1- normalizedColorPercent);//so we get color from current section and closer we are to the end of this section the bigger influence of this color, the closer we are to the begining the bigger the influence of previous section color
[ Info: }
[ Info: 
[ Info: 
[ Info: 
[ Info: float gnuclearMaskgetColorForMultiColor(float texelRes) {
[ Info:         float normalized = (texelRes/float(nuclearMaskValueRange))*3;
[ Info:         uint index = uint(floor(normalized)) ;// so we normalize floor  in order to get index of color from color list
[ Info:         float[3] colorFloats = float[3](0.0,0.0,0.0   )  ;
[ Info:         float normalizedColorPercent= float(index)-normalized ;
[ Info:         return texelRes;//colorFloats[index +1]*normalizedColorPercent+colorFloats[index]*(1- normalizedColorPercent);//so we get color from current section and closer we are to the end of this section the bigger influence of this color, the closer we are to the begining the bigger the influence of previous section color
[ Info: }
[ Info: 
[ Info: 
[ Info: 
[ Info: float bnuclearMaskgetColorForMultiColor(float texelRes) {
[ Info:         float normalized = (texelRes/float(nuclearMaskValueRange))*3;
[ Info:         uint index = uint(floor(normalized)) ;// so we normalize floor  in order to get index of color from color list
[ Info:         float[3] colorFloats = float[3](0.0,1.0,0.0   )  ;
[ Info:         float normalizedColorPercent= float(index)-normalized ;
[ Info:         return texelRes;//colorFloats[index +1]*normalizedColorPercent+colorFloats[index]*(1- normalizedColorPercent);//so we get color from current section and closer we are to the end of this section the bigger influence of this color, the closer we are to the begining the bigger the influence of previous section color
[ Info: }
[ Info: 
[ Info: 
[ Info: 
[ Info: float rdiffrenceColor(uint maskkA,uint maskkB)
[ Info: {
[ Info:   return   max(float(maskkA)-float(maskkB),0.0)*((mainLabColorMask.r + testLab1ColorMask.r)/2)*isMaskDiffrenceVis;
[ Info: }
[ Info: 
[ Info: 
[ Info: float gdiffrenceColor(uint maskkA,uint maskkB)
[ Info: {
[ Info:   return   max(float(maskkA)-float(maskkB),0.0)*((mainLabColorMask.g + testLab1ColorMask.g)/2)*isMaskDiffrenceVis;
[ Info: }
[ Info: 
[ Info: 
[ Info: float bdiffrenceColor(uint maskkA,uint maskkB)
[ Info: {
[ Info:   return   max(float(maskkA)-float(maskkB),0.0)*((mainLabColorMask.b + testLab1ColorMask.b)/2)*isMaskDiffrenceVis;
[ Info: }
[ Info: 
[ Info: 
[ Info: 
[ Info: 
[ Info:     void main()
[ Info:     {
[ Info: 
[ Info: uint mainLabRes = texture2D(mainLab, TexCoord0).r * mainLabisVisible  ;
[ Info: 
[ Info: 
[ Info: uint testLab1Res = texture2D(testLab1, TexCoord0).r * testLab1isVisible  ;
[ Info: 
[ Info: 
[ Info: uint testLab2Res = texture2D(testLab2, TexCoord0).r * testLab2isVisible  ;
[ Info: 
[ Info: 
[ Info: float nuclearMaskRes = texture2D(nuclearMask, TexCoord0).r * nuclearMaskisVisible  ;
[ Info: 
[ Info: 
[ Info: 
[ Info: 
[ Info: uint todiv =  mainLabisVisible  +  testLab1isVisible  +  testLab2isVisible  +  nuclearMaskisVisible + CTImisVisible+ isMaskDiffrenceVis;
[ Info:  vec4 CTImRes = mainColor(texture2D(CTIm, TexCoord0).r);
[ Info:    FragColor = vec4(( (mainLabColorMask.r *  mainLabRes)  +  (testLab1ColorMask.r *  testLab1Res)  +  (testLab2ColorMask.r *  testLab2Res) +rnuclearMaskgetColorForMultiColor(nuclearMaskRes)
[ Info:    +CTImRes.r+rdiffrenceColor(texture2D(mainLab, TexCoord0).r ,texture2D(testLab1, TexCoord0).r )
[ Info:     )/todiv
[ Info:    ,( (mainLabColorMask.g  * mainLabRes)  +  (testLab1ColorMask.g  * testLab1Res)  +  (testLab2ColorMask.g  * testLab2Res) +gnuclearMaskgetColorForMultiColor(nuclearMaskRes)
[ Info:    +CTImRes.g+gdiffrenceColor(texture2D(mainLab, TexCoord0).r ,texture2D(testLab1, TexCoord0).r ))
[ Info:    /todiv,
[ Info:    ( (mainLabColorMask.b  * mainLabRes)  +  (testLab1ColorMask.b  * testLab1Res)  +  (testLab2ColorMask.b  * testLab2Res) +bnuclearMaskgetColorForMultiColor(nuclearMaskRes)
[ Info:    +CTImRes.b+bdiffrenceColor(texture2D(mainLab, TexCoord0).r ,texture2D(testLab1, TexCoord0).r ) )
[ Info:    /todiv,
[ Info:    1.0  ); //long  product, if mask is invisible it just has full transparency
[ Info: 
[ Info:     }
[ Info: 
[ Info: 
[ Info: 