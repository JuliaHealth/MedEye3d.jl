using DrWatson
@quickactivate "Probabilistic medical segmentation"

"""
functions that will enable creation of long String that will be the code for custom fragment shader that will be suited for defined textures
"""
module CustomFragShad

using ModernGL, GeometryTypes, GLFW, Main.ForDisplayStructs

export createCustomFramgentShader,divideTexteuresToMainAndRest,addSamplerStr


```@doc
We divide textures into main imae texture and the rest
listOfTexturesToCreate - list of textures on the basis of which we will create custom fragment shader code
returns 3tuple where fist entr is the main image texture specification and second is rest of textures
  ```
function divideTexteuresToMainAndRest(listOfTexturesToCreate::Vector{TextureSpec})::Tuple{TextureSpec, Vector{TextureSpec}}
    mainTexture =   filter(it->it.isMainImage,listOfTexturesToCreate )[1] # main image texture
    notMainTextures = filter(it->!it.isMainImage,listOfTexturesToCreate) # textures associated with not main
return (mainTexture, notMainTextures)
end #divideTexteuresToMainAndRest




```@doc
We will in couple steps create code for fragment shader that will be based on the texture definitions we gave 
listOfTexturesToCreate - list of textures on the basis of which we will create custom fragment shader code

  ```
function createCustomFramgentShader(listOfTexturesToCreate::Vector{TextureSpec})::String
    mainTexture, notMainTextures=divideTexteuresToMainAndRest(listOfTexturesToCreate)
masksConstsants= map( x-> addMasksStrings(x), notMainTextures) |> 
(strings)-> join(strings, "")


 return """
$(initialStrings())
$(addMainTextureStrings(mainTexture))
$masksConstsants
$(mainFuncString(mainTexture,notMainTextures))
 """
end #createCustomFramgentShader

```@doc
some initial constants that are the same irrespective of textures
```
function initialStrings()::String
return """ 
#extension GL_EXT_gpu_shader4 : enable    //Include support for this extension, which defines usampler2D
out vec4 FragColor;    
in vec3 ourColor;
smooth in vec2 TexCoord0;
uniform int  min_shown_white = 400;//400 ;// value of cut off  - all values above will be shown as white 
uniform int  max_shown_black = -200;//value cut off - all values below will be shown as black
uniform float displayRange = 600.0;
"""
end #initialStrings

```@doc
managing main texture and on the basis of the type  we will initialize diffrent variables
mainTexture - specification of a texture related to main image
  ```
function addMainTextureStrings(mainTexture::TextureSpec)::String
    mainImageName= mainTexture.name
    return """
    uniform $(addSamplerStr(mainTexture, mainImageName)); // main image sampler
    uniform bool isVisible$(mainImageName) = true; // controllin main texture visibility

    $(addWindowingFunc(mainTexture))
    """
end #addMainTextureStrings


```@doc
setting string representing sampler depending on type
```
function addSamplerStr(textur::TextureSpec, samplerName::String)::String
   
    if supertype(Int)== supertype(textur.dataType)
        return "isampler2D $(samplerName)"
    elseif supertype(UInt)== supertype(textur.dataType)
        return "usampler2D $(samplerName)"
    else # so float ...
        return "sampler2D $(samplerName)"
    end
end #addSamplerStr


```@doc
giving variable name associated with given type```
function addTypeStr(textur::TextureSpec)::String
   
    if supertype(Int)== supertype(textur.dataType)
        return "int"
    elseif supertype(UInt)== supertype(textur.dataType)
        return "uint"
    else # so float ...
        return "float"
    end
end #addTypeStr



```@doc
giving string representing main function defining how windowing of main image should be performed
```
function addWindowingFunc(textur::TextureSpec)::String
    typeStr = addTypeStr(textur)
    mainImageName= textur.name

   return """

   //in case of $typeStr texture  controlling color display of main image we keep all above some value as white and all below some value as black
   vec4 mainColor(in $typeStr texel)
   {
       if(!isVisible$(mainImageName)){
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

   """

end #addWindowingFunc



```@doc
Adding string necessary for managing of masks textures
```
function addMasksStrings(textur::TextureSpec)
    textName = textur.name
return """

uniform $(addSamplerStr(textur,textName )); // mask image sampler
uniform vec4 $(textName)ColorMask; //controlling colors
uniform bool $(textName)isVisible= false; // controlling visibility

"""
end#addMasksStrings



```@doc
controlling main function - basically we need to return proper FragColor which represents pixel color in given spot
```
function mainFuncString( mainTexture::TextureSpec,notMainTextures::Vector{TextureSpec})::String

    mainImageName= mainTexture.name

    masksInfluences= map( x-> setMaskInfluence(x), notMainTextures) |> 
                    (strings)-> join(strings, "")

    return """
    void main()
    {        
    vec4 FragColorA; 
    FragColorA = mainColor(texture2D($(mainImageName), TexCoord0).r) ; 
    $(masksInfluences)

    FragColor = FragColorA;
    }

    """
end#mainFuncString




```@doc
setting conditional logic of masks - it should not affect final color if the associatedvaleue is above 0 and is visible string is set to true
```
function setMaskInfluence(textur::TextureSpec)
    textName = textur.name

return """

if( texture2D($(textName), TexCoord0).r >0 && $(textName)isVisible==true) {
    FragColorA= $(textName)ColorMask * FragColorA;
 }

"""
end#xxx


```@doc
used to display and debug  output - output can be also additionally checked using this tool http://evanw.github.io/glslx/


```
function debuggingOutput(listOfTexturesToCreate)
    strr= Main.CustomFragShad.createCustomFramgentShader(listOfTexturesToCreate)
    for st in split(strr, "\n")
    @info st
    end
end#



end #CustomFragShad module