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
function createCustomFramgentShader(listOfTexturesToCreate::Vector{TextureSpec}
                                    ,maskToSubtrastFrom::TextureSpec
                                    ,maskWeAreSubtracting::TextureSpec
    
                                     )::String
    mainTexture, notMainTextures=divideTexteuresToMainAndRest(listOfTexturesToCreate)
masksConstsants= map( x-> addMasksStrings(x), notMainTextures) |> 
(strings)-> join(strings, "")

#$(addMaskColorFunc())

 return """
$(initialStrings())


$(addUniformsForNuclearAndSubtr())

$(addMainTextureStrings(mainTexture))
$masksConstsants

$( getMasksSubtractionFunction(maskToSubtrastFrom,maskWeAreSubtracting))


$(mainFuncString(mainTexture,notMainTextures,maskToSubtrastFrom,maskWeAreSubtracting))
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
    uniform int $(mainImageName)isVisible = 1; // controllin main texture visibility

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
       if($(mainImageName)isVisible==0){
           return vec4(0.0, 0.0, 0.0, 1.0);
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
uniform vec4 $(textName)ColorMask= vec4(0.4,0.7,0.8,0.9); //controlling colors
uniform int $(textName)isVisible= 0; // controlling visibility

"""
end#addMasksStrings



```@doc
controlling main function - basically we need to return proper FragColor which represents pixel color in given spot
```
function mainFuncString( mainTexture::TextureSpec
                        ,notMainTextures::Vector{TextureSpec}
                        ,maskToSubtrastFrom::TextureSpec
                        ,maskWeAreSubtracting::TextureSpec)::String

    mainImageName= mainTexture.name

    masksInfluences= map( x-> setMaskInfluence(x), notMainTextures) |> 
                    (strings)-> join(strings, "")

   lll = length(notMainTextures)+1                 
   
   sumColorR= map( x-> " ($(x.name)ColorMask.r *  $(x.name)Res) ", notMainTextures) |> 
                    (strings)-> join(strings, " + ")     
   sumColorG= map( x-> " ($(x.name)ColorMask.g  * $(x.name)Res) ", notMainTextures) |> 
                    (strings)-> join(strings, " + ")     
   sumColorB= map( x-> " ($(x.name)ColorMask.b  * $(x.name)Res) ", notMainTextures) |> 
                    (strings)-> join(strings, " + ")     

   isVisibleList=  map( x-> " $(x.name)isVisible ", notMainTextures) |> 
   (strings)-> join(strings, " + ")                  
    return """
    void main()
    {      
$(masksInfluences)


uint todiv = $(isVisibleList)+ $(mainImageName)isVisible+ isMaskDiffrenceVis;
 vec4 $(mainImageName)Res = mainColor(texture2D($(mainImageName), TexCoord0).r);
  FragColor = vec4(($(sumColorR)
  +$(mainImageName)Res.r+rdiffrenceColor($(maskToSubtrastFrom.name)Res ,$(maskWeAreSubtracting.name)Res )
   )/todiv
  ,($(sumColorG)
  +$(mainImageName)Res.g+gdiffrenceColor($(maskToSubtrastFrom.name)Res ,$(maskWeAreSubtracting.name)Res ))
  /todiv, 
  ($(sumColorB)
  +$(mainImageName)Res.b+bdiffrenceColor($(maskToSubtrastFrom.name)Res ,$(maskWeAreSubtracting.name)Res ) )
  /todiv, 
  1.0  ); //long  product, if mask is invisible it just has full transparency

    }

    """
end#mainFuncString


```@doc
setting conditional logic of masks - it should not affect final color if the associatedvaleue is above 0 and is visible string is set to true
```
function setMaskInfluence(textur::TextureSpec)
    textName = textur.name

return """

$(addTypeStr(textur)) $(textName)Res = texture2D($(textName), TexCoord0).r * $(textName)isVisible  ;

"""
end#xxx



```@doc
on the basis of texture type gives proper function controlling color of the mask 
```
function chooseColorFonuction(textur::TextureSpec)::String

    if supertype(Int)== supertype(textur.dataType)
        return "imaskColor"
    elseif supertype(UInt)== supertype(textur.dataType)
        return "umaskColor"
    else # so float ...
        return "fmaskColor"
    end
end#chooseColorFonuction






```@doc
used to display and debug  output - output can be also additionally checked using this tool http://evanw.github.io/glslx/
```
function debuggingOutput(listOfTexturesToCreate)
    strr= Main.CustomFragShad.createCustomFramgentShader(listOfTexturesToCreate)
    for st in split(strr, "\n")
    @info st
    end
end#


getMasksSubtractionFunctionStr= """
used in order to enable subtracting one mask from the other - hence displaying 
pixels where value of mask a is present but mask b not (order is important)
automatically both masks will be set to be invisible and only the diffrence displayed

In order to achieve this  we need to have all of the samplers references stored in a list 
1) we need to set both masks yo invisible - it will be done from outside the shader
2) we set also from outside uniform marking visibility of diffrence to true
3) also from outside we need to set which texture to subtract from which we will achieve this by setting maskAtoSubtr and maskBtoSubtr int uniforms
    those integers will mark which samplers function will use
4) in shader function will be treated as any other mask and will give contribution to output color multiplied by its visibility(0 or 1)    
5) inside the function color will be defined as multiplication of two colors of mask A and mask B - colors will be acessed similarly to samplers
6) color will be returned only if value associated with  maskA is greater than mask B and proportional to this difffrence

In order to provide maximum performance and avoid branching inside shader multiple shader programs will be attached and one choosed  that will use diffrence needed
maskToSubtrastFrom,maskWeAreSubtracting - specifications o textures we are operating on 
"""
@doc getMasksSubtractionFunctionStr
function getMasksSubtractionFunction(maskToSubtrastFrom::TextureSpec
                                    ,maskWeAreSubtracting::TextureSpec)::String

  #letter - r g or b - will create specialized functions for r g and b colors


return map(letter-> """

float $(letter)diffrenceColor($(addTypeStr(maskToSubtrastFrom)) maskkA,$(addTypeStr(maskWeAreSubtracting)) maskkB)
{
  return 0.0;//(($(maskToSubtrastFrom.name)ColorMask.$(letter) + $(maskWeAreSubtracting.name)ColorMask.$(letter))/2) *( maskkA-maskkB  );
}

""",["r","g","b"]  )|>  x->join(x)

end #getMasksSubtractionFunction

getNuclearMaskFunctionStr =  """
Enable displaying nuclear medicine data by applying smoothly canging colors
to floating point data
1)in confguration phase we need to have minimum, maximum and range of possible values associated with nuclear mask
2)as uniform we would need to have set of vec4's - those will be used  to display colors 
3)colors will be set with algorithm presented below
    a)range will be divided into sections (value ranges) wich numbers will equal  length of colors vector - 1
    b)in each section  the output color will be mix of 2 colors one associated with this section and with next one 
        - contribution  of the color associated with given section will  vary from 100% at the begining of the section to 0%   
        in the end where 100% of color will be associated with color of next section
    
"""
@doc getNuclearMaskFunctionStr
function getNuclearMaskFunction()

end#getNuclearMaskFunction








addUniformsForNuclearAndSubtrStr= """
Add uniforms necessary for operation of mask subtraction and  for proper display of nuclear medicine masks
"""
@doc addUniformsForNuclearAndSubtrStr
function addUniformsForNuclearAndSubtr()::String
return """
// for mask difference display
uniform int isMaskDiffrenceVis=0 ;//1 if we want to display mask difference
uniform int maskAIndex=0 ;//marks index of first mask we want to get diffrence visualized
uniform int maskBIndex=0 ;//marks index of second mask we want to get diffrence visualized
//for nuclear mask properdisplay
uniform float minNuclearMaskVal = 0.0;//minimum possible value of nuclear mask
uniform float maxNuclearMaskVal = 0.0;//maximum possible value of nuclear mask
uniform float rangeOfNuclearMaskVal = 0.0;//precalculated maximum - minimum  possible values of nuclear mask
uniform sampler2D nuclearMaskSampler;
uniform int isNuclearMaskVis = 0;	

"""	
end#addUniformsForNuclearAndSubtr




# createReferenceListsStr= """
# On the basis of the list of textures characteristics  it creates the vector of references to 
# Samplers
# Colors associated with samplers
# """
# @doc createReferenceListsStr
# function createReferenceLists(listOfTexturesTocreate::Vector{TextureSpec})::String
# 	samplerStrings= map(textSpec->  textSpec.name  ,listOfTexturesTocreate ) |> (samplers)->  join(samplers, ",")
# 	colorStrings= map(textSpec-> "$(textSpec.name)ColorMask"  ,listOfTexturesTocreate ) |> (colors)->  join(colors, ",")
# lengthh=  length(listOfTexturesTocreate)

# return """
# uniform vec4[] colorsArr = vec4[$(lengthh)]($(colorStrings));
# uniform gsampler2D[] samplersArr = gsampler2D[$(lengthh)]($(samplerStrings));//gsampler2D is a supertype of usampler2d isampler2d ...
# """
# end #createReferenceLists



# ```@doc
# adding function enabling controll of the masks color
# ```
# function addMaskColorFunc()
# return """


# vec4 umaskColor(in uint maskTexel ,in bool isVisible ,in vec4 color  )
# {
#   if(maskTexel>0.0 && isVisible==true) {
#        return   color;
#     }
#     return vec4(0.0, 0.0, 0.0, 0.0);
#     }

# vec4 imaskColor(in int maskTexel,in bool isVisible ,in vec4 color  )
# {
#   if(maskTexel>0.0 && isVisible==true) {
#     return   color;
#     }
#     return vec4(0.0, 0.0, 0.0, 0.0);
#     }

# vec4 fmaskColor(in float maskTexel,in bool isVisible ,in vec4 color  )
# {
#   if(maskTexel>0.0 && isVisible==true) {
#     return   color;
#     }
#     return vec4(0.0, 0.0, 0.0, 0.0);
# }


# """


# end#addMaskColorFunc


end #CustomFragShad module