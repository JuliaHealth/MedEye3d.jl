"""
functions that will enable creation of long String that will be the code for custom fragment shader that will be suited for defined textures
"""
module CustomFragShad

using ModernGL, GeometryTypes, GLFW, ColorTypes
using ..ForDisplayStructs

export createCustomFramgentShader, divideTexteuresToMainAndRest, addSamplerStr



"""
We will in couple steps create code for fragment shader that will be based on the texture definitions we gave
listOfTexturesToCreate - list of textures on the basis of which we will create custom fragment shader code
maskToSubtrastFrom,maskWeAreSubtracting - texture specifications used in order to generate code needed to diplay diffrence between those two masks - every time we want diffrent diffrence we will need to recreate shader invoking this function
"""
function createCustomFramgentShader(listOfTexturesToCreate::Vector{TextureSpec})::String


    lengthOfTextures = length(listOfTexturesToCreate)
    masksConstants = map(x -> addMasksStrings(x, lengthOfTextures), listOfTexturesToCreate) |> (strings) -> join(strings, "")

    continuusColorMasks = filter(it -> it.isContinuusMask, listOfTexturesToCreate) # main image texture

    res = """
  $(initialStrings())
  $masksConstants
  $(mainFuncString(listOfTexturesToCreate))
   """
    # uncomment for debugging
    #   for st in split(res, "\n")
    #     @info st
    #     end
    return res

end #createCustomFramgentShader

"""
some initial constants that are the same irrespective of textures
very begining taken from :
    https://community.khronos.org/t/problem-with-layout-syntax/69034/5
"""
function initialStrings()::String
    return """
    out vec4 FragColor;
    in vec3 ourColor;
    smooth in vec2 TexCoord0;
    """
end #initialStrings



"""
setting string representing sampler depending on type
"""
function addSamplerStr(textur::TextureSpec, samplerName::String)::String

    if supertype(Int) == supertype(parameter_type(textur))
        return "isampler2D $(samplerName)"
    elseif supertype(UInt) == supertype(parameter_type(textur))
        return "usampler2D $(samplerName)"
    else # so float ...
        return "sampler2D $(samplerName)"
    end
end #addSamplerStr


"""
giving variable name associated with given type
"""
function addTypeStr(textur::TextureSpec)::String

    if supertype(Int) == supertype(parameter_type(textur))
        return "int"
    elseif supertype(UInt) == supertype(parameter_type(textur))
        return "uint"
    else # so float ...
        return "float"
    end
end #addTypeStr




"""
Adding string necessary for managing ..Uniforms of masks textures
"""
function addMasksStrings(textur::TextureSpec, lengthOfTextures)
    textName = textur.name
    return """

    uniform $(addSamplerStr(textur,textName )); // mask image sampler
    $(addColorUniform(textur))
    uniform int $(textName)isVisible= 0; // controlling visibility

    uniform $(addTypeStr(textur))  $(textName)minValue= $(textur.minAndMaxValue[1]); // minimum possible value set in configuration
    uniform $(addTypeStr(textur))  $(textName)maxValue= $(textur.minAndMaxValue[2]); // maximum possible value set in configuration
    uniform $(addTypeStr(textur))  $(textName)ValueRange= $(textur.minAndMaxValue[2] -textur.minAndMaxValue[1] ); // range of possible values calculated from above
    uniform float  $(textName)maskContribution=$(1/lengthOfTextures); //controls contribution of mask to output color

    """
end#addMasksStrings


"""
Adding ..Uniforms resopnsible for colors associated with given mask
"""
function addColorUniform(textur::TextureSpec)
    #in case of multiple colors used by single mask like in case of nuclearm medicine mask
    if (textur.isContinuusMask)
        # colors = textur.colorSet
        # colLen= length(colors)+1
        # colorStrings = join(map(it ->"vec4($(it.r),$(it.g),$(it.b) ,1.0)" , colors) ,",")
        # return "uniform vec4[$(colLen)] $(textur.name)ColorMask  = vec4[$(colLen)](vec4(0.0,0.0,0.0,1.0),$(colorStrings));// we add one so later function operating on this will make easier"
    end
    #in case mask uses only single color
    return "uniform vec4 $(textur.name)ColorMask= vec4(0.0,0.0,0.0,0.0); //controlling colors"


end#addColorUniform



"""
controlling main function - basically we need to return proper FragColor which represents pixel color in given spot
we generete separately r,g and b values by adding contributions from all textures
"""
function mainFuncString(textures::Vector{TextureSpec})::String



    texturesNotCont = filter(it -> !it.isContinuusMask, textures)#only single color associated
    texturesCont = filter(it -> length(it.colorSet) > 1, textures)#multiple colors associated

    masksInfluences = map(x -> setMaskInfluence(x), textures) |>
                      (strings) -> join(strings, "")


    #The step function returns 0.0 if x is smaller than edge and otherwise 1.0.

    # sumColors = map(letter ->
    #         map(x -> "  $(x.name)ColorMask.$(letter) * ( $(x.name)Res  * step($(x.name)minValue,$(x.name)Res) )/ $(x.name)ValueRange ", texturesNotCont) |>
    #         (strings) -> join(strings, " + "), ["r", "g", "b"])


    # sumColorR = sumColors[1]
    # sumColorG = sumColors[2]
    # sumColorB = sumColors[3]

    sumColors = map(letter ->
            map(x -> "  changeClip($(x.name)minValue,$(x.name)maxValue,$(x.name)Res,$(x.name)ColorMask.$(letter),$(x.name)ValueRange) ", texturesNotCont) |>
            (strings) -> join(strings, " + "), ["r", "g", "b"])


    sumColorR = sumColors[1]
    sumColorG = sumColors[2]
    sumColorB = sumColors[3]
    sumColorRCont = map(x -> "r$(x.name)getColorForMultiColor($(x.name)Res)", texturesCont) |>
                    (strings) -> join(strings, " + ")
    sumColorGCont = map(x -> "g$(x.name)getColorForMultiColor($(x.name)Res)", texturesCont) |>
                    (strings) -> join(strings, " + ")
    sumColorBCont = map(x -> "b$(x.name)getColorForMultiColor($(x.name)Res)", texturesCont) |>
                    (strings) -> join(strings, " + ")

    # Initialization sumColorRCont, sumColorGCont, sumColorBCont to 0.0 if they return an empty string
    if isempty(sumColorRCont) || isempty(sumColorBCont) || isempty(sumColorGCont)
        sumColorRCont = "0.0"
        sumColorGCont = "0.0"
        sumColorBCont = "0.0"

    elseif isempty(sumColorR) || isempty(sumColorB) || isempty(sumColorG)
        sumColorR = "0.0"
        sumColorG = "0.0"
        sumColorB = "0.0"

    end

    isVisibleList = map(x -> "$(x.name)isVisible *$(x.name)maskContribution", textures) |>
                    (strings) -> join(strings, " + ")

    return """
float changeClip(float min, float max, float value, float color, float range) {
    if (value < min) {
        return min;
    } else if (value > max) {
        return max;
    } else {
        return color * (value/ range);
    }
}
$(getMultiColorMaskFunctions(texturesCont))

void main() {
$(masksInfluences)

float todiv = $(isVisibleList);
FragColor = vec4(($(sumColorR) + $(sumColorRCont)) / todiv,
                 ($(sumColorG) + $(sumColorGCont)) / todiv,
                 ($(sumColorB) + $(sumColorBCont)) / todiv,
                 1.0); // long product, if mask is invisible it just has full transparency
}
"""
end#mainFuncString











"""
Giving value from texture f the texture is set to be visible otherwise 0
"""
function setMaskInfluence(textur::TextureSpec)
    textName = textur.name

    return """

    float $(textName)Res = texture2D($(textName), TexCoord0).r * $(textName)isVisible*$(textName)maskContribution ;

    """
end#setMaskInfluence



"""
on the basis of texture type gives proper function controlling color of the mask
"""
function chooseColorFonuction(textur::TextureSpec)::String

    if supertype(Int) == supertype(parameter_type(textur))
        return "imaskColor"
    elseif supertype(UInt) == supertype(parameter_type(textur))
        return "umaskColor"
    else # so float ...
        return "fmaskColor"
    end
end#chooseColorFonuction






# """
# used to display and debug  output - output can be also additionally checked using this tool http://evanw.github.io/glslx/
# """
# function debuggingOutput(listOfTexturesToCreate)
#     strr=  ..CustomFragShad.createCustomFramgentShader(listOfTexturesToCreate)
#     for st in split(strr, "\n")
#     @info st
#     end
# end#


# """
# used in order to enable subtracting one mask from the other - hence displaying
# pixels where value of mask a is present but mask b not (order is important)
# automatically both masks will be set to be invisible and only the diffrence displayed


# In order to provide maximum performance and avoid branching inside shader multiple shader programs will be attached and one choosed  that will use diffrence needed
# maskToSubtrastFrom,maskWeAreSubtracting - specifications o textures we are operating on
# """
# function getMasksSubtractionFunction(maskToSubtrastFrom::TextureSpec, maskWeAreSubtracting::TextureSpec)::String

#     #letter - r g or b - will create specialized functions for r g and b colors

#     return map(letter -> """

#      float $(letter)diffrenceColor($(addTypeStr(maskToSubtrastFrom)) maskkA,$(addTypeStr(maskWeAreSubtracting)) maskkB)
#      {
#        return   max(float(maskkA)-float(maskkB),0.0)*(($(maskToSubtrastFrom.name)ColorMask.$(letter) + $(maskWeAreSubtracting.name)ColorMask.$(letter))/2)*isMaskDiffrenceVis;
#      }

#      """, ["r", "g", "b"]) |> x -> join(x)

# end #getMasksSubtractionFunction
# """
# Enable displaying for example nuclear medicine data by applying smoothly changing colors
# to floating point data
# 1)in confguration phase we need to have minimum, maximum and range of possible values associated with nuclear mask
# 2)as uniform we would need to have set of vec4's - those will be used  to display colors
# 3)colors will be set with algorithm presented below
#     a)range will be divided into sections (value ranges) wich numbers will equal  length of colors vector - 1
#     b)in each section  the output color will be mix of 2 colors one associated with this section and with next one
#         - contribution  of the color associated with given section will  vary from 100% at the begining of the section to 0%
#         in the end where 100% of color will be associated with color of next section

# """
function getMultiColorMaskFunctions(continuusColorTextSpecs::Vector{TextureSpec})::String
    #Check in which range we are without if
    #Important first color in color list needs to be doubled in order to make algorithm cleaner - so we will start from index 1 and always there would be some previous index

    tuples = map(x -> [(x.name, [[a.r, a.g, a.b] for a in x.colorSet], "r", 1), (x.name, [[a.r, a.g, a.b] for a in x.colorSet], "g", 2), (x.name, [[a.r, a.g, a.b] for a in x.colorSet], "b", 3)], continuusColorTextSpecs)

    if (!isempty(tuples))
        tuples = reduce(vcat, tuples)
    end


    return join(map(x -> """

   float $(x[3])$(x[1])getColorForMultiColor(float innertexelRes) {


    float texelRes=  clamp(innertexelRes, $(x[1])minValue,$(x[1])maxValue );
           float normalized = (texelRes/float($(x[1])ValueRange))*$(length(x[2])+1);
           uint indexx = uint(floor(normalized)) ;// so we normalize floor  in order to get index of color from color list
           float[$(length(x[2])+1)] colorFloats = float[$(length(x[2])+1)](0.0,$( map(it->it[x[4]],x[2])|> (fls)-> join(fls,",")    )   )  ;
           float normalizedColorPercent= normalized-float(indexx) ;
           //return  clamp( colorFloats[indexx-1]*(1- normalizedColorPercent),0.0,1.0 );//so we get color from current section and closer we are to the end of this section the bigger influence of this color, the closer we are to the begining the bigger the influence of previous section color
           return  clamp(colorFloats[indexx]*normalizedColorPercent  +  colorFloats[ int(clamp(float(indexx-1),0.0,100.0))]*(1- normalizedColorPercent) ,0.0,1.0 );//so we get color from current section and closer we are to the end of this section the bigger influence of this color, the closer we are to the begining the bigger the influence of previous section color

           }

           """, tuples), " ")



end#getNuclearMaskFunction


# if( innertexelRes < $(x[1])minValue  ){
#     texelRes = $(x[1])minValue;
#       }
#       else if(innertexelRes > $(x[1])maxValue ){
#     texelRes = $(x[1])maxValue ;
#           }
#       else{
#     texelRes = innertexelRes ;
#    }





# """
# Add ..Uniforms necessary for operation of mask subtraction and  for proper display of nuclear medicine masks
# """
# function addUniformsForNuclearAndSubtr()::String
#     return """
#     // for mask difference display
#     uniform int isMaskDiffrenceVis=0 ;//1 if we want to display mask difference
#     //uniform int maskAIndex=0 ;//marks index of first mask we want to get diffrence visualized
#     //uniform int maskBIndex=0 ;//marks index of second mask we want to get diffrence visualized
#     """
# end#add..UniformsForNuclearAndSubtr






end #..CustomFragShad module


