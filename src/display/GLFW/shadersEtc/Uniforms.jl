
"""
managing  uniform values - global values in shaders
"""
module Uniforms
using Glutils,  ..ForDisplayStructs, Dictionaries, Parameters, ColorTypes

export coontrolMinMaxUniformVals,createStructsDict, setCTWindow,setMaskColor,setTextureVisibility, setTypeOfMainSampler!



"""
function cotrolling the window  for displaying CT scan  - min white and max max_shown_black
    uniformsStore - instantiated object holding references to uniforms controlling displayed window
 """
function setCTWindow(min_shown_whiteInner::Int32, max_shown_blackInner::Int32, uniformsStore ::MainImageUniforms)
    @uniforms! begin
    uniformsStore.min_shown_white:=min_shown_whiteInner
    uniformsStore.max_shown_black:=max_shown_blackInner
    uniformsStore.displayRange:= Float32(min_shown_whiteInner-max_shown_blackInner )
    end
end

"""
sets color of the mask

"""
function setMaskColor(color::RGB, uniformsStore ::MaskTextureUniforms)
    @uniforms! begin
    uniformsStore.colorsMaskRef:=Cfloat[color.r, color.g, color.b, 0.8]
    end

end#setMaskColor


"""
sets visibility of the texture
"""
function setTextureVisibility(isvisible::Bool, uniformsStore ::TextureUniforms)
    @uniforms! begin
    uniformsStore.isVisibleRef:= isvisible ? 1 : 0
    end

end#setTextureVisibility


"""
sets minimum and maximum value for display - 
    in case of continuus colors it will clamp values - so all above max will be equaled to max ; and min if smallert than min
    in case of main CT mask - it will controll min shown white and max shown black
    in case of maks with single color associated we will step data so if data is outside the rande it will return 0 - so will not affect display
"""
function coontrolMinMaxUniformVals(textur::TextureSpec
                                ,uniformsStore ::MaskTextureUniforms)
    newMin=textur.minAndMaxValue[1]
    newMax=textur.minAndMaxValue[2]
                                
    @uniforms! begin
    uniformsStore.maskMinValue:= newMin
    uniformsStore.maskMAxValue:= newMax
    uniformsStore.maskRangeValue:= newMax-newMin
    end

end#coontrolMinMaxUniformVals


end #module




