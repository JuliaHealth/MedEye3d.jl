
"""
managing  uniform values - global values in shaders
"""
module Uniforms
using Base: Int32, isvisible
using Glutils
using Main.ForDisplayStructs
using Dictionaries
using Parameters
using ColorTypes

export createStructsDict, setCTWindow,setMaskColor,setTextureVisibility, setTypeOfMainSampler!






```@doc
function cotrolling the window  for displaying CT scan  - min white and max max_shown_black
    uniformsStore - instantiated object holding references to uniforms controlling displayed window
 ```
function setCTWindow(min_shown_whiteInner::Int32, max_shown_blackInner::Int32, uniformsStore ::MainImageUniforms)
    @uniforms! begin
    uniformsStore.min_shown_white:=min_shown_whiteInner
    uniformsStore.max_shown_black:=max_shown_blackInner
    uniformsStore.displayRange:= Float32(min_shown_whiteInner-max_shown_blackInner )
    end
end

```@doc
sets color of the mask

```
function setMaskColor(color::RGB, uniformsStore ::MaskTextureUniforms)
    @uniforms! begin
    uniformsStore.colorsMaskRef:=Cfloat[color.r, color.g, color.b, 0.8]
    end

end#setMaskColor


```@doc
sets visibility of the texture
```
function setTextureVisibility(isvisible::Bool, uniformsStore ::TextureUniforms)
    @uniforms! begin
    uniformsStore.isVisibleRef:=isvisible
    end

end#setTextureVisibility


```@doc
sets typeOfMainSampler - value needed to choose proper sampler for main image
```
function setTypeOfMainSampler!(typeOfMainSamplerNumb::Int32, uniformsStore ::MainImageUniforms)
    @uniforms! begin
    uniformsStore.typeOfMainSamplerRef:=typeOfMainSamplerNumb
    end

end#setTextureVisibility




createStructsDictStr = """
Creates structures that enables controll over the uniforms in a shader  - most of this is done manually - TODO() apply metaprogramming to make this code shorter
program - reference to shader program of open gl from which we will take uniforms
1) dictionary of TextureUniforms where key is the type associated with the sampler - int/uint/float
2)main image uniforms 
"""
@doc createStructsDictStr
function createStructsDict(program::UInt32) ::Tuple{Vector{Pair{DataType, MaskTextureUniforms}}, MainImageUniforms}
# In order to make use of uniforms easier later in code we need to manually copy the uniform names in the tuple below
#What is also important that opengl removes all uniforms that are not used in shaders - hence it may lead to some hard to understand errors 
    @uniforms  (min_shown_white, max_shown_black, displayRange,
    iTexture0, uTexture0, fTexture0, typeOfMainSampler, isVisibleTexture0
    ,nuclearMask,isVisibleNuclearMask,
    uImask0,          uImask1,
          uImask2,          uImask3,          uImask4,          uImask5,          uImask6,          uImask7,          uImask8,
          imask0,          imask1,          imask2,          imask3,          imask4,          imask5,          imask6,          imask7,          imask8,          fmask0,
          fmask1,          fmask2,          fmask3,          fmask4,          fmask5,          fmask6,          fmask7,          fmask8,         uIcolorMask0,         uIcolorMask1,
         uIcolorMask2,         uIcolorMask3,         uIcolorMask4,         uIcolorMask5,         uIcolorMask6,         uIcolorMask7,         uIcolorMask8,         icolorMask0,
         icolorMask1,         icolorMask2,         icolorMask3,         icolorMask4,         icolorMask5,         icolorMask6,         icolorMask7,         icolorMask8,
         fcolorMask0,         fcolorMask1,         fcolorMask2,         fcolorMask3,         fcolorMask4,         fcolorMask5,         fcolorMask6,         fcolorMask7,
         fcolorMask8,        isVisibleTexture0,        isVisibleNuclearMask,                  uIisVisk0,       uIisVisk1,       uIisVisk2,       uIisVisk3,       uIisVisk4,
       uIisVisk5,       uIisVisk6,       uIisVisk7,       uIisVisk8,       iisVisk0,       iisVisk1,       iisVisk2,       iisVisk3,       iisVisk4,       iisVisk5,       iisVisk6,
       iisVisk7,       iisVisk8,       fisVisk0,       fisVisk1,       fisVisk2,       fisVisk3,       fisVisk4,       fisVisk5,       fisVisk6,       fisVisk7,       fisVisk8) = program
    

# structs holding references
    masksTuplList=   [ UInt => MaskTextureUniforms( samplerRef =uImask0, isVisibleRef=uIisVisk0, colorsMaskRef =uIcolorMask0) 
    ,UInt => MaskTextureUniforms( samplerRef =uImask1, isVisibleRef=uIisVisk1, colorsMaskRef =uIcolorMask1) 
    ,UInt => MaskTextureUniforms( samplerRef =uImask2, isVisibleRef=uIisVisk2, colorsMaskRef =uIcolorMask2) 
    ,UInt => MaskTextureUniforms( samplerRef =uImask3, isVisibleRef=uIisVisk3, colorsMaskRef =uIcolorMask3) 
    ,UInt => MaskTextureUniforms( samplerRef =uImask4, isVisibleRef=uIisVisk4, colorsMaskRef =uIcolorMask4) 
    ,UInt => MaskTextureUniforms( samplerRef =uImask5, isVisibleRef=uIisVisk5, colorsMaskRef =uIcolorMask5) 
    ,UInt => MaskTextureUniforms( samplerRef =uImask6, isVisibleRef=uIisVisk6, colorsMaskRef =uIcolorMask6) 
    ,UInt => MaskTextureUniforms( samplerRef =uImask7, isVisibleRef=uIisVisk7, colorsMaskRef =uIcolorMask7) 
    ,UInt => MaskTextureUniforms( samplerRef =uImask8, isVisibleRef=uIisVisk8, colorsMaskRef =uIcolorMask8) 
    #int
    ,Int => MaskTextureUniforms( samplerRef =imask0, isVisibleRef=iisVisk0, colorsMaskRef =icolorMask0) 
    ,Int => MaskTextureUniforms( samplerRef =imask1, isVisibleRef=iisVisk1, colorsMaskRef =icolorMask1) 
    ,Int => MaskTextureUniforms( samplerRef =imask2, isVisibleRef=iisVisk2, colorsMaskRef =icolorMask2) 
    ,Int => MaskTextureUniforms( samplerRef =imask3, isVisibleRef=iisVisk3, colorsMaskRef =icolorMask3) 
    ,Int => MaskTextureUniforms( samplerRef =imask4, isVisibleRef=iisVisk4, colorsMaskRef =icolorMask4) 
    ,Int => MaskTextureUniforms( samplerRef =imask5, isVisibleRef=iisVisk5, colorsMaskRef =icolorMask5) 
    ,Int => MaskTextureUniforms( samplerRef =imask6, isVisibleRef=iisVisk6, colorsMaskRef =icolorMask6) 
    ,Int => MaskTextureUniforms( samplerRef =imask7, isVisibleRef=iisVisk7, colorsMaskRef =icolorMask7) 
    ,Int => MaskTextureUniforms( samplerRef =imask8, isVisibleRef=iisVisk8, colorsMaskRef =icolorMask8) 
    #float
    ,Float => MaskTextureUniforms( samplerRef =umask0, isVisibleRef=fisVisk0, colorsMaskRef =fcolorMask0) 
    ,Float => MaskTextureUniforms( samplerRef =umask1, isVisibleRef=fisVisk1, colorsMaskRef =fcolorMask1) 
    ,Float => MaskTextureUniforms( samplerRef =umask2, isVisibleRef=fisVisk2, colorsMaskRef =fcolorMask2) 
    ,Float => MaskTextureUniforms( samplerRef =umask3, isVisibleRef=fisVisk3, colorsMaskRef =fcolorMask3) 
    ,Float => MaskTextureUniforms( samplerRef =umask4, isVisibleRef=fisVisk4, colorsMaskRef =fcolorMask4) 
    ,Float => MaskTextureUniforms( samplerRef =umask5, isVisibleRef=fisVisk5, colorsMaskRef =fcolorMask5) 
    ,Float => MaskTextureUniforms( samplerRef =umask6, isVisibleRef=fisVisk6, colorsMaskRef =fcolorMask6) 
    ,Float => MaskTextureUniforms( samplerRef =umask7, isVisibleRef=fisVisk7, colorsMaskRef =fcolorMask7) 
    ,Float => MaskTextureUniforms( samplerRef =imask8, isVisibleRef=fisVisk8, colorsMaskRef =fcolorMask8) ]

    mainImageUnifs = MainImageUniforms( samplerRef =Texture0
                                    ,isVisibleRef = isVisibleTexture0
                                    ,min_shown_white = min_shown_white
                                    ,max_shown_black = max_shown_black
                                    ,displayRange = displayRange
            )
   return (masksTuplList,mainImageUnifs)  
                     

end#createStructsDict



end #module




