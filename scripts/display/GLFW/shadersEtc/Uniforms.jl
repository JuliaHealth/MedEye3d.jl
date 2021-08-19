
"""
managing  uniform values - global values in shaders
"""
module Uniforms
using Glutils, Main.ForDisplayStructs, Dictionaries, Parameters, ColorTypes

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
    uniformsStore.isVisibleRef:= isvisible ? 1 : 0
    end

end#setTextureVisibility





end #module




