module ManageColorSets
using ColorSchemeTools

```@doc
creating grey scheme colors for proper display of medical image mainly CT scan
min_shown_white - max_shown_black range over which the gradint of greys will be shown
truemax - truemin the range of values in the image for which we are creating the scale
```
#taken from https://stackoverflow.com/questions/67727977/how-to-create-julia-color-scheme-for-displaying-ct-scan-makie-jl/67756158#67756158

function createMedicalImageColorScheme(min_shown_white,max_shown_black,truemax,truemin ) ::Vector{Any} 
return  [fill(ColorSchemeTools.colorant"black", max_shown_black - truemin + 1);
               collect(ColorSchemeTools.make_colorscheme(identity, identity, identity,
                   length = min_shown_white - max_shown_black - 1));
               fill(ColorSchemeTools.colorant"white", truemax - min_shown_white + 1)]

end




end #module