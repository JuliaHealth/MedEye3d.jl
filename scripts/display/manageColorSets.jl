module ManageColorSets
using Base: Tuple, indent_width
using ColorSchemeTools

```@doc
creating grey scheme colors for proper display of medical image mainly CT scan
min_shown_white - max_shown_black range over which the gradint of greys will be shown
truemax - truemin the range of values in the image for which we are creating the scale
```
#taken from https://stackoverflow.com/questions/67727977/how-to-create-julia-color-scheme-for-displaying-ct-scan-makie-jl/67756158#67756158

function createMedicalImageColorScheme(min_shown_white,max_shown_black,truemax,truemin,maxWithLabels )
  
    

 
  
#     gist_rainbow = (
#         (0.000, (1.00, 0.00, 0.16)),
#         (0.030, (1.00, 0.00, 0.00)),
#         (0.215, (1.00, 1.00, 0.00)),
#         (0.400, (0.00, 1.00, 0.00)),
#         (0.586, (0.00, 1.00, 1.00)),
#         (0.770, (0.00, 0.00, 1.00)),
#         (0.954, (1.00, 0.00, 1.00)),
#         (1.000, (1.00, 0.00, 0.75))
#  )
 
 return make_colorscheme(getGreyLevels(min_shown_white, max_shown_black,truemax,truemin,maxWithLabels))
end

 ```@doc
given upper and lower boundary it divides it value "buckets" and sets diffrent "greyness" values depending on the value uniformly in the range
min_shown_white - max_shown_black range over which the gradint of greys will be shown
maxWithLabels - max including additional values needed for labels
return list of rgb values for given range
 ```   
function getGreyLevels(
    min_shown_white, 
    max_shown_black,
    truemax,
    truemin,
    maxWithLabels)

    rangeSize =  maxWithLabels -truemin
    min_white =  (abs(truemin)+min_shown_white)/rangeSize
    maxBlack =  (abs(truemin)+max_shown_black)/rangeSize

    greys=(
        (0.000, (0.00, 0.00, 0.00)),
        (maxBlack, (0.00, 0.00, 0.00)),
        (min_white, (1.00, 1.00, 1.00)),
        (0.99, (1.00, 1.00, 1.00)),
        (0.99, (1.00, 0.1, 0.2)),
        (1.0, (1.00, 0.1, 0.2))
        )

    return greys
end


min_shown_white= 200
max_shown_black=-200
truemax = 1000
truemin = -1000
createMedicalImageColorScheme(min_shown_white,max_shown_black,1000,-1000,1001)


end #module


#createMedicalImageColorScheme(min_shown_white,max_shown_black,1000,-1000)

# greys = ((0:rangeSize)./rangeSize.*255) |>             # correct values and range but floats
# (floats) -> map(x-> round(Int, x)  ,floats) |> # rounding to integers
# (ints) -> map(x->(x,x,x) ,ints ) |># changing into RGB tuples
# # (res) -> reverse(res)

# out = map(x->(0.1, (0,0,0)),zeros(rangeSize+1))
# for (index, value) in enumerate(greys)
# out[index] = ( ((1/rangeSize) * index ),value)
# end

# return  tuple(out...)
# # [      fill(ColorSchemeTools.colorant"black", max_shown_black - truemin + 1);
# #               collect(ColorSchemeTools.make_colorscheme(identity, identity, identity,
# #                    length = min_shown_white - max_shown_black - 1));
# #                 collect(ColorSchemeTools.make_colorscheme(identity, identity, identity,
# #                    length = min_shown_white - max_shown_black - 1));
# #                fill(ColorSchemeTools.colorant"white", truemax - min_shown_white + 1);
# #               # fill(ColorSchemeTools.colorant"green", 10000)
# #                               ]
