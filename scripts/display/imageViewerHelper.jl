```@doc
functions responsible for helping in image viewer - those functions are  meant to be invoked on separate process
- in parallel
```
using DrWatson
@quickactivate "Probabilistic medical segmentation"

module imageViewerHelper
using Documenter
using ColorTypes
using Colors, ColorSchemeTools
using Makie
# using AbstractPlotting
```@doc
  given mouse event modifies mask accordingly
  maskArr - the 3 dimensional bit array  that has exactly the same dimensions as main Array storing image 
  event - mouse event passed from Makie
  sc - scene we are using in Makie
  ```
function calculateMouseAndSetmask(maskArr, event,sc,dims,sliceNumb) 
  #position from top left corner 
  xMouse= Makie.to_world(sc,event.data)[1]
  yMouse= Makie.to_world(sc,event.data)[2]
  #data about height and width in layout
  compBoxWidth = 510 
  compBoxHeight = 510 
  #image dimensions - number of pixels  from medical image for example ct scan
  pixelsNumbInX =dims[1]
  pixelsNumbInY =dims[2]
  #calculating over which image pixel we are
  calculatedXpixel =convert(Int32, round( (xMouse/compBoxWidth)*pixelsNumbInX) )
  calculatedYpixel =  convert(Int32,round( (yMouse/compBoxHeight)*pixelsNumbInY ))
  sliceNumbConv =convert(Int32,round( sliceNumb[] ))
  #appropriately modyfing wanted pixels in mask array
  return markMaskArrayPatch( maskArr ,CartesianIndex(calculatedXpixel, calculatedYpixel, sliceNumbConv  ),2)
end



```@doc
  maskArr - the 3 dimensional bit array  that has exactly the same dimensions as main Array storing image 
  point - cartesian coordinates of point around which we want to modify the 3 dimensional array from 0 to 1

```
function markMaskArrayPatch(maskArr, pointCart::CartesianIndex{3}, patchSize ::Int64)

  ones = CartesianIndex(patchSize,patchSize,patchSize) # cartesian 3 dimensional index used for calculations to get range of the cartesian indicis to analyze
  maskArrB = maskArr[]
  for J in (pointCart-ones):(pointCart+ones)
    diff = J - pointCart # diffrence between dimensions relative to point of origin
      if cartesianTolinear(diff) <= patchSize
        maskArrB[J]=1
      end
      end
return maskArrB
end
```@doc
works only for 3d cartesian coordinates
  cart - cartesian coordinates of point where we will add the dimensions ...
```
function cartesianTolinear(pointCart::CartesianIndex{3}) :: Int16
   abs(pointCart[1])+ abs(pointCart[2])+abs(pointCart[3])
end

```@doc
creating grey scheme colors for proper display of medical image mainly CT scan
min_shown_white - max_shown_black range over which the gradint of greys will be shown
truemax - truemin the range of values in the image for which we are creating the scale
```
#taken from https://stackoverflow.com/questions/67727977/how-to-create-julia-color-scheme-for-displaying-ct-scan-makie-jl/67756158#67756158

function createMedicalImageColorScheme(min_shown_white::Int,max_shown_black::Int,truemax::Int,truemin::Int ) ::Vector{Any} 

return  [fill(colorant"black", max_shown_black - truemin + 1);
               collect(make_colorscheme(identity, identity, identity,
                   length = min_shown_white - max_shown_black - 1));
               fill(colorant"white", truemax - min_shown_white + 1)]

end





end #module