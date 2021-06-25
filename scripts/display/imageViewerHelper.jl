```@doc
functions responsible for helping in image viewer - those functions are  meant to be invoked on separate process
- in parallel
```
using DrWatson
@quickactivate "Probabilistic medical segmentation"




module imageViewerHelper
using Core: print
using Base: Number
using Documenter
export calculateMouseAndSetmask
export createMedicalImageColorScheme
# using AbstractPlotting
```@doc
  given mouse event modifies mask accordingly
  maskArr - the 3 dimensional bit array  that has exactly the same dimensions as main Array storing image 
  dims - dimensions of a mask
  sliceNumb - number of sclice that is currently displaying
  xMouse - x coordinate where we clicked
  yMouse - y coordinate where we clicked
  patchSize - size in pixels of a patch
  return modified mask array where we added data  where we clicked
  ```
function calculateMouseAndSetmask(maskArr,dims::Tuple{Int64, Int64, Int64},sliceNumb
  ,xMouse::Number,yMouse::Number,patchSize::Int = 2,compBoxWidth::Int = 510 ,compBoxHeight::Int = 510  )  
  #position from top left corner 

  #image dimensions - number of pixels  from medical image for example ct scan
  pixelsNumbInX =dims[1]
  pixelsNumbInY =dims[2]
  #calculating over which image pixel we are
  calculatedXpixel =convert(Int32, round( (xMouse/compBoxWidth)*pixelsNumbInX) )
  calculatedYpixel =  convert(Int32,round( (yMouse/compBoxHeight)*pixelsNumbInY ))

  sliceNumbConv =convert(Int32,round( sliceNumb[] ))
  #appropriately modyfing wanted pixels in mask array

  pixelLoc = CartesianIndex(calculatedXpixel, calculatedYpixel, sliceNumbConv  )
  # calculating indices that surrounds the primary ones
  static = maskArr[]
  imageDim = size(static)
  return   markMaskArrayPatchTo!(static, cartesianCoordAroundPoint(pixelLoc,patchSize*2),4, imageDim)|>
  (_x)->markMaskArrayPatchTo!(_x, cartesianCoordAroundPoint(pixelLoc,patchSize),5, imageDim)|>
  (_x)->  markMaskArrayPatchTo!(_x, [pixelLoc],6, imageDim)

end


```@doc
  It modifies given array 
  maskArr - the 3 dimensional bit array  that has exactly the same dimensions as main Array storing image 
  points - cartesian coordinates of points where we want to apply modifications
  valueToSet - the new value by whic we will modify 
  return modified array 
```
function markMaskArrayPatchTo!(maskArr, points::Array{CartesianIndex{3}},valueToSet::Number,imageDim::Tuple{Int64, Int64, Int64} )
  #first we filter out all points tha are not in range
  points |>
  (x_)->  filter((ind) ->( ind[1]<= imageDim[1])  && ( ind[2]<= imageDim[2])  && ( ind[3]<= imageDim[3]) && ind[1]>0 && ind[2] >0 && ind[3]>0 , x_) |>
  (x_) -> maskArr[x_].=valueToSet
  return maskArr
end

```@doc
  point - cartesian coordinates of point around which we want the cartesian coordeinates
  return set of cartetian coordinates of given distance -patchSize from a point
```
function cartesianCoordAroundPoint(pointCart::CartesianIndex{3}, patchSize ::Int)::Array{CartesianIndex{3}}
  ones = CartesianIndex(patchSize,patchSize,patchSize) # cartesian 3 dimensional index used for calculations to get range of the cartesian indicis to analyze
  out = Array{CartesianIndex{3}}(UndefInitializer(), 6+2*patchSize^4)
  index =0
  for J in (pointCart-ones):(pointCart+ones)
    diff = J - pointCart # diffrence between dimensions relative to point of origin
      if cartesianTolinear(diff) <= patchSize
        index+=1
        out[index] = J
      end
      end
return out[1:index]
end






```@doc
works only for 3d cartesian coordinates
  cart - cartesian coordinates of point where we will add the dimensions ...
```
function cartesianTolinear(pointCart::CartesianIndex{3}) :: Int16
   abs(pointCart[1])+ abs(pointCart[2])+abs(pointCart[3])
end




end #module