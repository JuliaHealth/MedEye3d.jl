
```@doc
functions responsible for displaying medical image Data
```
using DrWatson
@quickactivate "Probabilistic medical segmentation"
using GLMakie
#using Makie
#using AbstractPlotting
using Plots 
using GeometryBasics
using GeometricalPredicates
using Distances


imageDim = size(arrr)
maskArr = Observable(BitArray(undef, imageDim))
singleCtScanDisplay(arrr, maskArr)


```@doc
simple display of single image - only in transverse plane we are adding also a mask  that 
arrr - main 3 dimensional data representing medical image for example in case of CT each voxel represents value of X ray attenuation
```
function singleCtScanDisplay(arrr ::Array{Number, 3}, maskArr::Observable{BitArray{3}} ) 
  
imageDim = size(arrr) # dimenstion of the primary image for example CT scan
slicesNumb =imageDim[3] # number of slices 

#defining layout variables
scene, layout = Makie.layoutscene(resolution = (600, 400))
ax1 = layout[1, 1] = Makie.Axis(scene, backgroundcolor = :transparent)
ax2 = layout[1, 1] = Makie.Axis(scene, backgroundcolor = :transparent)

#control widgets
sl_x =layout[2, 1]= Makie.Slider(scene, range = 1:1: slicesNumb , startvalue = slicesNumb/2 )
sliderXVal = sl_x.value


#color maps
cmwhite = cgrad(range(RGBA(10,10,10,0.01), stop=RGBA(0,0,255,0.4), length=10000));

####heatmaps

#main heatmap that holds for example Ct scan
currentSliceMain = Makie.@lift(arrr[:,:, convert(Int32,$sliderXVal)])
hm = Makie.heatmap!(ax1, currentSliceMain ,colormap = :grays) 

#helper heatmap designed to respond to both changes in slider and changes in the bit matrix
currentSliceMask = Makie.@lift($maskArr[:,:, convert(Int32,$sliderXVal)])
hmB = Makie.heatmap!(ax1, currentSliceMask ,colormap = cmwhite) 

#adding ability to be able to add information to mask  where we clicked so in casse of mit matrix we will set the point where we clicked to 1 
indicatorC(ax1,imageDim,scene,maskArr,sliderXVal)

#displaying
colorB = layout[1,2]= Colorbar(scene, hm)
Makie.translate!(hmB, Vec3f0(0,0,5))  
scene

end

```@doc
inspired by   https://github.com/JuliaPlots/Makie.jl/issues/810
Generaly thanks to this function  the viewer is able to respond to clicking on the slices and records it in the supplied 3 dimensional bitarray
  ax - Axis which store our heatmap slices which we want to observe wheather user clicked on them and where
  dims - dimensions of  main image for example CT
  sc - Scene where our axis is
  maskArr - the 3 dimensional bit array  that has exactly the same dimensions as main Array storing image 
  sliceNumb - represents on what slide we are on currently on - ussually it just give information from slider 
```
function indicatorC(ax::Axis,dims::Tuple{Int64, Int64, Int64},sc::Scene,maskArr::Observable{BitArray{3}},sliceNumb::Observable{Any})
  register_interaction!(ax, :indicator) do event::Makie.MouseEvent, axis
  if event.type === MouseEventTypes.leftclick
    #position from top left corner 
    xMouse= to_world(sc,event.data)[1]
    yMouse= to_world(sc,event.data)[2]
    #data about height and width in layout
    compBoxWidth = 510 
    compBoxHeight = 510 
    #image dimensions - number of pixels  from medical image for example ct scan
    pixelsNumbInX =dims[1]
    pixelsNumbInY =dims[2]
    #calculating over which image pixel we are
    calculatedXpixel =  (xMouse/compBoxWidth)*pixelsNumbInX 
    calculatedYpixel =   (yMouse/compBoxHeight)*pixelsNumbInY 
    sliceNumbConv = sliceNumb[] 
    #appropriately modyfing wanted pixels in mask array
    markMaskArrayPatch( maskArr ,Point3D(calculatedXpixel, calculatedYpixel, sliceNumbConv  ))



    #print("xMouse: $(xMouse)  yMouse: $(yMouse)   compBoxWidth: $(compBoxWidth)  compBoxHeight: $(compBoxHeight)   calculatedXpixel: $(calculatedXpixel)  calculatedYpixel: $(calculatedYpixel)      pixelsNumbInX  $(pixelsNumbInX)         ") 
  return true
  end
end
end



```@doc
  maskArr - the 3 dimensional bit array  that has exactly the same dimensions as main Array storing image 
  point - point around which we want to modify the bitArray from 0 to 1

```
function markMaskArrayPatch(maskArr::Observable{BitArray{3}}, point::Point3D)
  calculatedXpixel= convert(Int32, round(GeometricalPredicates.getx(point) ))
  calculatedYpixel= convert(Int32,round(GeometricalPredicates.gety(point) ))
  sliceNumbConv= convert(Int32,round(GeometricalPredicates.getz(point) ))

  maskArrB = maskArr[]
    
  maskArrB[calculatedXpixel, calculatedYpixel, sliceNumbConv]=1
  maskArrB[calculatedXpixel+1, calculatedYpixel+1, sliceNumbConv]=1
  maskArrB[calculatedXpixel+1, calculatedYpixel-1, sliceNumbConv]=1
  maskArrB[calculatedXpixel-1, calculatedYpixel+1, sliceNumbConv]=1
  maskArrB[calculatedXpixel-1, calculatedYpixel-1, sliceNumbConv]=1


  maskArr[] = maskArrB



end


```@doc
  we are looking in given 3 dimensional array closest points in distnace not bigger than dist measure in euclidean distance

  1)create a 3 dimensional matrix  that will have edge equal to distance
  2)take all cartesian coordinates from it
  3)add to those cartesian

  arr - the 3 dimensional bit array  that has exactly the same dimensions as main Array storing image 
  point - point around which we want to modify the bitArray from 0 to 1
  dist - maximal euclidean distance to which we want  to apply some operations
  returns series of cartesian coordinates of points closest to given point using euclidean distance measure 
```
function getIndexesclosestToPoint(arr::BitArray{3}, point::Point3D) 
  calculatedXpixel= convert(Int32, round(GeometricalPredicates.getx(point) ))
  calculatedYpixel= convert(Int32,round(GeometricalPredicates.gety(point) ))
  sliceNumbConv= convert(Int32,round(GeometricalPredicates.getz(point) ))

#
# - 


end

