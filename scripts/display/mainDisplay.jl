
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

include(DrWatson.scriptsdir("structs","forDisplayStructs.jl"))

exmpleH = @spawnat 2 Main.h5manag.getExample()
arrr = fetch(exmpleH)
imageDim = size(arrr)
maskArr = Observable(BitArray(undef, imageDim))
singleCtScanDisplay(arrr, maskArr)


```@doc
simple display of single image - only in transverse plane we are adding also a mask  that 
arrr - main 3 dimensional data representing medical image for example in case of CT each voxel represents value of X ray attenuation
```
function singleCtScanDisplay(arrr ::Array{Number, 3}, maskArr ) 
  
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
sl_x =layout[2, 1]
Makie.translate!(hmB, Vec3f0(0,0,5))  
scene

end

```@doc
inspired by   https://github.com/JuliaPlots/Makie.jl/issues/810
Generaly thanks to this function  the viewer is able to respond to clicking on the slices and records it in the supplied 3 dimensional AbstractArray
  ax - Axis which store our heatmap slices which we want to observe wheather user clicked on them and where
  dims - dimensions of  main image for example CT
  sc - Scene where our axis is
  maskArr - the 3 dimensional bit array  that has exactly the same dimensions as main Array storing image 
  sliceNumb - represents on what slide we are on currently on - ussually it just give information from slider 
```
function indicatorC(ax::Axis,dims::Tuple{Int64, Int64, Int64},sc::Scene,maskArr,sliceNumb::Observable{Any})
  register_interaction!(ax, :indicator) do event::Makie.MouseEvent, axis
  if event.type === MouseEventTypes.leftclick
    @async begin
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
      calculatedXpixel =convert(Int32, round( (xMouse/compBoxWidth)*pixelsNumbInX) )
      calculatedYpixel =  convert(Int32,round( (yMouse/compBoxHeight)*pixelsNumbInY ))
      sliceNumbConv =convert(Int32,round( sliceNumb[] ))
      #appropriately modyfing wanted pixels in mask array
      markMaskArrayPatch( maskArr ,CartesianIndex(calculatedXpixel, calculatedYpixel, sliceNumbConv  ),2)
    end
    return true
    #print("xMouse: $(xMouse)  yMouse: $(yMouse)   compBoxWidth: $(compBoxWidth)  compBoxHeight: $(compBoxHeight)   calculatedXpixel: $(calculatedXpixel)  calculatedYpixel: $(calculatedYpixel)      pixelsNumbInX  $(pixelsNumbInX)         ") 
  end
  
end
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
  maskArr[] = maskArrB
    end
end


```@doc
works only for 3d cartesian coordinates
  cart - cartesian coordinates of point where we will add the dimensions ...
```
function cartesianTolinear(pointCart::CartesianIndex{3}) :: Int16
   abs(pointCart[1])+ abs(pointCart[2])+abs(pointCart[3])
end


