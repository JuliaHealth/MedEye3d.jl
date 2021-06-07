
```@doc
functions responsible for displaying medical image Data
```
using DrWatson
@quickactivate "Probabilistic medical segmentation"
using GLMakie
using Makie
#using GeometryBasics
using GeometricalPredicates
using ColorTypes
using Distributed
using GLMakie

## getting id of workers 
dirToWorkerNumbs = DrWatson.scriptsdir("mainPipeline","processesDefinitions","workerNumbers.jl")
dirToImageHelper = DrWatson.scriptsdir("display","imageViewerHelper.jl")

include(dirToWorkerNumbs)
include(dirToImageHelper)
include(DrWatson.scriptsdir("structs","forDisplayStructs.jl"))

exmpleH = @spawnat persistenceWorker Main.h5manag.getExample()
arrr= fetch(exmpleH)
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
scene, layout = GLMakie.layoutscene(resolution = (600, 400))
ax1 = layout[1, 1] = GLMakie.Axis(scene, backgroundcolor = :transparent)
ax2 = layout[1, 1] = GLMakie.Axis(scene, backgroundcolor = :transparent)

#control widgets
sl_x =layout[2, 1]= GLMakie.Slider(scene, range = 1:1: slicesNumb , startvalue = slicesNumb/2 )
sliderXVal = sl_x.value


#color maps
cmwhite = cgrad(range(RGBA(10,10,10,0.01), stop=RGBA(0,0,255,0.4), length=10000));

####heatmaps

#main heatmap that holds for example Ct scan
currentSliceMain = GLMakie.@lift(arrr[:,:, convert(Int32,$sliderXVal)])
hm = GLMakie.heatmap!(ax1, currentSliceMain ,colormap = :grays) 

#helper heatmap designed to respond to both changes in slider and changes in the bit matrix
currentSliceMask = GLMakie.@lift($maskArr[:,:, convert(Int32,$sliderXVal)])
hmB = GLMakie.heatmap!(ax1, currentSliceMask ,colormap = cmwhite) 

#adding ability to be able to add information to mask  where we clicked so in casse of mit matrix we will set the point where we clicked to 1 
indicatorC(ax1,imageDim,scene,maskArr,sliderXVal)

#displaying
colorB = layout[1,2]= Colorbar(scene, hm)
GLMakie.translate!(hmB, Vec3f0(0,0,5))  

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
  register_interaction!(ax, :indicator) do event::GLMakie.MouseEvent, axis
  if event.type === MouseEventTypes.leftclick
    println("clicked")
    #@async begin
      #appropriately modyfing wanted pixels in mask array
   @async calculateMouseAndSetmaskWrap(maskArr, event,sc,dims,sliceNumb)            
    #  
    #  
    #  println("fetched" + fetch(maskA))

    #  finalize(maskA)
    #end
    return true
    #print("xMouse: $(xMouse)  yMouse: $(yMouse)   compBoxWidth: $(compBoxWidth)  compBoxHeight: $(compBoxHeight)   calculatedXpixel: $(calculatedXpixel)  calculatedYpixel: $(calculatedYpixel)      pixelsNumbInX  $(pixelsNumbInX)         ") 
  end
  
end
end
```@doc
wrapper for calculateMouseAndSetmask  - from imageViewerHelper module
  given mouse event modifies mask accordingly
  maskArr - the 3 dimensional bit array  that has exactly the same dimensions as main Array storing image 
  event - mouse event passed from Makie
  sc - scene we are using in Makie
  ```
function calculateMouseAndSetmaskWrap(maskArr, event,sc,dims,sliceNumb) 
  maskArr[] = calculateMouseAndSetmask(maskArr, event,sc,dims,sliceNumb)
end




