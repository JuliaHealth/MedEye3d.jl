
```@doc
functions responsible for displaying medical image Data
```
using DrWatson
@quickactivate "Probabilistic medical segmentation"


module MyImgeViewer
export singleCtScanDisplay

using GLMakie
#using Makie
#using GeometryBasics
using GeometricalPredicates
using ColorTypes
using Distributed
using GLMakie
using Main.imageViewerHelper
using Main.workerNumbers
## getting id of workers 
using Main.ForDisplayStructs
using Main.ManageColorSets

```@doc
simple display of single image - only in transverse plane we are adding also a mask  that 
arrr - main 3 dimensional data representing medical image for example in case of CT each voxel represents value of X ray attenuation
minimumm, maximumm - approximately minimum and maximum values we can have in our image
```
function singleCtScanDisplay(arrr ::Array{Number, 3}, masks::Array , minimumm::Int, maximumm::Int, maxWithLabels::Int) 
#we modify 2 pixels just in order to make the color range constant so slices will be displayed in the same windows in all projections
arrr[1,1,:].= minimumm 
arrr[2,1,:].= maxWithLabels 

arrr[1,:,1].= minimumm 
arrr[2,:,1].= maxWithLabels 

arrr[:,:,1].= minimumm 
arrr[:,:,2].= maxWithLabels 


imageDim = size(arrr) # dimenstion of the primary image for example CT scan
slicesNumb =imageDim[3] # number of slices 
observableArr = Observable(arrr)
#defining layout variables
scene, layout = GLMakie.layoutscene(resolution = (600, 400))
ax1 = layout[1, 1] = GLMakie.Axis(scene, backgroundcolor = :transparent)




#control widgets
sl_x =layout[2, 1]= GLMakie.Slider(scene, range = 1:1: slicesNumb , startvalue = slicesNumb/2 )
sliderXVal = sl_x.value

####heatmaps

#main heatmap that holds for example Ct scan
#currentSliceMain = GLMakie.@lift(arrr[convert(Int32,$sliderXVal),:,:])

currentSliceMain = GLMakie.@lift($observableArr[convert(Int32,$sliderXVal),:,:])


hm = GLMakie.heatmap!(ax1, currentSliceMain ,colormap = Main.ManageColorSets.createMedicalImageColorScheme(300,-500,maximumm, minimumm ,maxWithLabels)) 
indicatorC(ax1,imageDim,scene,observableArr,sliderXVal)

#helper heatmaps designed to respond to both changes in slider and changes in the bit matrix
# for mask in masks
#   createMaskMap!(mask,sliderXVal,ax1,scene,imageDim)
# end #for 


#displaying
# layout[1,2]= Colorbar(scene, hm)
scene

end


```@doc
creates heatmap that represents mask - some additional data displayed over the image, masks are generally modifiable  by user register_interaction
mask - mask object with all required data to create heatmap
sliderXVal - value associated with slider that controls which slice is displayed
ax1 - axis which is a basis for the Makie Layout
```
function createMaskMap!(mask,sliderXVal,ax1,scene,imageDim)
  mask.maskArrayObs[][1,1,:].= 1 # just for proper displaying
  cmwhite = cgrad(range(RGBA(10,10,10,0.01), stop=mask.colorRGBA, length=10));
  observableArr = mask.maskArrayObs
  currentSliceMask = GLMakie.@lift($observableArr[convert(Int32,$sliderXVal),:,:])
  hmB= GLMakie.heatmap!(ax1, currentSliceMask ,colormap = cmwhite) 
  #GLMakie.translate!(hmB, Vec3f0(0,0,10))  
  #adding ability to be able to add information to mask  where we clicked so in casse of mit matrix we will set the point where we clicked to 1 
  indicatorC(ax1,imageDim,scene,observableArr,sliderXVal)
  #@spawnat persistenceWorker Main.h5manag.saveMaskData!(Int16, mask)

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
function indicatorC(ax,dims::Tuple{Int64, Int64, Int64},sc,maskArr,sliceNumb::Observable{Any})
  GLMakie.register_interaction!(ax, :indicator) do event::GLMakie.MouseEvent, axis
  if event.type === MouseEventTypes.leftclick
   #@async 
   calculateMouseAndSetmaskWrap(maskArr, event,sc,dims,sliceNumb)            

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
  xMouse= GLMakie.to_world(sc,event.data)[1]
  yMouse= GLMakie.to_world(sc,event.data)[2]
  interm = calculateMouseAndSetmask(maskArr,dims,sliceNumb,xMouse,yMouse )
  print("clicked")
  maskArr[] = interm
end


end #module


