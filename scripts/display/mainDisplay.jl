
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
```@doc
simple display of single image - only in transverse plane
```
function singleCtScanDisplay(arr ::Array{Number, 3}) 
  
  slicesNumb =size(arr)[3]
  fig = Figure()
  sl_x = Slider(fig[2, 1], range = 1:1: slicesNumb , startvalue = convert(Int32,slicesNumb/2 ) )
  ax = Axis(fig[1, 1])
  hm = heatmap!(ax, lift(idx-> arr[:,:, convert(Int32,idx)], sl_x.value) ,colormap = :grays)
  Colorbar(fig[1, 2], hm)


end

#const arrr= getExample()

#display mouseposition inspired by   https://github.com/JuliaPlots/Makie.jl/issues/810
function indicatorC(ax::Axis,dims::Tuple{Int64, Int64, Int64},sc::Scene,maskArr::Observable{BitArray{3}},sliceNumb::Observable{Any})
  register_interaction!(ax, :indicator) do event::Makie.MouseEvent, axis
  if event.type === MouseEventTypes.leftclick
    position=(0,0)
    size=1
    #position from top left corner 
    compBoxOrigin = to_world(sc,ax1.layoutobservables.computedbbox[].origin)
    xMouse= to_world(sc,event.data)[1]
    yMouse= to_world(sc,event.data)[2]
    #data about height and width in layout
      #compBoxDims = ax1.layoutobservables.computedbbox[].widths
    compBoxWidth = 510 #Makie.to_world(sc,compBoxDims)[1] -compBoxOrigin[1]
    compBoxHeight = 510 #Makie.to_world(sc,compBoxDims)[2] -compBoxOrigin[1]
    #image dimensions - number of pixels  from medical image for example ct scan
    pixelsNumbInX =dims[1]
    pixelsNumbInY =dims[2]
    #calculating over which image pixel we are
    calculatedXpixel = round( (xMouse/compBoxWidth)*pixelsNumbInX )
    calculatedYpixel = round(  (yMouse/compBoxHeight)*pixelsNumbInY )
    maskArrB = maskArr[]
    maskArrB[ convert(Int32,calculatedXpixel), convert(Int32,calculatedYpixel), convert(Int32,sliceNumb[]) ]=1
    maskArr[] = maskArrB
    print("xMouse: $(xMouse)  yMouse: $(yMouse)   compBoxWidth: $(compBoxWidth)  compBoxHeight: $(compBoxHeight)   calculatedXpixel: $(calculatedXpixel)  calculatedYpixel: $(calculatedYpixel)      pixelsNumbInX  $(pixelsNumbInX)         ") 
    return false
  end
  end
end

# dimenstion of the primary image for example CT scan
imageDim = size(arrr)
maskArr = Observable(BitArray(undef, imageDim))
# number of slices 
slicesNumb =imageDim[3]

#defining layout variables
scene, layout = Makie.layoutscene(resolution = (600, 400))
ax1 = layout[1, 1] = Makie.Axis(scene, backgroundcolor = :transparent)
ax2 = layout[1, 1] = Makie.Axis(scene, backgroundcolor = :transparent)

#control widgets
sl_x =layout[2, 1]= Makie.Slider(scene, range = 1:1: slicesNumb , startvalue = slicesNumb/2 )
sliderXVal = sl_x.value

#color maps
cmwhite = cgrad(range(RGBA(10,10,10,0.01), stop=RGBA(0,0,255,0.4), length=10000));
#cmwhite = cgrad(ColorScheme([RGBA(10,10, 10,0.1),RGBA(220,0, 0,0.8)]))

    #cmwhite = cgrad(range(HSLA(0,0,1,0), stop=HSLA(0,0,1,1), length=100));

#heatmaps
#main heatmap that holds for example Ct scan
currentSliceMain = Makie.@lift(arrr[:,:, convert(Int32,$sliderXVal)])
hm = Makie.heatmap!(ax1, currentSliceMain ,colormap = :grays) 

#helper heatmap designed to respond to both changes in slider and changes in the bit matrix
currentSliceMask = Makie.@lift($maskArr[:,:, convert(Int32,$sliderXVal)])

hmB = Makie.heatmap!(ax1, currentSliceMask ,colormap = cmwhite) 


# #formatting
# hidedecorations!(ax1, grid = false)
# hidedecorations!(ax2, grid = false)

#display mouseposition  https://github.com/JuliaPlots/Makie.jl/issues/810

indicatorC(ax1,imageDim,scene,maskArr,sliderXVal)
#indicatorC(layout,imageDim)



#displaying
Makie.translate!(hmB, Vec3f0(0,0,5))  
scene





# function indicatorMy(ob)
#   print( "aaaa")
# end

# register_interaction!(indicatorMy,ax1, :indicatorMyInter) 



