
```@doc
functions responsible for displaying medical image Data
```
module mainDisplay

using DrWatson
@quickactivate "Probabilistic medical segmentation"
using GLMakie

```@doc
simple display of single image - only in transverse plane
```
function singleCtScanDisplay(arr ::Array{Number, 3}) 
  

  fig = Figure()
  sl_x = Slider(fig[2, 1], range = 0:1:100, startvalue = 40)
  ax = Axis(fig[1, 1])
  hm = heatmap!(ax, lift(idx-> arr[:,:, floor(idx)], sl_x.value) ,colormap = :grays)
  Colorbar(fig[1, 2], hm)
  fig
end


end
