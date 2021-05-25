
using DrWatson
@quickactivate "Probabilistic medical segmentation"
```@doc


```
using GLMakie
using Makie, GeometryTypes, Observables
using AbstractPlotting #: slider!, playbutton, vbox, hbox!
using MAT
#]add Makie#master AbstractPlotting#master

module mainDisplay 



x = range(0, 10, length=100)
y = sin.(x)
lines(x, y)





using HDF5

pathToHd5 = "C:\\Users\\jakub\\OneDrive\\Documents\\GitHub\\probabilisticSegmentation\\Probabilistic medical segmentation\\data\\hdf5Main\\mainHdDataBaseLiver07.hdf5"
g = h5open(pathToHd5, "r")


arr=[]
for obj in g["testScans"]
  arr= read( obj)
  break
end
arr





ex_volume = arr

style = Theme(raw = true, camera = campixel!)

# 3D Brain Image
axis    = range(0, stop = 1, length = size(ex_volume, 1))
scene3d = Makie.contour(axis, axis, axis, ex_volume, alpha = 0.1, levels = 4)
cntr    = last(scene3d);
volume  = cntr[4]

scene3d

# # XY-XZ-YZ Planes' Sliders
# sliders = ntuple(3) do i
  
#     s_scene = slider(style, 1:size(volume[], i), start = size(volume[], i) ÷ 2)
#     s = last(s_scene)
#     idx = s[:value]; 
    
#     plane = planes[i]
    
#     indices = map(1:3) do j; planes[j] == plane ? 1 : (:); end
    
#     hmap = last(heatmap!(
#                hscene, axis2, axis2, volume[][indices...],
#                fillrange = true, interpolate = true
#            ))
    
#     lift(idx, volume) do _idx, vol
#         idx = (i in (1, 2)) ? (size(vol, i) - _idx) + 1 : _idx
#         transform!(hmap, (plane, axis2[_idx]))
#         indices = map(1:3) do j; planes[j] == plane ? idx : (:); end
#         if checkbounds(Bool, vol, indices...)
#             hmap[3][] = view(vol, indices...)
#         end
#     end
#     s_scene
# end

# #XY-XZ-YZ  Planes Toggle Button
# b1 = button(style, "Plane:     "; dimensions = (150, 150))
# on(b1[end][:clicks]) do clicks
#     if b2.plots[1][1][] == "2D"
#         cam3d!(hscene)
#         cam = cameracontrols(hscene) 
#         cam.projectiontype[] = AbstractPlotting.Orthographic
#         if clicks % 3 == 1
#             cam.upvector[] = Vec3f0(0, 0, 1)
#             cam.eyeposition[] = Vec3f0(0, 300, 0)
#             cam.lookat[] = Vec3f0(0, 0, 0)
#             b1.plots[1][1][] = "Plane: XZ"
#         elseif clicks % 3 == 0
#             cam.upvector[] = Vec3f0(1, 0, 0)
#             cam.eyeposition[] = Vec3f0(0, 0, 300)
#             cam.lookat[] = Vec3f0(0, 0, 0)
#             b1.plots[1][1][] = "Plane: YZ"
#         elseif clicks % 3 == 2
#             cam.upvector[] = Vec3f0(0, 0, 1)
#             cam.eyeposition[] = Vec3f0(300, 0, 0)
#             cam.lookat[] = Vec3f0(0, 0, 0)
#             b1.plots[1][1][] = "Plane: XY"
#         end     
#         update_cam!(hscene, cam)
#     end
# end

# # 3D/2D Toggle Button
# b2 = button(style, "3D"; dimensions = (100, 100))
# on(b2[end][:clicks]) do clicks
#     @show clicks
#     if iseven(clicks)
#         cam3d!(hscene)
#         center!(hscene)
#         b1.plots[1][1][] = "Plane: XYZ"
#         b2.plots[1][1][] = "3D"
#     else
#         cam2d!(hscene)
#         center!(hscene)
#         cam3d!(hscene)
#         cam = cameracontrols(hscene) 
#         cam.projectiontype[] = AbstractPlotting.Orthographic
#         cam.upvector[] = Vec3f0(1, 0, 0)
#         cam.eyeposition[] = Vec3f0(0, 0, 300)
#         cam.lookat[] = Vec3f0(0, 0, 0)
#         update_cam!(hscene, cam)
#         b1.plots[1][1][] = "Plane: YZ"
#         b2.plots[1][1][] = "2D"
#     end 
# end

#Layout  
# hbox(
#     vbox(scene3d, hscene),
#     vbox(sliders...,t, b2, b1)
# )
hbox(
    vbox(scene3d)
)

x = range(0, 10, length=100)
y = sin.(x)
Makie.lines(x, y)

end


