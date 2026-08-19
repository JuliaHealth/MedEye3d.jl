using Core: println
include("src/structs/DataStructs.jl")
include("src/structs/ForDisplayStructs.jl")
include("src/display/GLFW/MakieEventHandlers.jl")
using .MedEye3d.DataStructs
using .MedEye3d.ForDisplayStructs
using .MedEye3d.SegmentationDisplay.MakieEventHandlers

dat = zeros(Float32, 10, 10, 10)
dat2 = zeros(Float32, 10, 10, 10)
dataToScroll = [TwoDimRawDat(Float32, "Bone_Surface", dat), TwoDimRawDat(Float32, "Bone_Marrow", dat2)]
onScrollData = ScrollStruct(1, 1, 10, dataToScroll, Dict("Bone_Surface"=>1, "Bone_Marrow"=>2))
state = StateDataFields(1, 1, false, false, 0.0, CartesianIndex(1,1,1), SingleSliceDat(), onScrollData, nothing, nothing, nothing, nothing, nothing, nothing, nothing, false)

MakieEventHandlers.current_active_lesion_id[] = 11
MakieEventHandlers.bone_subsegments_cache[11] = (CartesianIndex.(1:2, 1:2, 1:2), CartesianIndex.(3:4, 3:4, 3:4))

MakieEventHandlers.reactToShowMaskLayer(ShowMaskLayerEvent(2, true), [state])
println("Sum Bone_Surface: ", sum(dat))
println("Sum Bone_Marrow: ", sum(dat2))
