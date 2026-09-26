using Pkg
Pkg.activate("/workspaces/MedEye3d.jl")
using MedEye3d
println("Has app_is_loading? ", isdefined(MedEye3d.SegmentationDisplay.MakieEventHandlers, :app_is_loading))
