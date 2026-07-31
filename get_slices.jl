using Pkg
Pkg.activate(".")
include("scripts/run_goal_screenshot.jl")
println("Final slices:")
println(mainMedEye3dInstance.states[1].currentDisplayedSlice)
println(mainMedEye3dInstance.states[2].currentDisplayedSlice)
println(mainMedEye3dInstance.states[3].currentDisplayedSlice)
println(mainMedEye3dInstance.states[4].currentDisplayedSlice)
