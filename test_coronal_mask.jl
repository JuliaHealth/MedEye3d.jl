using Pkg
Pkg.activate(".")
include("scripts/run_goal_screenshot.jl")

coronal_mask_slice = vol_mask_coronal[:, :, y_center]
println("Coronal mask slice max: ", maximum(coronal_mask_slice))
println("Coronal mask slice sum: ", sum(coronal_mask_slice))
