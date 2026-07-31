using Pkg
Pkg.activate(".")
include("scripts/run_goal_screenshot.jl")

coronal_slice = vol_img_coronal[:, :, y_center]
println("Coronal slice max: ", maximum(coronal_slice))
println("Coronal slice min: ", minimum(coronal_slice))
println("Coronal slice mean: ", sum(coronal_slice)/length(coronal_slice))
