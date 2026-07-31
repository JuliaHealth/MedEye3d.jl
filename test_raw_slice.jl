using Pkg
Pkg.activate(".")
include("scripts/run_goal_screenshot.jl")
using Images, FileIO

slice_idx = 164
transverse_slice = vol_img_pad[:, :, slice_idx]

# Map -100 to 200 to 0.0 to 1.0
img_mapped = clamp.((transverse_slice .- (-100.0f0)) ./ 300.0f0, 0.0f0, 1.0f0)
img_gray = colorview(Gray, img_mapped')

save("data/test_raw_slice.png", img_gray)
println("Saved data/test_raw_slice.png")
