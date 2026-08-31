# Minimal production-like test that exercises the Makie window creation
using Pkg
Pkg.activate("/workspaces/MedEye3d.jl")

using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using MedEye3d.MakieEvents
using MedEye3d.SegmentationDisplay
using ColorTypes
using GLFW
using NIfTI

println("=== PRODUCTION-LIKE TEST WITH MAKIE WINDOW ===")

image_path = "/root/data_decathlon/Task09_Spleen/imagesTr/spleen_10.nii.gz"
mask_path = "/root/data_decathlon/Task09_Spleen/labelsTr/spleen_10.nii.gz"
nii_img = niread(image_path)
nii_mask = niread(mask_path)
vol_img = Float32.(nii_img.raw)
vol_mask = Float32.(nii_mask.raw)

colors_mapped = [RGB(1.0, 0.0, 0.0), RGB(0.0, 1.0, 0.0), RGB(0.0, 0.0, 1.0)]

ts_img = TextureSpec{Float32}(name="CT", isMainImage=true, color=RGB(1.,1.,1.), minAndMaxValue=Float32.([-150,250]))
ts_mask = TextureSpec{Float32}(
    name="Mask", isMainImage=false,
    isMultiDiscreteMask=true,
    colorSet=colors_mapped,
    maskContribution=Float32(0.5),
    minAndMaxValue=Float32.([0, length(colors_mapped)])
)

tsa = Vector{Vector{TextureSpec}}([TextureSpec[deepcopy(ts_img), deepcopy(ts_mask)]])

spacing = (1.,1.,1.); origin = (0.,0.,0.)
dts = Vector{DataToScrollDims}()
push!(dts, DataToScrollDims(imageSize=size(vol_img), voxelSize=spacing, dimensionToScroll=3))

inst = SegmentationDisplay.coordinateDisplay(
    tsa, Float32(1.0), dts, [spacing], [origin],
    Dict{String,Vector}("supervoxel_vertices"=>[],"supervoxel_indices"=>[]),
    Dict{Int64,Dict{Int64,Dict{String,Any}}}()
)

SD = FullScrollableDat(dataToScroll=[
    ThreeDimRawDat{Float32}(type=Float32, name="CT", dat=vol_img),
    ThreeDimRawDat{Float32}(type=Float32, name="Mask", dat=vol_mask)
], dimensionToScroll=3)
SegmentationDisplay.passDataForScrolling(inst, SD)
sleep(1)

println("Vulkan viewer window created OK.")

# Now try to create a GLMakie window — this is what was failing
println("Creating GLMakie window...")
try
    using GLMakie
    using Makie
    fig = Figure(size=(300, 200))
    Label(fig[1,1], "Test Makie Window")
    screen = GLMakie.Screen(fig.scene; renderloop=SegmentationDisplay.synchronized_makie_renderloop)
    println("✅ GLMakie window created successfully!")
    sleep(2)
    # Close the screen 
    GLMakie.close(screen)
catch e
    println("❌ GLMakie window creation failed: $e")
    println(stacktrace(catch_backtrace()))
end

# Take screenshot 
ch = inst.channel
done = Channel{Bool}(1)
put!(ch, 30); sleep(3)
put!(ch, ScreenshotEvent("/workspaces/MedEye3d.jl/data/scr/prod_test.png", done))
take!(done)
println("Screenshot saved")

println("=== TEST COMPLETE ===")
