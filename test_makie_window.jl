using Pkg
Pkg.activate("/workspaces/MedEye3d.jl")

using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using MedEye3d.MakieEvents
using MedEye3d.SegmentationDisplay
using MedEye3d.LesionMetadataWindow
using ColorTypes
using GLFW
using NIfTI
import Observables
import GLMakie

println("=== MAKIE WINDOW TEST ===")

image_path = "/root/data_decathlon/Task09_Spleen/imagesTr/spleen_10.nii.gz"
mask_path = "/root/data_decathlon/Task09_Spleen/labelsTr/spleen_10.nii.gz"
vol_img = Float32.(niread(image_path).raw)
vol_mask = Float32.(niread(mask_path).raw)

colors_mapped = [RGB(1.0, 0.0, 0.0), RGB(0.0, 1.0, 0.0), RGB(0.0, 0.0, 1.0)]
ts_img = TextureSpec{Float32}(name="CT", isMainImage=true, color=RGB(1.,1.,1.), minAndMaxValue=Float32.([-150,250]))
ts_mask = TextureSpec{Float32}(name="Mask", isMainImage=false, isMultiDiscreteMask=true, colorSet=colors_mapped, maskContribution=Float32(0.5), minAndMaxValue=Float32.([0, 3]))

tsa = Vector{Vector{TextureSpec}}([TextureSpec[deepcopy(ts_img), deepcopy(ts_mask)]])
spacing = (1.,1.,1.); origin = (0.,0.,0.)
dts = [DataToScrollDims(imageSize=size(vol_img), voxelSize=spacing, dimensionToScroll=3)]

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

println("Vulkan viewer created OK.")

# Now try the EXACT path that fails in production: create_metadata_window + display_metadata_window
println("Creating Makie control window (same as run_interactive_mrb.jl:905-906)...")
try
    active_lesion = Observables.Observable("(none)")
    lesion_ids = Observables.Observable(["(none)"])
    win = LesionMetadataWindow.create_metadata_window(active_lesion, lesion_ids, inst.channel)
    println("create_metadata_window OK.")
    screen = LesionMetadataWindow.display_metadata_window(win.fig)
    println("✅ display_metadata_window OK! GLMakie screen created.")
    sleep(3)
catch e
    println("❌ FAILED: $e")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

# Capture screenshot
ch = inst.channel
done = Channel{Bool}(1)
put!(ch, 30); sleep(3)
put!(ch, ScreenshotEvent("/workspaces/MedEye3d.jl/data/scr/makie_test.png", done))
take!(done)
println("Screenshot saved")

println("=== TEST COMPLETE ===")
