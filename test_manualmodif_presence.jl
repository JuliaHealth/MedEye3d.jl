using MedEye3d
using MedEye3d.SegmentationDisplay
using MedEye3d.SegmentationDisplay.MakieEventHandlers

println("=== STARTING TEST ===")
include("scripts/app/run_interactive_mrb.jl")

sleep(5)
st = SegmentationDisplay.mainMedEye3dInstance.states[1]
println("Panel 1 volumes:")
for dat in st.onScrollData.dataToScroll
    println("  - $(dat.name)")
end
