using MedEye3d
using MedEye3d.SegmentationDisplay
using MedEye3d.SegmentationDisplay.MakieEventHandlers

println("=== STARTING LOG TEST ===")
include("scripts/app/run_interactive_mrb.jl")

ch = Main.MRB_VIEWER_CHANNEL
sleep(5)
put!(ch, MedEye3d.ForDisplayStructs.AddAutoPetEvent("NNInteractive", ch))
sleep(3)
