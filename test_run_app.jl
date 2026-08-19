using MedEye3d
using MedEye3d.SegmentationDisplay
using MedEye3d.SegmentationDisplay.MakieEventHandlers

# Initialize channel before
Main.eval(:(MRB_VIEWER_CHANNEL = Channel(512)))

t = errormonitor(@async begin
    try
        include("scripts/app/run_interactive_mrb.jl")
    catch e
        println("APP CRASHED: ", e)
        println(sprint(showerror, e, catch_backtrace()))
    end
end)

sleep(40) # Wait for it to load

ch = Main.MRB_VIEWER_CHANNEL
println("Putting AI event on channel...")
put!(ch, MedEye3d.ForDisplayStructs.AddAutoPetEvent("NNInteractive", ch))
sleep(5)
println("Done.")
