using MedEye3d
using MedEye3d.SegmentationDisplay
using MedEye3d.SegmentationDisplay.MakieEventHandlers

println("=== STARTING AI FLOW REPRODUCTION ===")

# Run the app in a task so we can interact with it
t = errormonitor(@async begin
    include("scripts/app/run_interactive_mrb.jl")
end)

# Wait for it to initialize
sleep(30)

ch = Main.MRB_VIEWER_CHANNEL

println("Sending SyncLesionEvent(1)...")
put!(ch, MedEye3d.ForDisplayStructs.SyncLesionEvent(1))
sleep(0.5)

println("Sending PaintValEvent...")
put!(ch, MedEye3d.ForDisplayStructs.PaintValEvent(1, true))
sleep(0.5)

st = SegmentationDisplay.mainMedEye3dInstance.states[1]
for dat in st.onScrollData.dataToScroll
    if dat.name == "manualModif"
        dat.dat[33, 33, 33] = 1.0f0
        println("Simulated painted 1 voxel in manualModif")
        break
    end
end

println("Sending AddAutoPetEvent...")
put!(ch, MedEye3d.ForDisplayStructs.AddAutoPetEvent("NNInteractive", ch))

# Give it 5 seconds to run AI
sleep(10)

println("AI Status text: ", MakieEventHandlers.ai_status_text[])
