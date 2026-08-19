using MedEye3d
using MedEye3d.SegmentationDisplay
using MedEye3d.SegmentationDisplay.MakieEventHandlers

println("=== STARTING AI FLOW REPRODUCTION ===")
include("scripts/app/run_interactive_mrb.jl")

ch = Main.MRB_VIEWER_CHANNEL
sleep(3)

# 1. Select lesion 1
put!(ch, MedEye3d.ForDisplayStructs.SyncLesionEvent(1))
sleep(0.5)

# 2. Turn on painting with value 1
put!(ch, MedEye3d.ForDisplayStructs.PaintValEvent(1, true))
sleep(0.5)

# 3. Simulate painting some points in panel 1 manually, without simulating Mouse events,
# just by directly writing to the underlying array to simulate what drawing does.
st = SegmentationDisplay.mainMedEye3dInstance.states[1]
manualModifDat = nothing
for dat in st.onScrollData.dataToScroll
    if dat.name == "manualModif"
        manualModifDat = dat.dat
        break
    end
end
manualModifDat[33, 33, 33] = 1.0f0

println("Simulated painted 1 voxel in manualModif")

# 4. Trigger NNInteractive
put!(ch, MedEye3d.ForDisplayStructs.AddAutoPetEvent("NNInteractive", ch))

# Give it 5 seconds to run AI
sleep(5)

println("AI Status text: ", MakieEventHandlers.ai_status_text[])

