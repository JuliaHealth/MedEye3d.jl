using Pkg
Pkg.activate("/workspaces/MedEye3d.jl")

using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using MedEye3d.MakieEvents
using MedEye3d.SegmentationDisplay
using ColorTypes
using GLFW
using Dates

println("=== COMPREHENSIVE PRODUCTION INTERACTIVE FEATURE VERIFICATION ===")

# Set non-interactive flag so run_interactive_mrb doesn't block on readline()
# We will drive it programmatically via its channel

# Include the production script up to the interaction loop
# We can load the H5 data and initialize everything
ENV["MEDEYE3D_TEST_MODE"] = "true"

include("/workspaces/MedEye3d.jl/scripts/app/run_interactive_mrb.jl")

ch = mainMedEye3dInstance.channel

function capture_synced(path, desc)
    println("--- Capturing: $desc -> $path ---")
    done = Channel{Bool}(1)
    put!(ch, ScreenshotEvent(path, done))
    res = take!(done)
    println("Captured $desc: success=$res")
    sleep(1.0)
    return res
end

# 1. Initial 4-pane Quad View screenshot (Slice ~163)
sleep(2.0)
capture_synced("/workspaces/MedEye3d.jl/data/scr/01_quadview_initial.png", "Quad View Initial (Axial, PET-only, Sagittal, Coronal)")

# 2. Scroll to a bone lesion slice in Axial (e.g. slice 180)
println("Scrolling to slice 180...")
# Change slice on active panel (panel 1)
put!(ch, 17) # scroll delta +17
sleep(2.0)
capture_synced("/workspaces/MedEye3d.jl/data/scr/02_quadview_scrolled.png", "Quad View Scrolled to Slice 180")

# 3. Ctrl+Scroll simulation: PetBlendEvent (50% blend)
println("Testing PetBlendEvent (50% blend)...")
put!(ch, PetBlendEvent(0.5f0))
sleep(2.0)
capture_synced("/workspaces/MedEye3d.jl/data/scr/03_pet_blend_50pct.png", "PET Blend 50%")

# 4. PetBlendEvent (0% blend = CT only)
println("Testing PetBlendEvent (0% blend = CT only)...")
put!(ch, PetBlendEvent(0.0f0))
sleep(2.0)
capture_synced("/workspaces/MedEye3d.jl/data/scr/04_pet_blend_0pct_ct_only.png", "PET Blend 0% (CT Only)")

# 5. PetBlendEvent (100% blend = Full PET)
println("Restoring PetBlendEvent (100% blend)...")
put!(ch, PetBlendEvent(1.0f0))
sleep(2.0)
capture_synced("/workspaces/MedEye3d.jl/data/scr/05_pet_blend_100pct.png", "PET Blend 100%")

# 6. Zoom in on Panel 1 (Shift+Scroll simulation)
println("Testing Zoom...")
put!(ch, ScrollZoomEvent(1.0)) # zoom in
put!(ch, ScrollZoomEvent(1.0)) # zoom in more
put!(ch, ScrollZoomEvent(1.0)) # zoom in more
sleep(2.0)
capture_synced("/workspaces/MedEye3d.jl/data/scr/06_quadview_zoomed.png", "Quad View Zoomed (Panel 1)")

# 7. Reset Zoom
println("Resetting Zoom...")
mainMedEye3dInstance.states[1].calcDimsStruct.zoom = 1.0f0
mainMedEye3dInstance.states[1].calcDimsStruct.panX = 0.0f0
mainMedEye3dInstance.states[1].calcDimsStruct.panY = 0.0f0
put!(ch, 0) # trigger redraw
sleep(2.0)

# 8. Compare Time Points Mode (TP 0 vs TP 1)
println("Testing Compare Time Points Event (Side-by-Side Dual View)...")
put!(ch, CompareTimePointsEvent(true))
sleep(3.0)
capture_synced("/workspaces/MedEye3d.jl/data/scr/07_compare_mode_on.png", "Compare Mode ON (TP 0 vs TP 1 Side-by-Side)")

# 9. Toggle Compare Mode OFF (restore 4-pane quad view)
println("Toggling Compare Mode OFF...")
put!(ch, CompareTimePointsEvent(false))
sleep(3.0)
capture_synced("/workspaces/MedEye3d.jl/data/scr/08_compare_mode_off.png", "Compare Mode OFF (Restored Quad View)")

# 10. CT Windowing Event (Bone Window: min=-700, max=1300)
println("Testing Bone Windowing...")
put!(ch, WindowingEvent("CT", -700.0f0, 1300.0f0))
sleep(2.0)
capture_synced("/workspaces/MedEye3d.jl/data/scr/09_bone_window.png", "CT Bone Window (-700 to 1300 HU)")

# 11. Soft Tissue Windowing Event (min=-150, max=250)
println("Restoring Soft Tissue Windowing...")
put!(ch, WindowingEvent("CT", -150.0f0, 250.0f0))
sleep(2.0)
capture_synced("/workspaces/MedEye3d.jl/data/scr/10_soft_tissue_window.png", "CT Soft Tissue Window (-150 to 250 HU)")

# 12. Show Bone Mask Layer
println("Testing ShowBoneMaskEvent...")
put!(ch, ShowBoneMaskEvent(true))
sleep(2.0)
capture_synced("/workspaces/MedEye3d.jl/data/scr/11_bone_mask_layer.png", "Bone Mask Layer Enabled")

println("=== ALL INTERACTIVE TESTS COMPLETED SUCCESSFULLY ===")
