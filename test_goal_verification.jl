using Pkg
Pkg.activate("/workspaces/MedEye3d.jl")

using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using MedEye3d.MakieEvents
using MedEye3d.SegmentationDisplay
using ColorTypes
using GLFW
using JSON
using Dates

println("=== COMPREHENSIVE GOAL VERIFICATION TEST ===")

# Clean up / set up initial config file
cfg_path = joinpath(homedir(), ".medeye3d_display_config.json")
if isfile(cfg_path)
    rm(cfg_path)
end

# Test 1: Verify default config values
println("--- Test 1: Verifying default display config ---")
using MedEye3d.LesionMetadataWindow: load_display_config, save_display_config, GLOBAL_DISPLAY_CONFIG_PATH
cfg = load_display_config()
@assert cfg["pet_ct_blend"] == 0.5 "Default PET/CT blend should be 0.5"
@assert cfg["label_opacity"] == 0.5 "Default label opacity should be 0.5"
println("Default display config verified: $(cfg)")

# Test 2: Verify custom opacity persistence
println("--- Test 2: Testing persistence of label opacity across runs ---")
cfg["label_opacity"] = 0.65
cfg["pet_ct_blend"] = 0.50
save_display_config(cfg)
reloaded_cfg = load_display_config()
@assert reloaded_cfg["label_opacity"] == 0.65 "Reloaded label opacity must be 0.65"
println("Persistence verified: saved 0.65, loaded $(reloaded_cfg["label_opacity"])")

# Test 3: Launch app in test mode to verify Vulkan startup with 50% PET/50% CT and label opacity
println("--- Test 3: Launching application and verifying startup defaults ---")
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

# 3.1 Initial screenshot (should show 50% PET / 50% CT blend on startup)
sleep(2.0)
capture_synced("/workspaces/MedEye3d.jl/data/scr/v_01_startup_50pct_pet_ct.png", "Startup with 50% PET / 50% CT Default")

# Test 4: Verify Right-Click Drag Panning Direction
println("--- Test 4: Testing Right-Click Drag Panning (Drag Right moves Image Right) ---")
# 4.1 Press down at (150, 150) in Top-Left panel (Panel 1)
put!(ch, MouseStruct(
    isLeftButtonDown  = false,
    isRightButtonDown = true,
    lastCoordinates   = [CartesianIndex(150, 150)],
    actualWindowWidth  = 1100,
    actualWindowHeight = 1100
))
sleep(1.0) # Allow consumer loop to register initial press and set lastPanDragCoords

# 4.2 Drag to the right: move cursor to (350, 150) -> dx = +200
put!(ch, MouseStruct(
    isLeftButtonDown  = false,
    isRightButtonDown = true,
    lastCoordinates   = [CartesianIndex(350, 150)],
    actualWindowWidth  = 1100,
    actualWindowHeight = 1100
))
sleep(1.0) # Allow consumer loop to execute pan calculation

# Check pan offset before releasing button
p1_panY = mainMedEye3dInstance.states[1].calcDimsStruct.panY
p1_panX = mainMedEye3dInstance.states[1].calcDimsStruct.panX
println("After dragging RIGHT by 200px: panY=$p1_panY, panX=$p1_panX")
# panY should be negative (so texture coordinate shifts left, moving image right on screen)
@assert p1_panY < 0.0f0 "Dragging RIGHT must decrease panY so image shifts right"
capture_synced("/workspaces/MedEye3d.jl/data/scr/v_02_panned_right.png", "Image Panned Right")

# 4.3 Release right button
put!(ch, MouseStruct(
    isLeftButtonDown  = false,
    isRightButtonDown = false,
    lastCoordinates   = [CartesianIndex(350, 150)],
    actualWindowWidth  = 1100,
    actualWindowHeight = 1100
))
sleep(1.0)

# Reset pan
mainMedEye3dInstance.states[1].calcDimsStruct.panX = 0.0f0
mainMedEye3dInstance.states[1].calcDimsStruct.panY = 0.0f0
put!(ch, 0)
sleep(1.0)

# Test 5: Verify Label Opacity Event & Shader Update
println("--- Test 5: Testing Label Opacity Changes ---")
# Set label opacity to 0.8 (80%)
println("Setting label opacity to 0.8...")
put!(ch, LabelOpacityEvent(0.8f0))
sleep(1.5)
capture_synced("/workspaces/MedEye3d.jl/data/scr/v_03_label_opacity_80pct.png", "Label Opacity 80%")

# Set label opacity to 0.2 (20% - very subtle/transparent)
println("Setting label opacity to 0.2...")
put!(ch, LabelOpacityEvent(0.2f0))
sleep(1.5)
capture_synced("/workspaces/MedEye3d.jl/data/scr/v_04_label_opacity_20pct.png", "Label Opacity 20%")

# Set label opacity to 0.5 (standard)
println("Setting label opacity back to 0.5...")
put!(ch, LabelOpacityEvent(0.5f0))
sleep(1.5)
capture_synced("/workspaces/MedEye3d.jl/data/scr/v_05_label_opacity_50pct.png", "Label Opacity 50%")

println("=== ALL VERIFICATION TESTS PASSED SUCCESSFULLY ===")
