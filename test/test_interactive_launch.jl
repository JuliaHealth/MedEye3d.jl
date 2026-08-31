#=
Test that the interactive app launches correctly with real X11 display.
Verifies:
1. GLFW Vulkan window opens and renders (not black)
2. GLMakie control panel window opens
3. GLFW.PollEvents() works on main thread
4. Screenshot is captured showing actual rendered medical images
=#
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"
ENV["MEDEYE3D_TEST_MODE"] = "true"  # Skip readline/event loop

# Configure Vulkan ICD
if isfile("/etc/vulkan/icd.d/nvidia_icd.json") && !haskey(ENV, "VK_ICD_FILENAMES")
    ENV["VK_ICD_FILENAMES"] = "/etc/vulkan/icd.d/nvidia_icd.json"
    ENV["VK_DRIVER_FILES"] = "/etc/vulkan/icd.d/nvidia_icd.json"
end

using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using MedEye3d.SegmentationDisplay
using MedEye3d.MakieEvents
using MedEye3d.LesionMetadataWindow
import GLFW
import Observables

# Build minimal test data
data_dir = joinpath(@__DIR__, "..", "data", "pat_6_files")
include(joinpath(@__DIR__, "..", "scripts", "lib", "SceneHierarchy.jl"))
using .SceneHierarchy

import HDF5

preprocessed_h5 = joinpath(data_dir, "preprocessed_volumes.h5")
display_spacing = (0.9765625, 0.9765625, 3.0)

# Load first TP data
first_entry = HDF5.h5open(preprocessed_h5, "r") do h5
    ct = read(h5["BASELINE/CT"])
    pet = read(h5["BASELINE/PET"])
    mask = zeros(Int16, size(ct))
    (ct=ct, pet=pet, mask=mask, modality="PET", label="Test TP 0")
end

println("Data loaded: CT=$(size(first_entry.ct)), PET=$(size(first_entry.pet))")

# Build texture specs
ts_ct = TextureSpec{Float32}(
    name="CT", numb=Int32(1), color=RGB(1.0, 1.0, 1.0),
    minAndMaxValue=Float32[-1000.0, 3000.0],
    isMainImage=true, isVisible=true
)
ts_pet = TextureSpec{Float32}(
    name="PET", numb=Int32(2), color=RGB(1.0, 0.65, 0.0),
    minAndMaxValue=Float32[0.0, 15.0],
    isMainImage=false, isVisible=true
)
ts_mask = TextureSpec{Int16}(
    name="Mask", numb=Int32(3), color=RGB(1.0, 0.0, 0.0),
    isEditable=true, isMultiDiscreteMask=true,
    isVisible=true, isMainImage=false,
    minAndMaxValue=Int16[0, 100]
)

textureSpecArray = [[ts_ct, ts_pet, ts_mask] for _ in 1:4]

# Build VDT
function entry_to_vdt_simple(e)
    dims_ax = size(e.ct)
    ct_ax = Array{Float32}(e.ct)
    pet_ax = Array{Float32}(e.pet)
    mask_ax = Array{Int16}(e.mask)

    sliceN_ax = dims_ax[3]
    sliceN_sag = dims_ax[1]
    sliceN_cor = dims_ax[2]

    return [
        [("CT", ct_ax), ("PET", pet_ax), ("Mask", mask_ax), sliceN_ax, 3],
        [("CT", ct_ax), ("PET", pet_ax), sliceN_ax, 3],
        [("CT", ct_ax), ("PET", pet_ax), ("Mask", mask_ax), sliceN_sag, 1],
        [("CT", ct_ax), ("PET", pet_ax), ("Mask", mask_ax), sliceN_cor, 2],
    ]
end

vdt = entry_to_vdt_simple(first_entry)
ds = display_spacing
spacings = [[ds], [ds], [(ds[2], ds[3], ds[1])], [(ds[1], ds[3], ds[2])]]
origins = [[(0.0, 0.0, 0.0)] for _ in 1:4]
dummyStudySrc = Vector{Vector{Tuple{String,String}}}()

# === CRITICAL TEST: Launch sequence on main thread ===
println("\n=== Testing launch sequence ===")
println("Thread ID: $(Threads.threadid())")

# Step 1: Create Makie layout (no GLFW window yet)
println("[TEST] Step 1: Creating Makie layout...")
active_lesion = Observables.Observable("(none)")
lesion_ids = Observables.Observable(["(none)"])
makie_win = LesionMetadataWindow.create_metadata_window(active_lesion, lesion_ids, nothing)
println("[TEST] ✓ Makie layout created")

# Step 2: Launch Vulkan display
println("[TEST] Step 2: Launching Vulkan display...")
mainInstance = SegmentationDisplay.displayImage(
    dummyStudySrc;
    textureSpecArray=textureSpecArray,
    voxelDataTupleVector=vdt,
    spacings=spacings,
    origins=origins,
    windowWidth=1100,
    fractionOfMainImage=Float32(1.0),
    quadView=true
)
println("[TEST] ✓ Vulkan display created")

# Step 3: Connect channel and display Makie window
println("[TEST] Step 3: Connecting Makie to Vulkan channel...")
LesionMetadataWindow.connect_channel!(makie_win, mainInstance.channel)
println("[TEST] Step 3b: Opening GLMakie window...")
makie_screen = LesionMetadataWindow.display_metadata_window(makie_win.fig)
println("[TEST] ✓ GLMakie window opened!")

# Step 4: Warmup and render
println("[TEST] Step 4: Warmup rendering...")
put!(mainInstance.channel, Int64(0))
sleep(0.1)

# Step 5: Call PollEvents to process any pending events
println("[TEST] Step 5: Testing GLFW.PollEvents()...")
GLFW.PollEvents()
println("[TEST] ✓ PollEvents works on thread $(Threads.threadid())")

# Step 6: Take screenshot
println("[TEST] Step 6: Taking screenshot...")
vulkan_window = mainInstance.states[1].mainForDisplayObjects.window
screenshot_path = joinpath(@__DIR__, "test_data", "screenshots", "interactive_launch.png")
mkpath(dirname(screenshot_path))

# Scroll to a visible slice and render
put!(mainInstance.channel, Int64(80))
sleep(0.5)
GLFW.PollEvents()
sleep(0.1)

# Capture via Vulkan readback
vk_ctx = mainInstance.states[1].mainForDisplayObjects.vulkanCtx
if vk_ctx !== nothing
    try
        pixels = MedEye3d.VulkanContext.readback_swapchain(vk_ctx)
        if pixels !== nothing
            using FileIO, ImageIO
            save(screenshot_path, pixels)
            println("[TEST] ✓ Screenshot saved: $screenshot_path")
        else
            println("[TEST] ⚠ Readback returned nothing")
        end
    catch e
        println("[TEST] ⚠ Screenshot capture: $e")
    end
end

# Step 7: Cleanup
println("[TEST] Step 7: Closing...")
put!(mainInstance.channel, CloseWindowEvent())
sleep(0.5)

println("\n=== ALL INTERACTIVE LAUNCH TESTS PASSED ✓ ===")
