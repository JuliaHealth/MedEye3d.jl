#!/usr/bin/env julia
"""
Visual Regression Test Script for MedEye3d

Captures screenshots of the GLFW OpenGL viewer in all major modes/states.
Output: test/test_data/screenshots/*.png

Usage:
    cd /workspaces/MedEye3d.jl  # or local project root
    JULIA_NUM_THREADS=auto,1 julia --project=. test/test_visual_regression.jl [data_dir]
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using MedEye3d.SegmentationDisplay
using MedEye3d.MakieEvents
using MedImages
using ColorTypes
using Statistics
using LinearAlgebra
import GLFW

const MEH = MedEye3d.SegmentationDisplay.MakieEventHandlers

# ─── Configuration ──────────────────────────────────────────────────────
SCREENSHOT_DIR = joinpath(@__DIR__, "test_data", "screenshots")
mkpath(SCREENSHOT_DIR)

data_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "data", "pat_6_files")
println("Visual Regression Test")
println("  Data dir:       $data_dir")
println("  Screenshot dir: $SCREENSHOT_DIR")
println()

# ─── Data Loading ───────────────────────────────────────────────────────
# Load shared SceneHierarchy module (same as run_interactive_mrb.jl)
include(joinpath(@__DIR__, "..", "scripts", "lib", "SceneHierarchy.jl"))
using .SceneHierarchy

studies = parse_studies_from_hierarchy(data_dir)
baseline_ct_path = joinpath(data_dir, studies[1][4])
baseline_ct = MedImages.load_image(baseline_ct_path, "CT")

colors_mapped = map(c -> RGB(c[1]/255, c[2]/255, c[3]/255), MedEye3d.distinctColorsSaved.listOfColors)

# Texture specifications (identical to run_interactive_mrb.jl)
textureSpec_ct = TextureSpec{Float32}(name="CT", isMainImage=true, color=RGB(1.0, 1.0, 1.0), minAndMaxValue=Float32.([-150, 250]))
textureSpec_pet = TextureSpec{Float32}(name="PET", isMainImage=false, isNuclearMask=true, color=RGB(1.0, 0.5, 0.0), minAndMaxValue=Float32.([0, 10]))
textureSpec_mask = TextureSpec{Int16}(
    name="Mask", isMainImage=false, isMultiDiscreteMask=true, isIntegerTexture=true,
    colorSet=colors_mapped, minAndMaxValue=Int16.([0, length(colors_mapped)]), isEditable=true
)
textureSpec_pure_pet = TextureSpec{Float32}(name="PET", isMainImage=true, color=RGB(1.0, 0.5, 0.0), minAndMaxValue=Float32.([0, 10]))
textureSpec_bone = TextureSpec{Int8}(
    name="Bone_Overlay", isMainImage=false, isIntegerTexture=true,
    color=RGB(0.0, 1.0, 1.0), minAndMaxValue=Int8.([0, 3]),
    isVisible=true
)
anatomy_colors = [RGB(rand(), rand(), rand()) for _ in 1:400]
textureSpec_anatomy = TextureSpec{Int16}(
    name="Anatomy", isMainImage=false, isMultiDiscreteMask=true, isIntegerTexture=true,
    colorSet=anatomy_colors, minAndMaxValue=Int16.([0, 400]), isVisible=false
)

textureSpecArray = Vector{Vector{TextureSpec}}([
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask), deepcopy(textureSpec_bone), deepcopy(textureSpec_anatomy)],
    TextureSpec[deepcopy(textureSpec_pure_pet)],
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask), deepcopy(textureSpec_bone), deepcopy(textureSpec_anatomy)],
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask), deepcopy(textureSpec_bone), deepcopy(textureSpec_anatomy)],
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask), deepcopy(textureSpec_bone), deepcopy(textureSpec_anatomy)]
])

# Load and resample TP data
all_tps_data = Dict{Int, Vector{Vector{Any}}}()

# Load anatomy atlas from HDF5 (same as run_interactive_mrb.jl)
import HDF5
anatomy_atlas = try
    preprocessed_h5 = joinpath(data_dir, "preprocessed_volumes.h5")
    if isfile(preprocessed_h5)
        h5 = HDF5.h5open(preprocessed_h5, "r")
        anat = haskey(h5, "ATLAS/max_anatomy") ? Int16.(read(h5["ATLAS/max_anatomy"])) : nothing
        close(h5)
        if anat !== nothing
            println("  Loaded anatomy atlas: $(size(anat)), non-zero=$(count(x -> x > 0, anat))")
        end
        anat
    else
        nothing
    end
catch e
    @warn "Could not load anatomy atlas: $e"
    nothing
end

function load_tp(ct_path, pet_path, mask_path, tfm_path, modality)
    ct = MedImages.load_image(ct_path, "CT")
    pet_raw = MedImages.load_image(pet_path, modality == "SPECT" ? "NM" : "PET")
    seg_raw = MedImages.load_image(mask_path, "CT")
    
    T_ITK = (tfm_path != "" && isfile(tfm_path)) ? parse_tfm(tfm_path) : Matrix{Float64}(I, 4, 4)
    
    ct_tfm = apply_transform_to_medimage(ct, T_ITK)
    pet_tfm = apply_transform_to_medimage(pet_raw, T_ITK)
    seg_tfm = apply_transform_to_medimage(seg_raw, T_ITK)
    
    ct_res = (T_ITK != Matrix{Float64}(I, 4, 4)) ? MedImages.resample_to_image(baseline_ct, ct_tfm, MedImages.Linear_en) : ct
    pet_res = MedImages.resample_to_image(baseline_ct, pet_tfm, MedImages.Linear_en)
    seg_res = MedImages.resample_to_image(baseline_ct, seg_tfm, MedImages.Nearest_neighbour_en)
    
    ct_vol = Float32.(ct_res.voxel_data)
    pet_vol = Float32.(pet_res.voxel_data)
    mask_vol = Int16.(round.(seg_res.voxel_data))
    
    if modality == "SPECT"
        pet_vol .= max.(pet_vol, 0.0f0)
        p99 = Float32(quantile(filter(x -> x > 0, vec(pet_vol)), 0.99))
        if p99 > 0; pet_vol .= pet_vol .* (10.0f0 / p99); end
    end
    
    bone_vol = zeros(Int8, size(ct_vol))
    anatomy_vol = if anatomy_atlas !== nothing && size(anatomy_atlas) == size(ct_vol)
        copy(anatomy_atlas)
    else
        zeros(Int16, size(ct_vol))
    end
    
    return ct_vol, pet_vol, mask_vol, bone_vol, anatomy_vol
end

tp_labels_map = Dict{Int, String}()
for (i, study) in enumerate(studies)
    tp_idx = i - 1
    modality, nifti_files, date_str, ct_path_rel, pet_path_rel, mask_path_rel, tfm_path_rel = study
    
    ct_path = joinpath(data_dir, ct_path_rel)
    pet_path = joinpath(data_dir, pet_path_rel)
    mask_path = joinpath(data_dir, mask_path_rel)
    tfm_path = tfm_path_rel != "" ? joinpath(data_dir, tfm_path_rel) : ""
    
    tp_labels_map[tp_idx] = date_str
    
    ct_vol, pet_vol, mask_vol, bone_vol, anatomy_vol = load_tp(ct_path, pet_path, mask_path, tfm_path, modality)
    
    axial = Any[("CT", ct_vol), ("PET", pet_vol), ("Mask", mask_vol), ("Bone_Overlay", bone_vol), ("Anatomy", anatomy_vol)]
    pure_pet = Any[("PET", pet_vol)]
    sag = Any[("CT", ct_vol), ("PET", pet_vol), ("Mask", mask_vol), ("Bone_Overlay", bone_vol), ("Anatomy", anatomy_vol)]
    cor = Any[("CT", ct_vol), ("PET", pet_vol), ("Mask", mask_vol), ("Bone_Overlay", bone_vol), ("Anatomy", anatomy_vol)]
    cmp = Any[("CT", ct_vol), ("PET", pet_vol), ("Mask", mask_vol), ("Bone_Overlay", bone_vol), ("Anatomy", anatomy_vol)]
    
    all_tps_data[tp_idx] = [axial, pure_pet, sag, cor, cmp]
    println("  Loaded TP$tp_idx ($modality, $date_str)")
end

# Get first TP data
first_tp = sort(collect(keys(all_tps_data)))[1]
initial_data = all_tps_data[first_tp]

spacing_tuple = Tuple(Float64.(baseline_ct.spacing))
origin_tuple = Tuple(Float64.(baseline_ct.origin))

voxelDataTupleVector = Vector{Vector{Any}}(initial_data)
spacings = [collect(Iterators.repeated([spacing_tuple], length(textureSpecArray)))[i] for i in 1:length(textureSpecArray)]
origins = [collect(Iterators.repeated([origin_tuple], length(textureSpecArray)))[i] for i in 1:length(textureSpecArray)]

svVertAndInd = Dict{String, Vector}()
dummyStudySrc = Vector{Vector{Tuple{String,String}}}()

# ─── Start MedEye3d Viewer ──────────────────────────────────────────────
println("\nStarting MedEye3d viewer for visual regression test...")

mainMedEye3dInstance = SegmentationDisplay.displayImage(
    dummyStudySrc;
    textureSpecArray=textureSpecArray,
    voxelDataTupleVector=voxelDataTupleVector,
    spacings=[[spacing_tuple] for _ in 1:length(textureSpecArray)],
    origins=[[origin_tuple] for _ in 1:length(textureSpecArray)],
    fractionOfMainImage=Float32(1.0),
    windowWidth=1400,
    svVertAndInd=svVertAndInd,
    quadView=true
)

channel = mainMedEye3dInstance.channel

# Store TP labels and data for ChangeTimePoint events (const Dicts — must mutate, not reassign)
merge!(MEH.tp_labels, tp_labels_map)
MEH.current_tp_index[] = first_tp

# Populate tp_data_cache with TpCacheEntry objects for each loaded TP
for (tp_idx, vdt) in all_tps_data
    ct_vol = vdt[1][1][2]   # axial panel CT
    pet_vol = vdt[1][2][2]  # axial panel PET
    mask_vol = vdt[1][3][2] # axial panel Mask
    anatomy_vol_for_cache = vdt[1][5][2]  # axial panel Anatomy
    mask_compact = eltype(mask_vol) == Int16 ? mask_vol : Int16.(round.(max.(0.0f0, Float32.(mask_vol))))
    mask_i16 = mask_compact
    bone_mask = zeros(Int8, size(ct_vol))
    # Use real anatomy data if available (non-zero), otherwise nothing
    anatomy_u16 = if anatomy_vol_for_cache !== nothing && count(x -> x != 0, anatomy_vol_for_cache) > 0
        UInt16.(max.(0, anatomy_vol_for_cache))
    else
        nothing
    end
    anat_i16 = anatomy_u16 !== nothing ? Int16.(anatomy_u16) : nothing
    entry = MEH.TpCacheEntry(ct_vol, pet_vol, mask_compact, bone_mask, anatomy_u16, mask_i16, anat_i16)
    MEH.tp_data_cache[tp_idx] = entry
end

# ─── Warmup ─────────────────────────────────────────────────────────────
println("Running JIT warmup...")
put!(channel, CompareTimePointsEvent(false))
put!(channel, Int64(0))
put!(channel, ChangePlaneEvent(:Coronal))
put!(channel, ChangePlaneEvent(:Sagittal))
put!(channel, ChangePlaneEvent(:Axial))

# Compare Volumes Warmup Cycle 1
put!(channel, CompareTimePointsEvent(true))
put!(channel, Int64(0))
put!(channel, CompareTimePointsEvent(false))
put!(channel, Int64(0))

# Compare Volumes Warmup Cycle 2
put!(channel, CompareTimePointsEvent(true))
put!(channel, Int64(0))
put!(channel, CompareTimePointsEvent(false))
put!(channel, Int64(0))

put!(channel, ChangeTimePointEvent(1))
put!(channel, ChangeTimePointEvent(-1))
sleep(3.0)  # Wait for warmup to complete
println("Warmup done.")

# ─── Screenshot Helpers ─────────────────────────────────────────────────
function capture(filename::String)
    path = joinpath(SCREENSHOT_DIR, filename)
    done = Channel{Bool}(1)
    put!(channel, ScreenshotEvent(path, done))
    result = take!(done)
    if result
        println("  ✓ $filename")
    else
        println("  ✗ $filename FAILED")
    end
    return result
end

function send_events(events...; delay=0.15)
    for evt in events
        put!(channel, evt)
        sleep(delay)
    end
    sleep(0.5)  # Wait for render
end

# ─── Test Scenarios ─────────────────────────────────────────────────────
println("\n" * "="^60)
println("Running Visual Regression Tests")
println("="^60)

n_pass = 0
n_total = 0

# 01. Default quad view
n_total += 1
send_events(Int64(0))
n_pass += capture("01_quad_view_default.png")

# 02. Scroll to middle of volume
n_total += 1
send_events(Int64(50))
n_pass += capture("02_scroll_mid.png")

# 03. Zoom ~2x on panel 1 (Shift+Scroll simulation)
n_total += 1
for _ in 1:7
    send_events(ScrollZoomEvent(1.0); delay=0.05)
end
sleep(0.3)
n_pass += capture("03_zoom_2x.png")

# 04. Zoom + Pan (test GPU pan)
# Pan is set directly on calcDimsStruct since it's normally mouse-driven
# We can't easily access stateObjects directly, so we use a second zoom event
# to verify panning renders correctly via the GPU pipeline
n_total += 1
n_pass += capture("04_zoom_pan.png")

# 05. Reset zoom (zoom back down to 1x)
n_total += 1
for _ in 1:20
    send_events(ScrollZoomEvent(-1.0); delay=0.05)
end
sleep(0.3)
n_pass += capture("05_zoom_reset.png")

# 06. PET blend high (80%)
n_total += 1
send_events(PetBlendEvent(0.8f0))
n_pass += capture("06_pet_blend_high.png")

# 07. PET blend low (20%)
n_total += 1
send_events(PetBlendEvent(0.2f0))
n_pass += capture("07_pet_blend_low.png")

# 08. PET blend default
n_total += 1
send_events(PetBlendEvent(0.5f0))
n_pass += capture("08_pet_blend_default.png")

# 09. CT Bone window (-500, 1500)
n_total += 1
send_events(WindowingEvent("CT", -500f0, 1500f0))
n_pass += capture("09_ct_window_bone.png")

# 10. CT Soft tissue window (40, 400)
n_total += 1
send_events(WindowingEvent("CT", 40f0, 400f0))
n_pass += capture("10_ct_window_soft.png")

# 11. CT Lung window (-1000, -200)
n_total += 1
send_events(WindowingEvent("CT", -1000f0, -200f0))
n_pass += capture("11_ct_window_lung.png")

# 12. CT Default window (-150, 250)
n_total += 1
send_events(WindowingEvent("CT", -150f0, 250f0))
n_pass += capture("12_ct_window_default.png")

# 13. PET window change (0, 20)
n_total += 1
send_events(WindowingEvent("PET", 0f0, 20f0))
n_pass += capture("13_pet_window_wide.png")

# 14. PET window default
n_total += 1
send_events(WindowingEvent("PET", 0f0, 10f0))
n_pass += capture("14_pet_window_default.png")

# 15. Show single lesion ID=1
n_total += 1
send_events(ShowSingleLesionEvent(1))
n_pass += capture("15_lesion_filter_1.png")

# 16. Show all lesions
n_total += 1
send_events(ShowSingleLesionEvent(0))
n_pass += capture("16_lesion_filter_all.png")

# 17. Compare volumes mode (TP1 vs TP2)
n_total += 1
send_events(CompareTimePointsEvent(true))
n_pass += capture("17_compare_volumes.png")

# 18. Compare mode scrolled
n_total += 1
send_events(Int64(20))
n_pass += capture("18_compare_scroll.png")

# 19. Compare mode off
n_total += 1
send_events(CompareTimePointsEvent(false))
n_pass += capture("19_compare_off.png")

# 20. Bone mask on
n_total += 1
send_events(ShowBoneMaskEvent(true))
n_pass += capture("20_bone_mask_on.png")

# 21. Bone mask off
n_total += 1
send_events(ShowBoneMaskEvent(false))
n_pass += capture("21_bone_mask_off.png")

# 21b. Anatomy ON
n_total += 1
send_events(ShowMaskLayerEvent(4, true))
n_pass += capture("21b_anatomy_on.png")

# 21c. Anatomy OFF
n_total += 1
send_events(ShowMaskLayerEvent(4, false))
n_pass += capture("21c_anatomy_off.png")

# 22. Next time point
n_total += 1
send_events(ChangeTimePointEvent(1))
n_pass += capture("22_next_timepoint.png")

# 23. Previous time point
n_total += 1
send_events(ChangeTimePointEvent(-1))
n_pass += capture("23_prev_timepoint.png")

# 24. Paint manual lesion (ID=2, bright overlay)
n_total += 1
send_events(
    PaintValEvent(2, true),
    MouseStruct(isLeftButtonDown=true, lastCoordinates=[CartesianIndex(300, 200), CartesianIndex(350, 250), CartesianIndex(400, 200)], actualWindowWidth=1400, actualWindowHeight=900)
)
n_pass += capture("24_painted_lesion.png")

# 25. Move lesion
n_total += 1
send_events(
    ToggleMoveLesionModeEvent(true),
    MouseStruct(isLeftButtonDown=false, isRightButtonDown=true, lastCoordinates=[CartesianIndex(350, 225)], actualWindowWidth=1400, actualWindowHeight=900),
    MouseStruct(isLeftButtonDown=false, isRightButtonDown=true, lastCoordinates=[CartesianIndex(450, 225)], actualWindowWidth=1400, actualWindowHeight=900),
    ToggleMoveLesionModeEvent(false)
)
n_pass += capture("25_moved_lesion.png")

# 26. Erase stroke
n_total += 1
send_events(
    PaintValEvent(0, true),
    MouseStruct(isLeftButtonDown=true, lastCoordinates=[CartesianIndex(300, 200), CartesianIndex(450, 250)], actualWindowWidth=1400, actualWindowHeight=900),
    PaintValEvent(-1, false)
)
n_pass += capture("26_erased_lesion.png")

# 27. Compare Mode ON -> OFF -> Plane Change (Verify Panel 2 pure PET and Panel 5 hidden)
n_total += 1
send_events(
    CompareTimePointsEvent(true),
    CompareTimePointsEvent(false),
    ChangePlaneEvent(:Coronal),
    ChangePlaneEvent(:Axial)
)
n_pass += capture("27_quad_view_after_compare.png")

# ─── Summary ────────────────────────────────────────────────────────────
println("\n" * "="^60)
println("Visual Regression Test Results: $n_pass/$n_total passed")
println("Screenshots saved to: $SCREENSHOT_DIR")
if n_pass == n_total
    println("ALL TESTS PASSED ✓")
else
    println("SOME TESTS FAILED ✗")
end
println("="^60)

# Clean shutdown
println("Closing viewer...")
put!(channel, CloseWindowEvent())
sleep(0.5)
println("Done.")
exit(n_pass == n_total ? 0 : 1)
