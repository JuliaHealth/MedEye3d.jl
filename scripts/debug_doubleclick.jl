using Pkg
Pkg.activate(".")

using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using MedImages
using ColorTypes
using MedEye3d.StructsManag
using MedEye3d.SegmentationDisplay
using MedEye3d.ReactOnMouseClickAndDrag
using MedEye3d.ForDisplayStructs: DoubleClickEvent
using Statistics

println("=== DOUBLE-CLICK ZOOM DIAGNOSTIC ===")

# Load data
img_path  = "/root/data_decathlon/Task09_Spleen/imagesTr/spleen_10.nii.gz"
mask_path = "/root/data_decathlon/Task09_Spleen/labelsTr/spleen_10.nii.gz"

println("Loading NIfTI image: ", img_path)
med_img = MedImages.Load_and_save.load_image(img_path, "")
vol_img = Float32.(med_img.voxel_data)
spacing = Tuple(Float64.(med_img.spacing))

println("Loading NIfTI mask: ", mask_path)
med_mask = MedImages.Load_and_save.load_image(mask_path, "")
vol_mask = Float32.(med_mask.voxel_data)

function resample_z_nearest(arr::Array{Float32,3}, old_z::Float64, new_z::Float64)
    old_nz = size(arr, 3)
    new_nz = round(Int, old_nz * old_z / new_z)
    new_arr = zeros(Float32, size(arr, 1), size(arr, 2), new_nz)
    for z in 1:new_nz
        orig_z = clamp(round(Int, (z - 0.5) * new_z / old_z + 0.5), 1, old_nz)
        new_arr[:, :, z] = arr[:, :, orig_z]
    end
    return new_arr
end

target_z = spacing[1]
vol_img_iso  = resample_z_nearest(vol_img,  spacing[3], target_z)
vol_mask_iso = resample_z_nearest(vol_mask, spacing[3], target_z)

spleen_indices = findall(x -> x > 0, vol_mask_iso)
x_center = isempty(spleen_indices) ? size(vol_img_iso,1)÷2 : round(Int, mean([i[1] for i in spleen_indices]))
y_center = isempty(spleen_indices) ? size(vol_img_iso,2)÷2 : round(Int, mean([i[2] for i in spleen_indices]))
z_center = isempty(spleen_indices) ? size(vol_img_iso,3)÷2 : round(Int, mean([i[3] for i in spleen_indices]))

textureSpec_img  = TextureSpec{Float32}(name="MainImage", isMainImage=true,  color=RGB(1.0,1.0,1.0), minAndMaxValue=Float32.([-100,200]))
textureSpec_mask = TextureSpec{Float32}(name="Mask",      isMainImage=false, color=RGB(1.0,0.0,0.0), minAndMaxValue=Float32.([0,1]), maskContribution=0.5f0)
textureSpec_mask_as_main = TextureSpec{Float32}(name="MainImage", isMainImage=true, color=RGB(1.0,0.0,0.0), minAndMaxValue=Float32.([0,1]))

vol_img_sag  = permutedims(vol_img_iso,  (2,3,1));  vol_mask_sag = permutedims(vol_mask_iso, (2,3,1))
vol_img_cor  = permutedims(vol_img_iso,  (1,3,2));  vol_mask_cor = permutedims(vol_mask_iso, (1,3,2))

voxelDataTupleVector = Vector{Vector{Any}}([
    Any[("MainImage", vol_img_iso),  ("Mask", vol_mask_iso)],
    Any[("MainImage", vol_mask_iso), ("Mask", vol_mask_iso)],
    Any[("MainImage", vol_img_sag),  ("Mask", vol_mask_sag)],
    Any[("MainImage", vol_img_cor),  ("Mask", vol_mask_cor)]
])
textureSpecArray = Vector{Vector{TextureSpec}}([
    TextureSpec[deepcopy(textureSpec_img),          deepcopy(textureSpec_mask)],
    TextureSpec[deepcopy(textureSpec_mask_as_main), deepcopy(textureSpec_mask)],
    TextureSpec[deepcopy(textureSpec_img),          deepcopy(textureSpec_mask)],
    TextureSpec[deepcopy(textureSpec_img),          deepcopy(textureSpec_mask)]
])
spacings = [[(spacing[1],spacing[2],target_z)] for _ in 1:4]
origins  = [[(0.0,0.0,0.0)] for _ in 1:4]
svVertAndInd = Dict{String, Vector}()
dummyStudySrc = Vector{Vector{Tuple{String,String}}}()

println("Starting quad display...")
mainMedEye3dInstance = SegmentationDisplay.displayImage(
    dummyStudySrc;
    textureSpecArray=textureSpecArray,
    voxelDataTupleVector=voxelDataTupleVector,
    spacings=spacings,
    origins=origins,
    fractionOfMainImage=Float32(1.0),
    windowWidth=1200,
    svVertAndInd=svVertAndInd,
    quadView=true
)

println("Waiting for states to be ready...")
while length(mainMedEye3dInstance.states) < 4
    sleep(0.5)
end
sleep(4)

# ─── TEST 1: Inject double-click to zoom panel 1 ─────────────────────────────
println("\n=== TEST 1: Inject double-click (panel 1 - top-left) ===")
println("quadZoomState BEFORE: isZoomed=$(MedEye3d.ReactOnMouseClickAndDrag.quadZoomState.isZoomed)")

fakeClick = DoubleClickEvent(
    x = 300,
    y = 250,
    actualWindowWidth  = 1200,
    actualWindowHeight = 1011,
)

println("Injecting double-click into channel...")
put!(mainMedEye3dInstance.channel, fakeClick)
sleep(1)

println("quadZoomState AFTER: isZoomed=$(MedEye3d.ReactOnMouseClickAndDrag.quadZoomState.isZoomed), panel=$(MedEye3d.ReactOnMouseClickAndDrag.quadZoomState.zoomedPanel)")
if MedEye3d.ReactOnMouseClickAndDrag.quadZoomState.isZoomed
    println("PASS: Double-click zoom fired! Panel $(MedEye3d.ReactOnMouseClickAndDrag.quadZoomState.zoomedPanel) is zoomed.")
else
    println("FAIL: isZoomed is still false - zoom did not trigger.")
end

# ─── TEST 2: Second double-click to restore ───────────────────────────────────
sleep(1)
println("\n=== TEST 2: Second double-click (zoom out) ===")
fakeClick2 = DoubleClickEvent(
    x = 300,
    y = 250,
    actualWindowWidth  = 1200,
    actualWindowHeight = 1011,
)
put!(mainMedEye3dInstance.channel, fakeClick2)
sleep(1)

if !MedEye3d.ReactOnMouseClickAndDrag.quadZoomState.isZoomed
    println("PASS: Zoom-out works! 4-pane restored.")
else
    println("FAIL: Still zoomed after second double-click.")
end

println("\n=== Diagnostic done. Ctrl+C to exit. ===")
wait()
