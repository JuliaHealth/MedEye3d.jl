using Pkg
Pkg.activate(".")

using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using MedImages
using ColorTypes
using MedEye3d.StructsManag
using MedEye3d.SegmentationDisplay
using Statistics

# Paths to data
img_path = "/root/data_decathlon/Task09_Spleen/imagesTr/spleen_10.nii.gz"
mask_path = "/root/data_decathlon/Task09_Spleen/labelsTr/spleen_10.nii.gz"

println("Loading NIfTI image: ", img_path)
med_img = MedImages.Load_and_save.load_image(img_path, "")
vol_img = Float32.(med_img.voxel_data)
spacing = Tuple(Float64.(med_img.spacing))

println("Loading NIfTI mask: ", mask_path)
med_mask = MedImages.Load_and_save.load_image(mask_path, "")
vol_mask = Float32.(med_mask.voxel_data)

println("Loaded NIfTIs with spacing: $spacing")
println("Image size: $(size(vol_img)), Mask size: $(size(vol_mask))")

# Resample Z to make isotropic (z spacing is 5.0, xy spacing is ~0.977)
function resample_z_nearest(arr::Array{Float32,3}, old_spacing_z::Float64, new_spacing_z::Float64)
    old_nz = size(arr, 3)
    new_nz = round(Int, old_nz * old_spacing_z / new_spacing_z)
    new_arr = zeros(Float32, size(arr, 1), size(arr, 2), new_nz)
    for z in 1:new_nz
        orig_z = clamp(round(Int, (z - 0.5) * new_spacing_z / old_spacing_z + 0.5), 1, old_nz)
        new_arr[:, :, z] = arr[:, :, orig_z]
    end
    return new_arr
end

target_spacing_z = spacing[1]  # Make Z spacing same as X spacing
vol_img_iso = resample_z_nearest(vol_img, spacing[3], target_spacing_z)
vol_mask_iso = resample_z_nearest(vol_mask, spacing[3], target_spacing_z)
iso_spacing = (spacing[1], spacing[2], target_spacing_z)

println("After isotropic resampling: $(size(vol_img_iso)) with spacing $iso_spacing")

# Find spleen center from resampled label
spleen_indices = findall(x -> x > 0, vol_mask_iso)
if !isempty(spleen_indices)
    x_coords = [idx[1] for idx in spleen_indices]
    y_coords = [idx[2] for idx in spleen_indices]
    z_coords = [idx[3] for idx in spleen_indices]
    x_center = round(Int, mean(x_coords))
    y_center = round(Int, mean(y_coords))
    z_center = round(Int, mean(z_coords))
    println("Spleen center (isotropic): x=$x_center, y=$y_center, z=$z_center")
else
    x_center = size(vol_img_iso, 1) ÷ 2
    y_center = size(vol_img_iso, 2) ÷ 2
    z_center = size(vol_img_iso, 3) ÷ 2
    println("No spleen found, using volume center")
end

# Setup MedEye3d Display
textureSpec_img = TextureSpec{Float32}(
    name="MainImage",
    isMainImage=true,
    color=RGB(1.0, 1.0, 1.0),
    minAndMaxValue=Float32.([-100, 200])
)

textureSpec_mask = TextureSpec{Float32}(
    name="Mask",
    isMainImage=false,
    color=RGB(1.0, 0.0, 0.0),
    minAndMaxValue=Float32.([0, 1]),
    maskContribution=0.5f0
)

# Axial (Transverse) - z is 3rd dim (scroll dimension)
vol_img_axial = vol_img_iso
vol_mask_axial = vol_mask_iso
spacing_axial = iso_spacing

# Coronal - y becomes 3rd dim
vol_img_coronal = permutedims(vol_img_iso, (1, 3, 2))
vol_mask_coronal = permutedims(vol_mask_iso, (1, 3, 2))
spacing_coronal = iso_spacing

# Sagittal - x becomes 3rd dim
vol_img_sagittal = permutedims(vol_img_iso, (2, 3, 1))
vol_mask_sagittal = permutedims(vol_mask_iso, (2, 3, 1))
spacing_sagittal = iso_spacing

println("Axial size: $(size(vol_img_axial))")
println("Coronal size: $(size(vol_img_coronal))")
println("Sagittal size: $(size(vol_img_sagittal))")

# For panel 2 (label only), show the mask as the main image
textureSpec_mask_as_main = TextureSpec{Float32}(
    name="MainImage",
    isMainImage=true,
    color=RGB(1.0, 0.0, 0.0),
    minAndMaxValue=Float32.([0, 1])
)

# Create vectors of vectors for QuadImage (4 panels)
# Panel 1 (Top Left): Axial CT + label overlay
# Panel 2 (Top Right): Axial label only
# Panel 3 (Bottom Left): Sagittal CT + label overlay
# Panel 4 (Bottom Right): Coronal CT + label overlay
voxelDataTupleVector = Vector{Vector{Any}}([
    Any[("MainImage", vol_img_axial), ("Mask", vol_mask_axial)],
    Any[("MainImage", vol_mask_axial), ("Mask", vol_mask_axial)],
    Any[("MainImage", vol_img_sagittal), ("Mask", vol_mask_sagittal)],
    Any[("MainImage", vol_img_coronal), ("Mask", vol_mask_coronal)]
])

textureSpecArray = Vector{Vector{TextureSpec}}([
    TextureSpec[deepcopy(textureSpec_img), deepcopy(textureSpec_mask)],
    TextureSpec[deepcopy(textureSpec_mask_as_main), deepcopy(textureSpec_mask)],
    TextureSpec[deepcopy(textureSpec_img), deepcopy(textureSpec_mask)],
    TextureSpec[deepcopy(textureSpec_img), deepcopy(textureSpec_mask)]
])

spacings = [[spacing_axial], [spacing_axial], [spacing_sagittal], [spacing_coronal]]
origins = [[(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)]]

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

println("QuadImage Display running. Waiting for initialization...")

# Wait for the async consumer to initialize states
while length(mainMedEye3dInstance.states) < 4
    sleep(0.5)
end
sleep(3)

# Scroll panels to the spleen center slices via the channel
# Panel 1 (axial): scroll to z_center
# Panel 2 (axial label): scroll to z_center
# Panel 3 (sagittal): scroll to x_center (now 3rd dim after permute)
# Panel 4 (coronal): scroll to y_center (now 3rd dim after permute)
for (i, center) in [(1, z_center), (2, z_center), (3, x_center), (4, y_center)]
    mainMedEye3dInstance.states[1].switchIndex = i
    curr = mainMedEye3dInstance.states[i].currentDisplayedSlice
    diff = center - curr
    println("Panel $i: scrolling from $curr to $center (diff=$diff)")
    put!(mainMedEye3dInstance.channel, Int64(diff))
    sleep(0.5)
    put!(mainMedEye3dInstance.channel, Int64(0))
    sleep(0.5)
end

println("")
println("============================================")
println("  Interactive mode ready!")
println("============================================")
println("  Scroll: mouse wheel on any panel")
println("  Right-click: jump all planes to that point")
println("  Left-drag: paint on mask")
println("  Close window or Ctrl+C to exit")
println("============================================")
println("")

# Keep process alive for interactive use
# The GLFW event loop runs in a background task
wait()
