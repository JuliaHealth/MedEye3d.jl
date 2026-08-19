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
# We need to upsample z by factor of ~5.12 (5.0 / 0.977)
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
# Pad to 512x512x512 so all panels have uniform square textures
function pad_to_512(arr::Array{Float32,3}, pad_val::Float32)
    padded = fill(pad_val, 512, 512, 512)
    sx, sy, sz = size(arr)
    padded[1:min(sx,512), 1:min(sy,512), 1:min(sz,512)] = arr[1:min(sx,512), 1:min(sy,512), 1:min(sz,512)]
    return padded
end

vol_img_axial = pad_to_512(vol_img_iso, -1000.0f0)
vol_mask_axial = pad_to_512(vol_mask_iso, 0.0f0)
spacing_axial = iso_spacing

# Coronal - y becomes 3rd dim
vol_img_coronal = pad_to_512(permutedims(vol_img_iso, (1, 3, 2)), -1000.0f0)
vol_mask_coronal = pad_to_512(permutedims(vol_mask_iso, (1, 3, 2)), 0.0f0)
spacing_coronal = iso_spacing  # all isotropic now

# Sagittal - x becomes 3rd dim
vol_img_sagittal = pad_to_512(permutedims(vol_img_iso, (2, 3, 1)), -1000.0f0)
vol_mask_sagittal = pad_to_512(permutedims(vol_mask_iso, (2, 3, 1)), 0.0f0)
spacing_sagittal = iso_spacing  # all isotropic now

println("Axial size: $(size(vol_img_axial))")
println("Coronal size: $(size(vol_img_coronal))")
println("Sagittal size: $(size(vol_img_sagittal))")


# Verify data at center slices
ax_slice = vol_img_axial[:, :, z_center]
cor_slice = vol_img_coronal[:, :, y_center]
sag_slice = vol_img_sagittal[:, :, x_center]
println("Axial slice z=$z_center: min=$(minimum(ax_slice)), max=$(maximum(ax_slice))")
println("Coronal slice y=$y_center: min=$(minimum(cor_slice)), max=$(maximum(cor_slice))")
println("Sagittal slice x=$x_center: min=$(minimum(sag_slice)), max=$(maximum(sag_slice))")

# Create vectors of vectors for QuadImage (4 panels)
# Panel 1 (Top Left): Axial CT + label overlay
# Panel 2 (Top Right): Axial label only
# Panel 3 (Bottom Left): Sagittal CT + label overlay
# Panel 4 (Bottom Right): Coronal CT + label overlay

# For panel 2 (label only), show the mask as the main image
textureSpec_mask_as_main = TextureSpec{Float32}(
    name="MainImage",
    isMainImage=true,
    color=RGB(1.0, 0.0, 0.0),
    minAndMaxValue=Float32.([0, 1])
)

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
    # Send scroll event through the channel so the consumer processes it
    # (direct reactToScroll calls bypass the consumer's render loop)
    put!(mainMedEye3dInstance.channel, Int64(diff))
    sleep(0.5) # Give consumer time to process
    # Send a zero scroll to ensure double-buffer rendering
    put!(mainMedEye3dInstance.channel, Int64(0))
    sleep(0.5)
end


# Debug: dump vertex and texture info for each panel state
for (i, state) in enumerate(mainMedEye3dInstance.states)
    verts = state.calcDimsStruct.mainImageQuadVert
    tw = state.calcDimsStruct.imageTextureWidth
    th = state.calcDimsStruct.imageTextureHeight
    hwr = state.calcDimsStruct.heightToWithRatio
    println("Panel $i: texW=$tw texH=$th h2wRatio=$hwr slice=$(state.currentDisplayedSlice)")
    println("  TopRight:    x=$(verts[1]) y=$(verts[2])")
    println("  BottomRight: x=$(verts[9]) y=$(verts[10])")
    println("  BottomLeft:  x=$(verts[17]) y=$(verts[18])")
    println("  TopLeft:     x=$(verts[25]) y=$(verts[26])")
end

println("Taking screenshot...")
# Capture root window and crop to the GLFW window size (1200x1200)
run(`import -window root /tmp/full_screenshot.png`)
run(`convert /tmp/full_screenshot.png -crop 1200x1200+0+0 +repage /workspaces/MedEye3d.jl/data/medical_image_screenshot_000.png`)
println("Screenshot saved to /workspaces/MedEye3d.jl/data/medical_image_screenshot_000.png")
sleep(2)
println("Done.")
