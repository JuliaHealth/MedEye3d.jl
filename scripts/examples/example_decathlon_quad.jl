using Pkg
Pkg.activate(".")
Pkg.instantiate()
using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using GLMakie
using MedImages
using ColorTypes
using MedEye3d.StructsManag
using MedEye3d.SegmentationDisplay
using ModernGL

# Paths to the Decathlon Task09 dataset within the dev container
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
println("Max value in img: $(maximum(vol_img)), mask: $(maximum(vol_mask))")

# Setup MedEye3d Display
textureSpec_img = TextureSpec{Float32}(
    name="MainImage",
    isMainImage=true,
    color=RGB(1.0, 1.0, 1.0),
    minAndMaxValue=Float32.([-100, 200]) # Radiological windowing for CT
)

textureSpec_mask = TextureSpec{Float32}(
    name="Mask",
    isMainImage=false,
    color=RGB(1.0, 0.0, 0.0),
    minAndMaxValue=Float32.([0, 1])
)

# Axial (Transverse) - natively z is 3rd dim
vol_img_axial = deepcopy(vol_img)
vol_mask_axial = deepcopy(vol_mask)
spacing_axial = spacing

# Coronal - y is 3rd dim
vol_img_coronal = permutedims(vol_img, (1, 3, 2))
vol_mask_coronal = permutedims(vol_mask, (1, 3, 2))
spacing_coronal = (spacing[1], spacing[3], spacing[2])

# Sagittal - x is 3rd dim
vol_img_sagittal = permutedims(vol_img, (2, 3, 1))
vol_mask_sagittal = permutedims(vol_mask, (2, 3, 1))
spacing_sagittal = (spacing[2], spacing[3], spacing[1])

# Create vectors of vectors for QuadImage (4 panels)
# Panel 1 (Top Left): Axial image + mask
# Panel 2 (Top Right): Axial pure mask (mask only)
# Panel 3 (Bottom Left): Coronal image + mask
# Panel 4 (Bottom Right): Sagittal image + mask
voxelDataTupleVector = Vector{Vector{Any}}([
    Any[("MainImage", vol_img_axial), ("Mask", vol_mask_axial)],
    Any[("Mask", vol_mask_axial)],
    Any[("MainImage", vol_img_coronal), ("Mask", vol_mask_coronal)],
    Any[("MainImage", vol_img_sagittal), ("Mask", vol_mask_sagittal)]
])

textureSpecArray = Vector{Vector{TextureSpec}}([
    TextureSpec[deepcopy(textureSpec_img), deepcopy(textureSpec_mask)],
    TextureSpec[deepcopy(textureSpec_mask)],
    TextureSpec[deepcopy(textureSpec_img), deepcopy(textureSpec_mask)],
    TextureSpec[deepcopy(textureSpec_img), deepcopy(textureSpec_mask)]
])

# Calculate spacings and origins for all 4 panels
spacings = [[spacing_axial], [spacing_axial], [spacing_coronal], [spacing_sagittal]]
origins = [[(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)]]

# Provide dummy data for svVertAndInd
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

println("QuadImage Display is running. Triggering screenshot...")

# Wait for the async consumer to initialize states and populate onScrollData
while length(mainMedEye3dInstance.states) < 4
    sleep(0.5)
end
sleep(2) # Give consumer time to process FullScrollableDat

# Scroll to center slice for the axial views (z-axis)
z_center = size(vol_img, 3) ÷ 2
y_center = size(vol_img, 2) ÷ 2
x_center = size(vol_img, 1) ÷ 2

# Scroll panels to their respective center slices
# We call reactToScroll twice for each panel to ensure both OpenGL double buffers receive the rendered quad.
# Otherwise, alternating buffer swaps cause half the panels to disappear.
for (i, center) in [(1, z_center), (2, z_center), (3, y_center), (4, x_center)]
    mainMedEye3dInstance.states[1].switchIndex = i
    curr = mainMedEye3dInstance.states[i].currentDisplayedSlice
    MedEye3d.ReactToScroll.reactToScroll(center - curr, mainMedEye3dInstance.states, false)
    MedEye3d.ReactToScroll.reactToScroll(0, mainMedEye3dInstance.states, false)
end
sleep(2)

println("Taking screenshot via import...")
try
    run(`import -window root example_quad_decathlon.png`)
catch e
    println("Failed to run import: $e")
end

sleep(2)
println("Done.")
