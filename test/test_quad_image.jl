using Pkg
Pkg.activate(".")
Pkg.instantiate()
using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using GLMakie
using MedImages
using ColorTypes
using MedEye3d.DisplayWords
using MedEye3d.StructsManag
using MedEye3d.SegmentationDisplay
using MedEye3d.DataStructs: getThreeDims
using ModernGL

# Load NIfTI
med_img = MedImages.Load_and_save.load_image("data/synthetic_sphere.nii", "")
vol = Float32.(med_img.voxel_data)
spacing = Tuple(Float64.(med_img.spacing))
println("Loaded NIfTI with spacing: $spacing")
println("Max value in vol: $(maximum(vol)), sum: $(sum(vol))")

# Setup MedEye3d Display
textureSpec = TextureSpec{Float32}(
    name="Sphere",
    isMainImage=true,
    color=RGB(1.0, 1.0, 1.0),
    minAndMaxValue=Float32.([0, 1])
)

# Axial (Transverse) - natively z is 3rd dim
vol_axial = vol
spacing_axial = spacing

# Coronal - y is 3rd dim
vol_coronal = permutedims(vol, (1, 3, 2))
spacing_coronal = (spacing[1], spacing[3], spacing[2])

# Sagittal - x is 3rd dim
vol_sagittal = permutedims(vol, (2, 3, 1))
spacing_sagittal = (spacing[2], spacing[3], spacing[1])

# Create vectors of vectors for QuadImage (4 panels)
# Panel 1: Axial, Panel 2: Coronal, Panel 3: Sagittal, Panel 4: Axial (repeated)
voxelDataTupleVector = Vector{Vector{Any}}([
    Any[("Sphere", vol_axial)],
    Any[("Sphere", vol_coronal)],
    Any[("Sphere", vol_sagittal)],
    Any[("Sphere", vol_axial)]
])

textureSpecArray = Vector{Vector{TextureSpec}}([
    TextureSpec[textureSpec],
    TextureSpec[textureSpec],
    TextureSpec[textureSpec],
    TextureSpec[textureSpec]
])

# Calculate spacings and origins for all 4
spacings = [[spacing_axial], [spacing_coronal], [spacing_sagittal], [spacing_axial]]
origins = [[(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)]]

# Provide dummy data for svVertAndInd
svVertAndInd = Dict{String, Vector}()

println("Starting quad display...")
# Set fractionOfMainImage
fractionOfMainImage = Float32(0.8)

dummyStudySrc = Vector{Vector{Tuple{String,String}}}()

# Initialize UI
mainMedEye3dInstance = SegmentationDisplay.displayImage(
    dummyStudySrc;
    textureSpecArray=textureSpecArray,
    voxelDataTupleVector=voxelDataTupleVector,
    spacings=spacings,
    origins=origins,
    fractionOfMainImage=fractionOfMainImage,
    windowWidth=1000,
    svVertAndInd=svVertAndInd,
    quadView=true
)

println("QuadImage Display is running. Triggering screenshot...")

# Wait for the async consumer to initialize states and populate onScrollData
while length(mainMedEye3dInstance.states) < 4
    sleep(0.5)
end
sleep(2) # Give consumer time to process FullScrollableDat

println("Scrolling to center slice (64)...")
for i in 1:4
    mainMedEye3dInstance.states[1].switchIndex = i
    MedEye3d.ReactToScroll.reactToScroll(63, mainMedEye3dInstance.states, false)
end
sleep(2)

using ModernGL, GLFW

@info "Test script finished waiting."

for i in 1:4
    mainState = mainMedEye3dInstance.states[i]
    @info "Quadrant $i listOfTextSpecifications length: $(length(mainState.mainForDisplayObjects.listOfTextSpecifications))"
    for texSpec in mainState.mainForDisplayObjects.listOfTextSpecifications
        @info "Quadrant $i TextureSpec: name=$(texSpec.name), ID=$(texSpec.ID[])"
    end
    for (k, dts) in enumerate(mainState.onScrollData.dataToScroll)
        @info "dataToScroll[$k].name = '$(dts.name)'"
    end
end

println("Taking screenshot via import...")
try
    run(`import -window root screenshot_medeye_025.png`)
catch e
    println("Failed to run import: $e")
end

# Keep it alive for a few more seconds just in case
sleep(2)

println("Done.")
