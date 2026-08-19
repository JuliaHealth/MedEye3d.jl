using Pkg
Pkg.activate(".")
Pkg.instantiate()
using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.SegmentationDisplay
using ColorTypes
using MedImages

# 1. Load the NIfTI from the data directory
data_dir = "/workspace/data"
nii_path = joinpath(data_dir, "synthetic_sphere.nii")
nii_img = MedImages.Load_and_save.load_image(nii_path, "")
vol_img = Float32.(nii_img.voxel_data)

# 2. Configure Texture Specs
textureSpec_img = TextureSpec{Float32}(
    name="MainImage",
    isMainImage=true,
    color=RGB(1.0, 1.0, 1.0),
    minAndMaxValue=Float32.([minimum(vol_img), maximum(vol_img)])
)

# 3. Setup Display
voxelDataTupleVector = Vector{Vector{Any}}([Any[("MainImage", vol_img)]])
textureSpecArray = Vector{Vector{TextureSpec}}([TextureSpec[textureSpec_img]])

mainMedEye3dInstance = SegmentationDisplay.displayImage(
    Vector{Vector{Tuple{String,String}}}();
    textureSpecArray=textureSpecArray,
    voxelDataTupleVector=voxelDataTupleVector,
    spacings=[[(1.0, 1.0, 1.0)]],
    origins=[[(0.0, 0.0, 0.0)]],
    fractionOfMainImage=Float32(0.8),
    windowWidth=1000,
    quadView=false
)

# Wait for window to load
while length(mainMedEye3dInstance.states) < 1
    sleep(0.5)
end
sleep(2)

# Scroll to center
z_center = size(vol_img, 3) ÷ 2
curr = mainMedEye3dInstance.states[1].currentDisplayedSlice
MedEye3d.ReactToScroll.reactToScroll(z_center - curr, mainMedEye3dInstance.states, false)
MedEye3d.ReactToScroll.reactToScroll(0, mainMedEye3dInstance.states, false)

sleep(2)

# 4. Take Screenshot using scrot
println("Taking screenshot...")
screenshot_path = joinpath(data_dir, "sphere_screenshot.png")
try
    # Use scrot to capture the focused window
    run(`scrot -u $screenshot_path`)
catch e
    println("Failed to run scrot: ", e)
end

println("Saved screenshot to: ", screenshot_path)
