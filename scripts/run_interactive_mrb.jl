using Pkg
Pkg.activate(".")
Pkg.instantiate()

using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using MedImages
using ColorTypes
using MedEye3d.StructsManag
using MedEye3d.SegmentationDisplay
using MedEye3d.MakieEvents
using MedEye3d.LesionMetadataWindow
using Statistics
import GLFW

# Paths to data
data_dir = joinpath(@__DIR__, "..", "data", "pat6_pet_only_debug", "pat6_pet_only_debug")

println("Loading NIfTI data natively...")
nifti_dir = joinpath(@__DIR__, "..", "data", "pat6_pet_only_debug", "nifti_extracted")

println("Extracting TP1 data...")
ct_1 = MedImages.load_image(joinpath(nifti_dir, "Fixed_CT_Volume_0.nii.gz"), "CT")
pet_1_raw = MedImages.load_image(joinpath(nifti_dir, "SUV_PET_Image_0.nii.gz"), "PET")
seg_1_raw = MedImages.load_image(joinpath(nifti_dir, "NNUNET_OR_Lesions_CT_0_artficials_added.nii.gz"), "CT")

println("Extracting TP2 data...")
ct_2 = MedImages.load_image(joinpath(nifti_dir, "Fixed_CT_Volume_1.nii.gz"), "CT")
pet_2_raw = MedImages.load_image(joinpath(nifti_dir, "SUV_PET_Image_1.nii.gz"), "PET")
seg_2_raw = MedImages.load_image(joinpath(nifti_dir, "NNUNET_OR_Lesions_CT_1_artficials_added.nii.gz"), "CT")

println("Resampling PET and Mask to CT geometry...")
# Resample to common grid
pet_1 = MedImages.resample_to_image(ct_1, pet_1_raw, MedImages.Linear_en)
seg_1 = MedImages.resample_to_image(ct_1, seg_1_raw, MedImages.Nearest_neighbour_en)

pet_2 = MedImages.resample_to_image(ct_2, pet_2_raw, MedImages.Linear_en)
seg_2 = MedImages.resample_to_image(ct_2, seg_2_raw, MedImages.Nearest_neighbour_en)

# Map the distinct colors
colors_mapped = map(c -> RGB(c[1]/255, c[2]/255, c[3]/255), MedEye3d.distinctColorsSaved.listOfColors)

# Setup MedEye3d Display Textures
textureSpec_ct = TextureSpec{Float32}(name="CT", isMainImage=true, color=RGB(1.0, 1.0, 1.0), minAndMaxValue=Float32.([-150, 250]))
textureSpec_pet = TextureSpec{Float32}(name="PET", isMainImage=false, color=RGB(1.0, 0.5, 0.0), minAndMaxValue=Float32.([0, 10]))
textureSpec_mask = TextureSpec{Float32}(
    name="Mask", 
    isMainImage=false, 
    isMultiDiscreteMask=true,
    colorSet=colors_mapped,
    minAndMaxValue=Float32.([0, length(colors_mapped)])
)

ct_vol = Float32.(ct_1.voxel_data)
pet_vol = Float32.(pet_1.voxel_data)
mask_vol = Float32.(seg_1.voxel_data)
spacing = Tuple(Float64.(ct_1.spacing))

ct_vol_2 = Float32.(ct_2.voxel_data)
pet_vol_2 = Float32.(pet_2.voxel_data)
mask_vol_2 = Float32.(seg_2.voxel_data)
spacing_2 = Tuple(Float64.(ct_2.spacing))

# Apply correct orientations
vol_img_axial = reverse(ct_vol, dims=2)
vol_pet_axial = reverse(pet_vol, dims=2)
vol_mask_axial = reverse(mask_vol, dims=2)

vol_img_coronal = permutedims(ct_vol, (1, 3, 2))
vol_pet_coronal = permutedims(pet_vol, (1, 3, 2))
vol_mask_coronal = permutedims(mask_vol, (1, 3, 2))

vol_img_sagittal = permutedims(ct_vol, (2, 3, 1))
vol_pet_sagittal = permutedims(pet_vol, (2, 3, 1))
vol_mask_sagittal = permutedims(mask_vol, (2, 3, 1))

vol_img_axial_2 = reverse(ct_vol_2, dims=2)
vol_pet_axial_2 = reverse(pet_vol_2, dims=2)
vol_mask_axial_2 = reverse(mask_vol_2, dims=2)

textureSpec_pure_pet = TextureSpec{Float32}(name="PET", isMainImage=true, color=RGB(1.0, 0.5, 0.0), minAndMaxValue=Float32.([0, 10]))

voxelDataTupleVector = Vector{Vector{Any}}([
    Any[("CT", vol_img_axial), ("PET", vol_pet_axial), ("Mask", vol_mask_axial)],         # Top Left: Transverse
    Any[("PET", vol_pet_axial)],                                                          # Top Right: Pure PET Transverse
    Any[("CT", vol_img_sagittal), ("PET", vol_pet_sagittal), ("Mask", vol_mask_sagittal)],# Bottom Left: Sagittal
    Any[("CT", vol_img_coronal), ("PET", vol_pet_coronal), ("Mask", vol_mask_coronal)],   # Bottom Right: Coronal
    Any[("CT", vol_img_axial_2), ("PET", vol_pet_axial_2), ("Mask", vol_mask_axial_2)]    # Hidden: TP2 Transverse
])

textureSpecArray = Vector{Vector{TextureSpec}}([
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask)],
    TextureSpec[deepcopy(textureSpec_pure_pet)],
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask)],
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask)],
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask)]
])

spacings = [[spacing], [spacing], [(spacing[2], spacing[3], spacing[1])], [(spacing[1], spacing[3], spacing[2])], [spacing_2]]
origins = [[(0.0, 0.0, 0.0)] for _ in 1:5]

dummyStudySrc = Vector{Vector{Tuple{String,String}}}()

println("Launching MedEye3d Viewer...")
mainMedEye3dInstance = SegmentationDisplay.displayImage(
    dummyStudySrc;
    textureSpecArray=textureSpecArray,
    voxelDataTupleVector=voxelDataTupleVector,
    spacings=spacings,
    origins=origins,
    windowWidth=1200,
    fractionOfMainImage=Float32(1.0),
    quadView=true
)

# Initialize Quad View and hide Pane 5
put!(mainMedEye3dInstance.channel, CompareTimePointsEvent(false))

# Start Makie Control Window
println("Starting Makie Control Window...")
import Observables
using MedEye3d.LesionMetadataWindow

active_lesion = Observables.Observable("Lesion 1")
unique_vals = sort(unique(mask_vol))
lesion_ids_ints = filter(x -> x > 0, unique_vals)
lesion_list = isempty(lesion_ids_ints) ? ["Lesion 1"] : ["Lesion $(Int(i))" for i in lesion_ids_ints]
lesion_ids = Observables.Observable(lesion_list)

# Launch the Slicer Extension native port GUI
win = LesionMetadataWindow.create_metadata_window(active_lesion, lesion_ids, mainMedEye3dInstance.channel)
screen = display(win.fig)

# Run GLFW interaction loop manually
println("Interactive session ready!")
println("Press ENTER in this terminal to exit...")
readline()

println("Closing viewer...")
# Clean exit - GLFW will be terminated when Julia process exits
