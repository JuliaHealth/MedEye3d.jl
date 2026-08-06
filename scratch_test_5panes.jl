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

data_dir = joinpath(@__DIR__, "..", "data", "pat6_pet_only_debug", "pat6_pet_only_debug")
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
pet_1 = MedImages.resample_to_image(ct_1, pet_1_raw, MedImages.Linear_en)
seg_1 = MedImages.resample_to_image(ct_1, seg_1_raw, MedImages.Nearest_neighbour_en)
pet_2 = MedImages.resample_to_image(ct_2, pet_2_raw, MedImages.Linear_en)
seg_2 = MedImages.resample_to_image(ct_2, seg_2_raw, MedImages.Nearest_neighbour_en)

colors_mapped = map(c -> RGB(c[1]/255, c[2]/255, c[3]/255), MedEye3d.distinctColorsSaved.listOfColors)

textureSpec_ct = TextureSpec{Float32}(name="CT", isMainImage=true, color=RGB(1.0, 1.0, 1.0), minAndMaxValue=Float32.([-150, 250]))
textureSpec_pet = TextureSpec{Float32}(name="PET", isMainImage=false, color=RGB(1.0, 0.5, 0.0), minAndMaxValue=Float32.([0, 10]))
textureSpec_mask = TextureSpec{Float32}(name="Mask", isMainImage=false, isMultiDiscreteMask=true, colorSet=colors_mapped, minAndMaxValue=Float32.([0, length(colors_mapped)]))

ct_vol_1 = Float32.(ct_1.voxel_data); pet_vol_1 = Float32.(pet_1.voxel_data); mask_vol_1 = Float32.(seg_1.voxel_data)
ct_vol_2 = Float32.(ct_2.voxel_data); pet_vol_2 = Float32.(pet_2.voxel_data); mask_vol_2 = Float32.(seg_2.voxel_data)

spacing_1 = Tuple(Float64.(ct_1.spacing))
spacing_2 = Tuple(Float64.(ct_2.spacing))

# Apply correct orientations
vol_img_axial_1 = reverse(ct_vol_1, dims=2); vol_pet_axial_1 = reverse(pet_vol_1, dims=2); vol_mask_axial_1 = reverse(mask_vol_1, dims=2)
vol_img_axial_2 = reverse(ct_vol_2, dims=2); vol_pet_axial_2 = reverse(pet_vol_2, dims=2); vol_mask_axial_2 = reverse(mask_vol_2, dims=2)

vol_img_coronal_1 = permutedims(ct_vol_1, (1, 3, 2)); vol_pet_coronal_1 = permutedims(pet_vol_1, (1, 3, 2)); vol_mask_coronal_1 = permutedims(mask_vol_1, (1, 3, 2))
vol_img_sagittal_1 = permutedims(ct_vol_1, (2, 3, 1)); vol_pet_sagittal_1 = permutedims(pet_vol_1, (2, 3, 1)); vol_mask_sagittal_1 = permutedims(mask_vol_1, (2, 3, 1))

textureSpec_pure_pet = TextureSpec{Float32}(name="PET", isMainImage=true, color=RGB(1.0, 0.5, 0.0), minAndMaxValue=Float32.([0, 10]))

voxelDataTupleVector = Vector{Vector{Any}}([
    Any[("CT", vol_img_axial_1), ("PET", vol_pet_axial_1), ("Mask", vol_mask_axial_1)],         # Top Left: Transverse TP1
    Any[("PET", vol_pet_axial_1)],                                                              # Top Right: Pure PET Transverse TP1
    Any[("CT", vol_img_sagittal_1), ("PET", vol_pet_sagittal_1), ("Mask", vol_mask_sagittal_1)],# Bottom Left: Sagittal TP1
    Any[("CT", vol_img_coronal_1), ("PET", vol_pet_coronal_1), ("Mask", vol_mask_coronal_1)],   # Bottom Right: Coronal TP1
    Any[("CT", vol_img_axial_2), ("PET", vol_pet_axial_2), ("Mask", vol_mask_axial_2)]          # Hidden initially: Transverse TP2
])

textureSpecArray = Vector{Vector{TextureSpec}}([
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask)],
    TextureSpec[deepcopy(textureSpec_pure_pet)],
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask)],
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask)],
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask)]
])

spacings = [[spacing_1], [spacing_1], [(spacing_1[2], spacing_1[3], spacing_1[1])], [(spacing_1[1], spacing_1[3], spacing_1[2])], [spacing_2]]
origins = [[(0.0, 0.0, 0.0)] for _ in 1:5]
