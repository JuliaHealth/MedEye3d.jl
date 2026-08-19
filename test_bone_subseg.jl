using NIfTI
using MedImages

include("src/preprocessing/BoneSubsegmentation.jl")

ct_nii = niread("data/pat_6_files/Fixed_CT_Volume_0.nii.gz")
ct = Float32.(ct_nii.raw)
pet = Float32.(niread("data/pat_6_files/PET_Lesions_0.nii.gz").raw)

# Note: PET_Lesions is 200x200x326, CT is 512x512x326. We must resample PET to CT space
ct_med = load_image("data/pat_6_files/Fixed_CT_Volume_0.nii.gz", "CT")
pet_med = load_image("data/pat_6_files/PET_Lesions_0.nii.gz", "CT") # fake modality
pet_res = resample_to_image(ct_med, pet_med, Nearest_neighbour_en)
pet_vol = Float32.(pet_res.voxel_data)

spacing = Tuple(Float64.(ct_med.spacing))
println("Spacing: ", spacing)

bone_vol = Float32.(ct .>= 180.0f0)

println("Running generate_bone_subsegments for lesion 11...")
s, m = Main.BoneSubsegmentation.generate_bone_subsegments(pet_vol, bone_vol, spacing, 11)

println("Surface voxels: ", sum(s))
println("Marrow voxels: ", sum(m))
