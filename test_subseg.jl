using MedEye3d
using NIfTI
using MedEye3d.BoneSubsegmentation

# Let's load the data
ct_vol = Float32.(niread("data/pat_6_files/Fixed_CT_Volume_0.nii.gz").raw)
ct_vol = reverse(ct_vol, dims=2)

seg_vol = Float32.(niread("data/pat_6_files/PET_Lesions_0.nii.gz").raw)
seg_vol = reverse(seg_vol, dims=2)

bone_vol = Float32.(niread("data/pat_6_files/Skellytour_0.nii.gz").raw)
bone_vol = reverse(bone_vol, dims=2)

println("Running generate_bone_subsegments on Lesion 12...")
surf, marr = BoneSubsegmentation.generate_bone_subsegments(seg_vol, bone_vol, (1.5, 1.5, 2.0), 12)

println("Surf voxels: ", sum(surf))
println("Marr voxels: ", sum(marr))
