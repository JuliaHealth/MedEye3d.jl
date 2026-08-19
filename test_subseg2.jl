using MedEye3d
using NIfTI
using MedEye3d.BoneSubsegmentation

ct_vol = Float32.(niread("data/pat_6_files/Fixed_CT_Volume_0.nii.gz").raw)
ct_vol = reverse(ct_vol, dims=2)

seg_vol = Float32.(niread("data/pat_6_files/Segmentation_0.nii.gz").raw) # Wait, this doesn't exist
# Just create a fake lesion mask on the 512 grid
seg_vol = zeros(Float32, size(ct_vol))
seg_vol[128:138, 85:95, 65:75] .= 12.0f0

bone_vol = Float32.(niread("data/pat_6_files/Skellytour_0.nii.gz").raw)
bone_vol = reverse(bone_vol, dims=2)

println("Running generate_bone_subsegments on fake Lesion 12...")
surf, marr = BoneSubsegmentation.generate_bone_subsegments(seg_vol, bone_vol, (1.5, 1.5, 2.0), 12)

println("Surf voxels: ", sum(surf))
println("Marr voxels: ", sum(marr))
