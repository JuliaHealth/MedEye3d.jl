include("src/preprocessing/BoneSubsegmentation.jl")
using .BoneSubsegmentation
using NIfTI

ct = niread("data/pat_6_files/Fixed_CT_Volume_0.nii.gz").raw
skelly = niread("data/pat_6_files/Skellytour_0.nii.gz").raw
seg = niread("data/pat_6_files/PET_Lesions_0.nii.gz").raw

mask = BoneSubsegmentation.generate_bone_subsegments(Float32.(seg), Float32.(skelly), (1.5, 1.5, 2.0), 7)
println("Mask uniques surf: ", unique(mask[1]))
println("Mask sum surf: ", sum(mask[1]))
println("Mask uniques marr: ", unique(mask[2]))
println("Mask sum marr: ", sum(mask[2]))
