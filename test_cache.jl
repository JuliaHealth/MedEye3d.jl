using HDF5
h5_path = "/mnt/big/project_ssd/project_ssd/MedEye3d.jl/data/pat_6_files/Bone_Subsegments_0.h5"
bone_h5 = h5open(h5_path, "r")
a = read(bone_h5["PET_Lesions_0_lesion_24_surf"])
b = read(bone_h5["PET_Lesions_1_lesion_24_surf"])
println("a length: ", length(a))
println("b length: ", length(b))
println("Is exactly equal? ", a == b)
