using MedImages, LinearAlgebra
include("scripts/lib/SceneHierarchy.jl")
using .SceneHierarchy

println("Loading baseline CT...")
ct0 = MedImages.load_image("data/pat_6_files/SPECT_CT_Volume_0.nii.gz", "CT")

println("Loading TP1 CT...")
ct1 = MedImages.load_image("data/pat_6_files/SPECT_CT_Volume_1.nii.gz", "CT")

T_ITK = parse_tfm("data/pat_6_files/Transform_SPECT_to_Baseline_1.tfm")
println("Transform: ", T_ITK[1:3, 4])

# Try the original code (inv)
old_spacing = ct1.spacing
old_dir = transpose(reshape(collect(ct1.direction), 3, 3))
old_orig = ct1.origin

M_old = zeros(Float64, 4, 4)
for i in 1:3, j in 1:3; M_old[i, j] = old_dir[i, j] * old_spacing[j]; end
for i in 1:3; M_old[i, 4] = old_orig[i]; end
M_old[4, 4] = 1.0

M_new_inv = inv(T_ITK) * M_old
M_new_fwd = T_ITK * M_old

println("Baseline origin: ", ct0.origin)
println("TP1 origin before tfm: ", ct1.origin)
println("M_new_inv origin: ", M_new_inv[1:3, 4])
println("M_new_fwd origin: ", M_new_fwd[1:3, 4])
