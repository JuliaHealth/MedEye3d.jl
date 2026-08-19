include("src/display/InferenceClient.jl")
using .InferenceClient
using NIfTI
ct = niread("data/pat_6_files/Fixed_CT_Volume_0.nii.gz").raw
pet = niread("data/pat_6_files/PET_Lesions_0.nii.gz").raw
points = zeros(Float32, size(ct)...)
points[200, 200, 200] = 1.0f0

println("Testing NNInteractive...")
mask2 = InferenceClient.run_nninteractive(Float32.(ct), Float32.(pet), points, 200, 200, 200)
println("NNInteractive returned: ", sum(mask2))
