include("src/display/InferenceClient.jl")
using .InferenceClient
using NIfTI
ct = niread("data/pat_6_files/Fixed_CT_Volume_0.nii.gz").raw
pet = niread("data/pat_6_files/PET_Lesions_0.nii.gz").raw
points = zeros(Float32, size(ct)...)
points[200, 200, 200] = 1.0f0

println("Testing HelpNet...")
mask = InferenceClient.run_helpnet_inference(Float32.(ct), Float32.(pet), points, 200, 200, 200)

seg_vol = zeros(Float32, size(ct)...)
InferenceClient.insert_patch!(seg_vol, mask, 200, 200, 200, label_val=11.0f0)
println("Seg vol sum: ", sum(seg_vol))
