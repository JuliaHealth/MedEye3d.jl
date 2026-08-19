using Pkg
Pkg.activate("/mnt/big/project_ssd/project_ssd/MedEye3d.jl")

using MedEye3d.InferenceClient
using MedImages

InferenceClient.start_python_worker("/mnt/big/project_ssd/project_ssd/MedEye3d.jl/scripts/python_worker.py")
sleep(5) # wait for models to load

# Create dummy volumes
ct_vol = fill(-1000.0f0, 100, 100, 100)
pet_vol = fill(0.0f0, 100, 100, 100)
ct_vol[40:60, 40:60, 40:60] .= 40.0f0
pet_vol[48:52, 48:52, 48:52] .= 12.0f0

println("Running inference...")
mask = InferenceClient.run_helpnet_inference(ct_vol, pet_vol, 50, 50, 50)
if mask !== nothing
    println("Success! Mask size: ", size(mask))
    println("Mask sum: ", sum(mask))
else
    println("Failed.")
end
