using MedEye3d.InferenceClient

ct_vol = zeros(Float32, 256, 256, 256)
pet_vol = zeros(Float32, 256, 256, 256)
points_vol = zeros(Float32, 256, 256, 256)

# Paint a huge scribble from (50,50,50) to (150,150,150) -> max_dist = 100
points_vol[50:150, 50:150, 50:150] .= 1.0f0

println("Running run_nninteractive with large scribble...")
mask = InferenceClient.run_nninteractive(ct_vol, pet_vol, points_vol, 100, 100, 100)
if mask === nothing
    println("FAILED")
else
    println("SUCCESS, size: ", size(mask), ", nonzeros: ", count(mask .> 0))
end
