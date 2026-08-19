include("src/display/InferenceClient.jl")
using .InferenceClient
vol = zeros(Float32, 10, 10, 10)
patch = ones(Float32, 2, 2, 2)
InferenceClient.insert_patch!(vol, patch, 5, 5, 5)
