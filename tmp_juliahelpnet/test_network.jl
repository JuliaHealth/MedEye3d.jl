using Pkg; Pkg.activate(".")
include("network.jl")
using Random

rng = Xoshiro(0)
model = HELPNet_Lesion(3, 2)
ps, st = Lux.setup(rng, model)

# Julia array format: W, H, D, C, N
x = randn(Float32, 64, 64, 64, 3, 2)
y, st_ = model(x, ps, st)

println("Input size: ", size(x))
println("Output size: ", size(y))
