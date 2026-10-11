using Lux, CUDA, LuxCUDA, ComponentArrays, JLD2, NNlib
include("heat_diffuse.jl")
include("model.jl")
include("dataset.jl")
include("architectures/hed3d.jl")

dev = Lux.gpu_device()
ckpt = load("champion_model.jld2")
ps_edge = ckpt["ps"].edge_net
st_edge = ckpt["st"].edge_net

# Let's try deepcopy and copy to a new array
ps_edge = ComponentArray(ps_edge)
ps_edge = deepcopy(ps_edge) |> dev
st_edge = st_edge |> dev

edge_net = HED3D(3)
st_test = Lux.testmode(st_edge)

x = CUDA.rand(Float32, 64, 64, 64, 3, 1)
try
    edge_map, _ = edge_net(x, ps_edge, st_test)
    println("SUCCESS deepcopy")
catch e
    println("ERROR deepcopy: ", e)
end
