using Lux, CUDA, LuxCUDA, ComponentArrays, JLD2, NNlib, Random
include("heat_diffuse.jl")
include("model.jl")
include("dataset.jl")
include("architectures/hed3d.jl")

dev = Lux.gpu_device()
ckpt = load("champion_model.jld2")
ps_edge = ckpt["ps"].edge_net
st_edge = ckpt["st"].edge_net

edge_net = HED3D(3)
rng = Xoshiro(42)
ps_init, _ = Lux.setup(rng, edge_net)
ps_init_ca = ComponentArray(ps_init)
ps_init_ca .= ps_edge

ps_edge_gpu = ps_init_ca |> dev
st_edge_gpu = st_edge |> dev

st_test = Lux.testmode(st_edge_gpu)

x = CUDA.rand(Float32, 64, 64, 64, 3, 1)
try
    edge_map, _ = edge_net(x, ps_edge_gpu, st_test)
    println("SUCCESS Lux.setup copy")
catch e
    println("ERROR Lux.setup copy: ", e)
end
