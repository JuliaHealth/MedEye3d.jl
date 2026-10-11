using Pkg; Pkg.activate("/mnt/big/project_ssd/project_ssd/semiautomatic/JuliaHELPNet")
using LuxCUDA, Lux, NNlib, Random, ComponentArrays, JLD2, Printf, Statistics, CUDA, Zygote, Optimisers, LinearAlgebra
include("heat_diffuse.jl")
include("heat_diffuse_v2.jl")
include("model.jl")
include("dataset.jl")
using .HeatDiffusion, .HeatDiffusionV2, .HeatGDTModel, .HeatGDTDataset
include("architectures/hed3d.jl")

dev = Lux.gpu_device()
train_manifest = "/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/train_single_lesion_manifest.json"
train_ds = PatchHeatDataset(train_manifest)

struct HeatGDTFrameworkV2{M} <: Lux.AbstractLuxContainerLayer{(:edge_net,)}
    edge_net::M
    K::Int
    dt::Float32
    theta::Float32
    tau::Float32
    pde_type::Symbol
    edge_type::Symbol
end
function (m::HeatGDTFrameworkV2)((x, seed_mask), ps, st)
    net_out, st_edge = m.edge_net(x, ps.edge_net, st.edge_net)
    edge_map = net_out
    D = HeatGDTModel.compute_diffusivity(edge_map)
    u = HeatDiffusion.heat_diffusion(seed_mask, D, m.K, m.dt)
    pred_mask = HeatDiffusion.extract_mask(u, m.theta, m.tau)
    return (pred=pred_mask, heat=u, edge=edge_map, D=D), (edge_net=st_edge,)
end

include("losses.jl")

loss_fn = function(pred, heat, edge, gt_mask)
    dl = dice_loss(pred, gt_mask)
    ce = bce_loss(pred, gt_mask)
    sp = mean(edge)
    return dl + 0.5f0 * ce + 0.01f0 * sp, dl
end

model = HeatGDTFrameworkV2(HED3D(3), 40, 0.16f0, 0.001f0, 0.0001f0, :pure_diffusion, :hed3d)
rng = Xoshiro(42)
ps, st = Lux.setup(rng, model)
ps = ComponentArray(ps) |> dev
st = st |> dev

x_cpu, seed_cpu, gt_cpu = get_batch(train_ds, 1:2)
x_b = x_cpu |> dev
seed_b = seed_cpu |> dev
gt_b = gt_cpu |> dev

function compute_loss(p, s, xb, sb, gtb)
    out, st_new = model((xb, sb), p, s)
    total, dl = loss_fn(out.pred, out.heat, out.edge, gtb)
    return total, st_new, dl
end

println("Starting pullback...")
time_taken = @elapsed begin
    (loss_val, st_new, dl), back = Zygote.pullback(p -> compute_loss(p, st, x_b, seed_b, gt_b), ps)
end
println("Forward time: ", time_taken)

time_taken_back = @elapsed begin
    grads = back((one(loss_val), nothing, nothing))[1]
end
println("Backward time: ", time_taken_back)
println("Loss: ", loss_val, " | DL: ", dl)
println("Gradient L1 norm: ", sum(abs, grads))

out, _ = model((x_b, seed_b), ps, st)
println("Pred min/max: ", minimum(out.pred), " / ", maximum(out.pred))
println("Heat min/max: ", minimum(out.heat), " / ", maximum(out.heat))
println("Edge min/max: ", minimum(out.edge), " / ", maximum(out.edge))
