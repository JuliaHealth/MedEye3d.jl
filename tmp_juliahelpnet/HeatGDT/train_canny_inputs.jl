using Pkg; Pkg.activate("/mnt/big/project_ssd/project_ssd/semiautomatic/JuliaHELPNet")

using Lux
using Optimisers
using Zygote
using CUDA
using LuxCUDA
using ComponentArrays
using Printf
using Statistics
using Random
using JLD2
using NNlib
using JSON

include("heat_diffuse.jl")
include("heat_diffuse_v2.jl")
include("model.jl")
include("dataset_canny.jl")

include("losses.jl")

include("architectures/hed3d.jl")
include("architectures/hed3d_wide.jl")
include("architectures/dexined3d.jl")
include("architectures/dframenet.jl")
include("architectures/hybrid25d.jl")

using .HeatDiffusion
using .HeatDiffusionV2
using .HeatGDTModel
using .HeatGDTDataset

struct HeatGDTFrameworkV2{M} <: Lux.AbstractLuxContainerLayer{(:edge_net,)}
    edge_net::M
    K::Int
    dt::Float32
    theta::Float32
    tau::Float32
    pde_type::Symbol
    edge_type::Symbol
end

function create_edge_net(edge_type::Symbol, in_ch::Int=3)
    if edge_type == :hed3d
        return HED3D(in_ch)
    elseif edge_type == :hed3d_wide
        return HED3D_Wide(in_ch)
    elseif edge_type == :dexined3d
        return DexiNed3D(in_ch)
    elseif edge_type == :dframenet
        return DFrameNet(in_ch)
    elseif edge_type == :hybrid25d
        return Hybrid25D3D(in_ch)
    elseif edge_type == :heavy
        return HeatGDTModel.HeavyMultimodalEdgeNet3D(in_ch)
    elseif edge_type == :deep
        return HeatGDTModel.DeepMultimodalEdgeNet3D(in_ch)
    else
        return HeatGDTModel.MultimodalEdgeNet(in_ch)
    end
end

function HeatGDTFrameworkV2(;
    in_channels=3,
    edge_type::Symbol=:deep,
    pde_type::Symbol=:pure_diffusion,
    K=40, dt=0.16f0, theta=0.001f0, tau=0.0001f0
)
    edge_net = create_edge_net(edge_type, in_channels)
    return HeatGDTFrameworkV2(edge_net, K, dt, theta, tau, pde_type, edge_type)
end

function (m::HeatGDTFrameworkV2)((x, seed_mask), ps, st)
    net_out, st_edge = m.edge_net(x, ps.edge_net, st.edge_net)

    if m.edge_type == :dframenet
        edge_map = net_out.edges
        direction_field = net_out.direction
    else
        edge_map = net_out
        direction_field = nothing
    end

    D = HeatGDTModel.compute_diffusivity(edge_map)

    if m.pde_type == :pure_diffusion
        u = HeatDiffusion.heat_diffusion(seed_mask, D, m.K, m.dt)
    elseif m.pde_type == :reaction_diffusion
        beta = edge_map .* 2.0f0
        v = Zygote.ignore_derivatives() do
            CUDA.zeros(Float32, size(D, 1), size(D, 2), size(D, 3), 3, size(D, 4))
        end
        u = HeatDiffusionV2.heat_diffusion_v2(seed_mask, D, beta, v, m.K, m.dt)
    elseif m.pde_type == :advection_diffusion
        beta = Zygote.ignore_derivatives() do
            CUDA.zeros(Float32, size(D)...)
        end
        if direction_field !== nothing
            u = HeatDiffusionV2.heat_diffusion_v2(seed_mask, D, beta, direction_field, m.K, m.dt)
        else
            v = Zygote.ignore_derivatives() do
                CUDA.zeros(Float32, size(D, 1), size(D, 2), size(D, 3), 3, size(D, 4))
            end
            u = HeatDiffusionV2.heat_diffusion_v2(seed_mask, D, beta, v, m.K, m.dt)
        end
    elseif m.pde_type == :full_rda
        beta = edge_map .* 2.0f0
        if direction_field !== nothing
            u = HeatDiffusionV2.heat_diffusion_v2(seed_mask, D, beta, direction_field, m.K, m.dt)
        else
            v = Zygote.ignore_derivatives() do
                CUDA.zeros(Float32, size(D, 1), size(D, 2), size(D, 3), 3, size(D, 4))
            end
            u = HeatDiffusionV2.heat_diffusion_v2(seed_mask, D, beta, v, m.K, m.dt)
        end
    else
        u = HeatDiffusion.heat_diffusion(seed_mask, D, m.K, m.dt)
    end

    pred_mask = HeatDiffusion.extract_mask(u, m.theta, m.tau)

    return (pred=pred_mask, heat=u, edge=edge_map, D=D), (edge_net=st_edge,)
end

function get_loss_fn(loss_type::Symbol)
    if loss_type == :dice_bce_edge
        return function(pred, heat, edge, gt_mask)
            dl = dice_loss(pred, gt_mask)
            ce = bce_loss(pred, gt_mask)
            sp = mean(edge)
            total = dl + 0.5f0 * ce + 0.01f0 * sp
            return total, dl
        end
    elseif loss_type == :dice_dtm_edge || loss_type == :dice_dtm
        return function(pred, heat, edge, gt_mask)
            dl = dice_loss(pred, gt_mask)
            dtm = boundary_loss_dtm(pred, gt_mask)
            sp = mean(edge)
            total = dl + 0.4f0 * dtm + 0.01f0 * sp
            return total, dl
        end
    elseif loss_type == :focal_tversky_dtm
        return function(pred, heat, edge, gt_mask)
            ftl = focal_tversky_loss(pred, gt_mask)
            dtm = boundary_loss_dtm(pred, gt_mask)
            sp = mean(edge)
            total = ftl + 0.4f0 * dtm + 0.01f0 * sp
            return total, ftl
        end
    elseif loss_type == :compound_dtm || loss_type == :compound
        return function(pred, heat, edge, gt_mask)
            total, dl, bl, dtm_l, sp = compound_loss(pred, edge, gt_mask)
            return total, dl
        end
    else
        return function(pred, heat, edge, gt_mask)
            dl = dice_loss(pred, gt_mask)
            ce = bce_loss(pred, gt_mask)
            total = dl + 0.5f0 * ce
            return total, dl
        end
    end
end

function get_lr(ep, max_ep; eta_max=1e-4, eta_min=1e-6)
    return Float32(eta_min + 0.5 * (eta_max - eta_min) * (1.0 + cos(pi * ep / max_ep)))
end

function train_champion()
    dev = Lux.gpu_device()
    println("Device: ", dev)
    
    result_path = "hparam_results.json"
    results = JSON.parsefile(result_path)
    champion = sort(results, by=r -> -get(r, "best_val_dice_fixed", 0.0))[1]
    
    edge_type = Symbol(champion["edge_type"])
    pde_type = Symbol(champion["pde_type"])
    loss_type = Symbol(champion["loss_type"])
    
    println("🏆 Training Champion: $(champion["name"])")
    println("   Edge: $edge_type | PDE: $pde_type | Loss: $loss_type")

    train_manifest = "/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/canny_train_manifest.json"
    val_manifest = "/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/canny_val_manifest.json"

    train_ds = PatchHeatDataset(train_manifest)
    val_ds = PatchHeatDataset(val_manifest)

    rng = Xoshiro(42)
    model = HeatGDTFrameworkV2(;
        in_channels=3,
        edge_type=edge_type,
        pde_type=pde_type,
        K=40, dt=0.16f0, theta=0.001f0, tau=0.0001f0
    )

    ps, st = Lux.setup(rng, model)
    ps = ComponentArray(ps) |> dev
    st = st |> dev

    opt = Adam(1f-4)
    opt_st = Optimisers.setup(opt, ps)

    epochs = 1000
    patience = 5
    batch_size = 2
    n_per_epoch = 200

    loss_fn = get_loss_fn(loss_type)

    function compute_loss(p, s, x_b, seed_b, gt_b)
        out, st_new = model((x_b, seed_b), p, s)
        total, d_loss = loss_fn(out.pred, out.heat, out.edge, gt_b)
        return total, st_new, d_loss
    end

    best_val_dice = 0.0f0
    epochs_without_improvement = 0

    for ep in 1:epochs
        # Adjust learning rate (cosine annealing)
        lr = get_lr(ep - 1, epochs)
        Optimisers.adjust!(opt_st, lr)

        train_loss = 0.0f0
        train_dice = 0.0f0
        idxs = shuffle(rng, 1:length(train_ds))[1:(n_per_epoch*batch_size)]
        
        for i in 1:batch_size:length(idxs)
            batch_idx = idxs[i:min(i+batch_size-1, length(idxs))]
            x_cpu, seed_cpu, gt_cpu, _ = get_batch(train_ds, batch_idx)
            x_b = x_cpu |> dev
            seed_b = seed_cpu |> dev
            gt_b = gt_cpu |> dev

            (loss_val, st, dl), back = Zygote.pullback(p -> compute_loss(p, st, x_b, seed_b, gt_b), ps)
            grads = back((one(loss_val), nothing, nothing))[1]
            opt_st, ps = Optimisers.update(opt_st, ps, grads)

            train_loss += loss_val
            train_dice += (1.0f0 - dl)
        end
        
        train_loss /= n_per_epoch
        train_dice /= n_per_epoch

        val_dice = 0.0f0
        val_idx = 1:min(length(val_ds), 60)
        n_val = 0
        for i in 1:batch_size:length(val_idx)
            batch_idx = val_idx[i:min(i+batch_size-1, length(val_idx))]
            x_cpu, seed_cpu, gt_cpu, _ = get_batch(val_ds, batch_idx)
            x_b = x_cpu |> dev
            seed_b = seed_cpu |> dev
            gt_b = gt_cpu |> dev

            st_test = Lux.testmode(st)
            out, _ = model((x_b, seed_b), ps, st_test)
            dl = dice_loss(out.pred, gt_b)
            val_dice += (1.0f0 - dl)
            n_val += 1
        end
        val_dice /= max(n_val, 1)

        @printf("Epoch %03d | LR: %.6f | Train Loss: %.4f | Train Dice: %.4f | Val Dice: %.4f\n", 
                ep, lr, train_loss, train_dice, val_dice)

        if val_dice > best_val_dice
            best_val_dice = val_dice
            epochs_without_improvement = 0
            ps_cpu = ps |> Lux.cpu_device()
            st_cpu = st |> Lux.cpu_device()
            jldsave("champion_model.jld2"; ps=ps_cpu, st=st_cpu, config=champion)
            println("  -> Saved new best model")
        else
            epochs_without_improvement += 1
            println("  -> No improvement ($epochs_without_improvement / $patience)")
            if epochs_without_improvement >= patience
                println("Early stopping triggered at epoch $(ep)!")
                break
            end
        end
        
        # Free GPU memory and flush logs
        GC.gc()
        CUDA.reclaim()
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_champion()
end
