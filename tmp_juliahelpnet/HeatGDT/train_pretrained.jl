# train_pretrained.jl — Phase 3b: Fine-tune pretrained edge network on lesion segmentation
# Loads pretrained edge weights, trains through heat PDE with edge supervision loss.
# Supports all 3 architecture variants: :hed3d (14.1M), :heavy (23.5M), :hed3d_wide (31.8M)
#
# Usage:
#   julia --project=../JuliaHELPNet train_pretrained.jl --edge_type=hed3d
#   julia --project=../JuliaHELPNet train_pretrained.jl --edge_type=heavy
#   julia --project=../JuliaHELPNet train_pretrained.jl --edge_type=hed3d_wide

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
using NIfTI

include("heat_diffuse.jl")
include("heat_diffuse_v2.jl")
include("model.jl")
include("dataset.jl")
include("losses.jl")

include("architectures/hed3d.jl")
include("architectures/hed3d_wide.jl")
include("architectures/hed3d_ultra.jl")
include("architectures/dexined3d.jl")
include("architectures/dframenet.jl")
include("architectures/hybrid25d.jl")

using .HeatDiffusion
using .HeatDiffusionV2
using .HeatGDTModel
using .HeatGDTDataset

# Re-use the HeatGDTFrameworkV2 from train_champion.jl
struct PretrainedHeatGDT{M} <: Lux.AbstractLuxContainerLayer{(:edge_net,)}
    edge_net::M
    K::Int
    dt::Float32
    theta::Float32
    tau::Float32
end

function create_edge_net_ft(edge_type::Symbol, in_ch::Int=3)
    if edge_type == :hed3d
        return HED3D(in_ch)
    elseif edge_type == :hed3d_wide
        return HED3D_Wide(in_ch)
    elseif edge_type == :hed3d_ultra
        return HED3D_Ultra(in_ch)
    elseif edge_type == :heavy
        return HeatGDTModel.HeavyMultimodalEdgeNet3D(in_ch)
    else
        error("Unknown edge type: $edge_type")
    end
end

function PretrainedHeatGDT(; edge_type::Symbol=:hed3d, K=40, dt=0.16f0, theta=0.001f0, tau=0.0001f0)
    edge_net = create_edge_net_ft(edge_type, 3)
    return PretrainedHeatGDT(edge_net, K, dt, theta, tau)
end

function (m::PretrainedHeatGDT)((x, seed_mask), ps, st)
    edge_map, st_edge = m.edge_net(x, ps.edge_net, st.edge_net)
    D = HeatGDTModel.compute_diffusivity(edge_map)
    u = HeatDiffusion.heat_diffusion(seed_mask, D, m.K, m.dt)
    pred_mask = HeatDiffusion.extract_mask(u, m.theta, m.tau)
    return (pred=pred_mask, heat=u, edge=edge_map, D=D), (edge_net=st_edge,)
end

function get_lr_ft(ep, max_ep; eta_max=1e-4, eta_min=1e-6)
    return Float32(eta_min + 0.5 * (eta_max - eta_min) * (1.0 + cos(pi * ep / max_ep)))
end

function train_pretrained(; edge_type::Symbol=:hed3d)
    dev = Lux.gpu_device()
    
    pretrained_path = "pretrained_edge_$(edge_type).jld2"
    save_name = "champion_pretrained_$(edge_type).jld2"
    
    println("═══════════════════════════════════════════════════════")
    println("  Fine-tuning Pretrained Edge Net — $edge_type")
    println("  Loading pretrained weights: $pretrained_path")
    println("  Device: $dev")
    println("═══════════════════════════════════════════════════════")

    # Check pretrained checkpoint exists
    if !isfile(pretrained_path)
        error("Pretrained edge checkpoint not found: $pretrained_path. Run pretrain_edges.jl first.")
    end

    # Load datasets
    train_manifest = "/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/train_combined_manifest.json"
    val_manifest = "/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/val_single_lesion_manifest.json"

    train_ds = PatchHeatDataset(train_manifest)
    val_ds = PatchHeatDataset(val_manifest)

    println("Train: $(length(train_ds)) patches | Val: $(length(val_ds)) patches")

    # Create model
    rng = Xoshiro(42)
    model = PretrainedHeatGDT(; edge_type=edge_type, K=40, dt=0.16f0, theta=0.001f0, tau=0.0001f0)
    ps, st = Lux.setup(rng, model)

    # Load pretrained edge weights
    pretrained = load(pretrained_path)
    pretrained_ps = pretrained["ps"]
    pretrained_st = pretrained["st"]
    
    # Replace edge_net parameters with pretrained ones
    ps_edge = ps.edge_net
    pre_ps_edge = pretrained_ps
    
    # Merge pretrained edge_net params into model params
    ps = merge(ps, (edge_net=pre_ps_edge,))
    st = merge(st, (edge_net=pretrained_st,))
    
    println("✓ Loaded pretrained edge weights from $pretrained_path")

    ps = ComponentArray(ps) |> dev
    st = st |> dev

    nparams = length(ps)
    println("Total model parameters: $nparams ($(round(nparams/1e6, digits=2))M)")

    # Differential learning rates via OptimiserChain with parameter-specific rules
    # Edge net: slower LR (1e-5), PDE head: faster LR (1e-4)
    # We'll use a single optimizer but scale LR manually
    opt = Adam(1f-4)
    opt_st = Optimisers.setup(opt, ps)

    epochs = 1000
    patience = 5
    batch_size = 2
    n_per_epoch = 200
    
    # Edge supervision weight
    w_edge_sup = 0.1f0

    function compute_loss(p, s, x_b, seed_b, gt_b, edge_gt_b)
        out, st_new = model((x_b, seed_b), p, s)
        
        # Main segmentation losses
        dl = dice_loss(out.pred, gt_b)
        ce = bce_loss(out.pred, gt_b)
        sp = mean(out.edge)  # sparsity regularizer
        
        # Edge supervision loss (from TotalSeg GT)
        edge_sup = edge_supervision_loss(out.edge, edge_gt_b)
        
        # Total loss: segmentation + edge supervision
        total = dl + 0.5f0 * ce + 0.01f0 * sp + w_edge_sup * edge_sup
        
        return total, st_new, dl
    end

    best_val_dice = 0.0f0
    epochs_no_improve = 0

    for ep in 1:epochs
        # Cosine LR with lower max for fine-tuning
        lr = get_lr_ft(ep - 1, epochs; eta_max=5e-5, eta_min=1e-6)
        Optimisers.adjust!(opt_st, lr)

        train_loss = 0.0f0
        train_dice = 0.0f0
        idxs = shuffle(rng, 1:length(train_ds))[1:(n_per_epoch * batch_size)]

        st = Lux.trainmode(st)

        for i in 1:batch_size:length(idxs)
            batch_idx = idxs[i:min(i + batch_size - 1, length(idxs))]
            x_cpu, seed_cpu, gt_cpu, edge_gt_cpu = get_batch(train_ds, batch_idx; augment=true)
            x_b = x_cpu |> dev
            seed_b = seed_cpu |> dev
            gt_b = gt_cpu |> dev
            edge_gt_b = edge_gt_cpu |> dev

            (loss_val, st, dl), back = Zygote.pullback(p -> compute_loss(p, st, x_b, seed_b, gt_b, edge_gt_b), ps)
            grads = back((one(loss_val), nothing, nothing))[1]

            if any(isnan, grads)
                @warn "NaN gradient at epoch $ep, skipping batch"
                continue
            end

            opt_st, ps = Optimisers.update(opt_st, ps, grads)

            train_loss += loss_val
            train_dice += (1.0f0 - dl)
        end

        train_loss /= n_per_epoch
        train_dice /= n_per_epoch

        # Validation
        val_dice = 0.0f0
        val_idx = 1:min(length(val_ds), 60)
        n_val = 0
        st_test = Lux.testmode(st)

        for i in 1:batch_size:length(val_idx)
            batch_idx = val_idx[i:min(i + batch_size - 1, length(val_idx))]
            x_cpu, seed_cpu, gt_cpu, _ = get_batch(val_ds, batch_idx; augment=false)
            x_b = x_cpu |> dev
            seed_b = seed_cpu |> dev
            gt_b = gt_cpu |> dev

            out, _ = model((x_b, seed_b), ps, st_test)
            dl = dice_loss(out.pred, gt_b)
            val_dice += (1.0f0 - dl)
            n_val += 1
            
            if ep % 2 == 0 && i == 1
                out_pred = Array(out.pred)
                out_edge = Array(out.edge)
                for b in 1:size(x_cpu, 5)
                    mkpath("debug_val_full")
                    vol_x = x_cpu[:, :, :, :, b]
                    vol_gt = gt_cpu[:, :, :, 1, b]
                    vol_pred = out_pred[:, :, :, 1, b]
                    vol_edge = out_edge[:, :, :, 1, b]
                    niwrite("debug_val_full/ep$(ep)_b$(b)_pet.nii.gz", NIVolume(vol_x[:,:,:,2]))
                    niwrite("debug_val_full/ep$(ep)_b$(b)_gt.nii.gz", NIVolume(vol_gt))
                    niwrite("debug_val_full/ep$(ep)_b$(b)_pred.nii.gz", NIVolume(vol_pred))
                    niwrite("debug_val_full/ep$(ep)_b$(b)_edge.nii.gz", NIVolume(vol_edge))
                end
            end
        end
        val_dice /= max(n_val, 1)

        @printf("Epoch %03d | LR: %.6f | Train Loss: %.4f | Train Dice: %.4f | Val Dice: %.4f\n",
                ep, lr, train_loss, train_dice, val_dice)

        if val_dice > best_val_dice
            best_val_dice = val_dice
            epochs_no_improve = 0
            ps_cpu = ps |> Lux.cpu_device()
            st_cpu = st |> Lux.cpu_device()
            jldsave(save_name;
                ps=ps_cpu, st=st_cpu,
                edge_type=String(edge_type),
                best_val_dice=Float64(best_val_dice),
                pretrained_from=pretrained_path
            )
            println("  -> Saved new best: $save_name (val_dice=$(@sprintf("%.4f", best_val_dice)))")
        else
            epochs_no_improve += 1
            println("  -> No improvement ($epochs_no_improve / $patience)")
            if epochs_no_improve >= patience
                println("Early stopping at epoch $(ep)!")
                break
            end
        end

        GC.gc()
        CUDA.reclaim()
        flush(stdout)
    end

    println("\n✅ Fine-tuning complete for pretrained $edge_type")
    println("   Best val Dice: $(@sprintf("%.4f", best_val_dice))")
    println("   Saved to: $save_name")
end

# Parse command-line arguments
if abspath(PROGRAM_FILE) == @__FILE__
    local _edge_type = :hed3d  # default
    for arg in ARGS
        if startswith(arg, "--edge_type=")
            _edge_type = Symbol(split(arg, "=")[2])
        end
    end
    train_pretrained(; edge_type=_edge_type)
end
