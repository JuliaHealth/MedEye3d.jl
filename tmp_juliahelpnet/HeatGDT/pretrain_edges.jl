# pretrain_edges.jl — Phase 3a: Supervised edge pretraining on TotalSegmentator boundaries
# Trains edge network ONLY (no heat PDE) to reproduce organ boundary edges.
# Supports all 3 architecture variants: :hed3d (14.1M), :heavy (23.5M), :hed3d_wide (31.8M)
#
# Usage:
#   julia --project=../JuliaHELPNet pretrain_edges.jl --edge_type=hed3d
#   julia --project=../JuliaHELPNet pretrain_edges.jl --edge_type=heavy
#   julia --project=../JuliaHELPNet pretrain_edges.jl --edge_type=hed3d_wide

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

include("dataset.jl")
include("losses.jl")

include("architectures/hed3d.jl")
include("architectures/hed3d_wide.jl")
include("architectures/hed3d_ultra.jl")
include("architectures/dexined3d.jl")
include("architectures/dframenet.jl")
include("architectures/hybrid25d.jl")
include("model.jl")
using .HeatGDTModel
using .HeatGDTDataset

function create_edge_net_pretrain(edge_type::Symbol, in_ch::Int=3)
    if edge_type == :hed3d
        return HED3D(in_ch)
    elseif edge_type == :hed3d_wide
        return HED3D_Wide(in_ch)
    elseif edge_type == :hed3d_ultra
        return HED3D_Ultra(in_ch)
    elseif edge_type == :heavy
        return HeatGDTModel.HeavyMultimodalEdgeNet3D(in_ch)
    else
        error("Unknown edge type for pretraining: $edge_type. Use :hed3d, :heavy, or :hed3d_wide")
    end
end

"""
    filter_totalseg_indices(ds::PatchHeatDataset)

Returns indices of samples that have valid TotalSegmentator data (non-empty edge_gt).
During pretraining, we ONLY use patches with real TotalSeg supervision.
"""
function filter_totalseg_indices(ds::PatchHeatDataset)
    valid = Int[]
    for (i, item) in enumerate(ds.items)
        if get(item, "has_totalseg", false) == true
            push!(valid, i)
        end
    end
    return valid
end

"""
    get_lr_pretrain(ep, max_ep; eta_max=1e-4, eta_min=1e-6)

Cosine annealing learning rate schedule for edge pretraining.
"""
function get_lr_pretrain(ep, max_ep; eta_max=1e-4, eta_min=1e-6)
    return Float32(eta_min + 0.5 * (eta_max - eta_min) * (1.0 + cos(pi * ep / max_ep)))
end

function pretrain_edges(; edge_type::Symbol=:hed3d)
    dev = Lux.gpu_device()
    println("═══════════════════════════════════════════════════")
    println("  Edge Pretraining — Architecture: $edge_type")
    println("  Device: $dev")
    println("═══════════════════════════════════════════════════")

    # Load datasets
    train_manifest = "/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/train_combined_manifest.json"
    val_manifest = "/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/val_single_lesion_manifest.json"

    train_ds = PatchHeatDataset(train_manifest)
    val_ds = PatchHeatDataset(val_manifest)

    # Filter to only patches WITH TotalSegmentator
    train_valid = filter_totalseg_indices(train_ds)
    val_valid = filter_totalseg_indices(val_ds)

    println("Train patches with TotalSeg: $(length(train_valid)) / $(length(train_ds))")
    println("Val patches with TotalSeg: $(length(val_valid)) / $(length(val_ds))")

    # Create edge network
    rng = Xoshiro(42)
    edge_net = create_edge_net_pretrain(edge_type, 3)
    ps, st = Lux.setup(rng, edge_net)
    ps = ComponentArray(ps) |> dev
    st = st |> dev

    nparams = length(ps)
    println("Edge network parameters: $(nparams) ($(round(nparams/1e6, digits=2))M)")

    # Optimizer
    opt = Adam(1f-4)
    opt_st = Optimisers.setup(opt, ps)

    # Hyperparameters
    epochs = 100
    patience = 15
    batch_size = 2
    n_per_epoch = 400  # samples per epoch (from ~2400 valid patches)
    
    # Output paths
    save_name = "pretrained_edge_$(edge_type).jld2"

    best_val_loss = Inf32
    epochs_no_improve = 0

    for ep in 1:epochs
        # Cosine LR
        lr = get_lr_pretrain(ep - 1, epochs)
        Optimisers.adjust!(opt_st, lr)

        # === Training ===
        st = Lux.trainmode(st)
        train_loss_sum = 0.0f0
        train_dice_sum = 0.0f0
        n_train_batches = 0

        # Shuffle valid indices and take n_per_epoch
        shuffled = shuffle(rng, train_valid)
        epoch_indices = shuffled[1:min(n_per_epoch, length(shuffled))]

        for i in 1:batch_size:length(epoch_indices)
            batch_idx = epoch_indices[i:min(i + batch_size - 1, length(epoch_indices))]
            
            # Load batch with edge_gt
            x_batch, _, _, edge_gt_batch = get_batch(train_ds, batch_idx; augment=true)
            x_b = x_batch |> dev
            edge_gt_b = edge_gt_batch |> dev

            # Forward + backward
            function compute_edge_loss(p, s)
                edge_pred, st_new = edge_net(x_b, p, s)
                # Combined BCE + Dice loss for edge supervision
                loss = edge_supervision_loss(edge_pred, edge_gt_b)
                return loss, st_new
            end

            (loss_val, st), back = Zygote.pullback(p -> compute_edge_loss(p, st), ps)
            grads = back((one(loss_val), nothing))[1]

            # Check for NaN gradients
            if any(isnan, grads)
                @warn "NaN gradient at epoch $ep, skipping batch"
                continue
            end

            opt_st, ps = Optimisers.update(opt_st, ps, grads)

            train_loss_sum += loss_val
            n_train_batches += 1
        end

        avg_train_loss = n_train_batches > 0 ? train_loss_sum / n_train_batches : NaN32

        # === Validation ===
        st_test = Lux.testmode(st)
        val_loss_sum = 0.0f0
        val_dice_sum = 0.0f0
        n_val_batches = 0

        for i in 1:batch_size:length(val_valid)
            batch_idx = val_valid[i:min(i + batch_size - 1, length(val_valid))]
            x_batch, _, _, edge_gt_batch = get_batch(val_ds, batch_idx; augment=false)
            x_b = x_batch |> dev
            edge_gt_b = edge_gt_batch |> dev

            edge_pred, _ = edge_net(x_b, ps, st_test)
            loss = edge_supervision_loss(edge_pred, edge_gt_b)
            
            # Edge Dice for monitoring
            e_dice = 1.0f0 - dice_loss(edge_pred, edge_gt_b)
            
            if ep % 2 == 0 && i == 1
                out_edge = Array(edge_pred)
                for b in 1:size(x_batch, 5)
                    mkpath("debug_val_edges")
                    vol_x = x_batch[:, :, :, :, b]
                    vol_gt = edge_gt_batch[:, :, :, 1, b]
                    vol_pred = out_edge[:, :, :, 1, b]
                    niwrite("debug_val_edges/ep$(ep)_b$(b)_pet.nii.gz", NIVolume(vol_x[:,:,:,2]))
                    niwrite("debug_val_edges/ep$(ep)_b$(b)_gt.nii.gz", NIVolume(vol_gt))
                    niwrite("debug_val_edges/ep$(ep)_b$(b)_pred.nii.gz", NIVolume(vol_pred))
                end
            end
            
            val_loss_sum += loss
            val_dice_sum += e_dice
            n_val_batches += 1
        end

        avg_val_loss = n_val_batches > 0 ? val_loss_sum / n_val_batches : NaN32
        avg_val_dice = n_val_batches > 0 ? val_dice_sum / n_val_batches : NaN32

        @printf("Epoch %03d | LR: %.6f | Train Loss: %.4f | Val Loss: %.4f | Val Edge Dice: %.4f\n",
                ep, lr, avg_train_loss, avg_val_loss, avg_val_dice)

        # Early stopping
        if avg_val_loss < best_val_loss
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            ps_cpu = ps |> Lux.cpu_device()
            st_cpu = st |> Lux.cpu_device()
            jldsave(save_name;
                ps=ps_cpu, st=st_cpu,
                edge_type=String(edge_type),
                best_val_loss=best_val_loss,
                epoch=ep
            )
            println("  -> Saved best edge model: $save_name (val_loss=$(@sprintf("%.4f", best_val_loss)))")
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

    println("\n✅ Edge pretraining complete for $edge_type")
    println("   Best val loss: $(@sprintf("%.4f", best_val_loss))")
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
    pretrain_edges(; edge_type=_edge_type)
end
