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

include("heat_diffuse.jl")
include("model.jl")
include("dataset.jl")

using .HeatDiffusion
using .HeatGDTModel
using .HeatGDTDataset

function dice_loss(pred_mask, gt_mask)
    # pred_mask, gt_mask: (X, Y, Z, B)
    intersection = sum(pred_mask .* gt_mask, dims=(1, 2, 3))
    union = sum(pred_mask, dims=(1, 2, 3)) .+ sum(gt_mask, dims=(1, 2, 3))
    dsc = (2.0f0 .* intersection .+ 1f-5) ./ (union .+ 1f-5)
    return mean(1.0f0 .- dsc)
end

function bce_loss(pred_mask, gt_mask)
    p = clamp.(pred_mask, 1f-6, 1.0f0 - 1f-6)
    loss = -(gt_mask .* log.(p) .+ (1.0f0 .- gt_mask) .* log.(1.0f0 .- p))
    return mean(loss)
end

"""
    extract_3d_edges(gt)

Extracts morphological 3D surface boundary voxels ∂Y = Y \\ erode(Y) using GPU-accelerated max-pooling.
gt: (X, Y, Z, B)
returns: (X, Y, Z, 1, B)
"""
function extract_3d_edges(gt)
    X, Y, Z, B = size(gt)
    gt_5d = reshape(gt, X, Y, Z, 1, B)
    bg_5d = 1.0f0 .- gt_5d
    p = NNlib.maxpool(bg_5d, (3, 3, 3); stride=1, pad=1)
    return gt_5d .* p
end

function combined_loss_heavy(outputs, gt_mask)
    pred_mask = outputs.mask # (X, Y, Z, B)
    edge_map = outputs.edges # (X, Y, Z, 1, B)
    
    # Adjoint heat dice & bce loss
    dl = dice_loss(pred_mask, gt_mask)
    ce = bce_loss(pred_mask, gt_mask)
    
    # Surface edge boundary supervision
    edge_gt = extract_3d_edges(gt_mask)
    # Weighted BCE on edges to account for class imbalance (edge voxels are sparse)
    p_edge = clamp.(edge_map, 1f-6, 1.0f0 - 1f-6)
    edge_bce = -(5.0f0 .* edge_gt .* log.(p_edge) .+ (1.0f0 .- edge_gt) .* log.(1.0f0 .- p_edge))
    el = mean(edge_bce)
    
    sp = mean(edge_map)
    
    total = dl + 0.3f0 * ce + 0.5f0 * el + 0.01f0 * sp
    return total, dl, ce, el, sp
end

function compute_loss(model, ps, st, x, seed_mask, gt_mask)
    outputs, st_new = model((x, seed_mask), ps, st)
    loss, dl, ce, el, sp = combined_loss_heavy(outputs, gt_mask)
    return loss, st_new, dl, ce, el, sp
end

function evaluate(model, ps, st, val_ds, dev, batch_size=2; max_samples=40)
    st = Lux.testmode(st)
    total_dice = 0.0f0
    nb = 0
    
    n_eval = min(length(val_ds), max_samples)
    for i in 1:batch_size:n_eval
        idx = i:min(i+batch_size-1, n_eval)
        x_cpu, seed_cpu, gt_cpu = get_batch(val_ds, idx)
        x = x_cpu |> dev
        seed = seed_cpu |> dev
        gt = gt_cpu |> dev
        
        outputs, _ = model((x, seed), ps, st)
        p_bin = Array(outputs.mask) .>= 0.5f0
        t_bin = Array(gt) .>= 0.5f0
        
        for b in 1:size(p_bin, 4)
            pb = p_bin[:, :, :, b]
            tb = t_bin[:, :, :, b]
            inter = sum(pb .& tb)
            union = sum(pb) + sum(tb)
            dsc = (2.0f0 * inter + 1f-5) / (union + 1f-5)
            total_dice += dsc
            nb += 1
        end
    end
    return total_dice / max(nb, 1)
end

function train_heavy_heat_gdt()
    dev = Lux.gpu_device()
    println("="^80)
    println("Training Heavy 3D Multimodal Heat-GDT (~23.56M Parameters) on RTX 3090")
    println("Device: ", dev)
    println("="^80)
    
    # Hyperparameters
    batch_size = 2
    epochs = 10
    lr = 2e-4
    K_steps = 20
    dt = 0.12f0
    theta = 0.012f0
    tau = 0.03f0
    
    train_manifest = "/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/train_manifest.json"
    val_manifest = "/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/val_manifest.json"
    
    train_ds = PatchHeatDataset(train_manifest)
    val_ds = PatchHeatDataset(val_manifest)
    println("Loaded $(length(train_ds)) training patches, $(length(val_ds)) validation patches.")
    
    # Model Setup (Heavy 23.56M parameter network)
    rng = Xoshiro(42)
    model = HeatGDTFramework(; in_channels=3, edge_type=:heavy, K=K_steps, dt=dt, theta=theta, tau=tau)
    ps, st = Lux.setup(rng, model)
    ps = ComponentArray(ps) |> dev
    st = st |> dev
    println("Initialized Heavy Multimodal Model: ", length(ps), " parameters.")
    
    opt = Optimisers.Adam(lr)
    st_opt = Optimisers.setup(opt, ps)
    
    best_val_dice = 0.0f0
    ckpt_path = "/mnt/big/project_ssd/project_ssd/semiautomatic/JuliaHELPNet/HeatGDT/heat_gdt_heavy_best.jld2"
    
    println("\nRunning initial zero-shot validation...")
    val_dsc_init = evaluate(model, ps, st, val_ds, dev, batch_size; max_samples=20)
    @printf("Initial Validation Hard Dice: %.4f\n\n", val_dsc_init)
    
    n_train_epoch = 60 # 30 batches per epoch
    
    for epoch in 1:epochs
        st = Lux.trainmode(st)
        total_loss, total_dl, total_ce, total_el = 0.0f0, 0.0f0, 0.0f0, 0.0f0
        nb = 0
        
        indices = randperm(length(train_ds))[1:n_train_epoch]
        t0 = time()
        
        for i in 1:batch_size:length(indices)
            idx = indices[i:min(i+batch_size-1, length(indices))]
            x_cpu, seed_cpu, gt_cpu = get_batch(train_ds, idx)
            x = x_cpu |> dev
            seed = seed_cpu |> dev
            gt = gt_cpu |> dev
            
            # Forward + Backward via Zygote and custom GPU Adjoint
            (loss, st_new, dl, ce, el, sp), pb = Zygote.pullback(
                p -> compute_loss(model, p, st, x, seed, gt),
                ps
            )
            
            grads = pb((1.0f0, nothing, nothing, nothing, nothing, nothing))[1]
            st = st_new
            
            # Optimizer step
            st_opt, ps = Optimisers.update(st_opt, ps, grads)
            
            total_loss += loss
            total_dl += dl
            total_ce += ce
            total_el += el
            nb += 1
            
            if nb % 5 == 0 || nb == div(length(indices), batch_size)
                @printf("  [Ep %02d | Batch %02d/%02d] Loss: %.4f (DiceLoss: %.4f, EdgeBCE: %.4f)\n",
                        epoch, nb, div(length(indices), batch_size), loss, dl, el)
            end
        end
        
        dt_epoch = time() - t0
        avg_loss = total_loss / nb
        avg_dl = total_dl / nb
        
        # Validation
        val_dice = evaluate(model, ps, st, val_ds, dev, batch_size; max_samples=30)
        @printf("Epoch %02d [%.1fs] | Train Loss: %.4f | Train DiceLoss: %.4f | Val Hard Dice: %.4f\n",
                epoch, dt_epoch, avg_loss, avg_dl, val_dice)
                
        if val_dice > best_val_dice
            best_val_dice = val_dice
            println("  *** Saved new best heavy model (Val Dice: $(round(val_dice, digits=4))) to $(ckpt_path) ***")
            JLD2.save(ckpt_path, Dict("ps" => Array(ps), "st" => st, "epoch" => epoch, "val_dice" => val_dice))
        end
    end
    
    println("="^80)
    println("Heavy Heat-GDT Training Complete! Best Validation Dice: ", best_val_dice)
    println("="^80)
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_heavy_heat_gdt()
end
