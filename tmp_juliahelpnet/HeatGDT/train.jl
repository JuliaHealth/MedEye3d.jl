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

function combined_loss(outputs, gt_mask)
    pred_mask = outputs.mask
    edge_map = outputs.edges
    
    dl = dice_loss(pred_mask, gt_mask)
    ce = bce_loss(pred_mask, gt_mask)
    sparse = mean(edge_map)
    
    total = dl + 0.5f0 * ce + 0.02f0 * sparse
    return total, dl, ce, sparse
end

function compute_loss(model, ps, st, x, seed_mask, gt_mask)
    outputs, st_new = model((x, seed_mask), ps, st)
    loss, dl, ce, sp = combined_loss(outputs, gt_mask)
    return loss, st_new, dl, ce, sp
end

function evaluate(model, ps, st, val_ds, dev, batch_size=4; max_samples=40)
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

function train_heat_gdt()
    dev = Lux.gpu_device()
    println("="^70)
    println("Training Differentiable Heat-GDT Framework with Learned Edges")
    println("Device: ", dev)
    
    # Hyperparameters
    batch_size = 4
    epochs = 20
    lr = 5e-4
    K_steps = 20
    dt = 0.12f0
    theta = 0.012f0 # heat threshold for lesion boundary
    tau = 0.03f0    # soft thresholding temperature
    
    train_manifest = "/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/train_manifest.json"
    val_manifest = "/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/val_manifest.json"
    
    train_ds = PatchHeatDataset(train_manifest)
    val_ds = PatchHeatDataset(val_manifest)
    println("Loaded $(length(train_ds)) training patches, $(length(val_ds)) validation patches.")
    
    # Model Setup (Deep 3-Channel 1.43M param model)
    rng = Xoshiro(42)
    model = HeatGDTFramework(; in_channels=3, K=K_steps, dt=dt, theta=theta, tau=tau)
    ps, st = Lux.setup(rng, model)
    ps = ComponentArray(ps) |> dev
    st = st |> dev
    println("Model Parameters: ", length(ps))
    
    opt = Optimisers.Adam(lr)
    st_opt = Optimisers.setup(opt, ps)
    
    best_val_dice = 0.0f0
    
    # Initial validation before training
    println("\nRunning initial zero-shot validation...")
    val_dsc_init = evaluate(model, ps, st, val_ds, dev, batch_size; max_samples=20)
    @printf("Initial Validation Hard Dice: %.4f\n\n", val_dsc_init)
    
    # Training Loop
    # We sample a representative balanced subset per epoch for fast convergence demonstration
    n_train_epoch = 120
    for epoch in 1:epochs
        st = Lux.trainmode(st)
        total_loss, total_dl, total_ce, total_sp = 0.0f0, 0.0f0, 0.0f0, 0.0f0
        nb = 0
        
        # Shuffle indices
        indices = randperm(length(train_ds))[1:n_train_epoch]
        t0 = time()
        
        for i in 1:batch_size:length(indices)
            idx = indices[i:min(i+batch_size-1, length(indices))]
            x_cpu, seed_cpu, gt_cpu = get_batch(train_ds, idx)
            x = x_cpu |> dev
            seed = seed_cpu |> dev
            gt = gt_cpu |> dev
            
            # Forward + Backward with custom adjoint
            (loss, st_new, dl, ce, sp), back = Zygote.pullback(p -> compute_loss(model, p, st, x, seed, gt), ps)
            gs = back((1.0f0, nothing, nothing, nothing, nothing))[1]
            
            st_opt, ps = Optimisers.update(st_opt, ps, gs)
            st = st_new
            
            total_loss += loss
            total_dl += dl
            total_ce += ce
            total_sp += sp
            nb += 1
            
            if nb % 5 == 0
                @printf("  [Ep %02d | Batch %02d/%02d] Loss: %.4f (DiceLoss: %.4f, BCE: %.4f, EdgeSparse: %.4f)\n",
                        epoch, nb, div(n_train_epoch, batch_size), loss, dl, ce, sp)
                flush(stdout)
            end
        end
        
        epoch_time = time() - t0
        avg_loss = total_loss / nb
        avg_dl = total_dl / nb
        
        # Validation
        val_dsc = evaluate(model, ps, st, val_ds, dev, batch_size; max_samples=40)
        @printf("Epoch %02d [%.1fs] | Train Loss: %.4f | Train DiceLoss: %.4f | Val Hard Dice: %.4f\n",
                epoch, epoch_time, avg_loss, avg_dl, val_dsc)
        flush(stdout)
        
        if val_dsc > best_val_dice
            best_val_dice = val_dsc
            # Save checkpoint
            ps_cpu = ps |> Lux.cpu_device()
            st_cpu = st |> Lux.cpu_device()
            save_path = "/mnt/big/project_ssd/project_ssd/semiautomatic/JuliaHELPNet/HeatGDT/heat_gdt_deep_best.jld2"
            jldsave(save_path; ps=ps_cpu, st=st_cpu, val_dice=best_val_dice, epoch=epoch)
            println("  *** NEW BEST Validation Dice: $(round(best_val_dice, digits=4)) saved to $(save_path) ***")
        end
    end
    
    println("\nTraining complete! Best Validation Dice: ", best_val_dice)
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_heat_gdt()
end
