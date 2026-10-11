
using Lux, Optimisers, Zygote, NNlib, NIfTI, Random, JSON, ComponentArrays, Printf, Statistics, JLD2
import CUDA
using LuxCUDA

include("network.jl")

function dice_loss(pred, target)
    p = @view pred[:, :, :, 2, :]
    t = @view target[:, :, :, 2, :]
    intersection = sum(p .* t, dims=(1,2,3))
    union = sum(p, dims=(1,2,3)) .+ sum(t, dims=(1,2,3))
    dsc = (2.0f0 .* intersection .+ 1f-6) ./ (union .+ 1f-6)
    return mean(1.0f0 .- dsc)
end

function combined_loss(pred, target)
    ce = mean(-target .* log.(pred .+ 1f-6))
    dl = dice_loss(pred, target)
    return 0.5f0 * ce + 0.5f0 * dl, ce, dl
end

struct CaseRandomLesionDataset
    case_ids::Vector{String}
    case_to_lesions::Dict{String, Vector{Any}}
end

function CaseRandomLesionDataset(manifest_path)
    items = JSON.parsefile(manifest_path)
    case_to_lesions = Dict{String, Vector{Any}}()
    for item in items
        case_id = item["case_id"]
        if !haskey(case_to_lesions, case_id)
            case_to_lesions[case_id] = []
        end
        push!(case_to_lesions[case_id], item)
    end
    case_ids = sort(collect(keys(case_to_lesions)))
    return CaseRandomLesionDataset(case_ids, case_to_lesions)
end

Base.length(ds::CaseRandomLesionDataset) = length(ds.case_ids)

function get_item(ds::CaseRandomLesionDataset, idx)
    case_id = ds.case_ids[idx]
    lesions = ds.case_to_lesions[case_id]
    item = rand(lesions)
    return load_item(item)
end

struct NiftiPatchDataset
    items::Vector{Any}
end

function NiftiPatchDataset(manifest_path)
    items = JSON.parsefile(manifest_path)
    return NiftiPatchDataset(items)
end

Base.length(ds::NiftiPatchDataset) = length(ds.items)

function get_item(ds::NiftiPatchDataset, idx)
    return load_item(ds.items[idx])
end

function load_item(item)
    ct = Float32.(niread(item["ct_path"]).raw)
    pet = Float32.(niread(item["pet_path"]).raw)
    ts = Float32.(niread(item["totalseg_path"]).raw)
    gt = UInt8.(niread(item["gt_path"]).raw)
    
    ct_m = mean(ct)
    ct_s = std(ct) + 1f-6
    ct_norm = (ct .- ct_m) ./ ct_s
    
    ts_norm = ts ./ 117.0f0
    
    x = cat(ct_norm, pet, ts_norm, dims=4)
    y = zeros(Float32, size(gt, 1), size(gt, 2), size(gt, 3), 2)
    y[:, :, :, 1] .= (gt .== 0)
    y[:, :, :, 2] .= (gt .== 1)
    
    return x, y
end

function compute_loss(model, ps, st, x, y)
    pred, st_new = model(x, ps, st)
    loss, ce, dl = combined_loss(pred, y)
    return loss, st_new, ce, dl
end

function validate_epoch(model, ps, st, val_ds, dev, batch_size)
    st = Lux.testmode(st)
    total_loss = 0.0f0
    total_dice = 0.0f0
    nb = 0
    
    n_samples = length(val_ds)
    for i in 1:batch_size:n_samples
        idx = i:min(i+batch_size-1, n_samples)
        items = [get_item(val_ds, j) for j in idx]
        x = cat([it[1] for it in items]..., dims=5) |> dev
        y = cat([it[2] for it in items]..., dims=5) |> dev
        
        pred, _ = model(x, ps, st)
        loss, ce, dl = combined_loss(pred, y)
        total_loss += loss
        total_dice += (1.0f0 - dl)
        nb += 1
    end
    return total_loss / nb, total_dice / nb
end

function train_model()
    dev = Lux.gpu_device()
    println("Using device: ", dev)
    
    batch_size = 16
    epochs = 3000
    patience = 8
    lr_max = 1e-4
    lr_min = 1e-6
    
    train_ds = CaseRandomLesionDataset("/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/train_manifest.json")
    val_ds = NiftiPatchDataset("/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/val_manifest.json")
    
    rng = Xoshiro(42)
    model = HELPNet_Lesion(3, 2)
    ps, st = Lux.setup(rng, model)
    ps = ComponentArray(ps) |> dev
    st = st |> dev
    
    opt = Optimisers.Adam(lr_max)
    st_opt = Optimisers.setup(opt, ps)
    
    best_val_dice = 0.0f0
    epochs_no_improve = 0
    
    for epoch in 1:epochs
        # LR Cosine Annealing
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * (epoch - 1) / epochs))
        Optimisers.adjust!(st_opt, lr)
        
        st = Lux.trainmode(st)
        total_loss = 0.0f0
        nb = 0
        
        indices = randperm(length(train_ds))
        for i in 1:batch_size:length(indices)
            idx = indices[i:min(i+batch_size-1, length(indices))]
            items = [get_item(train_ds, j) for j in idx]
            x = cat([it[1] for it in items]..., dims=5) |> dev
            y = cat([it[2] for it in items]..., dims=5) |> dev
            
            if nb == 0 && epoch == 1
                println("Compiling model... (This will take a few minutes)")
                flush(stdout)
            end
            (loss, st_new, ce, dl), back = Zygote.pullback(p -> compute_loss(model, p, st, x, y), ps)
            gs = back((1.0f0, nothing, nothing, nothing))[1]
            # In Zygote, gradients wrt parameters are in gs
            st_opt, ps = Optimisers.update(st_opt, ps, gs)
            st = st_new
            
            total_loss += loss
            nb += 1
            if nb % 10 == 0
                @printf("Epoch %04d | Batch %04d | Loss: %.4f\n", epoch, nb, loss); flush(stdout)
            end
        end
        
        val_loss, val_dice = validate_epoch(model, ps, st, val_ds, dev, batch_size)
        @printf("Epoch %04d | Train Loss: %.4f | Val Loss: %.4f | Val Dice: %.4f | LR: %.6f\n", epoch, total_loss/nb, val_loss, val_dice, lr); flush(stdout)
        
        if val_dice > best_val_dice
            best_val_dice = val_dice
            epochs_no_improve = 0
            # Save model
            ps_cpu = ps |> Lux.cpu_device()
            st_cpu = st |> Lux.cpu_device()
            jldsave("helpnet_julia_best.jld2"; ps=ps_cpu, st=st_cpu)
            println("  *** NEW BEST Validation Dice: $(best_val_dice) saved! ***")
        else
            epochs_no_improve += 1
            println("  [Early Stopping] No improvement for $(epochs_no_improve)/$(patience) epochs")
        end
        
        if epochs_no_improve >= patience
            println("EARLY STOPPING TRIGGERED!")
            break
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_model()
end
