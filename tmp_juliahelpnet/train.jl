using Pkg; Pkg.activate(".")
using Statistics, Lux, Optimisers, Zygote, NNlib, NIfTI, Random, JSON, ComponentArrays, Printf
include("network.jl")

function dice_loss(pred, target)
    # pred, target have size (W, H, D, 2, N)
    # class 2 is lesion
    p = @view pred[:, :, :, 2, :]
    t = @view target[:, :, :, 2, :]
    intersection = sum(p .* t, dims=(1,2,3))
    union = sum(p, dims=(1,2,3)) .+ sum(t, dims=(1,2,3))
    dsc = (2.0f0 .* intersection .+ 1f-6) ./ (union .+ 1f-6)
    return mean(1.0f0 .- dsc)
end

function ce_loss(pred, target)
    loss = -target .* log.(pred .+ 1f-6)
    return mean(sum(loss, dims=(1,2,3,4))) # Wait, original is torch.mean.
    # torch.mean(target * log) over all elements.
    # In Julia, sum over all but batch, then mean? Or just mean over everything?
    # PyTorch's torch.mean(target * log) is mean over all elements (N * 2 * D * H * W).
    # Let's just do mean over all elements.
    # return mean(-target .* log.(pred .+ 1f-6))
end

function combined_loss(pred, target)
    # PyTorch original CE: -torch.mean(target * torch.log(pred + 1e-6)) -> this is mean over all elements!
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

function Base.length(ds::CaseRandomLesionDataset)
    return length(ds.case_ids)
end

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

function Base.length(ds::NiftiPatchDataset)
    return length(ds.items)
end

function get_item(ds::NiftiPatchDataset, idx)
    return load_item(ds.items[idx])
end

function load_item(item)
    ct = Float32.(niread(item["ct_path"]).raw)
    pet = Float32.(niread(item["pet_path"]).raw)
    ts = Float32.(niread(item["totalseg_path"]).raw)
    gt = UInt8.(niread(item["gt_path"]).raw)
    
    # CT z-score
    ct_m = mean(ct)
    ct_s = std(ct) + 1f-6
    ct_norm = (ct .- ct_m) ./ ct_s
    
    ts_norm = ts ./ 117.0f0
    
    # Stack [W, H, D, C]
    x = cat(ct_norm, pet, ts_norm, dims=4)
    
    # GT [W, H, D, 2]
    y = zeros(Float32, size(gt, 1), size(gt, 2), size(gt, 3), 2)
    y[:, :, :, 1] .= (gt .== 0)
    y[:, :, :, 2] .= (gt .== 1)
    
    return x, y, ct, pet, ts, gt
end

function get_batch(ds, indices)
    items = [get_item(ds, i) for i in indices]
    x_batch = cat([it[1] for it in items]..., dims=5)
    y_batch = cat([it[2] for it in items]..., dims=5)
    return x_batch, y_batch
end

function compute_loss(model, ps, st, x, y)
    pred, st_new = model(x, ps, st)
    loss, ce, dl = combined_loss(pred, y)
    return loss, st_new, (ce, dl, pred)
end

function train_model()
    # Parameters
    batch_size = 4  # Smaller batch size for Julia on CPU/GPU depending on setup, wait we have 1 GPU! But Zygote might use lots of memory. Wait, let's use batch_size 4 to test.
    epochs = 1
    
    println("Loading datasets...")
    train_ds = CaseRandomLesionDataset("/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/train_manifest.json")
    val_ds = NiftiPatchDataset("/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/val_manifest.json")
    println("Train cases: $(length(train_ds)), Val patches: $(length(val_ds))")
    
    rng = Xoshiro(42)
    model = HELPNet_Lesion(3, 2)
    ps, st = Lux.setup(rng, model)
    ps = ComponentArray(ps)
    
    
    # Needs CUDA if available, but Lux has specific GPU loading. We will use cu(...) or Lux.gpu(ps)
    # Actually wait, using ComponentArrays with gpu is a bit manual if Lux doesn't provide it.
    # We can just run on CPU first to verify correctness of 1 step!
    
    println("Running 1 batch forward...")
    x, y = get_batch(train_ds, 1:2)
    println("X size: ", size(x))
    
    pred, st_new = model(x, ps, Lux.testmode(st))
    loss, ce, dl = combined_loss(pred, y)
    println("Loss: ", loss, " CE: ", ce, " DL: ", dl)
    
    println("Running 1 backward step (Zygote)...")
    st = Lux.trainmode(st)
    gs = Zygote.gradient(p -> compute_loss(model, p, st, x, y)[1], ps)[1]
    println("Grad norm: ", sum(abs.(gs)))
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_model()
end
