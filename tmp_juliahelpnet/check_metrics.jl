using Pkg; Pkg.activate("/mnt/big/project_ssd/project_ssd/semiautomatic/JuliaHELPNet")
using Lux, Optimisers, Zygote, NNlib, NIfTI, Random, JSON, ComponentArrays, Printf, Statistics, JLD2
import CUDA
using LuxCUDA
include("network.jl")
include("full_train.jl")

dev = Lux.gpu_device()
model = HELPNet_Lesion(3, 2)
d = JLD2.load("/mnt/big/project_ssd/project_ssd/semiautomatic/JuliaHELPNet/helpnet_julia_best.jld2")
ps = ComponentArray(d["ps"]) |> dev
st = d["st"] |> dev
st = Lux.testmode(st)

val_ds = NiftiPatchDataset("/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/val_manifest.json")
batch_size = 16

all_hard_dice = Float32[]
for i in 1:batch_size:length(val_ds)
    idx = i:min(i+batch_size-1, length(val_ds))
    items = [get_item(val_ds, j) for j in idx]
    x = cat([it[1] for it in items]..., dims=5) |> dev
    y = cat([it[2] for it in items]..., dims=5) |> dev
    pred, _ = model(x, ps, st)
    
    # Compute hard dice
    p = Array(@view pred[:, :, :, 2, :]) .>= 0.5f0
    t = Array(@view y[:, :, :, 2, :]) .>= 0.5f0
    
    for b in 1:size(p, 4)
        pb = p[:,:,:,b]
        tb = t[:,:,:,b]
        inter = sum(pb .& tb)
        union = sum(pb) + sum(tb)
        dsc = (2.0f0 * inter + 1f-5) / (union + 1f-5)
        push!(all_hard_dice, dsc)
    end
    print(".")
end
println()
println("Julia Hard Dice: ", mean(all_hard_dice))
