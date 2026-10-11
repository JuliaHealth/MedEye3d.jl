using Pkg; Pkg.activate(".")
include("full_train.jl")

println("Overriding epochs and batch_size for testing...")
function train_model_test()
    dev = Lux.gpu_device()
    println("Using device: ", typeof(dev))
    
    batch_size = 2
    epochs = 1
    patience = 8
    lr_max = 1e-4
    lr_min = 1e-6
    
    train_ds = CaseRandomLesionDataset("/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/train_manifest.json")
    val_ds = NiftiPatchDataset("/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/val_manifest.json")
    
    # Trim datasets for testing
    train_ds.case_ids = train_ds.case_ids[1:4]
    val_ds.items = val_ds.items[1:4]
    
    rng = Xoshiro(42)
    model = HELPNet_Lesion(3, 2)
    ps, st = Lux.setup(rng, model)
    ps = ComponentArray(ps) |> dev
    st = st |> dev
    
    opt = Optimisers.Adam(lr_max)
    st_opt = Optimisers.setup(opt, ps)
    
    println("Starting test training loop...")
    st = Lux.trainmode(st)
    indices = [1, 2, 3, 4]
    for i in 1:batch_size:length(indices)
        idx = indices[i:min(i+batch_size-1, length(indices))]
        items = [get_item(train_ds, j) for j in idx]
        x = cat([it[1] for it in items]..., dims=5) |> dev
        y = cat([it[2] for it in items]..., dims=5) |> dev
        
        (loss, st_new, ce, dl), back = Zygote.pullback(p -> compute_loss(model, p, st, x, y), ps)
        gs = back((1.0f0, nothing, nothing, nothing))[1]
        
        st_opt, ps = Optimisers.update(st_opt, ps, gs)
        st = st_new
        println("Loss: ", loss)
    end
    
    println("Starting test validation loop...")
    val_loss, val_dice = validate_epoch(model, ps, st, val_ds, dev, batch_size)
    println("Val Loss: ", val_loss, " Val Dice: ", val_dice)
end
train_model_test()
