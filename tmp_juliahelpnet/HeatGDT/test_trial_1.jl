using Pkg; Pkg.activate("/mnt/big/project_ssd/project_ssd/semiautomatic/JuliaHELPNet")

include("hparam_search.jl")

function test_trial()
    dev = Lux.gpu_device()
    println("Testing single-lesion training on ", dev)

    train_manifest = "/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/train_single_lesion_manifest.json"
    val_manifest = "/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/val_single_lesion_manifest.json"

    train_ds = HeatGDTDataset.PatchHeatDataset(train_manifest)
    val_ds = HeatGDTDataset.PatchHeatDataset(val_manifest)
    println("Loaded $(length(train_ds)) train, $(length(val_ds)) val single lesions.")

    trial = Dict(
        "name" => "Test_SingleLesion_HED",
        "edge_type" => "hed3d",
        "pde_type" => "reaction_diffusion",
        "loss_type" => "dice_dtm_edge"
    )

    result = run_trial(trial, train_ds, val_ds, dev; epochs=1, batch_size=2, n_per_epoch=10)
    println("\n✓ Single-lesion test trial completed successfully!")
    println("Result: ", result)
end

test_trial()
