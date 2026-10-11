using Pkg; Pkg.activate("/mnt/big/project_ssd/project_ssd/semiautomatic/JuliaHELPNet")

using Lux
using Optimisers
using Zygote
using CUDA
using LuxCUDA
using ComponentArrays
using Random

include("heat_diffuse.jl")
include("heat_diffuse_v2.jl")
include("model.jl")
include("dataset.jl")
include("losses.jl")
include("adaptive_head.jl")

include("architectures/hed3d.jl")
include("architectures/dexined3d.jl")
include("architectures/dframenet.jl")
include("architectures/hybrid25d.jl")

using .HeatDiffusion
using .HeatDiffusionV2
using .HeatGDTModel
using .HeatGDTDataset

include("hparam_search.jl")

function smoke_test()
    dev = Lux.gpu_device()
    println("Testing architectures and PDE types on ", dev)

    configs = [
        (:hed3d, :pure_diffusion, :dice_bce_edge),
        (:hed3d, :reaction_diffusion, :dice_dtm_edge),
        (:dexined3d, :reaction_diffusion, :dice_dtm_edge),
        (:dframenet, :advection_diffusion, :dice_dtm),
        (:hybrid25d, :reaction_diffusion, :dice_dtm),
        (:heavy, :reaction_diffusion, :compound_dtm),
    ]

    # Create dummy batch of batch size 2 on CPU first (Lux BatchNorm requires batch_size >= 2)
    B = 2
    x = CUDA.rand(Float32, 16, 16, 16, 3, B)
    seed_cpu = zeros(Float32, 16, 16, 16, B)
    seed_cpu[8, 8, 8, :] .= 1.0f0
    seed = CuArray(seed_cpu)

    gt_cpu = zeros(Float32, 16, 16, 16, B)
    gt_cpu[7:9, 7:9, 7:9, :] .= 1.0f0
    gt = CuArray(gt_cpu)

    rng = Xoshiro(42)

    for (edge_t, pde_t, loss_t) in configs
        print("Testing edge=$edge_t, pde=$pde_t, loss=$loss_t ... ")
        model = HeatGDTFrameworkV2(;
            in_channels=3,
            edge_type=edge_t,
            pde_type=pde_t,
            K=5, dt=0.12f0, theta=0.012f0, tau=0.03f0
        )
        ps, st = Lux.setup(rng, model)
        ps = ComponentArray(ps) |> dev
        st = st |> dev

        # Test forward + backward with Zygote
        (loss, st_new, dl), pb = Zygote.pullback(ps) do p
            outputs, st_n = model((x, seed), p, st)
            loss_val, dl_val = compute_trial_loss(outputs, gt, loss_t)
            return loss_val, st_n, dl_val
        end

        grads = pb((1.0f0, nothing, nothing))[1]
        println("✓ PASS (loss = $(round(loss, digits=4)))")
        CUDA.reclaim()
    end

    println("\nAll configurations passed smoke test! Ready for full sweep.")
end

smoke_test()
