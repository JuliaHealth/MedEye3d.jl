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

# ── Include core modules ──────────────────────────────────────────────────────
include("heat_diffuse.jl")
include("heat_diffuse_v2.jl")
include("model.jl")
include("dataset.jl")
include("losses.jl")
include("adaptive_head.jl")

# Include architecture files
include("architectures/hed3d.jl")
include("architectures/dexined3d.jl")
include("architectures/dframenet.jl")
include("architectures/hybrid25d.jl")

using .HeatDiffusion
using .HeatDiffusionV2
using .HeatGDTModel
using .HeatGDTDataset

# ══════════════════════════════════════════════════════════════════════════════
# Extended Framework supporting all architectures + reaction-diffusion-advection
# ══════════════════════════════════════════════════════════════════════════════

"""
    HeatGDTFrameworkV2{M}

Extended framework supporting:
- Multiple edge architectures (:deep, :heavy, :hed3d, :dexined3d, :dframenet, :hybrid25d)
- PDE types (:pure_diffusion, :reaction_diffusion, :advection_diffusion, :full_rda)
- Compound loss with DTM boundary supervision
"""
struct HeatGDTFrameworkV2{M} <: Lux.AbstractLuxContainerLayer{(:edge_net,)}
    edge_net::M
    K::Int
    dt::Float32
    theta::Float32
    tau::Float32
    pde_type::Symbol      # :pure_diffusion, :reaction_diffusion, :advection_diffusion, :full_rda
    edge_type::Symbol     # :deep, :heavy, :hed3d, :dexined3d, :dframenet, :hybrid25d
end

function create_edge_net(edge_type::Symbol, in_ch::Int=3)
    if edge_type == :hed3d
        return HED3D(in_ch)
    elseif edge_type == :dexined3d
        return DexiNed3D(in_ch)
    elseif edge_type == :dframenet
        return DFrameNet(in_ch)
    elseif edge_type == :hybrid25d
        return Hybrid25D3D(in_ch)
    elseif edge_type == :heavy
        return HeatGDTModel.HeavyMultimodalEdgeNet3D(in_ch)
    elseif edge_type == :deep
        return HeatGDTModel.DeepMultimodalEdgeNet3D(in_ch)
    else
        return HeatGDTModel.MultimodalEdgeNet(in_ch)
    end
end

function HeatGDTFrameworkV2(;
    in_channels=3,
    edge_type::Symbol=:deep,
    pde_type::Symbol=:pure_diffusion,
    K=20, dt=0.12f0, theta=0.012f0, tau=0.03f0
)
    edge_net = create_edge_net(edge_type, in_channels)
    return HeatGDTFrameworkV2(edge_net, K, dt, theta, tau, pde_type, edge_type)
end

function (m::HeatGDTFrameworkV2)((x, seed_mask), ps, st)
    # x: (X, Y, Z, 3, B), seed_mask: (X, Y, Z, B)
    net_out, st_edge = m.edge_net(x, ps.edge_net, st.edge_net)

    # Handle DFrameNet (returns NamedTuple with edges + direction)
    if m.edge_type == :dframenet
        edge_map = net_out.edges
        direction_field = net_out.direction  # (X, Y, Z, 3, B)
    else
        edge_map = net_out
        direction_field = nothing
    end

    # Compute diffusivity from edge map
    D = HeatGDTModel.compute_diffusivity(edge_map)

    # PDE solve based on type
    if m.pde_type == :pure_diffusion
        u = HeatDiffusion.heat_diffusion(seed_mask, D, m.K, m.dt)
    elseif m.pde_type == :reaction_diffusion
        # Use edge map as absorption field β
        beta = dropdims(edge_map, dims=4) .* 2.0f0  # scale edges → absorption
        v = Zygote.ignore_derivatives() do
            CUDA.zeros(Float32, size(D, 1), size(D, 2), size(D, 3), 3, size(D, 4))
        end
        u = HeatDiffusionV2.heat_diffusion_v2(seed_mask, D, beta, v, m.K, m.dt)
    elseif m.pde_type == :advection_diffusion
        beta = Zygote.ignore_derivatives() do
            CUDA.zeros(Float32, size(D)...)
        end
        if direction_field !== nothing
            u = HeatDiffusionV2.heat_diffusion_v2(seed_mask, D, beta, direction_field, m.K, m.dt)
        else
            v = Zygote.ignore_derivatives() do
                CUDA.zeros(Float32, size(D, 1), size(D, 2), size(D, 3), 3, size(D, 4))
            end
            u = HeatDiffusionV2.heat_diffusion_v2(seed_mask, D, beta, v, m.K, m.dt)
        end
    elseif m.pde_type == :full_rda
        beta = dropdims(edge_map, dims=4) .* 2.0f0
        if direction_field !== nothing
            u = HeatDiffusionV2.heat_diffusion_v2(seed_mask, D, beta, direction_field, m.K, m.dt)
        else
            v = Zygote.ignore_derivatives() do
                CUDA.zeros(Float32, size(D, 1), size(D, 2), size(D, 3), 3, size(D, 4))
            end
            u = HeatDiffusionV2.heat_diffusion_v2(seed_mask, D, beta, v, m.K, m.dt)
        end
    else
        u = HeatDiffusion.heat_diffusion(seed_mask, D, m.K, m.dt)
    end

    pred_mask = HeatDiffusion.extract_mask(u, m.theta, m.tau)

    return (mask=pred_mask, edges=edge_map, heat=u, D=D), (edge_net=st_edge,)
end

# ══════════════════════════════════════════════════════════════════════════════
# Loss Function Wrappers for Each Trial Configuration
# ══════════════════════════════════════════════════════════════════════════════

"""Extract morphological surface boundary for edge BCE supervision."""
function extract_3d_edges(gt)
    X, Y, Z, B = size(gt)
    gt_5d = reshape(gt, X, Y, Z, 1, B)
    bg_5d = 1.0f0 .- gt_5d
    p = NNlib.maxpool(bg_5d, (3, 3, 3); stride=1, pad=1)
    return gt_5d .* p
end

function compute_trial_loss(outputs, gt_mask, loss_type::Symbol)
    pred = outputs.mask     # (X, Y, Z, B)
    edge = outputs.edges    # (X, Y, Z, 1, B)

    if loss_type == :dice_bce_edge
        # Original loss from train_heavy.jl
        dl = dice_loss(pred, gt_mask)
        ce = bce_loss(pred, gt_mask)
        edge_gt = Zygote.ignore_derivatives() do
            extract_3d_edges(gt_mask)
        end
        p_edge = clamp.(edge, 1f-6, 1.0f0 - 1f-6)
        edge_bce = -(5.0f0 .* edge_gt .* log.(p_edge) .+ (1.0f0 .- edge_gt) .* log.(1.0f0 .- p_edge))
        el = mean(edge_bce)
        sp = mean(edge)
        total = dl + 0.3f0 * ce + 0.5f0 * el + 0.01f0 * sp
        return total, dl

    elseif loss_type == :dice_dtm_edge
        # Dice + DTM boundary + edge sparsity
        dl = dice_loss(pred, gt_mask)
        dtm_l = boundary_loss_dtm(pred, gt_mask)
        edge_gt = Zygote.ignore_derivatives() do
            extract_3d_edges(gt_mask)
        end
        p_edge = clamp.(edge, 1f-6, 1.0f0 - 1f-6)
        edge_bce = -(5.0f0 .* edge_gt .* log.(p_edge) .+ (1.0f0 .- edge_gt) .* log.(1.0f0 .- p_edge))
        el = mean(edge_bce)
        sp = mean(edge)
        total = dl + 0.4f0 * dtm_l + 0.5f0 * el + 0.01f0 * sp
        return total, dl

    elseif loss_type == :focal_tversky_dtm
        # Focal Tversky + DTM boundary
        ftl = focal_tversky_loss(pred, gt_mask)
        dtm_l = boundary_loss_dtm(pred, gt_mask)
        sp = mean(edge)
        total = ftl + 0.4f0 * dtm_l + 0.01f0 * sp
        return total, ftl

    elseif loss_type == :dice_dtm
        # Simple Dice + DTM
        dl = dice_loss(pred, gt_mask)
        dtm_l = boundary_loss_dtm(pred, gt_mask)
        sp = mean(edge)
        total = dl + 0.4f0 * dtm_l + 0.01f0 * sp
        return total, dl

    elseif loss_type == :compound_dtm
        # Full compound loss
        total, dl, bl, dtm_l, sp = compound_loss(pred, edge, gt_mask)
        return total, dl

    else
        # Default: dice + bce
        dl = dice_loss(pred, gt_mask)
        ce = bce_loss(pred, gt_mask)
        total = dl + 0.5f0 * ce
        return total, dl
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# Training Loop for a Single Trial
# ══════════════════════════════════════════════════════════════════════════════

function run_trial(trial_config::Dict, train_ds, val_ds, dev; epochs=5, batch_size=2, n_per_epoch=60)
    trial_name = trial_config["name"]
    edge_type = Symbol(trial_config["edge_type"])
    pde_type = Symbol(trial_config["pde_type"])
    loss_type = Symbol(trial_config["loss_type"])

    println("\n" * "="^80)
    println("Trial: $trial_name")
    println("  Edge: $edge_type | PDE: $pde_type | Loss: $loss_type")
    println("="^80)

    rng = Xoshiro(42)
    model = HeatGDTFrameworkV2(;
        in_channels=3,
        edge_type=edge_type,
        pde_type=pde_type,
        K=20, dt=0.12f0, theta=0.012f0, tau=0.03f0
    )

    ps, st = Lux.setup(rng, model)
    ps = ComponentArray(ps) |> dev
    st = st |> dev

    n_params = length(ps)
    println("  Parameters: $n_params")

    # Check if model fits in GPU memory
    CUDA.reclaim()
    free_mem = CUDA.free_memory() / 1e9
    println("  Free GPU memory: $(round(free_mem, digits=2)) GB")

    opt = Optimisers.Adam(2e-4)
    st_opt = Optimisers.setup(opt, ps)

    best_val_dice = 0.0f0
    val_dice_history = Float32[]

    for epoch in 1:epochs
        st = Lux.trainmode(st)
        total_loss = 0.0f0
        total_dl = 0.0f0
        nb = 0

        indices = randperm(length(train_ds))[1:n_per_epoch]
        t0 = time()

        for i in 1:batch_size:length(indices)
            idx = indices[i:min(i+batch_size-1, length(indices))]
            x_cpu, seed_cpu, gt_cpu = get_batch(train_ds, idx)
            x = x_cpu |> dev
            seed = seed_cpu |> dev
            gt = gt_cpu |> dev

            (loss, st_new, dl), pb = try
                Zygote.pullback(ps) do p
                    outputs, st_n = model((x, seed), p, st)
                    loss_val, dl_val = compute_trial_loss(outputs, gt, loss_type)
                    return loss_val, st_n, dl_val
                end
            catch e
                println("  ⚠ Training error at epoch $epoch batch $nb: ", e)
                # Skip this batch on error
                continue
            end

            grads = try
                pb((1.0f0, nothing, nothing))[1]
            catch e
                println("  ⚠ Gradient error: ", e)
                continue
            end

            st = st_new
            st_opt, ps = Optimisers.update(st_opt, ps, grads)

            total_loss += loss
            total_dl += dl
            nb += 1
        end

        dt_epoch = time() - t0
        avg_loss = nb > 0 ? total_loss / nb : 0.0f0

        # ── Quick validation ──
        st = Lux.testmode(st)
        val_dice = 0.0f0
        n_val = 0
        max_val = 30

        for i in 1:batch_size:min(length(val_ds), max_val)
            idx = i:min(i+batch_size-1, min(length(val_ds), max_val))
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
                un = sum(pb) + sum(tb)
                dsc = (2.0f0 * inter + 1f-5) / (un + 1f-5)
                val_dice += dsc
                n_val += 1
            end
        end
        val_dice = n_val > 0 ? val_dice / n_val : 0.0f0
        push!(val_dice_history, val_dice)
        best_val_dice = max(best_val_dice, val_dice)

        @printf("  Epoch %02d [%.1fs] Loss: %.4f | Val Dice: %.4f (best: %.4f)\n",
                epoch, dt_epoch, avg_loss, val_dice, best_val_dice)
    end

    # ── Measure precomputation latency ──
    println("  Measuring latency...")
    x_test, seed_test, _ = get_batch(val_ds, [1])
    x_t = x_test |> dev
    seed_t = seed_test |> dev

    st = Lux.testmode(st)
    # Warmup
    model((x_t, seed_t), ps, st)
    CUDA.synchronize()

    # Time precomputation (edge network)
    precomp_times = Float64[]
    for _ in 1:10
        CUDA.synchronize()
        t0 = time()
        model((x_t, seed_t), ps, st)
        CUDA.synchronize()
        push!(precomp_times, (time() - t0) * 1000)
    end
    mean_precomp = mean(precomp_times)

    # Time interactive click (just heat diffusion, no edge net)
    D_test = CUDA.ones(Float32, 64, 64, 64, 1) .* 0.5f0
    CUDA.synchronize()
    click_times = Float64[]
    for _ in 1:20
        CUDA.synchronize()
        t0 = time()
        HeatDiffusion.heat_diffusion(seed_t, D_test, 20, 0.12f0)
        CUDA.synchronize()
        push!(click_times, (time() - t0) * 1000)
    end
    mean_click = mean(click_times)

    # ── Optimal slider Dice (sweep thresholds) ──
    println("  Computing optimal slider Dice...")
    optimal_dice = 0.0f0
    optimal_theta = 0.012f0

    n_slider = min(length(val_ds), 20)
    for theta_try in [0.005f0, 0.008f0, 0.010f0, 0.012f0, 0.015f0, 0.020f0, 0.025f0, 0.030f0, 0.035f0]
        slider_dice = 0.0f0
        n_s = 0
        for i in 1:batch_size:n_slider
            idx = i:min(i+batch_size-1, n_slider)
            x_cpu, seed_cpu, gt_cpu = get_batch(val_ds, idx)
            x = x_cpu |> dev
            seed = seed_cpu |> dev
            gt_arr = gt_cpu

            outputs, _ = model((x, seed), ps, st)
            heat_arr = Array(outputs.heat)
            for b in 1:size(heat_arr, 4)
                h = heat_arr[:, :, :, b]
                g = gt_arr[:, :, :, b] .>= 0.5f0
                p = h .>= theta_try
                inter = sum(p .& g)
                un = sum(p) + sum(g)
                dsc = (2.0f0 * inter + 1f-5) / (un + 1f-5)
                slider_dice += dsc
                n_s += 1
            end
        end
        avg_slider = n_s > 0 ? slider_dice / n_s : 0.0f0
        if avg_slider > optimal_dice
            optimal_dice = avg_slider
            optimal_theta = theta_try
        end
    end

    result = Dict(
        "name" => trial_name,
        "edge_type" => String(edge_type),
        "pde_type" => String(pde_type),
        "loss_type" => String(loss_type),
        "n_params" => n_params,
        "best_val_dice_fixed" => Float64(best_val_dice),
        "optimal_slider_dice" => Float64(optimal_dice),
        "optimal_theta" => Float64(optimal_theta),
        "val_dice_history" => Float64.(val_dice_history),
        "precomp_ms" => round(mean_precomp, digits=2),
        "click_ms" => round(mean_click, digits=2),
        "epochs" => epochs
    )

    @printf("  ✓ Trial %s: Fixed Dice=%.4f | Slider Dice=%.4f (θ=%.4f) | Precomp=%.1fms | Click=%.1fms\n",
            trial_name, best_val_dice, optimal_dice, optimal_theta, mean_precomp, mean_click)

    # Free GPU memory
    CUDA.reclaim()

    return result
end

# ══════════════════════════════════════════════════════════════════════════════
# MAIN: Define 12 Trial Configurations and Execute
# ══════════════════════════════════════════════════════════════════════════════

function main()
    dev = Lux.gpu_device()
    println("Device: ", dev)
    println("GPU: ", CUDA.name(CUDA.device()))
    println("VRAM: ", round(CUDA.total_memory() / 1e9, digits=2), " GB")

    train_manifest = "/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/train_single_lesion_manifest.json"
    val_manifest = "/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/val_single_lesion_manifest.json"

    train_ds = PatchHeatDataset(train_manifest)
    val_ds = PatchHeatDataset(val_manifest)
    println("Loaded $(length(train_ds)) single-lesion train samples / $(length(val_ds)) single-lesion val samples.")

    # ── 12 Trial Configurations (from approved plan) ──────────────────────
    trials = [
        Dict("name" => "H1_HED_PureDiff_DiceBCE",
             "edge_type" => "hed3d", "pde_type" => "pure_diffusion",
             "loss_type" => "dice_bce_edge"),

        Dict("name" => "H2_HED_ReacDiff_DiceDTM",
             "edge_type" => "hed3d", "pde_type" => "reaction_diffusion",
             "loss_type" => "dice_dtm_edge"),

        Dict("name" => "H3_HED_ReacDiff_FocalTversky",
             "edge_type" => "hed3d", "pde_type" => "reaction_diffusion",
             "loss_type" => "focal_tversky_dtm"),

        Dict("name" => "H4_DexiNed_PureDiff_DiceDTM",
             "edge_type" => "dexined3d", "pde_type" => "pure_diffusion",
             "loss_type" => "dice_dtm"),

        Dict("name" => "H5_DexiNed_ReacDiff_DiceDTM",
             "edge_type" => "dexined3d", "pde_type" => "reaction_diffusion",
             "loss_type" => "dice_dtm_edge"),

        Dict("name" => "H6_DexiNed_AdvecDiff",
             "edge_type" => "dexined3d", "pde_type" => "advection_diffusion",
             "loss_type" => "compound_dtm"),

        Dict("name" => "H7_DFrame_AdvecDiff_DTM",
             "edge_type" => "dframenet", "pde_type" => "advection_diffusion",
             "loss_type" => "dice_dtm"),

        Dict("name" => "H8_Hybrid25D_ReacDiff_DiceDTM",
             "edge_type" => "hybrid25d", "pde_type" => "reaction_diffusion",
             "loss_type" => "dice_dtm"),

        Dict("name" => "H9_Heavy_ReacDiff_CompoundDTM",
             "edge_type" => "heavy", "pde_type" => "reaction_diffusion",
             "loss_type" => "compound_dtm"),

        Dict("name" => "H10_Heavy_PureDiff_DiceDTM",
             "edge_type" => "heavy", "pde_type" => "pure_diffusion",
             "loss_type" => "dice_dtm"),

        Dict("name" => "H11_Deep_ReacDiff_DiceDTM",
             "edge_type" => "deep", "pde_type" => "reaction_diffusion",
             "loss_type" => "dice_dtm_edge"),

        Dict("name" => "H12_Heavy_FullRDA_Compound",
             "edge_type" => "heavy", "pde_type" => "full_rda",
             "loss_type" => "compound_dtm"),
    ]

    results = Dict[]
    result_path = "/mnt/big/project_ssd/project_ssd/semiautomatic/JuliaHELPNet/HeatGDT/hparam_results.json"

    for (i, trial) in enumerate(trials)
        println("\n\n" * "█"^80)
        println("  TRIAL $i / $(length(trials)): $(trial["name"])")
        println("█"^80)

        result = try
            run_trial(trial, train_ds, val_ds, dev; epochs=5, batch_size=2, n_per_epoch=60)
        catch e
            println("  ✗ TRIAL FAILED: ", e)
            Dict(
                "name" => trial["name"],
                "edge_type" => trial["edge_type"],
                "pde_type" => trial["pde_type"],
                "loss_type" => trial["loss_type"],
                "error" => string(e),
                "best_val_dice_fixed" => 0.0,
                "optimal_slider_dice" => 0.0,
            )
        end

        push!(results, result)

        # Save intermediate results after each trial
        open(result_path, "w") do io
            JSON.print(io, results, 2)
        end
        println("  Saved intermediate results → $(result_path)")

        # Force GC and GPU cleanup between trials
        GC.gc(true)
        CUDA.reclaim()
    end

    # ── Print Final Leaderboard ────────────────────────────────────────────
    println("\n\n" * "═"^100)
    println("                           HYPERPARAMETER SEARCH LEADERBOARD")
    println("═"^100)
    @printf("%-35s %10s %12s %12s %10s %8s\n",
            "Trial", "Params", "Fixed Dice", "Slider Dice", "Precomp", "Click")
    println("-"^100)

    # Sort by fixed dice descending
    sorted = sort(results, by=r -> -get(r, "best_val_dice_fixed", 0.0))
    for r in sorted
        @printf("%-35s %10s %12.4f %12.4f %8.1f ms %6.1f ms\n",
                get(r, "name", "?"),
                haskey(r, "n_params") ? string(r["n_params"]) : "ERR",
                get(r, "best_val_dice_fixed", 0.0),
                get(r, "optimal_slider_dice", 0.0),
                get(r, "precomp_ms", 0.0),
                get(r, "click_ms", 0.0))
    end
    println("═"^100)

    champion = sorted[1]
    println("\n🏆 CHAMPION: $(champion["name"])")
    println("   Fixed Dice: $(round(champion["best_val_dice_fixed"], digits=4))")
    println("   Slider Dice: $(round(champion["optimal_slider_dice"], digits=4))")
    println("   Config: edge=$(champion["edge_type"]) pde=$(champion["pde_type"]) loss=$(champion["loss_type"])")
    println("\n→ Next step: Train champion for 20 full epochs with train_champion.jl")

    # Save final
    open(result_path, "w") do io
        JSON.print(io, results, 2)
    end
    println("Final results saved → $(result_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
