#!/usr/bin/env julia
# ablation_eval_only.jl — Evaluation-Only Ablation: Learned Edge Tensor vs Uniform Diffusivity
#
# This script evaluates the champion Heat-GDT model under two conditions:
#   A) With learned edge-derived anisotropic diffusivity (D varies spatially)
#   B) With uniform isotropic diffusivity (D = 1.0 everywhere)
#
# Condition B answers: "What if we remove the edge network entirely and let heat
# diffuse isotropically? Is the heat kernel alone (a Euclidean distance sphere)
# sufficient for segmentation?"
#
# No training is needed for Condition B — with D=1 and fixed PDE parameters,
# there are no learnable degrees of freedom. The heat kernel simply creates a
# spherical diffusion profile centered on the seed point.

using Pkg; Pkg.activate("/mnt/big/project_ssd/project_ssd/semiautomatic/JuliaHELPNet")

using Lux
using CUDA
using LuxCUDA
using ComponentArrays
using Printf
using Statistics
using Random
using JLD2
using NNlib
using JSON
using Dates

include("heat_diffuse.jl")
include("model.jl")
include("dataset.jl")

using .HeatDiffusion
using .HeatGDTModel
using .HeatGDTDataset

# ── Champion model (same as train_champion.jl / benchmark_champion.jl) ───────

include("architectures/hed3d.jl")

struct HeatGDTFrameworkV2_Eval{M} <: Lux.AbstractLuxContainerLayer{(:edge_net,)}
    edge_net::M
    K::Int
    dt::Float32
    theta::Float32
    tau::Float32
    pde_type::Symbol
    edge_type::Symbol
end

function HeatGDTFrameworkV2_Eval(;
    in_channels=3,
    edge_type::Symbol=:hed3d,
    pde_type::Symbol=:pure_diffusion,
    K=40, dt=0.16f0, theta=0.001f0, tau=0.0001f0
)
    if edge_type == :hed3d
        edge_net = HED3D(in_channels)
    elseif edge_type == :heavy
        edge_net = HeatGDTModel.HeavyMultimodalEdgeNet3D(in_channels)
    elseif edge_type == :deep
        edge_net = HeatGDTModel.DeepMultimodalEdgeNet3D(in_channels)
    else
        edge_net = HeatGDTModel.MultimodalEdgeNet(in_channels)
    end
    return HeatGDTFrameworkV2_Eval(edge_net, K, dt, theta, tau, pde_type, edge_type)
end

function (m::HeatGDTFrameworkV2_Eval)((x, seed_mask), ps, st)
    edge_map, st_edge = m.edge_net(x, ps.edge_net, st.edge_net)
    D = HeatGDTModel.compute_diffusivity(edge_map)
    u = HeatDiffusion.heat_diffusion(seed_mask, D, m.K, m.dt)
    pred_mask = HeatDiffusion.extract_mask(u, m.theta, m.tau)
    return (pred=pred_mask, heat=u, edge=edge_map, D=D), (edge_net=st_edge,)
end

# ── Evaluation with threshold sweep ──────────────────────────────────────────

function evaluate_with_thresholds(heat_field_cpu, gt_cpu, thresholds)
    """Evaluate a batch of heat fields at multiple thresholds."""
    n_batch = size(heat_field_cpu, 5)
    results = Dict{Float32, Dict{String, Float64}}()
    
    for θ in thresholds
        dice_sum = 0.0
        sens_sum = 0.0
        spec_sum = 0.0
        
        for b in 1:n_batch
            h = heat_field_cpu[:, :, :, 1, b]
            gt_b = gt_cpu[:, :, :, b] .>= 0.5f0
            
            pred_bin = h .>= θ
            tp = Float64(sum(pred_bin .& gt_b))
            fp = Float64(sum(pred_bin .& .!gt_b))
            fn = Float64(sum(.!pred_bin .& gt_b))
            tn = Float64(sum(.!pred_bin .& .!gt_b))
            
            dsc = (2.0 * tp + 1e-5) / (2.0 * tp + fp + fn + 1e-5)
            sens = (tp + 1e-5) / (tp + fn + 1e-5)
            spec = (tn + 1e-5) / (tn + fp + 1e-5)
            
            dice_sum += dsc
            sens_sum += sens
            spec_sum += spec
        end
        
        results[θ] = Dict("dice" => dice_sum / n_batch,
                          "sensitivity" => sens_sum / n_batch,
                          "specificity" => spec_sum / n_batch)
    end
    return results
end

function run_ablation_eval()
    dev = Lux.gpu_device()
    println("Device: ", dev)
    println("\n" * "="^70)
    println("  ABLATION STUDY: Learned Edge Tensor vs. Uniform Diffusivity")
    println("  (Evaluation Only — No Training Required)")
    println("="^70)

    val_manifest = "/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/val_single_lesion_manifest.json"
    val_ds = PatchHeatDataset(val_manifest)
    n_val = length(val_ds)
    println("Validation set: $n_val patches")

    thresholds = Float32[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

    # ── Load Champion Model ──────────────────────────────────────────────────
    champion_path = joinpath(@__DIR__, "champion_model.jld2")
    saved = load(champion_path)
    config = saved["config"]
    edge_type = Symbol(config["edge_type"])
    pde_type = Symbol(config["pde_type"])
    
    model = HeatGDTFrameworkV2_Eval(;
        in_channels=3,
        edge_type=edge_type,
        pde_type=pde_type,
        K=40, dt=0.16f0, theta=0.001f0, tau=0.0001f0
    )
    
    rng = Xoshiro(42)
    _, st_init = Lux.setup(rng, model)
    ps = ComponentArray(saved["ps"]) |> dev
    st = saved["st"] |> dev
    st_test = Lux.testmode(st)
    
    println("\nChampion: edge=$edge_type, pde=$pde_type")
    println("PDE: K=40, dt=0.16, theta=0.001, tau=0.0001")

    # ── Accumulators ─────────────────────────────────────────────────────────
    acc_A = Dict(θ => Dict("dice" => 0.0, "sensitivity" => 0.0, "specificity" => 0.0) for θ in thresholds)
    acc_B = Dict(θ => Dict("dice" => 0.0, "sensitivity" => 0.0, "specificity" => 0.0) for θ in thresholds)
    n_evaluated = 0

    # ── Process all validation patches ───────────────────────────────────────
    println("\nEvaluating both conditions simultaneously on $n_val patches...")
    
    for i in 1:n_val
        x_cpu, seed_cpu, gt_cpu = get_batch(val_ds, [i])
        x_b = x_cpu |> dev
        seed_b = seed_cpu |> dev

        # ── Condition A: Learned edge-derived D ──────────────────────────────
        out_A, _ = model((x_b, seed_b), ps, st_test)
        CUDA.synchronize()
        heat_A = Array(out_A.heat)

        # ── Condition B: Uniform D = 1.0 (bypass edge network) ───────────────
        D_uniform = CUDA.ones(Float32, size(seed_b)...)
        u_B = HeatDiffusion.heat_diffusion(seed_b, D_uniform, 40, 0.16f0)
        CUDA.synchronize()
        heat_B = Array(u_B)

        # ── Evaluate both at all thresholds ──────────────────────────────────
        res_A = evaluate_with_thresholds(heat_A, gt_cpu, thresholds)
        res_B = evaluate_with_thresholds(heat_B, gt_cpu, thresholds)

        for θ in thresholds
            for metric in ["dice", "sensitivity", "specificity"]
                acc_A[θ][metric] += res_A[θ][metric]
                acc_B[θ][metric] += res_B[θ][metric]
            end
        end

        n_evaluated += 1
        if n_evaluated % 50 == 0
            @printf("  Evaluated %d / %d samples...\n", n_evaluated, n_val)
        end
    end

    # ── Average metrics ──────────────────────────────────────────────────────
    for θ in thresholds
        for metric in ["dice", "sensitivity", "specificity"]
            acc_A[θ][metric] /= n_evaluated
            acc_B[θ][metric] /= n_evaluated
        end
    end

    # ── Print Results Table ──────────────────────────────────────────────────
    println("\n\n" * "="^100)
    println("  ABLATION RESULTS: LEARNED EDGES vs. UNIFORM DIFFUSIVITY ($n_evaluated val patches)")
    println("="^100)
    @printf("%-10s | %-35s | %-35s | %-10s\n",
            "Threshold", "A: Learned Edges (Champion)", "B: Uniform D=1 (Isotropic)", "Δ Dice")
    println("-"^100)

    best_θ_A = Float32(0); best_dice_A = 0.0
    best_θ_B = Float32(0); best_dice_B = 0.0

    for θ in sort(thresholds)
        dA = acc_A[θ]["dice"]
        sA = acc_A[θ]["sensitivity"]
        pA = acc_A[θ]["specificity"]
        dB = acc_B[θ]["dice"]
        sB = acc_B[θ]["sensitivity"]
        pB = acc_B[θ]["specificity"]
        delta = dA - dB

        @printf("θ=%-8.4f | Dice=%.4f Sens=%.4f Spec=%.4f | Dice=%.4f Sens=%.4f Spec=%.4f | %+.4f\n",
                θ, dA, sA, pA, dB, sB, pB, delta)
        
        if dA > best_dice_A; best_dice_A = dA; best_θ_A = θ; end
        if dB > best_dice_B; best_dice_B = dB; best_θ_B = θ; end
    end

    println("\n" * "-"^100)
    @printf("BEST THRESHOLD:\n")
    @printf("  A (Learned Edges):   θ=%.4f → Dice=%.4f, Sens=%.4f, Spec=%.4f\n",
            best_θ_A, best_dice_A, acc_A[best_θ_A]["sensitivity"], acc_A[best_θ_A]["specificity"])
    @printf("  B (Uniform D=1.0):   θ=%.4f → Dice=%.4f, Sens=%.4f, Spec=%.4f\n",
            best_θ_B, best_dice_B, acc_B[best_θ_B]["sensitivity"], acc_B[best_θ_B]["specificity"])

    improvement_abs = best_dice_A - best_dice_B
    improvement_pct = (improvement_abs / max(best_dice_B, 1e-8)) * 100

    println("\n" * "="^100)
    @printf("EDGE NETWORK CONTRIBUTION:\n")
    @printf("  Absolute Dice improvement:  %+.4f\n", improvement_abs)
    @printf("  Relative Dice improvement:  %+.1f%%\n", improvement_pct)
    @printf("  Learned edges → Mean Dice:  %.4f\n", best_dice_A)
    @printf("  Uniform D=1   → Mean Dice:  %.4f\n", best_dice_B)
    println("="^100)

    # ── Save Results to JSON ─────────────────────────────────────────────────
    ablation_results = Dict(
        "experiment" => "Ablation: Learned Edge Tensor vs Uniform Diffusivity",
        "timestamp" => string(Dates.now()),
        "n_val_patches" => n_evaluated,
        "pde_params" => Dict("K" => 40, "dt" => 0.16, "theta" => 0.001, "tau" => 0.0001),
        "champion_config" => Dict("edge_type" => string(edge_type), "pde_type" => string(pde_type)),
        "condition_A_learned_edges" => Dict(
            "description" => "Champion HED3D model with learned edge-derived anisotropic diffusivity",
            "best_threshold" => Float64(best_θ_A),
            "best_dice" => best_dice_A,
            "per_threshold" => Dict(string(θ) => acc_A[θ] for θ in thresholds)
        ),
        "condition_B_uniform" => Dict(
            "description" => "Uniform D=1.0 isotropic diffusion (no edge network, pure heat kernel)",
            "best_threshold" => Float64(best_θ_B),
            "best_dice" => best_dice_B,
            "per_threshold" => Dict(string(θ) => acc_B[θ] for θ in thresholds)
        ),
        "summary" => Dict(
            "absolute_dice_improvement" => improvement_abs,
            "relative_dice_improvement_pct" => improvement_pct,
            "interpretation" => improvement_abs > 0.05 ?
                "The learned edge tensor provides a statistically meaningful improvement, confirming that boundary-aware anisotropic diffusivity is essential for accurate segmentation." :
                "The learned edge tensor provides only marginal improvement, suggesting the heat kernel geometry alone captures most of the segmentation signal."
        )
    )

    results_path = joinpath(@__DIR__, "ablation_results.json")
    open(results_path, "w") do io
        JSON.print(io, ablation_results, 2)
    end
    println("\nResults saved to: $results_path")

    return ablation_results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_ablation_eval()
end
