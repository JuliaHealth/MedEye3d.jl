# eval_pretrained.jl — Evaluate all pretrained edge models vs baseline
# Benchmarks: baseline (no pretrain), hed3d+pretrain, heavy+pretrain, hed3d_wide+pretrain
#
# Usage:
#   julia --project=../JuliaHELPNet eval_pretrained.jl

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

include("heat_diffuse.jl")
include("heat_diffuse_v2.jl")
include("model.jl")
include("dataset.jl")
include("losses.jl")

include("architectures/hed3d.jl")
include("architectures/hed3d_wide.jl")
include("architectures/hed3d_ultra.jl")
include("architectures/dexined3d.jl")
include("architectures/dframenet.jl")
include("architectures/hybrid25d.jl")

using .HeatDiffusion
using .HeatDiffusionV2
using .HeatGDTModel
using .HeatGDTDataset

function create_edge_net_eval(edge_type::Symbol, in_ch::Int=3)
    if edge_type == :hed3d
        return HED3D(in_ch)
    elseif edge_type == :hed3d_wide
        return HED3D_Wide(in_ch)
    elseif edge_type == :hed3d_ultra
        return HED3D_Ultra(in_ch)
    elseif edge_type == :heavy
        return HeatGDTModel.HeavyMultimodalEdgeNet3D(in_ch)
    else
        error("Unknown edge type: $edge_type")
    end
end

"""
    evaluate_model(checkpoint_path, edge_type, val_ds, dev; thresholds)

Evaluate a single model checkpoint across multiple thresholds.
Returns Dict with per-threshold Dice scores.
"""
function evaluate_model(checkpoint_path::String, edge_type::Symbol, val_ds, dev; pure_ct::Bool=false,
                        thresholds=[0.0001f0, 0.0005f0, 0.001f0, 0.005f0, 0.01f0, 0.05f0, 0.1f0],
                        K=40, dt=0.16f0, tau=0.0001f0)
    
    println("\n  Loading: $checkpoint_path")
    ckpt = load(checkpoint_path)
    
    edge_net = create_edge_net_eval(edge_type, 3)
    rng = Xoshiro(42)
    _, st_init = Lux.setup(rng, edge_net)
    
    # Load saved parameters
    ps_edge = ckpt["ps"]
    st_edge = ckpt["st"]
    
    # Handle both pretrained (edge params only) and full model checkpoints
    if haskey(ps_edge, :edge_net)
        ps_edge = ps_edge.edge_net
        st_edge = st_edge.edge_net
    end
    
    # To avoid scalar indexing issues with SubArrays on GPU,
    # copy the loaded weights into a freshly allocated ComponentArray
    ps_init_ca = ComponentArray(Lux.setup(rng, edge_net)[1])
    ps_init_ca .= ps_edge
    ps_edge = ps_init_ca |> dev
    st_edge = st_edge |> dev
    
    nparams = length(ps_edge)
    println("  Parameters: $nparams ($(round(nparams/1e6, digits=2))M)")
    
    # Evaluate at each threshold
    results = Dict{Float32, Dict{String, Float64}}()
    
    for theta in thresholds
        dice_sum = 0.0
        n_samples = 0
        
        for i in 1:length(val_ds)
            x_cpu, seed_cpu, gt_cpu, _ = get_batch(val_ds, [i])
            if pure_ct
                x_cpu[:, :, :, 2, :] .= 0.0f0
            end
            x_b = x_cpu |> dev
            seed_b = seed_cpu |> dev
            gt_b = gt_cpu |> dev
            
            st_test = Lux.testmode(st_edge)
            edge_map, _ = edge_net(x_b, ps_edge, st_test)
            D = HeatGDTModel.compute_diffusivity(edge_map)
            u = HeatDiffusion.heat_diffusion(seed_b, D, K, dt)
            pred = HeatDiffusion.extract_mask(u, theta, tau)
            
            dl = dice_loss(pred, gt_b)
            dice_sum += (1.0 - Float64(dl))
            n_samples += 1
        end
        
        avg_dice = dice_sum / max(n_samples, 1)
        results[theta] = Dict("dice" => avg_dice, "n_samples" => Float64(n_samples))
        @printf("    θ=%.4f → Dice=%.4f (n=%d)\n", theta, avg_dice, n_samples)
    end
    
    return results
end

function main()
    dev = Lux.gpu_device()
    println("═══════════════════════════════════════════════════")
    println("  Comprehensive Evaluation — All Edge Models")
    println("  Device: $dev")
    println("═══════════════════════════════════════════════════")

    val_manifest = "/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/val_single_lesion_manifest.json"
    val_ds = PatchHeatDataset(val_manifest)
    println("Val patches: $(length(val_ds))")

    thresholds = [0.0001f0, 0.0005f0, 0.001f0, 0.005f0, 0.01f0, 0.05f0, 0.1f0]

    # Define all conditions to evaluate
    # Evaluate only the ultra model since others are done
    conditions = [
        ("HED3D-Ultra + pretrain", "champion_pretrained_hed3d_ultra.jld2", :hed3d_ultra),
    ]

    all_results = isfile("pretrained_eval_results.json") ? JSON.parsefile("pretrained_eval_results.json") : Dict{String, Any}()

    for (name, ckpt_path, edge_type) in conditions
        if !isfile(ckpt_path)
            println("  ⚠ Checkpoint not found: $ckpt_path (skipping)")
            continue
        end
        
        for pure_ct in [false, true]
            modality = pure_ct ? "Pure CT" : "PET/CT"
            eval_name = "$name ($modality)"
            println("
══ $eval_name ══")
            
            try
                results = evaluate_model(ckpt_path, edge_type, val_ds, dev; thresholds=thresholds, pure_ct=pure_ct)
                
                # Find best threshold
                best_theta = argmax(Dict(k => v["dice"] for (k, v) in results))
                best_dice = results[best_theta]["dice"]
                
                all_results[eval_name] = Dict(
                    "checkpoint" => ckpt_path,
                    "edge_type" => String(edge_type),
                    "modality" => modality,
                    "best_theta" => Float64(best_theta),
                    "best_dice" => best_dice,
                    "per_threshold" => Dict(string(k) => v for (k, v) in results)
                )
                
                @printf("  ★ Best: θ=%.4f → Dice=%.4f
", best_theta, best_dice)
            catch e
                println("  ✗ Error evaluating $eval_name: $e")
                @error "Evaluation failed" exception=(e, catch_backtrace())
            end
            
            GC.gc()
            CUDA.reclaim()
        end
    end

    # Summary table
    println("

" * "=" ^ 85)
    println("  SUMMARY — All Conditions")
    println("=" ^ 85)
    @printf("%-35s | %10s | %8s | %10s | %10s
", "Condition", "Modality", "Params", "Best θ", "Best Dice")
    println("-" ^ 85)
    
    # We want to iterate the keys in the order we evaluated
    for (name, ckpt_path, edge_type) in conditions
        for pure_ct in [false, true]
            modality = pure_ct ? "Pure CT" : "PET/CT"
            eval_name = "$name ($modality)"
            if haskey(all_results, eval_name)
                r = all_results[eval_name]
                @printf("%-35s | %10s | %8s | %10.4f | %10.4f
", 
                        name, modality, r["edge_type"], r["best_theta"], r["best_dice"])
            end
        end
    end
    println("=" ^ 85)


    # Save results
    results_path = "pretrained_eval_results.json"
    open(results_path, "w") do f
        JSON.print(f, all_results, 2)
    end
    println("\n✅ Results saved to: $results_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
