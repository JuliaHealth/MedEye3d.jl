using Pkg; Pkg.activate("/mnt/big/project_ssd/project_ssd/semiautomatic/JuliaHELPNet")

using LuxCUDA
using Lux
using NNlib
using Random
using ComponentArrays
using JLD2
using Printf
using Statistics
using CUDA
using JSON

include("heat_diffuse.jl")
include("heat_diffuse_v2.jl")
include("model.jl")
include("dataset.jl")

include("architectures/hed3d.jl")
include("architectures/dexined3d.jl")
include("architectures/dframenet.jl")
include("architectures/hybrid25d.jl")

using .HeatDiffusion
using .HeatDiffusionV2
using .HeatGDTModel
using .HeatGDTDataset

struct HeatGDTFrameworkV2{M} <: Lux.AbstractLuxContainerLayer{(:edge_net,)}
    edge_net::M
    K::Int
    dt::Float32
    theta::Float32
    tau::Float32
    pde_type::Symbol
    edge_type::Symbol
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
    net_out, st_edge = m.edge_net(x, ps.edge_net, st.edge_net)

    if m.edge_type == :dframenet
        edge_map = net_out.edges
        direction_field = net_out.direction
    else
        edge_map = net_out
        direction_field = nothing
    end

    D = HeatGDTModel.compute_diffusivity(edge_map)

    if m.pde_type == :pure_diffusion
        u = HeatDiffusion.heat_diffusion(seed_mask, D, m.K, m.dt)
    elseif m.pde_type == :reaction_diffusion
        beta = edge_map .* 2.0f0
        v = CUDA.zeros(Float32, size(D, 1), size(D, 2), size(D, 3), 3, size(D, 4))
        u = HeatDiffusionV2.heat_diffusion_v2(seed_mask, D, beta, v, m.K, m.dt)
    elseif m.pde_type == :advection_diffusion
        beta = CUDA.zeros(Float32, size(D)...)
        if direction_field !== nothing
            u = HeatDiffusionV2.heat_diffusion_v2(seed_mask, D, beta, direction_field, m.K, m.dt)
        else
            v = CUDA.zeros(Float32, size(D, 1), size(D, 2), size(D, 3), 3, size(D, 4))
            u = HeatDiffusionV2.heat_diffusion_v2(seed_mask, D, beta, v, m.K, m.dt)
        end
    elseif m.pde_type == :full_rda
        beta = edge_map .* 2.0f0
        if direction_field !== nothing
            u = HeatDiffusionV2.heat_diffusion_v2(seed_mask, D, beta, direction_field, m.K, m.dt)
        else
            v = CUDA.zeros(Float32, size(D, 1), size(D, 2), size(D, 3), 3, size(D, 4))
            u = HeatDiffusionV2.heat_diffusion_v2(seed_mask, D, beta, v, m.K, m.dt)
        end
    else
        u = HeatDiffusion.heat_diffusion(seed_mask, D, m.K, m.dt)
    end

    pred_mask = HeatDiffusion.extract_mask(u, m.theta, m.tau)

    return (pred=pred_mask, heat=u, edge=edge_map, D=D), (edge_net=st_edge,)
end

function benchmark_champion()
    dev = Lux.gpu_device()
    println("Loading Champion Model...")
    
    saved = load("champion_model.jld2")
    config = saved["config"]
    edge_type = Symbol(config["edge_type"])
    pde_type = Symbol(config["pde_type"])
    
    model = HeatGDTFrameworkV2(;
        in_channels=3,
        edge_type=edge_type,
        pde_type=pde_type,
        K=40, dt=0.16f0, theta=0.001f0, tau=0.0001f0
    )
    
    ps_cpu = ComponentArray(saved["ps"])
    st_cpu = saved["st"]
    ps = ps_cpu |> dev
    st = st_cpu |> dev
    st = Lux.testmode(st)
    
    val_manifest = "/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/val_single_lesion_manifest.json"
    val_ds = PatchHeatDataset(val_manifest)
    println("Benchmarking on $(length(val_ds)) validation patches...")
    
    batch_size = 1
    total_dice = 0.0f0
    slider_dice = 0.0f0
    latencies = Float64[]
    n_samples = 0
    
    optimal_theta = Float32(config["optimal_theta"])
    println("Using dynamic slider threshold: ", optimal_theta)
    
    for i in 1:batch_size:length(val_ds)
        batch_idx = i:min(i+batch_size-1, length(val_ds))
        x_cpu, seed_cpu, gt_cpu = get_batch(val_ds, batch_idx)
        x_b = x_cpu |> dev
        seed_b = seed_cpu |> dev
        
        if i == 1
            model((x_b, seed_b), ps, st)
            CUDA.synchronize()
        end
        
        t0 = time_ns()
        out, _ = model((x_b, seed_b), ps, st)
        CUDA.synchronize()
        t1 = time_ns()
        
        push!(latencies, (t1 - t0) / 1e6)
        
        heat_arr = Array(out.heat)
        gt_arr = gt_cpu .>= 0.5f0
        
        for b in 1:length(batch_idx)
            h = heat_arr[:, :, :, 1, b]
            p_fixed = h .>= 0.001f0
            inter_f = sum(p_fixed .& gt_arr[:,:,:,b])
            un_f = sum(p_fixed) + sum(gt_arr[:,:,:,b])
            dsc_f = (2.0f0 * inter_f + 1f-5) / (un_f + 1f-5)
            total_dice += dsc_f
            
            p_slider = h .>= optimal_theta
            inter_s = sum(p_slider .& gt_arr[:,:,:,b])
            un_s = sum(p_slider) + sum(gt_arr[:,:,:,b])
            dsc_s = (2.0f0 * inter_s + 1f-5) / (un_s + 1f-5)
            slider_dice += dsc_s
            
            n_samples += 1
        end
    end
    
    avg_fixed = total_dice / n_samples
    avg_slider = slider_dice / n_samples
    avg_lat = mean(latencies[2:end]) # exclude warmup
    
    println("\n" * "="^50)
    println("CHAMPION BENCHMARK RESULTS")
    println("="^50)
    @printf("Fixed Dice (θ=0.001):  %.4f\n", avg_fixed)
    @printf("Slider Dice (θ=%.3f): %.4f\n", optimal_theta, avg_slider)
    @printf("Latency (ms/patch):    %.1f ms\n", avg_lat)
    println("="^50)
end

if abspath(PROGRAM_FILE) == @__FILE__
    benchmark_champion()
end
