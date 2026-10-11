using Pkg; Pkg.activate("/mnt/big/project_ssd/project_ssd/semiautomatic/JuliaHELPNet")
using JSON
using NIfTI
using Statistics
using Random
using Printf
using CUDA

include("heat_diffuse.jl")
include("model.jl")
include("dataset.jl")

using .HeatDiffusion
using .HeatGDTModel
using .HeatGDTDataset

function compute_dice(pred, gt)
    intersection = sum(pred .* gt)
    union = sum(pred) + sum(gt)
    return union > 0 ? 2.0 * intersection / union : 1.0
end

function eval_canny(manifest_path, channel=:ct)
    println("\nEvaluating Canny ablation for channel: ", channel)
    items = JSON.parsefile(manifest_path)
    
    K = 20
    dt = 0.12f0
    tau = 0.03f0
    thetas = Float32[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.012, 0.05]
    
    # store sum of dice for each theta
    dice_sums = zeros(Float64, length(thetas))
    valid_count = 0
    
    for (i, item) in enumerate(items)
        canny_ct_path = item["canny_ct_path"]
        canny_pet_path = item["canny_pet_path"]
        gt_path = item["gt_path"]
        center = item["seed_xyz"]
        
        # Load Canny edges
        if channel == :ct
            edge_nii = niread(canny_ct_path)
            edge_map = Float32.(edge_nii.raw)
        elseif channel == :pet
            edge_nii = niread(canny_pet_path)
            edge_map = Float32.(edge_nii.raw)
        elseif channel == :fused
            ct_edge = Float32.(niread(canny_ct_path).raw)
            pet_edge = Float32.(niread(canny_pet_path).raw)
            edge_map = max.(ct_edge, pet_edge)
        end
        
        edge_map = clamp.(edge_map, 0.0f0, 1.0f0)
        
        gt_nii = niread(gt_path)
        gt_raw = gt_nii.raw
        
        cx, cy, cz = center
        cx, cy, cz = Int(cx), Int(cy), Int(cz)
        seed = (cx, cy, cz)
        
        sz = size(gt_raw)
        seed_mask = zeros(Float32, sz)
        seed_mask[cx, cy, cz] = 1.0f0
        
        gt_filtered = HeatGDTDataset.extract_single_component(UInt8.(gt_raw .> 0), seed)
        if sum(gt_filtered) == 0
            continue
        end
        
        valid_count += 1
        
        edge_gpu = cu(reshape(edge_map, sz..., 1, 1))
        seed_gpu = cu(reshape(seed_mask, sz..., 1))
        gt_gpu = cu(gt_filtered)
        
        # We increase gamma to 10.0 to make Canny edges more impermeable, as they are thin
        D = HeatGDTModel.compute_diffusivity(edge_gpu, 0.01f0, 10.0f0)
        
        u = HeatDiffusion.heat_diffusion(seed_gpu, D, K, dt)
        
        gt_cpu = Array(gt_gpu)
        u_cpu = Array(u)[:, :, :, 1]
        
        for (t_idx, th) in enumerate(thetas)
            # manual extract mask
            pred_cpu = 1.0 ./ (1.0 .+ exp.(-(u_cpu .- th) ./ tau))
            dice = compute_dice(pred_cpu .> 0.5, gt_cpu .> 0.5)
            dice_sums[t_idx] += dice
        end
        
        if valid_count % 100 == 0
            print("  [$valid_count] ")
            for (t_idx, th) in enumerate(thetas)
                @printf("th=%.4f: %.4f | ", th, dice_sums[t_idx]/valid_count)
            end
            println()
        end
    end
    
    println("Final Results for $channel:")
    for (t_idx, th) in enumerate(thetas)
        @printf("  Threshold %.4f -> Mean Dice: %.4f\n", th, dice_sums[t_idx]/valid_count)
    end
end

manifest = "/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/canny_val_manifest.json"
eval_canny(manifest, :ct)
eval_canny(manifest, :pet)
eval_canny(manifest, :fused)
