#!/usr/bin/env julia
# Quick evaluation script: loads saved masks, resolves overlaps, evaluates Dice
using NIfTI
using JSON

include(joinpath(@__DIR__, "overlap_resolution.jl"))
include(joinpath(@__DIR__, "Evaluator.jl"))
using .Evaluator

function main()
    case_dir = "/mnt/big/project_ssd/project_ssd/lymph_node_rules/data/processed_cases_restored/FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat44"
    output_dir = joinpath(case_dir, "lymph_node_outputs")
    gold_nrrd_path = isfile(joinpath(case_dir, "Pat44_Combined_All_Lymph_Nodes.seg.nrrd")) ?
                     joinpath(case_dir, "Pat44_Combined_All_Lymph_Nodes.seg.nrrd") :
                     joinpath(case_dir, "All_Lymph_Node_Areas.seg.nrrd")
    spacing = (0.9765625, 0.9765625, 3.0)
    
    println("Loading generated masks from $output_dir...")
    gen_masks = Dict{String, Any}()
    for subdir in readdir(output_dir)
        sd = joinpath(output_dir, subdir)
        if !isdir(sd) continue end
        for f in readdir(sd)
            if endswith(f, ".nii.gz")
                name = replace(f, ".nii.gz" => "")
                try
                    ni = niread(joinpath(sd, f))
                    arr = UInt8.(ni.raw .> 0)
                    gen_masks[name] = arr
                catch e
                    println("Warning: Failed to load $f: $e")
                end
            end
        end
    end
    # Also check top-level
    for f in readdir(output_dir)
        if endswith(f, ".nii.gz")
            name = replace(f, ".nii.gz" => "")
            try
                ni = niread(joinpath(output_dir, f))
                arr = UInt8.(ni.raw .> 0)
                gen_masks[name] = arr
            catch e
                println("Warning: Failed to load $f: $e")
            end
        end
    end
    println("Loaded $(length(gen_masks)) masks")
    
    # Skip overlap resolution - masks already had overlaps resolved in GPU pipeline
    # println("\nResolving Overlaps...")
    # rules_dict = load_rules(joinpath(@__DIR__, "../../../jsons"))
    # resolve_overlaps_cpu!(gen_masks, rules_dict; spacing=spacing)
    
    println("\nEvaluating Dice against Gold Standard...")
    if isfile(gold_nrrd_path)
        avg_dice = evaluate_lymph_nodes(gen_masks, gold_nrrd_path)
        println("Average Dice: $avg_dice")
    else
        println("Gold standard not found at $gold_nrrd_path")
    end
end

main()
