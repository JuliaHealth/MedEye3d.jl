module Diagnostics

using HDF5
using NIfTI
using Statistics
using Printf

include("Evaluator.jl")

const GOLD_NRRD = "/mnt/big/project_ssd/project_ssd/lymph_node_rules/data/processed_cases_restored/FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat44/Pat44_Combined_All_Lymph_Nodes.seg.nrrd"
const OUTPUT_DIR = "/mnt/big/project_ssd/project_ssd/lymph_node_rules/data/processed_cases_restored/FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat44/lymph_node_outputs"

function compute_stats(mask::AbstractArray)
    idx = findall(mask .> 0)
    if isempty(idx)
        return (0, (0, 0, 0, 0, 0, 0), (0.0, 0.0, 0.0))
    end
    count = length(idx)
    min_x = minimum(c[1] for c in idx); max_x = maximum(c[1] for c in idx)
    min_y = minimum(c[2] for c in idx); max_y = maximum(c[2] for c in idx)
    min_z = minimum(c[3] for c in idx); max_z = maximum(c[3] for c in idx)
    c_x = mean(c[1] for c in idx)
    c_y = mean(c[2] for c in idx)
    c_z = mean(c[3] for c in idx)
    return (count, (min_x, max_x, min_y, max_y, min_z, max_z), (c_x, c_y, c_z))
end

function run_diagnostics()
    println("=============================================================================================================")
    println("                               SYSTEMATIC SEGMENT-BY-SEGMENT DIAGNOSTICS                                     ")
    println("=============================================================================================================")
    
    gold_masks = Evaluator.load_gold_standard_nrrd(GOLD_NRRD)
    
    # Collect all generated nii.gz files
    gen_files = Dict{String, String}()
    for (root, _, files) in walkdir(OUTPUT_DIR)
        for f in files
            if endswith(f, ".nii.gz")
                name = replace(f, ".nii.gz" => "")
                gen_files[lowercase(name)] = joinpath(root, f)
            end
        end
    end
    
    println(@sprintf("%-38s | %-8s | %-8s | %-6s | %-12s | %-12s | %-12s | %-5s", 
        "Segment Name", "GoldVox", "GenVox", "Ratio", "Gold BBox Z", "Gen BBox Z", "Centroid Shift", "Dice"))
    println("-"^125)
    
    aliases = Dict{String, Vector{String}}(
        "Axillary_I_left" => ["Axillary_Level_I_Left", "Thoracic_Axillary_Nodes_Left", "Axillary_I_left", "Axillary_Level_I_left"],
        "Axillary_I_right" => ["Axillary_Level_I_Right", "Thoracic_Axillary_Nodes_Right", "Axillary_I_right", "Axillary_Level_I_right"],
        "Axillary_II_left" => ["Axillary_Level_II_Left", "Axillary_II_left"],
        "Axillary_II_right" => ["Axillary_Level_II_Right", "Axillary_II_right"],
        "Axillary_III_left" => ["Axillary_Level_III_Left", "Axillary_III_left"],
        "Axillary_III_right" => ["Axillary_Level_III_Right", "Axillary_III_right"],
        "Axillary_Rotter_left" => ["Axillary_Rotter_Left", "Axillary_Rotter_left"],
        "Axillary_Rotter_right" => ["Axillary_Rotter_Right", "Axillary_Rotter_right"],
        "Deep_Inguinal_left" => ["Deep_Inguinal_Left", "Deep_Inguinal_left"],
        "Deep_Inguinal_right" => ["Deep_Inguinal_Right", "Deep_Inguinal_right"],
        "Superficial_Inguinal_left" => ["Superficial_Inguinal_Left", "Superficial_Inguinal_left"],
        "Superficial_Inguinal_right" => ["Superficial_Inguinal_Right", "Superficial_Inguinal_right"],
        "Neck_Level_IIa_Upper_Jugular_left" => ["Neck_Level_II_Upper_Jugular_left", "Neck_Level_IIa_Left", "Neck_Level_II_Left", "Neck_Level_IIa_Upper_Jugular_Left"],
        "Neck_Level_IIa_Upper_Jugular_right" => ["Neck_Level_II_Upper_Jugular_right", "Neck_Level_IIa_Right", "Neck_Level_II_Right", "Neck_Level_IIa_Upper_Jugular_Right"],
        "Neck_Level_IIb_Upper_Jugular_left" => ["Neck_Level_II_Upper_Jugular_left", "Neck_Level_IIb_Left", "Neck_Level_II_Left", "Neck_Level_IIb_Upper_Jugular_Left"],
        "Neck_Level_IIb_Upper_Jugular_right" => ["Neck_Level_II_Upper_Jugular_right", "Neck_Level_IIb_Right", "Neck_Level_II_Right", "Neck_Level_IIb_Upper_Jugular_Right"],
        "Neck_Level_V_Upper_Posterior_Triangle_left" => ["Neck_Level_V_Posterior_Triangle_left", "Thoracic_Supraclavicular_Left", "Neck_Level_V_Posterior_Triangle_and_Supraclavicular_left", "Neck_Level_V_Posterior_Triangle_Left"],
        "Neck_Level_V_Upper_Posterior_Triangle_right" => ["Neck_Level_V_Posterior_Triangle_right", "Thoracic_Supraclavicular_Right", "Neck_Level_V_Posterior_Triangle_and_Supraclavicular_right", "Neck_Level_V_Posterior_Triangle_Right"],
        "Neck_Level_Xb_Occipital_left" => ["Neck_Nuchal_left", "Neck_Nuchal_Left", "Neck_Nuchal"],
        "Neck_Level_Xb_Occipital_right" => ["Neck_Nuchal_right", "Neck_Nuchal_Right", "Neck_Nuchal"],
        "Neck_Parotid_left" => ["Neck_Parotid_Nodes_Left", "Neck_Parotid_Left"],
        "Neck_Parotid_right" => ["Neck_Parotid_Nodes_Right", "Neck_Parotid_Right"],
        "Thoracic_Station_5_Subaortic_Left" => ["Thoracic_Station_5_Subaortic_Left", "Thoracic_Station_5_Subaortic", "Thoracic_Station_5_Subaortic_left"],
        "Thoracic_Station_3A_Prevascular_Left" => ["Thoracic_Station_3A_Prevascular_Left", "Thoracic_Station_3A_Prevascular_left", "Thoracic_Station_3A_Prevascular"],
        "Thoracic_Station_3A_Prevascular_Right" => ["Thoracic_Station_3A_Prevascular_Right", "Thoracic_Station_3A_Prevascular_right", "Thoracic_Station_3A_Prevascular"],
        "Thoracic_Station_3P_Retrotracheal_left" => ["Thoracic_Station_3P_Retrotracheal_Left", "Thoracic_Station_3P_Retrotracheal_left", "Thoracic_Station_3P_Retrotracheal"],
        "Thoracic_Station_3P_Retrotracheal_right" => ["Thoracic_Station_3P_Retrotracheal_Right", "Thoracic_Station_3P_Retrotracheal_right", "Thoracic_Station_3P_Retrotracheal"],
        "Thoracic_Mammary_left" => ["Thoracic_Mammary_Left", "Thoracic_Mammary_left", "Thoracic_Mammary"],
        "Thoracic_Mammary_right" => ["Thoracic_Mammary_Right", "Thoracic_Mammary_right", "Thoracic_Mammary"],
        "Thoracic_Station_8_Paraoesophageal_left" => ["Thoracic_Station_8_Paraoesophageal_Left", "Thoracic_Station_8_Paraoesophageal_left", "Thoracic_Station_8_Paraesophageal_Left"],
        "Thoracic_Station_8_Paraoesophageal_right" => ["Thoracic_Station_8_Paraoesophageal_Right", "Thoracic_Station_8_Paraoesophageal_right", "Thoracic_Station_8_Paraesophageal_Right"],
        "Abdominal_Paraaortic" => ["Abdominal_Retroperitoneal", "Abdominal_Paraaortic"]
    )

    results = []
    
    for (gold_name, gold_m) in sort(collect(gold_masks), by=x->x[1])
        gen_m = nothing
        
        cands = [gold_name]
        if haskey(aliases, gold_name)
            append!(cands, aliases[gold_name])
        end
        # Add basic case variants
        for c in copy(cands)
            push!(cands, replace(c, "_left" => "_Left", "_right" => "_Right"))
            push!(cands, replace(c, "_left" => "", "_right" => ""))
        end
        
        matched_path = nothing
        for c in cands
            if haskey(gen_files, lowercase(c))
                matched_path = gen_files[lowercase(c)]
                break
            end
        end
        
        if matched_path !== nothing
            nii = niread(matched_path)
            gen_m = nii.raw .> 0
        end
        
        g_cnt, g_bbox, g_cent = compute_stats(gold_m)
        if gen_m !== nothing
            p_cnt, p_bbox, p_cent = compute_stats(gen_m)
            inter = count((gold_m .> 0) .& (gen_m .> 0))
            dice = (g_cnt + p_cnt) > 0 ? 2.0 * inter / (g_cnt + p_cnt) : 0.0
            ratio = g_cnt > 0 ? p_cnt / g_cnt : 0.0
            shift = (p_cent[1] - g_cent[1], p_cent[2] - g_cent[2], p_cent[3] - g_cent[3])
            shift_str = @sprintf("(%+.0f,%+.0f,%+.0f)", shift[1], shift[2], shift[3])
            g_z = @sprintf("[%d..%d]", g_bbox[5], g_bbox[6])
            p_z = @sprintf("[%d..%d]", p_bbox[5], p_bbox[6])
            
            push!(results, (gold_name, g_cnt, p_cnt, ratio, g_z, p_z, shift_str, dice))
        else
            push!(results, (gold_name, g_cnt, 0, 0.0, "N/A", "MISSING", "N/A", 0.0))
        end
    end
    
    # Sort by gold volume descending
    sort!(results, by=x->x[2], rev=true)
    
    for r in results
        println(@sprintf("%-38s | %8d | %8d | %6.2f | %-12s | %-12s | %-12s | %5.3f",
            r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]))
    end
    println("="^125)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    Diagnostics.run_diagnostics()
end
