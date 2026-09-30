module Evaluator

using NIfTI
using Printf

export evaluate_lymph_nodes, load_gold_standard_nrrd

"""
    load_gold_standard_nrrd(nrrd_path::String) -> Dict{String, Array{UInt8, 3}}
Reads a multi-segment NRRD (or single NRRD) and returns a dictionary of segment masks.
"""
function load_gold_standard_nrrd(nrrd_path::String)
    if !isfile(nrrd_path)
        error("Gold standard file not found: $nrrd_path")
    end
    
    # Read NRRD header and raw data
    lines = String[]
    header_bytes = 0
    open(nrrd_path, "r") do f
        while !eof(f)
            line = readline(f)
            header_bytes += length(line) + 1
            if isempty(strip(line))
                break
            end
            push!(lines, line)
        end
    end
    
    # Parse header for dimensions, encoding, segment names
    sizes = Int[]
    dimension = 3
    encoding = "raw"
    type_str = "uint8"
    segment_names = Dict{Int, String}()
    segment_label_values = Dict{Int, Int}()
    
    for line in lines
        if startswith(line, "sizes:")
            sizes = parse.(Int, split(split(line, ":")[2]))
        elseif startswith(line, "dimension:")
            dimension = parse(Int, strip(split(line, ":")[2]))
        elseif startswith(line, "encoding:")
            encoding = strip(split(line, ":")[2])
        elseif startswith(line, "type:")
            type_str = strip(split(line, ":")[2])
        elseif occursin("Segment", line) && occursin("Name:=", line)
            m = match(r"Segment(\d+)_Name:=(.+)", line)
            if m !== nothing
                seg_idx = parse(Int, m.captures[1])
                seg_name = strip(m.captures[2])
                segment_names[seg_idx] = seg_name
            end
        elseif occursin("Segment", line) && occursin("LabelValue:=", line)
            m = match(r"Segment(\d+)_LabelValue:=(\d+)", line)
            if m !== nothing
                seg_idx = parse(Int, m.captures[1])
                label_val = parse(Int, m.captures[2])
                segment_label_values[seg_idx] = label_val
            end
        end
    end
    
    # Read raw array data
    raw_bytes = read(nrrd_path)
    h_end = findfirst(b"\n\n", raw_bytes)
    data_start = h_end !== nothing ? last(h_end) + 1 : (findfirst(b"\r\n\r\n", raw_bytes) !== nothing ? last(findfirst(b"\r\n\r\n", raw_bytes)) + 1 : header_bytes + 1)
    data_bytes = raw_bytes[data_start:end]
    
    T = type_str == "uint8" || type_str == "unsigned char" ? UInt8 : (type_str == "int16" || type_str == "short" ? Int16 : UInt8)
    arr = reinterpret(T, data_bytes)
    
    segments = Dict{String, Array{UInt8, 3}}()
    
    if dimension == 4
        if length(sizes) == 4
            if sizes[1] == length(segment_names) || sizes[1] < 100
                n_segs = sizes[1]
                arr_4d = reshape(arr, (sizes[1], sizes[2], sizes[3], sizes[4]))
                for (idx, name) in segment_names
                    if idx + 1 <= n_segs
                        segments[name] = UInt8.(arr_4d[idx + 1, :, :, :] .> 0)
                    end
                end
            else
                n_segs = sizes[4]
                arr_4d = reshape(arr, (sizes[1], sizes[2], sizes[3], sizes[4]))
                for (idx, name) in segment_names
                    if idx + 1 <= n_segs
                        segments[name] = UInt8.(arr_4d[:, :, :, idx + 1] .> 0)
                    end
                end
            end
        end
    elseif dimension == 3
        arr_3d = reshape(arr, (sizes[1], sizes[2], sizes[3]))
        if isempty(segment_names)
            base_name = replace(basename(nrrd_path), ".seg.nrrd" => "", ".nrrd" => "")
            segments[base_name] = UInt8.(arr_3d .> 0)
        else
            for (idx, name) in segment_names
                lbl = get(segment_label_values, idx, idx + 1)
                segments[name] = UInt8.(arr_3d .== lbl)
            end
        end
    end
    
    return segments
end

"""
    evaluate_lymph_nodes(generated_masks, gold_standard_nrrd_path)
Compares all generated masks against the gold standard segments and prints a detailed report.
"""
function evaluate_lymph_nodes(generated_masks::Dict{String, <:Any}, gold_nrrd_path::String)
    gold_segments = load_gold_standard_nrrd(gold_nrrd_path)
    
    println("="^85)
    @printf("%-45s | %-10s | %-10s | %-8s\n", "Segment Name", "Gold Vox", "Gen Vox", "Dice")
    println("-"^85)
    
    dices = Float64[]
    matched_count = 0
    
    function norm_k(s)
        r = lowercase(s)
        r = replace(r, " " => "", "_" => "")
        r = replace(r, "paraoesophageal" => "paraesophageal")
        return r
    end
    
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
        "Neck_Level_IIa_Upper_Jugular_left" => ["Neck_Level_IIa_Upper_Jugular_Left", "Neck_Level_IIa_Upper_Jugular_left", "Neck_Level_IIa_Left", "Neck_Level_II_Upper_Jugular_left", "Neck_Level_II_Left"],
        "Neck_Level_IIa_Upper_Jugular_right" => ["Neck_Level_IIa_Upper_Jugular_Right", "Neck_Level_IIa_Upper_Jugular_right", "Neck_Level_IIa_Right", "Neck_Level_II_Upper_Jugular_right", "Neck_Level_II_Right"],
        "Neck_Level_IIb_Upper_Jugular_left" => ["Neck_Level_IIb_Upper_Jugular_Left", "Neck_Level_IIb_Upper_Jugular_left", "Neck_Level_IIb_Left", "Neck_Level_II_Left"],
        "Neck_Level_IIb_Upper_Jugular_right" => ["Neck_Level_IIb_Upper_Jugular_Right", "Neck_Level_IIb_Upper_Jugular_right", "Neck_Level_IIb_Right", "Neck_Level_II_Right"],
        "Neck_Level_V_Upper_Posterior_Triangle_left" => ["Neck_Level_V_Posterior_Triangle_left", "Thoracic_Supraclavicular_Left", "Neck_Level_V_Posterior_Triangle_and_Supraclavicular_left"],
        "Neck_Level_V_Upper_Posterior_Triangle_right" => ["Neck_Level_V_Posterior_Triangle_right", "Thoracic_Supraclavicular_Right", "Neck_Level_V_Posterior_Triangle_and_Supraclavicular_right"],
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
        "Thoracic_Station_Prepericardial_Left" => ["Thoracic_Station_Prepericardial_Left", "Thoracic_Prepericardial_Left", "Thoracic_Prepericardial_left", "Thoracic_Prepericardial"],
        "Thoracic_Station_Prepericardial_Right" => ["Thoracic_Station_Prepericardial_Right", "Thoracic_Prepericardial_Right", "Thoracic_Prepericardial_right", "Thoracic_Prepericardial"],
        "Abdominal_Common_Iliac_Left" => ["Abdominal_Common_Iliac_Left", "Abdominal_Common_Iliac_left", "Abdominal_Common_Iliac"],
        "Abdominal_Common_Iliac_Right" => ["Abdominal_Common_Iliac_Right", "Abdominal_Common_Iliac_right", "Abdominal_Common_Iliac"],
        "Abdominal_Paraaortic" => ["Abdominal_Retroperitoneal", "Abdominal_Paraaortic", "Abdominal_Paraaortic_Left", "Abdominal_Paraaortic_Right"]
    )
    
    for (seg_name, gold_mask) in sort(collect(gold_segments), by=x->x[1])
        gold_vox = count(gold_mask .> 0)
        
        gen_mask = nothing
        target_norm = norm_k(seg_name)
        
        # 1. Direct match
        for (k, v) in generated_masks
            if norm_k(k) == target_norm
                gen_mask = v
                break
            end
        end
        
        # 2. Try aliases
        if gen_mask === nothing && haskey(aliases, seg_name)
            for alias in aliases[seg_name]
                for (k, v) in generated_masks
                    if norm_k(k) == norm_k(alias)
                        gen_mask = v
                        break
                    end
                end
                if gen_mask !== nothing break end
            end
        end
        
        # 3. Bilateral split fallback
        if gen_mask !== nothing && (endswith(seg_name, "_left") || endswith(seg_name, "_Left") ||
                                     endswith(seg_name, "_right") || endswith(seg_name, "_Right"))
            is_left = endswith(lowercase(seg_name), "_left")
            midline = size(gen_mask, 1) ÷ 2
            split_mask = copy(gen_mask)
            if is_left
                split_mask[1:midline, :, :] .= 0
            else
                split_mask[midline+1:end, :, :] .= 0
            end
            split_vox = count(split_mask .> 0)
            full_vox = count(gen_mask .> 0)
            if split_vox > 0 && split_vox < full_vox && !haskey(generated_masks, seg_name)
                gen_mask = split_mask
            end
        end
        
        if gen_mask !== nothing
            gen_vox = count(gen_mask .> 0)
            intersect_vox = count((gold_mask .> 0) .& (gen_mask .> 0))
            dice = (gold_vox + gen_vox) > 0 ? (2.0 * intersect_vox) / (gold_vox + gen_vox) : 1.0
            push!(dices, dice)
            matched_count += 1
            @printf("%-45s | %-10d | %-10d | %0.4f\n", seg_name, gold_vox, gen_vox, dice)
        else
            @printf("%-45s | %-10d | %-10s | %-8s\n", seg_name, gold_vox, "MISSING", "N/A")
        end
    end
    
    println("-"^85)
    avg_dice = isempty(dices) ? 0.0 : sum(dices) / length(gold_segments)
    @printf("Average Dice: %0.4f across %d matched segments (%d total in gold standard).\n",
            avg_dice, matched_count, length(gold_segments))
    println("="^85)
    
    return avg_dice
end

end # module
