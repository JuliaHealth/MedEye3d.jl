#!/usr/bin/env julia
# Targeted re-computation of _meta_/segment_names.json and _meta_/organ_mapping
# Fixes the study-index keying bug and computes organ mapping for all TPs
using Pkg
Pkg.activate("/workspaces/MedEye3d.jl")

using HDF5, JSON

# Get data directory from args or use default
data_dir = length(ARGS) >= 1 ? ARGS[1] : "/workspaces/MedEye3d.jl/data/pat_6_files"
h5_path = joinpath(data_dir, "preprocessed_volumes.h5")

if !isfile(h5_path)
    error("HDF5 file not found: $h5_path")
end

println("=== Re-computing segment_names and organ_mapping ===")
println("  HDF5: $h5_path")

# Load scene hierarchy
scene_path = joinpath(data_dir, "scene_hierarchy.json")
if !isfile(scene_path)
    error("scene_hierarchy.json not found in $data_dir")
end

# We need to know the studies list. Parse it from scene hierarchy.
include("/workspaces/MedEye3d.jl/scripts/lib/SceneHierarchy.jl")
using .SceneHierarchy: parse_studies_from_hierarchy
studies = parse_studies_from_hierarchy(data_dir)
println("  Found $(length(studies)) studies")

h5file = h5open(h5_path, "r+")

try
    # ── 1. Fix segment_names.json ──
    h_tree = JSON.parse(read(scene_path, String))
    extracted_sn = Dict{String, Dict{String, String}}()
    
    function _walk_sn_fixed(nodes, studies_list)
        for nd in nodes
            nd_name = get(nd, "name", "")
            if get(nd, "type", "") == "vtkMRMLSegmentationNode" && haskey(nd, "segments")
                seg_base = replace(nd_name, ".nii.gz" => "")
                matched_idx = -1
                for (si, study) in enumerate(studies_list)
                    mask_base = replace(study[6], ".nii.gz" => "")
                    if seg_base == mask_base
                        matched_idx = si - 1  # 0-indexed study index
                        break
                    end
                end
                if matched_idx >= 0
                    segs = nd["segments"]
                    if segs isa AbstractVector
                        target = get!(extracted_sn, string(matched_idx), Dict{String, String}())
                        for (s_idx, s_item) in enumerate(segs)
                            target[string(s_idx)] = (s_item isa AbstractDict) ? get(s_item, "name", "Segment $s_idx") : string(s_item)
                        end
                    end
                end
            end
            if haskey(nd, "children")
                _walk_sn_fixed(nd["children"], studies_list)
            end
        end
    end
    
    _walk_sn_fixed(h_tree, studies)
    
    println("  segment_names keys: $(sort(collect(keys(extracted_sn))))")
    for k in sort(collect(keys(extracted_sn)))
        v = extracted_sn[k]
        println("    TP $k: $(length(v)) segments")
        for (sid, sname) in sort(collect(v), by=x->parse(Int, x[1]))
            println("      $sid: $sname")
        end
    end
    
    if haskey(h5file, "_meta_/segment_names.json")
        delete_object(h5file, "_meta_/segment_names.json")
    end
    h5file["_meta_/segment_names.json"] = JSON.json(extracted_sn)
    println("  Wrote _meta_/segment_names.json ($(length(extracted_sn)) TPs)")
    
    # ── 2. Fix organ_mapping for ALL TPs ──
    # Load TotalSegmentator atlas and names
    using MedEye3d
    LA = MedEye3d.LesionAssociation
    
    ts_names = Dict{Int, String}()
    if haskey(h5file, "_meta_/max_anatomy_labels.json")
        ts_raw = JSON.parse(read(h5file["_meta_/max_anatomy_labels.json"]))
        for (k, v) in ts_raw
            ts_names[parse(Int, k)] = v
        end
    end
    println("  Loaded $(length(ts_names)) TotalSegmentator labels")
    
    # Load the TotalSegmentator atlas (from BASELINE group)
    ts_atlas = nothing
    if haskey(h5file, "BASELINE/max_anatomy.nii.gz")
        ts_atlas = read(h5file["BASELINE/max_anatomy.nii.gz"])
        println("  Loaded TotalSegmentator atlas: $(size(ts_atlas))")
    end
    
    if ts_atlas !== nothing && !isempty(ts_names)
        all_organ_mappings = Dict{String, Dict{String, String}}()
        
        for (si, study) in enumerate(studies)
            tp_idx = si - 1
            mask_fname = study[6]
            group = study[8] == "" ? "BASELINE" : "TFM_" * study[8]
            h5_key = "$group/$mask_fname"
            
            if haskey(h5file, h5_key)
                mask_raw = read(h5file[h5_key])
                mask_f32 = Float32.(mask_raw)
                
                organ_mapping = LA.map_lesions_to_organs(mask_f32, ts_atlas, ts_names)
                
                tp_map = Dict{String, String}()
                for (k, v) in organ_mapping
                    tp_map[string(k)] = v
                end
                all_organ_mappings[string(tp_idx)] = tp_map
                println("    TP $tp_idx ($mask_fname): $(length(organ_mapping)) lesions mapped")
                for (lid, organ) in sort(collect(tp_map), by=x->parse(Int, x[1]))
                    println("      Lesion $lid: $organ")
                end
            else
                println("    TP $tp_idx ($mask_fname): mask not found at $h5_key, skipping")
            end
        end
        
        if haskey(h5file, "_meta_/organ_mapping")
            delete_object(h5file, "_meta_/organ_mapping")
        end
        h5file["_meta_/organ_mapping"] = JSON.json(all_organ_mappings)
        println("  Wrote _meta_/organ_mapping ($(length(all_organ_mappings)) TPs)")
    else
        println("  SKIPPED organ_mapping: atlas or labels not available")
    end

finally
    close(h5file)
end

println("=== Done! ===")
