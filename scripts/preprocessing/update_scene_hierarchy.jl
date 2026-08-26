"""
    update_scene_hierarchy.jl

Reads the existing scene_hierarchy.json, scans anatomy_out_* directories,
and injects per-timepoint max_anatomy + skellytour entries.
Replaces old TS_all_Segmentation entries.

Usage:
    julia scripts/preprocessing/update_scene_hierarchy.jl [data_dir]
"""

using JSON

function get_anatomy_suffix(ct_name::String)
    bn = replace(ct_name, ".nii.gz" => "")
    if startswith(bn, "Fixed_CT_Volume_")
        idx = replace(bn, "Fixed_CT_Volume_" => "")
        return "fixed_ct_$(idx)"
    elseif startswith(bn, "SPECT_CT_Volume_")
        idx = replace(bn, "SPECT_CT_Volume_" => "")
        return "spect_ct_$(idx)"
    else
        return nothing
    end
end

function get_anatomy_entries(data_dir::String, ct_name::String)
    suffix = get_anatomy_suffix(ct_name)
    if suffix === nothing
        return []
    end
    
    anat_dir = joinpath(data_dir, "anatomy_out_$(suffix)")
    entries = []
    
    # max_anatomy
    max_path = joinpath(anat_dir, "max_anatomy.nii.gz")
    labels_path = joinpath(anat_dir, "max_anatomy_labels.json")
    if isfile(max_path) && isfile(labels_path)
        labels = JSON.parsefile(labels_path)
        push!(entries, Dict{String,Any}(
            "name" => "max_anatomy_$(suffix)",
            "type" => "vtkMRMLSegmentationNode",
            "source" => "anatomy_out_$(suffix)/max_anatomy.nii.gz",
            "labels" => "anatomy_out_$(suffix)/max_anatomy_labels.json",
            "segments_count" => length(labels)
        ))
        println("  ✅ max_anatomy_$(suffix): $(length(labels)) classes")
    else
        @warn "max_anatomy not found in $anat_dir"
    end
    
    # Skellytour subseg
    if isdir(anat_dir)
        subseg = filter(f -> occursin("subseg_postprocessed", f) && endswith(f, ".nii.gz"), readdir(anat_dir))
        if !isempty(subseg)
            push!(entries, Dict{String,Any}(
                "name" => "skellytour_$(suffix)",
                "type" => "vtkMRMLSegmentationNode",
                "source" => "anatomy_out_$(suffix)/$(subseg[1])"
            ))
            println("  ✅ skellytour_$(suffix)")
        else
            @warn "No Skellytour subseg in $anat_dir"
        end
    end
    
    return entries
end

function update_hierarchy(data_dir::String)
    scene_json = joinpath(data_dir, "scene_hierarchy.json")
    if !isfile(scene_json)
        error("scene_hierarchy.json not found in $data_dir")
    end
    
    hierarchy = JSON.parse(read(scene_json, String))
    println("Loaded scene_hierarchy.json with $(length(hierarchy)) top-level nodes")
    
    # Remove old TS_all_Segmentation entries and any stale anatomy entries at top level
    filter!(n -> !startswith(get(n, "name", ""), "TS_all_Segmentation") &&
                 !startswith(get(n, "name", ""), "max_anatomy_") &&
                 !startswith(get(n, "name", ""), "skellytour_"), hierarchy)
    
    # Find baseline CT at top level and add anatomy entries
    baseline_ct = ""
    for node in hierarchy
        name = get(node, "name", "")
        if (startswith(name, "Fixed_CT_Volume_") || startswith(name, "SPECT_CT_Volume_")) &&
           get(node, "type", "") == "vtkMRMLScalarVolumeNode"
            baseline_ct = name * ".nii.gz"
            break
        end
    end
    
    if !isempty(baseline_ct)
        println("\nBaseline CT: $baseline_ct")
        for entry in get_anatomy_entries(data_dir, baseline_ct)
            push!(hierarchy, entry)
        end
    end
    
    # Add anatomy entries for transformed studies (as children of transform nodes)
    for node in hierarchy
        if get(node, "type", "") == "vtkMRMLLinearTransformNode"
            children = get(node, "children", Dict{String,Any}[])
            ct_name = ""
            for child in children
                name = get(child, "name", "")
                if occursin("CT", name) && get(child, "type", "") == "vtkMRMLScalarVolumeNode"
                    ct_name = name * ".nii.gz"
                    break
                end
            end
            
            if !isempty(ct_name)
                # Remove old anatomy entries from children
                filter!(c -> !startswith(get(c, "name", ""), "max_anatomy_") &&
                             !startswith(get(c, "name", ""), "skellytour_") &&
                             !startswith(get(c, "name", ""), "TS_all"), children)
                
                println("\nTransform $(node["name"]) → CT: $ct_name")
                append!(children, get_anatomy_entries(data_dir, ct_name))
                node["children"] = children
            end
        end
    end
    
    # Write back with pretty printing
    open(scene_json, "w") do f
        JSON.print(f, hierarchy, 4)
    end
    println("\n✅ Updated scene_hierarchy.json")
end

# Main
data_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "..", "data", "pat_6_files")
update_hierarchy(data_dir)
