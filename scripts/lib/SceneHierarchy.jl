"""
    SceneHierarchy

Shared module for parsing 3D Slicer scene_hierarchy.json files and applying
ITK affine transforms to MedImage spatial metadata. Used by both the interactive
app (`run_interactive_mrb.jl`) and the preprocessing pipeline (`preprocess_dataset.jl`).
"""
module SceneHierarchy

using JSON
using LinearAlgebra
using MedImages

export parse_studies_from_hierarchy, apply_transform_to_medimage, parse_tfm

"""
    parse_tfm(tfm_path::String) -> Matrix{Float64}

Parse a 3D Slicer .tfm file (ITK affine transform) and return a 4×4 homogeneous matrix.
Returns identity if the path is empty or the file cannot be parsed.
"""
function parse_tfm(tfm_path::String)
    if isempty(tfm_path) || !isfile(tfm_path)
        return Matrix{Float64}(I, 4, 4)
    end
    content = read(tfm_path, String)
    m = match(r"Parameters:\s*([\d\s\.\-e\+]+)", content)
    if m === nothing
        return Matrix{Float64}(I, 4, 4)
    end
    params = parse.(Float64, split(strip(m.captures[1])))
    if length(params) == 12
        T = Matrix{Float64}(I, 4, 4)
        T[1:3, 1:3] = reshape(params[1:9], 3, 3)'
        T[1:3, 4] = params[10:12]
        return T
    else
        return Matrix{Float64}(I, 4, 4)
    end
end

"""
    apply_transform_to_medimage(img::MedImage, T_ITK::Matrix{Float64}) -> MedImage

Apply an ITK affine transform to the spatial metadata (origin, spacing, direction)
of a MedImage. The voxel data is unchanged — this prepares the image for subsequent
resampling via `MedImages.resample_to_image`.

In ITK convention, T maps Fixed → Moving physical space. To resample the moving
image onto the fixed grid, we map `M_new = inv(T_ITK) * M_old`.
"""
function apply_transform_to_medimage(img::MedImage, T_ITK::Matrix{Float64})
    if T_ITK == Matrix{Float64}(I, 4, 4)
        return img
    end
    old_spacing = img.spacing
    old_dir = transpose(reshape(collect(img.direction), 3, 3))
    old_orig = img.origin
    
    M_old = zeros(Float64, 4, 4)
    for i in 1:3, j in 1:3
        M_old[i, j] = old_dir[i, j] * old_spacing[j]
    end
    for i in 1:3
        M_old[i, 4] = old_orig[i]
    end
    M_old[4, 4] = 1.0
    
    M_new = inv(T_ITK) * M_old
    
    new_orig = (M_new[1, 4], M_new[2, 4], M_new[3, 4])
    new_spacing = zeros(Float64, 3)
    new_dir = zeros(Float64, 3, 3)
    
    for j in 1:3
        col = M_new[1:3, j]
        s = norm(col)
        new_spacing[j] = s
        new_dir[:, j] = col / s
    end
    
    new_dir_tuple = Tuple(transpose(new_dir))
    return MedImages.update_voxel_and_spatial_data(img, img.voxel_data, new_orig, Tuple(new_spacing), new_dir_tuple)
end

"""
    parse_studies_from_hierarchy(data_dir::String) -> Vector{Tuple}

Parse a `scene_hierarchy.json` file and return a chronologically sorted list of
studies. Each study is a 12-element tuple:
    (modality, orig_tp, date_str, ct_fname, pet_fname, mask_fname, node_name, tfm_fname, ts_name,
     max_anatomy_source, max_anatomy_labels, skellytour_source)
"""
function parse_studies_from_hierarchy(data_dir)
    scene_json = joinpath(data_dir, "scene_hierarchy.json")
    if !isfile(scene_json)
        error("scene_hierarchy.json not found in $data_dir")
    end
    hierarchy = JSON.parse(read(scene_json, String))
    
    # Load metadata.json if available to extract actual acquisition dates and modalities
    meta_dates = Dict{String, String}()
    meta_modalities = Dict{String, String}()
    meta_json_path = joinpath(data_dir, "metadata.json")
    if isfile(meta_json_path)
        try
            meta_json = JSON.parsefile(meta_json_path)
            for item in meta_json
                for (k, v) in item
                    if v isa Dict
                        # Formatted date YYYY-MM-DD
                        d_str = if length(k) >= 8 && all(isdigit, k[1:8])
                            prefix = "$(k[1:4])-$(k[5:6])-$(k[7:8])"
                            suffix = length(k) > 8 ? " (" * replace(strip(c -> c == '_', k[9:end]), "_" => " ") * ")" : ""
                            prefix * suffix
                        else
                            k
                        end
                        for sub_k in keys(v)
                            sub_dict = v[sub_k]
                            if sub_dict isa Dict && haskey(sub_dict, "name")
                                v_name = sub_dict["name"]
                                meta_dates[v_name] = d_str
                                if haskey(sub_dict, "Modality")
                                    meta_modalities[v_name] = sub_dict["Modality"]
                                end
                            end
                        end
                    end
                end
            end
        catch e
            @warn "Could not parse metadata.json dates: $e"
        end
    end

    parsed_studies = []
    
    function extract_study(children, tfm_name)
        ct_name = ""
        pet_name = ""
        mask_name = ""
        ts_name = ""
        max_anatomy_source = ""
        max_anatomy_labels = ""
        skellytour_source = ""
        modality = "PET"
        for child in children
            if child["type"] == "vtkMRMLLinearTransformNode"
                continue
            end
            name = child["name"]
            child_mod = get(child, "modality", get(child, "Modality", get(meta_modalities, name, "")))
            if !isempty(child_mod)
                modality = child_mod
            end
            if child["type"] == "vtkMRMLScalarVolumeNode"
                # Check for functional / NM / PET / SPECT first
                if occursin("NM", name) || occursin("PET", name) || occursin("SUV", name)
                    pet_name = name * ".nii.gz"
                    if isempty(child_mod) && (occursin("NM", name) || occursin("SPECT", name))
                        modality = "SPECT"
                    end
                elseif occursin("CT", name) || occursin("MR", name) || occursin("T2", name) || occursin("ADC", name) || occursin("DWI", name) || occursin("T1", name)
                    ct_name = name * ".nii.gz"
                    if isempty(child_mod)
                        if occursin("ADC", name)
                            modality = "ADC"
                        elseif occursin("DWI", name) || occursin("BVAL", name)
                            modality = "DWI"
                        elseif occursin("T1", name)
                            modality = "T1"
                        elseif occursin("MR", name) || occursin("T2", name)
                            modality = "T2"
                        end
                    end
                end
            elseif child["type"] == "vtkMRMLSegmentationNode"
                if occursin("Lesions", name)
                    mask_name = name * ".nii.gz"
                    if !isfile(joinpath(data_dir, mask_name))
                        mask_name = name * ".seg.nrrd"
                    end
                elseif startswith(name, "max_anatomy_")
                    max_anatomy_source = get(child, "source", "")
                    max_anatomy_labels = get(child, "labels", "")
                elseif startswith(name, "skellytour_")
                    skellytour_source = get(child, "source", "")
                elseif occursin("TS_all", name) || occursin("TotalSegmentator", name)
                    ts_name = name * ".seg.nrrd"
                    if !isfile(joinpath(data_dir, ts_name))
                        ts_name = name * ".nii.gz"
                    end
                end
            end
        end
        
        if isempty(ct_name) || isempty(pet_name)
            return nothing
        end
        
        ct_base = replace(ct_name, ".nii.gz" => "")
        parts = split(ct_base, "_")
        orig_tp = tryparse(Int, parts[end])
        if orig_tp === nothing; orig_tp = 0; end
        
        # Look up exact date and modality from metadata if possible
        if haskey(meta_modalities, ct_base)
            modality = meta_modalities[ct_base]
        end
        pet_base = replace(pet_name, ".nii.gz" => "")
        date_str = get(meta_dates, ct_base, get(meta_dates, pet_base, "$modality TP $orig_tp"))
        
        # Strip extensions from mask name without regex
        mask_base = replace(replace(mask_name, ".seg.nrrd" => ""), ".nii.gz" => "")
        return (modality, orig_tp, date_str, ct_name, pet_name, mask_name, mask_base, tfm_name, ts_name,
                max_anatomy_source, max_anatomy_labels, skellytour_source)
    end
    
    b = extract_study(hierarchy, "")
    if b !== nothing; push!(parsed_studies, b); end
    
    for node in hierarchy
        if node["type"] == "vtkMRMLLinearTransformNode"
            tfm_name = node["name"]
            tfm_file = tfm_name * ".tfm"
            s = extract_study(get(node, "children", []), tfm_file)
            if s !== nothing; push!(parsed_studies, s); end
        end
    end
    
    # Sort chronologically by date prefix (YYYY-MM-DD), with orig_tp tie-breaker
    sort!(parsed_studies, by = x -> ((length(x[3]) >= 10 && isdigit(x[3][1])) ? x[3][1:10] : "9999-99-99", x[2]))
    return parsed_studies
end

end # module SceneHierarchy
