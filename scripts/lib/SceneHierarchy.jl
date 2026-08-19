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
image onto the fixed grid, we need `M_moving_mod = inv(T_ITK) * M_old`.
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
studies. Each study is a tuple:
    (modality, orig_tp, date_str, ct_fname, pet_fname, mask_fname, node_name, tfm_fname)
"""
function parse_studies_from_hierarchy(data_dir)
    scene_json = joinpath(data_dir, "scene_hierarchy.json")
    if !isfile(scene_json)
        error("scene_hierarchy.json not found in $data_dir")
    end
    hierarchy = JSON.parse(read(scene_json, String))
    
    parsed_studies = []
    
    function extract_study(children, tfm_name)
        ct_name = ""
        pet_name = ""
        mask_name = ""
        modality = "PET"
        for child in children
            if child["type"] == "vtkMRMLLinearTransformNode"
                continue
            end
            name = child["name"]
            if child["type"] == "vtkMRMLScalarVolumeNode"
                if occursin("CT", name)
                    ct_name = name * ".nii.gz"
                elseif occursin("PET", name) || occursin("SUV", name) || occursin("SPECT", name) || occursin("NM", name)
                    pet_name = name * ".nii.gz"
                    if occursin("SPECT", name) || occursin("NM", name)
                        modality = "SPECT"
                    end
                end
            elseif child["type"] == "vtkMRMLSegmentationNode" && occursin("Lesions", name)
                mask_name = name * ".nii.gz"
                if !isfile(joinpath(data_dir, mask_name))
                    mask_name = name * ".seg.nrrd"
                end
            end
        end
        
        if isempty(ct_name) || isempty(pet_name)
            return nothing
        end
        
        m = match(r"_(\d+)$", replace(ct_name, ".nii.gz" => ""))
        orig_tp = m !== nothing ? parse(Int, m.captures[1]) : 0
        date_str = "$modality TP $orig_tp"
        
        return (modality, orig_tp, date_str, ct_name, pet_name, mask_name, replace(mask_name, r"\..*" => ""), tfm_name)
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
    
    sort!(parsed_studies, by = x -> (x[2], x[1]))
    return parsed_studies
end

end # module SceneHierarchy
