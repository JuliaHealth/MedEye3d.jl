module HeuristicsEngine

using MedImages
using Statistics

export compute_heuristics

"""
Compute heuristics for a given lesion based on its CT mask.
This ports the logic from slicer_lesion_text_extension/src/core/HeuristicsEngine.py
and slicer_lesion_text_extension/docs/Lesion_Inference_Logic.md
"""
function compute_heuristics(
    ct_image::MedImage,
    pet_image::MedImage,
    lesion_mask::Array{UInt8, 3};
    bone_mask::Union{Array{UInt8, 3}, Nothing} = nothing,
    organ_mask::Union{Array{UInt8, 3}, Nothing} = nothing
)::Dict{String, String}
    
    heuristics = Dict{String, String}()
    
    # Extract coordinates where mask is > 0
    indices = findall(x -> x > 0, lesion_mask)
    if isempty(indices)
        return heuristics
    end
    
    # 1. Volume & Sphericity & Size
    # Very basic approximation
    voxel_volume = ct_image.spacing[1] * ct_image.spacing[2] * ct_image.spacing[3]
    volume_mm3 = length(indices) * voxel_volume
    
    # Surface area approximation (very rough bounding box surface area)
    coords_x = [idx[1] for idx in indices]
    coords_y = [idx[2] for idx in indices]
    coords_z = [idx[3] for idx in indices]
    
    dx = (maximum(coords_x) - minimum(coords_x) + 1) * ct_image.spacing[1]
    dy = (maximum(coords_y) - minimum(coords_y) + 1) * ct_image.spacing[2]
    dz = (maximum(coords_z) - minimum(coords_z) + 1) * ct_image.spacing[3]
    
    surface_area = 2 * (dx*dy + dy*dz + dx*dz)
    
    sphericity = (pi^(1.0/3.0) * (6 * volume_mm3)^(2.0/3.0)) / max(surface_area, 1.0)
    
    # 2. HU Density
    hu_values = Float32[]
    for idx in indices
        push!(hu_values, ct_image.voxel_data[idx])
    end
    
    mean_hu = mean(hu_values)
    std_hu = std(hu_values)
    
    # 3. Apply Heuristics mappings
    if mean_hu > 800
        heuristics["Inner Texture / Density / Attenuation"] = "Sclerotic / Blastic"
    elseif mean_hu < 30
        heuristics["Inner Texture / Density / Attenuation"] = "Lytic / Lucent"
    elseif std_hu > 300
        heuristics["Inner Texture / Density / Attenuation"] = "Mixed Lytic & Sclerotic"
    end
    
    if sphericity > 0.8
        heuristics["Lesion Shape"] = "Round"
    elseif 0.5 < sphericity <= 0.8
        heuristics["Lesion Shape"] = "Oval / Bean-Shaped"
    else
        heuristics["Lesion Shape"] = "Irregular"
    end
    
    # 4. Anatomic Location
    if bone_mask !== nothing
        bone_overlap = sum((lesion_mask .> 0) .& (bone_mask .> 0)) / length(indices)
        if bone_overlap > 0.5
            heuristics["Anatomic Location"] = "Axial Skeleton" # fallback to generic skeleton mapping
        end
    end
    
    if organ_mask !== nothing
        organ_overlap = sum((lesion_mask .> 0) .& (organ_mask .> 0)) / length(indices)
        if organ_overlap > 0.5
            heuristics["Anatomic Location"] = "Solid Organ / Viscera"
        end
    end
    
    return heuristics
end

end # module
