module MedImagesIO

using MedImages
using HDF5

export load_ct_volume, load_organ_masks, save_nifti_mask

"""
    load_ct_volume(ct_path) -> (voxel_arr, spacing, origin, direction)
Loads CT volume natively via MedImages.jl / ITKIOWrapper.
"""
function load_ct_volume(ct_path::String)
    med_img = MedImages.load_image(ct_path)
    return (
        med_img.voxel_data,
        med_img.spacing,
        med_img.origin,
        med_img.direction
    )
end

"""
    load_organ_masks(seg_dir; required_names=nothing) -> Dict{String, Array{UInt8, 3}}
Loads all .nii.gz organ segmentations in `seg_dir` directly into Julia UInt8 3D arrays using MedImages.jl.
"""
function load_organ_masks(seg_dir::String; required_names::Union{Nothing, Set{String}, Vector{String}}=nothing)
    masks = Dict{String, Array{UInt8, 3}}()
    
    if !isdir(seg_dir)
        @warn "Directory $seg_dir does not exist."
        return masks
    end
    
    req_set = required_names === nothing ? nothing : Set{String}([lowercase(n) for n in required_names])
    files = filter(f -> endswith(f, ".nii.gz") || endswith(f, ".nii"), readdir(seg_dir))
    
    loaded_count = 0
    for fname in files
        name = replace(fname, r"\.nii(\.gz)?$" => "")
        name = replace(name, r"_primary_totalsegmentator$" => "")
        name = replace(name, r"_fallback_2_slicerdentalsegmentator$" => "")
        if req_set !== nothing && !(lowercase(name) in req_set)
            continue
        end
        
        full_path = joinpath(seg_dir, fname)
        try
            med_img = MedImages.load_image(full_path)
            j_arr = UInt8.(med_img.voxel_data .> 0)
            masks[name] = j_arr
            loaded_count += 1
            if loaded_count % 10 == 0
                println("    Loaded $loaded_count masks...")
            end
        catch e
            @warn "Failed to load $fname: $e"
        end
    end
    return masks
end

"""
    save_nifti_mask(arr, ref_path, out_path)
Saves a 3D UInt8 array `arr` as a NIfTI file with geometry copied from `ref_path` via MedImages.jl.
"""
function save_nifti_mask(arr::AbstractArray{UInt8, 3}, ref_path::String, out_path::String)
    mkpath(dirname(out_path))
    ref_img = MedImages.load_image(ref_path)
    out_img = MedImages.update_voxel_data(ref_img, arr)
    MedImages.create_nii_from_medimage(out_img, out_path)
end

end # module MedImagesIO
