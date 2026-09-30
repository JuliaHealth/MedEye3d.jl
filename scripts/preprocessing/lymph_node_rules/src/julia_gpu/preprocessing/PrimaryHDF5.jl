module PrimaryHDF5

using HDF5
using JSON
using Dates
using LinearAlgebra
using NIfTI
using ..Landmarks

export build_primary_hdf5, load_primary_masks, load_primary_landmarks, load_primary_metadata,
       load_primary_ct, is_primary_hdf5_complete, save_primary_landmark!

function safe_read_nii(path::String)
    ni = NIfTI.niread(path)
    arr = ni.raw
    
    # Calculate ITK LPS coordinates from NIfTI sform matrix
    if ni.header.sform_code > 0
        S = Float64[
            ni.header.srow_x[1] ni.header.srow_x[2] ni.header.srow_x[3] ni.header.srow_x[4];
            ni.header.srow_y[1] ni.header.srow_y[2] ni.header.srow_y[3] ni.header.srow_y[4];
            ni.header.srow_z[1] ni.header.srow_z[2] ni.header.srow_z[3] ni.header.srow_z[4];
            0.0 0.0 0.0 1.0
        ]
        T_lps_ras = Float64[
            -1.0  0.0  0.0  0.0;
             0.0 -1.0  0.0  0.0;
             0.0  0.0  1.0  0.0;
             0.0  0.0  0.0  1.0
        ]
        M_lps = T_lps_ras * S
        col1 = M_lps[1:3, 1]
        col2 = M_lps[1:3, 2]
        col3 = M_lps[1:3, 3]
        sp1 = norm(col1)
        sp2 = norm(col2)
        sp3 = norm(col3)
        spacing = (sp1, sp2, sp3)
        origin = (M_lps[1, 4], M_lps[2, 4], M_lps[3, 4])
        dir_mat = [col1/sp1 col2/sp2 col3/sp3]
        direction = Tuple(Float64(x) for x in vec(dir_mat))
    else
        spacing = (Float64(ni.header.pixdim[2]), Float64(ni.header.pixdim[3]), Float64(ni.header.pixdim[4]))
        origin = (-Float64(ni.header.qoffset_x), -Float64(ni.header.qoffset_y), Float64(ni.header.qoffset_z))
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    end
    
    raw_header_attrs = Dict{String, Any}(
        "raw_srow_x" => collect(Float64.(ni.header.srow_x)),
        "raw_srow_y" => collect(Float64.(ni.header.srow_y)),
        "raw_srow_z" => collect(Float64.(ni.header.srow_z)),
        "raw_qoffset" => [Float64(ni.header.qoffset_x), Float64(ni.header.qoffset_y), Float64(ni.header.qoffset_z)],
        "raw_quaternions" => [Float64(ni.header.quatern_b), Float64(ni.header.quatern_c), Float64(ni.header.quatern_d)],
        "raw_qfac" => Float64(ni.header.pixdim[1])
    )
    return (arr, spacing, origin, direction, raw_header_attrs)
end

"""
    build_primary_hdf5(case_dir::String, h5_path::String; force::Bool=false, verbose::Bool=true)

Assembles the primary HDF5 container for a patient case.
Loads the CT volume and all Step 1 segmentations natively using NIfTI.jl and HDF5.jl.
Computes anatomical landmarks (carina, aortic arch, internal iliac, digastric, axillary)
and serializes everything into `h5_path` using MedImages HDF5 format.
"""
function build_primary_hdf5(case_dir::String, h5_path::String; force::Bool=false, verbose::Bool=true)
    if isfile(h5_path) && !force
        if verbose
            println("Primary HDF5 file already exists at: $h5_path (use force=true to rebuild).")
        end
        return h5_path
    end

    ct_path = joinpath(case_dir, "Fixed_CT_Volume.nii.gz")
    if !isfile(ct_path)
        error("Fixed_CT_Volume.nii.gz not found in case directory: $case_dir")
    end

    seg_dir = isdir(joinpath(case_dir, "segmentations")) ? joinpath(case_dir, "segmentations") : joinpath(case_dir, "segmentations_fast_backup")
    if !isdir(seg_dir)
        error("Segmentations directory not found in case directory: $case_dir")
    end

    if verbose
        println("Assembling Primary HDF5 for: $(basename(case_dir))")
        println("  -> CT Source: $ct_path")
        println("  -> Segmentation Source: $seg_dir")
    end

    # 1. Load CT Volume
    ct_arr, spacing, origin, direction, raw_meta = safe_read_nii(ct_path)
    dims = size(ct_arr)

    if verbose
        println("  -> CT Dimensions: $dims, Spacing: $spacing, Origin: $origin")
    end

    # Create / overwrite HDF5 file
    mkpath(dirname(h5_path))
    h5open(h5_path, "w") do f
        # Save CT Volume
        g_ct = create_group(f, "ct")
        g_ct["volume", chunk=(64, 64, 32), compress=1] = ct_arr
        attributes(g_ct["volume"])["spacing"] = collect(spacing)
        attributes(g_ct["volume"])["origin"] = collect(origin)
        attributes(g_ct["volume"])["direction"] = collect(direction)

        # 2. Scan and load all organ / muscle masks and details masks
        masks_dict = Dict{String, Array{UInt8, 3}}()
        
        # Collect from main segmentation dir and details subdir
        all_mask_paths = Dict{String, String}()
        for d in [seg_dir, joinpath(seg_dir, "details")]
            if isdir(d)
                for fn in readdir(d)
                    if endswith(fn, ".nii.gz") || endswith(fn, ".nii") || endswith(fn, ".h5")
                        mask_name = replace(fn, r"\.nii(\.gz)?$" => "")
                        mask_name = replace(mask_name, r"\.h5$" => "")

                        mask_name = replace(mask_name, r"_primary_totalsegmentator$" => "")
                        mask_name = replace(mask_name, r"_fallback_2_slicerdentalsegmentator$" => "")
                        if !haskey(all_mask_paths, mask_name)
                            all_mask_paths[mask_name] = joinpath(d, fn)
                        end
                    end
                end
            end
        end
        
        if verbose
            println("  -> Found $(length(all_mask_paths)) unique segmentation and detail mask files to process...")
        end

        g_masks = create_group(f, "masks")
        count_loaded = 0
        for (mask_name, mask_path) in all_mask_paths
            try
                if endswith(mask_path, ".h5")
                    m_arr = h5read(mask_path, "voxel_data")
                else
                    m_arr, _, _, _, _ = safe_read_nii(mask_path)
                end

                arr_u8 = UInt8.(m_arr .> 0)
                masks_dict[mask_name] = arr_u8
                
                g_masks[mask_name, chunk=(64, 64, 32), compress=1] = arr_u8
                attributes(g_masks[mask_name])["spacing"] = collect(spacing)
                attributes(g_masks[mask_name])["origin"] = collect(origin)
                attributes(g_masks[mask_name])["direction"] = collect(direction)
                count_loaded += 1
            catch e
                @warn "Failed to load/save mask $(basename(mask_path)): $e"
            end
        end

        if verbose
            println("  -> Successfully packed $count_loaded masks under '/masks/'.")
        end

        # 3. Compute and Merge Anatomical & Geometric Landmarks
        if verbose
            println("  -> Ingesting landmark cache and computing geometric landmarks...")
        end
        landmarks_dict = Dict{String, Any}()

        # Ingest existing cache if present
        for cache_path in [joinpath(seg_dir, "details", "computed_landmarks_cache.json"), joinpath(seg_dir, "computed_landmarks_cache.json")]
            if isfile(cache_path)
                try
                    c_data = JSON.parsefile(cache_path)
                    for (ck, cv) in c_data
                        landmarks_dict[ck] = cv
                    end
                    if verbose
                        println("  -> Ingested $(length(c_data)) cached landmark entries from $(basename(cache_path))")
                    end
                    break
                catch e
                    @warn "Failed to parse landmark cache $cache_path: $e"
                end
            end
        end

        # Carina
        if haskey(masks_dict, "trachea")
            car_z, car_mm = Landmarks.compute_carina_z(masks_dict["trachea"], spacing)
            landmarks_dict["carina_computed"] = car_mm
            landmarks_dict["carina_computed_vox"] = car_z
        end

        # Aortic Arch
        if haskey(masks_dict, "aorta")
            arch_z, arch_mm = Landmarks.compute_aortic_arch_z(masks_dict["aorta"], spacing)
            landmarks_dict["aortic_arch_computed"] = arch_mm
            landmarks_dict["aortic_arch_computed_vox"] = arch_z
        end

        # Internal Iliac (P1 & P2)
        for side in ("left", "right")
            art_k = "iliac_artery_internal_$side"
            if !haskey(masks_dict, art_k) && haskey(masks_dict, "iliac_artery_$side")
                art_k = "iliac_artery_$side"
            end
            if haskey(masks_dict, art_k) && (!haskey(landmarks_dict, "internal_iliac_p1_$side") || !haskey(landmarks_dict, "internal_iliac_p2_$side"))
                p1, p2 = Landmarks.compute_internal_iliac_points(masks_dict[art_k], spacing, origin, direction)
                landmarks_dict["internal_iliac_p1_$side"] = p1
                landmarks_dict["internal_iliac_p2_$side"] = p2
            end
        end

        # Digastric landmarks & plane
        for side in ("left", "right")
            m_arr = get(masks_dict, "mandible", nothing)
            h_arr = get(masks_dict, "hyoid", nothing)
            if (m_arr !== nothing || h_arr !== nothing) && !haskey(landmarks_dict, "digastric_plane_$side")
                dt, db, dpl = Landmarks.compute_digastric_landmarks(m_arr, h_arr, spacing, origin, side; is_lps=true)
                if dt !== nothing; landmarks_dict["dig_top_$side"] = dt; end
                if db !== nothing; landmarks_dict["dig_bot_$side"] = db; end
                if dpl !== nothing; landmarks_dict["digastric_plane_$side"] = (dt, dpl); end
            end
        end
        if haskey(landmarks_dict, "dig_top_left") && haskey(landmarks_dict, "dig_top_right")
            zl = landmarks_dict["dig_top_left"][3]
            zr = landmarks_dict["dig_top_right"][3]
            landmarks_dict["dig_top"] = zl > zr ? landmarks_dict["dig_top_left"] : landmarks_dict["dig_top_right"]
        end

        # Axillary Geometry
        for side in ("left", "right")
            scap = get(masks_dict, "scapula_$side", nothing)
            rib3 = get(masks_dict, "rib_$(side)_3", get(masks_dict, "rib_3_$side", get(masks_dict, "rib_3", nothing)))
            rib5 = get(masks_dict, "rib_$(side)_5", get(masks_dict, "rib_5_$side", get(masks_dict, "rib_5", nothing)))
            if scap !== nothing && rib3 !== nothing && rib5 !== nothing
                geom = Landmarks.compute_axillary_geometry(scap, rib3, rib5, spacing, origin, side)
                landmarks_dict["axillary_geometry_$side"] = geom
            end
        end

        # Save landmarks to HDF5 group
        g_lm = create_group(f, "landmarks")
        for (lk, lv) in landmarks_dict
            if lv isa Number
                g_lm[lk] = lv
            elseif lv isa Vector{<:Number}
                g_lm[lk] = lv
            else
                attributes(g_lm)[lk] = JSON.json(lv)
            end
        end

        # 4. Save metadata summary
        g_meta = create_group(f, "metadata")
        attributes(g_meta)["case_id"] = basename(case_dir)
        attributes(g_meta)["created_at"] = string(Dates.now())
        attributes(g_meta)["mask_count"] = count_loaded
        attributes(g_meta)["landmark_count"] = length(landmarks_dict)
        attributes(g_meta)["dimensions"] = collect(dims)
        attributes(g_meta)["spacing"] = collect(spacing)
        attributes(g_meta)["origin"] = collect(origin)
        attributes(g_meta)["direction"] = collect(direction)
        for (rk, rv) in raw_meta
            attributes(g_meta)[rk] = rv
        end
    end

    if verbose
        println("Primary HDF5 file successfully assembled at: $h5_path")
    end
    return h5_path
end

"""
    load_primary_masks(h5_path::String; required_names=nothing) -> Dict{String, BitArray{3}}

High-speed streaming of binary masks from `primary_masks.h5` into Julia memory.
"""
function load_primary_masks(h5_path::String; required_names::Union{Nothing, Set{String}, Vector{String}}=nothing)
    if !isfile(h5_path)
        error("Primary HDF5 file not found: $h5_path")
    end

    req_set = required_names === nothing ? nothing : Set{String}([lowercase(n) for n in required_names])
    masks = Dict{String, BitArray{3}}()

    h5open(h5_path, "r") do f
        if !haskey(f, "masks")
            @warn "HDF5 file does not contain a 'masks' group."
            return masks
        end

        g_masks = f["masks"]
        for k in keys(g_masks)
            if req_set !== nothing && !(lowercase(k) in req_set)
                continue
            end
            arr = read(g_masks[k])
            if length(size(arr)) == 3 && size(arr, 1) != 512 && size(arr, 3) == 512
                arr = permutedims(arr, (3, 2, 1))
            end
            masks[k] = BitArray(arr .> 0)
            arr = nothing
            if length(masks) % 20 == 0
                GC.gc(false)
            end
        end
    end

    return masks
end

"""
    load_primary_ct(h5_path::String) -> Array{Int16, 3}
"""
function load_primary_ct(h5_path::String)
    if !isfile(h5_path)
        error("Primary HDF5 file not found: $h5_path")
    end
    h5open(h5_path, "r") do f
        if haskey(f, "ct") && haskey(f["ct"], "volume")
            return read(f["ct"]["volume"])
        else
            error("Dataset 'ct/volume' not found in $h5_path")
        end
    end
end

"""
    load_primary_landmarks(h5_path::String) -> Dict{String, Any}
"""
function load_primary_landmarks(h5_path::String)
    if !isfile(h5_path)
        error("Primary HDF5 file not found: $h5_path")
    end

    landmarks = Dict{String, Any}()
    h5open(h5_path, "r") do f
        if haskey(f, "landmarks")
            g_lm = f["landmarks"]
            for k in keys(g_lm)
                landmarks[k] = read(g_lm[k])
            end
            attrs = attributes(g_lm)
            for ak in keys(attrs)
                av = read_attribute(g_lm, ak)
                try
                    landmarks[ak] = JSON.parse(av)
                catch
                    landmarks[ak] = av
                end
            end
        end
    end
    return landmarks
end

"""
    load_primary_metadata(h5_path::String) -> (dims, spacing, origin, direction)

Reads spatial metadata directly from the HDF5 `/metadata` attributes.
"""
function load_primary_metadata(h5_path::String)
    if !isfile(h5_path)
        error("Primary HDF5 file not found: $h5_path")
    end
    dims = (512, 512, 512)
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    h5open(h5_path, "r") do f
        if haskey(f, "metadata")
            g_meta = f["metadata"]
            attrs = attributes(g_meta)
            if haskey(attrs, "dimensions")
                d = read_attribute(g_meta, "dimensions")
                dims = (Int(d[1]), Int(d[2]), Int(d[3]))
            end
            if haskey(attrs, "spacing")
                s = read_attribute(g_meta, "spacing")
                spacing = (Float64(s[1]), Float64(s[2]), Float64(s[3]))
            end
            if haskey(attrs, "origin")
                o = read_attribute(g_meta, "origin")
                origin = (Float64(o[1]), Float64(o[2]), Float64(o[3]))
            end
            if haskey(attrs, "direction")
                dir_arr = read_attribute(g_meta, "direction")
                direction = Tuple(Float64(x) for x in dir_arr)
            end
        elseif haskey(f, "ct") && haskey(f["ct"], "volume")
            dims = size(f["ct"]["volume"])
        end
    end
    return dims, spacing, origin, direction
end

"""
    is_primary_hdf5_complete(h5_path::String, required_names::Set{String}) -> Bool
"""
function is_primary_hdf5_complete(h5_path::String, required_names::Set{String})::Bool
    if !isfile(h5_path)
        return false
    end
    try
        h5open(h5_path, "r") do f
            if !haskey(f, "ct") || !haskey(f["ct"], "volume") || !haskey(f, "masks")
                return false
            end
            g_masks = f["masks"]
            available = Set{String}([lowercase(k) for k in keys(g_masks)])
            for req in required_names
                if !(lowercase(req) in available)
                    return false
                end
            end
            return true
        end
    catch
        return false
    end
end

"""
    save_primary_landmark!(h5_path::String, name::String, value)
"""
function save_primary_landmark!(h5_path::String, name::String, value)
    h5open(h5_path, "r+") do f
        g_lm = haskey(f, "landmarks") ? f["landmarks"] : create_group(f, "landmarks")
        if value isa Number || value isa Vector{<:Number}
            if name in keys(g_lm); delete_object(g_lm, name); end
            g_lm[name] = value
        else
            attributes(g_lm)[name] = JSON.json(value)
        end
    end
end

end # module PrimaryHDF5
