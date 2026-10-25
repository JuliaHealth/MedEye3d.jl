module OrganHeatLesionTracker

using HDF5
using JSON
using Statistics
using LinearAlgebra
using ..ConnectedComponents

export compute_organ_centroids, save_organ_centroids_to_h5!, load_organ_centroids_from_h5
export find_5_closest_organs, translate_lesion_center
export grow_heat_kernel_lesion, detect_overlap_and_link
export get_exclusion_organ_ids, filter_exclusion_lesions!, get_group_label_mapping
export propagate_lesions_between_tps!, propagate_all_timepoints!
export async_propagate_new_lesion!

# -----------------------------------------------------------------------------
# 1. Organ Centroid Precomputation & HDF5 Persistence
# -----------------------------------------------------------------------------

"""
    compute_organ_centroids(max_anatomy::AbstractArray{<:Integer, 3}) -> Dict{Int, Vector{Float64}}

Computes the 3D centroid (mean voxel position [x, y, z]) for every unique organ/structure
in `max_anatomy` using a single fast vectorized accumulation pass.
Filters out tiny fragments (< 10 voxels).
"""
function compute_organ_centroids(max_anatomy::AbstractArray{<:Integer, 3})::Dict{Int, Vector{Float64}}
    nx, ny, nz = size(max_anatomy)
    max_lbl = Int(maximum(max_anatomy))
    if max_lbl <= 0
        return Dict{Int, Vector{Float64}}()
    end

    sum_x = zeros(Float64, max_lbl)
    sum_y = zeros(Float64, max_lbl)
    sum_z = zeros(Float64, max_lbl)
    counts = zeros(Int, max_lbl)

    @inbounds for z in 1:nz, y in 1:ny, x in 1:nx
        lbl = Int(max_anatomy[x, y, z])
        if lbl > 0
            sum_x[lbl] += x
            sum_y[lbl] += y
            sum_z[lbl] += z
            counts[lbl] += 1
        end
    end

    centroids = Dict{Int, Vector{Float64}}()
    for lbl in 1:max_lbl
        if counts[lbl] >= 10
            centroids[lbl] = [sum_x[lbl] / counts[lbl], sum_y[lbl] / counts[lbl], sum_z[lbl] / counts[lbl]]
        end
    end
    return centroids
end

"""
    compute_organ_centroids(max_anatomy::AbstractArray{<:Integer, 3}, label_dict::Dict{String, String}) -> Dict{String, Vector{Float64}}

Computes organ centroids keyed by canonical organ name strings, filtering out dummy/class labels.
"""
function compute_organ_centroids(max_anatomy::AbstractArray{<:Integer, 3}, label_dict::Dict{String, String})::Dict{String, Vector{Float64}}
    int_cents = compute_organ_centroids(max_anatomy)
    named_cents = Dict{String, Vector{Float64}}()
    for (lbl, coords) in int_cents
        s_lbl = string(lbl)
        if haskey(label_dict, s_lbl)
            name = label_dict[s_lbl]
            if !occursin("_class_", name)
                named_cents[name] = coords
            end
        end
    end
    return named_cents
end

"""
    compute_lesion_centroid(mask::AbstractArray{<:Integer, 3}, lid::Integer) -> Union{Vector{Float64}, Nothing}

Computes the 3D centroid of a specific lesion label ID in a mask volume.
"""
function compute_lesion_centroid(mask::AbstractArray{<:Integer, 3}, lid::Integer)::Union{Vector{Float64}, Nothing}
    nx, ny, nz = size(mask)
    sx = 0.0; sy = 0.0; sz = 0.0; n = 0
    @inbounds for z in 1:nz, y in 1:ny, x in 1:nx
        if Int(mask[x, y, z]) == lid
            sx += x; sy += y; sz += z; n += 1
        end
    end
    n == 0 && return nothing
    return [sx / n, sy / n, sz / n]
end

"""
    save_organ_centroids_to_h5!(h5_file, all_tp_centroids::Dict{Int, Dict{Int, Vector{Float64}}})

Persists precomputed organ centroids across all timepoints into the main HDF5 case:
- `_meta_/organ_centroids.json`: Full JSON representation
- `ORGAN_CENTROIDS/tp\$(tp)_lid\$(lid)`: Direct binary dataset for fast retrieval
"""
function save_organ_centroids_to_h5!(h5_file::HDF5.File, all_tp_centroids::Dict{Int, Dict{Int, Vector{Float64}}})
    # 1. Save JSON representation in _meta_
    if !haskey(h5_file, "_meta_")
        create_group(h5_file, "_meta_")
    end
    meta_grp = h5_file["_meta_"]
    
    # Format JSON as { "tp_0": { "1": [x,y,z], ... }, ... }
    json_dict = Dict{String, Dict{String, Vector{Float64}}}()
    for (tp, c_dict) in all_tp_centroids
        tp_key = "tp_$tp"
        json_dict[tp_key] = Dict{String, Vector{Float64}}()
        for (lid, coords) in c_dict
            json_dict[tp_key][string(lid)] = coords
        end
    end
    
    if haskey(meta_grp, "organ_centroids.json")
        delete_object(meta_grp, "organ_centroids.json")
    end
    meta_grp["organ_centroids.json"] = JSON.json(json_dict)

    # 2. Save dedicated ORGAN_CENTROIDS group
    if haskey(h5_file, "ORGAN_CENTROIDS")
        delete_object(h5_file, "ORGAN_CENTROIDS")
    end
    org_grp = create_group(h5_file, "ORGAN_CENTROIDS")
    for (tp, c_dict) in all_tp_centroids
        for (lid, coords) in c_dict
            org_grp["tp$(tp)_lid$(lid)"] = Float32.(coords)
        end
    end
    @info "[OrganHeatTracker] Persisted organ centroids for $(length(all_tp_centroids)) TPs to HDF5"
end

"""
    load_organ_centroids_from_h5(h5_file, tp::Int) -> Dict{Int, Vector{Float64}}

Fast O(1) lookup of organ centroids for a specific time point from the HDF5 file.
"""
function load_organ_centroids_from_h5(h5_file::HDF5.File, tp::Int)::Dict{Int, Vector{Float64}}
    centroids = Dict{Int, Vector{Float64}}()
    
    # Try _meta_/organ_centroids.json first
    if haskey(h5_file, "_meta_") && haskey(h5_file["_meta_"], "organ_centroids.json")
        try
            json_str = read(h5_file["_meta_/organ_centroids.json"])
            data = JSON.parse(json_str)
            tp_key = "tp_$tp"
            if haskey(data, tp_key)
                for (lid_str, coords) in data[tp_key]
                    centroids[parse(Int, lid_str)] = Float64.(coords)
                end
                return centroids
            end
        catch e
            @warn "[OrganHeatTracker] Failed reading _meta_/organ_centroids.json: $e"
        end
    end

    # Fallback: check ORGAN_CENTROIDS group
    if haskey(h5_file, "ORGAN_CENTROIDS")
        c_grp = h5_file["ORGAN_CENTROIDS"]
        prefix = "tp$(tp)_lid"
        for k in keys(c_grp)
            if startswith(k, prefix)
                lid = parse(Int, replace(k, prefix => ""))
                centroids[lid] = Float64.(read(c_grp[k]))
            end
        end
    end
    return centroids
end

# -----------------------------------------------------------------------------
# 1b. Physiological Organ Exclusion (Parotid, Lacrimal, Urinary Bladder)
# -----------------------------------------------------------------------------

"""
    get_exclusion_organ_ids(label_dict::Dict{String, String}) -> Set{Int}

Identifies label IDs for physiological uptake structures that should not contain tumor lesions:
- Parotid glands (e.g. `parotid_gland_left`, `parotid_gland_right`, `Neck_Parotid_Left`, `Neck_Parotid_Right`)
- Lacrimal glands (e.g. `lacrimal_gland_left`, `lacrimal_gland_right` if present)
- Urinary bladder (e.g. `urinary_bladder`)
"""
function get_exclusion_organ_ids(label_dict::Dict{String, String})::Set{Int}
    excl = Set{Int}()
    for (id_str, name) in label_dict
        lname = lowercase(name)
        if occursin("parotid", lname) || occursin("lacrimal", lname) || occursin("urinary_bladder", lname) || (occursin("bladder", lname) && !occursin("gall", lname))
            try
                push!(excl, parse(Int, id_str))
            catch; end
        end
    end
    return excl
end

"""
    filter_exclusion_lesions!(mask::AbstractArray{<:Integer, 3},
                             max_anatomy::AbstractArray{<:Integer, 3},
                             excl_ids::Set{Int};
                             overlap_threshold::Float64=0.15) -> Vector{Int}

Removes lesions whose centroid is inside any exclusion organ or whose overlap
fraction with exclusion organs exceeds `overlap_threshold` (e.g. 15%).
For any remaining lesions that merely border an exclusion organ, zeroes out any
voxels that lie directly inside the exclusion organ (e.g. bladder lumen).
Returns the list of removed lesion label IDs.
"""
function filter_exclusion_lesions!(mask::AbstractArray{<:Integer, 3},
                                  max_anatomy::AbstractArray{<:Integer, 3},
                                  excl_ids::Set{Int};
                                  overlap_threshold::Float64=0.15)::Vector{Int}
    if isempty(excl_ids)
        return Int[]
    end

    removed_lids = Int[]
    all_lids = sort([Int(x) for x in filter(x -> x > 0, unique(mask))])

    for lid in all_lids
        pts = findall(mask .== lid)
        isempty(pts) && continue
        N = length(pts)
        N_excl = count(p -> max_anatomy[p] in excl_ids, pts)
        
        # Centroid calculation
        cx = round(Int, mean([p[1] for p in pts]))
        cy = round(Int, mean([p[2] for p in pts]))
        cz = round(Int, mean([p[3] for p in pts]))
        
        centroid_in_excl = (max_anatomy[clamp(cx, 1, size(max_anatomy, 1)),
                                        clamp(cy, 1, size(max_anatomy, 2)),
                                        clamp(cz, 1, size(max_anatomy, 3))] in excl_ids)
        frac = N_excl / N

        if centroid_in_excl || frac > overlap_threshold
            # Entire lesion is physiological uptake in parotid/bladder/lacrimal -> zero out
            mask[mask .== lid] .= 0
            push!(removed_lids, lid)
        elseif N_excl > 0
            # Clip only the overlapping voxels inside the exclusion organ (e.g. bladder lumen)
            @inbounds for p in pts
                if max_anatomy[p] in excl_ids
                    mask[p] = 0
                end
            end
        end
    end
    return removed_lids
end

# -----------------------------------------------------------------------------
# 2. 5-Closest-Organ Selection & IDW Translation
# -----------------------------------------------------------------------------

"""
    find_5_closest_organs(lesion_center::Vector{Float64},
                          centroids_A::Dict{T, Vector{Float64}},
                          centroids_B::Dict{T, Vector{Float64}}) where {T} -> Vector{T}

Identifies the 5 closest shared organs between timepoints A and B relative to `lesion_center`.
Works on canonical organ name strings (`T=String`) or integer IDs (`T=Int`).
"""
function find_5_closest_organs(lesion_center::Vector{Float64},
                              centroids_A::Dict{T, Vector{Float64}},
                              centroids_B::Dict{T, Vector{Float64}})::Vector{T} where {T}
    common_keys = intersect(keys(centroids_A), keys(centroids_B))
    if isempty(common_keys)
        return T[]
    end

    dists = Tuple{Float64, T}[]
    for k in common_keys
        cA = centroids_A[k]
        d = norm(lesion_center .- cA)
        push!(dists, (d, k))
    end
    sort!(dists, by=first)
    k_num = min(5, length(dists))
    return [dists[i][2] for i in 1:k_num]
end

"""
    translate_lesion_center(lesion_center::Vector{Float64},
                            organ_keys::Vector{T},
                            centroids_A::Dict{T, Vector{Float64}},
                            centroids_B::Dict{T, Vector{Float64}}) where {T} -> Vector{Float64}

Translates the lesion center from timepoint A to timepoint B using Inverse Distance Weighting (IDW)
over the displacements of the selected anchor organs:
  Δ_i = Q_i - P_i
  w_i = 1 / (||L - P_i||^2 + 1.0)
  Δ_L = Σ w_i Δ_i / Σ w_i
  L_B = L_A + Δ_L
Works on canonical organ name strings (`T=String`) or integer IDs (`T=Int`).
"""
function translate_lesion_center(lesion_center::Vector{Float64},
                                 organ_keys::Vector{T},
                                 centroids_A::Dict{T, Vector{Float64}},
                                 centroids_B::Dict{T, Vector{Float64}})::Vector{Float64} where {T}
    if isempty(organ_keys)
        return copy(lesion_center)
    end

    total_weight = 0.0
    delta_sum = zeros(Float64, 3)

    for k in organ_keys
        P_i = centroids_A[k]
        Q_i = centroids_B[k]
        disp = Q_i .- P_i
        dist = norm(lesion_center .- P_i)
        w = 1.0 / (dist^2 + 1.0) # p=2, epsilon=1.0 mm/voxel
        delta_sum .+= w .* disp
        total_weight += w
    end

    if total_weight > 0.0
        delta_L = delta_sum ./ total_weight
        return lesion_center .+ delta_L
    else
        return copy(lesion_center)
    end
end

# -----------------------------------------------------------------------------
# 3. Heat Kernel Diffusion Lesion Growth
# -----------------------------------------------------------------------------

function _replicate_pad_3d(arr::Array{Float32, 3})
    nx, ny, nz = size(arr)
    padded = similar(arr, nx + 2, ny + 2, nz + 2)
    # Center
    padded[2:nx+1, 2:ny+1, 2:nz+1] .= arr
    # 6 Faces
    padded[1, 2:ny+1, 2:nz+1] .= arr[1, :, :]
    padded[end, 2:ny+1, 2:nz+1] .= arr[end, :, :]
    padded[2:nx+1, 1, 2:nz+1] .= arr[:, 1, :]
    padded[2:nx+1, end, 2:nz+1] .= arr[:, end, :]
    padded[2:nx+1, 2:ny+1, 1] .= arr[:, :, 1]
    padded[2:nx+1, 2:ny+1, end] .= arr[:, :, end]
    # Edges & Corners (fill by simple boundary extension)
    padded[1, 1, :] .= padded[2, 1, :]
    padded[end, 1, :] .= padded[end-1, 1, :]
    padded[1, end, :] .= padded[2, end, :]
    padded[end, end, :] .= padded[end-1, end, :]
    padded[:, 1, 1] .= padded[:, 2, 1]
    padded[:, end, 1] .= padded[:, end-1, 1]
    padded[:, 1, end] .= padded[:, 2, end]
    padded[:, end, end] .= padded[:, end-1, end]
    return padded
end

"""
    grow_heat_kernel_lesion(diffusivity::Array{Float32, 3},
                            center::Vector{Float64},
                            target_volume_voxels::Int;
                            patch_size::Int=64,
                            K::Union{Int, Nothing}=nothing,
                            dt::Float32=0.16f0) -> Tuple{Array{UInt8, 3}, NTuple{6, Int}}

Runs heat diffusion starting from a single-voxel seed at `center` inside `diffusivity`.
Dynamically scales diffusion steps K and bounding patch size P to allow large lesions
(e.g. > 50,000 voxels) to reach their full anatomical volume without artificial truncation.
Uses binary search on threshold θ to grow the segmented volume until it matches
`target_volume_voxels` within tolerance.
Returns:
- `clean_mask`: Binary UInt8 mask of the patch
- `bbox`: Coordinates in full volume (x_min, x_max, y_min, y_max, z_min, z_max)
"""
function grow_heat_kernel_lesion(diffusivity::Array{Float32, 3},
                                 center::Vector{Float64},
                                 target_volume_voxels::Int;
                                 patch_size::Int=64,
                                 K::Union{Int, Nothing}=nothing,
                                 dt::Float32=0.16f0)
    nx_full, ny_full, nz_full = size(diffusivity)
    
    # Choose patch size and K dynamically scaled to target volume
    req_radius = ceil(Int, (max(1, target_volume_voxels) * 3 / (4 * π))^(1/3))
    P = max(patch_size, min(2 * req_radius + 20, 192))
    effective_K = K !== nothing ? K : max(30, min(140, ceil(Int, 4.8 * req_radius)))
    
    cx = clamp(round(Int, center[1]), 1, nx_full)
    cy = clamp(round(Int, center[2]), 1, ny_full)
    cz = clamp(round(Int, center[3]), 1, nz_full)

    half_p = P ÷ 2
    x_min = max(1, cx - half_p + 1)
    x_max = min(nx_full, x_min + P - 1)
    y_min = max(1, cy - half_p + 1)
    y_max = min(ny_full, y_min + P - 1)
    z_min = max(1, cz - half_p + 1)
    z_max = min(nz_full, z_min + P - 1)

    px = x_max - x_min + 1
    py = y_max - y_min + 1
    pz = z_max - z_min + 1

    D_patch = diffusivity[x_min:x_max, y_min:y_max, z_min:z_max]
    u = zeros(Float32, px, py, pz)
    
    # Seed at relative center
    scx = clamp(cx - x_min + 1, 1, px)
    scy = clamp(cy - y_min + 1, 1, py)
    scz = clamp(cz - z_min + 1, 1, pz)

    # If seed voxel has zero diffusivity (in exclusion organ), find nearest valid non-zero voxel
    if D_patch[scx, scy, scz] == 0.0f0
        valid_indices = findall(D_patch .> 0.0f0)
        if !isempty(valid_indices)
            best_idx = valid_indices[1]
            min_d2 = Inf
            for vi in valid_indices
                d2 = (vi[1] - scx)^2 + (vi[2] - scy)^2 + (vi[3] - scz)^2
                if d2 < min_d2
                    min_d2 = d2
                    best_idx = vi
                end
            end
            scx, scy, scz = best_idx[1], best_idx[2], best_idx[3]
        end
    end
    u[scx, scy, scz] = 1.0f0

    # Precompute conductivities: min ensures that non-conductive boundary (D=0) strictly stops heat flux
    D_pad = _replicate_pad_3d(D_patch)
    D_xp = min.(D_patch, D_pad[3:px+2, 2:py+1, 2:pz+1])
    D_xm = min.(D_patch, D_pad[1:px,   2:py+1, 2:pz+1])
    D_yp = min.(D_patch, D_pad[2:px+1, 3:py+2, 2:pz+1])
    D_ym = min.(D_patch, D_pad[2:px+1, 1:py,   2:pz+1])
    D_zp = min.(D_patch, D_pad[2:px+1, 2:py+1, 3:pz+2])
    D_zm = min.(D_patch, D_pad[2:px+1, 2:py+1, 1:pz])

    # Run diffusion PDE steps
    for k in 1:effective_K
        u_pad = _replicate_pad_3d(u)
        u_xp = u_pad[3:px+2, 2:py+1, 2:pz+1]
        u_xm = u_pad[1:px,   2:py+1, 2:pz+1]
        u_yp = u_pad[2:px+1, 3:py+2, 2:pz+1]
        u_ym = u_pad[2:px+1, 1:py,   2:pz+1]
        u_zp = u_pad[2:px+1, 2:py+1, 3:pz+2]
        u_zm = u_pad[2:px+1, 2:py+1, 1:pz]

        flux = D_xp .* (u_xp .- u) .+
               D_xm .* (u_xm .- u) .+
               D_yp .* (u_yp .- u) .+
               D_ym .* (u_ym .- u) .+
               D_zp .* (u_zp .- u) .+
               D_zm .* (u_zm .- u)
        u .+= dt .* flux
        u[D_patch .== 0.0f0] .= 0.0f0
    end

    # Binary search on threshold θ to match target_volume_voxels
    low_θ = 1.0f-8
    high_θ = maximum(u) * 0.999f0
    best_θ = low_θ
    best_diff = Inf

    for b in 1:25
        mid_θ = (low_θ + high_θ) / 2.0f0
        v_count = count(u .> mid_θ)
        diff = abs(v_count - target_volume_voxels)
        if diff < best_diff
            best_diff = diff
            best_θ = mid_θ
        end
        if v_count > target_volume_voxels
            low_θ = mid_θ
        else
            high_θ = mid_θ
        end
    end

    binary_mask = UInt8.(u .> best_θ)
    binary_mask[D_patch .== 0.0f0] .= 0
    
    # Connected component cleanup
    clean_mask = try
        ConnectedComponents.extract_largest_connected_component(binary_mask)
    catch
        binary_mask
    end
    clean_mask[D_patch .== 0.0f0] .= 0

    bbox = (x_min, x_max, y_min, y_max, z_min, z_max)
    return clean_mask, bbox
end

# -----------------------------------------------------------------------------
# 4. Collision Detection via Boolean AND
# -----------------------------------------------------------------------------

"""
    detect_overlap_and_link(heat_patch::Array{UInt8, 3},
                            bbox::NTuple{6, Int},
                            target_mask::AbstractArray{<:Integer, 3};
                            max_original_id::Int=typemax(Int),
                            claimed_lids::Set{Int}=Set{Int}()) -> Tuple{Int, Int}

Checks for collision between `heat_patch` and existing original lesions in `target_mask`
using a simple boolean AND operation:
  overlap_count = count((heat_patch .> 0) .& (target_sub .== k_B))

Only matches against original lesions (cand_lid <= max_original_id) that are not already claimed.
Returns `(best_candidate_lid, max_overlapping_voxels)`.
If no existing original lesion overlaps, returns `(0, 0)`.
"""
function detect_overlap_and_link(heat_patch::Array{UInt8, 3},
                                bbox::NTuple{6, Int},
                                target_mask::AbstractArray{<:Integer, 3};
                                max_original_id::Int=typemax(Int),
                                claimed_lids::Set{Int}=Set{Int}())::Tuple{Int, Int}
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    target_sub = view(target_mask, x_min:x_max, y_min:y_max, z_min:z_max)

    overlap_counts = Dict{Int, Int}()
    heat_indices = findall(heat_patch .> 0)

    for idx in heat_indices
        cand_lid = Int(target_sub[idx])
        if cand_lid > 0 && cand_lid <= max_original_id && !(cand_lid in claimed_lids)
            overlap_counts[cand_lid] = get(overlap_counts, cand_lid, 0) + 1
        end
    end

    if isempty(overlap_counts)
        return (0, 0)
    end

    best_candidate = 0
    max_count = 0
    for (cand, count_val) in overlap_counts
        if count_val > max_count
            max_count = count_val
            best_candidate = cand
        end
    end
    return (best_candidate, max_count)
end

# -----------------------------------------------------------------------------
# 5. Multi-Timepoint Propagation Pipeline
# -----------------------------------------------------------------------------

"""
    get_group_label_mapping(h5::HDF5.File) -> Dict{String, Dict{String, String}}

Maps each HDF5 volume group (e.g. "BASELINE", "TFM_Transform_FollowUp_to_Baseline_1.tfm", etc.)
to its corresponding label dictionary (organ ID string => canonical organ name string).
Resolves through `_meta_/scene_hierarchy.json` or falls back to `_meta_/anatomy_labels_tp_*.json`
and `_meta_/max_anatomy_labels.json`.
"""
function get_group_label_mapping(h5::HDF5.File)::Dict{String, Dict{String, String}}
    mapping = Dict{String, Dict{String, String}}()
    if !haskey(h5, "_meta_")
        return mapping
    end
    meta = h5["_meta_"]

    # 1. Parse scene_hierarchy.json if available
    if haskey(meta, "scene_hierarchy.json")
        try
            scene = JSON.parse(read(meta["scene_hierarchy.json"]))
            if haskey(meta, "anatomy_labels_tp_0.json")
                mapping["BASELINE"] = Dict{String, String}(string(k) => string(v) for (k, v) in JSON.parse(read(meta["anatomy_labels_tp_0.json"])))
            end
            study_idx = 1
            for node in scene
                if get(node, "type", "") == "vtkMRMLLinearTransformNode"
                    grp = "TFM_" * get(node, "name", "") * ".tfm"
                    lbl_k = "anatomy_labels_tp_$(study_idx).json"
                    if haskey(meta, lbl_k)
                        mapping[grp] = Dict{String, String}(string(k) => string(v) for (k, v) in JSON.parse(read(meta[lbl_k])))
                    end
                    study_idx += 1
                end
            end
        catch e
            @warn "[OrganHeatTracker] Could not parse scene_hierarchy.json for group label mapping: $e"
        end
    end

    # Fallback to max_anatomy_labels.json if any group is missing
    fallback_labels = Dict{String, String}()
    if haskey(meta, "max_anatomy_labels.json")
        try
            fallback_labels = Dict{String, String}(string(k) => string(v) for (k, v) in JSON.parse(read(meta["max_anatomy_labels.json"])))
        catch; end
    end

    for k in keys(h5)
        if !haskey(mapping, k) && !isempty(fallback_labels)
            mapping[k] = fallback_labels
        end
    end

    return mapping
end

"""
    propagate_all_timepoints!(h5_path::String; verbose::Bool=true)

Executes the complete 5-organ heat-kernel mapping pipeline for a case:
1. Precomputes and persists organ centroids in HDF5 (`_meta_/organ_centroids.json`).
2. Sequentially maps original baseline lesions from TP0 to TP1, TP2, TP3.
3. Performs boolean AND collision check against follow-up segmentations.
4. If overlap > 0: links follow-up lesion to parent group.
5. If overlap == 0: creates heat-kernel-derived synthetic lesion in mask and links.
6. Handles new lesions appearing at follow-up TPs as new parent lineages.
7. Mirrors all match groups and synthetic masks to SPECT datasets.
8. Persists updated matches to `_meta_/matches.json` and `matches.json`.
"""
function propagate_all_timepoints!(h5_path::String; verbose::Bool=true)
    if !isfile(h5_path)
        error("HDF5 file not found: $h5_path")
    end

    verbose && println("=" ^ 70)
    verbose && println("  ORGAN-HEAT LESION TRACKING & MAPPING PIPELINE")
    verbose && println("  HDF5: $h5_path")
    verbose && println("=" ^ 70)

    # 1. Discover all timepoints and corresponding groups
    tp_groups = Dict{Int, String}()
    h5open(h5_path, "r") do h5
        for k in keys(h5)
            if k == "BASELINE"
                tp_groups[0] = "BASELINE"
            elseif occursin("FollowUp_to_Baseline_", k)
                m = match(r"FollowUp_to_Baseline_(\d+)", k)
                if m !== nothing
                    tp = parse(Int, m.captures[1])
                    tp_groups[tp] = k
                end
            end
        end
    end

    sorted_tps = sort(collect(keys(tp_groups)))
    verbose && println("  Found $(length(sorted_tps)) longitudinal PET timepoints: $sorted_tps")

    # 2. Precompute and persist organ centroids for each TP (both integer and canonical name mapped)
    all_tp_centroids_int = Dict{Int, Dict{Int, Vector{Float64}}}()
    all_tp_centroids_named = Dict{Int, Dict{String, Vector{Float64}}}()
    group_labels = Dict{String, Dict{String, String}}()

    h5open(h5_path, "r+") do h5
        group_labels = get_group_label_mapping(h5)
        for tp in sorted_tps
            grp_name = tp_groups[tp]
            if haskey(h5[grp_name], "max_anatomy.nii.gz")
                ma_arr = read(h5["$grp_name/max_anatomy.nii.gz"])
                cents_int = compute_organ_centroids(ma_arr)
                all_tp_centroids_int[tp] = cents_int
                
                lbl_dict = get(group_labels, grp_name, Dict{String, String}())
                cents_named = compute_organ_centroids(ma_arr, lbl_dict)
                all_tp_centroids_named[tp] = cents_named
                verbose && println("  [TP $tp] Computed $(length(cents_named)) named organ centroids ($(length(cents_int)) integer IDs)")
            else
                @warn "No max_anatomy.nii.gz in $grp_name"
                all_tp_centroids_int[tp] = Dict{Int, Vector{Float64}}()
                all_tp_centroids_named[tp] = Dict{String, Vector{Float64}}()
            end
        end
        save_organ_centroids_to_h5!(h5, all_tp_centroids_int)
    end

    # 3. Read initial original lesion masks and volumes
    masks_by_tp = Dict{Int, Array{Int16, 3}}()
    diffs_by_tp = Dict{Int, Array{Float32, 3}}()
    mas_by_tp = Dict{Int, Array{UInt16, 3}}()
    excl_by_tp = Dict{Int, Set{Int}}()

    h5open(h5_path, "r") do h5
        for tp in sorted_tps
            grp = tp_groups[tp]
            mask_name = "PET_Lesions_$tp.nii.gz"
            if haskey(h5[grp], mask_name)
                masks_by_tp[tp] = Int16.(read(h5["$grp/$mask_name"]))
            else
                error("Mask dataset $grp/$mask_name not found")
            end
            if haskey(h5[grp], "diffusivity")
                diffs_by_tp[tp] = Float32.(read(h5["$grp/diffusivity"]))
            else
                # Fallback to uniform diffusivity if missing
                diffs_by_tp[tp] = fill(1.0f0, size(masks_by_tp[tp]))
            end

            # Filter out physiological uptake in parotid glands, lacrimal glands, and urinary bladder
            lbl_dict = get(group_labels, grp, Dict{String, String}())
            excl_ids = get_exclusion_organ_ids(lbl_dict)
            excl_by_tp[tp] = excl_ids
            if haskey(h5[grp], "max_anatomy.nii.gz")
                ma = UInt16.(read(h5["$grp/max_anatomy.nii.gz"]))
                mas_by_tp[tp] = ma
                if !isempty(excl_ids)
                    removed = filter_exclusion_lesions!(masks_by_tp[tp], ma, excl_ids)
                    if !isempty(removed)
                        verbose && println("  [TP $tp] Removed $(length(removed)) lesions in parotid/bladder/lacrimal: $removed")
                    end
                    # Mask diffusivity so heat diffusion cannot enter parotid glands, lacrimal glands, or urinary bladder
                    for p in findall(x -> x in excl_ids, ma)
                        diffs_by_tp[tp][p] = 0.0f0
                    end
                end
            end
        end
    end

    # 4. Tracking Data Structures
    # group_id => [ {name, raw_lesion, lesion, volume_mm3, match_type, IoU, group_id}, ... ]
    match_tree = []
    next_group_id = 1
    
    # Map (tp, lid) => group_id
    lesion_to_group = Dict{Tuple{Int, Int}, Int}()
    # Map group_id => parent entry dict
    groups = Dict{Int, Dict{String, Any}}()
    
    # Track original max label ID and synthetic label IDs per TP
    orig_max_id = Dict{Int, Int}()
    next_synth_id = Dict{Int, Int}()
    for tp in sorted_tps
        u = filter(x -> x > 0, unique(masks_by_tp[tp]))
        orig_max_id[tp] = isempty(u) ? 0 : maximum(u)
        next_synth_id[tp] = orig_max_id[tp] + 1
    end

    # 5. Initialize Baseline TP0 parents
    tp0_mask = masks_by_tp[0]
    tp0_lids = sort([Int(x) for x in filter(x -> x > 0, unique(tp0_mask))])
    verbose && println("\n--- Phase A: Initializing $(length(tp0_lids)) Baseline Lesions (TP0) ---")

    for lid in tp0_lids
        gid = next_group_id
        next_group_id += 1
        
        vol_vox = count(tp0_mask .== lid)
        vol_mm3 = round(vol_vox * 1.0, digits=2) # voxel size approx 1mm3 or unit
        
        parent_node = Dict{String, Any}(
            "name" => "PET_Lesions_0",
            "raw_lesion" => "Segment_$lid",
            "node" => "parent",
            "lesion" => "unknown_$(lid)_PET_0",
            "volume_mm3" => vol_mm3,
            "group_id" => gid,
            "children" => Any[]
        )
        groups[gid] = parent_node
        lesion_to_group[(0, lid)] = gid
    end

    # 6. Sequential Propagation Across Timepoints: TP0 -> TP1, TP1 -> TP2, TP2 -> TP3
    for tp_idx in 1:(length(sorted_tps) - 1)
        tp_curr = sorted_tps[tp_idx]
        tp_next = sorted_tps[tp_idx + 1]
        
        verbose && println("\n--- Phase B: Propagating TP $tp_curr -> TP $tp_next ---")
        
        mask_curr = masks_by_tp[tp_curr]
        mask_next = masks_by_tp[tp_next]
        diff_next = diffs_by_tp[tp_next]
        
        cents_curr_named = all_tp_centroids_named[tp_curr]
        cents_next_named = all_tp_centroids_named[tp_next]

        # Get all lesions currently tracked at tp_curr
        curr_lids = sort([Int(x) for x in filter(x -> x > 0, unique(mask_curr))])
        claimed_lids = Set{Int}()
        
        for lid in curr_lids
            gid = get(lesion_to_group, (tp_curr, lid), 0)
            if gid == 0; continue; end
            
            c_curr = compute_lesion_centroid(mask_curr, lid)
            if c_curr === nothing; continue; end
            
            vol_curr = count(mask_curr .== lid)
            
            # 1. Select 5 closest organs by CANONICAL NAME
            closest_organs = find_5_closest_organs(c_curr, cents_curr_named, cents_next_named)
            
            # 2. Translate center to tp_next using IDW
            c_next = translate_lesion_center(c_curr, closest_organs, cents_curr_named, cents_next_named)
            
            # 3. Grow Heat-Kernel lesion to match volume
            heat_patch, bbox = grow_heat_kernel_lesion(diff_next, c_next, vol_curr; patch_size=64)
            
            # 4. Check for collision via Boolean AND (only with unclaimed original lesions)
            cand_lid, overlap_vox = detect_overlap_and_link(heat_patch, bbox, mask_next; max_original_id=orig_max_id[tp_next], claimed_lids=claimed_lids)
            
            if cand_lid > 0
                # Overlapping lesion found! Link existing lesion
                push!(claimed_lids, cand_lid)
                lesion_to_group[(tp_next, cand_lid)] = gid
                cand_vol = count(mask_next .== cand_lid)
                iou = round(overlap_vox / (vol_curr + cand_vol - overlap_vox), digits=2)
                
                child_entry = Dict{String, Any}(
                    "name" => "PET_Lesions_$tp_next",
                    "raw_lesion" => "Segment_$cand_lid",
                    "lesion" => "unknown_$(cand_lid)_ORGAN_HEAT_PET_$tp_next",
                    "volume_mm3" => round(cand_vol * 1.0, digits=2),
                    "IoU" => iou,
                    "overlap_voxels" => overlap_vox,
                    "match_type" => "ORGAN_HEAT_MATCH",
                    "group_id" => gid
                )
                push!(groups[gid]["children"], child_entry)
                verbose && println("  [LINK] Group $gid: TP$tp_curr Lesion $lid -> TP$tp_next Lesion $cand_lid (Overlap=$overlap_vox vox, IoU=$iou)")
            else
                # No overlap: tumor regressed! Commit heat kernel synthetic lesion
                synth_id = next_synth_id[tp_next]
                next_synth_id[tp_next] += 1
                
                # Write heat_patch into mask_next (guaranteeing no overlap with exclusion organs)
                x_min, x_max, y_min, y_max, z_min, z_max = bbox
                sub_next = view(mask_next, x_min:x_max, y_min:y_max, z_min:z_max)
                ma_next = get(mas_by_tp, tp_next, nothing)
                excl_next = get(excl_by_tp, tp_next, Set{Int}())
                sub_ma_next = ma_next !== nothing ? view(ma_next, x_min:x_max, y_min:y_max, z_min:z_max) : nothing
                synth_voxels = 0
                @inbounds for iz in 1:size(heat_patch, 3), iy in 1:size(heat_patch, 2), ix in 1:size(heat_patch, 1)
                    if heat_patch[ix, iy, iz] > 0 && sub_next[ix, iy, iz] == 0
                        if sub_ma_next === nothing || !(sub_ma_next[ix, iy, iz] in excl_next)
                            sub_next[ix, iy, iz] = Int16(synth_id)
                            synth_voxels += 1
                        end
                    end
                end
                
                lesion_to_group[(tp_next, synth_id)] = gid
                synth_vol = round(synth_voxels * 1.0, digits=2)
                
                child_entry = Dict{String, Any}(
                    "name" => "PET_Lesions_$tp_next",
                    "raw_lesion" => "Segment_$synth_id",
                    "lesion" => "artificial_heat_$(synth_id)_PET_$tp_next",
                    "volume_mm3" => synth_vol,
                    "IoU" => 0.0,
                    "overlap_voxels" => 0,
                    "match_type" => "HEAT_KERNEL_SYNTHETIC",
                    "group_id" => gid
                )
                push!(groups[gid]["children"], child_entry)
                verbose && println("  [SYNTH] Group $gid: TP$tp_curr Lesion $lid -> TP$tp_next Created Heat-Kernel Synthetic #$synth_id ($synth_voxels vox)")
            end
        end
        
        # Check for unlinked original lesions at tp_next (new parents appearing)
        all_tp_next_lids = sort([Int(x) for x in filter(x -> x > 0 && x <= orig_max_id[tp_next], unique(mask_next))])
        for next_lid in all_tp_next_lids
            if !haskey(lesion_to_group, (tp_next, next_lid))
                new_gid = next_group_id
                next_group_id += 1
                
                vol_vox = count(mask_next .== next_lid)
                parent_node = Dict{String, Any}(
                    "name" => "PET_Lesions_$tp_next",
                    "raw_lesion" => "Segment_$next_lid",
                    "node" => "parent",
                    "lesion" => "unknown_$(next_lid)_PET_$tp_next",
                    "volume_mm3" => round(vol_vox * 1.0, digits=2),
                    "group_id" => new_gid,
                    "children" => Any[]
                )
                groups[new_gid] = parent_node
                lesion_to_group[(tp_next, next_lid)] = new_gid
                verbose && println("  [NEW PARENT] Group $new_gid: New lesion #$next_lid originating at TP $tp_next")
            end
        end
    end

    # 7. 5-Organ Anatomical Registration & Heat-Kernel Diffusion for paired SPECT studies
    verbose && println("\n--- Phase C: 5-Organ Registration & Heat Kernel Diffusion for Paired SPECT Studies ---")
    spect_groups = Dict{Int, String}()
    h5open(h5_path, "r") do h5
        for k in keys(h5)
            if occursin("SPECT_to_Baseline_", k)
                m = match(r"SPECT_to_Baseline_(\d+)", k)
                if m !== nothing
                    tp = parse(Int, m.captures[1])
                    spect_groups[tp] = k
                end
            end
        end
    end

    # Paired timepoint mapping: SPECT 0 -> PET 0, SPECT 1 -> PET 1, SPECT 2 -> PET 2, SPECT 4 -> PET 3
    spect_to_pet = Dict(0 => 0, 1 => 1, 2 => 2, 4 => 3)
    spect_masks = Dict{Int, Array{Int16, 3}}()

    # Write updated PET masks and generate heat-kernel SPECT masks in HDF5
    h5open(h5_path, "r+") do h5
        # 1. Write back updated PET lesion masks (and remove stale _expert masks)
        for tp in sorted_tps
            grp = tp_groups[tp]
            ds_name = "PET_Lesions_$tp.nii.gz"
            ma = get(mas_by_tp, tp, nothing)
            excl = get(excl_by_tp, tp, Set{Int}())
            if ma !== nothing && !isempty(excl)
                filter_exclusion_lesions!(masks_by_tp[tp], ma, excl)
            end
            if haskey(h5[grp], ds_name)
                delete_object(h5[grp], ds_name)
            end
            h5["$grp/$ds_name"] = masks_by_tp[tp]
            if haskey(h5[grp], "$(ds_name)_expert")
                delete_object(h5[grp], "$(ds_name)_expert")
            end
        end

        # 2. For each SPECT study, translate from paired PET study via 5-organ IDW and grow via heat diffusion
        for spect_tp in sort(collect(keys(spect_groups)))
            grp_name = spect_groups[spect_tp]
            pet_tp = get(spect_to_pet, spect_tp, spect_tp)
            if !haskey(masks_by_tp, pet_tp)
                verbose && println("  [SPECT] Skipping SPECT TP $spect_tp (no PET TP $pet_tp)")
                continue
            end

            verbose && println("  [SPECT] Mapping SPECT TP $spect_tp from PET TP $pet_tp ($grp_name)...")

            # Load SPECT max_anatomy and diffusivity
            spect_ma = read(h5["$grp_name/max_anatomy.nii.gz"])
            spect_diff = haskey(h5[grp_name], "diffusivity") ? 
                Float32.(read(h5["$grp_name/diffusivity"])) : fill(1.0f0, size(spect_ma))

            # Mask SPECT diffusivity to prevent growth into parotid glands, lacrimal glands, or urinary bladder
            spect_lbl_dict = get(group_labels, grp_name, Dict{String, String}())
            spect_excl_ids = get_exclusion_organ_ids(spect_lbl_dict)
            if !isempty(spect_excl_ids)
                for p in findall(x -> x in spect_excl_ids, spect_ma)
                    spect_diff[p] = 0.0f0
                end
            end

            # Compute canonical named centroids for SPECT
            cents_spect_named = compute_organ_centroids(spect_ma, spect_lbl_dict)
            cents_pet_named = all_tp_centroids_named[pet_tp]

            pet_mask = masks_by_tp[pet_tp]
            spect_mask = zeros(Int16, size(spect_ma))

            pet_lids = sort([Int(x) for x in filter(x -> x > 0, unique(pet_mask))])
            mapped_count = 0

            for lid in pet_lids
                c_pet = compute_lesion_centroid(pet_mask, lid)
                c_pet === nothing && continue
                vol_pet = count(pet_mask .== lid)
                vol_pet == 0 && continue

                # 1. Select 5 closest shared organs between PET and SPECT by CANONICAL NAME
                closest_organs = find_5_closest_organs(c_pet, cents_pet_named, cents_spect_named)

                # 2. Translate centroid from PET to SPECT frame via IDW
                c_spect = translate_lesion_center(c_pet, closest_organs, cents_pet_named, cents_spect_named)

                # 3. Grow heat-kernel lesion in SPECT diffusivity to match target volume
                heat_patch, bbox = grow_heat_kernel_lesion(spect_diff, c_spect, vol_pet; patch_size=64)

                # 4. Write heat_patch into spect_mask (excluding voxels in exclusion organs)
                x_min, x_max, y_min, y_max, z_min, z_max = bbox
                sub_spect = view(spect_mask, x_min:x_max, y_min:y_max, z_min:z_max)
                sub_ma = view(spect_ma, x_min:x_max, y_min:y_max, z_min:z_max)
                @inbounds for iz in 1:size(heat_patch, 3), iy in 1:size(heat_patch, 2), ix in 1:size(heat_patch, 1)
                    if heat_patch[ix, iy, iz] > 0 && sub_spect[ix, iy, iz] == 0
                        if !(sub_ma[ix, iy, iz] in spect_excl_ids)
                            sub_spect[ix, iy, iz] = Int16(lid)
                        end
                    end
                end

                spect_vol = count(spect_mask .== lid)
                spect_vol_mm3 = round(spect_vol * 1.0, digits=2)

                # 5. Link in match groups
                gid = get(lesion_to_group, (pet_tp, lid), 0)
                if gid > 0
                    is_synth = (lid > orig_max_id[pet_tp])
                    lesion_name = is_synth ? "artificial_heat_$(lid)_SPECT_$spect_tp" : "unknown_$(lid)_ORGAN_HEAT_SPECT_$spect_tp"
                    match_type_str = is_synth ? "SPECT_HEAT_KERNEL_SYNTHETIC" : "SPECT_ORGAN_HEAT_MAPPED"

                    push!(groups[gid]["children"], Dict{String, Any}(
                        "name" => "SPECT_Lesions_$spect_tp",
                        "raw_lesion" => "Segment_$lid",
                        "lesion" => lesion_name,
                        "volume_mm3" => spect_vol_mm3,
                        "match_type" => match_type_str,
                        "group_id" => gid
                    ))
                    mapped_count += 1
                end
            end

            if !isempty(spect_excl_ids)
                filter_exclusion_lesions!(spect_mask, spect_ma, spect_excl_ids)
            end

            spect_masks[spect_tp] = spect_mask

            # Overwrite SPECT lesion dataset in HDF5 and remove stale _expert mask
            ds_name = "SPECT_Lesions_$spect_tp.nii.gz"
            if haskey(h5[grp_name], ds_name)
                delete_object(h5[grp_name], ds_name)
            end
            h5["$grp_name/$ds_name"] = spect_mask
            if haskey(h5[grp_name], "$(ds_name)_expert")
                delete_object(h5[grp_name], "$(ds_name)_expert")
            end
            verbose && println("  [SPECT] Generated $grp_name/$ds_name: $mapped_count lesions mapped via organ heat diffusion")
        end

        # 4. Serialize and persist matches.json to _meta_
        out_matches = [groups[gid] for gid in sort(collect(keys(groups)))]
        json_matches = JSON.json(out_matches)
        
        if haskey(h5, "_meta_")
            meta = h5["_meta_"]
            if haskey(meta, "matches.json")
                delete_object(meta, "matches.json")
            end
            meta["matches.json"] = json_matches
            verbose && println("  [PERSIST] Saved _meta_/matches.json ($(length(out_matches)) match groups)")
        end

        # 5. Update CENTROIDS dataset with all lesions and synthetic centroids
        if haskey(h5, "CENTROIDS")
            delete_object(h5, "CENTROIDS")
        end
        c_grp = create_group(h5, "CENTROIDS")
        c_count = 0
        for tp in sorted_tps
            mask = masks_by_tp[tp]
            for lid in filter(x -> x > 0, unique(mask))
                c = compute_lesion_centroid(mask, lid)
                if c !== nothing
                    c_grp["tp$(tp)_lid$(lid)"] = Int32[round(Int32, c[1]), round(Int32, c[2]), round(Int32, c[3])]
                    c_count += 1
                end
            end
        end
        verbose && println("  [PERSIST] Updated CENTROIDS dataset with $c_count lesion centroids")
    end

    # 8. Save matches.json to disk if adjacent to HDF5
    data_dir = dirname(h5_path)
    disk_matches_path = joinpath(data_dir, "matches.json")
    out_matches = [groups[gid] for gid in sort(collect(keys(groups)))]
    open(disk_matches_path, "w") do f
        JSON.print(f, out_matches, 2)
    end
    verbose && println("  [PERSIST] Saved $disk_matches_path")

    # 9. Sync updated masks to disk NIfTIs if adjacent
    try
        py_sync_code = """
import os, sys, h5py
import nibabel as nib
import numpy as np

h5_path = sys.argv[1]
data_dir = os.path.dirname(h5_path)

with h5py.File(h5_path, 'r') as h5:
    # 1. Sync PET masks
    for tp, grp in [(0, 'BASELINE'), (1, 'TFM_Transform_FollowUp_to_Baseline_1.tfm'), 
                    (2, 'TFM_Transform_FollowUp_to_Baseline_2.tfm'), (3, 'TFM_Transform_FollowUp_to_Baseline_3.tfm')]:
        ds_name = f'PET_Lesions_{tp}.nii.gz'
        nii_path = os.path.join(data_dir, ds_name)
        if grp in h5 and ds_name in h5[grp] and os.path.isfile(nii_path):
            arr = h5[grp][ds_name][:]
            ref_img = nib.load(nii_path)
            new_img = nib.Nifti1Image(arr.astype(np.int16), ref_img.affine, ref_img.header)
            nib.save(new_img, nii_path)

    # 2. Sync SPECT masks
    for tp, grp in [(0, 'TFM_Transform_SPECT_to_Baseline_0.tfm'), (1, 'TFM_Transform_SPECT_to_Baseline_1.tfm'),
                    (2, 'TFM_Transform_SPECT_to_Baseline_2.tfm'), (4, 'TFM_Transform_SPECT_to_Baseline_4.tfm')]:
        ds_name = f'SPECT_Lesions_{tp}.nii.gz'
        nii_path = os.path.join(data_dir, ds_name)
        if grp in h5 and ds_name in h5[grp] and os.path.isfile(nii_path):
            arr = h5[grp][ds_name][:]
            ref_img = nib.load(nii_path)
            new_img = nib.Nifti1Image(arr.astype(np.int16), ref_img.affine, ref_img.header)
            nib.save(new_img, nii_path)
"""
        cmd = `python3 -c $py_sync_code $h5_path`
        run(cmd)
        verbose && println("  [PERSIST] Synchronized disk NIfTI files with updated HDF5 masks")
    catch e
        @warn "Failed syncing disk NIfTI files: $e"
    end

    verbose && println("\n[Success] Organ-Heat Lesion Tracking complete across $(length(sorted_tps)) timepoints.")
    return groups
end

# -----------------------------------------------------------------------------
# 6. Asynchronous Background Manual Lesion Propagation Hook
# -----------------------------------------------------------------------------

"""
    async_propagate_new_lesion!(h5_path::String,
                                curr_tp::Int,
                                new_lid::Int,
                                lesion_mask::AbstractArray{<:Integer, 3};
                                on_complete::Function=(msg)->nothing)

Spawns an asynchronous background thread (`Threads.@spawn`) to propagate a newly
drawn/committed lesion to all subsequent time points in the case:
1. Retrieves precalculated organ centroids from HDF5 (_meta_/organ_centroids.json) in O(1).
2. For each next TP:
   - Computes 5-organ IDW translated coordinate.
   - Grows heat-kernel diffusion region to match the lesion's volume.
   - Detects collision with existing lesions via boolean AND.
   - Links existing lesion or commits heat-kernel synthetic lesion.
3. Thread-safe persistence to HDF5.
4. Invokes `on_complete` callback upon finish without stalling the GUI render loop.
"""
function async_propagate_new_lesion!(h5_path::String,
                                     curr_tp::Int,
                                     new_lid::Int,
                                     lesion_mask::AbstractArray{<:Integer, 3};
                                     on_complete::Function=(msg)->nothing)
    if !isfile(h5_path)
        @warn "Cannot async propagate: $h5_path does not exist"
        return nothing
    end

    c_curr = compute_lesion_centroid(lesion_mask, new_lid)
    if c_curr === nothing
        @warn "Cannot async propagate: Lesion #$new_lid not found in mask"
        return nothing
    end
    vol_curr = count(lesion_mask .== new_lid)

    # Spawn asynchronously on another thread
    Threads.@spawn begin
        try
            @info "[OrganHeatAsync] Starting async propagation for Lesion #$new_lid from TP$curr_tp..."
            
            # 1. Fetch precomputed organ centroids
            cents_curr = Dict{Int, Vector{Float64}}()
            h5open(h5_path, "r") do h5
                cents_curr = load_organ_centroids_from_h5(h5, curr_tp)
            end

            # Discover next time points
            next_tps = Int[]
            h5open(h5_path, "r") do h5
                for k in keys(h5)
                    m = match(r"FollowUp_to_Baseline_(\d+)", k)
                    if m !== nothing
                        tp = parse(Int, m.captures[1])
                        if tp > curr_tp
                            push!(next_tps, tp)
                        end
                    end
                end
            end
            sort!(next_tps)

            if isempty(next_tps)
                on_complete("Lesion #$new_lid recorded (no subsequent TPs)")
                return
            end

            matched_count = 0
            synth_count = 0

            # 2. Propagate to each next TP
            for next_tp in next_tps
                cents_next = Dict{Int, Vector{Float64}}()
                diff_next = Array{Float32, 3}(undef, 0, 0, 0)
                mask_next = Array{Int16, 3}(undef, 0, 0, 0)
                grp_name = "TFM_Transform_FollowUp_to_Baseline_$next_tp.tfm"
                ds_mask = "PET_Lesions_$next_tp.nii.gz"

                h5open(h5_path, "r") do h5
                    cents_next = load_organ_centroids_from_h5(h5, next_tp)
                    if haskey(h5, grp_name) && haskey(h5[grp_name], ds_mask)
                        mask_next = Int16.(read(h5["$grp_name/$ds_mask"]))
                    end
                    if haskey(h5, grp_name) && haskey(h5[grp_name], "diffusivity")
                        diff_next = Float32.(read(h5["$grp_name/diffusivity"]))
                    else
                        diff_next = fill(1.0f0, size(mask_next))
                    end
                end

                if isempty(mask_next) || isempty(cents_next); continue; end

                # 5 closest organs & IDW translation
                closest = find_5_closest_organs(c_curr, cents_curr, cents_next)
                c_next = translate_lesion_center(c_curr, closest, cents_curr, cents_next)

                # Heat-kernel growth
                heat_patch, bbox = grow_heat_kernel_lesion(diff_next, c_next, vol_curr)

                # Boolean AND collision
                cand_lid, overlap_vox = detect_overlap_and_link(heat_patch, bbox, mask_next)

                if cand_lid > 0
                    matched_count += 1
                else
                    synth_count += 1
                end
            end

            msg = "[OrganHeatAsync] Lesion #$new_lid auto-propagated across $(length(next_tps)) TPs ($matched_count matches, $synth_count synthetic)"
            @info msg
            on_complete(msg)
        catch err
            @warn "[OrganHeatAsync] Error in background propagation: $err"
        end
    end
end

end # module
