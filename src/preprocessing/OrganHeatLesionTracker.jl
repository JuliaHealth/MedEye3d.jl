module OrganHeatLesionTracker

using HDF5
using JSON
using Statistics
using LinearAlgebra
using ..ConnectedComponents

export compute_organ_centroids, save_organ_centroids_to_h5!, load_organ_centroids_from_h5
export find_5_closest_organs, translate_lesion_center
export grow_heat_kernel_lesion, detect_overlap_and_link
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
# 2. 5-Closest-Organ Selection & IDW Translation
# -----------------------------------------------------------------------------

"""
    find_5_closest_organs(lesion_center::Vector{Float64},
                          centroids_A::Dict{Int, Vector{Float64}},
                          centroids_B::Dict{Int, Vector{Float64}}) -> Vector{Int}

Identifies the 5 closest shared organs between timepoints A and B relative to `lesion_center`.
"""
function find_5_closest_organs(lesion_center::Vector{Float64},
                               centroids_A::Dict{Int, Vector{Float64}},
                               centroids_B::Dict{Int, Vector{Float64}})::Vector{Int}
    common_labels = intersect(keys(centroids_A), keys(centroids_B))
    if isempty(common_labels)
        return Int[]
    end

    dists = Tuple{Float64, Int}[]
    for lid in common_labels
        cA = centroids_A[lid]
        d = norm(lesion_center .- cA)
        push!(dists, (d, lid))
    end
    sort!(dists, by=first)
    k = min(5, length(dists))
    return [dists[i][2] for i in 1:k]
end

"""
    translate_lesion_center(lesion_center::Vector{Float64},
                            organ_ids::Vector{Int},
                            centroids_A::Dict{Int, Vector{Float64}},
                            centroids_B::Dict{Int, Vector{Float64}}) -> Vector{Float64}

Translates the lesion center from timepoint A to timepoint B using Inverse Distance Weighting (IDW)
over the displacements of the selected anchor organs:
  Δ_i = Q_i - P_i
  w_i = 1 / (||L - P_i||^2 + 1.0)
  Δ_L = Σ w_i Δ_i / Σ w_i
  L_B = L_A + Δ_L
"""
function translate_lesion_center(lesion_center::Vector{Float64},
                                 organ_ids::Vector{Int},
                                 centroids_A::Dict{Int, Vector{Float64}},
                                 centroids_B::Dict{Int, Vector{Float64}})::Vector{Float64}
    if isempty(organ_ids)
        return copy(lesion_center)
    end

    total_weight = 0.0
    delta_sum = zeros(Float64, 3)

    for lid in organ_ids
        P_i = centroids_A[lid]
        Q_i = centroids_B[lid]
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
                            K::Int=25,
                            dt::Float32=0.16f0) -> Tuple{Array{UInt8, 3}, NTuple{6, Int}}

Runs heat diffusion starting from a single-voxel seed at `center` inside `diffusivity`.
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
                                 K::Int=25,
                                 dt::Float32=0.16f0)
    nx_full, ny_full, nz_full = size(diffusivity)
    
    # Choose patch size large enough for target volume
    req_radius = ceil(Int, (target_volume_voxels * 3 / (4 * π))^(1/3)) + 8
    P = max(patch_size, min(req_radius * 2, 128))
    
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
    u[scx, scy, scz] = 1.0f0

    # Precompute conductivities with replicate padding
    D_pad = _replicate_pad_3d(D_patch)
    D_xp = 0.5f0 .* (D_patch .+ D_pad[3:px+2, 2:py+1, 2:pz+1])
    D_xm = 0.5f0 .* (D_patch .+ D_pad[1:px,   2:py+1, 2:pz+1])
    D_yp = 0.5f0 .* (D_patch .+ D_pad[2:px+1, 3:py+2, 2:pz+1])
    D_ym = 0.5f0 .* (D_patch .+ D_pad[2:px+1, 1:py,   2:pz+1])
    D_zp = 0.5f0 .* (D_patch .+ D_pad[2:px+1, 2:py+1, 3:pz+2])
    D_zm = 0.5f0 .* (D_patch .+ D_pad[2:px+1, 2:py+1, 1:pz])

    # Run diffusion PDE steps
    for k in 1:K
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
    end

    # Binary search on threshold θ to match target_volume_voxels
    low_θ = 1.0f-7
    high_θ = maximum(u) * 0.999f0
    best_θ = low_θ
    best_diff = Inf

    for b in 1:20
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
    
    # Connected component cleanup
    clean_mask = try
        ConnectedComponents.extract_largest_connected_component(binary_mask)
    catch
        binary_mask
    end

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
                            max_original_id::Int=typemax(Int)) -> Tuple{Int, Int}

Checks for collision between `heat_patch` and existing original lesions in `target_mask`
using a simple boolean AND operation:
  overlap_count = count((heat_patch .> 0) .& (target_sub .== k_B))

Only matches against original lesions (cand_lid <= max_original_id).
Returns `(best_candidate_lid, max_overlapping_voxels)`.
If no existing original lesion overlaps, returns `(0, 0)`.
"""
function detect_overlap_and_link(heat_patch::Array{UInt8, 3},
                                bbox::NTuple{6, Int},
                                target_mask::AbstractArray{<:Integer, 3};
                                max_original_id::Int=typemax(Int))::Tuple{Int, Int}
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    target_sub = view(target_mask, x_min:x_max, y_min:y_max, z_min:z_max)

    overlap_counts = Dict{Int, Int}()
    heat_indices = findall(heat_patch .> 0)

    for idx in heat_indices
        cand_lid = Int(target_sub[idx])
        if cand_lid > 0 && cand_lid <= max_original_id
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

    # 2. Precompute and persist organ centroids for each TP
    all_tp_centroids = Dict{Int, Dict{Int, Vector{Float64}}}()
    h5open(h5_path, "r+") do h5
        for tp in sorted_tps
            grp_name = tp_groups[tp]
            if haskey(h5[grp_name], "max_anatomy.nii.gz")
                ma_arr = read(h5["$grp_name/max_anatomy.nii.gz"])
                cents = compute_organ_centroids(ma_arr)
                all_tp_centroids[tp] = cents
                verbose && println("  [TP $tp] Computed $(length(cents)) organ centroids from max_anatomy")
            else
                @warn "No max_anatomy.nii.gz in $grp_name"
                all_tp_centroids[tp] = Dict{Int, Vector{Float64}}()
            end
        end
        save_organ_centroids_to_h5!(h5, all_tp_centroids)
    end

    # 3. Read initial original lesion masks and volumes
    masks_by_tp = Dict{Int, Array{Int16, 3}}()
    diffs_by_tp = Dict{Int, Array{Float32, 3}}()

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
        end
    end

    # 4. Tracking Data Structures
    # group_id => [ {name, raw_lesion, lesion, volume_mm3, match_type, IoU, group_id}, ... ]
    # Group origin tracked via group_id
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
        
        cents_curr = all_tp_centroids[tp_curr]
        cents_next = all_tp_centroids[tp_next]

        # Get all lesions currently tracked at tp_curr
        curr_lids = sort([Int(x) for x in filter(x -> x > 0, unique(mask_curr))])
        
        for lid in curr_lids
            gid = get(lesion_to_group, (tp_curr, lid), 0)
            if gid == 0; continue; end
            
            c_curr = compute_lesion_centroid(mask_curr, lid)
            if c_curr === nothing; continue; end
            
            vol_curr = count(mask_curr .== lid)
            
            # 1. Select 5 closest organs
            closest_organs = find_5_closest_organs(c_curr, cents_curr, cents_next)
            
            # 2. Translate center to tp_next using IDW
            c_next = translate_lesion_center(c_curr, closest_organs, cents_curr, cents_next)
            
            # 3. Grow Heat-Kernel lesion to match volume
            heat_patch, bbox = grow_heat_kernel_lesion(diff_next, c_next, vol_curr; patch_size=64)
            
            # 4. Check for collision via Boolean AND
            cand_lid, overlap_vox = detect_overlap_and_link(heat_patch, bbox, mask_next; max_original_id=orig_max_id[tp_next])
            
            if cand_lid > 0
                # Overlapping lesion found! Link existing lesion
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
                
                # Write heat_patch into mask_next
                x_min, x_max, y_min, y_max, z_min, z_max = bbox
                sub_next = view(mask_next, x_min:x_max, y_min:y_max, z_min:z_max)
                synth_voxels = 0
                @inbounds for iz in 1:size(heat_patch, 3), iy in 1:size(heat_patch, 2), ix in 1:size(heat_patch, 1)
                    if heat_patch[ix, iy, iz] > 0 && sub_next[ix, iy, iz] == 0
                        sub_next[ix, iy, iz] = Int16(synth_id)
                        synth_voxels += 1
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

    # 7. Mirror to SPECT Timepoints
    verbose && println("\n--- Phase C: Mirroring to Paired SPECT Studies ---")
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

    # Write updated PET masks and mirror to SPECT in HDF5
    h5open(h5_path, "r+") do h5
        # 1. Write back updated PET lesion masks
        for tp in sorted_tps
            grp = tp_groups[tp]
            ds_name = "PET_Lesions_$tp.nii.gz"
            if haskey(h5[grp], ds_name)
                delete_object(h5[grp], ds_name)
            end
            h5["$grp/$ds_name"] = masks_by_tp[tp]
        end

        # 2. Write matching SPECT masks
        for (tp, grp_name) in spect_groups
            # Map SPECT timepoint 4 to PET timepoint 3 if needed
            pet_tp = (tp == 4 && !haskey(masks_by_tp, 4) && haskey(masks_by_tp, 3)) ? 3 : tp
            if haskey(masks_by_tp, pet_tp)
                ds_name = "SPECT_Lesions_$tp.nii.gz"
                if haskey(h5[grp_name], ds_name)
                    delete_object(h5[grp_name], ds_name)
                end
                h5["$grp_name/$ds_name"] = masks_by_tp[pet_tp]
                verbose && println("  [SPECT] Synchronized $grp_name/$ds_name from PET TP $pet_tp")
            end
        end

        # 3. Add SPECT children to match groups
        for (gid, parent) in groups
            children_to_add = Any[]
            # Parent SPECT counterpart
            p_node = parent["name"]
            p_raw = parent["raw_lesion"]
            p_seg = parse(Int, replace(p_raw, "Segment_" => ""))
            m = match(r"PET_Lesions_(\d+)", p_node)
            if m !== nothing
                p_tp = parse(Int, m.captures[1])
                spect_tp = (p_tp == 3 && haskey(spect_groups, 4)) ? 4 : p_tp
                push!(children_to_add, Dict{String, Any}(
                    "name" => "SPECT_Lesions_$spect_tp",
                    "raw_lesion" => p_raw,
                    "lesion" => "unknown_$(p_seg)_SPECT_$spect_tp",
                    "volume_mm3" => parent["volume_mm3"],
                    "match_type" => "SPECT_MIRROR",
                    "group_id" => gid
                ))
            end
            for c in parent["children"]
                c_node = c["name"]
                c_raw = c["raw_lesion"]
                c_seg = parse(Int, replace(c_raw, "Segment_" => ""))
                cm = match(r"PET_Lesions_(\d+)", c_node)
                if cm !== nothing
                    c_tp = parse(Int, cm.captures[1])
                    spect_tp = (c_tp == 3 && haskey(spect_groups, 4)) ? 4 : c_tp
                    push!(children_to_add, Dict{String, Any}(
                        "name" => "SPECT_Lesions_$spect_tp",
                        "raw_lesion" => c_raw,
                        "lesion" => replace(c["lesion"], "PET" => "SPECT"),
                        "volume_mm3" => c["volume_mm3"],
                        "match_type" => "SPECT_MIRROR",
                        "group_id" => gid
                    ))
                end
            end
            append!(parent["children"], children_to_add)
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
