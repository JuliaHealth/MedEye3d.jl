module HeatGDTDataset

using JSON
using NIfTI
using Statistics
using Random

export PatchHeatDataset, get_batch, get_batch_edge_only, extract_single_component

struct PatchHeatDataset
    items::Vector{Any}
end

function PatchHeatDataset(manifest_path::String)
    items = JSON.parsefile(manifest_path)
    return PatchHeatDataset(items)
end

Base.length(ds::PatchHeatDataset) = length(ds.items)

"""
    extract_single_component(gt::Array{UInt8, 3}, seed::Tuple{Int,Int,Int})

Extracts the single 3D connected component (26-connectivity) from binary ground truth `gt`
that contains the seed coordinate `seed`. All other disconnected lesions and background
are set to 0.0f0.
"""
function extract_single_component(gt::Array{UInt8, 3}, seed::Tuple{Int,Int,Int})
    sz = size(gt)
    mask = zeros(Float32, sz)
    (gt[seed...] != 1) && return mask

    mask[seed...] = 1.0f0
    queue = [seed]
    sizehint!(queue, 20000)

    # 26-connectivity neighborhood offsets
    offsets = Tuple{Int,Int,Int}[]
    for dz in -1:1, dy in -1:1, dx in -1:1
        (dx == 0 && dy == 0 && dz == 0) && continue
        push!(offsets, (dx, dy, dz))
    end

    head = 1
    while head <= length(queue)
        cx, cy, cz = queue[head]
        head += 1
        for (dx, dy, dz) in offsets
            nx, ny, nz = cx + dx, cy + dy, cz + dz
            if 1 <= nx <= sz[1] && 1 <= ny <= sz[2] && 1 <= nz <= sz[3]
                if gt[nx, ny, nz] == 1 && mask[nx, ny, nz] == 0.0f0
                    mask[nx, ny, nz] = 1.0f0
                    push!(queue, (nx, ny, nz))
                end
            end
        end
    end
    return mask
end

"""
    find_seed_point(gt, pet)

Finds the deterministic seed point (peak SUV within the lesion component).
"""
function find_seed_point(gt::Array{UInt8, 3}, pet::Array{Float32, 3})
    lesion_indices = findall(gt .== 1)
    if isempty(lesion_indices)
        sz = size(gt)
        return (div(sz[1], 2), div(sz[2], 2), div(sz[3], 2))
    end

    max_idx = lesion_indices[argmax([pet[idx] for idx in lesion_indices])]
    return Tuple(max_idx)
end

"""
    load_edge_gt(item, fallback_size)

Loads the edge ground truth for a patch item.
Returns Float32 array (X, Y, Z) with soft edge values in [0, 1].
Returns zeros if edge_gt_path is missing or empty.
"""
function load_edge_gt(item, fallback_size::Tuple{Int,Int,Int}=(64, 64, 64))
    edge_path = get(item, "edge_gt_path", "")
    if edge_path != "" && isfile(edge_path)
        return Float32.(niread(edge_path).raw)
    else
        return zeros(Float32, fallback_size...)
    end
end

"""
    load_sample(item)

Loads a single preprocessed NIfTI patch item.
Guarantees training and testing on ONE LESION AT A TIME:
- `x`: (X, Y, Z, 3) [CT, PET, TS]
- `seed_mask`: (X, Y, Z) with 1.0f0 at the seed voxel of the target lesion
- `gt_mask`: (X, Y, Z) binary mask of ONLY the target lesion connected component
- `edge_gt`: (X, Y, Z) soft edge ground truth from TotalSegmentator (zeros if unavailable)
"""
function load_sample(item; augment::Bool=false)
    ct = Float32.(niread(item["ct_path"]).raw)
    pet = Float32.(niread(item["pet_path"]).raw)
    
    if augment
        pet_scale = rand(Float32) * 1.19f0 + 0.01f0
        pet .*= pet_scale
    end

    ts = Float32.(niread(item["totalseg_path"]).raw)
    gt = UInt8.(niread(item["gt_path"]).raw)

    # CT z-score normalization
    ct_m = mean(ct)
    ct_s = std(ct) + 1f-6
    ct_norm = (ct .- ct_m) ./ ct_s

    # TotalSegmentator normalized to [0, 1]
    ts_norm = ts ./ 117.0f0

    # Combine into 3-channel input: [CT, PET, TS]
    x = cat(ct_norm, pet, ts_norm, dims=4)

    # Determine seed point coordinate
    seed_idx = if haskey(item, "seed_xyz")
        Tuple(Int.(item["seed_xyz"]))
    else
        find_seed_point(gt, pet)
    end

    seed_mask = zeros(Float32, size(gt)...)
    if 1 <= seed_idx[1] <= size(gt, 1) && 1 <= seed_idx[2] <= size(gt, 2) && 1 <= seed_idx[3] <= size(gt, 3)
        seed_mask[seed_idx...] = 1.0f0
    end

    # Isolate single lesion component (one lesion at a time)
    gt_mask = extract_single_component(gt, seed_idx)

    # Load edge ground truth (soft edges from TotalSegmentator boundaries)
    edge_gt = load_edge_gt(item, size(gt))

    return x, seed_mask, gt_mask, edge_gt
end

"""
    get_batch(ds, indices; augment=false)

Batches samples for index list:
- x: (X, Y, Z, 3, B)
- seed_mask: (X, Y, Z, 1, B)
- gt: (X, Y, Z, 1, B)
- edge_gt: (X, Y, Z, 1, B)
"""
function get_batch(ds::PatchHeatDataset, indices::AbstractVector{Int}; augment::Bool=false)
    samples = [load_sample(ds.items[i]; augment=augment) for i in indices]
    x_batch = cat([s[1] for s in samples]..., dims=5)
    seed_batch = reshape(cat([s[2] for s in samples]..., dims=4), size(samples[1][2])..., 1, length(indices))
    gt_batch = reshape(cat([s[3] for s in samples]..., dims=4), size(samples[1][3])..., 1, length(indices))
    edge_gt_batch = reshape(cat([s[4] for s in samples]..., dims=4), size(samples[1][4])..., 1, length(indices))
    return x_batch, seed_batch, gt_batch, edge_gt_batch
end

"""
    get_batch_edge_only(ds, indices; augment=false)

Batches samples for edge pretraining (no seed/gt needed):
- x: (X, Y, Z, 3, B)
- edge_gt: (X, Y, Z, 1, B)
"""
function get_batch_edge_only(ds::PatchHeatDataset, indices::AbstractVector{Int}; augment::Bool=false)
    samples = [load_sample(ds.items[i]; augment=augment) for i in indices]
    x_batch = cat([s[1] for s in samples]..., dims=5)
    edge_gt_batch = reshape(cat([s[4] for s in samples]..., dims=4), size(samples[1][4])..., 1, length(indices))
    return x_batch, edge_gt_batch
end

end # module
