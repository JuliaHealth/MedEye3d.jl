module LesionEditsRecorder

using Dates
using HDF5

export set_case_h5_path!, get_edits_h5_path, reset_recorder!
export compute_patch_bounds, extract_patch_64, are_voxels_connected_26
export record_heatgdt_event!, record_brush_voxels!, record_posthoc_event!
export flush_all_edits!, flush_all_edits_async!, has_pending_edits

const PATCH_SIZE = (64, 64, 64)
const _REC_LOCK = ReentrantLock()
const _H5_IO_LOCK = ReentrantLock()

const _case_dir_ref = Ref{String}("")
const _source_h5_ref = Ref{String}("")
const _edits_h5_ref = Ref{String}("")
const _case_id_ref = Ref{String}("")
const _spacing_ref = Ref{NTuple{3, Float64}}((1.0, 1.0, 1.0))
const _origin_ref = Ref{NTuple{3, Float64}}((0.0, 0.0, 0.0))
const _seq_counter = Ref{Int}(0)

Base.@kwdef mutable struct PendingEditEvent
    seq::Int
    event_type::String          # "heat_gdt", "manual_paint", "manual_erase", "posthoc_size", "posthoc_dilation", "posthoc_smoothing"
    timestamp::String
    seed_vox::NTuple{3, Int}
    theta::Float32 = 0.001f0
    K::Int = 40
    dilation::Int = 0
    smoothing::Float32 = 0.0f0
    edge_modality::String = "combined"
    brush_radius::Int = 0
    subpatch_idx::Int = 0       # 0 = primary patch, >0 = overflow subpatch index
    patch_data::Array{UInt8, 3} # 64x64x64 binary mask or event delta
end

Base.@kwdef mutable struct OverflowSubpatch
    subpatch_idx::Int
    origin_vox::NTuple{3, Int}
    center_vox::NTuple{3, Int}
    center_mm::NTuple{3, Float64}
    final_mask::Array{UInt8, 3}
    manual_painted_mask::Array{UInt8, 3}
    manual_erased_mask::Array{UInt8, 3}
    dirty::Bool = true
end

Base.@kwdef mutable struct LesionEditRecord
    uid::String
    tp_index::Int
    lesion_id::Int
    seed_vox::NTuple{3, Int}
    last_seed_vox::NTuple{3, Int}
    origin_vox::NTuple{3, Int}
    center_vox::NTuple{3, Int}
    center_mm::NTuple{3, Float64}
    initial_heat_mask::Array{UInt8, 3}
    final_mask::Array{UInt8, 3}
    manual_painted_mask::Array{UInt8, 3}
    manual_erased_mask::Array{UInt8, 3}
    subpatches::Vector{OverflowSubpatch} = OverflowSubpatch[]
    pending_events::Vector{PendingEditEvent} = PendingEditEvent[]
    next_event_seq::Int = 1
    dirty::Bool = true
end

# Active records keyed by (tp_index, lesion_id) -> Vector{LesionEditRecord}
# Multiple records for the same (tp_index, lesion_id) can exist if they are disconnected components.
const _records = Dict{Tuple{Int, Int}, Vector{LesionEditRecord}}()

function reset_recorder!()
    lock(_REC_LOCK) do
        empty!(_records)
        _seq_counter[] = 0
    end
end

"""
    set_case_h5_path!(source_h5::String; spacing=(1.0,1.0,1.0), origin=(0.0,0.0,0.0), edits_h5_path="")

Configure the separate independent `lesion_edits.h5` path for the active patient.
Never modifies `source_h5` (`preprocessed_volumes.h5`).
"""
function set_case_h5_path!(source_h5::String;
                           spacing::NTuple{3, Float64}=(1.0, 1.0, 1.0),
                           origin::NTuple{3, Float64}=(0.0, 0.0, 0.0),
                           edits_h5_path::String="")
    lock(_REC_LOCK) do
        empty!(_records)
        _seq_counter[] = 0
        _source_h5_ref[] = source_h5
        _spacing_ref[] = spacing
        _origin_ref[] = origin
        if !isempty(edits_h5_path)
            _edits_h5_ref[] = edits_h5_path
            _case_dir_ref[] = dirname(edits_h5_path)
            _case_id_ref[] = basename(dirname(edits_h5_path))
        elseif !isempty(source_h5)
            cdir = dirname(source_h5)
            _case_dir_ref[] = cdir
            _case_id_ref[] = basename(cdir)
            base = replace(basename(source_h5), r"\.h5$"i => "")
            fname = base == "preprocessed_volumes" ? "lesion_edits.h5" : "$(base)_lesion_edits.h5"
            _edits_h5_ref[] = joinpath(cdir, fname)
        else
            _case_dir_ref[] = ""
            _case_id_ref[] = ""
            _edits_h5_ref[] = ""
        end
    end
    return _edits_h5_ref[]
end

get_edits_h5_path()::String = _edits_h5_ref[]

function has_pending_edits()::Bool
    lock(_REC_LOCK) do
        for recs in values(_records)
            for rec in recs
                (rec.dirty || !isempty(rec.pending_events)) && return true
            end
        end
        return false
    end
end

"""
    compute_patch_bounds(cx, cy, cz, vol_size; patch_size=(64,64,64))

Compute `[x0:x1, y0:y1, z0:z1]` clamped to `vol_size`, plus `origin_vox` (where
chunk index `(1,1,1)` maps in the full volume) and `center_vox` / `center_mm`.
When the seed is near a volume border, `origin_vox` is shifted so the `64×64×64`
patch stays inside `1..vol_size` whenever `vol_size[d] >= patch_size[d]`.
"""
function compute_patch_bounds(cx::Int, cy::Int, cz::Int, vol_size::NTuple{3, Int};
                              patch_size::NTuple{3, Int}=PATCH_SIZE,
                              spacing::NTuple{3, Float64}=_spacing_ref[],
                              origin::NTuple{3, Float64}=_origin_ref[])
    half = (patch_size[1] ÷ 2, patch_size[2] ÷ 2, patch_size[3] ÷ 2)
    starts = ntuple(3) do d
        c = (cx, cy, cz)[d]
        max_start = max(1, vol_size[d] - patch_size[d] + 1)
        clamp(c - half[d], 1, max_start)
    end
    stops = ntuple(3) do d
        min(vol_size[d], starts[d] + patch_size[d] - 1)
    end
    center_vox = (
        round(Int, (starts[1] + stops[1]) / 2.0),
        round(Int, (starts[2] + stops[2]) / 2.0),
        round(Int, (starts[3] + stops[3]) / 2.0),
    )
    center_mm = (
        origin[1] + (center_vox[1] - 1) * spacing[1],
        origin[2] + (center_vox[2] - 1) * spacing[2],
        origin[3] + (center_vox[3] - 1) * spacing[3],
    )
    return (
        x_range = starts[1]:stops[1],
        y_range = starts[2]:stops[2],
        z_range = starts[3]:stops[3],
        origin_vox = starts,
        center_vox = center_vox,
        center_mm = center_mm,
    )
end

"""
    extract_patch_64(seg_vol, lesion_id, origin_vox; patch_size=(64,64,64)) -> Array{UInt8, 3}

Extract a strictly `64×64×64` `UInt8` binary mask (`1` where `seg_vol == lesion_id`, else `0`).
Zero-pads if the volume itself is smaller than `64` along any axis.
"""
function extract_patch_64(seg_vol::AbstractArray{T, 3}, lesion_id::Integer,
                          origin_vox::NTuple{3, Int};
                          patch_size::NTuple{3, Int}=PATCH_SIZE)::Array{UInt8, 3} where {T}
    patch = zeros(UInt8, patch_size...)
    W, H, D = size(seg_vol)
    x0, y0, z0 = origin_vox
    x1 = min(W, x0 + patch_size[1] - 1)
    y1 = min(H, y0 + patch_size[2] - 1)
    z1 = min(D, z0 + patch_size[3] - 1)
    (x1 < x0 || y1 < y0 || z1 < z0) && return patch
    target = T(lesion_id)
    @inbounds for z in z0:z1
        pz = z - z0 + 1
        for y in y0:y1
            py = y - y0 + 1
            for x in x0:x1
                if seg_vol[x, y, z] == target
                    patch[x - x0 + 1, py, pz] = UInt8(1)
                end
            end
        end
    end
    return patch
end

@inline function _in_patch(x::Int, y::Int, z::Int, origin_vox::NTuple{3, Int})::Bool
    return (origin_vox[1] <= x < origin_vox[1] + PATCH_SIZE[1]) &&
           (origin_vox[2] <= y < origin_vox[2] + PATCH_SIZE[2]) &&
           (origin_vox[3] <= z < origin_vox[3] + PATCH_SIZE[3])
end

"""
    are_voxels_connected_26(seg_vol, lesion_id, seed_a, seed_b; margin=32) -> Bool

Memory-cautious 26-connected BFS bounded to the local bounding box around `seed_a`
and `seed_b` (no full-volume allocation). Returns `true` if `seed_a` (or an adjacent
`lesion_id` voxel) is in the same connected component of `seg_vol .== lesion_id` as `seed_b`.
"""
function are_voxels_connected_26(seg_vol::AbstractArray{T, 3}, lesion_id::Integer,
                                 seed_a::NTuple{3, Int}, seed_b::NTuple{3, Int};
                                 margin::Int=40)::Bool where {T}
    W, H, D = size(seg_vol)
    target = T(lesion_id)

    # Locate nearest target voxel within 3x3x3 around a seed if the exact seed was erased/shifted
    function _find_local_seed(s::NTuple{3, Int})
        sx, sy, sz = clamp(s[1], 1, W), clamp(s[2], 1, H), clamp(s[3], 1, D)
        seg_vol[sx, sy, sz] == target && return (sx, sy, sz)
        for dz in -2:2, dy in -2:2, dx in -2:2
            nx, ny, nz = sx + dx, sy + dy, sz + dz
            if 1 <= nx <= W && 1 <= ny <= H && 1 <= nz <= D && seg_vol[nx, ny, nz] == target
                return (nx, ny, nz)
            end
        end
        return nothing
    end

    a = _find_local_seed(seed_a)
    b = _find_local_seed(seed_b)
    (a === nothing || b === nothing) && return false
    a == b && return true

    x0 = max(1, min(a[1], b[1]) - margin)
    x1 = min(W, max(a[1], b[1]) + margin)
    y0 = max(1, min(a[2], b[2]) - margin)
    y1 = min(H, max(a[2], b[2]) + margin)
    z0 = max(1, min(a[3], b[3]) - margin)
    z1 = min(D, max(a[3], b[3]) + margin)

    bw, bh, bd = x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1
    visited = falses(bw, bh, bd)
    queue = NTuple{3, Int}[a]
    visited[a[1] - x0 + 1, a[2] - y0 + 1, a[3] - z0 + 1] = true

    head = 1
    @inbounds while head <= length(queue)
        cx, cy, cz = queue[head]
        head += 1
        for dz in -1:1, dy in -1:1, dx in -1:1
            (dx == 0 && dy == 0 && dz == 0) && continue
            nx, ny, nz = cx + dx, cy + dy, cz + dz
            if x0 <= nx <= x1 && y0 <= ny <= y1 && z0 <= nz <= z1
                lx, ly, lz = nx - x0 + 1, ny - y0 + 1, nz - z0 + 1
                if !visited[lx, ly, lz] && seg_vol[nx, ny, nz] == target
                    (nx, ny, nz) == b && return true
                    visited[lx, ly, lz] = true
                    push!(queue, (nx, ny, nz))
                end
            end
        end
    end
    return false
end

function _find_connected_record(tp_idx::Int, lesion_id::Int,
                                seg_vol::AbstractArray{T, 3},
                                seed_vox::NTuple{3, Int}) where {T}
    recs = get(_records, (tp_idx, lesion_id), nothing)
    (recs === nothing || isempty(recs)) && return nothing
    for rec in reverse(recs)
        if are_voxels_connected_26(seg_vol, lesion_id, rec.last_seed_vox, seed_vox) ||
           are_voxels_connected_26(seg_vol, lesion_id, rec.seed_vox, seed_vox)
            return rec
        end
    end
    return nothing
end

function _find_record_for_voxel(tp_idx::Int, lesion_id::Int, vox::NTuple{3, Int})
    recs = get(_records, (tp_idx, lesion_id), nothing)
    (recs === nothing || isempty(recs)) && return nothing
    for rec in reverse(recs)
        _in_patch(vox[1], vox[2], vox[3], rec.origin_vox) && return rec
        for sp in rec.subpatches
            _in_patch(vox[1], vox[2], vox[3], sp.origin_vox) && return rec
        end
    end
    return nothing
end

function _sync_record_masks!(rec::LesionEditRecord, seg_vol::AbstractArray{T, 3}) where {T}
    rec.final_mask .= extract_patch_64(seg_vol, rec.lesion_id, rec.origin_vox)
    for sp in rec.subpatches
        sp.final_mask .= extract_patch_64(seg_vol, rec.lesion_id, sp.origin_vox)
        sp.dirty = true
    end
    rec.dirty = true
end

function _ensure_subpatch_for_seed!(rec::LesionEditRecord, seg_vol::AbstractArray{T, 3},
                                    seed_vox::NTuple{3, Int})::Int where {T}
    _in_patch(seed_vox[1], seed_vox[2], seed_vox[3], rec.origin_vox) && return 0
    for sp in rec.subpatches
        if _in_patch(seed_vox[1], seed_vox[2], seed_vox[3], sp.origin_vox)
            return sp.subpatch_idx
        end
    end
    bounds = compute_patch_bounds(seed_vox[1], seed_vox[2], seed_vox[3], size(seg_vol))
    sp_idx = length(rec.subpatches) + 1
    sp_mask = extract_patch_64(seg_vol, rec.lesion_id, bounds.origin_vox)
    push!(rec.subpatches, OverflowSubpatch(
        subpatch_idx = sp_idx,
        origin_vox = bounds.origin_vox,
        center_vox = bounds.center_vox,
        center_mm = bounds.center_mm,
        final_mask = sp_mask,
        manual_painted_mask = zeros(UInt8, PATCH_SIZE...),
        manual_erased_mask = zeros(UInt8, PATCH_SIZE...),
        dirty = true,
    ))
    return sp_idx
end

"""
    record_heatgdt_event!(tp_idx, lesion_id, seed_vox, seg_vol; theta=0.001f0, K=40, dilation=0, smoothing=0.0f0, edge_modality="combined")

Record a Heat-GDT segmentation/growth event. If `(tp_idx, lesion_id)` already has a
record whose voxels are 26-connected to `seed_vox` (single connected component, e.g.,
elongated lesion), appends to that same lesion record (creating an overflow `64×64×64`
sub-patch if needed). Otherwise creates a new lesion record centered on `seed_vox`.
"""
function record_heatgdt_event!(tp_idx::Int, lesion_id::Int, seed_vox::NTuple{3, Int},
                               seg_vol::AbstractArray{T, 3};
                               theta::Float32=0.001f0, K::Int=40, dilation::Int=0,
                               smoothing::Float32=0.0f0, edge_modality::String="combined") where {T}
    isempty(_edits_h5_ref[]) && return nothing
    lesion_id <= 0 && return nothing
    ts = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS.sss")
    lock(_REC_LOCK) do
        rec = _find_connected_record(tp_idx, lesion_id, seg_vol, seed_vox)
        if rec === nothing
            _seq_counter[] += 1
            bounds = compute_patch_bounds(seed_vox[1], seed_vox[2], seed_vox[3], size(seg_vol))
            init_mask = extract_patch_64(seg_vol, lesion_id, bounds.origin_vox)
            uid = "lesion_$(lesion_id)_tp$(tp_idx)_c$(seed_vox[1])_$(seed_vox[2])_$(seed_vox[3])_$(_seq_counter[])"
            rec = LesionEditRecord(
                uid = uid,
                tp_index = tp_idx,
                lesion_id = lesion_id,
                seed_vox = seed_vox,
                last_seed_vox = seed_vox,
                origin_vox = bounds.origin_vox,
                center_vox = bounds.center_vox,
                center_mm = bounds.center_mm,
                initial_heat_mask = copy(init_mask),
                final_mask = copy(init_mask),
                manual_painted_mask = zeros(UInt8, PATCH_SIZE...),
                manual_erased_mask = zeros(UInt8, PATCH_SIZE...),
            )
            push!(get!(_records, (tp_idx, lesion_id), LesionEditRecord[]), rec)
            push!(rec.pending_events, PendingEditEvent(
                seq = rec.next_event_seq,
                event_type = "heat_gdt",
                timestamp = ts,
                seed_vox = seed_vox,
                theta = theta,
                K = K,
                dilation = dilation,
                smoothing = smoothing,
                edge_modality = edge_modality,
                subpatch_idx = 0,
                patch_data = init_mask,
            ))
            rec.next_event_seq += 1
        else
            sp_idx = _ensure_subpatch_for_seed!(rec, seg_vol, seed_vox)
            rec.last_seed_vox = seed_vox
            _sync_record_masks!(rec, seg_vol)
            ev_origin = sp_idx == 0 ? rec.origin_vox : rec.subpatches[sp_idx].origin_vox
            ev_mask = extract_patch_64(seg_vol, lesion_id, ev_origin)
            push!(rec.pending_events, PendingEditEvent(
                seq = rec.next_event_seq,
                event_type = "heat_gdt",
                timestamp = ts,
                seed_vox = seed_vox,
                theta = theta,
                K = K,
                dilation = dilation,
                smoothing = smoothing,
                edge_modality = edge_modality,
                subpatch_idx = sp_idx,
                patch_data = ev_mask,
            ))
            rec.next_event_seq += 1
        end
        return rec.uid
    end
end

"""
    record_posthoc_event!(tp_idx, lesion_id, event_type, seg_vol; theta=0.001f0, dilation=0, smoothing=0.0f0)

Record a post-hoc Size ± / Dilation / Smoothing edit on the latest active record for `(tp_idx, lesion_id)`.
"""
function record_posthoc_event!(tp_idx::Int, lesion_id::Int, event_type::String,
                               seg_vol::AbstractArray{T, 3};
                               theta::Float32=0.001f0, dilation::Int=0,
                               smoothing::Float32=0.0f0) where {T}
    isempty(_edits_h5_ref[]) && return nothing
    lesion_id <= 0 && return nothing
    ts = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS.sss")
    lock(_REC_LOCK) do
        recs = get(_records, (tp_idx, lesion_id), nothing)
        (recs === nothing || isempty(recs)) && return nothing
        rec = recs[end]
        _sync_record_masks!(rec, seg_vol)
        push!(rec.pending_events, PendingEditEvent(
            seq = rec.next_event_seq,
            event_type = event_type,
            timestamp = ts,
            seed_vox = rec.last_seed_vox,
            theta = theta,
            dilation = dilation,
            smoothing = smoothing,
            subpatch_idx = 0,
            patch_data = copy(rec.final_mask),
        ))
        rec.next_event_seq += 1
        return rec.uid
    end
end

"""
    record_brush_voxels!(tp_idx, lesion_id, coords_3d, is_erase, seg_vol; brush_radius=1)

Record manual brush paint (`is_erase=false`) or erase (`is_erase=true`) voxels onto
any matching Heat-GDT lesion patch (or overflow sub-patch).
"""
function record_brush_voxels!(tp_idx::Int, lesion_id::Int,
                              coords_3d::Vector{NTuple{3, Int}},
                              is_erase::Bool,
                              seg_vol::AbstractArray{T, 3};
                              brush_radius::Int=1) where {T}
    isempty(_edits_h5_ref[]) && return nothing
    (lesion_id <= 0 || isempty(coords_3d)) && return nothing
    ts = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS.sss")
    lock(_REC_LOCK) do
        recs = get(_records, (tp_idx, lesion_id), nothing)
        (recs === nothing || isempty(recs)) && return nothing

        # Group changed voxels by matching record & patch (0=primary, >0=subpatch)
        per_rec_deltas = Dict{Tuple{String, Int}, Tuple{LesionEditRecord, Array{UInt8, 3}}}()
        for v in coords_3d
            rec = _find_record_for_voxel(tp_idx, lesion_id, v)
            rec === nothing && continue
            if _in_patch(v[1], v[2], v[3], rec.origin_vox)
                lx = v[1] - rec.origin_vox[1] + 1
                ly = v[2] - rec.origin_vox[2] + 1
                lz = v[3] - rec.origin_vox[3] + 1
                key = (rec.uid, 0)
                if !haskey(per_rec_deltas, key)
                    per_rec_deltas[key] = (rec, zeros(UInt8, PATCH_SIZE...))
                end
                per_rec_deltas[key][2][lx, ly, lz] = UInt8(1)
                if is_erase
                    rec.manual_erased_mask[lx, ly, lz] = UInt8(1)
                else
                    rec.manual_painted_mask[lx, ly, lz] = UInt8(1)
                end
            else
                for sp in rec.subpatches
                    if _in_patch(v[1], v[2], v[3], sp.origin_vox)
                        lx = v[1] - sp.origin_vox[1] + 1
                        ly = v[2] - sp.origin_vox[2] + 1
                        lz = v[3] - sp.origin_vox[3] + 1
                        key = (rec.uid, sp.subpatch_idx)
                        if !haskey(per_rec_deltas, key)
                            per_rec_deltas[key] = (rec, zeros(UInt8, PATCH_SIZE...))
                        end
                        per_rec_deltas[key][2][lx, ly, lz] = UInt8(1)
                        if is_erase
                            sp.manual_erased_mask[lx, ly, lz] = UInt8(1)
                        else
                            sp.manual_painted_mask[lx, ly, lz] = UInt8(1)
                        end
                        sp.dirty = true
                        break
                    end
                end
            end
        end

        ev_type = is_erase ? "manual_erase" : "manual_paint"
        for ((_, sp_idx), (rec, delta_patch)) in per_rec_deltas
            _sync_record_masks!(rec, seg_vol)
            push!(rec.pending_events, PendingEditEvent(
                seq = rec.next_event_seq,
                event_type = ev_type,
                timestamp = ts,
                seed_vox = coords_3d[1],
                brush_radius = brush_radius,
                subpatch_idx = sp_idx,
                patch_data = delta_patch,
            ))
            rec.next_event_seq += 1
        end
        return length(per_rec_deltas)
    end
end

function _write_compressed_ds!(grp, name::String, arr::AbstractArray)
    if haskey(grp, name)
        HDF5.delete_object(grp, name)
    end
    chk = ntuple(d -> min(size(arr, d), 64), ndims(arr))
    dset = HDF5.create_dataset(grp, name, datatype(eltype(arr)), dataspace(size(arr)); chunk=chk, deflate=4)
    write(dset, arr)
    return dset
end

function _ensure_base_volumes_in_h5!(h5::HDF5.File, ct_vol, pet_vol)
    m_grp = haskey(h5, "_meta_") ? h5["_meta_"] : HDF5.create_group(h5, "_meta_")
    attrs(m_grp)["case_id"] = _case_id_ref[]
    attrs(m_grp)["source_h5"] = _source_h5_ref[]
    attrs(m_grp)["spacing_mm"] = collect(_spacing_ref[])
    attrs(m_grp)["origin_mm"] = collect(_origin_ref[])

    if ct_vol !== nothing && !haskey(h5, "CT/volume")
        ct_grp = haskey(h5, "CT") ? h5["CT"] : HDF5.create_group(h5, "CT")
        _write_compressed_ds!(ct_grp, "volume", ct_vol)
        attrs(m_grp)["ct_size"] = collect(size(ct_vol))
    end
    if pet_vol !== nothing && !haskey(h5, "PET/volume")
        pet_grp = haskey(h5, "PET") ? h5["PET"] : HDF5.create_group(h5, "PET")
        _write_compressed_ds!(pet_grp, "volume", pet_vol)
        attrs(m_grp)["pet_size"] = collect(size(pet_vol))
    end
end

"""
    flush_all_edits!(; ct_vol=nothing, pet_vol=nothing) -> Bool

Flush all dirty lesion `64×64×64` final patches and pending edit events to the
separate `lesion_edits.h5` file, freeing flushed event arrays from RAM.
"""
function flush_all_edits!(; ct_vol=nothing, pet_vol=nothing)::Bool
    edits_path = _edits_h5_ref[]
    isempty(edits_path) && return false

    # Snapshot dirty records and drain pending_events under _REC_LOCK
    to_flush = Tuple{LesionEditRecord, Vector{PendingEditEvent}}[]
    lock(_REC_LOCK) do
        for recs in values(_records)
            for rec in recs
                if rec.dirty || !isempty(rec.pending_events)
                    evs = copy(rec.pending_events)
                    empty!(rec.pending_events)
                    rec.dirty = false
                    push!(to_flush, (rec, evs))
                end
            end
        end
    end
    isempty(to_flush) && return false

    lock(_H5_IO_LOCK) do
        mkpath(dirname(edits_path))
        mode = isfile(edits_path) ? "r+" : "w"
        HDF5.h5open(edits_path, mode) do h5
            _ensure_base_volumes_in_h5!(h5, ct_vol, pet_vol)
            lesions_grp = haskey(h5, "lesions") ? h5["lesions"] : HDF5.create_group(h5, "lesions")

            for (rec, evs) in to_flush
                lgrp = haskey(lesions_grp, rec.uid) ? lesions_grp[rec.uid] : HDF5.create_group(lesions_grp, rec.uid)
                la = attrs(lgrp)
                la["lesion_id"] = rec.lesion_id
                la["tp_index"] = rec.tp_index
                la["seed_vox"] = collect(rec.seed_vox)
                la["last_seed_vox"] = collect(rec.last_seed_vox)
                la["chunk_origin_vox"] = collect(rec.origin_vox)
                la["chunk_size"] = collect(PATCH_SIZE)
                la["chunk_center_vox"] = collect(rec.center_vox)
                la["chunk_center_mm"] = collect(rec.center_mm)

                if !haskey(lgrp, "initial_heat_mask")
                    _write_compressed_ds!(lgrp, "initial_heat_mask", rec.initial_heat_mask)
                end
                _write_compressed_ds!(lgrp, "final_mask", rec.final_mask)
                _write_compressed_ds!(lgrp, "manual_painted_mask", rec.manual_painted_mask)
                _write_compressed_ds!(lgrp, "manual_erased_mask", rec.manual_erased_mask)

                if !isempty(rec.subpatches)
                    sp_root = haskey(lgrp, "subpatches") ? lgrp["subpatches"] : HDF5.create_group(lgrp, "subpatches")
                    for sp in rec.subpatches
                        if sp.dirty || !haskey(sp_root, "subpatch_$(sp.subpatch_idx)")
                            sp_name = "subpatch_$(sp.subpatch_idx)"
                            sp_grp = haskey(sp_root, sp_name) ? sp_root[sp_name] : HDF5.create_group(sp_root, sp_name)
                            spa = attrs(sp_grp)
                            spa["chunk_origin_vox"] = collect(sp.origin_vox)
                            spa["chunk_center_vox"] = collect(sp.center_vox)
                            spa["chunk_center_mm"] = collect(sp.center_mm)
                            _write_compressed_ds!(sp_grp, "final_mask", sp.final_mask)
                            _write_compressed_ds!(sp_grp, "manual_painted_mask", sp.manual_painted_mask)
                            _write_compressed_ds!(sp_grp, "manual_erased_mask", sp.manual_erased_mask)
                            sp.dirty = false
                        end
                    end
                end

                ev_root = haskey(lgrp, "events") ? lgrp["events"] : HDF5.create_group(lgrp, "events")
                for ev in evs
                    ev_name = lpad(ev.seq, 4, '0') * "_" * ev.event_type
                    ev_ds = _write_compressed_ds!(ev_root, ev_name, ev.patch_data)
                    eva = attrs(ev_ds)
                    eva["event_type"] = ev.event_type
                    eva["timestamp"] = ev.timestamp
                    eva["seed_vox"] = collect(ev.seed_vox)
                    eva["theta"] = ev.theta
                    eva["K"] = ev.K
                    eva["dilation"] = ev.dilation
                    eva["smoothing"] = ev.smoothing
                    eva["edge_modality"] = ev.edge_modality
                    eva["brush_radius"] = ev.brush_radius
                    eva["subpatch_idx"] = ev.subpatch_idx
                end
            end
        end
    end
    return true
end

"""
    flush_all_edits_async!(; ct_vol=nothing, pet_vol=nothing) -> Task

Asynchronously flush all dirty lesion patches and edit events to `lesion_edits.h5`
on a background worker thread without blocking the UI or render loop.
"""
function flush_all_edits_async!(; ct_vol=nothing, pet_vol=nothing)::Task
    return Threads.@spawn begin
        try
            flush_all_edits!(ct_vol=ct_vol, pet_vol=pet_vol)
        catch e
            @warn "LesionEditsRecorder async flush failed" exception=(e, catch_backtrace())
            false
        end
    end
end

end # module LesionEditsRecorder
