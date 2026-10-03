module DagVm
using ImageMorphology



struct LevelOutputDict
    mapping::Dict{String, Int}
    buf::AbstractArray
    backend::Any
end
import Base: getindex, setindex!, haskey, keys, empty!, get, length, iterate

function LevelOutputDict(buf, backend)
    LevelOutputDict(Dict{String, Int}(), buf, backend)
end

function Base.empty!(d::LevelOutputDict)
    empty!(d.mapping)
    return d
end

function Base.haskey(d::LevelOutputDict, key::String)
    return haskey(d.mapping, key)
end

function Base.keys(d::LevelOutputDict)
    return keys(d.mapping)
end

function Base.length(d::LevelOutputDict)
    return length(d.mapping)
end

function Base.getindex(d::LevelOutputDict, key::String)
    if haskey(d.mapping, key)
        return view(d.buf, :, :, :, d.mapping[key])
    end
    throw(KeyError(key))
end

function Base.get(d::LevelOutputDict, key::String, default)
    if haskey(d.mapping, key)
        return view(d.buf, :, :, :, d.mapping[key])
    end
    return default
end

function Base.setindex!(d::LevelOutputDict, val, key::String)
    if !haskey(d.mapping, key)
        d.mapping[key] = length(d.mapping) + 1
        if d.mapping[key] > size(d.buf, 4)
            error("LevelOutputDict capacity exceeded!")
        end
    end
    if val === nothing
        return val
    end
    idx = d.mapping[key]
    v = view(d.buf, :, :, :, idx)
    if val !== v
        if val isa CuArray
            copyto!(v, val)
        else
            v .= adapt(d.backend, UInt8.(val .> 0))
        end
    end
    return val
end

function Base.iterate(d::LevelOutputDict, state...)
    it = iterate(d.mapping, state...)
    if it === nothing
        return nothing
    end
    (k, idx), next_state = it
    return (k => view(d.buf, :, :, :, idx)), next_state
end

struct GpuMasksDict <: AbstractDict{String, Any}
    masks::Dict{String, Any}
    get_mask::Function
end
Base.haskey(d::GpuMasksDict, k::String) = haskey(d.masks, k) || d.get_mask(k) !== nothing
Base.getindex(d::GpuMasksDict, k::String) = (m = d.get_mask(k); m !== nothing ? m : d.masks[k])
Base.length(d::GpuMasksDict) = length(d.masks)
Base.iterate(d::GpuMasksDict, state...) = iterate(d.masks, state...)



using JSON
using Adapt
using KernelAbstractions
using CUDA
using Statistics

# ── Debug Save Flag ────────────────────────────────────────────────────────────
# Set JL_DEBUG_SAVE=1 to enable per-stage intermediate mask saves.
# Production runs must NOT set this — it causes significant IO overhead.
# Optional: JL_DEBUG_SEGMENT="Neck_Parotid_left"  → only save that segment
#           JL_DEBUG_DIR="/tmp/jl_debug"           → output directory
const DEBUG_SAVE    = get(ENV, "JL_DEBUG_SAVE",    "0") == "1"
const DEBUG_SEGMENT = get(ENV, "JL_DEBUG_SEGMENT", "")   # empty = all segments
const DEBUG_DIR     = get(ENV, "JL_DEBUG_DIR",     "/tmp/jl_debug")
const DEBUG_VERBOSE = get(ENV, "JL_DEBUG_VERBOSE", "0") == "1"  # Per-rule voxel loss tracing
# ──────────────────────────────────────────────────────────────────────────────

include("StaticArena.jl")
using .StaticArena
include("GpuCCL.jl")
using .GpuCCL
include("MaskPacker.jl")
include("VMOpcodes.jl")
include("VMCompiler.jl")
include("VMKernel.jl")

using .VMOpcodes
using .VMCompiler
using .VMKernel
include("InstructionSet.jl")
include("VmKernel.jl")
include("RuleExecutors.jl")
include("kernels_custom_rules.jl")
include("kernels_convexhull.jl")
include("layer_wise_propagation.jl")
include("../preprocessing/Landmarks.jl")
include("../preprocessing/GpuLandmarks.jl")
include("../preprocessing/PrimaryHDF5.jl")

using .MaskPacker
using .InstructionSet
using .VmKernel
using .RuleExecutors
using .CustomRules
using .Landmarks
using .GpuLandmarks
using .PrimaryHDF5
include("overlap_resolution.jl")

export run_dag_vm_pipeline, run_pipeline, resolve_overlaps_cpu!

"""
    run_pipeline(h5_path::String; json_dir::String="jsons", backend=CUDA.functional() ? CUDABackend() : CPU(), resolve_overlaps::Bool=true)
"""
function run_pipeline(h5_path::String; json_dir::String="jsons", backend=CUDA.functional() ? CUDABackend() : CPU(), resolve_overlaps::Bool=true)
    dims, spacing, origin, direction = PrimaryHDF5.load_primary_metadata(h5_path)
    masks = PrimaryHDF5.load_primary_masks(h5_path; required_names=nothing)
    computed_landmarks = PrimaryHDF5.load_primary_landmarks(h5_path)
    # Coracoid process Z (most anterior voxel of scapula)
    for side in ["left", "right"]
        scap_m = get(masks, "scapula_$side", nothing)
        if scap_m === nothing; scap_m = get(masks, "scapula", nothing); end
        if scap_m !== nothing
            scap_m_gpu = backend isa CPU ? scap_m : KernelAbstractions.adapt(backend, scap_m)
            min_y_cpu, max_y_cpu = RuleExecutors.get_slice_y_bounds_gpu(backend, scap_m_gpu, dims)
            min_y = dims[2] + 1
            best_k = 0
            for k in 1:dims[3]
                if max_y_cpu[k] > 0 && min_y_cpu[k] < min_y
                    min_y = min_y_cpu[k]
                    best_k = k
                end
            end
            if best_k > 0
                z_mm = origin[3] + (best_k - 1) * spacing[3]
                computed_landmarks["coracoid_process_$side"] = [0.0, 0.0, z_mm]
                println("    [DEBUG] Computed coracoid_process_$side Z = $z_mm mm")
            end
        end
    end


    gen_masks = run_dag_vm_pipeline(json_dir, masks, computed_landmarks, dims, spacing, origin, direction; backend=backend)
    # Overlaps are now natively resolved inside run_dag_vm_pipeline using PackedTensor
    return gen_masks
end

gpu_count(m) = m === nothing ? 0 : Int(mapreduce(x -> Int64(x > UInt8(0)), +, m; init=Int64(0)))

"""
    load_all_rules(json_dir::String) -> Dict{String, Dict}
"""
function load_all_rules(json_dir::String)
    rules = Dict{String, Dict}()
    for f in readdir(json_dir, join=true)
        if endswith(f, ".json") && !occursin("MANUAL_DEFS_V21.json", f) && !occursin("_v21", f)
            try
                d = JSON.parsefile(f)
                if d isa Dict
                    for (k, v) in d
                        rules[String(k)] = v
                    end
                end
            catch e
                println("Warning loading rule $f: $e")
            end
        end
    end
    return rules
end

"""
    build_dag_levels(rules::Dict{String, Dict}) -> Vector{Vector{String}}
Stratifies the rule dependency graph into minimal topological levels.
"""
function build_dag_levels(rules::AbstractDict; max_per_level::Int=15)
    function extract_deps(params)
        deps = Set{String}()
        function _ext(o)
            if o isa String
                push!(deps, o)
                base_s = replace(o, r"(_left|_right|_Left|_Right|_l|_r|_L|_R)$" => "")
                if base_s != o
                    # It already has a suffix, so do NOT expand laterally!
                    push!(deps, base_s)
                else
                    # It is a base name, so expand to catch bilateral outputs
                    for suffix in ["_Left", "_Right", "_left", "_right"]
                        push!(deps, "$(o)$(suffix)")
                    end
                end
            elseif o isa Dict
                for (k, v) in o
                    if k == "overlap_precedence" continue end
                    _ext(v)
                end
            elseif o isa Vector
                for i in o; _ext(i); end
            end
        end
        _ext(params)
        return Set([d for d in deps if haskey(rules, d)])
    end

        dep_graph = Dict(k => extract_deps(v) for (k, v) in rules)
    for (k, v) in dep_graph
        delete!(v, k)
    end
    resolved = Set{String}()
    levels = Vector{Vector{String}}()
    remaining = copy(dep_graph)

    while !isempty(remaining)
        curr = [k for (k, d) in remaining if issubset(d, resolved)]
        if isempty(curr)
            # Break cycles if any by picking remaining with smallest deps
            min_k = sort(collect(keys(remaining)), by=k -> length(remaining[k]))[1]
            curr = [min_k]
        end
        if length(curr) > max_per_level
            for chunk in Iterators.partition(sort(curr), max_per_level)
                push!(levels, collect(chunk))
            end
        else
            push!(levels, sort(curr))
        end
        union!(resolved, curr)
        for k in curr
            delete!(remaining, k)
        end
    end

    return levels
end

"""
    resolve_mask_from_spec(spec, get_mask_fn, dims, backend)
Resolves a mask specification which may be a String, a Vector of names (Union), or a Dict.
"""
function resolve_mask_from_spec(spec, get_mask_fn::Function, dims::Tuple{Int, Int, Int}, backend)
    if spec === nothing
        return nothing
    elseif spec isa String
        return get_mask_fn(spec)
    elseif spec isa Vector
        res = KernelAbstractions.zeros(backend, UInt8, dims)
        found_any = false
        for item in spec
            m = resolve_mask_from_spec(item, get_mask_fn, dims, backend)
            if m !== nothing
                m_u8 = map(x -> x > 0 ? UInt8(1) : UInt8(0), m)
                res .|= adapt(typeof(res), m_u8)
                found_any = true
            end
        end
        return found_any ? res : nothing
    elseif spec isa Dict
        name = get(spec, "landmark", get(spec, "name", get(spec, "base_landmark", "")))
        return !isempty(name) ? get_mask_fn(String(name)) : nothing
    else
        return nothing
    end
end

"""
    run_dag_vm_pipeline(...)
Executes the full 6-step VM GPU pipeline.
"""
function run_dag_vm_pipeline(
    json_dir::String,
    raw_masks::AbstractDict,
    computed_landmarks::Dict{String, Any},
    dims::Tuple{Int, Int, Int},
    spacing::Tuple{Float64, Float64, Float64},
    origin::Tuple{Float64, Float64, Float64},
    direction::Tuple;
    backend::KernelAbstractions.Backend = CUDA.functional() ? CUDABackend() : CPU()
)
    println("================================================================================")
    println("           DAG VM GPU Pipeline Execution (KernelAbstractions)                   ")
    println("================================================================================")
    println("  -> Backend: $backend | Dimensions: $dims | Spacing: $spacing")
    
    # 1. Step 1: Input Collection & Bilateral Unions (CPU memory efficient)
    println("\n[Step 1] Initializing Input Mask Collection...")
    masks = Dict{String, Any}()
    for (k, v) in raw_masks
        masks[k] = v
    end
    
    # Bilateral unions for paired organs
    paired = ["clavicle", "clavicula", "femur", "hip", "scapula", "humerus", "pectoralis_major", "pectoralis_minor",
              "internal_jugular_vein", "common_carotid_artery", "internal_carotid_artery", "subclavian_artery", "subclavian_vein",
              "parotid_gland", "submandibular_gland", "iliac_artery", "iliac_vena", "lung", "kidney",
              "colon", "small_bowel", "sternocleidomastoid", "latissimus_dorsi", "serratus_anterior",
              "iliopsoas", "psoas_major", "psoas", "obturator_internus", "obturator_externus", "obturator_foramen",
              "gluteus_maximus", "gluteus_medius", "gluteus_minimus", "masseter", "pulmonary_artery", "pulmonary_vein",
              "acetabulum_proxy"]
    
    for p in paired
        ml = get(masks, "$(p)_left", nothing)
        mr = get(masks, "$(p)_right", nothing)
        if ml !== nothing && mr !== nothing
            masks[p] = ml .| mr
        elseif ml !== nothing
            masks[p] = copy(ml)
        elseif mr !== nothing
            masks[p] = copy(mr)
        end
    end
    if haskey(masks, "clavicula_left"); masks["clavicle_left"] = masks["clavicula_left"]; end
    if haskey(masks, "clavicula_right"); masks["clavicle_right"] = masks["clavicula_right"]; end
    # Presacral sacrum: ensure fused sacrum includes S1 (vertebrae_S1)
    s1_mask = get(masks, "vertebrae_S1", get(masks, "vertebrae_s1", nothing))
    if haskey(masks, "sacrum")
        if s1_mask !== nothing
            fused_sac = masks["sacrum"] .| s1_mask
            masks["helper_sacrum_fused"] = fused_sac
            masks["sacrum"] = fused_sac
        else
            masks["helper_sacrum_fused"] = copy(masks["sacrum"])
        end
    elseif s1_mask !== nothing
        masks["helper_sacrum_fused"] = copy(s1_mask)
        masks["sacrum"] = copy(s1_mask)
    end
    if haskey(masks, "urinary_bladder"); masks["bladder"] = masks["urinary_bladder"]; end

    # Build fused vertebral column: pairwise coronal+sagittal convex hull bridging
    # between adjacent vertebrae to fill intervertebral disc gaps while preserving curvature
    println("  -> Building fused vertebral column (pairwise hull bridging)...")
    vertebrae_order = [
        "vertebrae_C1", "vertebrae_C2", "vertebrae_C3", "vertebrae_C4", "vertebrae_C5", "vertebrae_C6", "vertebrae_C7",
        "vertebrae_T1", "vertebrae_T2", "vertebrae_T3", "vertebrae_T4", "vertebrae_T5", "vertebrae_T6", "vertebrae_T7",
        "vertebrae_T8", "vertebrae_T9", "vertebrae_T10", "vertebrae_T11", "vertebrae_T12",
        "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5",
        "vertebrae_S1", "sacrum"
    ]
    present_verts = [k for k in vertebrae_order if haskey(masks, k) && masks[k] !== nothing && any(masks[k] .> 0)]
    if length(present_verts) >= 2
        fused_spine_gpu = KernelAbstractions.zeros(backend, UInt8, dims)
        # Base union of all present vertebrae
        for k in present_verts
            m_gpu = backend isa CPU ? masks[k] : adapt(backend, masks[k])
            fused_spine_gpu .|= (m_gpu .> UInt8(0))
        end
        # Pairwise convex hull bridging between adjacent vertebrae
        n_bridged = 0
        for i in 1:(length(present_verts) - 1)
            k1, k2 = present_verts[i], present_verts[i+1]
            m1 = backend isa CPU ? masks[k1] : adapt(backend, masks[k1])
            m2 = backend isa CPU ? masks[k2] : adapt(backend, masks[k2])
            try
                cor = RuleExecutors.execute_convex_hull_bridge(backend, m1, m2, dims, spacing, origin, direction; plane="coronal")
                sag = RuleExecutors.execute_convex_hull_bridge(backend, m1, m2, dims, spacing, origin, direction; plane="sagittal")
                # Intersect coronal & sagittal bridges and accumulate
                fused_spine_gpu .|= (cor .& sag)
                n_bridged += 1
            catch e
                # Skip pairs that fail (e.g. too few voxels for hull)
            end
        end
        fused_spine_cpu = Array(fused_spine_gpu)
        masks["fused_spine"] = fused_spine_cpu
        masks["vertebral_column_fused"] = fused_spine_cpu
        println("  -> Fused spine: $(sum(fused_spine_cpu .> 0)) voxels, $(n_bridged) bridges from $(length(present_verts)) vertebrae")
    end
    if haskey(masks, "thorax_wall"); masks["chest_wall"] = masks["thorax_wall"]; end
    if haskey(masks, "psoas_major"); masks["psoas"] = masks["psoas_major"]; end
    if !haskey(masks, "bronchi_left") && haskey(masks, "bronchi_main_left"); masks["bronchi_left"] = masks["bronchi_main_left"]; end
    if !haskey(masks, "bronchi_right") && haskey(masks, "bronchi_main_right"); masks["bronchi_right"] = masks["bronchi_main_right"]; end

    # Precalculate all dynamic geometric and anatomical landmarks on GPU directly
    t_landmarks_start = time()
    println("  -> Precalculating dynamic anatomical landmarks & bounding planes on GPU...")
    GpuLandmarks.precalculate_all_gpu_landmarks!(computed_landmarks, masks, spacing, origin, direction; backend=backend)
    t_landmarks_end = time()
    println("  -> Computed $(length(computed_landmarks)) dynamic GPU landmarks and planes. [$(round(t_landmarks_end - t_landmarks_start, digits=2))s]")
    
    t_arena_start = time()
    StaticArena.init_edt_arena(backend, dims)
    println("  -> Initialized StaticArena (EDT Arena preallocated).")
    StaticArena.init_vm_arena(backend, dims; max_rules=32)
    println("  -> Initialized StaticArena (VM level_output Arena preallocated: 32 channels UInt8).")
    StaticArena.init_mask_arena(backend, dims; num_masks=12)
    println("  -> Initialized StaticArena (MASK_ARENA preallocated: 20 × UInt8 3D buffers).")
    StaticArena.init_ccl_arena(backend, dims)
    println("  -> Initialized StaticArena (CCL_ARENA preallocated: labels+counts+output).")
    # FT_ARENA disabled — CPU bounding-box crop is faster for typical anatomy masks
    # StaticArena.init_ft_arena(backend, dims)
    t_arena_end = time()
    println("  -> Arena init: $(round(t_arena_end - t_arena_start, digits=2))s")

    # 2. Step 2: Non-overlapping 4D Integer-packed Tensor
    t_packing_start = time()
    println("\n[Step 2] Packing $(length(masks)) input masks into non-overlapping 4D Integer Tensor...")
    GC.gc(false); CUDA.reclaim(); packed_tensor = MaskPacker.pack_masks(masks; backend=backend)
    t_packing_end = time()
    println("  -> Packing done: $(round(t_packing_end - t_packing_start, digits=2))s")
    
    # 3. Load rules & stratify DAG
    rules = load_all_rules(json_dir)
    levels = build_dag_levels(rules; max_per_level=20)
    println("\n[DAG Structure] Stratified $(length(rules)) rules into $(length(levels)) levels:")
    for (idx, lvl) in enumerate(levels)
        GC.gc(false); CUDA.reclaim()
        println("  -> Level $idx: $(length(lvl)) rules")
    end

    gen_masks = Dict{String, Any}()

    # Fix 17: Add mask aliases for aortic_arch and carina so that z_plane_restriction can
    # resolve "aortic_arch_bottom" (strips _bottom → looks for "aortic_arch" via get_mask_fn).
    # The HDF5 has "aortic_arch_computed" mask, not "aortic_arch". Adding to ALIASES below allows
    # resolve_landmark_z to use the mask's full bbox for eff_part="min"/"max" computation.
    # NOTE: "aortic_arch" and "carina" aliases are added directly to ALIASES dict below.

    ALIASES = Dict{String, String}(
        "lungs" => "lung",
        "kidneys" => "kidney",
        "clavicles" => "clavicula",
        "clavicle" => "clavicula",
        "manubrium" => "sternum",
        "manubrium_left" => "sternum",
        "manubrium_right" => "sternum",
        "bronchi_sanitized" => "bronchi",
        "bronchus_sanitized" => "bronchi",
        "hyoid" => "hyoid",
        "hyoid_bone" => "hyoid",
        "femurs" => "femur",
        "scapulae" => "scapula",
        "scapulas" => "scapula",
        "ribs" => "rib",
        "vertebral_column_fused" => "fused_spine",
        "first_rib_top" => "rib_left_1",
        "bladder" => "urinary_bladder",
        "chest_wall" => "thorax_wall",
        "psoas" => "psoas_major",
        "psoas_major_muscle" => "psoas_major",
        "iliopsoas_muscle" => "iliopsoas",
        "obturator_internus_muscle" => "obturator_internus",
        "obturator_externus_muscle" => "obturator_externus",
        "bronchi_right" => "bronchi_main_right",
        "bronchi_left" => "bronchi_main_left",
        "bronchus_right" => "bronchi_main_right",
        "bronchus_left" => "bronchi_main_left",
        "inguinal_ligament" => "Inguinal_Ligament",
        "inguinal_ligament_left" => "Inguinal_Ligament_Left",
        "inguinal_ligament_right" => "Inguinal_Ligament_Right",
        "inguinal_ligament_proxy" => "Inguinal_Ligament",
        "inguinal_plane" => "inguinal_plane_superior",
        "inguinal_plane_left" => "inguinal_plane_superior_left",
        "inguinal_plane_right" => "inguinal_plane_superior_right",
        "thoracic_prepericardial" => "Thoracic_Prepericardial",
        "thoracic_prepericardial_left" => "Thoracic_Prepericardial_Left",
        "thoracic_prepericardial_right" => "Thoracic_Prepericardial_Right",
        "thoracic_station_3a_prevascular" => "Thoracic_Station_3A_Prevascular",
        "thoracic_station_3a_prevascular_left" => "Thoracic_Station_3A_Prevascular_Left",
        "thoracic_station_3a_prevascular_right" => "Thoracic_Station_3A_Prevascular_Right",
        "thoracic_station_6_paraaortic" => "Thoracic_Station_6_Paraaortic",
        "lung_vessels" => "pulmonary_artery",
        "skull_base" => "skull",
        "clavicle_left" => "clavicula_left",
        "clavicle_right" => "clavicula_right",
        # NOTE: clavicle_*_lower_half is a runtime-computed landmark in Python (inferior 50% of clavicle).
        # It is NOT available as a precomputed mask in the HDF5 file. When Python cannot find it,
        # it skips the MedialTo constraint entirely. To match this behavior, we do NOT alias these
        # names to the full clavicula (which would create an incorrect constraint boundary).
        # Removing these aliases causes get_mask("clavicle_left_lower_half") = nothing → constraint skipped.
        # "clavicle_left_lower_half" => "clavicula_left",   # REMOVED: causes conflicting constraints
        # "clavicle_right_lower_half" => "clavicula_right",  # REMOVED: causes conflicting constraints
        # "clavicula_left_lower_half" => "clavicula_left",   # REMOVED: causes conflicting constraints
        # "clavicula_right_lower_half" => "clavicula_right"  # REMOVED: causes conflicting constraints
        # Fix 17: Aliases for aortic_arch and carina mask lookups.
        # resolve_landmark_z strips "_bottom"/"_top" suffixes → looks for "aortic_arch" in get_mask_fn.
        # The HDF5 has "aortic_arch_computed" mask (not "aortic_arch"), and "carina_computed" (not "carina").
        # These aliases allow the mask bbox to be computed correctly for eff_part="min"/"max".
        "aortic_arch" => "aortic_arch_computed",
        "carina" => "carina_computed",
        "plane_trachea_left" => "trachea",
        "plane_trachea_right" => "trachea",
        "plane_trachea" => "trachea",
        "internal_carotid_artery" => "common_carotid_artery",
        "internal_carotid_artery_left" => "common_carotid_artery_left",
        "internal_carotid_artery_right" => "common_carotid_artery_right",
        "lev_vert" => "levator_scapulae",
        "lev_vert_left" => "levator_scapulae_left",
        "lev_vert_right" => "levator_scapulae_right",
        "adrenals" => "adrenal_gland",
        "adrenal" => "adrenal_gland",
        "adrenals_left" => "adrenal_gland_left",
        "adrenals_right" => "adrenal_gland_right",
        "adrenal_left" => "adrenal_gland_left",
        "adrenal_right" => "adrenal_gland_right",
        # Fix 27: Missing exclusion alias mappings
        "sigmoid" => "colon",
        "bladder" => "urinary_bladder",
        "bladder_left" => "urinary_bladder",
        "bladder_right" => "urinary_bladder",
        "chest_wall" => "thorax_wall",
    )

    composite_cache = Dict{String, Any}()

    function get_mask(name::String)
        if isempty(name) return nothing end
        nl = lowercase(name)
        if haskey(composite_cache, nl) return composite_cache[nl] end
        if haskey(ALIASES, nl) && ALIASES[nl] != nl
            alias_target = ALIASES[nl]
            if haskey(gen_masks, alias_target)
                m = gen_masks[alias_target]
                if !(m isa CuArray) && !(backend isa CPU) && CUDA.functional()
                    res = adapt(backend, map(x -> x > 0 ? UInt8(1) : UInt8(0), m))
                    composite_cache[nl] = res
                    return res
                end
                return m
            end
            if haskey(packed_tensor.registry, alias_target)
                res = MaskPacker.unpack_mask(packed_tensor, alias_target)
                composite_cache[nl] = res
                return res
            end
            if haskey(masks, alias_target)
                m = masks[alias_target]
                res = !(backend isa CPU) && CUDA.functional() ? adapt(backend, map(x -> x > 0 ? UInt8(1) : UInt8(0), m)) : map(x -> x > 0 ? UInt8(1) : UInt8(0), m)
                composite_cache[nl] = res
                return res
            end
        end

        if nl in ["skull_base", "skull_base_plane"]
            sb_l = get_mask("skull_base_plane_left")
            sb_r = get_mask("skull_base_plane_right")
            if sb_l !== nothing || sb_r !== nothing
                sb_u = KernelAbstractions.zeros(backend, UInt8, dims)
                if sb_l !== nothing; sb_u .|= (sb_l .> UInt8(0)); end
                if sb_r !== nothing; sb_u .|= (sb_r .> UInt8(0)); end
                composite_cache[nl] = sb_u
                return sb_u
            end
        end

        if haskey(gen_masks, name)
            m = gen_masks[name]
            if !(m isa CuArray) && !(backend isa CPU) && CUDA.functional()
                res = adapt(backend, map(x -> x > 0 ? UInt8(1) : UInt8(0), m))
                composite_cache[nl] = res
                return res
            end
            return m
        elseif haskey(packed_tensor.registry, name)
            res = MaskPacker.unpack_mask(packed_tensor, name)
            composite_cache[nl] = res
            return res
        elseif haskey(masks, name)
            m = masks[name]
            res = !(backend isa CPU) && CUDA.functional() ? adapt(backend, map(x -> x > 0 ? UInt8(1) : UInt8(0), m)) : map(x -> x > 0 ? UInt8(1) : UInt8(0), m)
            composite_cache[nl] = res
            return res
        end
        # Case insensitive fallback
        for k in keys(gen_masks)
            if lowercase(k) == nl
                m = gen_masks[k]
                if !(m isa CuArray) && !(backend isa CPU) && CUDA.functional()
                    res = adapt(backend, map(x -> x > 0 ? UInt8(1) : UInt8(0), m))
                    composite_cache[nl] = res
                    return res
                end
                return m
            end
        end
        for k in keys(packed_tensor.registry)
            if lowercase(k) == nl
                res = MaskPacker.unpack_mask(packed_tensor, k)
                composite_cache[nl] = res
                return res
            end
        end
        if nl in ["fused_spine", "spine", "vertebrae"]
            spine_u = KernelAbstractions.zeros(backend, UInt8, dims)
            found = false
            for k in keys(masks)
                if lowercase(k) == "fused_spine" || startswith(lowercase(k), "vertebrae_")
                    m = masks[k]
                    spine_u .|= ((backend isa CPU ? m : adapt(backend, m)) .> UInt8(0))
                    found = true
                end
            end
            if found
                composite_cache[nl] = spine_u
                return spine_u
            end
        end
        if nl in ["vertebrae_t", "vertebrae_thoracic", "thoracic_vertebrae"]
            vert_t = KernelAbstractions.zeros(backend, UInt8, dims)
            found = false
            for k in keys(masks)
                if startswith(lowercase(k), "vertebrae_t")
                    m = masks[k]
                    vert_t .|= ((backend isa CPU ? m : adapt(backend, m)) .> UInt8(0))
                    found = true
                end
            end
            if found
                composite_cache[nl] = vert_t
                return vert_t
            end
        end
        if nl in ["vertebrae_c", "vertebrae_cervical", "cervical_vertebrae"]
            vert_c = KernelAbstractions.zeros(backend, UInt8, dims)
            found = false
            for k in keys(masks)
                if startswith(lowercase(k), "vertebrae_c")
                    m = masks[k]
                    vert_c .|= ((backend isa CPU ? m : adapt(backend, m)) .> UInt8(0))
                    found = true
                end
            end
            if found
                composite_cache[nl] = vert_c
                return vert_c
            end
        end
        if nl in ["vertebrae_l", "vertebrae_lumbar", "lumbar_vertebrae"]
            vert_l = KernelAbstractions.zeros(backend, UInt8, dims)
            found = false
            for k in keys(masks)
                if startswith(lowercase(k), "vertebrae_l")
                    m = masks[k]
                    vert_l .|= ((backend isa CPU ? m : adapt(backend, m)) .> UInt8(0))
                    found = true
                end
            end
            if found
                composite_cache[nl] = vert_l
                return vert_l
            end
        end
        if nl in ["vertebrae_s", "vertebrae_sacral", "sacral_vertebrae"]
            vert_s = KernelAbstractions.zeros(backend, UInt8, dims)
            found = false
            for k in keys(masks)
                if startswith(lowercase(k), "vertebrae_s")
                    m = masks[k]
                    vert_s .|= ((backend isa CPU ? m : adapt(backend, m)) .> UInt8(0))
                    found = true
                end
            end
            if found
                composite_cache[nl] = vert_s
                return vert_s
            end
        end
        for k in keys(masks)
            if lowercase(k) == nl
                m = masks[k]
                res = !(backend isa CPU) && CUDA.functional() ? adapt(backend, map(x -> x > 0 ? UInt8(1) : UInt8(0), m)) : map(x -> x > 0 ? UInt8(1) : UInt8(0), m)
                composite_cache[nl] = res
                return res
            end
        end

        # Bilateral union fallback: if "organ" is requested and "organ_left" / "organ_right" exist
        if !endswith(nl, "_left") && !endswith(nl, "_right")
            m_l = haskey(masks, "$(nl)_left") ? masks["$(nl)_left"] : (haskey(masks, "$(nl)_l") ? masks["$(nl)_l"] : nothing)
            m_r = haskey(masks, "$(nl)_right") ? masks["$(nl)_right"] : (haskey(masks, "$(nl)_r") ? masks["$(nl)_r"] : nothing)
            if m_l !== nothing || m_r !== nothing
                u_b = KernelAbstractions.zeros(backend, UInt8, dims)
                if m_l !== nothing; u_b .|= ((backend isa CPU ? m_l : adapt(backend, m_l)) .> UInt8(0)); end
                if m_r !== nothing; u_b .|= ((backend isa CPU ? m_r : adapt(backend, m_r)) .> UInt8(0)); end
                composite_cache[nl] = u_b
                return u_b
            end
        end

        if nl in ["lung", "lungs"]
            lung_u = KernelAbstractions.zeros(backend, UInt8, dims)
            found = false
            for k in keys(masks)
                if lowercase(k) in ["lung", "lung_left", "lung_right"]
                    m = masks[k]
                    lung_u .|= ((backend isa CPU ? m : adapt(backend, m)) .> UInt8(0))
                    found = true
                end
            end
            for k in keys(packed_tensor.registry)
                if lowercase(k) in ["lung", "lung_left", "lung_right"]
                    m = MaskPacker.unpack_mask(packed_tensor, k)
                    lung_u .|= (m .> UInt8(0))
                    found = true
                end
            end
            if found
                composite_cache[nl] = lung_u
                return lung_u
            end
        end

        # pharynx_muscles composite: REMOVED (Fix 11)
        # In Python, pharynx_muscles is NOT a precomputed HDF5 key → get_nifti_image returns None.
        # When Julia had the pharynx_muscles composite (union of 3 constrictors), the PosteriorTo
        # pharynx_muscles constraint in Neck_Retropharyngeal was applied at z-slices where only
        # constrictors exist (not oropharynx/nasopharynx), making Retropharyngeal over-constrained.
        # Now pharynx_muscles is treated as absent (returns nothing) → matches Python behavior.
        # if nl == "pharynx_muscles"
        #     ... (commented out)
        # end


        if nl == "body"

            if haskey(masks, "body")
                res = (backend isa CPU ? map(x -> x > 0 ? UInt8(1) : UInt8(0), masks["body"]) : adapt(backend, map(x -> x > 0 ? UInt8(1) : UInt8(0), masks["body"])))
                composite_cache[nl] = res
                return res
            end
            b_t = get_mask("body_trunc")
            b_e = get_mask("body_extremities")
            if b_t !== nothing || b_e !== nothing
                b_u = KernelAbstractions.zeros(backend, UInt8, dims)
                if b_t !== nothing; b_u .|= (b_t .> UInt8(0)); end
                if b_e !== nothing; b_u .|= (b_e .> UInt8(0)); end
                composite_cache[nl] = b_u
                return b_u
            end
        end

        if nl in ["pelvis", "pelvic_bones"]
            h_l = get_mask("hip_left"); h_r = get_mask("hip_right"); sac = get_mask("sacrum")
            pelvis_u = KernelAbstractions.zeros(backend, UInt8, dims)
            if h_l !== nothing; pelvis_u .|= (h_l .> UInt8(0)); end
            if h_r !== nothing; pelvis_u .|= (h_r .> UInt8(0)); end
            if sac !== nothing; pelvis_u .|= (sac .> UInt8(0)); end
            composite_cache[nl] = pelvis_u
            return pelvis_u
        end

        if nl == "rib_1"
            r_l = get_mask("rib_left_1"); r_r = get_mask("rib_right_1")
            if r_l !== nothing || r_r !== nothing
                rib1_u = KernelAbstractions.zeros(backend, UInt8, dims)
                if r_l !== nothing; rib1_u .|= (r_l .> UInt8(0)); end
                if r_r !== nothing; rib1_u .|= (r_r .> UInt8(0)); end
                composite_cache[nl] = rib1_u
                return rib1_u
            end
        end

        # Generic rib_N -> rib_left_N + rib_right_N (handles rib_2 through rib_12)
        m_rib = match(r"^rib_(\d+)$", nl)
        if m_rib !== nothing
            n = m_rib.captures[1]
            r_l = get_mask("rib_left_$(n)"); r_r = get_mask("rib_right_$(n)")
            if r_l !== nothing || r_r !== nothing
                rib_u = KernelAbstractions.zeros(backend, UInt8, dims)
                if r_l !== nothing; rib_u .|= (r_l .> UInt8(0)); end
                if r_r !== nothing; rib_u .|= (r_r .> UInt8(0)); end
                composite_cache[nl] = rib_u
                return rib_u
            end
        end


        if nl == "manubrium"
            stern = get_mask("sternum")
            if stern !== nothing
                composite_cache[nl] = stern
                return stern
            end
        end

        # Bilateral union fallback (e.g. "iliac_artery" -> "iliac_artery_left" | "iliac_artery_right")
        m_l = nothing; m_r = nothing
        targets_to_check = [name]
        if haskey(ALIASES, nl) && ALIASES[nl] != name
            push!(targets_to_check, ALIASES[nl])
        end
        for t in targets_to_check
            for suffix in ["_left", "_Left", "_l", "_L"]
                cand = "$(t)$(suffix)"
                if haskey(gen_masks, cand); m_l = gen_masks[cand]; break; end
                if haskey(packed_tensor.registry, cand); m_l = MaskPacker.unpack_mask(packed_tensor, cand); break; end
                if haskey(masks, cand); m_l = masks[cand]; break; end
            end
            if m_l !== nothing break end
        end
        for t in targets_to_check
            for suffix in ["_right", "_Right", "_r", "_R"]
                cand = "$(t)$(suffix)"
                if haskey(gen_masks, cand); m_r = gen_masks[cand]; break; end
                if haskey(packed_tensor.registry, cand); m_r = MaskPacker.unpack_mask(packed_tensor, cand); break; end
                if haskey(masks, cand); m_r = masks[cand]; break; end
            end
            if m_r !== nothing break end
        end
        if m_l !== nothing && m_r !== nothing
            m_l_gpu = (m_l isa CuArray) ? m_l : adapt(backend, map(x -> x > 0 ? UInt8(1) : UInt8(0), m_l))
            m_r_gpu = (m_r isa CuArray) ? m_r : adapt(backend, map(x -> x > 0 ? UInt8(1) : UInt8(0), m_r))
            return m_l_gpu .| m_r_gpu
        elseif m_l !== nothing
            return (m_l isa CuArray) ? m_l : adapt(backend, map(x -> x > 0 ? UInt8(1) : UInt8(0), m_l))
        elseif m_r !== nothing
            return (m_r isa CuArray) ? m_r : adapt(backend, map(x -> x > 0 ? UInt8(1) : UInt8(0), m_r))
        end

        # Unilateral split fallback (for paired bilateral muscles segmented as one, or bilateral intermediate stations)
        for (suf, is_l) in [("_left", true), ("_Left", true), ("_l", true), ("_L", true),
                            ("_right", false), ("_Right", false), ("_r", false), ("_R", false)]
            if endswith(name, suf)
                parent_k = name[1:end-length(suf)]
                parent_low = lowercase(parent_k)
                # STRICTLY FORBID midline/unpaired structures from unilateral splitting
                is_midline = occursin(r"(?i)(trachea|esophagus|spine|vertebra|aorta|sternum|manubrium|heart|hyoid|cricoid|thyroid|cord|vena_cava|carotid|azygos|brainstem|bladder|rectum|prostate|uterus|stomach|liver|pancreas|spleen|sacrum|coccyx)", parent_low)
                if !is_midline
                    # Check if parent is a known paired bilateral muscle or an intermediate generated mask
                    is_paired_muscle = occursin(r"(?i)(subscapularis|pectoralis|latissimus|serratus|rhomboid|gluteus|rectus_abdominis|trapezius|sternocleidomastoid|masseter|pterygoid|digastric|scalene|infraspinatus|supraspinatus|teres|iliopsoas|psoas)", parent_low)
                    is_gen = haskey(gen_masks, parent_k) || haskey(gen_masks, parent_low)
                    
                    if is_paired_muscle || is_gen
                        parent_m = get_mask(parent_k)
                        if parent_m !== nothing
                            mid_x = dims[1] ÷ 2
                            res = copy(parent_m)
                            if is_l
                                res[1:mid_x-1, :, :] .= UInt8(0)
                            else
                                res[mid_x:end, :, :] .= UInt8(0)
                            end
                            composite_cache[nl] = res
                            return res
                        end
                    end
                end
            end
        end

        return nothing
    end

    function get_mask_into(name::String, out_buf::AbstractArray{UInt8, 3})
        if haskey(packed_tensor.registry, name)
            return MaskPacker.unpack_mask!(out_buf, packed_tensor, name)
        end
        nl = lowercase(name)
        if haskey(ALIASES, nl) && haskey(packed_tensor.registry, ALIASES[nl])
            return MaskPacker.unpack_mask!(out_buf, packed_tensor, ALIASES[nl])
        end
        for (k, v) in packed_tensor.registry
            if lowercase(k) == nl
                return MaskPacker.unpack_mask!(out_buf, packed_tensor, k)
            end
        end
        m = get_mask(name)
        if m !== nothing
            copyto!(out_buf, m)
            return true
        end
        return false
    end

    gpu_masks = GpuMasksDict(masks, get_mask)

    # Timing accumulators for benchmark analysis
    timing_rule_gen = Float64[]     # Rule generation phase per level
    timing_postproc = Float64[]     # Post-processing (constraints, z-restr, excl, LCC)
    timing_copy_arena = Float64[]   # Copy level outputs to VM_ARENA
    timing_vm_compile = Float64[]   # VM instruction compilation
    timing_megakernel = Float64[]   # MegaKernel execution
    timing_repack = Float64[]       # Repack into PackedTensor
    timing_gen_masks = Float64[]    # Copy to gen_masks BitArrays (GPU->CPU)
    timing_level_total = Float64[]  # Total per level
    timing_gc_reclaim = Float64[]
    last_reclaim = time()
    last_reclaim = time()   # GC/CUDA.reclaim time per level

    for (lvl_idx, level_nodes) in enumerate(levels)
        t_level_start = time()
        println("\n--------------------------------------------------------------------------------")
        println("  Executing DAG Level $lvl_idx / $(length(levels)) ($(length(level_nodes)) rules)...")
        println("--------------------------------------------------------------------------------")
        flush(stdout)
        
        t_rule_gen_start = time()
        level_outputs = LevelOutputDict(StaticArena.VM_ARENA[:level_output], backend); fill!(StaticArena.VM_ARENA[:level_output], UInt8(0))
        for node_name in level_nodes
            rule_def = rules[node_name]
            rule_type = get(rule_def, "rule", "")
            
            if VMCompiler.can_compile_directly(rule_def, packed_tensor, ALIASES, node_name)
                level_outputs[node_name] = nothing
                empty!(composite_cache)
                continue
            end
            
            params_dict = copy(rule_def)
            params_dict["spacing_x"] = spacing[1]
            params_dict["spacing_y"] = spacing[2]
            params_dict["spacing_z"] = spacing[3]
            params_dict["origin_x"] = origin[1]
            params_dict["origin_y"] = origin[2]
            params_dict["origin_z"] = origin[3]
            params_dict["direction_x"] = direction[1]
            params_dict["direction_y"] = direction[5]
            params_dict["direction_z"] = direction[9]

            # Check if this rule is a specialized rule (Axillary, Celiac, Morphological Expansion)
            if rule_type in ["AxillaryRTOG", "AxillaryRTOGRelaxed"]
                side = get(rule_def, "side", occursin("right", lowercase(node_name)) ? "right" : "left")
                ax_results = RuleExecutors.execute_axillary_rtog(backend, get_mask, get_mask_into, packed_tensor, side, dims, spacing, origin, direction)
                for (k, v) in ax_results
                    level_outputs[k] = v
                end
                empty!(composite_cache)
                continue
            elseif rule_type in ["ThoracicStation3ACustom", "Station3APrevascular"]
                side = get(rule_def, "side", occursin("right", lowercase(node_name)) ? "right" : "left")
                out_mask = RuleExecutors.execute_station_3a_prevascular(backend, get_mask, side, dims, spacing, origin, direction)
                if out_mask !== nothing
                    mask_arr = out_mask
                    level_outputs[node_name] = mask_arr
                else
                    empty!(composite_cache)
                    continue
                end

            
            elseif rule_type == "SubscapularisBand"
                side = lowercase(get(rule_def, "side", "left"))
                subscap_m = resolve_mask_from_spec("subscapularis_$side", get_mask, dims, backend)
                if subscap_m === nothing
                    subscap_m = resolve_mask_from_spec("subscapularis", get_mask, dims, backend)
                end
                if subscap_m !== nothing
                    shift_mm = Float64(get(rule_def, "shift_mm", 60.0))
                    mask_arr = RuleExecutors.execute_subscapularis_band(backend, subscap_m, side == "left", shift_mm, spacing, dims)
                    level_outputs[node_name] = mask_arr
                else
                    empty!(composite_cache)
                    continue
                end
            elseif rule_type == "AortaMedialPlane"
                aorta_m = resolve_mask_from_spec("aorta", get_mask, dims, backend)
                if aorta_m !== nothing
                    mask_arr = RuleExecutors.execute_aorta_medial_plane(backend, aorta_m, dims)
                    level_outputs[node_name] = mask_arr
                else
                    empty!(composite_cache)
                    continue
                end
            elseif rule_type == "RelativeGeometricRegion"
                # Celiac geometric region
                sup_ref_spec = get(rule_def, "superior_ref", "pancreas")
                ant_ref_spec = get(rule_def, "anterior_ref", "aorta")
                sup_mask = resolve_mask_from_spec(sup_ref_spec, get_mask, dims, backend)
                ant_mask = resolve_mask_from_spec(ant_ref_spec, get_mask, dims, backend)
                
                if sup_mask !== nothing && ant_mask !== nothing
                    sup_z_bounds = RuleExecutors.get_z_bounds_physical(sup_mask, dims, spacing, origin, direction)
                    sup_z_max = sup_z_bounds[2]
                    ext_mm = Float32(get(rule_def, "extension_mm", 60.0))
                    z_limit = sup_z_max + ext_mm
                    
                    ant_y_bounds = RuleExecutors.get_y_bounds_physical(ant_mask, dims, spacing, origin, direction)
                    ant_x_bounds = RuleExecutors.get_x_bounds_physical(ant_mask, dims, spacing, origin, direction)
                    
                    ant_y_min = ant_y_bounds[1]
                    ext_y_mm = Float32(get(rule_def, "extension_y_mm", 40.0))
                    y_limit = ant_y_min - ext_y_mm
                    margin_y_post = Float32(get(rule_def, "margin_y_posterior_mm", 10.0))
                    y_post = ant_y_min + margin_y_post
                    
                    margin_lat = Float32(get(rule_def, "margin_lateral_mm", 40.0))
                    x_min = ant_x_bounds[1] - margin_lat
                    x_max = ant_x_bounds[2] + margin_lat
                    
                    inst = VmInstruction(
                        OP_RELATIVE_GEOMETRIC,
                        Int32(0), Int32(0), Int32(0), Int32(0),
                        out_idx,
                        sup_z_max, z_limit,
                        y_limit, y_post,
                        x_min, x_max
                    )
                    push!(instructions, inst)
                    inst_idx_to_name[out_idx] = node_name
                    out_arr = KernelAbstractions.zeros(backend, UInt8, dims)
                    level_outputs[node_name] = out_arr
                    out_idx += Int32(1)
                end
                empty!(composite_cache)
                continue
            elseif rule_type in ["DistanceExpansion", "AnisotropicMargin", "Morphology", "GenericMorphologyDAG", "DilatedMask", "MarginAroundLandmark"]
                base_spec = get(rule_def, "base_landmark", get(rule_def, "input", get(rule_def, "input_mask", get(rule_def, "mask_name", get(rule_def, "landmark", get(rule_def, "landmarks", get(rule_def, "input_landmarks", "")))))))
                
                exp_m = nothing
                if base_spec isa Dict && !haskey(base_spec, "landmark") && !haskey(base_spec, "name") && !haskey(base_spec, "base_landmark")
                    # Multi-organ specification mapping organ_name => margin_dict
                    is_bilat_rule = get(rule_def, "is_bilateral", false)
                    if is_bilat_rule
                        for s in ["Left", "Right"]
                            side_lower = lowercase(s)
                            side_exp = nothing
                            for (organ_name, organ_margins) in base_spec
                                sub_m = resolve_mask_from_spec("$(organ_name)_$side_lower", get_mask, dims, backend)
                                if sub_m === nothing
                                    sub_m = resolve_mask_from_spec("$(organ_name)_$(s[1:1])", get_mask, dims, backend)
                                end
                                if sub_m === nothing
                                    sub_m = resolve_mask_from_spec(organ_name, get_mask, dims, backend)
                                end
                                if sub_m !== nothing
                                    m_dict = organ_margins isa Dict ? organ_margins : Dict("all" => Float64(organ_margins))
                                    sub_exp = RuleExecutors.execute_anisotropic_expansion(backend, sub_m, dims, spacing, m_dict; side=side_lower)
                                    side_exp = side_exp === nothing ? sub_exp : (side_exp .| sub_exp)
                                end
                            end
                            if side_exp !== nothing
                                level_outputs["$(node_name)_$s"] = side_exp
                            end
                        end
                        l_m = get(level_outputs, "$(node_name)_Left", nothing)
                        r_m = get(level_outputs, "$(node_name)_Right", nothing)
                        if l_m !== nothing && r_m !== nothing
                            level_outputs[node_name] = l_m .| r_m
                        elseif l_m !== nothing
                            level_outputs[node_name] = l_m
                        elseif r_m !== nothing
                            level_outputs[node_name] = r_m
                        end
                    else
                        for (organ_name, organ_margins) in base_spec
                            sub_m = resolve_mask_from_spec(organ_name, get_mask, dims, backend)
                            if sub_m !== nothing
                                m_dict = organ_margins isa Dict ? organ_margins : Dict("all" => Float64(organ_margins))
                                sub_exp = RuleExecutors.execute_anisotropic_expansion(backend, sub_m, dims, spacing, m_dict)
                                exp_m = exp_m === nothing ? sub_exp : (exp_m .| sub_exp)
                            println("    [DEBUG] $node_name base_spec=$base_spec sub_exp=$(sum(sub_exp .> 0)) exp_m=$(sum(exp_m .> 0))")
                            end
                        end
                        if exp_m !== nothing
                            level_outputs[node_name] = exp_m
                        end
                    end
                else
                    base_m = resolve_mask_from_spec(base_spec, get_mask, dims, backend)
                    if base_m !== nothing
                        margins = Dict{String, Any}()
                        if rule_type in ["DistanceExpansion", "DilatedMask"]
                            d = Float64(get(rule_def, "distance_mm", get(rule_def, "dilation_mm", 20.0)))
                            margins["all"] = d
                        elseif rule_type in ["Morphology", "GenericMorphologyDAG"]
                            r = Float64(get(rule_def, "radius_mm", 20.0))
                            op = lowercase(get(rule_def, "operation", "dilate"))
                            margins["all"] = (op == "erode" || op == "erosion") ? -r : r
                        elseif rule_type == "MarginAroundLandmark"
                            m = Float64(get(rule_def, "margin_mm", get(rule_def, "margin", 10.0)))
                            margins["all"] = m
                        else
                            if haskey(rule_def, "margins_mm")
                                m_raw = rule_def["margins_mm"]
                                if m_raw isa Dict
                                    margins = m_raw
                                elseif m_raw isa Number
                                    margins = Dict("all" => Float64(m_raw))
                                else
                                    margins = Dict{String, Float64}()
                                end
                            elseif haskey(rule_def, "margin_mm")
                                margins = Dict("all" => Float64(rule_def["margin_mm"]))
                            else
                                margins = Dict{String, Float64}()
                            end
                        end
                        is_bilat_rule = get(rule_def, "is_bilateral", false)
                        if is_bilat_rule
                            m_l = if base_spec isa Vector
                                sub_res = nothing
                                for item in base_spec
                                    sub_m = resolve_mask_from_spec("$(item)_left", get_mask, dims, backend)
                                    if sub_m === nothing; sub_m = resolve_mask_from_spec("$(item)_l", get_mask, dims, backend); end
                                    if sub_m === nothing; sub_m = resolve_mask_from_spec(item, get_mask, dims, backend); end
                                    if sub_m !== nothing
                                        sub_res = sub_res === nothing ? copy(sub_m) : (sub_res .|= sub_m)
                                    end
                                end
                                sub_res
                            else
                                sub_l = resolve_mask_from_spec("$(base_spec)_left", get_mask, dims, backend)
                                if sub_l === nothing; sub_l = resolve_mask_from_spec("$(base_spec)_l", get_mask, dims, backend); end
                                sub_l
                            end
                            
                            m_r = if base_spec isa Vector
                                sub_res = nothing
                                for item in base_spec
                                    sub_m = resolve_mask_from_spec("$(item)_right", get_mask, dims, backend)
                                    if sub_m === nothing; sub_m = resolve_mask_from_spec("$(item)_r", get_mask, dims, backend); end
                                    if sub_m === nothing; sub_m = resolve_mask_from_spec(item, get_mask, dims, backend); end
                                    if sub_m !== nothing
                                        sub_res = sub_res === nothing ? copy(sub_m) : (sub_res .|= sub_m)
                                    end
                                end
                                sub_res
                            else
                                sub_r = resolve_mask_from_spec("$(base_spec)_right", get_mask, dims, backend)
                                if sub_r === nothing; sub_r = resolve_mask_from_spec("$(base_spec)_r", get_mask, dims, backend); end
                                sub_r
                            end
                            if m_l === nothing && m_r === nothing
                                mid_x = dims[1] ÷ 2
                                m_l = copy(base_m); m_l[1:mid_x-1, :, :] .= UInt8(0)
                                m_r = copy(base_m); m_r[mid_x:end, :, :] .= UInt8(0)
                            end
                            exp_l = RuleExecutors.execute_anisotropic_expansion(backend, m_l, dims, spacing, margins; side="left")
                            exp_r = RuleExecutors.execute_anisotropic_expansion(backend, m_r, dims, spacing, margins; side="right")
                            
                            if rule_type == "DistanceExpansion" || (rule_type == "MarginAroundLandmark" && !get(rule_def, "include_landmark", false))
                                inv_l = adapt(backend, map(x -> x == 0 ? UInt8(1) : UInt8(0), m_l))
                                inv_r = adapt(backend, map(x -> x == 0 ? UInt8(1) : UInt8(0), m_r))
                                exp_l = exp_l .& inv_l
                                exp_r = exp_r .& inv_r
                            end
                            
                            level_outputs["$(node_name)_Left"] = exp_l
                            level_outputs["$(node_name)_Right"] = exp_r
                            level_outputs[node_name] = exp_l .| exp_r
                        else
                            # For non-bilateral single-side rules, extract side from JSON to correctly
                            # resolve lateral/medial margin directions (e.g. Supraclavicular_Left, Retrostyloid)
                            rule_side_str = get(rule_def, "side", nothing)
                            if rule_side_str === nothing
                                # Infer side from node name as fallback
                                nn_lower = lowercase(node_name)
                                if occursin("_left", nn_lower) || endswith(nn_lower, "_l")
                                    rule_side_str = "left"
                                elseif occursin("_right", nn_lower) || endswith(nn_lower, "_r")
                                    rule_side_str = "right"
                                end
                            end
                            exp_m = RuleExecutors.execute_anisotropic_expansion(backend, base_m, dims, spacing, margins; side=rule_side_str)
                            base_sum = gpu_count(base_m)
                            exp_sum = gpu_count(exp_m)
                            println("    [DEBUG-DILATED] $node_name base_spec=$base_spec base_voxels=$base_sum dilated_voxels=$exp_sum")
                            
                            # DistanceExpansion mathematically excludes the base mask in Python
                            if rule_type == "DistanceExpansion" || (rule_type == "MarginAroundLandmark" && !get(rule_def, "include_landmark", false))
                                inv_base = adapt(backend, map(x -> x == 0 ? UInt8(1) : UInt8(0), base_m))
                                exp_m = exp_m .& inv_base
                            end
                            
                            if !get(rule_def, "keep_all_components", false)
                                exp_m = RuleExecutors.get_largest_connected_component(exp_m)
                            end
                            level_outputs[node_name] = exp_m

                        end
                    end
                end
                # We DO NOT continue here, we want to let it fall through to apply_constraints and exclusions!
                # Wait, if it doesn't continue, it will just fall through...
                # BUT wait, the generic Mask block is at the END of the if/elseif chain!
            elseif rule_type == "PrimaryVector"
                prim_spec = get(rule_def, "primary_landmark", get(rule_def, "base_landmark", ""))
                vec_spec = get(rule_def, "vector_landmarks", Dict())
                base_margin = Float64(get(rule_def, "base_margin_mm", 5.0))
                is_bilat = get(rule_def, "is_bilateral", false)
                
                if is_bilat
                    for s in ["Left", "Right"]
                        s_lower = lowercase(s)
                        prim_side_spec = if prim_spec isa Vector
                            ["$(p)_$s_lower" for p in prim_spec]
                        elseif prim_spec isa String
                            "$(prim_spec)_$s_lower"
                        else
                            prim_spec
                        end
                        prim_m = resolve_mask_from_spec(prim_side_spec, get_mask, dims, backend)
                        if prim_m === nothing; prim_m = resolve_mask_from_spec(prim_spec, get_mask, dims, backend); end
                        
                        if prim_m !== nothing
                            res_m = nothing
                            if vec_spec isa AbstractString
                                exp_mm = Float64(get(rule_def, "expansion_mm", 10.0))
                                vec_s_name = "$(vec_spec)_$s_lower"
                                vec_m = resolve_mask_from_spec(vec_s_name, get_mask, dims, backend)
                                if vec_m === nothing; vec_m = resolve_mask_from_spec(vec_spec, get_mask, dims, backend); end
                                if vec_m !== nothing
                                    res_m = RuleExecutors.execute_primary_vector(backend, prim_m, vec_m, dims, spacing, origin, direction, base_margin, exp_mm)
                                end
                            elseif vec_spec isa Vector
                                exp_mm = Float64(get(rule_def, "expansion_mm", 10.0))
                                for v_name in vec_spec
                                    vec_s_name = "$(v_name)_$s_lower"
                                    vec_m = resolve_mask_from_spec(vec_s_name, get_mask, dims, backend)
                                    if vec_m === nothing; vec_m = resolve_mask_from_spec(string(v_name), get_mask, dims, backend); end
                                    if vec_m !== nothing
                                        sub_res = RuleExecutors.execute_primary_vector(backend, prim_m, vec_m, dims, spacing, origin, direction, base_margin, exp_mm)
                                        res_m = res_m === nothing ? sub_res : (res_m .| sub_res)
                                    end
                                end
                            elseif vec_spec isa Dict
                                for (v_name, v_params) in vec_spec
                                    exp_mm = Float64(v_params isa Dict ? get(v_params, "expansion_mm", 10.0) : v_params)
                                    vec_s_name = "$(v_name)_$s_lower"
                                    vec_m = resolve_mask_from_spec(vec_s_name, get_mask, dims, backend)
                                    if vec_m === nothing; vec_m = resolve_mask_from_spec(v_name, get_mask, dims, backend); end
                                    if vec_m !== nothing
                                        sub_res = RuleExecutors.execute_primary_vector(backend, prim_m, vec_m, dims, spacing, origin, direction, base_margin, exp_mm)
                                        res_m = res_m === nothing ? sub_res : (res_m .| sub_res)
                                    end
                                end
                            end
                            if res_m === nothing
                                res_m = RuleExecutors.execute_anisotropic_expansion(backend, prim_m, dims, spacing, Dict("all" => base_margin); side=s_lower)
                            end
                            level_outputs["$(node_name)_$s"] = res_m
                            level_outputs["$(node_name)_$s_lower"] = res_m
                        end
                    end
                    l_m = get(level_outputs, "$(node_name)_Left", nothing)
                    r_m = get(level_outputs, "$(node_name)_Right", nothing)
                    if l_m !== nothing && r_m !== nothing
                        level_outputs[node_name] = l_m .| r_m
                    elseif l_m !== nothing
                        level_outputs[node_name] = l_m
                    elseif r_m !== nothing
                        level_outputs[node_name] = r_m
                    end
                else
                    prim_m = resolve_mask_from_spec(prim_spec, get_mask, dims, backend)
                    if prim_m !== nothing
                        res_m = nothing
                        if vec_spec isa AbstractString
                            exp_mm = Float64(get(rule_def, "expansion_mm", 10.0))
                            vec_m = resolve_mask_from_spec(vec_spec, get_mask, dims, backend)
                            if vec_m !== nothing
                                res_m = RuleExecutors.execute_primary_vector(backend, prim_m, vec_m, dims, spacing, origin, direction, base_margin, exp_mm)
                            end
                        elseif vec_spec isa Vector
                            exp_mm = Float64(get(rule_def, "expansion_mm", 10.0))
                            for v_name in vec_spec
                                vec_m = resolve_mask_from_spec(string(v_name), get_mask, dims, backend)
                                if vec_m !== nothing
                                    sub_res = RuleExecutors.execute_primary_vector(backend, prim_m, vec_m, dims, spacing, origin, direction, base_margin, exp_mm)
                                    res_m = res_m === nothing ? sub_res : (res_m .| sub_res)
                                end
                            end
                        elseif vec_spec isa Dict
                            for (v_name, v_params) in vec_spec
                                exp_mm = Float64(v_params isa Dict ? get(v_params, "expansion_mm", 10.0) : v_params)
                                vec_m = resolve_mask_from_spec(v_name, get_mask, dims, backend)
                                if vec_m !== nothing
                                    sub_res = RuleExecutors.execute_primary_vector(backend, prim_m, vec_m, dims, spacing, origin, direction, base_margin, exp_mm)
                                    res_m = res_m === nothing ? sub_res : (res_m .| sub_res)
                                end
                            end
                        end
                        if res_m === nothing
                            res_m = RuleExecutors.execute_anisotropic_expansion(backend, prim_m, dims, spacing, Dict("all" => base_margin))
                        end
                        level_outputs[node_name] = res_m
                    end
                end
                empty!(composite_cache)
                continue
            elseif rule_type == "GeometricPrimitive"
                p1_def = get(rule_def, "p1", "")
                p2_def = get(rule_def, "p2", "")
                rad = Float32(get(rule_def, "radius_mm", 20.0))
                
                function _resolve_p(p_def, s_suffix)
                    if p_def isa Vector && length(p_def) >= 3
                        return (Float32(p_def[1]), Float32(p_def[2]), Float32(p_def[3]))
                    elseif p_def isa String
                        if !isempty(s_suffix) && haskey(computed_landmarks, "$(p_def)_$s_suffix")
                            v = computed_landmarks["$(p_def)_$s_suffix"]
                            if v isa Vector && length(v) >= 3 return (Float32(v[1]), Float32(v[2]), Float32(v[3])) end
                        end
                        if haskey(computed_landmarks, p_def)
                            v = computed_landmarks[p_def]
                            if v isa Vector && length(v) >= 3 return (Float32(v[1]), Float32(v[2]), Float32(v[3])) end
                        end
                        if startswith(p_def, "internal_iliac_p1") || startswith(p_def, "internal_iliac_p2")
                            side_s = !isempty(s_suffix) ? s_suffix : (occursin("right", p_def) ? "right" : "left")
                            if startswith(p_def, "internal_iliac_p1")
                                art_k = "iliac_artery_$side_s"
                                m_art = get_mask(art_k)
                                if m_art === nothing; m_art = get_mask("iliac_artery_internal_$side_s"); end
                                if m_art === nothing; m_art = get_mask("iliac_artery"); end
                                m_l5 = get_mask("vertebrae_L5")
                                if m_art !== nothing
                                    p1_dyn = Landmarks.compute_internal_iliac_p1(m_art isa CuArray ? m_art : adapt(backend, m_art), m_l5 !== nothing ? (m_l5 isa CuArray ? m_l5 : adapt(backend, m_l5)) : nothing, spacing, origin, direction)
                                    return (Float32(p1_dyn[1]), Float32(p1_dyn[2]), Float32(p1_dyn[3]))
                                end
                            else
                                m_sacrum = get_mask("sacrum")
                                m_hip = get_mask("hip_$side_s")
                                if m_hip === nothing; m_hip = get_mask("hip"); end
                                if m_sacrum !== nothing && m_hip !== nothing
                                    p2_dyn = Landmarks.compute_internal_iliac_p2(m_sacrum isa CuArray ? m_sacrum : adapt(backend, m_sacrum), m_hip isa CuArray ? m_hip : adapt(backend, m_hip), spacing, origin, direction)
                                    return (Float32(p2_dyn[1]), Float32(p2_dyn[2]), Float32(p2_dyn[3]))
                                end
                            end
                        end
                        m = get_mask(p_def)
                        if m !== nothing return RuleExecutors.find_most_anterior_point(m, dims, spacing, origin, direction) end
                    elseif p_def isa Dict
                        lm_k = get(p_def, "landmark", get(p_def, "name", ""))
                        lm_lookup = !isempty(s_suffix) ? "$(lm_k)_$s_suffix" : lm_k
                        m = get_mask(lm_lookup)
                        if m === nothing; m = get_mask(lm_k); end
                        if m !== nothing
                            idx = adapt(Array, findall(m .> UInt8(0)))
                            if !isempty(idx)
                                dir_00 = Float32(direction[1]); dir_11 = Float32(direction[5]); dir_22 = Float32(direction[9])
                                split_mode = get(p_def, "split", "")
                                if split_mode in ["medial", "lateral"]
                                    xs_idx = [ci[1] for ci in idx]
                                    param = Statistics.median(xs_idx)
                                    is_left_side = lowercase(s_suffix) == "left"
                                    if is_left_side
                                        idx = split_mode == "medial" ? filter(ci -> ci[1] < param, idx) : filter(ci -> ci[1] >= param, idx)
                                    else
                                        idx = split_mode == "medial" ? filter(ci -> ci[1] >= param, idx) : filter(ci -> ci[1] < param, idx)
                                    end
                                end
                                if isempty(idx)
                                    idx = adapt(Array, findall(m .> UInt8(0)))
                                end
                                dir_spec = lowercase(get(p_def, "direction", "anterior"))
                                dir_vec = if dir_spec == "anterior"; (0.0f0, -1.0f0, 0.0f0)
                                elseif dir_spec == "posterior"; (0.0f0, 1.0f0, 0.0f0)
                                elseif dir_spec == "superior"; (0.0f0, 0.0f0, 1.0f0)
                                elseif dir_spec == "inferior"; (0.0f0, 0.0f0, -1.0f0)
                                elseif dir_spec == "left"; (1.0f0, 0.0f0, 0.0f0)
                                elseif dir_spec == "right"; (-1.0f0, 0.0f0, 0.0f0)
                                else; (0.0f0, -1.0f0, 0.0f0)
                                end
                                scores = [Float32(ci[1]-1)*dir_vec[1]*dir_00 + Float32(ci[2]-1)*dir_vec[2]*dir_11 + Float32(ci[3]-1)*dir_vec[3]*dir_22 for ci in idx]
                                thresh = Statistics.quantile(scores, 0.99)
                                top_idx = filter(i -> scores[i] >= thresh, 1:length(scores))
                                if isempty(top_idx); top_idx = [argmax(scores)]; end
                                ci_mean_x = mean([idx[i][1]-1 for i in top_idx])
                                ci_mean_y = mean([idx[i][2]-1 for i in top_idx])
                                ci_mean_z = mean([idx[i][3]-1 for i in top_idx])
                                return (Float32(origin[1]) + Float32(ci_mean_x)*Float32(spacing[1])*dir_00,
                                        Float32(origin[2]) + Float32(ci_mean_y)*Float32(spacing[2])*dir_11,
                                        Float32(origin[3]) + Float32(ci_mean_z)*Float32(spacing[3])*dir_22)
                            end
                        end
                    end
                    return nothing
                end
                
                is_bilat = get(rule_def, "is_bilateral", false)
                if is_bilat
                    pt1_l = _resolve_p(p1_def, "left"); pt2_l = _resolve_p(p2_def, "left")
                    pt1_r = _resolve_p(p1_def, "right"); pt2_r = _resolve_p(p2_def, "right")
                    
                    len_mod = Float32(get(rule_def, "length_modifier", 1.0))
                    cyl_total = KernelAbstractions.zeros(backend, UInt8, dims)
                    if pt1_l !== nothing && pt2_l !== nothing
                        p2l_mod = (pt1_l[1] + (pt2_l[1]-pt1_l[1])*len_mod, pt1_l[2] + (pt2_l[2]-pt1_l[2])*len_mod, pt1_l[3] + (pt2_l[3]-pt1_l[3])*len_mod)
                        cyl_l = RuleExecutors.execute_cylinder_primitive(backend, pt1_l, p2l_mod, rad, dims, spacing, origin, direction)
                        cyl_total .|= cyl_l
                        level_outputs["$(node_name)_Left"] = cyl_l
                        level_outputs["$(node_name)_left"] = cyl_l
                    end
                    if pt1_r !== nothing && pt2_r !== nothing
                        p2r_mod = (pt1_r[1] + (pt2_r[1]-pt1_r[1])*len_mod, pt1_r[2] + (pt2_r[2]-pt1_r[2])*len_mod, pt1_r[3] + (pt2_r[3]-pt1_r[3])*len_mod)
                        cyl_r = RuleExecutors.execute_cylinder_primitive(backend, pt1_r, p2r_mod, rad, dims, spacing, origin, direction)
                        cyl_total .|= cyl_r
                        level_outputs["$(node_name)_Right"] = cyl_r
                        level_outputs["$(node_name)_right"] = cyl_r
                    end
                    level_outputs[node_name] = cyl_total
                    empty!(composite_cache)
                    continue
                else
                    pt1 = _resolve_p(p1_def, ""); pt2 = _resolve_p(p2_def, "")
                    if pt1 !== nothing && pt2 !== nothing
                        len_mod = Float32(get(rule_def, "length_modifier", 1.0))
                        p2_mod = (pt1[1] + (pt2[1]-pt1[1])*len_mod, pt1[2] + (pt2[2]-pt1[2])*len_mod, pt1[3] + (pt2[3]-pt1[3])*len_mod)
                        level_outputs[node_name] = RuleExecutors.execute_cylinder_primitive(backend, pt1, p2_mod, rad, dims, spacing, origin, direction)
                    end
                    empty!(composite_cache)
                    continue
                end

            elseif rule_type == "Station1LowCervical"
                is_bilat = get(rule_def, "is_bilateral", false)
                deps = get(rule_def, "dependencies", Dict())
                in_str = get(rule_def, "input", "body_trunc")
                in_m = resolve_mask_from_spec(in_str, get_mask, dims, backend)
                
                manubrium = get_mask(get(deps, "manubrium", "manubrium")); if manubrium === nothing; manubrium = get_mask("sternum"); end
                clavicle = get_mask(get(deps, "clavicle", "clavicula"))
                
                if in_m === nothing || (manubrium === nothing && clavicle === nothing)
                    println("    [INFO] Missing required anatomy for $node_name; skipping area calculation.")
                    out_arr = KernelAbstractions.zeros(backend, UInt8, dims)
                    if is_bilat
                        for s in ["Left", "Right"]
                            level_outputs["$(node_name)_$s"] = copy(out_arr)
                            level_outputs["$(node_name)_$(lowercase(s))"] = copy(out_arr)
                        end
                        level_outputs[node_name] = out_arr
                    else
                        level_outputs[node_name] = out_arr
                    end
                    empty!(composite_cache)
                    continue
                end
                
                if manubrium === nothing; manubrium = KernelAbstractions.zeros(backend, UInt8, dims); end
                if clavicle === nothing; clavicle = KernelAbstractions.zeros(backend, UInt8, dims); end
                cricoid = get_mask(get(deps, "cricoid", "cricoid_cartilage")); if cricoid === nothing; cricoid = KernelAbstractions.zeros(backend, UInt8, dims); end
                trachea = get_mask(get(deps, "trachea", "trachea")); if trachea === nothing; trachea = KernelAbstractions.zeros(backend, UInt8, dims); end
                esophagus = get_mask(get(deps, "esophagus", "esophagus")); if esophagus === nothing; esophagus = KernelAbstractions.zeros(backend, UInt8, dims); end
                thyroid = get_mask(get(deps, "thyroid", "thyroid_gland")); if thyroid === nothing; thyroid = KernelAbstractions.zeros(backend, UInt8, dims); end
                lung_l = get_mask(get(deps, "lung_l", "lung_left")); if lung_l === nothing; lung_l = KernelAbstractions.zeros(backend, UInt8, dims); end
                lung_r = get_mask(get(deps, "lung_r", "lung_right")); if lung_r === nothing; lung_r = KernelAbstractions.zeros(backend, UInt8, dims); end
                scm = get_mask(get(deps, "scm", "sternocleidomastoid")); if scm === nothing; scm = KernelAbstractions.zeros(backend, UInt8, dims); end
                scalene = get_mask(get(deps, "scalene", "anterior_scalene")); if scalene === nothing; scalene = KernelAbstractions.zeros(backend, UInt8, dims); end
                spine = get_mask("vertebrae_T1"); if spine === nothing; spine = get_mask("fused_spine"); end; if spine === nothing; spine = get_mask("spine"); end; if spine === nothing; spine = KernelAbstractions.zeros(backend, UInt8, dims); end
                
                function _run_station1(s_str)
                    out_arr = KernelAbstractions.zeros(backend, UInt8, dims)
                    plane_key = "plane_station_2_sup_" * lowercase(s_str)
                    plane_data = get(computed_landmarks, plane_key, nothing)
                    CustomRules.Station1LowCervical((backend=backend,), out_arr, params_dict, in_m, cricoid, trachea, manubrium, clavicle, esophagus, thyroid, lung_l, lung_r, scm, scalene, spine, plane_data, s_str, spacing, origin, direction)
                    return out_arr
                end

                if is_bilat
                    out_total = KernelAbstractions.zeros(backend, UInt8, dims)
                    for s in ["Left", "Right"]
                        res_s = _run_station1(s)
                        level_outputs["$(node_name)_$s"] = res_s
                        level_outputs["$(node_name)_$(lowercase(s))"] = res_s
                        out_total .|= res_s
                    end
                    level_outputs[node_name] = out_total
                else
                    side_str = occursin("right", lowercase(node_name)) ? "right" : "left"
                    res = _run_station1(side_str)
                    level_outputs[node_name] = res
                end
                empty!(composite_cache)
                continue

            elseif rule_type in ["HilarAnteriorHelper", "AnteriorGrowthMask", "ExtractMainBronchi", "PosteriorGrowthMask", "LimitZByLandmark", "AnteriorExtrusion", "PleuralSpaceCustom", "PresacralAnteriorCustom", "SplitConnectedComponents", "HelperSplenicPDCustom", "StomachLongAxisHelper", "PancreasSplitHelper", "PyloricSplitHelper", "ErosionHelper"]
                out_arr = KernelAbstractions.zeros(backend, UInt8, dims)
                params_dict = copy(rule_def)
                params_dict["spacing_x"] = Float64(spacing[1])
                params_dict["spacing_y"] = Float64(spacing[2])
                params_dict["spacing_z"] = Float64(spacing[3])
                params_dict["spacing"] = spacing
                params_dict["origin"] = origin
                params_dict["direction"] = direction
                params_dict["dims"] = dims
                
                in_str = get(rule_def, "lung_landmark", get(rule_def, "input", get(rule_def, "input_mask", get(rule_def, "input_image", get(rule_def, "mask_name", get(rule_def, "landmark", get(rule_def, "hilar_landmark", "")))))))
                in_m = resolve_mask_from_spec(in_str, get_mask, dims, backend)
                if in_m === nothing; in_m = KernelAbstractions.zeros(backend, UInt8, dims); end
                
                if rule_type == "HilarAnteriorHelper"
                    CustomRules.HilarAnteriorHelper((backend=backend,), out_arr, params_dict, in_m)
                elseif rule_type == "ExtractMainBronchi"
                    CustomRules.ExtractMainBronchi((backend=backend,), out_arr, params_dict, in_m)
                elseif rule_type == "AnteriorGrowthMask"
                    CustomRules.AnteriorGrowthMask((backend=backend,), out_arr, params_dict, in_m)
                elseif rule_type == "PosteriorGrowthMask"
                    CustomRules.PosteriorGrowthMask((backend=backend,), out_arr, params_dict, in_m)
                elseif rule_type == "LimitZByLandmark"
                    lm_spec = get(rule_def, "landmark", "")
                    s_str = lowercase(get(rule_def, "side", ""))
                    mode_str = get(rule_def, "mode", "bounds")
                    
                    target_z_vox = nothing
                    keys_to_try = [lm_spec]
                    if !isempty(s_str)
                        pushfirst!(keys_to_try, "$(lm_spec)_$(s_str)")
                        pushfirst!(keys_to_try, "$(lm_spec)_top_$(s_str)_z")
                        pushfirst!(keys_to_try, "$(lm_spec)_top_$(s_str)")
                        pushfirst!(keys_to_try, "$(lm_spec)_$(s_str)_z")
                    end
                    if occursin("coracoid", lowercase(lm_spec))
                        if occursin("left", lowercase(node_name)) || s_str == "left"
                            pushfirst!(keys_to_try, "coracoid_process_left_z")
                            pushfirst!(keys_to_try, "coracoid_process_left")
                        elseif occursin("right", lowercase(node_name)) || s_str == "right"
                            pushfirst!(keys_to_try, "coracoid_process_right_z")
                            pushfirst!(keys_to_try, "coracoid_process_right")
                        end
                    elseif occursin("pectoralis", lowercase(lm_spec))
                        if occursin("left", lowercase(node_name)) || s_str == "left"
                            pushfirst!(keys_to_try, "pectoralis_minor_top_left_z")
                            pushfirst!(keys_to_try, "pectoralis_minor_top_left")
                        elseif occursin("right", lowercase(node_name)) || s_str == "right"
                            pushfirst!(keys_to_try, "pectoralis_minor_top_right_z")
                            pushfirst!(keys_to_try, "pectoralis_minor_top_right")
                        end
                    end
                    
                    for k in keys_to_try
                        if haskey(computed_landmarks, k)
                            val = computed_landmarks[k]
                            if val isa Integer
                                target_z_vox = Int(val)
                                break
                            elseif val isa Number
                                dir_22 = direction[9]
                                target_z_vox = Int(round((Float64(val) - origin[3]) / (spacing[3] * dir_22))) + 1
                                break
                            elseif (val isa Tuple || val isa Vector) && length(val) >= 1 && (val[1] isa Vector || val[1] isa Tuple)
                                pt = val[1]
                                dir_22 = direction[9]
                                target_z_vox = Int(round((Float64(pt[3]) - origin[3]) / (spacing[3] * dir_22))) + 1
                                break
                            end
                        end
                    end
                    
                    if target_z_vox !== nothing
                        z_max = clamp(target_z_vox, 1, dims[3])
                        z_min = (mode_str == "bottom_only") ? z_max : 1
                        if mode_str == "bottom_only"
                            z_max = dims[3]
                        end
                        kernel! = CustomRules.limit_z_by_landmark_kernel!(backend)
                        kernel!(out_arr, in_m, dims, Int(z_min), Int(z_max), ndrange=dims)
                        KernelAbstractions.synchronize(backend)
                    else
                        lm_m = resolve_mask_from_spec(lm_spec, get_mask, dims, backend)
                        if lm_m !== nothing
                            # Get bounds of lm_m
                            bbox = RuleExecutors.gpu_bounding_box(backend, lm_m)
                            if bbox !== nothing
                                z_min = (mode_str in ["top_only", "upper_only"]) ? 1 : bbox[5]
                                z_max = (mode_str == "bottom_only") ? dims[3] : bbox[6]
                                kernel! = CustomRules.limit_z_by_landmark_kernel!(backend)
                                kernel!(out_arr, in_m, dims, Int(z_min), Int(z_max), ndrange=dims)
                                KernelAbstractions.synchronize(backend)
                            else
                                println("FALLBACK FOR LimitZ: ", rule_type, " ", node_name, " in_m sum=", sum(in_m)); copyto!(out_arr, in_m)
                            end
                        else
                            println("FALLBACK FOR LimitZ: ", rule_type, " ", node_name, " in_m sum=", sum(in_m)); copyto!(out_arr, in_m)
                        end
                    end
                elseif rule_type == "AnteriorExtrusion"
                    CustomRules.AnteriorExtrusion((backend=backend,), out_arr, params_dict, in_m)
                elseif rule_type == "PleuralSpaceCustom"
                    CustomRules.PleuralSpaceCustom((backend=backend,), out_arr, params_dict, in_m)
                elseif rule_type == "HelperSplenicPDCustom"
                    CustomRules.HelperSplenicPDCustom((backend=backend,), out_arr, params_dict, in_m)
                elseif rule_type == "StomachLongAxisHelper"
                    CustomRules.StomachLongAxisHelper((backend=backend,), out_arr, params_dict, in_m)
                elseif rule_type == "PancreasSplitHelper"
                    # Second input is duodenum for Z reference
                    duo_m = nothing
                    if haskey(gpu_masks, "duodenum")
                        duo_m = gpu_masks["duodenum"]
                    end
                    CustomRules.PancreasSplitHelper((backend=backend,), out_arr, params_dict, in_m, duo_m)
                elseif rule_type == "PyloricSplitHelper"
                    CustomRules.PyloricSplitHelper((backend=backend,), out_arr, params_dict, in_m)
                elseif rule_type == "PresacralAnteriorCustom"
                    CustomRules.PresacralAnteriorCustom((backend=backend,), out_arr, params_dict, in_m)
                elseif rule_type == "SplitConnectedComponents"
                    CustomRules.SplitConnectedComponents((backend=backend,), out_arr, params_dict, in_m)
                elseif rule_type == "ErosionHelper"
                    CustomRules.ErosionHelper((backend=backend,), out_arr, params_dict, in_m)
                end
                mask_arr = out_arr
                level_outputs[node_name] = out_arr

            elseif rule_type == "AxillaryHelperB"
                out_arr = KernelAbstractions.zeros(backend, UInt8, dims)
                deps = get(rule_def, "dependencies", Dict())
                pec = get_mask(get(deps, "pectoralis", "pectoralis_major"))
                sub = get_mask(get(deps, "subclavius", "subclavius"))
                if pec === nothing || sub === nothing
                    println("    [INFO] Missing required anatomy for $node_name; skipping area calculation.")
                    mask_arr = out_arr
                    level_outputs[node_name] = out_arr
                    continue
                end
                CustomRules.AxillaryHelperB((backend=backend,), out_arr, params_dict, pec, sub)
                mask_arr = out_arr
                level_outputs[node_name] = out_arr

            elseif rule_type == "InternalIliacCustom"
                out_arr = KernelAbstractions.zeros(backend, UInt8, dims)
                prim = nothing
                if get(rule_def, "primitive", "") == "Cylinder"
                    side_str = lowercase(get(rule_def, "side", "Left"))
                    p1_dyn = nothing
                    p2_dyn = nothing
                    
                    # NEW: p1 = center of top slice of External Iliac
                    ext_iliac_key = "Abdominal_External_Iliac_$(uppercase(first(side_str)))$(side_str[2:end])"
                    m_ext = get_mask(ext_iliac_key)
                    if m_ext !== nothing
                        # GPU path: find top slice via Z-projection, then centroid per slice
                        z_sums = Array(dropdims(sum(m_ext, dims=(1, 2)), dims=(1, 2)))
                        top_k = 0
                        for kk in length(z_sums):-1:1
                            if z_sums[kk] > 0
                                top_k = kk
                                break
                            end
                        end
                        if top_k > 0
                            # Get centroids per slice on GPU, only transfer 1D results (~6 KB)
                            sum_x_cpu, sum_y_cpu, count_cpu = RuleExecutors.get_centroid_per_slice_gpu(backend, m_ext, dims)
                            cnt = count_cpu[top_k]
                            if cnt > 0
                                ci = sum_x_cpu[top_k] / cnt; cj = sum_y_cpu[top_k] / cnt
                                dir_00 = Float64(direction[1]); dir_11 = Float64(direction[5]); dir_22 = Float64(direction[9])
                                p1_x = Float64(origin[1]) + (ci - 1.0) * Float64(spacing[1]) * dir_00
                                p1_y = Float64(origin[2]) + (cj - 1.0) * Float64(spacing[2]) * dir_11
                                p1_z = Float64(origin[3]) + (Float64(top_k) - 1.0) * Float64(spacing[3]) * dir_22
                                p1_dyn = (p1_x, p1_y, p1_z)
                                println("    [DEBUG] $node_name: p1 from External Iliac top slice $top_k center=($ci,$cj) -> physical=($p1_x,$p1_y,$p1_z)")
                            end
                        end
                    end
                    
                    # p2 = sacrum/hip direction (unchanged)
                    m_sacrum = get_mask("sacrum")
                    m_hip = get_mask("hip_$side_str")
                    if m_hip === nothing; m_hip = get_mask("hip"); end
                    if m_sacrum !== nothing && m_hip !== nothing
                        p2_res = Landmarks.compute_internal_iliac_p2(m_sacrum isa CuArray ? m_sacrum : adapt(backend, m_sacrum), m_hip isa CuArray ? m_hip : adapt(backend, m_hip), spacing, origin, direction)
                        if p2_res !== nothing
                            p2_dyn = (p2_res[1], p2_res[2], p2_res[3])
                        end
                    end
                    if p1_dyn !== nothing && p2_dyn !== nothing
                        pt1 = (Float32(p1_dyn[1]), Float32(p1_dyn[2]), Float32(p1_dyn[3]))
                        length_mm = Float64(get(rule_def, "length_mm", 40.0))
                        vec = (p2_dyn[1]-p1_dyn[1], p2_dyn[2]-p1_dyn[2], p2_dyn[3]-p1_dyn[3])
                        

                        
                        norm_vec = sqrt(vec[1]^2 + vec[2]^2 + vec[3]^2)
                        if norm_vec > 1e-4
                            p2_dyn = (p1_dyn[1] + (vec[1]/norm_vec)*length_mm, p1_dyn[2] + (vec[2]/norm_vec)*length_mm, p1_dyn[3] + (vec[3]/norm_vec)*length_mm)
                            pt2 = (Float32(p2_dyn[1]), Float32(p2_dyn[2]), Float32(p2_dyn[3]))
                            rad = Float32(get(rule_def, "radius_mm", 7.0))
                            println("    [DEBUG] $node_name cylinder pts: p1=$pt1 -> p2=$pt2 (length=$(sqrt((pt2[1]-pt1[1])^2 + (pt2[2]-pt1[2])^2 + (pt2[3]-pt1[3])^2)))")
                            prim = RuleExecutors.execute_cylinder_primitive(backend, pt1, pt2, rad, dims, spacing, origin, direction)
                        end
                    end
                    if prim === nothing
                        println("    [INFO] Missing required anatomy/landmark for $node_name cylinder; skipping area calculation.")
                        mask_arr = out_arr
                        level_outputs[node_name] = out_arr
                        continue
                    end
                else
                    in_str = get(rule_def, "input", get(rule_def, "input_mask", get(rule_def, "mask_name", get(rule_def, "landmark", ""))))
                    prim = resolve_mask_from_spec(in_str, get_mask, dims, backend)
                    if prim === nothing
                        println("    [INFO] Missing required input mask for $node_name; skipping area calculation.")
                        mask_arr = out_arr
                        level_outputs[node_name] = out_arr
                        continue
                    end
                end
                
                art_l = get_mask("iliac_artery_left")
                if art_l === nothing; art_l = get_mask("iliac_artery_internal_left"); end
                if art_l === nothing; art_l = KernelAbstractions.zeros(backend, UInt8, dims); end
                art_r = get_mask("iliac_artery_right")
                if art_r === nothing; art_r = get_mask("iliac_artery_internal_right"); end
                if art_r === nothing; art_r = KernelAbstractions.zeros(backend, UInt8, dims); end
                
                CustomRules.InternalIliacCustom((backend=backend,), out_arr, params_dict, prim, art_l, art_r)
                mask_arr = out_arr
                level_outputs[node_name] = out_arr

            elseif rule_type in ["CommonIliacCustom", "ExternalIliacCustom", "IliacBifurcationCustom"]
                out_arr = KernelAbstractions.zeros(backend, UInt8, dims)
                
                art_l = get_mask("iliac_artery_left")
                if art_l === nothing; art_l = get_mask("iliac_artery_common_left"); end
                if art_l === nothing; art_l = get_mask("iliac_artery_external_left"); end
                if art_l === nothing; art_l = KernelAbstractions.zeros(backend, UInt8, dims); end
                
                art_r = get_mask("iliac_artery_right")
                if art_r === nothing; art_r = get_mask("iliac_artery_common_right"); end
                if art_r === nothing; art_r = get_mask("iliac_artery_external_right"); end
                if art_r === nothing; art_r = KernelAbstractions.zeros(backend, UInt8, dims); end
                
                ven_l = get_mask("iliac_vena_left")
                if ven_l === nothing; ven_l = get_mask("iliac_vena_common_left"); end
                if ven_l === nothing; ven_l = get_mask("iliac_vena_external_left"); end
                if ven_l === nothing; ven_l = KernelAbstractions.zeros(backend, UInt8, dims); end
                
                ven_r = get_mask("iliac_vena_right")
                if ven_r === nothing; ven_r = get_mask("iliac_vena_common_right"); end
                if ven_r === nothing; ven_r = get_mask("iliac_vena_external_right"); end
                if ven_r === nothing; ven_r = KernelAbstractions.zeros(backend, UInt8, dims); end

                if rule_type == "IliacBifurcationCustom"
                    CustomRules.IliacBifurcationCustom((backend=backend,), out_arr, params_dict, computed_landmarks, art_l, art_r, ven_l, ven_r)
                elseif rule_type == "CommonIliacCustom"
                    aorta = get_mask("aorta")
                    if aorta === nothing; aorta = KernelAbstractions.zeros(backend, UInt8, dims); end
                    CustomRules.CommonIliacCustom((backend=backend,), out_arr, params_dict, computed_landmarks, aorta, art_l, art_r, ven_l, ven_r)
                elseif rule_type == "ExternalIliacCustom"

                    side_str = lowercase(get(rule_def, "side", "left"))
                    p1_key = "internal_iliac_p1_" * side_str
                    if !haskey(computed_landmarks, p1_key)
                        art_k = "iliac_artery_$side_str"
                        m_art = get_mask(art_k)
                        if m_art === nothing; m_art = get_mask("iliac_artery"); end
                        m_l5 = get_mask("vertebrae_L5")
                        if m_art !== nothing && m_l5 !== nothing
                            p1_res = Landmarks.compute_internal_iliac_p1(m_art isa CuArray ? m_art : adapt(backend, m_art), m_l5 isa CuArray ? m_l5 : adapt(backend, m_l5), spacing, origin, direction)
                            if p1_res !== nothing
                                computed_landmarks[p1_key] = p1_res
                            end
                        end
                    end
                    in_str = get(rule_def, "input", get(rule_def, "input_mask", get(rule_def, "mask_name", get(rule_def, "landmark", ""))))
                    prim = resolve_mask_from_spec(in_str, get_mask, dims, backend)
                    if prim === nothing; prim = KernelAbstractions.zeros(backend, UInt8, dims); end
                    
                    side = lowercase(get(rule_def, "side", "left"))
                    femur = get_mask("femur_$side")
                    if femur === nothing; femur = get_mask("femur"); end
                    if femur === nothing; femur = KernelAbstractions.zeros(backend, UInt8, dims); end
                    
                    CustomRules.ExternalIliacCustom((backend=backend,), out_arr, params_dict, computed_landmarks, prim, femur, art_l, art_r, ven_l, ven_r)
                end
                mask_arr = out_arr
                level_outputs[node_name] = out_arr

            elseif rule_type == "EllipsoidFromBBox"
                lm_spec = get(rule_def, "landmark", get(rule_def, "base_landmark", get(rule_def, "proxy_landmark", "")))
                is_bilat = get(rule_def, "is_bilateral", false)
                lat_lm = get(rule_def, "lateral_landmark", "femur")
                inf_lm = get(rule_def, "inferior_landmark", "obturator_foramen")
                exp_mm = Float64(get(rule_def, "medial_expansion_mm", 30.0))
                rad_mult = Float64(get(rule_def, "radius_multiplier", 1.0))
                cx_mode = get(rule_def, "center_x_mode", "medial_edge")
                
                if is_bilat
                    ell_total = KernelAbstractions.zeros(backend, UInt8, dims)
                    for side in ["left", "right"]
                        lm_side = resolve_mask_from_spec("$(lm_spec)_$side", get_mask, dims, backend)
                        if lm_side === nothing
                            lm_side = resolve_mask_from_spec(lm_spec, get_mask, dims, backend)
                        end
                        
                        if lm_side !== nothing
                            femur_m = resolve_mask_from_spec("$(lat_lm)_$side", get_mask, dims, backend)
                            if femur_m === nothing; femur_m = resolve_mask_from_spec(lat_lm, get_mask, dims, backend); end
                            foram_m = resolve_mask_from_spec("$(inf_lm)_$side", get_mask, dims, backend)
                            if foram_m === nothing; foram_m = resolve_mask_from_spec(inf_lm, get_mask, dims, backend); end
                            
                            ell_side = RuleExecutors.execute_ellipsoid_from_bbox(
                                backend, lm_side, side, dims, spacing, origin, direction;
                                femur_mask=femur_m, foramen_mask=foram_m,
                                expand_medial_mm=exp_mm, radius_multiplier=rad_mult,
                                center_x_mode=cx_mode
                            )
                            ell_total .|= ell_side
                            s_cap = uppercase(side[1:1]) * side[2:end]
                            level_outputs["$(node_name)_$s_cap"] = ell_side
                        end
                    end
                    level_outputs[node_name] = ell_total
                else
                    lm_m = resolve_mask_from_spec(lm_spec, get_mask, dims, backend)
                    if lm_m !== nothing
                        femur_m = resolve_mask_from_spec(lat_lm, get_mask, dims, backend)
                        foram_m = resolve_mask_from_spec(inf_lm, get_mask, dims, backend)
                        ell_m = RuleExecutors.execute_ellipsoid_from_bbox(
                            backend, lm_m, "left", dims, spacing, origin, direction;
                            femur_mask=femur_m, foramen_mask=foram_m,
                            expand_medial_mm=exp_mm, radius_multiplier=rad_mult,
                            center_x_mode=cx_mode
                        )
                        level_outputs[node_name] = ell_m
                    end
                end
                empty!(composite_cache)
                continue
            elseif rule_type == "VolumetricBoundary2DLateralGrowth"
                # Read landmark lists and directions from JSON (fully parameterized, not hardcoded)
                boundaries_def = get(rule_def, "boundaries", Dict())
                
                # Start / Medial keys
                start_keys_cfg = get(boundaries_def, "medial", get(boundaries_def, "start", get(boundaries_def, "vessel", ["internal_jugular_vein", "internal_carotid_artery", "common_carotid_artery"])))
                start_keys_cfg = start_keys_cfg isa String ? [start_keys_cfg] : Vector{String}(start_keys_cfg)
                
                # Obstacle / Lateral keys (strictly lateral obstacles, no skull)
                obstacle_keys_cfg = get(boundaries_def, "lateral", get(boundaries_def, "obstacle", ["sternocleidomastoid", "parotid_gland"]))
                obstacle_keys_cfg = obstacle_keys_cfg isa String ? [obstacle_keys_cfg] : Vector{String}(obstacle_keys_cfg)
                
                # Limit / Anterior keys
                limit_keys_cfg = get(boundaries_def, "anterior", get(boundaries_def, "limit", []))
                limit_keys_cfg = limit_keys_cfg isa String ? [limit_keys_cfg] : Vector{String}(limit_keys_cfg)
                
                # Split landmark (e.g. internal_jugular_vein for IIa/IIb)
                split_lm_cfg = get(rule_def, "split_landmark", get(boundaries_def, "split", "internal_jugular_vein"))
                
                # Superior / Inferior keys
                superior_key = get(boundaries_def, "superior", "skull")
                inferior_key = get(boundaries_def, "inferior", "hyoid")
                
                is_bilat_vbg = get(rule_def, "is_bilateral", false)
                
                function _run_vbg_for_side(s_str)
                    # 1. Start mask
                    start_in = KernelAbstractions.zeros(backend, UInt8, dims)
                    for base_name in start_keys_cfg
                        for ks in ["$(base_name)_$s_str", base_name]
                            vm = get_mask(ks)
                            if vm !== nothing; start_in .|= vm; break; end
                        end
                    end
                    
                    # Compute Z bounds from superior / inferior landmarks FIRST so we can extrude SCM
                    sup_z_phys = RuleExecutors.resolve_landmark_z(superior_key, "max", 0.0, get_mask, computed_landmarks, dims, spacing, origin, direction; side=s_str)
                    inf_z_phys = RuleExecutors.resolve_landmark_z(inferior_key, "min", 0.0, get_mask, computed_landmarks, dims, spacing, origin, direction; side=s_str)
                    stern_z_phys = RuleExecutors.resolve_landmark_z("sternum", "max", 0.0, get_mask, computed_landmarks, dims, spacing, origin, direction; side=s_str)
                    
                    z_sup_k = (sup_z_phys !== nothing) ? Int(floor((Float64(sup_z_phys) - origin[3]) / spacing[3])) + 1 : dims[3]
                    z_inf_k = (inf_z_phys !== nothing) ? Int(floor((Float64(inf_z_phys) - origin[3]) / spacing[3])) + 1 : 1
                    if stern_z_phys !== nothing
                        z_stern_k = Int(floor((Float64(stern_z_phys) - origin[3]) / spacing[3])) + 1
                        z_inf_k = max(z_inf_k, z_stern_k)
                    end
                    z_min = max(1, min(z_sup_k, z_inf_k))
                    z_max = min(dims[3], max(z_sup_k, z_inf_k))

                    # 2. Obstacle mask (strictly obstacle keys)
                    obs_in = KernelAbstractions.zeros(backend, UInt8, dims)
                    for base_name in obstacle_keys_cfg
                        for ks in ["$(base_name)_$s_str", "$(base_name)_$(uppercase(s_str[1:1]))$(s_str[2:end])", base_name]
                            om = get_mask(ks)
                            if om !== nothing
                                obs_in .|= om
                                break
                            end
                        end
                    end
                    
                    # 3. Limit mask (e.g. anterior limit)
                    limit_in = nothing
                    has_limit = false
                    if !isempty(limit_keys_cfg)
                        lim_arr = KernelAbstractions.zeros(backend, UInt8, dims)
                        found_lim = false
                        for base_name in limit_keys_cfg
                            for ks in ["$(base_name)_$s_str", "$(base_name)_$(uppercase(s_str[1:1]))$(s_str[2:end])", base_name]
                                lm = get_mask(ks)
                                if lm !== nothing; lim_arr .|= lm; found_lim = true; break; end
                            end
                        end
                        if found_lim
                            limit_in = lim_arr
                            has_limit = true
                        end
                    end
                    
                    # 4. Split mask
                    split_in = nothing
                    has_split = false
                    if !isempty(split_lm_cfg)
                        for ks in ["$(split_lm_cfg)_$s_str", split_lm_cfg]
                            sm = get_mask(ks)
                            if sm !== nothing; split_in = sm; has_split = true; break; end
                        end
                    end
                    
                    p = copy(params_dict)
                    p["side"] = s_str
                    p["z_min"] = z_min
                    p["z_max"] = z_max
                    p["has_limit"] = has_limit
                    p["has_split"] = has_split
                    p["spacing_x"] = Float32(spacing[1])
                    p["spacing_y"] = Float32(spacing[2])
                    p["spacing_z"] = Float32(spacing[3])
                    
                    for k in ["growth_axis", "growth_target", "sublevel", "max_dist_mm", "limit_to_obstacle_min_if_no_hit"]
                        if haskey(rule_def, k) && !haskey(p, k)
                            p[k] = rule_def[k]
                        end
                    end
                    
                    out_arr = KernelAbstractions.zeros(backend, UInt8, dims)
                    CustomRules.VolumetricBoundary2DLateralGrowth(
                        (backend=backend,), out_arr, p,
                        start_in, obs_in, limit_in, split_in
                    )
                    return out_arr
                end
                
                if is_bilat_vbg
                    for s in ["Left", "Right"]
                        s_str = lowercase(s)
                        out = _run_vbg_for_side(s_str)
                        level_outputs["$(node_name)_$s"] = out
                    end
                else
                    side_str = get(rule_def, "side", occursin("right", lowercase(node_name)) ? "right" : "left")
                    level_outputs[node_name] = _run_vbg_for_side(side_str)
                end
                empty!(composite_cache)
                continue

            elseif rule_type in ["VolumetricBoundary2D", "VolumetricBoundary"]
                bounds = get(rule_def, "boundaries", Dict())
                growth_mode = get(rule_def, "growth_mode", "convex_hull")
                posterior_offset_mm = Float64(get(rule_def, "posterior_offset_mm", 5.0))
                keep_all = get(rule_def, "keep_all_components", false)
                is_bilat_rule = get(rule_def, "is_bilateral", false)
                
                function extract_bounds_for_side(target_side)
                    m_arrs = Any[]; l_arrs = Any[]; p_arrs = Any[]; a_arrs = Any[]; s_arrs = Any[]
                    resolve_spec(spec) = begin
                        if !isempty(target_side)
                            m = resolve_mask_from_spec("$(spec)_$(lowercase(target_side))", get_mask, dims, backend)
                            if m === nothing; m = resolve_mask_from_spec("$(spec)_$(target_side)", get_mask, dims, backend); end
                            if m === nothing; m = resolve_mask_from_spec(spec, get_mask, dims, backend); end
                            return m
                        else
                            return resolve_mask_from_spec(spec, get_mask, dims, backend)
                        end
                    end
                    
                    if haskey(bounds, "medial")
                        for m in (bounds["medial"] isa Vector ? bounds["medial"] : [bounds["medial"]])
                            arr = resolve_spec(m)
                            if arr !== nothing; push!(m_arrs, arr); end
                        end
                    end
                    if haskey(bounds, "lateral")
                        for l in (bounds["lateral"] isa Vector ? bounds["lateral"] : [bounds["lateral"]])
                            arr = resolve_spec(l)
                            if arr !== nothing; push!(l_arrs, arr); end
                        end
                    end
                    if haskey(bounds, "posterior")
                        for p in (bounds["posterior"] isa Vector ? bounds["posterior"] : [bounds["posterior"]])
                            arr = resolve_spec(p)
                            if arr !== nothing; push!(p_arrs, arr); end
                        end
                    end
                    if haskey(bounds, "anterior")
                        for a in (bounds["anterior"] isa Vector ? bounds["anterior"] : [bounds["anterior"]])
                            arr = resolve_spec(a)
                            if arr !== nothing; push!(a_arrs, arr); end
                        end
                    end
                    for (k, v) in bounds
                        for i_v in (v isa Vector ? v : [v])
                            arr = resolve_spec(i_v)
                            if arr !== nothing; push!(s_arrs, arr); end
                        end
                    end
                    return s_arrs, m_arrs, l_arrs, p_arrs, a_arrs
                end
                
                if is_bilat_rule
                    s_l, m_l, l_l, p_l, a_l = extract_bounds_for_side("Left")
                    s_r, m_r, l_r, p_r, a_r = extract_bounds_for_side("Right")
                    
                    mask_l = RuleExecutors.execute_volumetric_boundary_2d_gpu(
                        backend, dims, spacing,
                        s_l, m_l, l_l, p_l, a_l;
                        posterior_offset_mm=posterior_offset_mm,
                        growth_mode=growth_mode,
                        keep_all_components=keep_all
                    )
                    mask_r = RuleExecutors.execute_volumetric_boundary_2d_gpu(
                        backend, dims, spacing,
                        s_r, m_r, l_r, p_r, a_r;
                        posterior_offset_mm=posterior_offset_mm,
                        growth_mode=growth_mode,
                        keep_all_components=keep_all
                    )
                    mask_arr = mask_l .| mask_r
                else
                    s_arrs, m_arrs, l_arrs, p_arrs, a_arrs = extract_bounds_for_side("")
                    mask_arr = RuleExecutors.execute_volumetric_boundary_2d_gpu(
                        backend, dims, spacing,
                        s_arrs, m_arrs, l_arrs, p_arrs, a_arrs;
                        posterior_offset_mm=posterior_offset_mm,
                        growth_mode=growth_mode,
                        keep_all_components=keep_all
                    )
                end
                level_outputs[node_name] = mask_arr

            elseif rule_type in ["ConvexHullBridge", "ConvexHullBridging"]
                m1 = nothing
                m2 = nothing
                if haskey(rule_def, "bridge_landmarks") && length(rule_def["bridge_landmarks"]) >= 2
                    m1 = resolve_mask_from_spec(rule_def["bridge_landmarks"][1], get_mask, dims, backend)
                    m2 = resolve_mask_from_spec(rule_def["bridge_landmarks"][2], get_mask, dims, backend)
                else
                    lm1_spec = get(rule_def, "landmark_1", "bronchi_left")
                    lm2_spec = get(rule_def, "landmark_2", "bronchi_right")
                    m1 = resolve_mask_from_spec(lm1_spec, get_mask, dims, backend)
                    m2 = resolve_mask_from_spec(lm2_spec, get_mask, dims, backend)
                end
                if m1 !== nothing && m2 !== nothing
                    side_str = lowercase(get(rule_def, "side", ""))
                    if side_str == "left" || side_str == "right"
                        keep_str = side_str == "left" ? "max" : "min"
                        m1 = RuleExecutors.execute_split_mask(backend, m1, "x", "image_center", keep_str, dims)
                        m2 = RuleExecutors.execute_split_mask(backend, m2, "x", "image_center", keep_str, dims)
                    end
                    plane_str = lowercase(get(rule_def, "plane", "axial"))
                    mask_arr = RuleExecutors.execute_convex_hull_bridge(backend, m1, m2, dims, spacing, origin, direction; plane=plane_str)
                else
                    mask_arr = KernelAbstractions.zeros(backend, UInt8, dims)
                end
                level_outputs[node_name] = mask_arr

            elseif rule_type in ["ConvexHull2D", "ConvexHull"]
                targets = get(rule_def, "targets", get(rule_def, "dependencies", []))
                if isempty(targets)
                    if haskey(rule_def, "input_a") && haskey(rule_def, "input_b")
                        targets = [rule_def["input_a"], rule_def["input_b"]]
                    elseif haskey(rule_def, "inputs")
                        targets = rule_def["inputs"] isa Vector ? rule_def["inputs"] : [rule_def["inputs"]]
                    end
                end
                if targets isa String
                    targets = [targets]
                end
                merged_m = KernelAbstractions.zeros(backend, UInt8, dims)
                for tgt in targets
                    t_m = resolve_mask_from_spec(tgt, get_mask, dims, backend)
                    if t_m !== nothing
                        merged_m = merged_m .| t_m
                    end
                end
                planes = get(rule_def, "planes", ["axial"])
                if planes isa String; planes = [planes]; end
                combined = KernelAbstractions.zeros(backend, UInt8, dims)
                for plane in planes
                    hull = RuleExecutors.execute_convex_hull_bridge(backend, merged_m, merged_m, dims, spacing, origin, direction; plane=String(plane))
                    combined = combined .| hull
                    hull = nothing
                end
                mask_arr = combined
                level_outputs[node_name] = mask_arr
            elseif rule_type in ["BooleanOperation", "Union", "BitwiseOr", "Combine"]
                # Multi-input Union / Intersection
                in_list = get(rule_def, "inputs", get(rule_def, "landmarks", get(rule_def, "targets", get(rule_def, "target_landmarks", get(rule_def, "components", get(rule_def, "input", []))))))
                if !(in_list isa Vector)
                    in_list = isempty(in_list) ? [] : [in_list]
                end
                is_intersect = lowercase(get(rule_def, "method", "union")) in ["intersection", "intersect", "and"]
                is_bilat = get(rule_def, "is_bilateral", false)
                dil_val = Float64(get(rule_def, "dilate", get(rule_def, "dilation_mm", 0.0)))
                
                if is_bilat
                    for s in ["Left", "Right"]
                        res_side = nothing
                        for inp in in_list
                            inp_s = resolve_mask_from_spec("$(inp)_$s", get_mask, dims, backend)
                            if inp_s === nothing
                                inp_s = resolve_mask_from_spec("$(inp)_$(lowercase(s))", get_mask, dims, backend)
                            end
                            if inp_s === nothing
                                inp_s_raw = resolve_mask_from_spec(inp, get_mask, dims, backend)
                                if inp_s_raw !== nothing
                                    inp_s = copy(inp_s_raw)
                                    mid_x = div(dims[1], 2)
                                    if lowercase(s) == "left"
                                        inp_s[1:mid_x-1, :, :] .= UInt8(0)
                                    else
                                        inp_s[mid_x:end, :, :] .= UInt8(0)
                                    end
                                end
                            end
                            if inp_s !== nothing
                                if res_side === nothing
                                    res_side = copy(inp_s)
                                else
                                    if is_intersect
                                        res_side .&= (inp_s .> UInt8(0))
                                    else
                                        res_side .|= (inp_s .> UInt8(0))
                                    end
                                end
                                inp_s = nothing
                            end
                        end
                        if res_side !== nothing
                            if dil_val > 0.0
                                res_side = RuleExecutors.execute_anisotropic_expansion(backend, res_side, dims, spacing, Dict("all" => dil_val); side=lowercase(s))
                            end
                            level_outputs["$(node_name)_$s"] = res_side
                            level_outputs["$(node_name)_$(lowercase(s))"] = res_side
                        end
                    end
                    l_m = get(level_outputs, "$(node_name)_Left", nothing)
                    r_m = get(level_outputs, "$(node_name)_Right", nothing)
                    if l_m !== nothing && r_m !== nothing
                        level_outputs[node_name] = l_m .| r_m
                    elseif l_m !== nothing
                        level_outputs[node_name] = l_m
                    elseif r_m !== nothing
                        level_outputs[node_name] = r_m
                    end
                else
                    res_m = nothing
                    for inp in in_list
                        inp_m = resolve_mask_from_spec(inp, get_mask, dims, backend)
                        if inp_m !== nothing
                            if res_m === nothing
                                res_m = copy(inp_m)
                            else
                                if is_intersect
                                    res_m .&= (inp_m .> UInt8(0))
                                else
                                    res_m .|= (inp_m .> UInt8(0))
                                end
                            end
                            inp_m = nothing
                        end
                    end
                    if res_m !== nothing
                        if dil_val > 0.0
                            res_m = RuleExecutors.execute_anisotropic_expansion(backend, res_m, dims, spacing, Dict("all" => dil_val))
                        end
                        level_outputs[node_name] = res_m
                    end
                end
                empty!(composite_cache)
                continue
            elseif rule_type in ["Exclude", "Subtract", "Difference"]
                deps = get(rule_def, "Dependencies", get(rule_def, "dependencies", get(rule_def, "inputs", [])))
                if length(deps) >= 2
                    b_spec = deps[1]
                    b_img = resolve_mask_from_spec(b_spec, get_mask, dims, backend)
                    if b_img !== nothing
                        res = copy(b_img)
                        for i in 2:length(deps)
                            e_spec = deps[i]
                            e_img = resolve_mask_from_spec(e_spec, get_mask, dims, backend)
                            if e_img !== nothing
                                res .= ifelse.(e_img .> UInt8(0), UInt8(0), res)
                            end
                        end
                        level_outputs[node_name] = res
                    end
                elseif haskey(rule_def, "input_a") && haskey(rule_def, "input_b")
                    b_img = resolve_mask_from_spec(rule_def["input_a"], get_mask, dims, backend)
                    e_img = resolve_mask_from_spec(rule_def["input_b"], get_mask, dims, backend)
                    if b_img !== nothing && e_img !== nothing
                        res = copy(b_img)
                        res .= ifelse.(e_img .> UInt8(0), UInt8(0), res)
                        level_outputs[node_name] = res
                    elseif b_img !== nothing
                        level_outputs[node_name] = copy(b_img)
                    end
                end
                empty!(composite_cache)
                continue
            elseif rule_type in ["GeometricConstraint", "Intersect", "BitwiseAnd"]
                inp_spec = get(rule_def, "mask_name", get(rule_def, "input", get(rule_def, "landmark", get(rule_def, "inputs", get(rule_def, "base_landmark", "")))))
                is_bilat = get(rule_def, "is_bilateral", false)
                constraints = get(rule_def, "constraints", [])
                
                c_type = get(rule_def, "constraint_type", "")
                if !isempty(c_type)
                    c_dict = Dict(
                        "constraint_type" => c_type,
                        "landmark" => get(rule_def, "landmark", ""),
                        "boundary_part" => get(rule_def, "boundary_part", "max"),
                        "offset_mm" => Float64(get(rule_def, "offset_mm", 0.0))
                    )
                    push!(constraints, c_dict)
                end
                
                if is_bilat
                    for s in ["Left", "Right"]
                        inp_s = resolve_mask_from_spec("$(inp_spec)_$s", get_mask, dims, backend)
                        if inp_s === nothing
                            inp_s = resolve_mask_from_spec("$(inp_spec)_$(lowercase(s))", get_mask, dims, backend)
                        end
                        if inp_s === nothing
                            inp_s = resolve_mask_from_spec(inp_spec, get_mask, dims, backend)
                        end
                        if inp_s !== nothing
                            res_side = copy(inp_s)
                            if !isempty(constraints)
                                res_side = RuleExecutors.apply_constraints(backend, res_side, constraints, get_mask, computed_landmarks, dims, spacing, origin, direction; side=lowercase(s))
                            end
                            level_outputs["$(node_name)_$s"] = res_side
                        end
                    end
                else
                    inp_m = resolve_mask_from_spec(inp_spec, get_mask, dims, backend)
                    if inp_m !== nothing
                        res_m = copy(inp_m)
                        if !isempty(constraints)
                            res_m = RuleExecutors.apply_constraints(backend, res_m, constraints, get_mask, computed_landmarks, dims, spacing, origin, direction)
                        end
                        level_outputs[node_name] = res_m
                    end
                end
            elseif rule_type == "PropagateZ"
                seed_spec = get(rule_def, "base_landmark", get(rule_def, "input", ""))
                dir_str = get(rule_def, "direction", "inferior")
                term_spec = get(rule_def, "terminus", nothing)
                is_bilat = get(rule_def, "is_bilateral", false)
                
                if is_bilat
                    for s in ["Left", "Right"]
                        seed_s = resolve_mask_from_spec("$(seed_spec)_$s", get_mask, dims, backend)
                        if seed_s === nothing; seed_s = resolve_mask_from_spec("$(seed_spec)_$(lowercase(s))", get_mask, dims, backend); end
                        if seed_s === nothing; seed_s = resolve_mask_from_spec(seed_spec, get_mask, dims, backend); end
                        
                        stop_s = nothing
                        if term_spec !== nothing
                            stop_s = resolve_mask_from_spec("$(term_spec)_$s", get_mask, dims, backend)
                            if stop_s === nothing; stop_s = resolve_mask_from_spec("$(term_spec)_$(lowercase(s))", get_mask, dims, backend); end
                            if stop_s === nothing; stop_s = resolve_mask_from_spec(term_spec, get_mask, dims, backend); end
                        end
                        
                        if seed_s !== nothing
                            level_outputs["$(node_name)_$s"] = RuleExecutors.execute_z_propagation(backend, seed_s, dir_str, stop_s, direction, dims)
                        end
                    end
                    l_m = get(level_outputs, "$(node_name)_Left", nothing)
                    r_m = get(level_outputs, "$(node_name)_Right", nothing)
                    if l_m !== nothing && r_m !== nothing; level_outputs[node_name] = l_m .| r_m
                    elseif l_m !== nothing; level_outputs[node_name] = l_m
                    elseif r_m !== nothing; level_outputs[node_name] = r_m; end
                else
                    rule_side_str = get(rule_def, "side", nothing)
                    if rule_side_str === nothing
                        nn_lower = lowercase(node_name)
                        if occursin("_left", nn_lower) || endswith(nn_lower, "_l")
                            rule_side_str = "left"
                        elseif occursin("_right", nn_lower) || endswith(nn_lower, "_r")
                            rule_side_str = "right"
                        end
                    end
                    
                    seed_m = nothing
                    if rule_side_str !== nothing
                        rule_side_cap = titlecase(rule_side_str)
                        seed_m = resolve_mask_from_spec("$(seed_spec)_$(rule_side_cap)", get_mask, dims, backend)
                        if seed_m === nothing
                            seed_m = resolve_mask_from_spec("$(seed_spec)_$(rule_side_str)", get_mask, dims, backend)
                        end
                    end
                    if seed_m === nothing
                        seed_m = resolve_mask_from_spec(seed_spec, get_mask, dims, backend)
                    end
                    
                    stop_m = nothing
                    if term_spec !== nothing
                        if rule_side_str !== nothing
                            rule_side_cap = titlecase(rule_side_str)
                            stop_m = resolve_mask_from_spec("$(term_spec)_$(rule_side_cap)", get_mask, dims, backend)
                            if stop_m === nothing
                                stop_m = resolve_mask_from_spec("$(term_spec)_$(rule_side_str)", get_mask, dims, backend)
                            end
                        end
                        if stop_m === nothing
                            stop_m = resolve_mask_from_spec(term_spec, get_mask, dims, backend)
                        end
                    end
                    
                    if seed_m !== nothing
                        level_outputs[node_name] = RuleExecutors.execute_z_propagation(backend, seed_m, dir_str, stop_m, direction, dims)
                    end
                end
            elseif rule_type == "SplitMask"
                lm_spec = get(rule_def, "landmark", get(rule_def, "base_landmark", get(rule_def, "input", "")))
                axis_str = get(rule_def, "axis", "x")
                method_str = get(rule_def, "split_method", "center")
                keep_str = get(rule_def, "side_to_keep", get(rule_def, "keep", "min"))
                is_bilat = get(rule_def, "is_bilateral", false)
                
                if is_bilat
                    for s in ["Left", "Right"]
                        lm_s = resolve_mask_from_spec("$(lm_spec)_$s", get_mask, dims, backend)
                        if lm_s === nothing; lm_s = resolve_mask_from_spec("$(lm_spec)_$(lowercase(s))", get_mask, dims, backend); end
                        if lm_s === nothing; lm_s = resolve_mask_from_spec(lm_spec, get_mask, dims, backend); end
                        if lm_s !== nothing
                            res_s = RuleExecutors.execute_split_mask(backend, lm_s, axis_str, method_str, keep_str, dims)
                            if !get(rule_def, "keep_all_components", false)
                                res_s = RuleExecutors.get_largest_connected_component(res_s)
                            end
                            level_outputs["$(node_name)_$s"] = res_s
                        end
                    end
                    l_m = get(level_outputs, "$(node_name)_Left", nothing)
                    r_m = get(level_outputs, "$(node_name)_Right", nothing)
                    if l_m !== nothing && r_m !== nothing; level_outputs[node_name] = l_m .| r_m
                    elseif l_m !== nothing; level_outputs[node_name] = l_m
                    elseif r_m !== nothing; level_outputs[node_name] = r_m; end
                else
                    lm_m = resolve_mask_from_spec(lm_spec, get_mask, dims, backend)
                    if lm_m !== nothing
                        res_m = RuleExecutors.execute_split_mask(backend, lm_m, axis_str, method_str, keep_str, dims)
                        if !get(rule_def, "keep_all_components", false)
                            res_m = RuleExecutors.get_largest_connected_component(res_m)
                        end
                        level_outputs[node_name] = res_m
                    end
                end
                empty!(composite_cache)
                continue
            else
                # Default Pointwise / Direct copy (handles Mask, LateralBridge, unknown rule types)
                # Check all common source keys including mask_name for the Mask rule type
                base_spec = get(rule_def, "base_landmark", get(rule_def, "mask_name",
                               get(rule_def, "input", get(rule_def, "landmark", ""))))
                base_m = resolve_mask_from_spec(base_spec, get_mask, dims, backend)
                if base_m !== nothing
                    level_outputs[node_name] = copy(base_m)
                end
            end
            empty!(composite_cache)
        end # closes for node_name in level_nodes
        t_rule_gen_end = time()

        # Post-process level outputs (constraints, Z-plane restrictions, exclusions) matching Python engine order
        t_postproc_start = time()
        for (node_name, mask_arr) in collect(level_outputs)
            if mask_arr === nothing
                empty!(composite_cache)
                continue
            end
            base_rule_name = node_name
            side_eff = ""
            if !haskey(rules, base_rule_name)
                for suffix in ["_Left", "_Right", "_left", "_right"]
                    if endswith(base_rule_name, suffix)
                        candidate = base_rule_name[1:end-length(suffix)]
                        if haskey(rules, candidate)
                            base_rule_name = candidate
                            side_eff = lowercase(suffix[2:end])
                            break
                        end
                    end
                end
            end
            rule_def = get(rules, base_rule_name, Dict())
            
            if isempty(side_eff)
                if occursin("_left", lowercase(node_name)) || endswith(lowercase(node_name), "_l")
                    side_eff = "left"
                elseif occursin("_right", lowercase(node_name)) || endswith(lowercase(node_name), "_r")
                    side_eff = "right"
                elseif haskey(rule_def, "side")
                    side_eff = lowercase(string(rule_def["side"]))
                end
            end
            base_clean = replace(node_name, r"(_Left|_Right|_left|_right|_L|_R)$" => "")

            # CUDA.reclaim() removed for performance - done per-level only
            if DEBUG_VERBOSE; pre_count = gpu_count(mask_arr); end
            
            # 1. Append landmarks / Dilation ref (pre-constraint)
            for app_lm in get(rule_def, "append_landmarks", [])
                app_lookup = !isempty(side_eff) ? "$(app_lm)_$side_eff" : app_lm
                app_m = resolve_mask_from_spec(app_lookup, get_mask, dims, backend)
                if app_m === nothing; app_m = resolve_mask_from_spec(app_lm, get_mask, dims, backend); end
                if app_m !== nothing
                    app_gpu = adapt(typeof(mask_arr), map(x -> x > 0 ? UInt8(1) : UInt8(0), app_m))
                    mask_arr .|= (app_gpu .> UInt8(0))
                end
            end
            
            dil_ref = get(rule_def, "dilation_ref", "")
            if !isempty(dil_ref)
                dil_lookup = !isempty(side_eff) ? "$(dil_ref)_$side_eff" : dil_ref
                dil_m = resolve_mask_from_spec(dil_lookup, get_mask, dims, backend)
                if dil_m === nothing; dil_m = resolve_mask_from_spec(dil_ref, get_mask, dims, backend); end
                if dil_m !== nothing
                    dil_mm = Float64(get(rule_def, "dilation_mm", 10.0))
                    dil_exp = RuleExecutors.execute_anisotropic_expansion(backend, dil_m, dims, spacing, Dict("all" => dil_mm))
                    mask_arr .|= (dil_exp .> UInt8(0))
                end
            end

            # 2. Geometric Constraints (Step 0 in Python)
            constraints = get(rule_def, "constraints", [])
            
            # Fix 30: Override PosteriorTo fused_spine for Neck_Nuchal
            # fused_spine includes skull at cervical level, pushing Y center to ~340
            # which removes the true occipital node region at Y=255-320.
            # Use vertebral_column_fused (vertebral bodies only) for proper Y boundary.
            if (base_rule_name == "Neck_Nuchal" || base_clean == "Neck_Nuchal") && !isempty(constraints)
                vert_col = get_mask("vertebral_column_fused")
                if vert_col === nothing; vert_col = get_mask("vertebrae_C1"); end
                if vert_col !== nothing
                    min_y_cpu, max_y_cpu = RuleExecutors.get_slice_y_bounds_gpu(backend, vert_col, dims)
                    RuleExecutors.zero_anterior_to_y_gpu!(backend, mask_arr, min_y_cpu, dims)
                    println("    [FIX30] Neck_Nuchal: Applied per-slice PosteriorTo vertebral_column_fused (replacing fused_spine)")
                end
                # Remove the original PosteriorTo constraint to avoid double-application
                constraints = filter(c -> !(get(c, "constraint_type", "") == "PosteriorTo"), constraints)
            end
            
            # Cache compiled status once for all postproc steps (used for exclusion skipping)
            compiled_directly = VMCompiler.can_compile_directly(rule_def, packed_tensor, ALIASES, base_rule_name)
            
            if !isempty(constraints)
                mask_arr = RuleExecutors.apply_constraints(backend, mask_arr, constraints, get_mask, computed_landmarks, dims, spacing, origin, direction; side=side_eff)
            end

            if haskey(rule_def, "limit_landmark")
                limit_spec = rule_def["limit_landmark"]
                limit_dir = get(rule_def, "limit_direction", "superior")
                c_type = limit_dir == "superior" ? "InferiorTo" : "SuperiorTo"
                mask_arr = RuleExecutors.apply_constraints(backend, mask_arr, [Dict("constraint_type" => c_type, "landmark" => limit_spec, "boundary_part" => "max")], get_mask, computed_landmarks, dims, spacing, origin, direction; side=side_eff)
            end
            if DEBUG_VERBOSE; post_constr_count = gpu_count(mask_arr); end
            
            # 3. Z-Plane restriction (Step 1 in Python)
            z_restr = get(rule_def, "z_plane_restriction", nothing)
            if z_restr isa Dict
                mask_arr = RuleExecutors.apply_z_plane_restriction(backend, mask_arr, z_restr, get_mask, computed_landmarks, dims, spacing, origin, direction; side=side_eff)
            end
            if DEBUG_VERBOSE; post_z_count = gpu_count(mask_arr); end

            # 3.5 Layer-wise propagation (e.g. for Abdominal_Hiatus)
            if haskey(rule_def, "layer_wise_propagation")
                lwp = rule_def["layer_wise_propagation"]
                targets = get(lwp, "target_landmark", get(lwp, "targets", []))
                dir_lwp = get(lwp, "direction", "inferior")
                step_mm = Float64(get(lwp, "step_dilation_mm", 2.0))
                max_mm = Float64(get(lwp, "max_mm", 200.0))
                step_px = max(1, round(Int, step_mm / spacing[1]))
                max_slices = round(Int, max_mm / spacing[3])
                
                target_m = resolve_mask_from_spec(targets, get_mask, dims, backend)
                if target_m !== nothing && gpu_count(target_m) > 0
                    RuleExecutors.run_layer_wise_propagation_gpu!(backend, mask_arr, target_m, dir_lwp, step_px, max_slices, dims)
                end
            end

            # 4. Exclusions with non-zero margin (only 3 rules in entire dataset)
            # All zero-margin exclusions are executed directly by the MegaKernel from packed_tensor!
            ex_list = unique(vcat(get(rule_def, "exclude", []), get(rule_def, "exclude_structures", []), get(rule_def, "exclude_landmarks", [])))
            for ex_raw in ex_list
                ex_margin = ex_raw isa Dict ? Float64(get(ex_raw, "margin_mm", 0.0)) : 0.0
                ex_name = ex_raw isa Dict ? string(get(ex_raw, "landmark", get(ex_raw, "name", ""))) : string(ex_raw)
                isempty(ex_name) && continue
                ex_lookup = !isempty(side_eff) ? "$(ex_name)_$side_eff" : ex_name
                
                # If the rule was compiled directly, MegaKernel already handled zero-margin exclusions in packed_tensor.
                # Check if exclusion exists in packed_tensor directly or via alias
                reg = haskey(packed_tensor.registry, ex_lookup) ? packed_tensor.registry[ex_lookup] :
                      (haskey(packed_tensor.registry, ex_name) ? packed_tensor.registry[ex_name] :
                      VMCompiler.resolve_registry_entry(packed_tensor, ex_lookup, ALIASES))
                if reg === nothing
                    reg = VMCompiler.resolve_registry_entry(packed_tensor, ex_name, ALIASES)
                end
                is_in_packed = reg !== nothing
                
                if !compiled_directly && ex_margin == 0.0 && is_in_packed
                    ch, id = reg
                    if id > UInt8(0)
                        mask_arr .= ifelse.(view(packed_tensor.data, :, :, :, ch) .== id, UInt8(0), mask_arr)
                    end
                elseif !compiled_directly && ex_margin == 0.0 && !isempty(VMCompiler.resolve_exclusion_constituents(packed_tensor, ex_name, ALIASES))
                    constituents = VMCompiler.resolve_exclusion_constituents(packed_tensor, ex_name, ALIASES)
                    for c_name in constituents
                        c_reg = VMCompiler.resolve_registry_entry(packed_tensor, c_name, ALIASES)
                        if c_reg !== nothing
                            c_ch, c_id = c_reg
                            if c_id > UInt8(0)
                                mask_arr .= ifelse.(view(packed_tensor.data, :, :, :, c_ch) .== c_id, UInt8(0), mask_arr)
                            end
                        end
                    end
                elseif ex_margin != 0.0 || !is_in_packed || !compiled_directly
                    ex_m = resolve_mask_from_spec(ex_lookup, get_mask, dims, backend)
                    if ex_m === nothing; ex_m = resolve_mask_from_spec(ex_name, get_mask, dims, backend); end
                    if ex_m !== nothing
                        if ex_margin > 0.0
                            ex_m = RuleExecutors.execute_anisotropic_expansion(backend, ex_m, dims, spacing, Dict("all" => ex_margin); side=side_eff)
                        elseif ex_margin < 0.0
                            erode_mm = abs(ex_margin)
                            # Compute complement directly on GPU (no CPU allocation)
                            complement = map(x -> x > UInt8(0) ? UInt8(0) : UInt8(1), ex_m)
                            dilated_complement = RuleExecutors.execute_anisotropic_expansion(backend, complement, dims, spacing, Dict("all" => erode_mm); side=side_eff)
                            ex_m = map((e, d) -> (e > UInt8(0) && d == UInt8(0)) ? UInt8(1) : UInt8(0), ex_m, dilated_complement)
                        end
                        mask_arr .= ifelse.(ex_m .> UInt8(0), UInt8(0), mask_arr)
                        ex_m = nothing
                    end
                    if length(composite_cache) >= 2
                        empty!(composite_cache)
                    end
                end
            end
            if DEBUG_VERBOSE; post_json_excl_count = gpu_count(mask_arr); end

            # 5.5 Fat Intersection (ignore_fat=false means INTERSECT with fat)
            if haskey(rule_def, "ignore_fat") && rule_def["ignore_fat"] == false && gpu_count(mask_arr) > 0
                fat_mask = get_mask("tissue_fat")
                if fat_mask !== nothing
                    mask_arr .= ifelse.(fat_mask .> UInt8(0), mask_arr, UInt8(0))
                    println("    [ignore_fat=false] Intersected $node_name with tissue_fat: $(gpu_count(mask_arr)) voxels remain")
                end
            end

            # 6. Largest Connected Component (matching Python engine.py:1773)
            # Python only skips LCC when keep_all_components is True in JSON.
            # Only run host LCC if dilate_val > 0.0 requires LCC prior to dilation;
            # otherwise GPU CCL runs on level_output_buf in-place at step 2125 with zero allocations.
            dilate_val = Float64(get(rule_def, "dilate", get(rule_def, "dilation", 0.0)))
            if dilate_val > 0.0 && !get(rule_def, "keep_all_components", false) && gpu_count(mask_arr) > 0
                mask_arr = RuleExecutors.get_largest_connected_component(mask_arr)
            end

            # 6.5 Optional Dilation (matching Python engine.py:1783, runs AFTER LCC)
            if dilate_val > 0.0 && gpu_count(mask_arr) > 0
                mask_arr = RuleExecutors.execute_anisotropic_expansion(backend, mask_arr, dims, spacing, Dict("all" => dilate_val); side=side_eff)
            end

            # 6. per_slice_lcc (optimized: slice-by-slice transfer instead of whole volume)
            if get(rule_def, "per_slice_lcc", false) && gpu_count(mask_arr) > 0
                z_sums = Array(dropdims(sum(mask_arr, dims=(1, 2)), dims=(1, 2)))
                for z in 1:dims[3]
                    if z_sums[z] > 0
                        slice_gpu = view(mask_arr, :, :, z)
                        slice_cpu = Array(slice_gpu)
                        labels = ImageMorphology.label_components(slice_cpu)
                        max_label = maximum(labels)
                        if max_label > 1
                            best_label = 0
                            max_size = 0
                            for l in 1:max_label
                                s = sum(labels .== l)
                                if s > max_size
                                    max_size = s
                                    best_label = l
                                end
                            end
                            for i in 1:size(slice_cpu, 1), j in 1:size(slice_cpu, 2)
                                if slice_cpu[i, j] > 0 && labels[i, j] != best_label
                                    slice_cpu[i, j] = UInt8(0)
                                end
                            end
                            copyto!(slice_gpu, slice_cpu)
                        end
                    end
                end
            end

            # Final voxel count — always computed (1 GPU sync per rule)
            final_count = gpu_count(mask_arr)
            if DEBUG_VERBOSE
                println("    [DEBUG] $node_name voxels lost: pre=$pre_count → constr=$post_constr_count → z_restr=$post_z_count → json_excl=$post_json_excl_count → lcc=$final_count → final=$final_count")
            end
            
            level_outputs[node_name] = mask_arr
            if !isempty(composite_cache)
                empty!(composite_cache)
            end
        end # closes for (node_name, mask_arr) in collect(level_outputs)
        t_postproc_end = time()

        # Copy level outputs to preallocated VM_ARENA[:level_output]
        t_copy_start = time()
        max_idx = length(level_outputs)
        output_names = Vector{String}(undef, max_idx)
        for (k, idx) in level_outputs.mapping
            output_names[idx] = k
        end
        num_out = length(output_names)
        # StaticArena.reset_level_output!(backend, num_out) # NO! We wrote to it already!
        level_output_buf = StaticArena.VM_ARENA[:level_output]
        # We don't empty!(level_outputs) yet because we might need it later, wait, actually we can empty it at the end of the level.
        t_copy_end = time()

        t_gc_start = time()
        GC.gc(false)
        t_gc_end = time()
        
        # Compile VM instruction stream for this DAG level
        t_compile_start = time()
        instructions = VMCompiler.compile_level_instructions(
            output_names, rules, packed_tensor, spacing, dims,
            computed_landmarks, origin, direction, ALIASES
        )
        t_compile_end = time()
        
        # Execute single MegaKernel launch for all rules in this level
        t_mk_start = time()
        if !isempty(instructions)
            VMKernel.execute_megakernel_level!(backend, level_output_buf, packed_tensor.data, instructions; spacing=spacing)
        end
        
        # GPU CCL for rules requiring largest connected component
        for (r_idx, out_k) in enumerate(output_names)
            base_rule_name = out_k
            if !haskey(rules, base_rule_name)
                for suffix in ["_Left", "_Right", "_left", "_right"]
                    if endswith(base_rule_name, suffix)
                        cand = base_rule_name[1:end-length(suffix)]
                        if haskey(rules, cand); base_rule_name = cand; break; end
                    end
                end
            end
            rdef = get(rules, base_rule_name, Dict())
            adjacent_to = get(rdef, "keep_component_adjacent_to", nothing)
            if adjacent_to !== nothing
                # Select largest component that overlaps with anchor mask (dilated by ~5mm)
                v_ch = view(level_output_buf, :, :, :, r_idx)
                if gpu_count(v_ch) > 0
                    # Get anchor mask from packed_tensor
                    anchor_name = adjacent_to
                    # For bilateral rules, append side suffix
                    if endswith(out_k, "_Left") && !haskey(packed_tensor.registry, anchor_name)
                        anchor_name = anchor_name * "_left"
                    elseif endswith(out_k, "_Right") && !haskey(packed_tensor.registry, anchor_name)
                        anchor_name = anchor_name * "_right"
                    end
                    anchor_mask = haskey(packed_tensor.registry, anchor_name) ? MaskPacker.unpack_mask(packed_tensor, anchor_name) : nothing
                    if anchor_mask !== nothing
                        # Dilate anchor by 5mm on CPU for adjacency tolerance
                        anchor_cpu = Array(anchor_mask)
                        dilate_vox = max(2, round(Int, 10.0 / minimum(spacing)))
                        # Simple 3D dilation on CPU: expand any nonzero voxel by dilate_vox in each direction
                        anchor_dilated_cpu = copy(anchor_cpu)
                        nz_indices = findall(x -> x > UInt8(0), anchor_cpu)
                        for idx in nz_indices
                            ci = Tuple(idx)
                            for dz in -dilate_vox:dilate_vox, dy in -dilate_vox:dilate_vox, dx in -dilate_vox:dilate_vox
                                nz_i = ci[1] + dz
                                ny_i = ci[2] + dy
                                nx_i = ci[3] + dx
                                if 1 <= nz_i <= dims[1] && 1 <= ny_i <= dims[2] && 1 <= nx_i <= dims[3]
                                    anchor_dilated_cpu[nz_i, ny_i, nx_i] = UInt8(1)
                                end
                            end
                        end
                        
                        # Run CCL steps 1-4 (label + count)
                        n_voxels = Int32(dims[1] * dims[2] * dims[3])
                        fill!(StaticArena.CCL_ARENA[:counts], Int32(0))
                        kernel_init = GpuCCL.ccl_init_kernel!(backend, 256)
                        kernel_init(StaticArena.CCL_ARENA[:labels], v_ch, n_voxels; ndrange=Int(n_voxels))
                        KernelAbstractions.synchronize(backend)
                        kernel_merge = GpuCCL.ccl_merge_kernel!(backend, (8, 4, 4))
                        kernel_merge(StaticArena.CCL_ARENA[:labels], v_ch, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]); ndrange=(Int(dims[1]), Int(dims[2]), Int(dims[3])))
                        KernelAbstractions.synchronize(backend)
                        kernel_compress = GpuCCL.ccl_compress_kernel!(backend, 256)
                        kernel_compress(StaticArena.CCL_ARENA[:labels], n_voxels; ndrange=Int(n_voxels))
                        KernelAbstractions.synchronize(backend)
                        kernel_count = GpuCCL.ccl_count_kernel!(backend, 256)
                        kernel_count(StaticArena.CCL_ARENA[:counts], StaticArena.CCL_ARENA[:labels], n_voxels; ndrange=Int(n_voxels))
                        KernelAbstractions.synchronize(backend)
                        
                        # Transfer labels to CPU for component analysis
                        labels_cpu = Array(StaticArena.CCL_ARENA[:labels])
                        counts_cpu = Array(StaticArena.CCL_ARENA[:counts])
                        
                        # Find all unique labels that overlap with dilated anchor
                        adjacent_labels = Set{Int32}()
                        anchor_flat = vec(anchor_dilated_cpu)
                        for i in 1:length(labels_cpu)
                            if labels_cpu[i] > 0 && anchor_flat[i] > 0
                                push!(adjacent_labels, labels_cpu[i])
                            end
                        end
                        
                        if !isempty(adjacent_labels)
                            # Pick largest among adjacent components
                            best_label = Int32(0)
                            best_count = Int32(0)
                            for lbl in adjacent_labels
                                if counts_cpu[lbl] > best_count
                                    best_count = counts_cpu[lbl]
                                    best_label = lbl
                                end
                            end
                            # Extract the winning component
                            kernel_extract = GpuCCL.ccl_extract_kernel!(backend, 256)
                            kernel_extract(v_ch, StaticArena.CCL_ARENA[:labels], best_label, n_voxels; ndrange=Int(n_voxels))
                            KernelAbstractions.synchronize(backend)
                            println("    [CCL-Adjacent] $out_k: selected component $best_label ($best_count voxels) adjacent to $anchor_name ($(length(adjacent_labels)) candidates)")
                        else
                            println("    [CCL-Adjacent] $out_k: no component adjacent to $anchor_name, falling back to standard LCC")
                            GpuCCL.gpu_largest_connected_component!(
                                backend, v_ch, v_ch, dims,
                                StaticArena.CCL_ARENA[:labels], StaticArena.CCL_ARENA[:counts];
                                block_val=get(StaticArena.CCL_ARENA, :block_val, nothing),
                                block_idx=get(StaticArena.CCL_ARENA, :block_idx, nothing)
                            )
                        end
                    else
                        println("    [CCL-Adjacent] $out_k: anchor '$anchor_name' not found, falling back to standard LCC")
                        GpuCCL.gpu_largest_connected_component!(
                            backend, v_ch, v_ch, dims,
                            StaticArena.CCL_ARENA[:labels], StaticArena.CCL_ARENA[:counts];
                            block_val=get(StaticArena.CCL_ARENA, :block_val, nothing),
                            block_idx=get(StaticArena.CCL_ARENA, :block_idx, nothing)
                        )
                    end
                end
            elseif !get(rdef, "keep_all_components", false)
                v_ch = view(level_output_buf, :, :, :, r_idx)
                if gpu_count(v_ch) > 0
                    GpuCCL.gpu_largest_connected_component!(
                        backend, v_ch, v_ch, dims,
                        StaticArena.CCL_ARENA[:labels], StaticArena.CCL_ARENA[:counts];
                        block_val=get(StaticArena.CCL_ARENA, :block_val, nothing),
                        block_idx=get(StaticArena.CCL_ARENA, :block_idx, nothing)
                    )
                end
            end
        end
        t_mk_end = time()
        
        # Repack level outputs directly into packed_tensor in-place with zero dynamic array allocations
        t_repack_start = time()
        MaskPacker.pack_level_outputs!(packed_tensor, level_output_buf, output_names; backend=backend)
        t_repack_end = time()
        
        t_genmasks_start = time()
        for (r_idx, k) in enumerate(output_names)
            v_view = view(level_output_buf, :, :, :, r_idx)
            c = gpu_count(v_view)
            println("    -> Generated '$k': $c voxels")
            if c > 0
                bbox = RuleExecutors.gpu_bounding_box(backend, v_view)
                if bbox !== nothing
                    packed_tensor.bboxes[k] = bbox
                end
            end
            if k == "Thoracic_Station_5_Subaortic"
                ch, id = packed_tensor.registry[k]
                packed_tensor.registry["Thoracic_Station_5_Subaortic_Left"] = (ch, id)
                if haskey(packed_tensor.bboxes, k)
                    packed_tensor.bboxes["Thoracic_Station_5_Subaortic_Left"] = packed_tensor.bboxes[k]
                end
            end
        end
        t_genmasks_end = time()
        
        empty!(level_outputs)
        empty!(composite_cache)

        t_level_end = time()

        # Record timings
        push!(timing_rule_gen, t_rule_gen_end - t_rule_gen_start)
        push!(timing_postproc, t_postproc_end - t_postproc_start)
        push!(timing_copy_arena, t_copy_end - t_copy_start)
        push!(timing_vm_compile, t_compile_end - t_compile_start)
        push!(timing_megakernel, t_mk_end - t_mk_start)
        push!(timing_repack, t_repack_end - t_repack_start)
        push!(timing_gen_masks, t_genmasks_end - t_genmasks_start)
        push!(timing_gc_reclaim, t_gc_end - t_gc_start)
        last_reclaim = time()
        push!(timing_level_total, t_level_end - t_level_start)

        println("  [TIMING Level $lvl_idx] rule_gen=$(round(t_rule_gen_end - t_rule_gen_start, digits=2))s  postproc=$(round(t_postproc_end - t_postproc_start, digits=2))s  copy=$(round(t_copy_end - t_copy_start, digits=2))s  gc=$(round(t_gc_end - t_gc_start, digits=2))s  compile=$(round(t_compile_end - t_compile_start, digits=2))s  megakernel=$(round(t_mk_end - t_mk_start, digits=2))s  repack=$(round(t_repack_end - t_repack_start, digits=2))s  gen_masks=$(round(t_genmasks_end - t_genmasks_start, digits=2))s  TOTAL=$(round(t_level_end - t_level_start, digits=2))s")

        println("  [PackedTensor Status] Total packed channels: $(packed_tensor.num_channels_used)")
        flush(stdout)
    end

    # Print overall timing summary
    println("\n================================================================================")
    println("  DETAILED TIMING SUMMARY (all DAG levels)")
    println("================================================================================")
    println("  Phase              Total (s)   Avg/Level (s)   % of Processing")
    println("  ─────────────────  ──────────  ─────────────   ───────────────")
    total_all = sum(timing_level_total)
    for (name, arr) in [
        ("Rule Generation", timing_rule_gen),
        ("Post-Processing", timing_postproc),
        ("Copy to Arena", timing_copy_arena),
        ("GC/Reclaim", timing_gc_reclaim),
        ("VM Compile", timing_vm_compile),
        ("MegaKernel Exec", timing_megakernel),
        ("Repack Tensor", timing_repack),
        ("Gen Masks (D→H)", timing_gen_masks),
    ]
        t = sum(arr)
        avg = t / length(arr)
        pct = total_all > 0 ? 100.0 * t / total_all : 0.0
        println("  $(rpad(name, 20)) $(lpad(round(t, digits=2), 8))   $(lpad(round(avg, digits=3), 12))   $(lpad(round(pct, digits=1), 6))%")
    end
    println("  $(rpad("TOTAL", 20)) $(lpad(round(total_all, digits=2), 8))")
    println("================================================================================")

    # 5. Post-Processing: Bilateral Splits
    t_postdag_start = time()
    println("[Post-Processing] Applying Inguinal Anterior Refinement (Ligament Proxy) via GPU Kernel...")
    for side in ["left", "right"]
        hip_m = get_mask("hip_$(side)")
        if hip_m === nothing
            hip_m = get_mask("hip_$(side[1:1])")
        end
        if hip_m !== nothing
            hip_2d = adapt(Array, dropdims(any(hip_m .> 0, dims=3), dims=3))
            if any(hip_2d)
                x_indices = findall(any(hip_2d, dims=2)[:])
                if !isempty(x_indices)
                    x_min, x_max = minimum(x_indices), maximum(x_indices)
                    x_mid = (x_min + x_max) / 2.0
                    
                    side_cap = titlecase(side)
                    
                    med_mask = zeros(Bool, size(hip_2d))
                    lat_mask = zeros(Bool, size(hip_2d))
                    
                    if side == "left"
                        med_mask[1:floor(Int, x_mid), :] .= hip_2d[1:floor(Int, x_mid), :]
                        lat_mask[floor(Int, x_mid)+1:end, :] .= hip_2d[floor(Int, x_mid)+1:end, :]
                    else
                        med_mask[floor(Int, x_mid)+1:end, :] .= hip_2d[floor(Int, x_mid)+1:end, :]
                        lat_mask[1:floor(Int, x_mid), :] .= hip_2d[1:floor(Int, x_mid), :]
                    end
                    
                    ys_m, xs_m = [], []
                    for ci in findall(med_mask)
                        push!(ys_m, ci[2])
                        push!(xs_m, ci[1])
                    end
                    ys_l, xs_l = [], []
                    for ci in findall(lat_mask)
                        push!(ys_l, ci[2])
                        push!(xs_l, ci[1])
                    end
                    
                    if !isempty(ys_m) && !isempty(ys_l)
                        med_y_idx = argmin(ys_m)
                        med_y = ys_m[med_y_idx]
                        med_x = xs_m[med_y_idx]
                        
                        lat_y_idx = argmin(ys_l)
                        lat_y = ys_l[lat_y_idx]
                        lat_x = xs_l[lat_y_idx]
                        
                        dy_pixels = 5.0 / spacing[2]
                        p1_x, p1_y = med_x, med_y + dy_pixels
                        p2_x, p2_y = lat_x, lat_y + dy_pixels
                        
                        if p1_x != p2_x
                            for target_name in ["Superficial_Inguinal", "Deep_Inguinal"]
                                full_name = "$(target_name)_$(side_cap)"
                                ch_info = get(packed_tensor.registry, full_name, nothing)
                                if ch_info !== nothing
                                    ch, id = ch_info
                                    kernel! = VMKernel.refine_inguinal_kernel!(backend)
                                    kernel!(packed_tensor.data, Int32(ch), UInt8(id), Float32(p1_x), Float32(p1_y), Float32(p2_x), Float32(p2_y), Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=dims)
                                    KernelAbstractions.synchronize(backend)
                                    println("    -> Refined $(full_name) using Inguinal Anterior Refinement (Ligament Proxy)")
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    println("\n[Post-Processing] Applying Bilateral Landmark Splits via GPU Kernels...")
    trachea_m = get_mask("trachea")
    sternum_m = get_mask("sternum")
    esophagus_m = get_mask("esophagus")
    sacrum_m = get_mask("sacrum")
    if sacrum_m === nothing; sacrum_m = get_mask("vertebrae_L5"); end
    
    mid_x_default = dims[1] ÷ 2
    
    # We iterate over a copy of registry keys because we will delete and add keys
    registered_masks = collect(keys(rules))
    for node_name in registered_masks
        rule_def = get(rules, node_name, Dict())
        is_bilateral = get(rule_def, "is_bilateral", false)
        is_force_bilat = any(k -> occursin(k, node_name), ["Retrotracheal", "Mammary", "Paraoesophageal", "Prepericardial", "Nuchal", "Common_Iliac", "Thoracic_Station_3A", "Thoracic_Station_3P", "Thoracic_Station_4", "Thoracic_Station_2", "Thoracic_Station_1", "Thoracic_Station_8", "Obturator", "Level_Va", "Level_Vb", "Level_V_Helper", "Level_Ib", "Level_II", "Level_III", "Level_IV", "Internal_Iliac", "External_Iliac", "Iliac_Bifurcation"])
        
        if (is_bilateral || is_force_bilat) && !endswith(node_name, "_left") && !endswith(node_name, "_right") &&
           !endswith(node_name, "_Left") && !endswith(node_name, "_Right") &&
           !occursin("Helper", node_name) && !occursin("helper", node_name)
            
            if haskey(packed_tensor.registry, "$(node_name)_Left") || haskey(packed_tensor.registry, "$(node_name)_left") ||
               haskey(packed_tensor.registry, "$(node_name)_Right") || haskey(packed_tensor.registry, "$(node_name)_right")
                delete!(packed_tensor.registry, node_name)
            end
        end
        if haskey(packed_tensor.registry, "$(node_name)_left") && haskey(packed_tensor.registry, "$(node_name)_Left")
            delete!(packed_tensor.registry, "$(node_name)_Left")
        end
        if haskey(packed_tensor.registry, "$(node_name)_right") && haskey(packed_tensor.registry, "$(node_name)_Right")
            delete!(packed_tensor.registry, "$(node_name)_Right")
        end
        if (is_bilateral || is_force_bilat) && !endswith(node_name, "_left") && !endswith(node_name, "_right") &&
           !endswith(node_name, "_Left") && !endswith(node_name, "_Right") &&
           !occursin("Helper", node_name) && !occursin("helper", node_name)
            
            if haskey(packed_tensor.registry, "$(node_name)_Left") || haskey(packed_tensor.registry, "$(node_name)_left") ||
               haskey(packed_tensor.registry, "$(node_name)_Right") || haskey(packed_tensor.registry, "$(node_name)_right")
                empty!(composite_cache)
                continue
            end
            
            slice_split_x = fill(mid_x_default, dims[3])
            split_lm_m = nothing
            split_lm_name = get(rule_def, "split_landmark", "")
            if !isempty(split_lm_name)
                split_lm_m = resolve_mask_from_spec(split_lm_name, get_mask, dims, backend)
            end
            
            if split_lm_m === nothing
                if any(k -> occursin(k, node_name), ["Retrotracheal", "LowCervical", "UpperParatracheal", "LowerParatracheal", "Thoracic_Station_3A", "Thoracic_Station_3P", "Thoracic_Station_1", "Thoracic_Station_2", "Thoracic_Station_4"])
                    split_lm_m = trachea_m
                elseif occursin("Paraoesophageal", node_name) || occursin("Thoracic_Station_8", node_name)
                    split_lm_m = esophagus_m
                elseif occursin("Mammary", node_name) || occursin("Prepericardial", node_name)
                    split_lm_m = sternum_m
                elseif occursin("Internal_Iliac", node_name) || occursin("Common_Iliac", node_name) || occursin("Presacral", node_name)
                    split_lm_m = sacrum_m
                elseif occursin("Nuchal", node_name)
                    fused_spine_m = get_mask("fused_spine")
                    if fused_spine_m === nothing; fused_spine_m = get_mask("vertebrae_C1"); end
                    split_lm_m = fused_spine_m
                end
            end
            
            if split_lm_m !== nothing
                min_x_cpu, max_x_cpu = RuleExecutors.get_slice_bounds_gpu(backend, split_lm_m, dims)
                all_xs = Int[]
                for k in 1:dims[3]
                    if max_x_cpu[k] > 0
                        push!(all_xs, div(min_x_cpu[k] + max_x_cpu[k], 2))
                    end
                end
                def_x = !isempty(all_xs) ? round(Int, sum(all_xs) / length(all_xs)) : mid_x_default
                for k in 1:dims[3]
                    if max_x_cpu[k] > 0
                        slice_split_x[k] = div(min_x_cpu[k] + max_x_cpu[k], 2)
                    else
                        slice_split_x[k] = def_x
                    end
                end
            end
            
            ch_info = get(packed_tensor.registry, node_name, nothing)
            if ch_info !== nothing
                ch, id = ch_info
                slice_split_x_gpu = adapt(backend, slice_split_x)
                
                # Clear level output buf
                level_output_buf = StaticArena.VM_ARENA[:level_output]
                fill!(level_output_buf, UInt8(0))
                out_left = view(level_output_buf, :, :, :, 1)
                out_right = view(level_output_buf, :, :, :, 2)
                
                kernel! = VMKernel.split_packed_mask_kernel!(backend)
                kernel!(packed_tensor.data, out_left, out_right, slice_split_x_gpu, Int32(ch), UInt8(id), Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=dims)
                KernelAbstractions.synchronize(backend)
                
                MaskPacker.pack_level_outputs!(packed_tensor, level_output_buf, ["$(node_name)_Left", "$(node_name)_Right"]; backend=backend)
                delete!(packed_tensor.registry, node_name)
                println("    -> Bilateral split for $node_name via GPU")
            end
        end
    end

    # Split Neck Level II into IIa and IIb based on IJV posterior wall
    ijv_m = get_mask("internal_jugular_vein")
    registered_masks = collect(keys(rules))
    for node_name in registered_masks
        if occursin("Neck_Level_II_Upper_Jugular", node_name) && !occursin("IIa", node_name) && !occursin("IIb", node_name)
            is_left = occursin("Left", node_name) || occursin("left", node_name)
            ijv_side = get_mask(is_left ? "internal_jugular_vein_left" : "internal_jugular_vein_right")
            if ijv_side === nothing; ijv_side = ijv_m; end
            if ijv_side !== nothing
                ch_info = get(packed_tensor.registry, node_name, nothing)
                if ch_info !== nothing
                    ch, id = ch_info
                    
                    _, ijv_post_y_gpu = RuleExecutors.get_slice_y_bounds_gpu(backend, ijv_side, dims; return_gpu=true)
                    
                    level_output_buf = StaticArena.VM_ARENA[:level_output]
                fill!(level_output_buf, UInt8(0))
                    out_2a = view(level_output_buf, :, :, :, 1)
                    out_2b = view(level_output_buf, :, :, :, 2)
                    
                    kernel! = VMKernel.split_neck_level_2_kernel!(backend)
                    kernel!(packed_tensor.data, out_2a, out_2b, ijv_post_y_gpu, Int32(ch), UInt8(id), Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), ndrange=dims)
                    KernelAbstractions.synchronize(backend)
                    
                    side_str = is_left ? "Left" : "Right"
                    side_str_l = is_left ? "left" : "right"
                    
                    a_name = "Neck_Level_IIa_Upper_Jugular_$side_str"
                    b_name = "Neck_Level_IIb_Upper_Jugular_$side_str"
                    
                    MaskPacker.pack_level_outputs!(packed_tensor, level_output_buf, [a_name, b_name]; backend=backend)
                    delete!(packed_tensor.registry, node_name)
                    println("    -> Split Neck Level II ($node_name) into IIa and IIb via GPU")
                end
            end
        end
    end


    # Split Axillary II based on Pectoralis Minor and reassign medial portion to Axillary III
    for side_str in ["Left", "Right"]
        ax2_name = "Axillary_Level_II_$side_str"
        ax3_name = "Axillary_Level_III_$side_str"
        pec_name = "pectoralis_minor_$(lowercase(side_str))"
        
        if haskey(packed_tensor.registry, ax2_name) && haskey(packed_tensor.registry, ax3_name)
            pec_m = get_mask(pec_name)
            if pec_m === nothing; pec_m = get_mask("pectoralis_minor"); end
            if pec_m !== nothing
                is_left = side_str == "Left"
                
                # Get medial edge of pectoralis minor
                min_x_gpu, max_x_gpu = RuleExecutors.get_slice_bounds_gpu(backend, pec_m, dims; return_gpu=true)
                pec_x_gpu = is_left ? min_x_gpu : max_x_gpu # Left: +X is lateral, min_x is medial. Right: -X is lateral, max_x is medial.
                
                # Unpack 2 and 3
                in2 = MaskPacker.unpack_mask(packed_tensor, ax2_name)
                in3 = MaskPacker.unpack_mask(packed_tensor, ax3_name)
                
                level_output_buf = StaticArena.VM_ARENA[:level_output]
                fill!(level_output_buf, UInt8(0))
                out2 = view(level_output_buf, :, :, :, 1)
                out3 = view(level_output_buf, :, :, :, 2)
                
                kernel! = VMKernel.axillary_2_3_reassign_kernel!(backend)
                kernel!(out2, out3, in2, in3, pec_x_gpu, is_left, Int32(dims[3]), ndrange=dims)
                KernelAbstractions.synchronize(backend)
                
                # Delete old
                delete!(packed_tensor.registry, ax2_name)
                delete!(packed_tensor.registry, ax3_name)
                
                # Repack
                MaskPacker.pack_level_outputs!(packed_tensor, level_output_buf, [ax2_name, ax3_name]; backend=backend)
                println("    -> Reassigned medial portion of $ax2_name to $ax3_name based on Pectoralis Minor")
            end
        end
    end

    t_postdag_end = time()

    println("\n================================================================================")
    println("  DAG VM Pipeline Execution Finished Successfully! ($(length(packed_tensor.registry)) total masks in PackedTensor)")
    println("  Pre-level setup: landmarks=$(round(t_landmarks_end - t_landmarks_start, digits=2))s  arena=$(round(t_arena_end - t_arena_start, digits=2))s  packing=$(round(t_packing_end - t_packing_start, digits=2))s")
    println("  DAG Levels total: $(round(total_all, digits=2))s")
    println("  Post-DAG (splits): $(round(t_postdag_end - t_postdag_start, digits=2))s")
    println("================================================================================")

    # Finally, resolve overlaps using the GPU PackedTensor directly
    println("\n[4/4] Resolving Overlaps to match Gold Standard Labelmap via GPU...")
    # overlap_resolution.jl defines resolve_overlaps_gpu!
    t_overlap_start = time()
    final_masks_cpu = resolve_overlaps_gpu!(packed_tensor, rules; spacing=spacing, anatomy_masks=raw_masks, backend=backend)
    t_overlap_end = time()
    println("  Overlap resolution completed in $(round(t_overlap_end - t_overlap_start, digits=2))s")
    
    # Post-overlap LCC for tagged masks
    println("\n[5/5] Post-overlap LCC for tagged masks...")
    n_lcc = 0
    for (name, mask_cpu) in final_masks_cpu
        rname = replace(name, r"(_Left|_Right|_left|_right)$" => "")
        rdef = get(rules, rname, Dict())
        if !get(rdef, "post_overlap_lcc", false)
            rdef = get(rules, name, rdef)
        end
        if get(rdef, "post_overlap_lcc", false) && sum(mask_cpu) > 0
            # 3D flood-fill LCC on CPU
            dims_m = size(mask_cpu)
            labeled = zeros(Int32, dims_m)
            label_count = 0
            label_sizes = Int[]
            for k in 1:dims_m[3], j in 1:dims_m[2], i in 1:dims_m[1]
                if mask_cpu[i,j,k] > UInt8(0) && labeled[i,j,k] == 0
                    label_count += 1
                    push!(label_sizes, 0)
                    queue = Tuple{Int,Int,Int}[(i,j,k)]
                    labeled[i,j,k] = label_count
                    while !isempty(queue)
                        ci, cj, ck = popfirst!(queue)
                        label_sizes[label_count] += 1
                        for (di,dj,dk) in ((-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1))
                            ni, nj, nk = ci+di, cj+dj, ck+dk
                            if 1<=ni<=dims_m[1] && 1<=nj<=dims_m[2] && 1<=nk<=dims_m[3] &&
                               mask_cpu[ni,nj,nk] > UInt8(0) && labeled[ni,nj,nk] == 0
                                labeled[ni,nj,nk] = label_count
                                push!(queue, (ni,nj,nk))
                            end
                        end
                    end
                end
            end
            if label_count > 1
                best_label = argmax(label_sizes)
                for k in 1:dims_m[3], j in 1:dims_m[2], i in 1:dims_m[1]
                    mask_cpu[i,j,k] = labeled[i,j,k] == best_label ? UInt8(1) : UInt8(0)
                end
                old_sum = sum(mask_cpu .> UInt8(0)) + sum(label_sizes) - label_sizes[best_label]
                println("    Post-overlap LCC: $name — kept 1 of $label_count components (component #$best_label, $(label_sizes[best_label]) of $(sum(label_sizes)) voxels)")
                n_lcc += 1
            end
        end
    end
    println("  Applied post-overlap LCC to $n_lcc masks")
    
    return final_masks_cpu

end

end # module