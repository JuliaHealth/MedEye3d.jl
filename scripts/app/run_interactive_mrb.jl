using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

# HDF5 thread-safe configuration (must be set before first HDF5 open)
ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Configure Vulkan ICD if NVIDIA driver is present
if isfile("/etc/vulkan/icd.d/nvidia_icd.json") && !haskey(ENV, "VK_ICD_FILENAMES")
    ENV["VK_ICD_FILENAMES"] = "/etc/vulkan/icd.d/nvidia_icd.json"
    ENV["VK_DRIVER_FILES"] = "/etc/vulkan/icd.d/nvidia_icd.json"
end

using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using MedImages
using ColorTypes
using MedEye3d.StructsManag
using MedEye3d.SegmentationDisplay
using MedEye3d.MakieEvents
using MedEye3d.LesionMetadataWindow
using MedEye3d.LesionAssociation
using Statistics
using LinearAlgebra
import GLFW
import JSON
import HDF5

const MEH = MedEye3d.SegmentationDisplay.MakieEventHandlers

# Paths to data
data_dir_pat6 = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "..", "data", "pat_6_files")

t_startup = time_ns()
println("Starting MedEye3d (HDF5-only mode)...")

# Load shared SceneHierarchy module
include(joinpath(@__DIR__, "..", "lib", "SceneHierarchy.jl"))
using .SceneHierarchy

studies = parse_studies_from_hierarchy(data_dir_pat6)

preprocessed_h5 = joinpath(data_dir_pat6, "preprocessed_volumes.h5")
if !isfile(preprocessed_h5)
    error("preprocessed_volumes.h5 not found in $data_dir_pat6. Run: julia scripts/preprocessing/preprocess_dataset.jl $data_dir_pat6")
end

# =======================================================================
# Phase 1: Read spatial metadata from HDF5 (replaces baseline_ct NIfTI)
# =======================================================================

h5_init = HDF5.h5open(preprocessed_h5, "r")
base_ct_fname = studies[1][4]
base_mask_fname = studies[1][6]
first_spacing = Tuple(Float64.(read(HDF5.attributes(h5_init["BASELINE/$base_ct_fname"])["spacing"])))
display_spacing = first_spacing
println("Spacing from HDF5: $first_spacing")

is_preflipped = haskey(h5_init, "_meta_/preflipped") && read(h5_init["_meta_/preflipped"]) == 1
raw_first_mask = read(h5_init["BASELINE/$base_mask_fname"])
first_mask = is_preflipped ? Float32.(raw_first_mask) : reverse(Float32.(raw_first_mask), dims=2)

# =======================================================================
# Phase 2: Read atlas, labels, organ mapping, centroids from HDF5
# =======================================================================

ts_atlas_aligned = nothing
ts_names = Dict{Int,String}()
bone_atlas = nothing
skelly_atlas = nothing
organ_mapping = Dict{Int, String}()

if haskey(h5_init, "ATLAS/max_anatomy")
    ts_atlas_aligned = read(h5_init["ATLAS/max_anatomy"])
    println("  Loaded ATLAS/max_anatomy from HDF5 ($(size(ts_atlas_aligned)))")
end

if haskey(h5_init, "_meta_/max_anatomy_labels.json")
    raw_labels = JSON.parse(read(h5_init["_meta_/max_anatomy_labels.json"]))
    ts_names = Dict{Int,String}(parse(Int, k) => v for (k, v) in raw_labels)
    println("  Loaded $(length(ts_names)) anatomy labels from HDF5")
end

if haskey(h5_init, "ATLAS/skellytour")
    skelly_atlas = read(h5_init["ATLAS/skellytour"])
    println("  Loaded ATLAS/skellytour from HDF5 ($(size(skelly_atlas)))")
end

if haskey(h5_init, "ATLAS/bone_atlas")
    bone_atlas = read(h5_init["ATLAS/bone_atlas"])
    println("  Loaded ATLAS/bone_atlas from HDF5 ($(count(bone_atlas .> 0)) bone voxels)")
end

if haskey(h5_init, "_meta_/organ_mapping")
    raw_organ = JSON.parse(read(h5_init["_meta_/organ_mapping"]))
    organ_mapping = Dict{Int,String}(parse(Int, k) => v for (k, v) in raw_organ)
    println("  Loaded organ_mapping from HDF5 ($(length(organ_mapping)) lesions)")
end

# Per-TP anatomy labels
for (s_idx, study) in enumerate(studies)
    tp_i = s_idx - 1
    lbl_key = "_meta_/anatomy_labels_tp_$(tp_i).json"
    if haskey(h5_init, lbl_key)
        tp_raw = JSON.parse(read(h5_init[lbl_key]))
        merged = copy(ts_names)
        for (k_str, v) in tp_raw
            k_int = parse(Int, k_str)
            if !occursin("class_", v)
                merged[k_int] = v
            end
        end
        MEH.anatomy_labels_cache[tp_i] = merged
    else
        MEH.anatomy_labels_cache[tp_i] = ts_names
    end
end

# Centroids
centroid_count = Ref(0)
if haskey(h5_init, "CENTROIDS")
    for key in keys(h5_init["CENTROIDS"])
        parts = split(key, "_lid")
        if length(parts) == 2
            tp_str = replace(parts[1], "tp" => "")
            tp_idx = parse(Int, tp_str)
            lid = parse(Int, parts[2])
            coords = read(h5_init["CENTROIDS/$key"])
            c = [Int(coords[1]), Int(coords[2]), Int(coords[3])]
            MEH.lesion_centroids_cache[(tp_idx, lid)] = c
            node_name = get(Dict(s_idx - 1 => study[7] for (s_idx, study) in enumerate(studies)), tp_idx, "")
            if !isempty(node_name)
                MEH.lesion_centroids_cache[(node_name, lid)] = c
            end
            if tp_idx == 0
                MEH.lesion_centroids_cache[lid] = c
            end
            centroid_count[] += 1
        end
    end
    println("  Loaded $(centroid_count[]) centroids from HDF5")
end

# Bone subsegments
bone_subseg_count = 0
node_to_tp = Dict{String, Int}(study[7] => s_idx - 1 for (s_idx, study) in enumerate(studies))

function _load_bone_subseg_from_group!(h5_group)
    count = 0
    vol_size = ts_atlas_aligned !== nothing ? size(ts_atlas_aligned) : (512, 512, 326)
    cis = CartesianIndices(vol_size)
    
    for obj in keys(h5_group)
        if endswith(obj, "_surf")
            marr_key = replace(obj, "_surf" => "_marr")
            if haskey(h5_group, marr_key)
                try
                    surf_data = read(h5_group[obj])
                    marr_data = read(h5_group[marr_key])
                    surf_pts = ndims(surf_data) == 1 ? cis[surf_data] : findall(surf_data .> 0)
                    marr_pts = ndims(marr_data) == 1 ? cis[marr_data] : findall(marr_data .> 0)
                    
                    obj_base = replace(obj, "_surf" => "")
                    parts = split(obj_base, "_lesion_")
                    if length(parts) == 2
                        prefix = String(parts[1])
                        lid = parse(Int, parts[2])
                        
                        if haskey(node_to_tp, prefix)
                            tp_i = node_to_tp[prefix]
                            MEH.bone_subsegments_cache[(tp_i, lid)] = (surf_pts, marr_pts)
                            MEH.bone_subsegments_cache[(prefix, lid)] = (surf_pts, marr_pts)
                        elseif startswith(prefix, "tp_")
                            tp_num = tryparse(Int, replace(prefix, "tp_" => ""))
                            if tp_num !== nothing
                                MEH.bone_subsegments_cache[(tp_num, lid)] = (surf_pts, marr_pts)
                                MEH.bone_subsegments_cache[(prefix, lid)] = (surf_pts, marr_pts)
                            end
                        else
                            MEH.bone_subsegments_cache[(prefix, lid)] = (surf_pts, marr_pts)
                        end
                    elseif startswith(obj_base, "lesion_")
                        lid = parse(Int, replace(obj_base, "lesion_" => ""))
                        MEH.bone_subsegments_cache[(0, lid)] = (surf_pts, marr_pts)
                    end
                    count += 1
                catch err
                    @warn "Failed to parse bone subseg $obj: $err"
                end
            end
        end
    end
    return count
end

if haskey(h5_init, "BONE_SUBSEG")
    global bone_subseg_count = _load_bone_subseg_from_group!(h5_init["BONE_SUBSEG"])
    println("  Loaded $bone_subseg_count bone subsegment pairs from HDF5 BONE_SUBSEG/")
end
close(h5_init)

# Fallback: try legacy Bone_Subsegments_0.h5
if bone_subseg_count == 0
    legacy_bone_h5 = joinpath(data_dir_pat6, "Bone_Subsegments_0.h5")
    if isfile(legacy_bone_h5)
        println("  Falling back to legacy Bone_Subsegments_0.h5...")
        HDF5.h5open(legacy_bone_h5, "r") do h5
            global bone_subseg_count = _load_bone_subseg_from_group!(h5)
        end
        println("  Loaded $bone_subseg_count bone subsegment pairs from legacy file")
    end
end

tp_nodes_map = Dict{Int, String}()
for (s_idx, study) in enumerate(studies)
    tp_i = s_idx - 1
    tp_nodes_map[tp_i] = study[7]
    tp_entries = count(k -> k isa Tuple{Int, Int} && k[1] == tp_i, keys(MEH.bone_subsegments_cache))
    if tp_entries > 0
        println("    Bone cache: tp_$(tp_i) ($(study[7])) = $(tp_entries) lesion pairs")
    end
end

# =======================================================================
# Phase 3: Setup textures and display configuration
# =======================================================================

colors_mapped = map(c -> RGB(c[1]/255, c[2]/255, c[3]/255), MedEye3d.distinctColorsSaved.listOfColors)

display_cfg = LesionMetadataWindow.load_display_config()
init_pet_blend = Float32(get(display_cfg, "pet_ct_blend", 0.5))
init_label_opacity = Float32(get(display_cfg, "label_opacity", 0.5))

textureSpec_ct = TextureSpec{Float32}(name="CT", isMainImage=true, color=RGB(1.0, 1.0, 1.0), minAndMaxValue=Float32.([-150, 250]))
textureSpec_pet = TextureSpec{Float32}(name="PET", isMainImage=false, isNuclearMask=true, color=RGB(1.0, 0.5, 0.0), minAndMaxValue=Float32.([0, 10]), maskContribution=init_pet_blend)
textureSpec_mask = TextureSpec{Int16}(
    name="Mask", isMainImage=false, isMultiDiscreteMask=true, isIntegerTexture=true,
    colorSet=colors_mapped, minAndMaxValue=Int16.([0, length(colors_mapped)]),
    isEditable=true, maskContribution=init_label_opacity
)
textureSpec_pure_pet = TextureSpec{Float32}(name="PET", isMainImage=true, color=RGB(1.0, 0.5, 0.0), minAndMaxValue=Float32.([0, 10]))
# Combined bone overlay: surface=1 (cyan), marrow=2 (yellow), both=3 (green)
textureSpec_bone = TextureSpec{Int8}(
    name="Bone_Overlay", isMainImage=false, isIntegerTexture=true,
    color=RGB(0.0, 1.0, 1.0), minAndMaxValue=Int8.([0, 3]),
    isVisible=true, maskContribution=init_label_opacity
)
anatomy_colors = [RGB(rand(), rand(), rand()) for _ in 1:400]
textureSpec_anatomy = TextureSpec{Int16}(
    name="Anatomy", isMainImage=false, isMultiDiscreteMask=true, isIntegerTexture=true,
    colorSet=anatomy_colors, minAndMaxValue=Int16.([0, 400]),
    isVisible=false, maskContribution=init_label_opacity
)

textureSpecArray = Vector{Vector{TextureSpec}}([
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask), deepcopy(textureSpec_bone), deepcopy(textureSpec_anatomy)],
    TextureSpec[deepcopy(textureSpec_pure_pet)],
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask), deepcopy(textureSpec_bone), deepcopy(textureSpec_anatomy)],
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask), deepcopy(textureSpec_bone), deepcopy(textureSpec_anatomy)],
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask), deepcopy(textureSpec_bone), deepcopy(textureSpec_anatomy)]
])

const HIRES_FACTOR = 1.0

# =======================================================================
# Phase 4: Define TP loader + load TP 0 and TP 1 IN PARALLEL
# =======================================================================

tp_labels_map = Dict{Int, String}()
for (s_idx, study) in enumerate(studies)
    tp_i = s_idx - 1
    modality = study[1]
    orig_tp = study[2]
    date_str = study[3]
    node_name = study[7]
    lbl = "$modality $date_str (TP $orig_tp)"
    tp_labels_map[tp_i] = lbl
    tp_nodes_map[tp_i] = node_name
    MEH.tp_labels[tp_i] = lbl
    MEH.tp_modalities[tp_i] = modality
    MEH.tp_node_names[tp_i] = node_name
end

function load_single_tp_from_h5(tp_i::Int)
    t_total = time_ns()
    if tp_i < 0 || tp_i >= length(studies)
        return nothing
    end
    study = studies[tp_i + 1]
    modality, orig_tp, date_str, ct_fname, pet_fname, mask_fname, node_name, tfm_fname = study[1:8]
    group = tfm_fname == "" ? "BASELINE" : "TFM_" * tfm_fname
    
    # Open independent file handle (thread-safe with HDF5_USE_FILE_LOCKING=FALSE)
    HDF5.h5open(preprocessed_h5, "r") do h5_file
        is_pf = haskey(h5_file, "_meta_/preflipped") && read(h5_file["_meta_/preflipped"]) == 1
        
        t_read = @elapsed begin
            ct_vol = Float32.(read(h5_file["$group/$ct_fname"]))
            pet_vol = Float32.(read(h5_file["$group/$pet_fname"]))
            mask_vol = read(h5_file["$group/$mask_fname"])
        end
        println("    [BENCH-H5] read from $group: $(round(t_read, digits=3))s ($(size(ct_vol)), mask=$(eltype(mask_vol)))"); flush(stdout)
        
        if modality == "SPECT"
            pos_dat = pet_vol[pet_vol .> 0]
            p99 = isempty(pos_dat) ? 1.0f0 : Float32(quantile(pos_dat, 0.99))
            scale_factor = 8.0f0 / max(p99, 1.0f0)
            pet_vol = max.(0.0f0, pet_vol .* scale_factor)
        end
        
        needs_reverse = !is_pf
        if needs_reverse
            ct_vol_base = reverse(ct_vol, dims=2)
            pet_vol_base = reverse(pet_vol, dims=2)
            mask_vol_base = reverse(mask_vol, dims=2)
        else
            ct_vol_base = ct_vol; pet_vol_base = pet_vol; mask_vol_base = mask_vol
        end
        
        # Thread-safe write to shared Dict (parallel startup can race)
        lock(MEH._centroids_lock) do
            MEH.pet_volumes_cache[tp_i] = pet_vol_base
        end
        
        # Always use Int16 for mask — matches preprocessing (Int16 in HDF5) and TextureSpec{Int16}
        if eltype(mask_vol_base) == Int16
            mask_compact = mask_vol_base
        elseif eltype(mask_vol_base) <: Integer
            mask_compact = Int16.(mask_vol_base)
        else
            mask_vol_base = max.(0.0f0, mask_vol_base)
            mask_compact = Int16.(round.(mask_vol_base))
        end
        
        sz = size(ct_vol_base)
        bone_mask = zeros(Int8, sz)  # Combined bone overlay: surface=1, marrow=2, both=3
        
        anatomy_vol = nothing
        try
            for k in keys(h5_file[group])
                if k == "max_anatomy.nii.gz" || startswith(k, "max_anatomy")
                    raw_anat = read(h5_file["$group/$k"])
                    if needs_reverse
                        raw_anat = reverse(Float32.(raw_anat), dims=2)
                    end
                    anatomy_vol = eltype(raw_anat) <: Integer ? UInt16.(raw_anat) : UInt16.(round.(max.(0.0f0, Float32.(raw_anat))))
                    break
                end
            end
        catch; end
        
        # Integer textures: Int16 for mask/anatomy (no Float32 conversion needed)
        mask_i16 = Int16.(mask_compact)
        anat_i16 = anatomy_vol !== nothing ? Int16.(anatomy_vol) : nothing
        MEH.precompute_mask_centroids!(mask_compact, tp_i, node_name)
        try; MedEye3d.LesionMetadataWindow.precompute_all_volumes!(mask_compact, tp_i); catch e; @warn "Volume precompute skipped: $e"; end
        
        t_total_ms = (time_ns() - t_total) / 1e6
        println("    [BENCH-H5] LOAD TP $tp_i TOTAL: $(round(t_total_ms, digits=1))ms"); flush(stdout)
        return MEH.TpCacheEntry(ct_vol_base, pet_vol_base, mask_compact, bone_mask, anatomy_vol, mask_i16, anat_i16)
    end
end

MEH.DEBUG_VERBOSE[] = true
MEH.register_tp_loader!(load_single_tp_from_h5)

# --- Load TP 0 and TP 1 in PARALLEL ---
println("Loading Time Points 0 and 1 in parallel...")
t_parallel = @elapsed begin
    tp0_task = Threads.@spawn load_single_tp_from_h5(0)
    tp1_task = length(studies) > 1 ? Threads.@spawn(load_single_tp_from_h5(1)) : nothing
    
    first_entry = fetch(tp0_task)
    MEH.tp_data_cache[0] = first_entry
    
    if tp1_task !== nothing
        tp1_entry = fetch(tp1_task)
        if tp1_entry !== nothing
            MEH.tp_data_cache[1] = tp1_entry
        end
    end
end
println("  TP 0+1 loaded in $(round(t_parallel, digits=1))s (parallel)")

# Convert TpCacheEntry to vdt format for initial displayImage call
function entry_to_vdt(e::MEH.TpCacheEntry)
    mask_i16 = e.mask_i16
    bone_i8 = e.bone_mask  # Combined Int8 overlay (surface=1, marrow=2)
    anat_i16 = if e.anat_i16 !== nothing
        e.anat_i16
    elseif ts_atlas_aligned !== nothing
        Int16.(ts_atlas_aligned)
    else
        zeros(Int16, size(e.ct))
    end
    
    sz = size(e.ct)
    Vector{Vector{Any}}([
        Any[("CT", e.ct), ("PET", e.pet), ("Mask", mask_i16), ("Bone_Overlay", bone_i8), ("Anatomy", anat_i16)],
        Any[("PET", e.pet)],
        Any[("CT", PermutedDimsArray(e.ct, (2,3,1))), ("PET", PermutedDimsArray(e.pet, (2,3,1))), ("Mask", PermutedDimsArray(mask_i16, (2,3,1))), ("Bone_Overlay", zeros(Int8, sz[2], sz[3], sz[1])), ("Anatomy", PermutedDimsArray(anat_i16, (2,3,1)))],
        Any[("CT", PermutedDimsArray(e.ct, (1,3,2))), ("PET", PermutedDimsArray(e.pet, (1,3,2))), ("Mask", PermutedDimsArray(mask_i16, (1,3,2))), ("Bone_Overlay", zeros(Int8, sz[1], sz[3], sz[2])), ("Anatomy", PermutedDimsArray(anat_i16, (1,3,2)))],
        Any[("CT", e.ct), ("PET", e.pet), ("Mask", mask_i16), ("Bone_Overlay", zeros(Int8, sz)), ("Anatomy", anat_i16)]
    ])
end
first_voxelDataTupleVector = entry_to_vdt(first_entry)

ds = display_spacing
spacings = [[ds], [ds], [(ds[2], ds[3], ds[1])], [(ds[1], ds[3], ds[2])], [ds]]
origins = [[(0.0, 0.0, 0.0)] for _ in 1:5]
dummyStudySrc = Vector{Vector{Tuple{String,String}}}()

# =======================================================================
# Phase 5: Launch Vulkan display + Makie GUI in parallel
# =======================================================================

import Observables
using MedEye3d.LesionMetadataWindow

# Parse segment names for display
nrrd_path = joinpath(data_dir_pat6, "PET_Lesions_0.seg.nrrd")
segment_names = isfile(nrrd_path) ? LesionAssociation.parse_nrrd_segment_names(nrrd_path) : Dict{Int, String}()

unique_vals = sort(unique(first_mask))
lesion_ids_ints = filter(x -> x > 0, unique_vals)

# Load match groups
LesionAssociation.load_matches_from_h5(preprocessed_h5)
match_groups = LesionAssociation.get_match_groups()

# Build lesion list
lesion_list = if isempty(lesion_ids_ints)
    ["(none)"]
else
    map(lesion_ids_ints) do i
        seg_int = Int(i)
        display_name = get(organ_mapping, seg_int, "")
        if isempty(display_name); display_name = get(segment_names, seg_int, ""); end
        if isempty(display_name); display_name = "Segment_$seg_int"; end
        
        found_gid = nothing; found_matches = 0
        node_name_0 = get(tp_nodes_map, 0, "PET_Lesions_0")
        for (gid, members) in match_groups
            for (node, s_int, _) in members
                if node == node_name_0 && s_int == seg_int
                    found_gid = gid; found_matches = length(members); break
                end
            end
            found_gid !== nothing && break
        end
        found_gid !== nothing ? "$seg_int: $display_name [Grp $found_gid, $(found_matches) TPs]" : "$seg_int: $display_name"
    end
end

active_lesion = Observables.Observable("(none)")
if !isempty(lesion_list) && lesion_list[1] != "(none)"
    active_lesion[] = lesion_list[1]
end
lesion_ids = Observables.Observable(lesion_list)

# Create Makie GUI data (Figure + widgets) on main thread — no GLFW window yet.
# GLFW requires all window creation on thread 1 (main thread), so we cannot use
# Threads.@spawn for GLMakie.Screen(). Instead: build layout first, then launch
# Vulkan display, then open the Makie GLFW window last (all on thread 1).
println("  [MAKIE] Creating control panel layout..."); flush(stdout)
t_makie = @elapsed begin
    # Pass nothing as channel = controls greyed out until connected
    makie_win = LesionMetadataWindow.create_metadata_window(active_lesion, lesion_ids, nothing)
end
println("  [MAKIE] Layout created in $(round(t_makie, digits=1))s"); flush(stdout)

# --- Launch Vulkan display (main thread) ---
println("Launching MedEye3d Vulkan display...")
mainMedEye3dInstance = SegmentationDisplay.displayImage(
    dummyStudySrc;
    textureSpecArray=textureSpecArray,
    voxelDataTupleVector=first_voxelDataTupleVector,
    spacings=spacings,
    origins=origins,
    windowWidth=1100,
    fractionOfMainImage=Float32(1.0),
    quadView=true
)

# Connect Makie to Vulkan channel and open the GLMakie window (main thread)
println("  [MAKIE] Connecting to Vulkan event channel..."); flush(stdout)
LesionMetadataWindow.connect_channel!(makie_win, mainMedEye3dInstance.channel)
makie_screen = LesionMetadataWindow.display_metadata_window(makie_win.fig)
println("  [MAKIE] Control window ready!"); flush(stdout)

# =======================================================================
# Phase 6: JIT warmup + metadata population
# =======================================================================

println("Running JIT warmup...")
# Initial restore & plane switching warmup
put!(mainMedEye3dInstance.channel, CompareTimePointsEvent(false))
put!(mainMedEye3dInstance.channel, Int64(0))
put!(mainMedEye3dInstance.channel, ChangePlaneEvent(:Coronal))
put!(mainMedEye3dInstance.channel, ChangePlaneEvent(:Sagittal))
put!(mainMedEye3dInstance.channel, ChangePlaneEvent(:Axial))

# Compare Volumes Warmup Cycle 1 (precompiles panel 5 Vulkan pipelines, UBOs, and descriptors)
put!(mainMedEye3dInstance.channel, CompareTimePointsEvent(true))
put!(mainMedEye3dInstance.channel, Int64(0))
put!(mainMedEye3dInstance.channel, CompareTimePointsEvent(false))
put!(mainMedEye3dInstance.channel, Int64(0))

# Compare Volumes Warmup Cycle 2 (precompiles texture re-upload, quad layout switches, and slice registration)
put!(mainMedEye3dInstance.channel, CompareTimePointsEvent(true))
put!(mainMedEye3dInstance.channel, Int64(0))
put!(mainMedEye3dInstance.channel, CompareTimePointsEvent(false))
put!(mainMedEye3dInstance.channel, Int64(0))

# TP navigation & overlay features warmup
put!(mainMedEye3dInstance.channel, ChangeTimePointEvent(1))
put!(mainMedEye3dInstance.channel, ChangeTimePointEvent(-1))
put!(mainMedEye3dInstance.channel, SyncLesionEvent(1))
put!(mainMedEye3dInstance.channel, ShowBoneMaskEvent(true))
put!(mainMedEye3dInstance.channel, ShowBoneMaskEvent(false))
put!(mainMedEye3dInstance.channel, ShowSingleLesionEvent(1))
put!(mainMedEye3dInstance.channel, ShowSingleLesionEvent(0))

# Start Background Inference Worker
worker_script = joinpath(@__DIR__, "..", "ai", "python_worker.py")
if isfile(worker_script)
    try
        println("Starting background inference worker on port 5005...")
        MedEye3d.InferenceClient.start_python_worker(worker_script)
    catch e
        @warn "Failed to auto-start python worker: $e"
    end
end

# Populate global references
MEH.global_bone_atlas[] = skelly_atlas !== nothing ? skelly_atlas : (bone_atlas !== nothing ? bone_atlas : zeros(Float32, 1, 1, 1))
MEH.global_organ_mapping[] = organ_mapping
MEH.global_ts_atlas[] = ts_atlas_aligned
MEH.global_ts_names[] = ts_names
MEH.patient_id[] = basename(data_dir_pat6)
MEH.h5_path_ref[] = preprocessed_h5
MEH.current_tp_index[] = 0
MEH.volume_z_size[] = size(first_mask, 3)

println("Loaded $(length(match_groups)) match groups from HDF5")

# Parse radiological descriptions from HDF5 embedded metadata
try
    HDF5.h5open(preprocessed_h5, "r") do h5
        if haskey(h5, "_meta_/metadata.json")
            meta_json = JSON.parse(read(h5["_meta_/metadata.json"]))
            date_descriptions = Dict{String, String}()
            date_english = Dict{String, String}()
            for item in meta_json
                for (k, v) in item
                    if v isa Dict
                        haskey(v, "Description") && (date_descriptions[k] = v["Description"])
                        haskey(v, "EnglishDescription") && (date_english[k] = v["EnglishDescription"])
                    end
                end
            end
            sorted_dates = sort(collect(keys(date_descriptions)))
            for (s_idx, study_tuple) in enumerate(studies)
                tp_idx = s_idx - 1
                ct_fname = length(study_tuple) >= 4 ? study_tuple[4] : ""
                pet_fname = length(study_tuple) >= 5 ? study_tuple[5] : ""
                matched = false
                for date_key in sorted_dates
                    if haskey(date_descriptions, date_key) && (occursin(date_key, ct_fname) || occursin(date_key, pet_fname))
                        MEH.tp_descriptions[tp_idx] = date_descriptions[date_key]
                        haskey(date_english, date_key) && (MEH.tp_english_descriptions[tp_idx] = date_english[date_key])
                        matched = true; break
                    end
                end
                if !matched && tp_idx < length(sorted_dates)
                    date_key = sorted_dates[tp_idx + 1]
                    !haskey(MEH.tp_descriptions, tp_idx) && haskey(date_descriptions, date_key) && (MEH.tp_descriptions[tp_idx] = date_descriptions[date_key])
                    !haskey(MEH.tp_english_descriptions, tp_idx) && haskey(date_english, date_key) && (MEH.tp_english_descriptions[tp_idx] = date_english[date_key])
                end
            end
            println("Loaded radiological descriptions for $(length(MEH.tp_descriptions)) time points")
        end
    end
catch e
    @warn "Failed to load metadata descriptions: $e"
end

# Pre-cache background SUVs asynchronously
@async for tp_idx in keys(MEH.pet_volumes_cache)
    try; LesionMetadataWindow.get_background_suvs(tp_idx); catch; end
end

println("Makie GUI fully connected.")

t_total_startup = (time_ns() - t_startup) / 1e9
println("\n=== MedEye3d ready in $(round(t_total_startup, digits=1))s ===")
println("  Data sources: HDF5 only (no NIfTI at runtime)")
println("  TPs cached: $(length(MEH.tp_data_cache)) ($(join(sort(collect(keys(MEH.tp_data_cache))), ", ")))")
println("  Centroids: $(length(MEH.lesion_centroids_cache))")
println("  Bone subseg: $(length(MEH.bone_subsegments_cache)) pairs")

# Run GLFW interaction loop on main thread (thread 1).
# GLFW.PollEvents() must be called on the main thread to process mouse, keyboard,
# scroll, and window events for BOTH the Vulkan viewer and the GLMakie control panel.
# Without this, all GLFW callbacks are dead and both windows appear frozen/black.
println("Interactive session ready!")
if get(ENV, "MEDEYE3D_TEST_MODE", "false") != "true"
    println("Close the viewer window to exit.")
    vulkan_window = mainMedEye3dInstance.states[1].mainForDisplayObjects.window
    try
        while !GLFW.WindowShouldClose(vulkan_window)
            GLFW.PollEvents()
            sleep(0.008)  # ~120 Hz poll rate, yield to other Julia tasks
        end
    catch e
        if !(e isa InterruptException)
            @warn "Event loop error: $e"
        end
    end
    println("Closing viewer...")
end
