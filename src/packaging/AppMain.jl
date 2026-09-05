module MedEye3dApp

using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using MedEye3d.StructsManag
using MedEye3d.SegmentationDisplay
using MedEye3d.MakieEvents
using MedEye3d.LesionMetadataWindow
using MedEye3d.LesionAssociation
using MedImages
using ColorTypes
using Logging
using Statistics
using LinearAlgebra
using Dates
using GLMakie
import Observables
import GLFW
import HDF5
import JSON

const MEH = MedEye3d.SegmentationDisplay.MakieEventHandlers

# ── Clipboard safety patch ────────────────────────────────────────────────────
# Makie's editabletext calls InteractiveUtils.clipboard() directly on Ctrl+C/V.
# In Docker / headless environments without xclip/xsel this throws a hard error.
# Monkey-patch to silently degrade instead of crashing the app.
try
    import InteractiveUtils
    if Sys.islinux()
        try
            InteractiveUtils.clipboard("__clipboard_test__")
        catch
            @eval InteractiveUtils function clipboard(x::AbstractString)
                try
                    open(`xclip -selection clipboard`, "w") do io
                        print(io, x)
                    end
                catch
                    @warn "Clipboard not available (no xclip/xsel). Text not copied." maxlog=1
                end
            end
            @eval InteractiveUtils function clipboard()
                try
                    return read(`xclip -selection clipboard -o`, String)
                catch
                    return ""
                end
            end
            @info "[STARTUP] Patched InteractiveUtils.clipboard for headless/Docker environment"
        end
    end
catch; end

export julia_main

"""
    get_writable_log_dir() -> String

Returns a writable directory path for application log files on Windows.
Prefers `%APPDATA%\\MedEye3D\\logs` or `%LOCALAPPDATA%\\MedEye3D\\logs`, falling back to `tempdir()`.
"""
function get_writable_log_dir()::String
    appdata = get(ENV, "APPDATA", "")
    base_dir = if !isempty(appdata) && isdir(appdata)
        joinpath(appdata, "MedEye3D", "logs")
    else
        localappdata = get(ENV, "LOCALAPPDATA", "")
        if !isempty(localappdata) && isdir(localappdata)
            joinpath(localappdata, "MedEye3D", "logs")
        else
            joinpath(tempdir(), "MedEye3D_logs")
        end
    end
    try
        mkpath(base_dir)
        return base_dir
    catch
        return tempdir()
    end
end

"""
    run_viewer_loop(mainViewer)

Continuously pumps GLFW OS window events on the main thread and keeps the visualizer and Makie control panel responsive until closed.
"""
function run_viewer_loop(mainViewer)
    window = if !isempty(mainViewer.states) && mainViewer.states[1].mainForDisplayObjects !== nothing
        mainViewer.states[1].mainForDisplayObjects.window
    else
        nothing
    end

    println("MedEye3D main event loop active. Window: ", window !== nothing ? "ready" : "none")
    try
        while isopen(mainViewer.channel) && (window === nothing || !GLFW.WindowShouldClose(window))
            GLFW.PollEvents()
            sleep(0.005)
        end
    catch e
        # Channel closed or window terminated
    end
    println("MedEye3D session finished.")
end

"""
    launch_simple_volume(vol_ct::Array{Float32, 3}, title::String; spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), quad::Bool=true)

Visualizes a single 3D scalar volume with optional multi-planar QuadView.
"""
function launch_simple_volume(vol_ct::Array{Float32, 3}, title::String; spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), quad::Bool=true)
    dim_x, dim_y, dim_z = size(vol_ct)
    vol_mask = zeros(Float32, dim_x, dim_y, dim_z)

    min_ct, max_ct = Float32(minimum(vol_ct)), Float32(maximum(vol_ct))
    if min_ct == max_ct
        max_ct += 1.0f0
    end

    textureSpec_ct = TextureSpec{Float32}(
        name="CT",
        isMainImage=true,
        color=RGB(1.0, 1.0, 1.0),
        minAndMaxValue=Float32.([min_ct, min(max_ct, 1500.0f0)])
    )

    textureSpec_mask = TextureSpec{Float32}(
        name="Mask",
        isMainImage=false,
        color=RGB(1.0, 0.0, 0.0),
        minAndMaxValue=Float32.([0, 1]),
        maskContribution=0.5f0,
        isEditable=true
    )

    specs = TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_mask)]
    tuples = Any[("CT", vol_ct), ("Mask", vol_mask)]

    if quad
        vol_ct_coronal = PermutedDimsArray(vol_ct, (1, 3, 2))
        vol_mask_coronal = PermutedDimsArray(vol_mask, (1, 3, 2))
        spacing_coronal = (spacing[1], spacing[3], spacing[2])

        vol_ct_sagittal = PermutedDimsArray(vol_ct, (2, 3, 1))
        vol_mask_sagittal = PermutedDimsArray(vol_mask, (2, 3, 1))
        spacing_sagittal = (spacing[2], spacing[3], spacing[1])

        voxelDataTupleVector = Vector{Vector{Any}}([
            tuples,
            Any[("Mask", vol_mask)],
            Any[("CT", vol_ct_coronal), ("Mask", vol_mask_coronal)],
            Any[("CT", vol_ct_sagittal), ("Mask", vol_mask_sagittal)]
        ])

        textureSpecArray = Vector{Vector{TextureSpec}}([
            specs,
            TextureSpec[deepcopy(textureSpec_mask)],
            TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_mask)],
            TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_mask)]
        ])

        spacings = [[spacing], [spacing], [spacing_coronal], [spacing_sagittal]]
        origins = [[origin], [origin], [origin], [origin]]
    else
        voxelDataTupleVector = Vector{Vector{Any}}([tuples])
        textureSpecArray = Vector{Vector{TextureSpec}}([specs])
        spacings = [[spacing]]
        origins = [[origin]]
    end

    dummyStudySrc = Vector{Vector{Tuple{String,String}}}()
    println("Opening MedEye3D window for: ", title)
    mainViewer = SegmentationDisplay.displayImage(
        dummyStudySrc;
        textureSpecArray=textureSpecArray,
        voxelDataTupleVector=voxelDataTupleVector,
        spacings=spacings,
        origins=origins,
        fractionOfMainImage=Float32(1.0),
        windowWidth=1280,
        quadView=quad
    )

    run_viewer_loop(mainViewer)
end

"""
    launch_demo(; quad::Bool=true)

Launches a standalone interactive 3D medical visualizer with synthetic CT and segmentation mask.
"""
function launch_demo(; quad::Bool=true)
    println("Initializing MedEye3D Demo Visualization...")

    # Create synthetic CT volume
    dim_x, dim_y, dim_z = 128, 128, 64
    vol_ct = zeros(Float32, dim_x, dim_y, dim_z)
    vol_mask = zeros(Float32, dim_x, dim_y, dim_z)

    cx, cy, cz = dim_x ÷ 2, dim_y ÷ 2, dim_z ÷ 2
    for z in 1:dim_z, y in 1:dim_y, x in 1:dim_x
        vol_ct[x, y, z] = -100.0f0 + 40.0f0 * sin(0.08f0 * x) * cos(0.08f0 * y)
        dx = (x - cx) / (dim_x * 0.35f0)
        dy = (y - cy) / (dim_y * 0.35f0)
        dz = (z - cz) / (dim_z * 0.35f0)
        r2 = dx*dx + dy*dy + dz*dz
        if r2 < 1.0f0
            vol_ct[x, y, z] = 40.0f0 + 120.0f0 * (1.0f0 - Float32(sqrt(r2)))
        end
        lx = (x - (cx + 16)) / 10.0f0
        ly = (y - (cy + 8)) / 10.0f0
        lz = (z - (cz - 4)) / 8.0f0
        if (lx*lx + ly*ly + lz*lz) < 1.0f0
            vol_mask[x, y, z] = 1.0f0
        end
    end

    spacing = (1.0, 1.0, 1.5)
    origin = (0.0, 0.0, 0.0)

    textureSpec_ct = TextureSpec{Float32}(name="CT", isMainImage=true, color=RGB(1.0, 1.0, 1.0), minAndMaxValue=Float32.([-150, 250]))
    textureSpec_mask = TextureSpec{Float32}(name="Mask", isMainImage=false, color=RGB(1.0, 0.2, 0.2), minAndMaxValue=Float32.([0, 1]), maskContribution=0.5f0, isEditable=true)

    if quad
        vol_ct_coronal = PermutedDimsArray(vol_ct, (1, 3, 2))
        vol_mask_coronal = PermutedDimsArray(vol_mask, (1, 3, 2))
        spacing_coronal = (spacing[1], spacing[3], spacing[2])

        vol_ct_sagittal = PermutedDimsArray(vol_ct, (2, 3, 1))
        vol_mask_sagittal = PermutedDimsArray(vol_mask, (2, 3, 1))
        spacing_sagittal = (spacing[2], spacing[3], spacing[1])

        voxelDataTupleVector = Vector{Vector{Any}}([
            Any[("CT", vol_ct), ("Mask", vol_mask)],
            Any[("Mask", vol_mask)],
            Any[("CT", vol_ct_coronal), ("Mask", vol_mask_coronal)],
            Any[("CT", vol_ct_sagittal), ("Mask", vol_mask_sagittal)]
        ])

        textureSpecArray = Vector{Vector{TextureSpec}}([
            TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_mask)],
            TextureSpec[deepcopy(textureSpec_mask)],
            TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_mask)],
            TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_mask)]
        ])

        spacings = [[spacing], [spacing], [spacing_coronal], [spacing_sagittal]]
        origins = [[origin], [origin], [origin], [origin]]
    else
        voxelDataTupleVector = Vector{Vector{Any}}([Any[("CT", vol_ct), ("Mask", vol_mask)]])
        textureSpecArray = Vector{Vector{TextureSpec}}([TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_mask)]])
        spacings = [[spacing]]
        origins = [[origin]]
    end

    dummyStudySrc = Vector{Vector{Tuple{String,String}}}()
    println("Opening MedEye3D Demo window...")
    mainViewer = SegmentationDisplay.displayImage(
        dummyStudySrc;
        textureSpecArray=textureSpecArray,
        voxelDataTupleVector=voxelDataTupleVector,
        spacings=spacings,
        origins=origins,
        fractionOfMainImage=Float32(1.0),
        windowWidth=1280,
        quadView=quad
    )

    run_viewer_loop(mainViewer)
end

"""
    launch_from_h5(h5_path::String; quad::Bool=true)

Loads and visualizes multi-modal medical datasets from a preprocessed HDF5 database (`preprocessed_volumes.h5`).
Initializes the complete clinical workflow:
- 5-layer textures (CT, Nuclear PET overlay, discrete Mask, Bone Surface/Marrow overlay, Anatomy)
- Panel 2 pure PET view and multi-planar QuadView (Axial, Coronal, Sagittal)
- Makie metadata control panel window (sliders, lesion dropdown, AI segmentation triggers, report generation)
- Real-time timepoint navigation (PET, SPECT, followups)
"""
function launch_from_h5(h5_path::String; quad::Bool=true)
    println("Opening HDF5 medical dataset: ", h5_path)
    if !isfile(h5_path)
        @error "Provided HDF5 file does not exist: $h5_path"
        return
    end

    ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"
    
    h5_init = try
        HDF5.h5open(h5_path, "r")
    catch e
        @error "Failed to open HDF5 file: $e"
        return
    end

    # Check if this is a preprocessed multi-study dataset
    is_multimodal = haskey(h5_init, "BASELINE")
    
    if !is_multimodal
        root_keys = keys(h5_init)
        vol_ct = Float32.(read(h5_init[first(root_keys)]))
        close(h5_init)
        return launch_simple_volume(vol_ct, basename(h5_path); quad=quad)
    end

    # 1. Parse studies and time points from scene_hierarchy / metadata
    meta_dates = Dict{String, String}()
    meta_modalities = Dict{String, String}()
    if haskey(h5_init, "_meta_/metadata.json")
        try
            meta_json = JSON.parse(read(h5_init["_meta_/metadata.json"]))
            for item in meta_json
                for (k, v) in item
                    if v isa Dict
                        d_str = if length(k) >= 8 && all(isdigit, k[1:8])
                            prefix = "$(k[1:4])-$(k[5:6])-$(k[7:8])"
                            suffix = length(k) > 8 ? " (" * replace(strip(c -> c == '_', k[9:end]), "_" => " ") * ")" : ""
                            prefix * suffix
                        else
                            k
                        end
                        for sub_k in keys(v)
                            sub_dict = v[sub_k]
                            if sub_dict isa Dict && haskey(sub_dict, "name")
                                v_name = sub_dict["name"]
                                meta_dates[v_name] = d_str
                                if haskey(sub_dict, "Modality")
                                    meta_modalities[v_name] = sub_dict["Modality"]
                                end
                            end
                        end
                    end
                end
            end
        catch e
            @warn "Could not parse metadata.json dates: $e"
        end
    end

    studies = []
    if haskey(h5_init, "_meta_/scene_hierarchy.json")
        try
            hierarchy = JSON.parse(read(h5_init["_meta_/scene_hierarchy.json"]))
            function extract_study(children, tfm_name)
                ct_name = ""; pet_name = ""; mask_name = ""; ts_name = ""; max_anatomy_source = ""; max_anatomy_labels = ""; skellytour_source = ""; modality = "PET"
                for child in children
                    if child["type"] == "vtkMRMLLinearTransformNode"; continue; end
                    name = child["name"]
                    child_mod = get(child, "modality", get(child, "Modality", get(meta_modalities, name, "")))
                    if !isempty(child_mod)
                        modality = child_mod
                    end
                    if child["type"] == "vtkMRMLScalarVolumeNode"
                        if occursin("NM", name) || occursin("PET", name) || occursin("SUV", name)
                            pet_name = name * ".nii.gz"
                            if isempty(child_mod) && (occursin("NM", name) || occursin("SPECT", name)); modality = "SPECT"; end
                        elseif occursin("CT", name) || occursin("MR", name) || occursin("T2", name) || occursin("ADC", name) || occursin("DWI", name) || occursin("T1", name)
                            ct_name = name * ".nii.gz"
                            if isempty(child_mod)
                                if occursin("ADC", name)
                                    modality = "ADC"
                                elseif occursin("DWI", name) || occursin("BVAL", name)
                                    modality = "DWI"
                                elseif occursin("T1", name)
                                    modality = "T1"
                                elseif occursin("MR", name) || occursin("T2", name)
                                    modality = "T2"
                                end
                            end
                        end
                    elseif child["type"] == "vtkMRMLSegmentationNode"
                        if occursin("Lesions", name)
                            mask_name = name * ".nii.gz"
                        elseif startswith(name, "max_anatomy_")
                            max_anatomy_source = get(child, "source", "")
                            max_anatomy_labels = get(child, "labels", "")
                        elseif startswith(name, "skellytour_")
                            skellytour_source = get(child, "source", "")
                        elseif occursin("TS_all", name) || occursin("TotalSegmentator", name)
                            ts_name = name * ".nii.gz"
                        end
                    end
                end
                if isempty(ct_name) || isempty(pet_name); return nothing; end
                ct_base = replace(ct_name, ".nii.gz" => "")
                parts = split(ct_base, "_")
                orig_tp = tryparse(Int, parts[end])
                if orig_tp === nothing; orig_tp = 0; end
                if haskey(meta_modalities, ct_base)
                    modality = meta_modalities[ct_base]
                end
                pet_base = replace(pet_name, ".nii.gz" => "")
                date_str = get(meta_dates, ct_base, get(meta_dates, pet_base, "$modality TP $orig_tp"))
                mask_base = replace(replace(mask_name, ".seg.nrrd" => ""), ".nii.gz" => "")
                return (modality, orig_tp, date_str, ct_name, pet_name, mask_name, mask_base, tfm_name, ts_name, max_anatomy_source, max_anatomy_labels, skellytour_source)
            end

            b = extract_study(hierarchy, "")
            if b !== nothing; push!(studies, b); end
            for node in hierarchy
                if node["type"] == "vtkMRMLLinearTransformNode"
                    tfm_name = node["name"]
                    tfm_file = tfm_name * ".tfm"
                    s = extract_study(get(node, "children", []), tfm_file)
                    if s !== nothing; push!(studies, s); end
                end
            end
            sort!(studies, by = x -> ((length(x[3]) >= 10 && isdigit(x[3][1])) ? x[3][1:10] : "9999-99-99", x[2]))

            # Harvest original RTOG / clinical segment names per timepoint
            function harvest_segments!(nodes, cur_tp=0)
                for node in nodes
                    tp = cur_tp
                    name = get(node, "name", "")
                    parts = split(replace(name, ".nii.gz" => ""), "_")
                    last_int = tryparse(Int, parts[end])
                    if last_int !== nothing
                        tp = last_int
                    end
                    if get(node, "type", "") == "vtkMRMLSegmentationNode" && haskey(node, "segments")
                        segs = node["segments"]
                        if segs isa AbstractVector
                            dict = get!(MEH.tp_segment_names, tp, Dict{Int, String}())
                            for (idx, item) in enumerate(segs)
                                if item isa AbstractDict
                                    dict[idx] = get(item, "name", "Segment $idx")
                                else
                                    dict[idx] = string(item)
                                end
                            end
                        end
                    end
                    if haskey(node, "children")
                        harvest_segments!(node["children"], tp)
                    end
                end
            end
            harvest_segments!(hierarchy, 0)
        catch e
            @warn "Failed to parse scene hierarchy: $e"
        end
    end

    if isempty(studies)
        base_ct = first(filter(k -> occursin("CT", k) || occursin("ct", k), keys(h5_init["BASELINE"])))
        base_pet = first(filter(k -> occursin("PET", k) || occursin("SUV", k), keys(h5_init["BASELINE"])))
        base_mask = first(filter(k -> occursin("Lesion", k) || occursin("mask", k), keys(h5_init["BASELINE"])))
        push!(studies, ("PET", 0, "BASELINE", base_ct, base_pet, base_mask, "PET_Lesions_0", "", "", "", "", ""))
    end

    base_ct_fname = studies[1][4]
    base_mask_fname = studies[1][6]
    first_spacing = Tuple(Float64.(read(HDF5.attributes(h5_init["BASELINE/$base_ct_fname"])["spacing"])))
    display_spacing = first_spacing

    is_preflipped = haskey(h5_init, "_meta_/preflipped") && read(h5_init["_meta_/preflipped"]) == 1
    raw_first_mask = read(h5_init["BASELINE/$base_mask_fname"])
    first_mask = is_preflipped ? Float32.(raw_first_mask) : reverse(Float32.(raw_first_mask), dims=2)

    ts_atlas_aligned = nothing
    ts_names = Dict{Int,String}()
    bone_atlas = nothing
    skelly_atlas = nothing
    organ_mapping = Dict{Int, String}()

    if haskey(h5_init, "ATLAS/max_anatomy")
        ts_atlas_aligned = read(h5_init["ATLAS/max_anatomy"])
    end

    if haskey(h5_init, "_meta_")
        for k in sort(collect(keys(h5_init["_meta_"])))
            if startswith(k, "anatomy_labels_tp_")
                try
                    tp_raw = JSON.parse(read(h5_init["_meta_/$k"]))
                    for (id_str, name) in tp_raw
                        id = parse(Int, id_str)
                        if !occursin("_class_", name) && !haskey(ts_names, id)
                            ts_names[id] = name
                        end
                    end
                    break
                catch; end
            end
        end
        if haskey(h5_init, "_meta_/max_anatomy_labels.json") && isempty(ts_names)
            try
                raw_labels = JSON.parse(read(h5_init["_meta_/max_anatomy_labels.json"]))
                ts_names = Dict{Int,String}(parse(Int, k) => v for (k, v) in raw_labels)
            catch; end
        end
        if haskey(h5_init, "_meta_/organ_mapping")
            try
                raw_organ = JSON.parse(read(h5_init["_meta_/organ_mapping"]))
                organ_mapping = Dict{Int,String}(parse(Int, k) => v for (k, v) in raw_organ)
            catch; end
        end
        if haskey(h5_init, "_meta_/segment_names.json")
            try
                raw_sn = JSON.parse(read(h5_init["_meta_/segment_names.json"]))
                for (tp_str, sdict) in raw_sn
                    tp_i = parse(Int, tp_str)
                    target = get!(MEH.tp_segment_names, tp_i, Dict{Int, String}())
                    for (sid_str, sname) in sdict
                        target[parse(Int, sid_str)] = string(sname)
                    end
                end
            catch; end
        end
    end

    if haskey(h5_init, "ATLAS/skellytour")
        skelly_atlas = read(h5_init["ATLAS/skellytour"])
    end
    if haskey(h5_init, "ATLAS/bone_atlas")
        bone_atlas = read(h5_init["ATLAS/bone_atlas"])
    end

    # Centroids
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
            end
        end
    end

    # Bone subsegments precomputed cache loading
    if haskey(h5_init, "BONE_SUBSEG")
        vol_size = ts_atlas_aligned !== nothing ? size(ts_atlas_aligned) : (512, 512, 326)
        cis_b = CartesianIndices(vol_size)
        node_to_tp = Dict{String, Int}(study[7] => s_idx - 1 for (s_idx, study) in enumerate(studies))
        bone_grp = h5_init["BONE_SUBSEG"]
        
        for obj in keys(bone_grp)
            if endswith(obj, "_surf")
                marr_key = replace(obj, "_surf" => "_marr")
                if haskey(bone_grp, marr_key)
                    try
                        surf_data = read(bone_grp[obj])
                        marr_data = read(bone_grp[marr_key])
                        surf_pts = ndims(surf_data) == 1 ? cis_b[surf_data] : findall(surf_data .> 0)
                        marr_pts = ndims(marr_data) == 1 ? cis_b[marr_data] : findall(marr_data .> 0)
                        
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
                    catch err
                        @warn "Failed to parse bone subseg $obj: $err"
                    end
                end
            end
        end
    end

    close(h5_init)

    # Initialize MEH global state early so event handlers and UI widgets have full atlas data
    MEH.global_bone_atlas[] = skelly_atlas !== nothing ? skelly_atlas : (bone_atlas !== nothing ? bone_atlas : zeros(Float32, 1, 1, 1))
    MEH.global_organ_mapping[] = organ_mapping
    MEH.global_ts_atlas[] = ts_atlas_aligned
    MEH.global_ts_names[] = ts_names
    MEH.patient_id[] = basename(h5_path)
    MEH.h5_path_ref[] = h5_path
    MEH.current_tp_index[] = 0
    MEH.volume_z_size[] = size(first_mask, 3)

    # 2. Textures configuration
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

    tp_labels_map = Dict{Int, String}()
    tp_nodes_map = Dict{Int, String}()
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

    # Load prostate-specific anatomy labels for MRI timepoints (must run after tp_modalities is populated)
    HDF5.h5open(h5_path, "r") do h5_r
        if haskey(h5_r, "_meta_/prostate_anatomy_labels.json")
            try
                raw_prostate = JSON.parse(read(h5_r["_meta_/prostate_anatomy_labels.json"]))
                prostate_labels = Dict{Int,String}(parse(Int, k) => v for (k, v) in raw_prostate)
                for (tp_i, mod) in MEH.tp_modalities
                    if uppercase(mod) in ("T2", "MRI", "MR", "T1", "ADC", "DWI")
                        MEH.anatomy_labels_cache[tp_i] = prostate_labels
                    end
                end
                @info "Loaded prostate anatomy labels for MRI TPs: $(collect(values(prostate_labels)))"
            catch; end
        end
    end

    function load_single_tp_from_h5(tp_i::Int)
        if tp_i < 0 || tp_i >= length(studies)
            return nothing
        end
        study = studies[tp_i + 1]
        modality, orig_tp, date_str, ct_fname, pet_fname, mask_fname, node_name, tfm_fname = study[1:8]
        group = tfm_fname == "" ? "BASELINE" : "TFM_" * tfm_fname
        
        HDF5.h5open(h5_path, "r") do h5_file
            is_pf = haskey(h5_file, "_meta_/preflipped") && read(h5_file["_meta_/preflipped"]) == 1
            ct_vol = Float32.(read(h5_file["$group/$ct_fname"]))
            pet_vol = Float32.(read(h5_file["$group/$pet_fname"]))
            mask_vol = read(h5_file["$group/$mask_fname"])
            
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
            
            lock(MEH._centroids_lock) do
                MEH.pet_volumes_cache[tp_i] = pet_vol_base
            end
            
            if eltype(mask_vol_base) == Int16
                mask_compact = mask_vol_base
            elseif eltype(mask_vol_base) <: Integer
                mask_compact = Int16.(mask_vol_base)
            else
                mask_vol_base = max.(0.0f0, mask_vol_base)
                mask_compact = Int16.(round.(mask_vol_base))
            end
            
            sz = size(ct_vol_base)
            bone_mask = zeros(Int8, sz)
            
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
            # Fallback: if no per-TP anatomy, reuse ATLAS or BASELINE anatomy
            # (MRI TPs are registered to baseline space, so the atlas is valid)
            if anatomy_vol === nothing
                for fallback_path in ["ATLAS/max_anatomy", "BASELINE/max_anatomy.nii.gz"]
                    if haskey(h5_file, fallback_path)
                        try
                            raw_anat = read(h5_file[fallback_path])
                            if needs_reverse
                                raw_anat = reverse(Float32.(raw_anat), dims=2)
                            end
                            anatomy_vol = eltype(raw_anat) <: Integer ? UInt16.(raw_anat) : UInt16.(round.(max.(0.0f0, Float32.(raw_anat))))
                            println("  [TP $tp_i] Using fallback anatomy from $fallback_path"); flush(stdout)
                            break
                        catch; end
                    end
                end
            end
            
            mask_i16 = Int16.(mask_compact)
            anat_i16 = anatomy_vol !== nothing ? Int16.(anatomy_vol) : nothing
            MEH.precompute_mask_centroids!(mask_compact, tp_i, node_name)
            return MEH.TpCacheEntry(ct_vol_base, pet_vol_base, mask_compact, bone_mask, anatomy_vol, mask_i16, anat_i16)
        end
    end

    MEH.register_tp_loader!(load_single_tp_from_h5)
    MEH.register_h5_mask_saver!(h5_path, studies)
    first_entry = load_single_tp_from_h5(0)
    MEH.tp_data_cache[0] = first_entry

    function entry_to_vdt(e::MEH.TpCacheEntry)
        mask_i16 = e.mask_i16
        bone_i8 = e.bone_mask
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

    # 3. Lesion List & Match Groups
    try; LesionAssociation.load_matches_from_h5(h5_path); catch; end
    match_groups = LesionAssociation.get_match_groups()

    unique_vals = sort(unique(first_mask))
    lesion_ids_ints = filter(x -> x > 0, unique_vals)
    tp0_sn = get(MEH.tp_segment_names, 0, Dict{Int, String}())
    for sid in keys(tp0_sn)
        if !(sid in lesion_ids_ints)
            push!(lesion_ids_ints, sid)
        end
    end
    sort!(lesion_ids_ints)

    lesion_list = if isempty(lesion_ids_ints)
        ["(none)"]
    else
        map(lesion_ids_ints) do i
            seg_int = Int(i)
            display_name = if haskey(tp0_sn, seg_int) && !isempty(tp0_sn[seg_int])
                tp0_sn[seg_int]
            elseif haskey(organ_mapping, seg_int)
                organ_mapping[seg_int]
            else
                "Segment_$seg_int"
            end
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

    # 4. Create Makie control window layout
    println("Creating Makie control panel layout...")
    makie_win = LesionMetadataWindow.create_metadata_window(active_lesion, lesion_ids, nothing)

    # 5. Launch Vulkan display
    println("Launching MedEye3D Vulkan display...")
    mainViewer = SegmentationDisplay.displayImage(
        dummyStudySrc;
        textureSpecArray=textureSpecArray,
        voxelDataTupleVector=first_voxelDataTupleVector,
        spacings=spacings,
        origins=origins,
        windowWidth=1100,
        fractionOfMainImage=Float32(1.0),
        quadView=true
    )

    # 6. Connect Makie window to Vulkan channel & display
    println("Connecting Makie window to Vulkan channel...")
    LesionMetadataWindow.connect_channel!(makie_win, mainViewer.channel)
    makie_screen = LesionMetadataWindow.display_metadata_window(makie_win.fig)

    # 7. Warmup JIT & initial event synchronization
    put!(mainViewer.channel, CompareTimePointsEvent(false))
    put!(mainViewer.channel, Int64(0))

    println("MedEye3D interactive clinical workflow initialized.")
    
    # 8. Pre-build E-PSMA structured reports (async, non-blocking)
    @async try
        MedEye3d.EPSMAStructuredReport.prebuild_reports!()
    catch e
        @warn "[STARTUP] E-PSMA report pre-build failed" exception=(e, catch_backtrace())
    end
    
    run_viewer_loop(mainViewer)
end

"""
    launch_from_file(file_path::String; quad::Bool=true)

Loads and visualizes a medical image (NIfTI, DICOM, HDF5, etc.) from `file_path`.
"""
function launch_from_file(file_path::String; quad::Bool=true)
    println("Loading medical image: ", file_path)
    if !isfile(file_path)
        @error "Provided file path does not exist: $file_path"
        return
    end

    if endswith(lowercase(file_path), ".h5") || endswith(lowercase(file_path), ".hdf5")
        return launch_from_h5(file_path; quad=quad)
    end

    med_img = MedImages.load_image(file_path, "")
    vol_img = Float32.(med_img.voxel_data)
    spacing = Tuple(Float64.(med_img.spacing))
    origin = Tuple(Float64.(med_img.origin))

    launch_simple_volume(vol_img, basename(file_path); spacing=spacing, origin=origin, quad=quad)
end

"""
    run_app(args::Vector{String})

Dispatches CLI actions: help, version, image loading, or interactive study selector GUI.
"""
function run_app(args::Vector{String})
    println("MedEye3D standalone runtime initializing...")
    println("Arguments: ", args)
    println("Julia version: ", VERSION)
    println("Threads available: ", Threads.nthreads())

    if "--help" in args || "-h" in args
        println("""
        MedEye3D - High-Performance 3D Medical Image Annotation & Visualization
        
        Usage:
          MedEye3D.exe [options] [image_file]

        Options:
          -h, --help        Show this help message
          -v, --version     Show version information
          --demo            Launch default test case or synthetic 3D phantom viewer
          [file_path]       Open medical image file (.nii, .nii.gz, .mha, .h5)
        """)
        return
    end

    if "--version" in args || "-v" in args
        println("MedEye3D Version 0.5.8 (x86_64-w64-mingw32)")
        return
    end

    MedEye3d.Telemetry.log_action("APP_START", Dict("args" => args))

    default_h5_candidates = [
        "D:\\MedEye3d.jl\\data\\preprocessed_volumes.h5",
        joinpath(@__DIR__, "..", "..", "data", "preprocessed_volumes.h5"),
        joinpath(get(ENV, "APPDATA", ""), "MedEye3D", "data", "preprocessed_volumes.h5"),
        joinpath(homedir(), "Downloads", "preprocessed_volumes.h5"),
        joinpath(homedir(), "Desktop", "preprocessed_volumes.h5"),
        joinpath(homedir(), "Documents", "preprocessed_volumes.h5"),
        joinpath(pwd(), "preprocessed_volumes.h5")
    ]
    default_h5_idx = findfirst(isfile, default_h5_candidates)
    default_h5 = default_h5_idx !== nothing ? default_h5_candidates[default_h5_idx] : nothing

    file_args = filter(a -> !startswith(a, "-"), args)
    if !isempty(file_args) && isfile(file_args[1])
        MedEye3d.Telemetry.log_action("CLI_LOAD_FILE", Dict("path" => basename(file_args[1])))
        launch_from_file(file_args[1]; quad=true)
    elseif "--demo" in args
        if default_h5 !== nothing
            println("Launching default test dataset: ", default_h5)
            MedEye3d.Telemetry.log_action("CLI_LOAD_DEMO_DATASET")
            launch_from_h5(default_h5; quad=true)
        else
            MedEye3d.Telemetry.log_action("CLI_LOAD_SYNTHETIC_PHANTOM")
            launch_demo(; quad=true)
        end
    else
        println("Opening MedEye3D File Selector / Demo Launcher...")
        action, path = MedEye3d.StudySelectorWindow.prompt_open_or_demo()
        if action == :file && isfile(path)
            println("Selected file: ", path)
            MedEye3d.Telemetry.log_action("GUI_LOAD_FILE", Dict("path" => basename(path)))
            launch_from_file(path; quad=true)
        elseif default_h5 !== nothing
            println("No file selected. Launching default test dataset: ", default_h5)
            MedEye3d.Telemetry.log_action("GUI_LOAD_DEFAULT_DATASET")
            launch_from_h5(default_h5; quad=true)
        else
            println("No file selected. Launching interactive 3D demo phantom visualizer...")
            MedEye3d.Telemetry.log_action("GUI_LOAD_SYNTHETIC_PHANTOM")
            launch_demo(; quad=true)
        end
    end
end

"""
    julia_main()::Cint

Standard C-compatible entrypoint function required by PackageCompiler.jl.
Captures stdio and stderr into real-time flushed log files in `%APPDATA%\\MedEye3D\\logs`,
initializes loggers, catches unhandled exceptions, and returns exit code 0.
"""
function julia_main()::Cint
    log_dir = get_writable_log_dir()
    out_log_path = joinpath(log_dir, "medeye3d_output.log")
    err_log_path = joinpath(log_dir, "medeye3d_error.log")
    session_log_path = joinpath(log_dir, "medeye3d_session.log")

    # Rotate previous logs safely
    try
        if isfile(out_log_path)
            mv(out_log_path, joinpath(log_dir, "medeye3d_output_prev.log"), force=true)
        end
        if isfile(err_log_path)
            mv(err_log_path, joinpath(log_dir, "medeye3d_error_prev.log"), force=true)
        end
    catch
    end

    out_file = open(out_log_path, "w")
    err_file = open(err_log_path, "w")

    init_msg = """
    =======================================================
     MedEye3D Session Started: $(Dates.now())
     Log Directory: $log_dir
     Output Log:    $out_log_path
     Error Log:     $err_log_path
     Julia Version: $VERSION
     Threads:       $(Threads.nthreads())
     Arguments:     $ARGS
    =======================================================
    """
    println(out_file, init_msg)
    flush(out_file)

    try
        open(session_log_path, "a") do sess
            println(sess, init_msg)
            flush(sess)
        end
    catch
    end

    exit_code = Cint(0)
    redirect_stdio(stdout=out_file, stderr=err_file) do
        logger = SimpleLogger(err_file, Logging.Info)
        with_logger(logger) do
            try
                run_app(ARGS)
            catch e
                exit_code = Cint(1)
                err_msg = "FATAL UNHANDLED EXCEPTION: $(sprint(showerror, e))\n$(sprint(Base.show_backtrace, catch_backtrace()))"
                println(stderr, err_msg)
                flush(stderr)
                try
                    open(session_log_path, "a") do sess
                        println(sess, "ERROR [$(Dates.now())]: $err_msg")
                        flush(sess)
                    end
                catch
                end
            finally
                end_msg = "MedEye3D Session Ended: $(Dates.now()) (Exit code: $exit_code)\n"
                println(stdout, end_msg)
                flush(stdout)
                flush(stderr)
                try
                    open(session_log_path, "a") do sess
                        println(sess, end_msg)
                        flush(sess)
                    end
                catch
                end
                try close(out_file) catch end
                try close(err_file) catch end
            end
        end
    end

    return exit_code
end

end # module MedEye3dApp
