using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

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

# Paths to data
data_dir_pat6 = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "..", "data", "pat_6_files")

println("Loading NIfTI data natively with registration transforms...")

# Load shared SceneHierarchy module
include(joinpath(@__DIR__, "..", "lib", "SceneHierarchy.jl"))
using .SceneHierarchy

studies = parse_studies_from_hierarchy(data_dir_pat6)

# Load baseline reference CT for spatial resampling grid
baseline_ct_path = joinpath(data_dir_pat6, studies[1][4])
baseline_ct = MedImages.load_image(baseline_ct_path, "CT")

# Map the distinct colors
colors_mapped = map(c -> RGB(c[1]/255, c[2]/255, c[3]/255), MedEye3d.distinctColorsSaved.listOfColors)

# Setup MedEye3d Display Textures
textureSpec_ct = TextureSpec{Float32}(name="CT", isMainImage=true, color=RGB(1.0, 1.0, 1.0), minAndMaxValue=Float32.([-150, 250]))
textureSpec_pet = TextureSpec{Float32}(name="PET", isMainImage=false, color=RGB(1.0, 0.5, 0.0), minAndMaxValue=Float32.([0, 10]))
textureSpec_mask = TextureSpec{Float32}(
    name="Mask", 
    isMainImage=false, 
    isMultiDiscreteMask=true,
    colorSet=colors_mapped,
    minAndMaxValue=Float32.([0, length(colors_mapped)]), isEditable=true
)
textureSpec_pure_pet = TextureSpec{Float32}(name="PET", isMainImage=true, color=RGB(1.0, 0.5, 0.0), minAndMaxValue=Float32.([0, 10]))
textureSpec_surface = TextureSpec{Float32}(name="Bone_Surface", isMainImage=false, color=RGB(0.0, 1.0, 1.0), minAndMaxValue=Float32.([0.5, 1.5]), isVisible=true)
textureSpec_marrow = TextureSpec{Float32}(name="Bone_Marrow", isMainImage=false, color=RGB(1.0, 1.0, 0.0), minAndMaxValue=Float32.([0.5, 1.5]), isVisible=true)

textureSpecArray = Vector{Vector{TextureSpec}}([
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask), deepcopy(textureSpec_surface), deepcopy(textureSpec_marrow)],
    TextureSpec[deepcopy(textureSpec_pure_pet)],
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask), deepcopy(textureSpec_surface), deepcopy(textureSpec_marrow)],
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask), deepcopy(textureSpec_surface), deepcopy(textureSpec_marrow)],
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask), deepcopy(textureSpec_surface), deepcopy(textureSpec_marrow)]
])

# To hold all TP data
all_tps_data = Dict{Int, Vector{Vector{Any}}}()

# Helper to load, transform, and resample one TP to baseline CT grid
function load_tp(ct_path, pet_path, mask_path, tfm_path, modality)
    ct = MedImages.load_image(ct_path, "CT")
    pet_raw = MedImages.load_image(pet_path, modality == "SPECT" ? "NM" : "PET")
    seg_raw = MedImages.load_image(mask_path, "CT")
    
    T_ITK = (tfm_path != "" && isfile(tfm_path)) ? parse_tfm(tfm_path) : Matrix{Float64}(I, 4, 4)
    
    ct_tfm = apply_transform_to_medimage(ct, T_ITK)
    pet_tfm = apply_transform_to_medimage(pet_raw, T_ITK)
    seg_tfm = apply_transform_to_medimage(seg_raw, T_ITK)
    
    ct_res = (T_ITK != Matrix{Float64}(I, 4, 4)) ? MedImages.resample_to_image(baseline_ct, ct_tfm, MedImages.Linear_en) : ct
    pet_res = MedImages.resample_to_image(baseline_ct, pet_tfm, MedImages.Linear_en)
    seg_res = MedImages.resample_to_image(baseline_ct, seg_tfm, MedImages.Nearest_neighbour_en)
    
    ct_vol = Float32.(ct_res.voxel_data)
    pet_vol = Float32.(pet_res.voxel_data)
    mask_vol = Float32.(seg_res.voxel_data)
    
    # Calibrate SPECT intensity: clamp negative reconstruction artifacts and scale to match PET SUV dynamic range
    if modality == "SPECT"
        pos_dat = pet_vol[pet_vol .> 0]
        p99 = isempty(pos_dat) ? 1.0f0 : Float32(quantile(pos_dat, 0.99))
        scale_factor = 8.0f0 / max(p99, 1.0f0)
        pet_vol = max.(0.0f0, pet_vol .* scale_factor)
    end
    
    ct_vol_base = reverse(ct_vol, dims=2)
    pet_vol_base = reverse(pet_vol, dims=2)
    mask_vol_base = reverse(mask_vol, dims=2)

    vol_img_axial = ct_vol_base
    vol_pet_axial = pet_vol_base
    vol_mask_axial = mask_vol_base

    vol_img_coronal = permutedims(ct_vol_base, (1, 3, 2))
    vol_pet_coronal = permutedims(pet_vol_base, (1, 3, 2))
    vol_mask_coronal = permutedims(mask_vol_base, (1, 3, 2))

    vol_img_sagittal = permutedims(ct_vol_base, (2, 3, 1))
    vol_pet_sagittal = permutedims(pet_vol_base, (2, 3, 1))
    vol_mask_sagittal = permutedims(mask_vol_base, (2, 3, 1))

    surf_axial = zeros(Float32, size(ct_vol_base))
    marr_axial = zeros(Float32, size(ct_vol_base))
    surf_sag = zeros(Float32, size(vol_img_sagittal))
    marr_sag = zeros(Float32, size(vol_img_sagittal))
    surf_cor = zeros(Float32, size(vol_img_coronal))
    marr_cor = zeros(Float32, size(vol_img_coronal))

    return Vector{Vector{Any}}([
        Any[("CT", vol_img_axial), ("PET", vol_pet_axial), ("Mask", vol_mask_axial), ("Bone_Surface", surf_axial), ("Bone_Marrow", marr_axial)],         
        Any[("PET", vol_pet_axial)],                                                          
        Any[("CT", vol_img_sagittal), ("PET", vol_pet_sagittal), ("Mask", vol_mask_sagittal), ("Bone_Surface", surf_sag), ("Bone_Marrow", marr_sag)],
        Any[("CT", vol_img_coronal), ("PET", vol_pet_coronal), ("Mask", vol_mask_coronal), ("Bone_Surface", surf_cor), ("Bone_Marrow", marr_cor)],   
        Any[("CT", vol_img_axial), ("PET", vol_pet_axial), ("Mask", vol_mask_axial), ("Bone_Surface", surf_axial), ("Bone_Marrow", marr_axial)]    
    ]), Tuple(Float64.(baseline_ct.spacing)), mask_vol_base
end

idx = 0
first_voxelDataTupleVector = nothing
first_spacing = nothing
first_mask = nothing
tp_labels_map = Dict{Int, String}()
tp_nodes_map = Dict{Int, String}()

# Chronological study list ordered by acquisition date from metadata.json:
# 1. 2022-03-10: PET TP 0 (Baseline)
# 2. 2022-05-19: SPECT TP 0 (Transform_SPECT_to_Baseline_0)
# 3. 2022-07-14: SPECT TP 1 (Transform_SPECT_to_Baseline_1)

studies = parse_studies_from_hierarchy(data_dir_pat6)

preprocessed_h5 = joinpath(data_dir_pat6, "preprocessed_volumes.h5")
if isfile(preprocessed_h5)
    println("Found preprocessed_volumes.h5! Fast loading...")
    import HDF5
    h5_file = HDF5.h5open(preprocessed_h5, "r")
    
    # Read Baseline spacing from the baseline CT
    base_ct_fname = studies[1][4]
    base_mask_fname = studies[1][6]
    global first_spacing = Tuple(Float64.(read(HDF5.attributes(h5_file["BASELINE/$base_ct_fname"])["spacing"])))
    global first_mask = reverse(Float32.(read(h5_file["BASELINE/$base_mask_fname"])), dims=2)
    first_mask_base = first_mask
    
    for (modality, orig_tp, date_str, ct_fname, pet_fname, mask_fname, node_name, tfm_fname) in studies
        lbl = "$modality $date_str (TP $orig_tp)"
        println("Loading $lbl from HDF5 (queue index $idx)...")
        
        group = tfm_fname == "" ? "BASELINE" : "TFM_" * tfm_fname
        
        ct_vol = Float32.(read(h5_file["$group/$ct_fname"]))
        pet_vol = Float32.(read(h5_file["$group/$pet_fname"]))
        mask_vol = Float32.(read(h5_file["$group/$mask_fname"]))
        
        if modality == "SPECT"
            pos_dat = pet_vol[pet_vol .> 0]
            p99 = isempty(pos_dat) ? 1.0f0 : Float32(quantile(pos_dat, 0.99))
            scale_factor = 8.0f0 / max(p99, 1.0f0)
            pet_vol = max.(0.0f0, pet_vol .* scale_factor)
        end
        
        ct_vol_base = reverse(ct_vol, dims=2)
        pet_vol_base = reverse(pet_vol, dims=2)
        mask_vol_base = reverse(mask_vol, dims=2)
        
        vol_img_sagittal = permutedims(ct_vol_base, (2, 3, 1))
        vol_pet_sagittal = permutedims(pet_vol_base, (2, 3, 1))
        vol_mask_sagittal = permutedims(mask_vol_base, (2, 3, 1))
        
        vol_img_coronal = permutedims(ct_vol_base, (1, 3, 2))
        vol_pet_coronal = permutedims(pet_vol_base, (1, 3, 2))
        vol_mask_coronal = permutedims(mask_vol_base, (1, 3, 2))
        
        surf_axial = zeros(Float32, size(ct_vol_base))
        marr_axial = zeros(Float32, size(ct_vol_base))
        surf_sag = zeros(Float32, size(vol_img_sagittal))
        marr_sag = zeros(Float32, size(vol_img_sagittal))
        surf_cor = zeros(Float32, size(vol_img_coronal))
        marr_cor = zeros(Float32, size(vol_img_coronal))
        
        vdt = Vector{Vector{Any}}([
            Any[("CT", ct_vol_base), ("PET", pet_vol_base), ("Mask", mask_vol_base), ("Bone_Surface", surf_axial), ("Bone_Marrow", marr_axial)],         
            Any[("PET", pet_vol_base)],                                                          
            Any[("CT", vol_img_sagittal), ("PET", vol_pet_sagittal), ("Mask", vol_mask_sagittal), ("Bone_Surface", surf_sag), ("Bone_Marrow", marr_sag)],
            Any[("CT", vol_img_coronal), ("PET", vol_pet_coronal), ("Mask", vol_mask_coronal), ("Bone_Surface", surf_cor), ("Bone_Marrow", marr_cor)],   
            Any[("CT", ct_vol_base), ("PET", pet_vol_base), ("Mask", mask_vol_base), ("Bone_Surface", surf_axial), ("Bone_Marrow", marr_axial)]    
        ])
        
        all_tps_data[idx] = vdt
        tp_labels_map[idx] = lbl
        tp_nodes_map[idx] = node_name
        if idx == 0
            global first_voxelDataTupleVector = vdt
        end
        global idx += 1
    end
    close(h5_file)
else
    # Legacy Slow Path
    for (modality, orig_tp, date_str, ct_fname, pet_fname, mask_fname, node_name, tfm_fname) in studies
        ct_file = joinpath(data_dir_pat6, ct_fname)
        pet_file = joinpath(data_dir_pat6, pet_fname)
        mask_file = joinpath(data_dir_pat6, mask_fname)
        tfm_file = tfm_fname != "" ? joinpath(data_dir_pat6, tfm_fname) : ""
        if isfile(ct_file) && isfile(pet_file) && isfile(mask_file)
            lbl = "$modality $date_str (TP $orig_tp)"
            println("Loading & registering $lbl (queue index $idx)...")
            vdt, spc, msk = load_tp(ct_file, pet_file, mask_file, tfm_file, modality)
            all_tps_data[idx] = vdt
            tp_labels_map[idx] = lbl
            tp_nodes_map[idx] = node_name
            if idx == 0
                global first_voxelDataTupleVector = vdt
                global first_spacing = spc
                global first_mask = msk
            end
            global idx += 1
        end
    end
end

spacings = [[first_spacing], [first_spacing], [(first_spacing[2], first_spacing[3], first_spacing[1])], [(first_spacing[1], first_spacing[3], first_spacing[2])], [first_spacing]]
origins = [[(0.0, 0.0, 0.0)] for _ in 1:5]

dummyStudySrc = Vector{Vector{Tuple{String,String}}}()

println("Launching MedEye3d Viewer...")
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

# Start Background Inference Worker via Docker (HELPNet & nnInteractive)
worker_script = joinpath(@__DIR__, "..", "ai", "python_worker.py")
if isfile(worker_script)
    try
        println("Starting background inference worker on port 5005...")
        MedEye3d.InferenceClient.start_python_worker(worker_script)
    catch e
        @warn "Failed to auto-start python worker: $e"
    end
end

# Populate TP data cache for navigation
MEH = MedEye3d.SegmentationDisplay.MakieEventHandlers
for (k, v) in all_tps_data
    MEH.tp_data_cache[k] = v
end
for (k, v) in tp_labels_map
    MEH.tp_labels[k] = v
end
for (k, v) in tp_nodes_map
    MEH.tp_node_names[k] = v
end

# Parse radiological descriptions from metadata.json
metadata_json_path = joinpath(data_dir_pat6, "metadata.json")
if isfile(metadata_json_path)
    try
        meta_json = JSON.parsefile(metadata_json_path)
        for item in meta_json
            for (k, v) in item
                if v isa Dict && haskey(v, "Description")
                    desc_text = v["Description"]
                    for (s_idx, (m, otp, ds, _...)) in enumerate(studies)
                        date_clean = replace(ds, "-" => "")
                        if date_clean == k
                            MEH.tp_descriptions[s_idx - 1] = desc_text
                        end
                    end
                end
            end
        end
        println("Loaded radiological descriptions for $(length(MEH.tp_descriptions)) time points.")
    catch e
        @warn "Failed to load metadata.json descriptions: $e"
    end
end

MEH.current_tp_index[] = 0

# Initialize Quad View and hide Pane 5
put!(mainMedEye3dInstance.channel, CompareTimePointsEvent(false))

# Start Makie Control Window
println("Starting Makie Control Window...")
import Observables
using MedEye3d.LesionMetadataWindow

active_lesion = Observables.Observable("(none)")

# Parse segment names from NRRD header for display
nrrd_path = joinpath(data_dir_pat6, "PET_Lesions_0.seg.nrrd")
segment_names = if isfile(nrrd_path)
    LesionAssociation.parse_nrrd_segment_names(nrrd_path)
else
    Dict{Int, String}()
end

# Load TotalSegmentator atlas for anatomical organ name lookup
ts_nrrd_path = joinpath(data_dir_pat6, "TS_all_Segmentation_0.seg.nrrd")
organ_mapping = Dict{Int, String}()
ts_atlas_aligned = nothing
if isfile(ts_nrrd_path)
    println("Loading TotalSegmentator atlas for organ names...")
    ts_names = LesionAssociation.parse_nrrd_segment_names(ts_nrrd_path)
    ts_atlas, ts_sizes = LesionAssociation.load_nrrd_labelmap(ts_nrrd_path)
    if ts_atlas !== nothing && !isempty(ts_names)
        # Apply Y-reversal on TS atlas to match the OpenGL display convention of first_mask
        ts_atlas_aligned = reverse(ts_atlas, dims=2)
        organ_mapping = LesionAssociation.map_lesions_to_organs(first_mask, ts_atlas_aligned, ts_names)
        
        bone_keywords = ["femur", "hip", "vertebra", "rib", "sacrum", "clavicula", "humerus", "scapula", "sternum", "skull", "palate", "bone", "spine"]
        bone_labels = Set{Int}()
        for (k, v) in ts_names
            v_low = lowercase(v)
            if any(kw -> occursin(kw, v_low), bone_keywords)
                push!(bone_labels, k)
            end
        end
        ct_vox = Float32.(baseline_ct.voxel_data)
        ct_aligned = reverse(ct_vox, dims=2)
        bone_atlas = Float32.(in.(ts_atlas_aligned, Ref(bone_labels)) .| (ct_aligned .>= 180.0f0))
        MEH.global_bone_atlas[] = bone_atlas
    end
else
    @warn "TotalSegmentator atlas not found at $ts_nrrd_path — using NRRD names only"
end

skelly_path = joinpath(data_dir_pat6, "Skellytour_0.nii.gz")
if isfile(skelly_path)
    println("Loading Skellytour Bone Subsegmentation mask for AI inference...")
    # Load Skellytour mask using the standard load_tp function or NIfTI directly
    using NIfTI
    skelly_nii = NIfTI.niread(skelly_path)
    skelly_vox = Float32.(skelly_nii.raw)
    skelly_aligned = reverse(skelly_vox, dims=2) # match OpenGL display convention
    MEH.global_bone_atlas[] = skelly_aligned
else
    @warn "Skellytour bone segmentation not found at $skelly_path. AI Bone Subsegmentation will fail!"
end

# Preload precomputed KernelAbstractions Bone Subsegments into MakieEventHandlers cache
println("Loading precomputed KernelAbstractions Bone Subsegmentation for bone lesions...")
output_h5 = joinpath(data_dir_pat6, "Bone_Subsegments_0.h5")
if isfile(output_h5)
    import HDF5
    try
        cis = CartesianIndices((512, 512, 326))
        HDF5.h5open(output_h5, "r") do file
            for obj in keys(file)
                if endswith(obj, "_surf")
                    lid_str = replace(replace(obj, "lesion_" => ""), "_surf" => "")
                    try
                        lid_int = parse(Int, lid_str)
                        marr_key = "lesion_$(lid_int)_marr"
                        if haskey(file, marr_key)
                            surf_data = read(file[obj])
                            marr_data = read(file[marr_key])
                            surf_pts = if ndims(surf_data) == 1
                                cis[surf_data]
                            else
                                findall(surf_data .> 0)
                            end
                            marr_pts = if ndims(marr_data) == 1
                                cis[marr_data]
                            else
                                findall(marr_data .> 0)
                            end
                            MEH.bone_subsegments_cache[lid_int] = (surf_pts, marr_pts)
                        end
                    catch err
                        @warn "Failed to parse lesion $obj: $err"
                    end
                end
            end
        end
        println("Loaded precomputed bone subsegments for $(length(MEH.bone_subsegments_cache)) lesions.")
    catch e
        @warn "Failed to read bone subsegments HDF5: $e"
    end
else
    @warn "Precomputed bone subsegments not found. Run scripts/preprocessing/precompute_bone_subsegments.jl to generate them."
end

unique_vals = sort(unique(first_mask))
lesion_ids_ints = filter(x -> x > 0, unique_vals)

# Precalculate lesion centroids for instant navigation (<1us)
for lid in lesion_ids_ints
    c = MEH.find_lesion_center(first_mask, Float32(lid))
    if c !== nothing
        MEH.lesion_centroids_cache[Int(lid)] = c
    end
end
println("Cached centroids for $(length(MEH.lesion_centroids_cache)) lesions.")

lesion_list = if isempty(lesion_ids_ints)
    ["(none)"]
else
    match_groups = LesionAssociation.get_match_groups()
    
    map(lesion_ids_ints) do i
        seg_int = Int(i)
        
        # 1. Priority: anatomical name from TS atlas
        display_name = get(organ_mapping, seg_int, "")
        
        # 2. Fallback: name from NRRD header
        if isempty(display_name)
            display_name = get(segment_names, seg_int, "")
        end
        
        # 3. Last resort: generic name
        if isempty(display_name)
            display_name = "Segment_$seg_int"
        end
        
        # Look up match group from matches.json
        found_gid = nothing
        found_matches = 0
        for (gid, members) in match_groups
            for (node, s_int, _) in members
                if node == "PET_Lesions_0" && s_int == seg_int
                    found_gid = gid
                    found_matches = length(members)
                    break
                end
            end
            found_gid !== nothing && break
        end
        
        if found_gid !== nothing
            "$seg_int: $display_name [Grp $found_gid, $(found_matches) TPs]"
        else
            "$seg_int: $display_name"
        end
    end
end

if !isempty(lesion_list) && lesion_list[1] != "(none)"
    active_lesion[] = lesion_list[1]
end
lesion_ids = Observables.Observable(lesion_list)

# Launch the Slicer Extension native port GUI
win = LesionMetadataWindow.create_metadata_window(active_lesion, lesion_ids, mainMedEye3dInstance.channel)
screen = LesionMetadataWindow.display_metadata_window(win.fig)



# Run GLFW interaction loop manually
println("Interactive session ready!")
println("Press ENTER to close the viewer...")
readline()
println("Closing viewer...")

