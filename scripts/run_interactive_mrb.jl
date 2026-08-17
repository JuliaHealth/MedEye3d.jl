using Pkg
Pkg.activate(".")
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

# Paths to data
data_dir_pat6 = joinpath(@__DIR__, "..", "data", "pat_6_files")

println("Loading NIfTI data natively with registration transforms...")

# Helper to parse ITK Affine Transform (.tfm)
function parse_tfm(tfm_path)
    if !isfile(tfm_path)
        return Matrix{Float64}(I, 4, 4)
    end
    lines = readlines(tfm_path)
    params = Float64[]
    fixed_params = Float64[0.0, 0.0, 0.0]
    for line in lines
        if startswith(line, "Parameters:")
            params = parse.(Float64, split(replace(line, "Parameters:" => "")))
        elseif startswith(line, "FixedParameters:")
            fixed_params = parse.(Float64, split(replace(line, "FixedParameters:" => "")))
        end
    end
    A = transpose(reshape(params[1:9], 3, 3))
    t = params[10:12]
    c = fixed_params
    offset = t + c - A * c
    T_RAS = Matrix{Float64}(I, 4, 4)
    T_RAS[1:3, 1:3] = A
    T_RAS[1:3, 4] = offset
    return T_RAS
end

# Helper to apply 4x4 RAS transform to MedImage spatial metadata
function apply_transform_to_medimage(img::MedImage, T_RAS::Matrix{Float64})
    if T_RAS == Matrix{Float64}(I, 4, 4)
        return img
    end
    L = Diagonal([-1.0, -1.0, 1.0, 1.0])
    T_LPS = L * T_RAS * L
    
    old_spacing = img.spacing
    old_dir = transpose(reshape(collect(img.direction), 3, 3))
    old_orig = img.origin
    
    M_old = zeros(Float64, 4, 4)
    for i in 1:3, j in 1:3
        M_old[i, j] = old_dir[i, j] * old_spacing[j]
    end
    for i in 1:3
        M_old[i, 4] = old_orig[i]
    end
    # Apply forward ITK LPS transformation matrix to place the moving image into baseline coordinate space
    M_new = T_LPS * M_old
    
    new_orig = (M_new[1, 4], M_new[2, 4], M_new[3, 4])
    new_spacing = zeros(Float64, 3)
    new_dir = zeros(Float64, 3, 3)
    
    for j in 1:3
        col = M_new[1:3, j]
        s = norm(col)
        new_spacing[j] = s
        new_dir[:, j] = col / s
    end
    
    new_dir_tuple = Tuple(transpose(new_dir))
    return MedImages.update_voxel_and_spatial_data(img, img.voxel_data, new_orig, Tuple(new_spacing), new_dir_tuple)
end

# Load baseline reference CT (Fixed_CT_Volume_0) for spatial resampling grid
baseline_ct_path = joinpath(data_dir_pat6, "Fixed_CT_Volume_0.nii.gz")
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
    minAndMaxValue=Float32.([0, length(colors_mapped)])
)
textureSpec_pure_pet = TextureSpec{Float32}(name="PET", isMainImage=true, color=RGB(1.0, 0.5, 0.0), minAndMaxValue=Float32.([0, 10]))

textureSpecArray = Vector{Vector{TextureSpec}}([
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask)],
    TextureSpec[deepcopy(textureSpec_pure_pet)],
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask)],
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask)],
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask)]
])

# To hold all TP data
all_tps_data = Dict{Int, Vector{Vector{Any}}}()

# Helper to load, transform, and resample one TP to baseline CT grid
function load_tp(ct_path, pet_path, mask_path, tfm_path, modality)
    ct = MedImages.load_image(ct_path, "CT")
    pet_raw = MedImages.load_image(pet_path, modality == "SPECT" ? "NM" : "PET")
    seg_raw = MedImages.load_image(mask_path, "CT")
    
    T_RAS = (tfm_path != "" && isfile(tfm_path)) ? parse_tfm(tfm_path) : Matrix{Float64}(I, 4, 4)
    
    ct_tfm = apply_transform_to_medimage(ct, T_RAS)
    pet_tfm = apply_transform_to_medimage(pet_raw, T_RAS)
    seg_tfm = apply_transform_to_medimage(seg_raw, T_RAS)
    
    ct_res = (T_RAS != Matrix{Float64}(I, 4, 4)) ? MedImages.resample_to_image(baseline_ct, ct_tfm, MedImages.Linear_en) : ct
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

    return Vector{Vector{Any}}([
        Any[("CT", vol_img_axial), ("PET", vol_pet_axial), ("Mask", vol_mask_axial)],         
        Any[("PET", vol_pet_axial)],                                                          
        Any[("CT", vol_img_sagittal), ("PET", vol_pet_sagittal), ("Mask", vol_mask_sagittal)],
        Any[("CT", vol_img_coronal), ("PET", vol_pet_coronal), ("Mask", vol_mask_coronal)],   
        Any[("CT", vol_img_axial), ("PET", vol_pet_axial), ("Mask", vol_mask_axial)]    
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
# 4. 2022-08-25: PET TP 1 (Transform_FollowUp_to_Baseline_1)
# 5. 2022-09-08: SPECT TP 2 (Transform_SPECT_to_Baseline_2)
# 6. 2022-12-08: PET TP 2 (Transform_FollowUp_to_Baseline_2)
# 7. 2023-01-23: SPECT TP 4 (Transform_SPECT_to_Baseline_4)
# 8. 2023-06-15: PET TP 3 (Transform_FollowUp_to_Baseline_3)
studies = [
    ("PET",   0, "2022-03-10", "Fixed_CT_Volume_0.nii.gz", "SUV_PET_Image_0.nii.gz",        "PET_Lesions_0.nii.gz",   "PET_Lesions_0",   ""),
    ("SPECT", 0, "2022-05-19", "SPECT_CT_Volume_0.nii.gz", "SPECT_NM_Vendor_Volume_0.nii.gz", "SPECT_Lesions_0.nii.gz", "SPECT_Lesions_0", "Transform_SPECT_to_Baseline_0.tfm"),
    ("SPECT", 1, "2022-07-14", "SPECT_CT_Volume_1.nii.gz", "SPECT_NM_Vendor_Volume_1.nii.gz", "SPECT_Lesions_1.nii.gz", "SPECT_Lesions_1", "Transform_SPECT_to_Baseline_1.tfm"),
    ("PET",   1, "2022-08-25", "Fixed_CT_Volume_1.nii.gz", "SUV_PET_Image_1.nii.gz",        "PET_Lesions_1.nii.gz",   "PET_Lesions_1",   "Transform_FollowUp_to_Baseline_1.tfm"),
    ("SPECT", 2, "2022-09-08", "SPECT_CT_Volume_2.nii.gz", "SPECT_NM_Vendor_Volume_2.nii.gz", "SPECT_Lesions_2.nii.gz", "SPECT_Lesions_2", "Transform_SPECT_to_Baseline_2.tfm"),
    ("PET",   2, "2022-12-08", "Fixed_CT_Volume_2.nii.gz", "SUV_PET_Image_2.nii.gz",        "PET_Lesions_2.nii.gz",   "PET_Lesions_2",   "Transform_FollowUp_to_Baseline_2.tfm"),
    ("SPECT", 4, "2023-01-23", "SPECT_CT_Volume_4.nii.gz", "SPECT_NM_Vendor_Volume_4.nii.gz", "SPECT_Lesions_4.nii.gz", "SPECT_Lesions_4", "Transform_SPECT_to_Baseline_4.tfm"),
    ("PET",   3, "2023-06-15", "Fixed_CT_Volume_3.nii.gz", "SUV_PET_Image_3.nii.gz",        "PET_Lesions_3.nii.gz",   "PET_Lesions_3",   "Transform_FollowUp_to_Baseline_3.tfm"),
]

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

# Start Background Inference Worker (HELPNet & nnInteractive)
worker_script = joinpath(@__DIR__, "python_worker.py")
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
    end
else
    @warn "TotalSegmentator atlas not found at $ts_nrrd_path — using NRRD names only"
end

# Embed precomputed KernelAbstractions Bone Subsegments into Mask volumes
println("Loading precomputed KernelAbstractions Bone Subsegmentation for bone lesions...")
output_h5 = joinpath(data_dir_pat6, "Bone_Subsegments_0.h5")
if isfile(output_h5)
    import HDF5
    HDF5.h5open(output_h5, "r") do file
        for (k, vdt) in all_tps_data
            for item in vdt
                for (tname, tvol) in item
                    if tname == "Mask"
                        lids = filter(x -> x > 0, unique(tvol))
                        for lid in lids
                            lid_int = Int(lid)
                            if haskey(file, "lesion_$(lid_int)_surf")
                                surf = file["lesion_$(lid_int)_surf"][:]
                                marr = file["lesion_$(lid_int)_marr"][:]
                                tvol[surf .> 0] .= 2.0f0
                                tvol[marr .> 0] .= 3.0f0
                            end
                        end
                    end
                end
            end
        end
    end
else
    @warn "Precomputed bone subsegments not found. Run scripts/preprocessing/precompute_bone_subsegments.jl to generate them."
end

unique_vals = sort(unique(first_mask))
lesion_ids_ints = filter(x -> x > 0, unique_vals)
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
println("Press ENTER in this terminal to exit...")
readline()

println("Closing viewer...")
