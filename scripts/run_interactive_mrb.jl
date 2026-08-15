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
import GLFW

# Paths to data
data_dir_pat6 = joinpath(@__DIR__, "..", "data", "pat_6_files")

println("Loading NIfTI data natively...")

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

# Helper to load and prepare one TP
function load_tp(ct_path, pet_path, mask_path)
    ct = MedImages.load_image(ct_path, "CT")
    pet_raw = MedImages.load_image(pet_path, "PET")
    seg_raw = MedImages.load_image(mask_path, "CT")

    pet = MedImages.resample_to_image(ct, pet_raw, MedImages.Linear_en)
    seg = MedImages.resample_to_image(ct, seg_raw, MedImages.Nearest_neighbour_en)

    ct_vol = Float32.(ct.voxel_data)
    pet_vol = Float32.(pet.voxel_data)
    mask_vol = Float32.(seg.voxel_data)
    
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
    ]), Tuple(Float64.(ct.spacing)), mask_vol_base
end

idx = 0
first_voxelDataTupleVector = nothing
first_spacing = nothing
first_mask = nothing
tp_labels_map = Dict{Int, String}()
tp_nodes_map = Dict{Int, String}()

# Chronological study list ordered by acquisition date from metadata.json:
# 1. 2022-03-10: PET TP 0
# 2. 2022-05-19: SPECT TP 0
# 3. 2022-07-14: SPECT TP 1
# 4. 2022-08-25: PET TP 1
# 5. 2022-09-08: SPECT TP 2
# 6. 2022-12-08: PET TP 2
# 7. 2023-01-23: SPECT TP 4
# 8. 2023-06-15: PET TP 3
studies = [
    ("PET",   0, "2022-03-10", "Fixed_CT_Volume_0.nii.gz", "SUV_PET_Image_0.nii.gz",        "PET_Lesions_0.nii.gz",   "PET_Lesions_0"),
    ("SPECT", 0, "2022-05-19", "SPECT_CT_Volume_0.nii.gz", "SPECT_NM_Vendor_Volume_0.nii.gz", "SPECT_Lesions_0.nii.gz", "SPECT_Lesions_0"),
    ("SPECT", 1, "2022-07-14", "SPECT_CT_Volume_1.nii.gz", "SPECT_NM_Vendor_Volume_1.nii.gz", "SPECT_Lesions_1.nii.gz", "SPECT_Lesions_1"),
    ("PET",   1, "2022-08-25", "Fixed_CT_Volume_1.nii.gz", "SUV_PET_Image_1.nii.gz",        "PET_Lesions_1.nii.gz",   "PET_Lesions_1"),
    ("SPECT", 2, "2022-09-08", "SPECT_CT_Volume_2.nii.gz", "SPECT_NM_Vendor_Volume_2.nii.gz", "SPECT_Lesions_2.nii.gz", "SPECT_Lesions_2"),
    ("PET",   2, "2022-12-08", "Fixed_CT_Volume_2.nii.gz", "SUV_PET_Image_2.nii.gz",        "PET_Lesions_2.nii.gz",   "PET_Lesions_2"),
    ("SPECT", 4, "2023-01-23", "SPECT_CT_Volume_4.nii.gz", "SPECT_NM_Vendor_Volume_4.nii.gz", "SPECT_Lesions_4.nii.gz", "SPECT_Lesions_4"),
    ("PET",   3, "2023-06-15", "Fixed_CT_Volume_3.nii.gz", "SUV_PET_Image_3.nii.gz",        "PET_Lesions_3.nii.gz",   "PET_Lesions_3"),
]

for (modality, orig_tp, date_str, ct_fname, pet_fname, mask_fname, node_name) in studies
    ct_file = joinpath(data_dir_pat6, ct_fname)
    pet_file = joinpath(data_dir_pat6, pet_fname)
    mask_file = joinpath(data_dir_pat6, mask_fname)
    if isfile(ct_file) && isfile(pet_file) && isfile(mask_file)
        lbl = "$modality $date_str (TP $orig_tp)"
        println("Loading $lbl (queue index $idx)...")
        vdt, spc, msk = load_tp(ct_file, pet_file, mask_file)
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
    windowWidth=1200,
    fractionOfMainImage=Float32(1.0),
    quadView=true
)

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

# Load cross-TP match groups from matches.json
matches_path = joinpath(data_dir_pat6, "matches.json")
if isfile(matches_path)
    LesionAssociation.load_matches_json(matches_path)
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

# Dispatch initial lesion synchronization so the viewer centers on Lesion 1
if !isempty(lesion_list) && lesion_list[1] != "(none)"
    m = match(r"\d+", lesion_list[1])
    if m !== nothing
        initial_lesion_id = parse(Int, m.match)
        @async begin
            sleep(0.5)  # Allow GLFW consumer to initialize
            put!(mainMedEye3dInstance.channel, MedEye3d.MakieEvents.SyncLesionEvent(initial_lesion_id))
            println("Initial lesion sync dispatched: Lesion $initial_lesion_id")
        end
    end
end

# Run GLFW interaction loop manually
println("Interactive session ready!")
println("Press ENTER in this terminal to exit...")
readline()

println("Closing viewer...")
