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

# Load PET
for i in 0:3
    ct_file = joinpath(data_dir_pat6, "Fixed_CT_Volume_$i.nii.gz")
    pet_file = joinpath(data_dir_pat6, "SUV_PET_Image_$i.nii.gz")
    mask_file = joinpath(data_dir_pat6, "PET_Lesions_$i.nii.gz")
    if isfile(ct_file) && isfile(pet_file) && isfile(mask_file)
        println("Loading PET TP $i...")
        vdt, spc, msk = load_tp(ct_file, pet_file, mask_file)
        all_tps_data[idx] = vdt
        if idx == 0
            global first_voxelDataTupleVector = vdt
            global first_spacing = spc
            global first_mask = msk
        end
        global idx += 1
    end
end

# Load SPECT
for i in [0, 1, 2, 4]
    ct_file = joinpath(data_dir_pat6, "SPECT_CT_Volume_$i.nii.gz")
    pet_file = joinpath(data_dir_pat6, "SPECT_NM_Vendor_Volume_$i.nii.gz")
    mask_file = joinpath(data_dir_pat6, "SPECT_Lesions_$i.nii.gz")
    if isfile(ct_file) && isfile(pet_file) && isfile(mask_file)
        println("Loading SPECT TP $i...")
        vdt, spc, msk = load_tp(ct_file, pet_file, mask_file)
        all_tps_data[idx] = vdt
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
# Add labels: PET TPs 0-3, then SPECT TPs
pet_count = count(i -> isfile(joinpath(data_dir_pat6, "SUV_PET_Image_$i.nii.gz")), 0:3)
for i in 0:(pet_count-1)
    MEH.tp_labels[i] = "PET TP$i"
end
spect_indices = [0, 1, 2, 4]
for (offset, si) in enumerate(spect_indices)
    tp_idx = pet_count + offset - 1
    if haskey(all_tps_data, tp_idx)
        MEH.tp_labels[tp_idx] = "SPECT TP$si"
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
if isfile(ts_nrrd_path)
    println("Loading TotalSegmentator atlas for organ names...")
    ts_names = LesionAssociation.parse_nrrd_segment_names(ts_nrrd_path)
    ts_atlas, ts_sizes = LesionAssociation.load_nrrd_labelmap(ts_nrrd_path)
    if ts_atlas !== nothing && !isempty(ts_names)
        # The TS atlas is in the SAME voxel space as first_mask (both aligned to CT)
        # However, the TS atlas may have different dimensions if the CT was resampled.
        # If dimensions match, do direct lookup. Otherwise, scale coordinates.
        if size(ts_atlas) == size(first_mask)
            organ_mapping = LesionAssociation.map_lesions_to_organs(first_mask, ts_atlas, ts_names)
        else
            @warn "TS atlas size $(size(ts_atlas)) != mask size $(size(first_mask)) — attempting coordinate scaling"
            # Scale coordinates from mask space to atlas space
            organ_mapping = LesionAssociation.map_lesions_to_organs(first_mask, ts_atlas, ts_names)
        end
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
screen = display(win.fig)

# Run GLFW interaction loop manually
println("Interactive session ready!")
println("Press ENTER in this terminal to exit...")
readline()

println("Closing viewer...")
