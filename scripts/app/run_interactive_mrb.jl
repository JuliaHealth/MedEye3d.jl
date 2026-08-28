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

const MEH = MedEye3d.SegmentationDisplay.MakieEventHandlers

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
textureSpec_pet = TextureSpec{Float32}(name="PET", isMainImage=false, isNuclearMask=true, color=RGB(1.0, 0.5, 0.0), minAndMaxValue=Float32.([0, 10]))
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
# Anatomy overlay: multi-discrete mask with up to 400 classes, invisible by default (shown on click)
anatomy_colors = [RGB(rand(), rand(), rand()) for _ in 1:400]
textureSpec_anatomy = TextureSpec{Float32}(
    name="Anatomy",
    isMainImage=false,
    isMultiDiscreteMask=true,
    colorSet=anatomy_colors,
    minAndMaxValue=Float32.([0, 400]),
    isVisible=false
)

textureSpecArray = Vector{Vector{TextureSpec}}([
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask), deepcopy(textureSpec_surface), deepcopy(textureSpec_marrow), deepcopy(textureSpec_anatomy)],
    TextureSpec[deepcopy(textureSpec_pure_pet)],
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask), deepcopy(textureSpec_surface), deepcopy(textureSpec_marrow), deepcopy(textureSpec_anatomy)],
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask), deepcopy(textureSpec_surface), deepcopy(textureSpec_marrow), deepcopy(textureSpec_anatomy)],
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_mask), deepcopy(textureSpec_surface), deepcopy(textureSpec_marrow), deepcopy(textureSpec_anatomy)]
])

# ── Hi-resolution display: resample to half in-plane spacing ─────────────
# Doubles X,Y resolution while keeping Z unchanged (4x memory per volume)
const HIRES_FACTOR = 2.0  # 2.0 = double resolution, 1.0 = native

"""
    hires_resample(vol::Array{Float32,3}, native_sp, half_sp, interp) -> Array{Float32,3}

Resample a Float32 3D volume from `native_sp` to `half_sp` using MedImages.
`interp` is MedImages.Linear_en for CT/PET or MedImages.Nearest_neighbour_en for masks.
"""
function hires_resample(vol::Array{Float32,3}, native_sp::Tuple{Float64,Float64,Float64},
                        half_sp::Tuple{Float64,Float64,Float64}, interp)
    if HIRES_FACTOR <= 1.0
        return vol
    end
    dummy_origin = (0.0, 0.0, 0.0)
    dummy_dir = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    im = MedImage(voxel_data=vol, spacing=native_sp, origin=dummy_origin,
                  direction=dummy_dir,
                  image_type=MedImages.MedImage_data_struct.MRI_type,
                  image_subtype=MedImages.MedImage_data_struct.CT_subtype,
                  patient_id="hires")
    resampled = MedImages.resample_to_spacing(im, half_sp, interp)
    return Float32.(resampled.voxel_data)
end

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

    # Hi-res resample
    native_sp = Tuple(Float64.(baseline_ct.spacing))
    disp_sp = (native_sp[1] / HIRES_FACTOR, native_sp[2] / HIRES_FACTOR, native_sp[3])
    ct_vol_base = hires_resample(ct_vol_base, native_sp, disp_sp, MedImages.Linear_en)
    pet_vol_base = hires_resample(pet_vol_base, native_sp, disp_sp, MedImages.Linear_en)
    mask_vol_base = hires_resample(mask_vol_base, native_sp, disp_sp, MedImages.Nearest_neighbour_en)

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
    # Panel 5 (compare right) needs its own independent bone arrays
    surf_axial_p5 = zeros(Float32, size(ct_vol_base))
    marr_axial_p5 = zeros(Float32, size(ct_vol_base))
    anat_axial = zeros(Float32, size(ct_vol_base))
    anat_sag = zeros(Float32, size(vol_img_sagittal))
    anat_cor = zeros(Float32, size(vol_img_coronal))

    return Vector{Vector{Any}}([
        Any[("CT", vol_img_axial), ("PET", vol_pet_axial), ("Mask", vol_mask_axial), ("Bone_Surface", surf_axial), ("Bone_Marrow", marr_axial), ("Anatomy", anat_axial)],         
        Any[("PET", vol_pet_axial)],                                                          
        Any[("CT", vol_img_sagittal), ("PET", vol_pet_sagittal), ("Mask", vol_mask_sagittal), ("Bone_Surface", surf_sag), ("Bone_Marrow", marr_sag), ("Anatomy", anat_sag)],
        Any[("CT", vol_img_coronal), ("PET", vol_pet_coronal), ("Mask", vol_mask_coronal), ("Bone_Surface", surf_cor), ("Bone_Marrow", marr_cor), ("Anatomy", anat_cor)],   
        Any[("CT", vol_img_axial), ("PET", vol_pet_axial), ("Mask", vol_mask_axial), ("Bone_Surface", surf_axial_p5), ("Bone_Marrow", marr_axial_p5), ("Anatomy", anat_axial)]    
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
    println("Found preprocessed_volumes.h5! Initializing on-demand streaming...")
    import HDF5
    h5_init = HDF5.h5open(preprocessed_h5, "r")
    
    # Read Baseline spacing from the baseline CT
    base_ct_fname = studies[1][4]
    base_mask_fname = studies[1][6]
    global first_spacing = Tuple(Float64.(read(HDF5.attributes(h5_init["BASELINE/$base_ct_fname"])["spacing"])))
    
    # Compute display spacing (halved in-plane for hi-res)
    display_spacing = (first_spacing[1] / HIRES_FACTOR, first_spacing[2] / HIRES_FACTOR, first_spacing[3])
    println("Native spacing: $first_spacing → Display spacing: $display_spacing ($(HIRES_FACTOR)x)")
    
    global first_mask = reverse(Float32.(read(h5_init["BASELINE/$base_mask_fname"])), dims=2)
    # Resample first_mask to display resolution for centroid/organ mapping later
    first_mask = hires_resample(first_mask, first_spacing, display_spacing, MedImages.Nearest_neighbour_en)
    first_mask_base = first_mask
    close(h5_init)
    
    function load_single_tp_from_h5(tp_i::Int)
        t_total = time_ns()
        if tp_i < 0 || tp_i >= length(studies)
            return nothing
        end
        study = studies[tp_i + 1]
        modality, orig_tp, date_str, ct_fname, pet_fname, mask_fname, node_name, tfm_fname = study[1:8]
        group = tfm_fname == "" ? "BASELINE" : "TFM_" * tfm_fname
        
        HDF5.h5open(preprocessed_h5, "r") do h5_file
            # Check if volumes were pre-flipped during preprocessing
            is_preflipped = haskey(h5_file, "_meta_/preflipped") && read(h5_file["_meta_/preflipped"]) == 1
            
            # Try pre-resampled display-resolution group first
            display_group = group * "_DISPLAY"
            use_display = haskey(h5_file, display_group) && 
                          haskey(h5_file[display_group], ct_fname) &&
                          haskey(h5_file[display_group], pet_fname) &&
                          haskey(h5_file[display_group], mask_fname)
            src_group = use_display ? display_group : group
            
            t_read = @elapsed begin
                ct_vol = Float32.(read(h5_file["$src_group/$ct_fname"]))
                pet_vol = Float32.(read(h5_file["$src_group/$pet_fname"]))
                mask_vol = read(h5_file["$src_group/$mask_fname"])  # Keep native type (Int8/Int16)
            end
            println("    [BENCH-H5] read from $src_group: $(round(t_read, digits=3))s ($(size(ct_vol)), mask=$(eltype(mask_vol)))"); flush(stdout)
            
            if modality == "SPECT"
                pos_dat = pet_vol[pet_vol .> 0]
                p99 = isempty(pos_dat) ? 1.0f0 : Float32(quantile(pos_dat, 0.99))
                scale_factor = 8.0f0 / max(p99, 1.0f0)
                pet_vol = max.(0.0f0, pet_vol .* scale_factor)
            end
            
            # Reverse is needed unless reading from pre-flipped DISPLAY group
            # (Native groups are NEVER pre-flipped, only DISPLAY groups are)
            needs_reverse = !is_preflipped || !use_display
            if needs_reverse
                t_reverse = @elapsed begin
                    ct_vol_base = reverse(ct_vol, dims=2)
                    pet_vol_base = reverse(pet_vol, dims=2)
                    mask_vol_base = reverse(mask_vol, dims=2)
                end
                println("    [BENCH-H5] reverse: $(round(t_reverse*1000, digits=1))ms"); flush(stdout)
            else
                ct_vol_base = ct_vol; pet_vol_base = pet_vol; mask_vol_base = mask_vol
                println("    [BENCH-H5] reverse: SKIPPED (pre-flipped DISPLAY)"); flush(stdout)
            end
            
            # Cache PET volume for SUV computation (use native-res for accuracy)
            MEH.pet_volumes_cache[tp_i] = pet_vol_base
            
            # Only resample if reading from native-resolution fallback group
            if !use_display
                t_resample = @elapsed begin
                    ct_vol_base = hires_resample(ct_vol_base, first_spacing, display_spacing, MedImages.Linear_en)
                    pet_vol_base = hires_resample(pet_vol_base, first_spacing, display_spacing, MedImages.Linear_en)
                    mask_vol_base = hires_resample(mask_vol_base, first_spacing, display_spacing, MedImages.Nearest_neighbour_en)
                end
                println("    [BENCH-H5] hires_resample ×3: $(round(t_resample, digits=3))s (FALLBACK - native res)"); flush(stdout)
            else
                println("    [BENCH-H5] hires_resample: SKIPPED (pre-resampled)"); flush(stdout)
            end
            
            # Convert mask to compact int type (skip if already compact from preprocessing)
            t_mask = @elapsed begin
                if eltype(mask_vol_base) <: Integer
                    mask_compact = mask_vol_base  # already Int8/Int16 from preprocessing
                else
                    # Ensure no negative background values (like -1024 from CT resampling)
                    mask_vol_base = max.(0.0f0, mask_vol_base)
                    max_id = round(Int, maximum(mask_vol_base))
                    mask_compact = if max_id + 5 <= 127
                        Int8.(round.(mask_vol_base))
                    else
                        Int16.(round.(mask_vol_base))
                    end
                end
            end
            println("    [BENCH-H5] mask compact: $(round(t_mask*1000, digits=1))ms ($(eltype(mask_compact)))"); flush(stdout)
            
            # Bone arrays start as empty BitArrays
            sz = size(ct_vol_base)
            bone_surf = falses(sz...)
            bone_marr = falses(sz...)
            
            # Load per-TP max_anatomy atlas (prefer pre-registered/resampled volume from HDF5)
            anatomy_vol = nothing
            try
                max_anat_src = study[10]
                max_anat_lbl = study[11]
                
                # Try finding pre-registered anatomy in HDF5 group first
                anat_h5_key = ""
                if haskey(h5_file, src_group)
                    for k in keys(h5_file[src_group])
                        if k == "max_anatomy.nii.gz" || endswith(k, ".seg.nrrd") || startswith(k, "TS_all") || startswith(k, "max_anatomy")
                            anat_h5_key = "$src_group/$k"
                            break
                        end
                    end
                end
                
                if !isempty(anat_h5_key) && haskey(h5_file, anat_h5_key)
                    raw_anat = read(h5_file[anat_h5_key])
                    # Reverse needed unless reading from pre-flipped DISPLAY group
                    if needs_reverse
                        raw_anat = reverse(Float32.(raw_anat), dims=2)
                    end
                    if !use_display && size(raw_anat) != size(ct_vol_base)
                        raw_anat = hires_resample(Float32.(raw_anat), first_spacing, display_spacing, MedImages.Nearest_neighbour_en)
                    end
                    # Auto-detect type: if already UInt16 from preprocessing, use directly
                    anatomy_vol = eltype(raw_anat) <: Integer ? UInt16.(raw_anat) : UInt16.(round.(max.(0.0f0, Float32.(raw_anat))))
                    println("    [BENCH-H5] Loaded max_anatomy from HDF5 ($anat_h5_key): $(size(anatomy_vol)) ($(eltype(raw_anat)))"); flush(stdout)
                elseif !isempty(max_anat_src)
                    # Fallback: load from NIfTI (for old HDF5 files without max_anatomy)
                    anat_path = joinpath(data_dir_pat6, max_anat_src)
                    if isfile(anat_path)
                        anat_nii = NIfTI.niread(anat_path)
                        anat_raw = Float32.(anat_nii.raw)
                        anat_aligned = reverse(anat_raw, dims=2)
                        if HIRES_FACTOR > 1.0
                            anat_aligned = hires_resample(anat_aligned, first_spacing, display_spacing, MedImages.Nearest_neighbour_en)
                        end
                        anatomy_vol = UInt16.(round.(max.(0.0f0, anat_aligned)))
                        println("    [BENCH-H5] Loaded max_anatomy from NIfTI fallback: $(size(anatomy_vol))")
                    end
                end
                
                # Build merged label dictionary for cursor readout:
                # Start with global names, then overlay per-TP real names (skip class_XX)
                if !isempty(max_anat_lbl)
                    tp_labels_path = joinpath(data_dir_pat6, max_anat_lbl)
                    if isfile(tp_labels_path)
                        tp_raw_labels = JSON.parsefile(tp_labels_path)
                        merged = copy(MEH.global_ts_names[])  # start with all 201 global real names
                        for (k_str, v) in tp_raw_labels
                            k_int = parse(Int, k_str)
                            if !occursin("class_", v)
                                merged[k_int] = v  # use per-TP real name (overrides global)
                            end
                            # class_XX entries fall through to the global name for this integer
                        end
                        MEH.anatomy_labels_cache[tp_i] = merged
                        println("    [BENCH-H5] Merged per-TP labels: $(length(merged)) entries ($(length(tp_raw_labels)) per-TP, $(length(MEH.global_ts_names[])) global)")
                    else
                        MEH.anatomy_labels_cache[tp_i] = MEH.global_ts_names[]
                    end
                else
                    MEH.anatomy_labels_cache[tp_i] = MEH.global_ts_names[]
                end
            catch e
                println("    [BENCH-H5] max_anatomy load failed for TP $tp_i: $e")
            end
            
            # Pre-convert mask/anatomy to Float32 once (avoids ~800ms per TP switch)
            sz = size(ct_vol_base)
            mask_f32 = Float32.(mask_compact)
            anat_f32 = anatomy_vol !== nothing ? Float32.(anatomy_vol) : zeros(Float32, sz)
            
            t_total_ms = (time_ns() - t_total) / 1e6
            println("    [BENCH-H5] LOAD TP TOTAL: $(round(t_total_ms, digits=1))ms"); flush(stdout)
            return MEH.TpCacheEntry(ct_vol_base, pet_vol_base, mask_compact, bone_surf, bone_marr, anatomy_vol, mask_f32, anat_f32)
        end
    end
    
    # Register fast on-demand loader with event handlers
    MEH.DEBUG_VERBOSE[] = true  # Enable per-step [BENCH] logs for TP switch
    MEH.register_tp_loader!(load_single_tp_from_h5)
    
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
        MEH.tp_node_names[tp_i] = node_name  # Populate for cross-TP bone cache lookups
    end
    
    println("Loading initial Time Point 0 on-demand...")
    first_entry = load_single_tp_from_h5(0)
    MEH.tp_data_cache[0] = first_entry
    
    # Convert TpCacheEntry to old vdt format for initial displayImage call
    # (displayImage still expects Vector{Vector{Any}} with concrete Array{Float32,3})
    function entry_to_vdt(e::MEH.TpCacheEntry)
        mask_f32 = e.mask_f32
        bone_s_f32 = Float32.(e.bone_surf)
        bone_m_f32 = Float32.(e.bone_marr)
        anat_f32 = if e.anatomy !== nothing
            e.anat_f32
        elseif MEH.global_ts_atlas[] !== nothing
            Float32.(MEH.global_ts_atlas[])
        else
            zeros(Float32, size(e.ct))
        end
        
        # collect() required here because displayImage (SegmentationDisplay.jl:1210)
        # has strict type: Union{Vector{Array{Float32,3}}, Vector{Vector{Array{Float32,3}}}}
        # PermutedDimsArray is AbstractArray, not Array. This is one-time startup cost.
        # TP switching uses _load_tp_from_entry! which uses zero-copy PermutedDimsArray views.
        # Each panel MUST have independent bone arrays — reactToSyncLesion writes
        # per-panel bone subseg indices, so shared arrays cause doubling.
        Vector{Vector{Any}}([
            Any[("CT", e.ct), ("PET", e.pet), ("Mask", mask_f32), ("Bone_Surface", bone_s_f32), ("Bone_Marrow", bone_m_f32), ("Anatomy", anat_f32)],
            Any[("PET", e.pet)],
            Any[("CT", collect(PermutedDimsArray(e.ct, (2,3,1)))), ("PET", collect(PermutedDimsArray(e.pet, (2,3,1)))), ("Mask", collect(PermutedDimsArray(mask_f32, (2,3,1)))), ("Bone_Surface", zeros(Float32, size(e.ct, 2), size(e.ct, 3), size(e.ct, 1))), ("Bone_Marrow", zeros(Float32, size(e.ct, 2), size(e.ct, 3), size(e.ct, 1))), ("Anatomy", collect(PermutedDimsArray(anat_f32, (2,3,1))))],
            Any[("CT", collect(PermutedDimsArray(e.ct, (1,3,2)))), ("PET", collect(PermutedDimsArray(e.pet, (1,3,2)))), ("Mask", collect(PermutedDimsArray(mask_f32, (1,3,2)))), ("Bone_Surface", zeros(Float32, size(e.ct, 1), size(e.ct, 3), size(e.ct, 2))), ("Bone_Marrow", zeros(Float32, size(e.ct, 1), size(e.ct, 3), size(e.ct, 2))), ("Anatomy", collect(PermutedDimsArray(anat_f32, (1,3,2))))],
            Any[("CT", e.ct), ("PET", e.pet), ("Mask", mask_f32), ("Bone_Surface", zeros(Float32, size(bone_s_f32))), ("Bone_Marrow", zeros(Float32, size(bone_m_f32))), ("Anatomy", anat_f32)]
        ])
    end
    global first_voxelDataTupleVector = entry_to_vdt(first_entry)
else
    # Legacy Slow Path
    for study in studies
        modality = study[1]
        orig_tp = study[2]
        date_str = study[3]
        ct_fname = study[4]
        pet_fname = study[5]
        mask_fname = study[6]
        node_name = study[7]
        tfm_fname = study[8]
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

# Compute display_spacing for slow path if not already set
if !@isdefined(display_spacing)
    display_spacing = (first_spacing[1] / HIRES_FACTOR, first_spacing[2] / HIRES_FACTOR, first_spacing[3])
    println("Native spacing: $first_spacing → Display spacing: $display_spacing ($(HIRES_FACTOR)x)")
end

ds = display_spacing
spacings = [[ds], [ds], [(ds[2], ds[3], ds[1])], [(ds[1], ds[3], ds[2])], [ds]]
origins = [[(0.0, 0.0, 0.0)] for _ in 1:5]

dummyStudySrc = Vector{Vector{Tuple{String,String}}}()

# Preload ALL remaining TPs before launching GUI (prevents premature clicking)
println("Preloading all remaining TPs before GUI launch...")
tp_indices_sorted = sort(collect(keys(tp_labels_map)))
for tp_idx in tp_indices_sorted
    if tp_idx != 0 && !haskey(MEH.tp_data_cache, tp_idx)
        t_preload = @elapsed begin
            entry = load_single_tp_from_h5(tp_idx)
            MEH.tp_data_cache[tp_idx] = entry
        end
        println("  [PRELOAD] TP $tp_idx loaded in $(round(t_preload, digits=1))s"); flush(stdout)
    end
end
println("All $(length(tp_indices_sorted)) TPs preloaded. Launching GUI..."); flush(stdout)

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

# (All TPs already preloaded before GUI launch — no background preload needed)

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
        
        # Collect all date→description entries (German and English)
        date_descriptions = Dict{String, String}()
        date_english = Dict{String, String}()
        for item in meta_json
            for (k, v) in item
                if v isa Dict
                    if haskey(v, "Description")
                        date_descriptions[k] = v["Description"]
                    end
                    if haskey(v, "EnglishDescription")
                        date_english[k] = v["EnglishDescription"]
                    end
                end
            end
        end
        
        # Sort dates chronologically → same order as studies (which are sorted by orig_tp)
        sorted_dates = sort(collect(keys(date_descriptions)))
        
        # Map studies to descriptions by matching CT/PET filenames that contain dates
        # or by chronological index if no date found in filenames
        for (s_idx, study_tuple) in enumerate(studies)
            tp_idx = s_idx - 1  # 0-indexed
            
            # Try to find date in the CT or PET filename
            ct_fname = length(study_tuple) >= 4 ? study_tuple[4] : ""
            pet_fname = length(study_tuple) >= 5 ? study_tuple[5] : ""
            matched = false
            
            for date_key in sorted_dates
                # Check if the CT/PET metadata entry's node name matches our study
                if haskey(date_descriptions, date_key)
                    # Direct date match in filename (e.g. "Fixed_CT_Volume_20220310")
                    if occursin(date_key, ct_fname) || occursin(date_key, pet_fname)
                        MEH.tp_descriptions[tp_idx] = date_descriptions[date_key]
                        if haskey(date_english, date_key)
                            MEH.tp_english_descriptions[tp_idx] = date_english[date_key]
                        end
                        matched = true
                        break
                    end
                end
            end
            
            # Fallback: use chronological index if no date match found
            if !matched && tp_idx < length(sorted_dates)
                # Interleaved PET/SPECT studies may double up indices,
                # so only match the first study per date
                date_key = sorted_dates[tp_idx + 1]
                if haskey(date_descriptions, date_key) && !haskey(MEH.tp_descriptions, tp_idx)
                    MEH.tp_descriptions[tp_idx] = date_descriptions[date_key]
                end
                if haskey(date_english, date_key) && !haskey(MEH.tp_english_descriptions, tp_idx)
                    MEH.tp_english_descriptions[tp_idx] = date_english[date_key]
                end
            end
        end
        
        println("Loaded radiological descriptions for $(length(MEH.tp_descriptions)) time points ($(length(date_descriptions)) dates in metadata.json, $(length(MEH.tp_english_descriptions)) in English)")
    catch e
        @warn "Failed to load metadata.json descriptions: $e"
    end
end

# Cache PET volumes per TP for SUV computation (axial orientation, Y-reversed)
for (tp_idx, vdt) in all_tps_data
    # vdt[1] is the axial panel: [("CT", ct_vol), ("PET", pet_vol), ("Mask", mask_vol), ...]
    for (name, vol) in vdt[1]
        if name == "PET"
            MEH.pet_volumes_cache[tp_idx] = vol
            break
        end
    end
end
println("Cached PET volumes for $(length(MEH.pet_volumes_cache)) time points.")

for study in studies
    modality = study[1]
    date_str = study[3]
    tp_idx_found = -1
    for (k, v) in tp_labels_map
        if occursin(date_str, v)
            tp_idx_found = k
            break
        end
    end
    if tp_idx_found >= 0
        MEH.tp_modalities[tp_idx_found] = modality
    end
end

# Extract patient ID from data directory name
MEH.patient_id[] = basename(data_dir_pat6)
# Register HDF5 path (single source of truth for JSON metadata)
MEH.h5_path_ref[] = preprocessed_h5

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

# Load per-timepoint max_anatomy atlas (319 classes) for anatomical organ name lookup
# No fallback to old TS seg.nrrd — max_anatomy is required
baseline_study = studies[1]
max_anatomy_source = baseline_study[10]
max_anatomy_labels_file = baseline_study[11]
skellytour_source = baseline_study[12]

if isempty(max_anatomy_source) || isempty(max_anatomy_labels_file)
    error("No max_anatomy in scene_hierarchy.json for baseline. Run:\n  julia scripts/preprocessing/update_scene_hierarchy.jl\n  bash scripts/ai/run_all_timepoints.sh")
end

max_anat_path = joinpath(data_dir_pat6, max_anatomy_source)
max_labels_path = joinpath(data_dir_pat6, max_anatomy_labels_file)

# Always try to use the true anatomic names if they exist, rather than synthetic fallback names
real_labels_path = joinpath(data_dir_pat6, "anatomy_out", "max_anatomy_labels.json")
if isfile(real_labels_path)
    max_labels_path = real_labels_path
end


if !isfile(max_anat_path)
    error("max_anatomy.nii.gz not found: $max_anat_path\nRun: bash scripts/ai/run_all_timepoints.sh")
end
if !isfile(max_labels_path)
    error("max_anatomy_labels.json not found: $max_labels_path\nRun: bash scripts/ai/run_all_timepoints.sh")
end

println("Loading max_anatomy atlas (319 classes) from $(basename(dirname(max_anatomy_source)))...")
using NIfTI
using JSON
nii_anat = NIfTI.niread(max_anat_path)
ts_atlas_raw = UInt16.(nii_anat.raw)
ts_atlas_aligned = reverse(ts_atlas_raw, dims=2)
if HIRES_FACTOR > 1.0
    ts_atlas_aligned = round.(UInt16, hires_resample(Float32.(ts_atlas_aligned), first_spacing, display_spacing, MedImages.Nearest_neighbour_en))
end
raw_labels = JSON.parsefile(max_labels_path)
ts_names = Dict{Int,String}(parse(Int, k) => v for (k, v) in raw_labels)
println("  Loaded $(length(ts_names)) anatomical classes")

organ_mapping = LesionAssociation.map_lesions_to_organs(first_mask, ts_atlas_aligned, ts_names)

bone_keywords = ["femur", "hip", "vertebra", "rib", "sacrum", "clavicula", "humerus",
                  "scapula", "sternum", "skull", "palate", "bone", "spine", "mandible",
                  "costal"]
bone_labels = Set{Int}()
for (k, v) in ts_names
    v_low = lowercase(v)
    if any(kw -> occursin(kw, v_low), bone_keywords)
        push!(bone_labels, k)
    end
end
ct_vox = Float32.(baseline_ct.voxel_data)
ct_aligned = reverse(ct_vox, dims=2)
if HIRES_FACTOR > 1.0
    ct_aligned = hires_resample(ct_aligned, first_spacing, display_spacing, MedImages.Linear_en)
end
bone_atlas = Float32.(in.(ts_atlas_aligned, Ref(bone_labels)))
MEH.global_bone_atlas[] = bone_atlas
MEH.global_organ_mapping[] = organ_mapping
# Cache atlas + names for SUV background organ computation
MEH.global_ts_atlas[] = ts_atlas_aligned
MEH.global_ts_names[] = ts_names
MEH.anatomy_labels_cache[0] = ts_names  # TP 0 anatomy labels for cursor readout

# Load per-timepoint Skellytour from hierarchy (overrides bone atlas for AI bone subsegmentation)
if isempty(skellytour_source)
    error("No Skellytour in scene_hierarchy.json for baseline. Run:\n  julia scripts/preprocessing/update_scene_hierarchy.jl")
end
skelly_path = joinpath(data_dir_pat6, skellytour_source)
if !isfile(skelly_path)
    error("Skellytour file not found: $skelly_path\nRun anatomy segmentation for baseline first.")
end
println("Loading Skellytour from $(basename(dirname(skellytour_source)))...")
skelly_nii = NIfTI.niread(skelly_path)
skelly_vox = Float32.(skelly_nii.raw)
skelly_aligned = reverse(skelly_vox, dims=2)
if HIRES_FACTOR > 1.0
    skelly_aligned = hires_resample(skelly_aligned, first_spacing, display_spacing, MedImages.Nearest_neighbour_en)
end
MEH.global_bone_atlas[] = skelly_aligned

# Preload precomputed KernelAbstractions Bone Subsegments into MakieEventHandlers cache
println("Loading precomputed KernelAbstractions Bone Subsegmentation for bone lesions...")
output_h5 = joinpath(data_dir_pat6, "Bone_Subsegments_0.h5")
if isfile(output_h5)
    import HDF5
    try
        cis = CartesianIndices((512, 512, 326))
        
        function scale_indices(pts, factor)
            if factor <= 1.0; return pts; end
            new_pts = CartesianIndex{3}[]
            f = Int(round(factor))
            for p in pts
                x, y, z = p.I
                for dx in 0:(f-1)
                    for dy in 0:(f-1)
                        push!(new_pts, CartesianIndex((x-1)*f + dx + 1, (y-1)*f + dy + 1, z))
                    end
                end
            end
            return new_pts
        end

        node_to_tp = Dict{String, Int}(study[7] => s_idx - 1 for (s_idx, study) in enumerate(studies))

        HDF5.h5open(output_h5, "r") do file
            for obj in keys(file)
                if endswith(obj, "_surf")
                    marr_key = replace(obj, "_surf" => "_marr")
                    if haskey(file, marr_key)
                        try
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
                            surf_pts = scale_indices(surf_pts, HIRES_FACTOR)
                            marr_pts = scale_indices(marr_pts, HIRES_FACTOR)
                            
                            # Parse key format without regex
                            # Formats: "PET_Lesions_0_lesion_24_surf", "tp_0_lesion_24_surf", "lesion_24_surf"
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
                            @warn "Failed to parse lesion $obj: $err"
                        end
                    end
                end
            end
        end
        println("Loaded precomputed bone subsegments for $(length(MEH.bone_subsegments_cache)) keys.")
        # Per-TP summary to verify all time points loaded
        for tp_i in 0:(length(studies)-1)
            tp_entries = count(k -> k isa Tuple{Int, Int} && k[1] == tp_i, keys(MEH.bone_subsegments_cache))
            node = get(tp_nodes_map, tp_i, "?")
            println("  Bone cache: tp_$(tp_i) ($node) = $(tp_entries) lesion pairs")
        end
        sample_keys = first(collect(keys(MEH.bone_subsegments_cache)), min(10, length(MEH.bone_subsegments_cache)))
        println("  Sample bone keys: $sample_keys")
    catch e
        @warn "Failed to read bone subsegments HDF5: $e"
    end
else
    @warn "Precomputed bone subsegments not found. Run scripts/preprocessing/precompute_bone_subsegments.jl to generate them."
end

unique_vals = sort(unique(first_mask))
lesion_ids_ints = filter(x -> x > 0, unique_vals)

# Precalculate lesion centroids for ALL studies for instant navigation (<1us)
if isfile(preprocessed_h5)
    HDF5.h5open(preprocessed_h5, "r") do h5
        for (s_idx, study) in enumerate(studies)
            tp_idx = s_idx - 1
            node_name = study[7]
            mask_fname = study[6]
            tfm_fname = study[8]
            group = tfm_fname == "" ? "BASELINE" : "TFM_" * tfm_fname
            
            if haskey(h5, group) && haskey(h5[group], mask_fname)
                mask_raw = reverse(Float32.(read(h5["$group/$mask_fname"])), dims=2)
                u_lids = filter(x -> x > 0, unique(mask_raw))
                for lid_f in u_lids
                    lid = Int(lid_f)
                    c_native = MEH.find_lesion_center(mask_raw, lid_f)
                    if c_native !== nothing
                        c_hires = [round(Int, c_native[1] * HIRES_FACTOR), round(Int, c_native[2] * HIRES_FACTOR), c_native[3]]
                        MEH.lesion_centroids_cache[(tp_idx, lid)] = c_hires
                        MEH.lesion_centroids_cache[(node_name, lid)] = c_hires
                        if tp_idx == 0
                            MEH.lesion_centroids_cache[lid] = c_hires
                        end
                    end
                end
            end
        end
    end
    println("Cached centroids for all 8 time points: $(length(MEH.lesion_centroids_cache)) entries.")
else
    for lid in lesion_ids_ints
        c = MEH.find_lesion_center(first_mask, Float32(lid))
        if c !== nothing
            MEH.lesion_centroids_cache[Int(lid)] = c
            MEH.lesion_centroids_cache[(0, Int(lid))] = c
        end
    end
    println("Cached centroids for $(length(MEH.lesion_centroids_cache)) lesions.")
end

# Store volume Z dimension for edge-slice artefact detection
MEH.volume_z_size[] = size(first_mask, 3)

# Load cross-TP match groups from HDF5 attribute (single source of truth)
LesionAssociation.load_matches_from_h5(preprocessed_h5)
println("Loaded $(length(LesionAssociation.get_match_groups())) match groups from HDF5")

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
        
        # Look up match group from HDF5 matches attribute
        found_gid = nothing
        found_matches = 0
        node_name_0 = get(tp_nodes_map, 0, "PET_Lesions_0")
        for (gid, members) in match_groups
            for (node, s_int, _) in members
                if node == node_name_0 && s_int == seg_int
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

