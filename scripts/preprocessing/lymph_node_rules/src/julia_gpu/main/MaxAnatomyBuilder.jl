module MaxAnatomyBuilder

using NIfTI
using JSON
using Adapt
using HDF5

export build_and_save_max_anatomy, build_and_save_max_anatomy_from_h5

# Layering logic keywords (mirrors build_max_anatomy.py)
const BONE_KW = ["vertebrae", "rib_", "scapula_", "clavicula", "femur", "hip_", "humerus", "sacrum", "skull", "sternum", "costal"]
const MUSCLE_KW = ["muscle", "gluteus", "autochthon", "iliopsoas", "pectoralis", "subscapularis",
                   "supraspinatus", "infraspinatus", "latissimus", "rectus_abdominis", "oblique", "erector",
                   "trapezius", "deltoid", "sartorius", "quadriceps", "scalene", "platysma", "masseter",
                   "temporalis", "pterygoid", "coracobrachial", "serratus", "teres_major", "triceps",
                   "psoas", "quadratus", "sternocleidomastoid", "pharyngeal", "prevertebral", "tongue",
                   "digastric", "thigh_medial", "thigh_posterior", "levator_scapulae", "sterno_thyroid",
                   "thyrohyoid", "transversospinalis"]
const COARSE_PARENTS = ["heart", "colon", "liver", "kidney_left", "kidney_right"]
const FINE_CHILDREN = ["atrial_appendage_left", "rectum", "kidney_cyst_left", "kidney_cyst_right",
                       "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
                       "lung_middle_lobe_right", "lung_lower_lobe_right", "pulmonary_artery",
                       "pulmonary_vein", "celiac_trunk", "portal_vein_and_splenic_vein"]


const BANNED_KW = [
    # Tissues, Fat, Body Envelopes
    "body_trunc", "body_extremities", "skin", "tissue_", "subcutaneous", "torso_fat", "intermuscular", "fat_visceral",
    "skeletal_muscle",
    # Cavities, Spaces, Compartments, Envelopes
    "cavity", "space", "pericardium", "mediastinum", "thorax_wall", "compartment", "air", "effusion",
    # Helper Planes, Points, Lines, Proxies
    "plane", "proxy", "helper", "_base", "_exclusion", "ligament", "bridge", 
    "point", "line", "bound", "margin", "floor", "wall", "box",
    # Processing Intermediate Tags
    "clean", "sanitized", "filled", "real", "dilated", "computed", "subtask", "split", "half", "foramen", "lesion", "tumor", "nodule", "pleural", "_p1", "_p2", "_p3", "_p4", "_p5", "combined", "fused", "hemorrhage", "lumen", "head", "lung_trachea_bronchia", "portal_vein_and_splenic_vein", "vertebrae_body", "ventricle_frontal_horn", "autochthon", "quadriceps", "lung_arteries", "lung_veins", "teeth", "intervertebral_discs", "sinus_maxillary", "outflow_tract", "liver_vessels", "fixed_ct", "spect_ct", "skellytour"
]

const CLINICAL_LN_STATIONS_LOWER = Set{String}(String[
    "neck_level_ia_submental", "neck_level_ib_submandibular_left", "neck_level_ib_submandibular_right",
    "neck_level_iia_upper_jugular_left", "neck_level_iia_upper_jugular_right", "neck_level_iib_upper_jugular_left",
    "neck_level_iib_upper_jugular_right", "neck_level_iii_middle_jugular_left", "neck_level_iii_middle_jugular_right",
    "neck_level_iv_lower_jugular_left", "neck_level_iv_lower_jugular_right", "neck_level_va_upper_posterior_triangle_left",
    "neck_level_va_upper_posterior_triangle_right", "neck_level_vi_anterior_cervical", "neck_level_xb_occipital_left",
    "neck_level_xb_occipital_right", "neck_parotid_left", "neck_parotid_right", "neck_retropharyngeal",
    "thoracic_station_2_upperparatracheal_left", "thoracic_station_2_upperparatracheal_right", "thoracic_station_3a_prevascular_left",
    "thoracic_station_3a_prevascular_right", "thoracic_station_3p_retrotracheal_left", "thoracic_station_3p_retrotracheal_right",
    "thoracic_station_4_lowerparatracheal_left", "thoracic_station_4_lowerparatracheal_right", "thoracic_station_5_subaortic_left",
    "thoracic_station_6_paraaortic", "thoracic_station_7_subcarinial", "thoracic_station_8_paraoesophageal_left",
    "thoracic_station_8_paraoesophageal_right", "thoracic_station_hilar_interlobar_left", "thoracic_station_hilar_interlobar_right",
    "thoracic_station_prepericardial_left", "thoracic_station_prepericardial_right", "thoracic_mammary_left",
    "thoracic_mammary_right", "axillary_level_i_left", "axillary_level_i_right", "axillary_level_ii_left",
    "axillary_level_ii_right", "axillary_level_iii_left", "axillary_level_iii_right", "axillary_rotter_left",
    "axillary_rotter_right", "abdominal_station_1_right_paracardial", "abdominal_station_2_left_paracardial",
    "abdominal_station_3_lesser_curvature", "abdominal_station_4_greater_curvature", "abdominal_station_5_suprapyloric",
    "abdominal_station_6_infrapyloric", "abdominal_station_7_left_gastric", "abdominal_station_8_common_hepatic",
    "abdominal_station_9_celiac", "abdominal_station_10_splenic_hilum", "abdominal_station_11_splenic_artery",
    "abdominal_station_13_posterior_pancreaticoduodenal", "abdominal_station_17_anterior_pancreaticoduodenal",
    "abdominal_station_14_sma", "abdominal_station_inferior_pancreatic", "abdominal_renal_hilar_left",
    "abdominal_renal_hilar_right", "abdominal_common_iliac_left", "abdominal_common_iliac_right",
    "abdominal_external_iliac_left", "abdominal_external_iliac_right", "abdominal_iliac_bifurcation_left",
    "abdominal_iliac_bifurcation_right", "abdominal_internal_iliac_left", "abdominal_internal_iliac_right",
    "abdominal_mesenteric_interenteric", "abdominal_obturator_left", "abdominal_obturator_right",
    "abdominal_station_16a1_aortic_hiatus", "abdominal_station_16a2_upper_middle_paraaortic",
    "abdominal_station_16b1_lower_middle_paraaortic", "abdominal_station_16b2_caudal_paraaortic",
    "abdominal_pararectal", "abdominal_presacral", "deep_inguinal_left", "deep_inguinal_right",
    "superficial_inguinal_left", "superficial_inguinal_right"
])

function is_banned(name::String, base_names::Union{Dict, Set{String}}, gen_names::Union{Dict, Set{String}, Nothing}=nothing)
    ln = lowercase(name)
    
    # If this is from gen_masks (a lymph node area), it MUST be in the clinical list.
    # Otherwise it is a deprecated station (like LowCervical, Supraclavicular) or a helper.
    if gen_names !== nothing
        is_in_gen = gen_names isa Dict ? haskey(gen_names, name) : in(name, gen_names)
        if is_in_gen && !(ln in CLINICAL_LN_STATIONS_LOWER)
            return true
        end
    end
    
    # 1. Check strict banned keywords
    for kw in BANNED_KW
        if occursin(kw, ln)
            return true
        end
    end
    if ln == "body" || ln == "fat" || ln == "muscle" || ln == "hip_left" || ln == "hip_right" || ln == "rib_1" || ln == "rib_2" || ln == "rib_3" || ln == "rib_5"
        return true
    end
    
    # Hard bans from subagent
    if ln in ["renal_arteries", "coronary_arteries", "venous_sinuses", "pulmonary_vein", "costal_cartilages", "subclavian_artery", "triceps_brachii", "infraspinatus", "pectoralis_minor", "deltoid", "subscapularis", "supraspinatus", "teres_major", "coracobrachial"]
        return true
    end
    
    has_base(n) = base_names isa Dict ? haskey(base_names, n) : in(n, base_names)

    # 2. Dynamic Parent Banning
    if ln == "aorta" && has_base("aorta_ascending") return true end
    if (ln == "lung_left" || ln == "lung_right" || ln == "lung") && has_base("lung_upper_lobe_left") return true end
    if ln == "liver" && has_base("liver_segment_1") return true end
    if ln == "heart" && has_base("heart_myocardium") return true end
    if ln == "scapula" && has_base("scapula_left") return true end
    if (ln == "clavicle" || ln == "clavicula") && has_base("clavicula_left") return true end
    if ln == "femur" && has_base("femur_left") return true end
    if ln == "iliac_artery_left" && has_base("iliac_artery_common_left") return true end
    if ln == "iliac_artery_right" && has_base("iliac_artery_common_right") return true end
    if ln == "iliac_vena_left" && has_base("iliac_vena_common_left") return true end
    if ln == "iliac_vena_right" && has_base("iliac_vena_common_right") return true end
    if ln == "obturator_internus" && has_base("obturator_internus_left") return true end
    if ln == "pulmonary_artery" && has_base("pulmonary_artery_left") return true end
    if ln == "bronchi" && has_base("bronchi_main_left") return true end
    if ln == "bronchi_left" && has_base("bronchi_main_left") return true end
    if ln == "bronchi_right" && has_base("bronchi_main_right") return true end
    if ln == "sternum" && has_base("manubrium") return true end
    if ln == "skull" && has_base("mandible") return true end
    if ln == "levator_scapulae" && has_base("levator_scapulae_left") return true end
    if ln == "latissimus_dorsi" && has_base("latissimus_dorsi_left") return true end
    if ln == "pectoralis_major" && has_base("pectoralis_major_left") return true end
    if ln == "trapezius" && has_base("trapezius_left") return true end
    if ln == "serratus_anterior" && has_base("serratus_anterior_left") return true end
    if ln == "iliopsoas" && has_base("iliopsoas_left") return true end
    if ln == "iliopsoas_left" && has_base("psoas_major_left") return true end
    if ln == "iliopsoas_right" && has_base("psoas_major_right") return true end
    if ln == "ventricle" && has_base("heart_ventricle_left") return true end
    if ln == "brain" && has_base("cerebellum") return true end
    if ln == "pharynx" && has_base("oropharynx") return true end
    if ln == "digastric_muscle_left" && has_base("digastric_left") return true end
    if ln == "digastric_muscle_right" && has_base("digastric_right") return true end
    if ln == "femoral_vena_left" && has_base("femoral_vein_left") return true end

    return false
end


function is_muscle(name::String)
    for kw in MUSCLE_KW
        if occursin(kw, lowercase(name))
            return true
        end
    end
    return false
end

function is_bone(name::String)
    for kw in BONE_KW
        if occursin(kw, lowercase(name)) && !is_muscle(name)
            return true
        end
    end
    return false
end

function build_and_save_max_anatomy(base_masks::Dict, gen_masks::Dict, ref_hdr::NIfTI.NIfTI1Header, out_dir::String)
    # Get dimensions from the first mask
    dims = size(first(values(base_masks)))
    max_anatomy = zeros(UInt16, dims...)
    label_dict = Dict{Int, String}()
    current_id = 1

    function apply_mask!(name::String, mask::AbstractArray)
        mask_cpu = adapt(Array, mask)
        coords = findall(x -> x > 0, mask_cpu)
        if !isempty(coords)
            max_anatomy[coords] .= current_id
            label_dict[current_id] = name
            current_id += 1
        end
    end

    println("  [MaxAnatomy] Pass 0: Lymph Nodes")
    for (name, mask) in gen_masks
        if !is_banned(name, base_masks, gen_masks)
            apply_mask!(name, mask)
        end
    end

    println("  [MaxAnatomy] Pass 1a: Coarse parent organs")
    for name in COARSE_PARENTS
        if haskey(base_masks, name) && !is_banned(name, base_masks, gen_masks)
            apply_mask!(name, base_masks[name])
        end
    end

    println("  [MaxAnatomy] Pass 1b: Regular organs and vessels")
    for (name, mask) in base_masks
        if !(name in COARSE_PARENTS) && !(name in FINE_CHILDREN) && !is_bone(name) && !is_muscle(name) && name != "mandible" && !is_banned(name, base_masks, gen_masks)
            apply_mask!(name, mask)
        end
    end

    println("  [MaxAnatomy] Pass 1c: Fine sub-organs")
    for name in FINE_CHILDREN
        if haskey(base_masks, name) && !is_banned(name, base_masks, gen_masks)
            apply_mask!(name, base_masks[name])
        end
    end

    println("  [MaxAnatomy] Pass 2: Skeletal bones and vertebrae")
    for (name, mask) in base_masks
        if is_bone(name) && !is_banned(name, base_masks, gen_masks)
            apply_mask!(name, mask)
        end
    end

    println("  [MaxAnatomy] Pass 3: Mandible")
    if haskey(base_masks, "mandible")
        apply_mask!("mandible", base_masks["mandible"])
    end

    println("  [MaxAnatomy] Pass 4: Granular muscles")
    for (name, mask) in base_masks
        if is_muscle(name) && !is_banned(name, base_masks, gen_masks)
            apply_mask!(name, mask)
        end
    end

    println("  [MaxAnatomy] Saving max_anatomy.nii.gz")
    mkpath(out_dir)
    out_path = joinpath(out_dir, "max_anatomy.nii.gz")
    
    # Save NIfTI
    hdr = deepcopy(ref_hdr)
    hdr.datatype = Int16(512) # UINT16
    hdr.bitpix = Int16(16)
    ni = NIfTI.NIVolume(hdr, max_anatomy)
    NIfTI.niwrite(out_path, ni)

    # Save JSON labels
    json_path = joinpath(out_dir, "max_anatomy_labels.json")
    open(json_path, "w") do f
        JSON.print(f, label_dict, 4)
    end
    
    println("  [MaxAnatomy] Saved with $(current_id - 1) classes.")
end

function build_and_save_max_anatomy_from_h5(base_h5::String, ln_h5::String, ref_hdr::NIfTI.NIfTI1Header, out_dir::String)
    println("  [MaxAnatomy] Building Max Anatomy streaming from HDF5...")
    t0 = time()
    h5open(base_h5, "r") do f_base
        base_grp = haskey(f_base, "masks") ? f_base["masks"] : f_base
        base_names = Set{String}(keys(base_grp))
        
        h5open(ln_h5, "r") do f_ln
            ln_names = Set{String}(keys(f_ln))
            
            # Get dimensions from first dataset in base_h5
            first_ds = base_grp[first(base_names)]
            dims = size(first_ds)
            max_anatomy = zeros(UInt16, dims...)
            label_dict = Dict{Int, String}()
            current_id = 1

            function apply_ds!(name::String, ds)
                mask = read(ds)
                coords = findall(x -> x > 0, mask)
                if !isempty(coords)
                    max_anatomy[coords] .= current_id
                    label_dict[current_id] = name
                    current_id += 1
                end
            end

            println("  [MaxAnatomy] Pass 0: Lymph Nodes")
            for name in sort(collect(ln_names))
                if !is_banned(name, base_names, ln_names)
                    if typeof(f_ln[name]) <: HDF5.Dataset
                        apply_ds!(name, f_ln[name])
                    end
                end
            end
            println("    -> Clinical lymph nodes added: $(current_id - 1)")

            println("  [MaxAnatomy] Pass 1a: Coarse parent organs")
            for name in COARSE_PARENTS
                if in(name, base_names) && !is_banned(name, base_names, nothing)
                    apply_ds!(name, base_grp[name])
                end
            end

            println("  [MaxAnatomy] Pass 1b: Regular organs and vessels")
            for name in sort(collect(base_names))
                if !(name in COARSE_PARENTS) && !(name in FINE_CHILDREN) && !is_bone(name) && !is_muscle(name) && name != "mandible" && !is_banned(name, base_names, nothing)
                    apply_ds!(name, base_grp[name])
                end
            end

            println("  [MaxAnatomy] Pass 1c: Fine sub-organs")
            for name in FINE_CHILDREN
                if in(name, base_names) && !is_banned(name, base_names, nothing)
                    apply_ds!(name, base_grp[name])
                end
            end

            println("  [MaxAnatomy] Pass 2: Skeletal bones and vertebrae")
            for name in sort(collect(base_names))
                if is_bone(name) && !is_banned(name, base_names, nothing)
                    apply_ds!(name, base_grp[name])
                end
            end

            println("  [MaxAnatomy] Pass 3: Mandible")
            if in("mandible", base_names)
                apply_ds!("mandible", base_grp["mandible"])
            end

            println("  [MaxAnatomy] Pass 4: Granular muscles")
            for name in sort(collect(base_names))
                if is_muscle(name) && !is_banned(name, base_names, nothing)
                    apply_ds!(name, base_grp[name])
                end
            end

            println("  [MaxAnatomy] Saving max_anatomy.nii.gz ($(current_id - 1) classes) in $(round(time() - t0, digits=2))s")
            mkpath(out_dir)
            out_path = joinpath(out_dir, "max_anatomy.nii.gz")
            
            # Save NIfTI
            hdr = deepcopy(ref_hdr)
            hdr.datatype = Int16(512) # UINT16
            hdr.bitpix = Int16(16)
            ni = NIfTI.NIVolume(hdr, max_anatomy)
            NIfTI.niwrite(out_path, ni)

            # Save JSON labels
            json_path = joinpath(out_dir, "max_anatomy_labels.json")
            open(json_path, "w") do f
                JSON.print(f, label_dict, 4)
            end
            
            println("  [MaxAnatomy] Successfully saved $(out_path) and $(json_path)")
        end
    end
end

function build_and_save_max_anatomy_from_h5(base_h5::String, ln_h5::String, out_dir::String)
    # Find reference header
    nii_files = filter(f -> endswith(f, ".nii.gz"), readdir(out_dir))
    ref_file = isfile(joinpath(out_dir, "max_anatomy.nii.gz")) ? 
        joinpath(out_dir, "max_anatomy.nii.gz") : 
        (!isempty(nii_files) ? joinpath(out_dir, nii_files[1]) : "")
    
    if !isempty(ref_file) && isfile(ref_file)
        ref_hdr = deepcopy(NIfTI.niread(ref_file).header)
    else
        # Fallback: create header with spacing from primary_masks.h5
        ref_hdr = NIfTI.NIfTI1Header()
        h5open(base_h5, "r") do f
            if haskey(f, "metadata") && haskey(attrs(f["metadata"]), "spacing")
                sp = attrs(f["metadata"])["spacing"]
                ref_hdr.pixdim = NTuple{8, Float32}([1.0, Float32(sp[1]), Float32(sp[2]), Float32(sp[3]), 0.0, 0.0, 0.0, 0.0])
            end
        end
    end
    build_and_save_max_anatomy_from_h5(base_h5, ln_h5, ref_hdr, out_dir)
end

end
