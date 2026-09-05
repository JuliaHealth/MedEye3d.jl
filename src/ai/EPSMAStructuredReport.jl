module EPSMAStructuredReport

using JSON
using Statistics
using Dates
import ..LLMDictation

export EPSMALesionRow,
       EPSMATechnicalParams,
       EPSMAReport,
       compute_psma_v,
       compute_reader_confidence,
       classify_mitnm,
       build_epsma_data,
       prebuild_reports!,
       get_or_build_report,
       invalidate_report!,
       enrich_with_llm!,
       export_to_docx,
       to_dict

# ── Data Structures ──────────────────────────────────────────────────────────
Base.@kwdef struct EPSMALesionRow
    id::Int = 0
    location::String = ""
    mitnm::String = "miM0"
    size_str::String = ""
    volume_cc::Float64 = 0.0
    diameter_mm::Float64 = 0.0
    num_lesions::Int = 1
    psma_q::String = ""
    suv_max::Float32 = 0.0f0
    psma_v::String = "Score 0"
    psma_v_num::Int = 0
    reader_confidence::Int = 5
    recip_status::String = "BASELINE"
    delta_str::String = ""
    is_new::Bool = false
    comment::String = ""
end

Base.@kwdef struct EPSMATechnicalParams
    radiotracer::String = "(not specified)"
    injected_activity::String = "(not specified)"
    uptake_time::String = "(not specified)"
    acquisition_type::String = "(not specified)"
    ct_protocol::String = "(not specified)"
    contrast::String = "(not specified)"
    diuretic::String = "(not specified)"
end

Base.@kwdef mutable struct EPSMAReport
    patient_id::String = "PAT_001"
    tp_index::Int = 0
    tp_label::String = "PET TP 0"
    modality::String = "PET/CT"
    study_date::String = ""
    tech_params::EPSMATechnicalParams = EPSMATechnicalParams()
    
    # Background SUVs
    background_suv::Dict{String, Float32} = Dict{String, Float32}("liver" => 2.3f0, "blood" => 1.5f0, "parotid" => 1.8f0)
    biodistribution_text_en::String = "(Please enter biodistribution statement.)"
    biodistribution_text_de::String = "(Bitte Biodistributionsbeschreibung eingeben.)"
    
    # Technical narrative (prose paragraph matching E-PSMA example style)
    tech_narrative_en::String = ""
    tech_narrative_de::String = ""
    
    # Patient History
    history_text_en::String = ""
    history_text_de::String = ""
    
    # Regional Findings (Narrative)
    findings_prostate_en::String = "No abnormal PSMA uptake detected in prostate / prostate bed."
    findings_prostate_de::String = "Keine pathologische PSMA-Mehranreicherung in der Prostata bzw. im Prostatabett nachweisbar."
    findings_lymph_en::String = "No suspicious PSMA-avid pelvic or extra-pelvic lymph nodes identified."
    findings_lymph_de::String = "Keine suspekten PSMA-positiven pelvinen oder extra-pelvinen Lymphknotenmetastasen abgrenzbar."
    findings_bone_en::String = "No focal abnormal PSMA uptake in axial or appendicular skeleton."
    findings_bone_de::String = "Kein Nachweis fokaler PSMA-Mehranreicherungen im Achsen- oder Extremitätenskelett."
    findings_visceral_en::String = "No PSMA-avid non-nodal visceral metastases or incidental findings."
    findings_visceral_de::String = "Keine PSMA-positiven viszeralen Fernmetastasen oder suspekten Nebenbefunde."
    findings_artifacts_en::String = ""
    findings_artifacts_de::String = ""
    skeletal_burden_en::String = "No bone metastases"
    skeletal_burden_de::String = "Keine Knochenmetastasen"
    
    # Synoptic Table 2 Rows (Only PCa metastases)
    synoptic_rows::Vector{EPSMALesionRow} = EPSMALesionRow[]
    # Incidental / Technical Artifact Rows (Table 6 of E-PSMA)
    artifact_rows::Vector{EPSMALesionRow} = EPSMALesionRow[]
    
    # Overall Assessment
    final_mitnm::String = "miT0 miN0 miM0"
    tmtv_cc::Float64 = 0.0
    tmtv_delta_pct::Float64 = 0.0
    overall_recip::String = "BASELINE (Initial Staging)"
    conclusion_en::String = ""
    conclusion_de::String = ""
    default_lang::String = "EN"
end

# ── Utility Helpers ──────────────────────────────────────────────────────────
"""English ordinal suffix: 1→1st, 2→2nd, 3→3rd, 4→4th, 11→11th, 12→12th, 13→13th, 21→21st, etc."""
function _ordinal(n::Int)::String
    if n % 100 in (11, 12, 13)
        return "$(n)th"
    end
    return n % 10 == 1 ? "$(n)st" : n % 10 == 2 ? "$(n)nd" : n % 10 == 3 ? "$(n)rd" : "$(n)th"
end

# ── Dynamic Module Helpers ───────────────────────────────────────────────────
function _get_meh()
    isdefined(Main, :MedEye3d) && isdefined(Main.MedEye3d, :SegmentationDisplay) &&
        isdefined(Main.MedEye3d.SegmentationDisplay, :MakieEventHandlers) &&
        return Main.MedEye3d.SegmentationDisplay.MakieEventHandlers
    return nothing
end

function _get_lmw()
    isdefined(Main, :MedEye3d) && isdefined(Main.MedEye3d, :LesionMetadataWindow) &&
        return Main.MedEye3d.LesionMetadataWindow
    return nothing
end

# ── Classification Logic (E-PSMA Tables 2, 3, 4) ─────────────────────────────
"""
    compute_psma_v(suv_max, liver_suv, blood_suv, parotid_suv) -> (score_str, score_int)

E-PSMA Table 2 4-point visual score:
- Score 0: Below blood pool
- Score 1: Equal to or above blood pool and lower than liver
- Score 2: Equal to or above liver and lower than parotid gland
- Score 3: Equal to or above parotid gland
"""
function compute_psma_v(suv_max::Real, liver_suv::Real, blood_suv::Real, parotid_suv::Real)::Tuple{String, Int}
    if isnan(suv_max) || isinf(suv_max) || suv_max <= 0
        return ("Score 0 (< Blood Pool)", 0)
    end
    if suv_max < blood_suv
        return ("Score 0 (< Blood Pool)", 0)
    elseif suv_max < liver_suv
        return ("Score 1 (>= Blood Pool, < Liver)", 1)
    elseif suv_max < parotid_suv
        return ("Score 2 (>= Liver, < Parotid)", 2)
    else
        return ("Score 3 (>= Parotid)", 3)
    end
end

"""
    compute_reader_confidence(organ, suv_max, liver_suv, is_bone) -> Int

E-PSMA Table 4: 5-point scale:
- 5: Definitive evidence of PCa (intense uptake in typical site with CT substrate)
- 4: Probably PCa (intense uptake in typical site without definitive CT substrate)
- 3: Equivocal finding (faint uptake in typical site or intense in atypical site)
- 2: Probably benign (faint uptake in atypical site)
- 1: Benign lesion without abnormal PSMA uptake
"""
function compute_reader_confidence(organ::String, suv_max::Real, liver_suv::Real, is_bone::Bool)::Int
    lo = lowercase(organ)
    is_typical = occursin("prostat", lo) || occursin("iliac", lo) || occursin("obturat", lo) ||
                 occursin("presacr", lo) || occursin("lymph", lo) || is_bone
    
    if suv_max >= liver_suv
        return is_typical ? 5 : 3
    elseif suv_max >= 2.0
        return is_typical ? 4 : 2
    else
        return is_typical ? 3 : 1
    end
end

"""
    classify_mitnm(organ, sublocation, details, lesion_type="", seg_name="") -> String

PROMISE / E-PSMA Table 3: Regional classification
- Local tumor (T): miT0, miT2 (organ-confined), miT3a (ECE), miT3b (SVI), miT4, miTr (bed recurrence)
- Regional nodes (N): miN0, miN1 (pelvic nodes)
- Distant metastases (M): miM1a (extra-pelvic nodes), miM1b (bone), miM1c (visceral)

Classification priority:
1. Muscle/artifact detection
2. Prostate keywords
3. LesionType button value (Bone Meta, Lymph Node, Prostate, Organ Meta)
4. Location/segment name keywords (incl. German: Knochen, Lymphknoten)
5. Fallthrough → miN1 (lymph node, most common in PSMA PET)
"""
function classify_mitnm(organ::String, sublocation::String, details::String, lesion_type::String="", seg_name::String="")::String
    lo = lowercase(organ * " " * sublocation * " " * details * " " * seg_name)
    ltype_lo = lowercase(lesion_type)
    
    muscle_keywords = ["gluteus", "autochthon", "iliopsoas", "pectoralis", "subscapularis",
                       "supraspinatus", "infraspinatus", "latissimus", "rectus_abdominis",
                       "oblique", "erector", "trapezius", "deltoid", "sartorius", "quadriceps",
                       "scalene", "platysma", "masseter", "temporalis", "pterygoid",
                       "coracobrachial", "serratus", "teres_major", "triceps", "psoas",
                       "quadratus", "sternocleidomastoid", "pharyngeal", "prevertebral",
                       "tongue", "digastric", "thigh_medial", "thigh_posterior",
                       "levator_scapulae", "sterno_thyroid", "thyrohyoid", "transversospinalis", "muscle", "muskel"]
    is_muscle = any(k -> occursin(k, lo), muscle_keywords)
    if (is_muscle || occursin("artifact", lo) || occursin("artefakt", lo) || occursin("false positive", lo) ||
       ltype_lo == "technical artifact") && ltype_lo != "lymph node" && ltype_lo != "lymph node meta"
        return "Artifact"
    end
    
    # ── 1. Prostate (from keywords OR LesionType) ──
    if occursin("prostat", lo) || ltype_lo == "prostate"
        if occursin("bed", lo) || occursin("bett", lo) || occursin("recurren", lo) || occursin("rezidiv", lo)
            return "miTr"
        elseif occursin("seminal", lo) || occursin("bläschen", lo) || occursin("vesic", lo)
            return "miT3b"
        elseif occursin("capsul", lo) || occursin("kapsel", lo) || occursin("extracaps", lo)
            return "miT3a"
        elseif occursin("rectum", lo) || occursin("bladder", lo) || occursin("blase", lo)
            return "miT4"
        else
            return "miT2"
        end
    end
    
    # ── 2. Lymph nodes: respect explicit LesionType FIRST (before bone keywords) ──
    # This ensures auto-inferred or user-set "Lymph Node" type takes priority
    # over atlas-derived bone/vascular keywords in the organ name.
    is_ln_type = ltype_lo == "lymph node" || ltype_lo == "lymph node meta"
    if is_ln_type
        if occursin("retroperiton", lo) || occursin("paraort", lo) || occursin("para-aort", lo) || occursin("mediastin", lo) ||
           occursin("supraclav", lo) || occursin("inguin", lo) || occursin("axill", lo) ||
           occursin("aorta", lo) || occursin("aortic", lo) || occursin("vena_cava", lo) || occursin("caval", lo)
            return "miM1a"
        else
            return "miN1"
        end
    end

    # ── 3. Bone / Skeleton (from keywords OR LesionType = "Bone Meta") ──
    bone_keywords = ["femur", "ilium", "ischium", "pubis", "vertebra", "spine", "rib", "humerus", "scapula", "clavicle", "sacrum", "bone", "knochen", "skelett", "wirbel", "rippe", "becken", "crest", "kamm", "acetabulum", "skull", "calvarium", "sternum", "mandible"]
    is_bone_kw = !is_muscle && any(k -> occursin(k, lo), bone_keywords) && !occursin("lymph", lo) && !occursin("node", lo) && !occursin("knoten", lo)
    if is_bone_kw || ltype_lo == "bone meta"
        return "miM1b"
    end

    # ── 4. Lymph nodes (from keywords in organ name) ──
    is_ln_kw = occursin("lymph", lo) || occursin("node", lo) || occursin("knoten", lo) || occursin("obturat", lo) || occursin("presacr", lo) || (occursin("iliac", lo) && !occursin("crest", lo))
    if is_ln_kw
        if occursin("retroperiton", lo) || occursin("paraort", lo) || occursin("para-aort", lo) || occursin("mediastin", lo) ||
           occursin("supraclav", lo) || occursin("inguin", lo) || occursin("axill", lo) ||
           occursin("aorta", lo) || occursin("aortic", lo) || occursin("vena_cava", lo) || occursin("caval", lo)
            return "miM1a"
        else
            return "miN1"
        end
    end
    
    # ── 5. Visceral ──
    visc_keywords = ["lung", "lunge", "liver", "leber", "brain", "gehirn", "pleura", "peritoneum", "adrenal", "nebenniere", "spleen", "milz"]
    if any(k -> occursin(k, lo), visc_keywords) || ltype_lo == "organ meta"
        return "miM1c"
    end
    
    # ── 6. Fallthrough → lymph node (most common unknown in PSMA PET) ──
    return is_muscle ? "Artifact" : "miN1"
end

# ── Anatomy Resolution Engine (max_anatomy + ontology) ─────────────────────────
const _ontology_cache = Ref{Union{Nothing, Dict{String, Any}}}(nothing)

function _lookup_ontology(raw_name::String)::Union{Nothing, Dict{String, Any}}
    LMW = _get_lmw()
    if LMW !== nothing
        try
            res = LMW.lookup_anatomy(raw_name)
            res !== nothing && return res
        catch; end
    end
    if _ontology_cache[] === nothing
        json_path = joinpath(@__DIR__, "..", "..", "data", "max_anatomy_to_ontology.json")
        if isfile(json_path)
            try
                _ontology_cache[] = JSON.parsefile(json_path)
            catch e
                @warn "Failed to load ontology in EPSMAStructuredReport: $e"
                _ontology_cache[] = Dict{String, Any}()
            end
        else
            _ontology_cache[] = Dict{String, Any}()
        end
    end
    cache = _ontology_cache[]
    if cache !== nothing && haskey(cache, raw_name)
        return cache[raw_name]
    end
    rn_low = lowercase(raw_name)
    if cache !== nothing
        for (k, v) in cache
            if lowercase(k) == rn_low
                return v
            end
        end
    end
    return nothing
end

"""
    resolve_anatomical_location(lid, state, tp_idx, _MEH, LMW) -> (clean_loc, is_artifact, is_muscle)

Resolves clean anatomical location from state or max_anatomy.
Never returns generic "Lesion \$lid" strings!
"""
function resolve_anatomical_location(lid::Integer, state::Union{AbstractDict, Nothing}, tp_idx::Integer=0, _MEH=nothing, LMW=nothing)::Tuple{String, Bool, Bool, String}
    st = state isa AbstractDict ? state : Dict{String, Any}()
    base_anat = string(get(st, "BaseAnatomy", ""))
    side = string(get(st, "BaseAnatomySide", ""))
    subloc = string(get(st, "Anatomical Sublocation", ""))
    details = string(get(st, "Anatomical Details", ""))
    alt_hyp = string(get(st, "Alternative Hypothesis (False Positive)", ""))
    certainty = string(get(st, "Certainty", ""))
    ltype = string(get(st, "LesionType", ""))

    raw_organ = ""
    # 1. If base_anat is empty or generic, check _MEH.global_organ_mapping
    if isempty(base_anat) || occursin("lesion", lowercase(base_anat))
        if _MEH !== nothing
            raw_organ = get(_MEH.global_organ_mapping[], lid, "")
            if raw_organ == "Unknown"
                raw_organ = ""
            end
        end
    else
        raw_organ = base_anat
    end

    # 2. If still empty, query max_anatomy atlas via centroid
    if isempty(raw_organ) && _MEH !== nothing && _MEH.global_ts_atlas[] !== nothing
        try
            atlas = _MEH.global_ts_atlas[]
            ts_nm = _MEH.global_ts_names[]
            centroid = get(_MEH.lesion_centroids_cache, (tp_idx, lid), nothing)
            if centroid === nothing
                centroid = get(_MEH.lesion_centroids_cache, lid, nothing)
            end
            if centroid !== nothing
                cx = clamp(round(Int, centroid[1]), 1, size(atlas, 1))
                cy = clamp(round(Int, centroid[2]), 1, size(atlas, 2))
                cz = clamp(round(Int, centroid[3]), 1, size(atlas, 3))
                val = Int(atlas[cx, cy, cz])
                if val > 0 && haskey(ts_nm, val)
                    raw_organ = ts_nm[val]
                    _MEH.global_organ_mapping[][lid] = raw_organ
                end
            end
        catch e
            @warn "Atlas centroid lookup failed for lesion $lid: $e"
        end
    end

    # 3. Look up in ontology mapping
    ont_entry = !isempty(raw_organ) ? _lookup_ontology(raw_organ) : nothing
    
    # 4. Check muscle and artifact flags
    muscle_kws = ["gluteus", "autochthon", "iliopsoas", "pectoralis", "subscapularis",
                  "supraspinatus", "infraspinatus", "latissimus", "rectus_abdominis",
                  "oblique", "erector", "trapezius", "deltoid", "sartorius", "quadriceps",
                  "scalene", "platysma", "masseter", "temporalis", "pterygoid",
                  "coracobrachial", "serratus", "teres_major", "triceps", "psoas",
                  "quadratus", "sternocleidomastoid", "pharyngeal", "prevertebral",
                  "tongue", "digastric", "thigh_medial", "thigh_posterior",
                  "levator_scapulae", "sterno_thyroid", "thyrohyoid", "transversospinalis", "muscle", "muskel"]
    combined_check = lowercase(raw_organ * " " * base_anat * " " * subloc * " " * details * " " * ltype)
    is_muscle = any(k -> occursin(k, combined_check), muscle_kws) || 
                (ont_entry !== nothing && (get(ont_entry, "is_muscle", false) || get(ont_entry, "anatomic_location", "") == "General Soft Tissue (Muscles, Subcutaneous)"))
    
    is_artifact = alt_hyp == "Technical Artifact" || certainty == "0" || ltype == "Technical Artifact" || is_muscle ||
                  occursin("artifact", combined_check) || occursin("artefakt", combined_check) || occursin("bladder", combined_check)

    # 5. Format human-readable clinical location
    clean_loc = ""
    clean_side = !isempty(side) ? side : (ont_entry !== nothing ? get(ont_entry, "side", "") : "")

    if ont_entry !== nothing
        detailed = get(ont_entry, "detailed", "")
        detailed = replace(detailed, r"\s*\(UBERON\)" => "")
        detailed = replace(detailed, r"\s*\(RadLex\)" => "")
        detailed = titlecase(detailed)

        if occursin("femur", lowercase(detailed))
            clean_loc = !isempty(clean_side) ? "$clean_side Femur" : "Femur"
        elseif occursin("sacral", lowercase(detailed)) || occursin("sacrum", lowercase(detailed))
            clean_loc = "Sacrum"
        elseif occursin("innominate", lowercase(detailed)) || occursin("hip", lowercase(raw_organ)) || occursin("ilium", lowercase(raw_organ))
            clean_loc = !isempty(clean_side) ? "$clean_side Pelvic Bone (Ilium)" : "Pelvic Bone (Ilium)"
        elseif occursin("ischium", lowercase(raw_organ))
            clean_loc = !isempty(clean_side) ? "$clean_side Ischium" : "Ischium"
        elseif occursin("pubis", lowercase(raw_organ))
            clean_loc = !isempty(clean_side) ? "$clean_side Pubic Bone" : "Pubic Bone"
        elseif occursin("thoracic vertebra", lowercase(detailed)) || startswith(lowercase(raw_organ), "vertebrae_t")
            v_num = match(r"T\d+", raw_organ, 1)
            num_str = v_num !== nothing ? v_num.match : uppercase(replace(raw_organ, r".*?(\d+).*" => s"\1"))
            clean_loc = "Thoracic Vertebra $num_str"
        elseif occursin("lumbar vertebra", lowercase(detailed)) || startswith(lowercase(raw_organ), "vertebrae_l")
            v_num = match(r"L\d+", raw_organ, 1)
            num_str = v_num !== nothing ? v_num.match : uppercase(replace(raw_organ, r".*?(\d+).*" => s"\1"))
            clean_loc = "Lumbar Vertebra $num_str"
        elseif occursin("cervical vertebra", lowercase(detailed)) || startswith(lowercase(raw_organ), "vertebrae_c")
            v_num = match(r"C\d+", raw_organ, 1)
            num_str = v_num !== nothing ? v_num.match : uppercase(replace(raw_organ, r".*?(\d+).*" => s"\1"))
            clean_loc = "Cervical Vertebra $num_str"
        elseif occursin("rib", lowercase(raw_organ))
            m_rib = match(r"rib_([a-z]+)_(\d+)", lowercase(raw_organ))
            if m_rib !== nothing
                r_side = titlecase(m_rib.captures[1])
                r_num = m_rib.captures[2]
                clean_loc = "$(_ordinal(parse(Int, r_num))) $r_side Rib"
            else
                clean_loc = !isempty(clean_side) ? "$clean_side Rib" : "Rib"
            end
        elseif occursin("scapula", lowercase(raw_organ))
            clean_loc = !isempty(clean_side) ? "$clean_side Scapula" : "Scapula"
        elseif occursin("skull", lowercase(raw_organ)) || occursin("calvarium", lowercase(raw_organ))
            clean_loc = "Skull / Calvarium"
        elseif occursin("mandible", lowercase(raw_organ))
            clean_loc = "Mandible"
        elseif occursin("clavicle", lowercase(raw_organ)) || occursin("clavicula", lowercase(raw_organ))
            clean_loc = !isempty(clean_side) ? "$clean_side Clavicle" : "Clavicle"
        elseif occursin("prostate", lowercase(raw_organ))
            clean_loc = "Prostate Gland"
        elseif occursin("quadriceps", lowercase(raw_organ))
            clean_loc = !isempty(clean_side) ? "$clean_side Quadriceps Femoris Muscle" : "Quadriceps Femoris Muscle"
        elseif occursin("sartorius", lowercase(raw_organ))
            clean_loc = !isempty(clean_side) ? "$clean_side Sartorius Muscle" : "Sartorius Muscle"
        elseif occursin("thigh_medial", lowercase(raw_organ))
            clean_loc = !isempty(clean_side) ? "$clean_side Thigh Medial Muscle Compartment" : "Thigh Medial Muscle Compartment"
        elseif occursin("gluteus", lowercase(raw_organ))
            clean_loc = !isempty(clean_side) ? "$clean_side Gluteal Muscle" : "Gluteal Muscle"
        elseif occursin("aorta", lowercase(raw_organ))
            clean_loc = "Aorta (Blood Pool / Vascular)"
        else
            clean_loc = detailed
            if !isempty(clean_side) && !occursin(lowercase(clean_side), lowercase(clean_loc))
                clean_loc = "$clean_side $clean_loc"
            end
        end
    elseif !isempty(raw_organ)
        clean_loc = titlecase(replace(raw_organ, "_" => " "))
        if !isempty(clean_side) && !occursin(lowercase(clean_side), lowercase(clean_loc))
            clean_loc = "$clean_side $clean_loc"
        end
    else
        clean_loc = if occursin("prostate", lowercase(ltype))
            "Prostate Gland"
        elseif occursin("bone", lowercase(ltype))
            "Osseous Skeleton Focus"
        elseif occursin("lymph", lowercase(ltype))
            "Pelvic Lymph Node Station"
        else
            "Soft Tissue / Visceral Focus"
        end
        if !isempty(clean_side)
            clean_loc = "$clean_side $clean_loc"
        end
    end

    # Only append sub-details if they were manually entered by the user
    # (not auto-filled from ontology — we don't have sub-organ imaging resolution)
    json_default_subloc = ""
    if LMW !== nothing
        try
            entry = LMW.lookup_anatomy(raw_organ)
            json_default_subloc = entry !== nothing ? string(get(entry, "anatomical_sublocation", "")) : ""
        catch; end
    end

    if !isempty(subloc) && subloc != json_default_subloc &&
       subloc != "N/A (General Organ)" && subloc != "Medullary Cavity (Intramedullary/Marrow)" &&
       !occursin(lowercase(subloc), lowercase(clean_loc))
        clean_loc *= " ($subloc)"
    elseif !isempty(details) && !occursin(lowercase(details), lowercase(clean_loc))
        clean_loc *= " ($details)"
    end

    return (clean_loc, is_artifact, is_muscle, raw_organ)
end

resolve_anatomical_location(lid::Integer, state::Any, tp_idx::Integer=0, _MEH=nothing, LMW=nothing)::Tuple{String, Bool, Bool, String} =
    resolve_anatomical_location(Int(lid), state isa AbstractDict ? state : nothing, Int(tp_idx), _MEH, LMW)


# ── Automatic Report Aggregation ─────────────────────────────────────────────
"""
    build_epsma_data(tp_idx::Int; lang="EN") -> EPSMAReport

Aggregates all metadata, background references, quantitative metrics, and longitudinal trajectories
into a structured E-PSMA report object conforming to the EANM v1.0 consensus.
"""
function build_epsma_data(tp_idx::Int; lang::String = "EN")::EPSMAReport
    _MEH = _get_meh()
    LMW = _get_lmw()
    
    patient_id = _MEH !== nothing ? _MEH.patient_id[] : "PAT_001"
    if isempty(patient_id)
        patient_id = "PAT_001"
    end
    tp_label = _MEH !== nothing ? get(_MEH.tp_labels, tp_idx, "PET TP $tp_idx") : "PET TP $tp_idx"
    modality = _MEH !== nothing ? get(_MEH.tp_modalities, tp_idx, "PET/CT") : "PET/CT"
    desc_de = _MEH !== nothing ? get(_MEH.tp_descriptions, tp_idx, "") : ""
    desc_en = _MEH !== nothing ? get(_MEH.tp_english_descriptions, tp_idx, "") : ""
    
    # Background SUV
    bg = if LMW !== nothing
        try LMW.get_background_suvs(tp_idx) catch; Dict{String, Float32}("liver" => 2.3f0, "blood" => 1.5f0, "parotid" => 1.8f0) end
    else
        Dict{String, Float32}("liver" => 2.3f0, "blood" => 1.5f0, "parotid" => 1.8f0)
    end
    l_suv = get(bg, "liver", 2.3f0)
    b_suv = get(bg, "blood", 1.5f0)
    p_suv = get(bg, "parotid", 1.8f0)
    
    # Lesions at this TP
    l_ids = if LMW !== nothing
        try LMW.get_mask_ids(tp_idx) catch; Int[] end
    else
        Int[]
    end
    # Always merge with organ_mapping (contains all lesions from preprocessing)
    if _MEH !== nothing && !isempty(_MEH.global_organ_mapping[])
        om_ids = collect(keys(_MEH.global_organ_mapping[]))
        l_ids = sort(unique(vcat(l_ids, om_ids)))
    end
    # Also merge with tp_segment_names (contains segment names from scene_hierarchy)
    if _MEH !== nothing && haskey(_MEH.tp_segment_names, tp_idx)
        seg_ids = collect(keys(_MEH.tp_segment_names[tp_idx]))
        l_ids = sort(unique(vcat(l_ids, seg_ids)))
    end
    @info "[E-PSMA] Discovered $(length(l_ids)) lesion IDs for TP $tp_idx: $l_ids"
    
    db = if LMW !== nothing
        try LMW.get_active_lesion_db() catch; Dict{String, Any}() end
    else
        Dict{String, Any}()
    end
    
    synoptic_rows = EPSMALesionRow[]
    artifact_rows = EPSMALesionRow[]
    
    has_t = false
    highest_t = "miT0"
    has_n = false
    has_m1a = false
    has_m1b = false
    has_m1c = false
    
    # Subregional narrative containers
    prostate_gland_en = String[]; prostate_gland_de = String[]
    prostate_bed_en   = String[]; prostate_bed_de   = String[]
    
    pelvic_ln_en      = String[]; pelvic_ln_de      = String[]
    distant_ln_en     = String[]; distant_ln_de     = String[]
    
    bone_axial_en     = String[]; bone_axial_de     = String[]
    bone_append_en    = String[]; bone_append_de    = String[]
    
    visceral_pca_en   = String[]; visceral_pca_de   = String[]
    artifact_lines_en = String[]; artifact_lines_de = String[]
    
    all_prior_tp_indices = _MEH !== nothing ? filter(t -> t < tp_idx, collect(keys(_MEH.tp_labels))) : Int[]
    has_prior_studies = !isempty(all_prior_tp_indices)
    
    for lid in l_ids
        state = if LMW !== nothing && !isempty(db)
            try LMW.get_lesion_state(db, string(lid)) catch; Dict{String, String}() end
        else
            Dict{String, String}()
        end
        
        # 1. Resolve clean anatomical location & artifact status (NO generic "Lesion $lid"!)
        (loc_full, is_artifact, is_muscle, raw_organ_name) = resolve_anatomical_location(lid, state, tp_idx, _MEH, LMW)
        sublocation = get(state, "Anatomical Details", "")
        ltype = get(state, "LesionType", "Lesion")
        comment = get(state, "Comment", "")
        
        # 1b. Fetch original RTOG/clinical segment name for classification hints
        seg_name = ""
        if _MEH !== nothing && isdefined(_MEH, :tp_segment_names)
            tp_segs = get(_MEH.tp_segment_names, tp_idx, Dict{Int,String}())
            seg_name = get(tp_segs, lid, "")
        end
        
        # 1c. Auto-infer LesionType from segment name if current type is generic
        if ltype in ("Lesion", "Organ Meta", "") && !isempty(seg_name)
            seg_lo = lowercase(seg_name)
            if occursin("knochen", seg_lo) || occursin("bone", seg_lo)
                ltype = "Bone Meta"
            elseif occursin("lymph", seg_lo) || occursin("knoten", seg_lo) || occursin("node", seg_lo)
                ltype = "Lymph Node"
            elseif occursin("prostat", seg_lo)
                ltype = "Prostate"
            end
        end
        
        # 1d. Auto-infer LesionType from atlas organ name for known LN stations
        # TotalSegmentator has no lymph node labels, but lesions near iliac vessels
        # or the aorta are almost certainly lymph node metastases (users don't annotate
        # normal blood pool as lesions).
        if ltype in ("Lesion", "Organ Meta", "") && !is_artifact
            organ_lo = lowercase(raw_organ_name)
            if occursin("iliac", organ_lo) && !occursin("crest", organ_lo)
                ltype = "Lymph Node"  # Pelvic LN station (near iliac vessels)
                @info "[E-PSMA] Auto-inferred Lymph Node for lesion $lid (atlas: '$raw_organ_name')"
            elseif occursin("aorta", organ_lo) || occursin("vena_cava", organ_lo) || occursin("caval", organ_lo)
                ltype = "Lymph Node"  # Para-aortic / retroperitoneal LN
                @info "[E-PSMA] Auto-inferred Lymph Node for lesion $lid (atlas: '$raw_organ_name')"
            end
        end
        
        # 1e. Override artifact/muscle flag when segment name explicitly identifies a lymph node
        # The atlas centroid may land on adjacent muscle/bone (e.g., iliopsoas), but the
        # clinical annotation from Slicer says it IS a lymph node — trust the annotation.
        if ltype == "Lymph Node" && (is_artifact || is_muscle)
            @info "[E-PSMA] Overriding artifact flag for lesion $lid: seg_name='$seg_name' says Lymph Node but atlas='$raw_organ_name' triggered muscle/artifact"
            is_artifact = false
            is_muscle = false
        end
        
        # 1f. Derive proper lymph node location name from segment name when available
        # The atlas gives names like "iliac_artery_right" but the clinical annotation
        # from Slicer says "LN01 Lymphknoten iliaca externa links" — use that info.
        if ltype == "Lymph Node" && !isempty(seg_name)
            seg_lo = lowercase(seg_name)
            if occursin("inguinal", seg_lo)
                side_str = occursin("rechts", seg_lo) || occursin("right", seg_lo) ? "Right" :
                           occursin("links", seg_lo) || occursin("left", seg_lo) ? "Left" : ""
                loc_full = isempty(side_str) ? "Inguinal Lymph Node" : "$side_str Inguinal Lymph Node"
            elseif occursin("iliaca externa", seg_lo) || occursin("external iliac", seg_lo)
                side_str = occursin("rechts", seg_lo) || occursin("right", seg_lo) ? "Right" :
                           occursin("links", seg_lo) || occursin("left", seg_lo) ? "Left" : ""
                loc_full = isempty(side_str) ? "External Iliac Lymph Node" : "$side_str External Iliac Lymph Node"
            elseif occursin("iliaca interna", seg_lo) || occursin("internal iliac", seg_lo)
                side_str = occursin("rechts", seg_lo) || occursin("right", seg_lo) ? "Right" :
                           occursin("links", seg_lo) || occursin("left", seg_lo) ? "Left" : ""
                loc_full = isempty(side_str) ? "Internal Iliac Lymph Node" : "$side_str Internal Iliac Lymph Node"
            elseif occursin("obturator", seg_lo) || occursin("obturat", seg_lo)
                side_str = occursin("rechts", seg_lo) || occursin("right", seg_lo) ? "Right" :
                           occursin("links", seg_lo) || occursin("left", seg_lo) ? "Left" : ""
                loc_full = isempty(side_str) ? "Obturator Lymph Node" : "$side_str Obturator Lymph Node"
            elseif occursin("retroperiton", seg_lo) || occursin("paraaort", seg_lo) || occursin("para-aort", seg_lo)
                loc_full = "Retroperitoneal / Para-Aortic Lymph Node"
            elseif occursin("mediastin", seg_lo)
                loc_full = "Mediastinal Lymph Node"
            elseif occursin("supraclavicul", seg_lo)
                side_str = occursin("rechts", seg_lo) || occursin("right", seg_lo) ? "Right" :
                           occursin("links", seg_lo) || occursin("left", seg_lo) ? "Left" : ""
                loc_full = isempty(side_str) ? "Supraclavicular Lymph Node" : "$side_str Supraclavicular Lymph Node"
            elseif occursin("lymphknoten", seg_lo) || occursin("lymph node", seg_lo)
                # Generic lymph node from segment name — use atlas side info
                side_str = occursin("rechts", seg_lo) || occursin("right", seg_lo) ? "Right" :
                           occursin("links", seg_lo) || occursin("left", seg_lo) ? "Left" : ""
                loc_full = isempty(side_str) ? "Pelvic Lymph Node" : "$side_str Pelvic Lymph Node"
            end
        end
        
        # 1g. Fallback location name for atlas-inferred lymph nodes (no clinical segment name)
        # When auto-inferred from vascular atlas names (step 1d) with generic segment names
        if ltype == "Lymph Node" && (isempty(seg_name) || !occursin("lymph", lowercase(seg_name)) && !occursin("knoten", lowercase(seg_name)))
            organ_lo = lowercase(raw_organ_name)
            if occursin("aorta", organ_lo)
                loc_full = "Para-Aortic Lymph Node"
            elseif occursin("iliac", organ_lo) && !occursin("crest", organ_lo)
                side_str = occursin("right", organ_lo) ? "Right" : occursin("left", organ_lo) ? "Left" : ""
                loc_full = isempty(side_str) ? "Pelvic Iliac Lymph Node" : "$side_str Pelvic Iliac Lymph Node"
            elseif occursin("vena_cava", organ_lo) || occursin("caval", organ_lo)
                loc_full = "Paracaval Lymph Node"
            end
        end
        
        # 2. Volume & Diameter
        vol_cc = 0.0
        diam_mm = 0.0
        if LMW !== nothing
            try
                v_res = LMW.compute_lesion_volume(lid, tp_idx)
                vol_cc = get(v_res, "volume_cc", 0.0)
                diam_mm = get(v_res, "diameter_mm", 0.0)
            catch; end
        end
        
        # 3. SUVmax
        suv_max = 0.0f0
        if LMW !== nothing
            try suv_max = LMW._get_suv_max(lid, tp_idx) catch; end
        end
        
        # 4. Visual score & Confidence
        (psma_v_str, psma_v_num) = compute_psma_v(suv_max, l_suv, b_suv, p_suv)
        is_bone_lesion = !is_artifact && (
            occursin("bone", lowercase(ltype)) || occursin("knochen", lowercase(ltype)) ||
            occursin("femur", lowercase(loc_full)) || occursin("pelvic", lowercase(loc_full)) ||
            occursin("vertebra", lowercase(loc_full)) || occursin("rib", lowercase(loc_full)) ||
            occursin("sacrum", lowercase(loc_full))
        )
        conf = compute_reader_confidence(loc_full, suv_max, l_suv, is_bone_lesion)
        
        # 5. Longitudinal match
        recip_status = "BASELINE"
        delta_str = ""
        is_new = false
        if LMW !== nothing
            try
                m_res = LMW.compute_match_analysis(lid, tp_idx)
                if m_res !== nothing && m_res.recip != "N/A (baseline)"
                    recip_status = m_res.recip
                    sgn_v = m_res.vol_delta_pct >= 0 ? "+" : ""
                    sgn_s = m_res.suv_delta_abs >= 0 ? "+" : ""
                    delta_str = "ΔVol: $(sgn_v)$(round(m_res.vol_delta_pct, digits=1))% | ΔSUV: $(sgn_s)$(round(m_res.suv_delta_pct, digits=1))%"
                elseif has_prior_studies
                    recip_status = "NEW LESION"
                    is_new = true
                    delta_str = "New Hypermetabolic Lesion"
                end
            catch; end
        end
        
        size_display = diam_mm > 0 ? "$(round(diam_mm, digits=1)) mm ($(round(vol_cc, digits=2)) cc)" : "$(round(vol_cc, digits=2)) cc"
        q_display = "SUVmax $(round(suv_max, digits=1))"
        
        # 6. Separate PCa metastases from Technical Artifacts / False Positives
        if is_artifact
            mitnm = "Artifact"
            row = EPSMALesionRow(
                id = lid,
                location = loc_full,
                mitnm = "Artifact",
                size_str = size_display,
                volume_cc = vol_cc,
                diameter_mm = diam_mm,
                num_lesions = 1,
                psma_q = q_display,
                suv_max = suv_max,
                psma_v = psma_v_str,
                psma_v_num = psma_v_num,
                reader_confidence = 1, # Benign / Artifact
                recip_status = "ARTIFACT",
                delta_str = "Technical Artifact / False Positive",
                is_new = false,
                comment = !isempty(comment) ? comment : (is_muscle ? "Muscular uptake without CT substrate (false positive)" : "Artifact")
            )
            push!(artifact_rows, row)
            push!(artifact_lines_en, "• $loc_full: Presumed technical artifact / muscular uptake (Certainty 0, $size_display, $q_display). Excluded from PCa staging.")
            push!(artifact_lines_de, "• $loc_full: Vermutliches technisches Artefakt / Muskelmehranreicherung (Sicherheit 0, $size_display, $q_display). Nicht als PCa-Metastase gewertet.")
        else
            # Valid Prostate Cancer Lesion!
            mitnm = classify_mitnm(loc_full, sublocation, ltype, ltype, seg_name)
            @info "[E-PSMA] Lesion $lid: loc='$loc_full' ltype='$ltype' seg='$(seg_name)' → $mitnm"
            if startswith(mitnm, "miT")
                has_t = true
                highest_t = mitnm
            elseif mitnm == "miN1"
                has_n = true
            elseif mitnm == "miM1a"
                has_m1a = true
            elseif mitnm == "miM1b"
                has_m1b = true
            elseif mitnm == "miM1c"
                has_m1c = true
            end
            
            row = EPSMALesionRow(
                id = lid,
                location = loc_full,
                mitnm = mitnm,
                size_str = size_display,
                volume_cc = vol_cc,
                diameter_mm = diam_mm,
                num_lesions = 1,
                psma_q = q_display,
                suv_max = suv_max,
                psma_v = psma_v_str,
                psma_v_num = psma_v_num,
                reader_confidence = conf,
                recip_status = recip_status,
                delta_str = delta_str,
                is_new = is_new,
                comment = comment
            )
            push!(synoptic_rows, row)
            
            finding_line_en = "• $loc_full: $size_display, $q_display ($psma_v_str, Conf $conf). $(is_new ? "*** NEW LESION ***" : delta_str)"
            finding_line_de = "• $loc_full: $size_display, $q_display ($psma_v_str, Konfidenz $conf). $(is_new ? "*** NEUE LÄSION ***" : delta_str)"
            
            # Route finding line into Delphi consensus subregions
            if mitnm == "miTr"
                push!(prostate_bed_en, finding_line_en)
                push!(prostate_bed_de, finding_line_de)
            elseif startswith(mitnm, "miT")
                push!(prostate_gland_en, finding_line_en)
                push!(prostate_gland_de, finding_line_de)
            elseif mitnm == "miN1"
                push!(pelvic_ln_en, finding_line_en)
                push!(pelvic_ln_de, finding_line_de)
            elseif mitnm == "miM1a"
                push!(distant_ln_en, finding_line_en)
                push!(distant_ln_de, finding_line_de)
            elseif mitnm == "miM1b"
                is_append = occursin("femur", lowercase(loc_full)) || occursin("humerus", lowercase(loc_full)) ||
                             occursin("tibia", lowercase(loc_full)) || occursin("scapula", lowercase(loc_full)) ||
                             occursin("radius", lowercase(loc_full)) || occursin("ulna", lowercase(loc_full))
                if is_append
                    push!(bone_append_en, finding_line_en)
                    push!(bone_append_de, finding_line_de)
                else
                    push!(bone_axial_en, finding_line_en)
                    push!(bone_axial_de, finding_line_de)
                end
            else
                push!(visceral_pca_en, finding_line_en)
                push!(visceral_pca_de, finding_line_de)
            end
        end
    end
    
    # Synthesize Final miTNM (ONLY from valid PCa lesions)
    m_part = if has_m1c
        "miM1c"
    elseif has_m1b
        "miM1b"
    elseif has_m1a
        "miM1a"
    else
        "miM0"
    end
    n_part = has_n ? "miN1" : "miN0"
    t_part = has_t ? highest_t : "miT0"
    final_mitnm = "$t_part $n_part $m_part"
    
    # Cumulative TMTV (ONLY sums true malignant PCa lesions!)
    total_tmtv_cc = round(sum(r.volume_cc for r in synoptic_rows; init=0.0), digits=2)
    
    # Overall RECIP (ONLY on PCa lesions)
    overall_recip = if !has_prior_studies
        "BASELINE (Initial Staging / Erststaging)"
    elseif any(r -> r.is_new, synoptic_rows)
        "PROGRESSIVE METABOLIC DISEASE (PMD / RECIP-PD) - New Lesion Emerged"
    elseif any(r -> r.recip_status == "RECIP-PD", synoptic_rows)
        "PROGRESSIVE METABOLIC DISEASE (PMD / RECIP-PD) - Lesion Growth"
    elseif !isempty(synoptic_rows) && all(r -> r.recip_status == "RECIP-CR" || r.volume_cc < 0.001, synoptic_rows)
        "COMPLETE METABOLIC RESPONSE (CMR / RECIP-CR)"
    elseif any(r -> r.recip_status == "RECIP-PR", synoptic_rows)
        "PARTIAL METABOLIC RESPONSE (PMR / RECIP-PR)"
    else
        "STABLE METABOLIC DISEASE (SMD / RECIP-SD)"
    end
    
    # Format Subregional Narrative Sections
    # ── 1. Prostate & Prostate Bed ──
    prostate_txt_en = if isempty(prostate_gland_en) && isempty(prostate_bed_en)
        "• Prostate Gland & Bed: No abnormal focal PSMA uptake detected (miT0)."
    else
        parts = String[]
        if !isempty(prostate_gland_en)
            push!(parts, ">> Subregion Prostate Gland (Primary Tumor):\n" * join(prostate_gland_en, "\n"))
        end
        if !isempty(prostate_bed_en)
            push!(parts, ">> Subregion Prostate Bed (Local Recurrence - miTr):\n" * join(prostate_bed_en, "\n"))
        end
        join(parts, "\n\n")
    end
    prostate_txt_de = if isempty(prostate_gland_de) && isempty(prostate_bed_de)
        "• Prostata & Prostatabett: Keine pathologische fokale PSMA-Mehranreicherung nachweisbar (miT0)."
    else
        parts = String[]
        if !isempty(prostate_gland_de)
            push!(parts, ">> Subregion Prostatadrüse (Primärtumor):\n" * join(prostate_gland_de, "\n"))
        end
        if !isempty(prostate_bed_de)
            push!(parts, ">> Subregion Prostatabett (Lokalrezidiv - miTr):\n" * join(prostate_bed_de, "\n"))
        end
        join(parts, "\n\n")
    end

    # ── 2. Lymph Nodes (Pelvic vs Extra-pelvic) ──
    lymph_parts_en = String[]
    if !isempty(pelvic_ln_en)
        push!(lymph_parts_en, ">> Regional Pelvic Lymph Nodes (miN1):\n" * join(pelvic_ln_en, "\n"))
    else
        push!(lymph_parts_en, ">> Regional Pelvic Lymph Nodes (miN0):\n• No suspicious PSMA-avid pelvic lymph nodes identified.")
    end
    if !isempty(distant_ln_en)
        push!(lymph_parts_en, ">> Extra-Pelvic Distant Lymph Nodes (miM1a):\n" * join(distant_ln_en, "\n"))
    else
        push!(lymph_parts_en, ">> Extra-Pelvic Distant Lymph Nodes (miM0):\n• No suspicious extra-pelvic or retroperitoneal lymph node metastases.")
    end
    lymph_txt_en = join(lymph_parts_en, "\n\n")

    lymph_parts_de = String[]
    if !isempty(pelvic_ln_de)
        push!(lymph_parts_de, ">> Regionale pelvine Lymphknoten (miN1):\n" * join(pelvic_ln_de, "\n"))
    else
        push!(lymph_parts_de, ">> Regionale pelvine Lymphknoten (miN0):\n• Keine suspekten PSMA-positiven pelvinen Lymphknotenmetastasen.")
    end
    if !isempty(distant_ln_de)
        push!(lymph_parts_de, ">> Extra-pelvine Fern-Lymphknoten (miM1a):\n" * join(distant_ln_de, "\n"))
    else
        push!(lymph_parts_de, ">> Extra-pelvine Fern-Lymphknoten (miM0):\n• Keine suspekten extra-pelvinen oder retroperitonealen Lymphknotenmetastasen.")
    end
    lymph_txt_de = join(lymph_parts_de, "\n\n")

    # ── 3. Osseous Skeleton (Axial vs Appendicular + Burden) ──
    n_bone = length(bone_axial_en) + length(bone_append_en)
    burden_en = n_bone == 0 ? "No osseous metastases" : (n_bone == 1 ? "Solitary bone lesion" : (n_bone <= 5 ? "Oligometastatic skeleton ($n_bone lesions)" : "Polymetastatic / disseminated skeleton ($n_bone lesions)"))
    burden_de = n_bone == 0 ? "Kein Nachweis von Knochenmetastasen" : (n_bone == 1 ? "Solitäre Knochenmetastase" : (n_bone <= 5 ? "Oligometastasiertes Skelett ($n_bone Herde)" : "Polymetastasiertes / disseminiertes Skelett ($n_bone Herde)"))

    bone_parts_en = String["[Skeletal Pattern: $burden_en]"]
    if n_bone == 0
        push!(bone_parts_en, "• No focal abnormal PSMA uptake in axial or appendicular skeleton (miM0).")
    else
        if !isempty(bone_axial_en)
            push!(bone_parts_en, ">> Axial Skeleton (Spine, Pelvis, Thorax, Skull - miM1b):\n" * join(bone_axial_en, "\n"))
        else
            push!(bone_parts_en, ">> Axial Skeleton (miM0):\n• No focal abnormal PSMA uptake in spine, pelvis, or thorax.")
        end
        if !isempty(bone_append_en)
            push!(bone_parts_en, ">> Appendicular Skeleton (Limbs, Scapulae - miM1b):\n" * join(bone_append_en, "\n"))
        else
            push!(bone_parts_en, ">> Appendicular Skeleton (miM0):\n• No focal abnormal PSMA uptake in extremities.")
        end
    end
    bone_txt_en = join(bone_parts_en, "\n\n")

    bone_parts_de = String["[Befallsmuster Skelett: $burden_de]"]
    if n_bone == 0
        push!(bone_parts_de, "• Kein Nachweis fokaler pathologischer PSMA-Mehranreicherungen im Skelettsystem (miM0).")
    else
        if !isempty(bone_axial_de)
            push!(bone_parts_de, ">> Achsenskelett (Wirbelsäule, Becken, Thorax, Schädel - miM1b):\n" * join(bone_axial_de, "\n"))
        else
            push!(bone_parts_de, ">> Achsenskelett (miM0):\n• Keine fokale pathologische Mehranreicherung in Wirbelsäule, Becken oder Thorax.")
        end
        if !isempty(bone_append_de)
            push!(bone_parts_de, ">> Extremitätenskelett (Gliedmaßen, Skapula - miM1b):\n" * join(bone_append_de, "\n"))
        else
            push!(bone_parts_de, ">> Extremitätenskelett (miM0):\n• Keine fokale pathologische Mehranreicherung in den Extremitäten.")
        end
    end
    bone_txt_de = join(bone_parts_de, "\n\n")

    # ── 4. Visceral & Incidental / Technical Artifacts ──
    visc_parts_en = String[]
    if !isempty(visceral_pca_en)
        push!(visc_parts_en, ">> Non-Nodal Visceral PCa Metastases (miM1c):\n" * join(visceral_pca_en, "\n"))
    else
        push!(visc_parts_en, ">> Non-Nodal Visceral Organs (miM0):\n• No PSMA-avid visceral metastases (liver, lungs, brain unremarkable).")
    end
    if !isempty(artifact_lines_en)
        push!(visc_parts_en, ">> Incidental Findings & Technical Artifacts (Certainty 0 - Excluded from PCa staging):\n" * join(artifact_lines_en, "\n"))
    else
        push!(visc_parts_en, ">> Incidental Findings & Artifacts:\n• No significant non-prostatic incidental findings or artifacts.")
    end
    visc_txt_en = join(visc_parts_en, "\n\n")

    visc_parts_de = String[]
    if !isempty(visceral_pca_de)
        push!(visc_parts_de, ">> Nicht-nodale viszerale PCa-Metastasen (miM1c):\n" * join(visceral_pca_de, "\n"))
    else
        push!(visc_parts_de, ">> Nicht-nodale viszerale Organe (miM0):\n• Keine PSMA-positiven Organmetastasen (Leber, Lunge, ZNS unauffällig).")
    end
    if !isempty(artifact_lines_de)
        push!(visc_parts_de, ">> Nebenbefunde & Technische Artefakte (Sicherheit 0 - Nicht als PCa-Metastase gewertet):\n" * join(artifact_lines_de, "\n"))
    else
        push!(visc_parts_de, ">> Nebenbefunde & Artefakte:\n• Keine signifikanten nicht-prostatabezogenen Nebenbefunde oder Artefakte.")
    end
    visc_txt_de = join(visc_parts_de, "\n\n")

    # Default Conclusions
    concl_en = if isempty(synoptic_rows)
        "No definitive evidence of PSMA-avid local recurrence or metastatic prostate cancer (final miTNM: miT0 miN0 miM0)."
    else
        "1. Evidence of PSMA-avid prostate cancer lesions as detailed above. Final molecular imaging staging: $final_mitnm.\n2. Total Metabolic Tumor Volume (TMTV): $(total_tmtv_cc) cc. Overall Response Status: $overall_recip.\n3. Recommend multidisciplinary tumor board review."
    end
    
    concl_de = if isempty(synoptic_rows)
        "Kein Anhalt für ein PSMA-positives Lokalrezidiv oder metastasiertes Prostatakarzinom (finales miTNM: miT0 miN0 miM0)."
    else
        "1. Nachweis PSMA-positiver Prostatakarzinom-Läsionen wie oben detailliert beschrieben. Finales molekulares Tumorstadium: $final_mitnm.\n2. Gesamt-Tumorvolumen (TMTV): $(total_tmtv_cc) ml. Verlauf/Therapieansprechen: $overall_recip.\n3. Interdisziplinäre Tumorkonferenz empfohlen."
    end
    
    # Query clinical information entered in main panel
    clin_info = Dict{String, Any}()
    if haskey(db, "Clinical_Info_TP$(tp_idx)")
        clin_info = db["Clinical_Info_TP$(tp_idx)"]
    elseif haskey(db, "Patient_Clinical_Info")
        clin_info = db["Patient_Clinical_Info"]
    end
    
    c_ind = get(clin_info, "Indication", "")
    c_psa = get(clin_info, "PSA", "")
    c_psa_dt = get(clin_info, "PSADoublingTime", "")
    c_gleason = get(clin_info, "Gleason", "")
    c_tnm = get(clin_info, "TNMStage", "")
    c_prior = get(clin_info, "PriorTherapies", "")
    c_notes = get(clin_info, "ClinicalNotes", "")
    
    history_en_parts = String[]
    history_de_parts = String[]
    if !isempty(c_ind)
        push!(history_en_parts, "• Clinical Indication: $c_ind")
        push!(history_de_parts, "• Klinische Indikation: $c_ind")
    end
    if !isempty(c_psa)
        psa_s = isempty(c_psa_dt) ? "$c_psa ng/mL" : "$c_psa ng/mL ($c_psa_dt)"
        push!(history_en_parts, "• Current PSA: $psa_s")
        push!(history_de_parts, "• Aktueller PSA-Wert: $psa_s")
    end
    if !isempty(c_gleason) || !isempty(c_tnm)
        hist_s_en = !isempty(c_gleason) ? (!isempty(c_tnm) ? "$c_gleason, Stage $c_tnm" : c_gleason) : "Stage $c_tnm"
        hist_s_de = !isempty(c_gleason) ? (!isempty(c_tnm) ? "$c_gleason, Stadium $c_tnm" : c_gleason) : "Stadium $c_tnm"
        push!(history_en_parts, "• Histology & Initial Staging: $hist_s_en")
        push!(history_de_parts, "• Histologie & Initiales Stadium: $hist_s_de")
    end
    if !isempty(c_prior)
        push!(history_en_parts, "• Prior Therapies: $c_prior")
        push!(history_de_parts, "• Vortherapien: $c_prior")
    end
    if !isempty(c_notes)
        push!(history_en_parts, "• Clinical History / Anamnese: $c_notes")
        push!(history_de_parts, "• Klinische Anamnese: $c_notes")
    end
    
    full_history_en = if !isempty(history_en_parts)
        join(history_en_parts, "\n") * (!isempty(desc_en) ? "\n\n" * desc_en : "")
    else
        !isempty(desc_en) ? desc_en : (!isempty(desc_de) ? desc_de : "")
    end
    
    full_history_de = if !isempty(history_de_parts)
        join(history_de_parts, "\n") * (!isempty(desc_de) ? "\n\n" * desc_de : "")
    else
        !isempty(desc_de) ? desc_de : ""
    end

    return EPSMAReport(
        patient_id = patient_id,
        default_lang = lang,
        tp_index = tp_idx,
        tp_label = tp_label,
        modality = modality,
        study_date = Dates.format(Dates.today(), "yyyy-mm-dd"),
        tech_params = EPSMATechnicalParams(),
        background_suv = bg,
        biodistribution_text_en = "(Please enter biodistribution statement.)",
        biodistribution_text_de = "(Bitte Biodistributionsbeschreibung eingeben.)",
        tech_narrative_en = "",
        tech_narrative_de = "",
        history_text_en = full_history_en,
        history_text_de = full_history_de,
        findings_prostate_en = prostate_txt_en,
        findings_prostate_de = prostate_txt_de,
        findings_lymph_en = lymph_txt_en,
        findings_lymph_de = lymph_txt_de,
        findings_bone_en = bone_txt_en,
        findings_bone_de = bone_txt_de,
        findings_visceral_en = visc_txt_en,
        findings_visceral_de = visc_txt_de,
        findings_artifacts_en = join(artifact_lines_en, "\n"),
        findings_artifacts_de = join(artifact_lines_de, "\n"),
        synoptic_rows = synoptic_rows,
        artifact_rows = artifact_rows,
        skeletal_burden_en = burden_en,
        skeletal_burden_de = burden_de,
        final_mitnm = final_mitnm,
        tmtv_cc = total_tmtv_cc,
        tmtv_delta_pct = 0.0,
        overall_recip = overall_recip,
        conclusion_en = concl_en,
        conclusion_de = concl_de
    )
end

# ── Report Caching (auto-build on startup, invalidate on changes) ────────────
const _report_cache = Dict{Int, EPSMAReport}()

"""
    prebuild_reports!()

Pre-build E-PSMA reports for all TPs that have mask data.
Call after startup when TP data and segment names are populated.
"""
function prebuild_reports!()
    _MEH = _get_meh()
    if _MEH === nothing
        @warn "[E-PSMA] Cannot pre-build reports: MakieEventHandlers not available"
        return
    end
    for tp in sort(collect(keys(_MEH.tp_labels)))
        try
            _report_cache[tp] = build_epsma_data(tp)
            n_rows = length(_report_cache[tp].synoptic_rows)
            n_art = length(_report_cache[tp].artifact_rows)
            @info "[E-PSMA] Pre-built report for TP $tp: $n_rows PCa lesions, $n_art artifacts, miTNM=$(_report_cache[tp].final_mitnm)"
        catch e
            @warn "[E-PSMA] Failed to pre-build report for TP $tp" exception=(e, catch_backtrace())
        end
    end
end

"""
    get_or_build_report(tp_idx::Int; lang="EN") -> EPSMAReport

Return cached report if available, otherwise build and cache it.
"""
function get_or_build_report(tp_idx::Int; lang::String="EN")::EPSMAReport
    if !haskey(_report_cache, tp_idx)
        _report_cache[tp_idx] = build_epsma_data(tp_idx; lang=lang)
    end
    return _report_cache[tp_idx]
end

"""
    invalidate_report!(tp_idx::Int)

Invalidate cached report for a TP. Call after metadata changes, lesion add/delete, etc.
"""
function invalidate_report!(tp_idx::Int)
    if haskey(_report_cache, tp_idx)
        delete!(_report_cache, tp_idx)
        @info "[E-PSMA] Invalidated cached report for TP $tp_idx"
    end
end

# ── Serialization to Dict ────────────────────────────────────────────────────
function to_dict(rep::EPSMAReport)::Dict{String, Any}
    return Dict{String, Any}(
        "patient_id" => rep.patient_id,
        "tp_index" => rep.tp_index,
        "tp_label" => rep.tp_label,
        "modality" => rep.modality,
        "study_date" => rep.study_date,
        "tech_params" => Dict{String, Any}(
            "radiotracer" => rep.tech_params.radiotracer,
            "injected_activity" => rep.tech_params.injected_activity,
            "uptake_time" => rep.tech_params.uptake_time,
            "acquisition_type" => rep.tech_params.acquisition_type,
            "ct_protocol" => rep.tech_params.ct_protocol,
            "contrast" => rep.tech_params.contrast,
            "diuretic" => rep.tech_params.diuretic
        ),
        "background_suv" => Dict{String, Float64}(
            "liver" => Float64(get(rep.background_suv, "liver", 2.3f0)),
            "blood" => Float64(get(rep.background_suv, "blood", 1.5f0)),
            "parotid" => Float64(get(rep.background_suv, "parotid", 1.8f0))
        ),
        "biodistribution_text_en" => rep.biodistribution_text_en,
        "biodistribution_text_de" => rep.biodistribution_text_de,
        "tech_narrative_en" => rep.tech_narrative_en,
        "tech_narrative_de" => rep.tech_narrative_de,
        "history_text_en" => rep.history_text_en,
        "history_text_de" => rep.history_text_de,
        "findings_prostate_en" => rep.findings_prostate_en,
        "findings_prostate_de" => rep.findings_prostate_de,
        "findings_lymph_en" => rep.findings_lymph_en,
        "findings_lymph_de" => rep.findings_lymph_de,
        "findings_bone_en" => rep.findings_bone_en,
        "findings_bone_de" => rep.findings_bone_de,
        "findings_visceral_en" => rep.findings_visceral_en,
        "findings_visceral_de" => rep.findings_visceral_de,
        "findings_artifacts_en" => rep.findings_artifacts_en,
        "findings_artifacts_de" => rep.findings_artifacts_de,
        "skeletal_burden_en" => rep.skeletal_burden_en,
        "skeletal_burden_de" => rep.skeletal_burden_de,
        "synoptic_rows" => [
            Dict{String, Any}(
                "id" => r.id,
                "location" => r.location,
                "mitnm" => r.mitnm,
                "size_str" => r.size_str,
                "volume_cc" => r.volume_cc,
                "diameter_mm" => r.diameter_mm,
                "num_lesions" => r.num_lesions,
                "psma_q" => r.psma_q,
                "suv_max" => r.suv_max,
                "psma_v" => r.psma_v,
                "psma_v_num" => r.psma_v_num,
                "reader_confidence" => r.reader_confidence,
                "recip_status" => r.recip_status,
                "delta_str" => r.delta_str,
                "is_new" => r.is_new,
                "comment" => r.comment
            ) for r in rep.synoptic_rows
        ],
        "artifact_rows" => [
            Dict{String, Any}(
                "id" => r.id,
                "location" => r.location,
                "mitnm" => r.mitnm,
                "size_str" => r.size_str,
                "volume_cc" => r.volume_cc,
                "diameter_mm" => r.diameter_mm,
                "num_lesions" => r.num_lesions,
                "psma_q" => r.psma_q,
                "suv_max" => r.suv_max,
                "psma_v" => r.psma_v,
                "psma_v_num" => r.psma_v_num,
                "reader_confidence" => r.reader_confidence,
                "recip_status" => r.recip_status,
                "delta_str" => r.delta_str,
                "is_new" => r.is_new,
                "comment" => r.comment
            ) for r in rep.artifact_rows
        ],
        "final_mitnm" => rep.final_mitnm,
        "tmtv_cc" => rep.tmtv_cc,
        "tmtv_delta_pct" => rep.tmtv_delta_pct,
        "overall_recip" => rep.overall_recip,
        "conclusion_en" => rep.conclusion_en,
        "conclusion_de" => rep.conclusion_de
    )
end

# ── Export to Word (.docx) ───────────────────────────────────────────────────
"""Recursively sanitize NaN/Inf values in a Dict/Vector tree for JSON compatibility."""
function _sanitize_json!(d)
    if d isa Dict
        for (k, v) in d
            d[k] = _sanitize_val(v)
        end
    elseif d isa AbstractVector
        for (i, v) in enumerate(d)
            d[i] = _sanitize_val(v)
        end
    end
end
function _sanitize_val(v)
    if v isa AbstractFloat && (isnan(v) || isinf(v))
        return 0.0
    elseif v isa Dict || v isa AbstractVector
        _sanitize_json!(v)
        return v
    else
        return v
    end
end

"""
    export_to_docx(report::EPSMAReport, out_path::String; lang="EN") -> String

Exports the E-PSMA report to a formatted Microsoft Word document (.docx).
"""
function export_to_docx(report::EPSMAReport, out_path::String; lang::String="EN")::String
    json_data = to_dict(report)
    _sanitize_json!(json_data)
    json_str = JSON.json(json_data)
    
    # Robust path resolution (before try so it's in scope for catch)
    script_path = abspath(joinpath(@__DIR__, "..", "..", "scripts", "export", "generate_epsma_docx.py"))
    if !isfile(script_path)
        script_path = "/workspaces/MedEye3d.jl/scripts/export/generate_epsma_docx.py"
    end
    if !isfile(script_path)
        error("Python export script not found. Searched: $(abspath(joinpath(@__DIR__, "..", "..", "scripts", "export", "generate_epsma_docx.py")))")
    end
    
    # Use project data dir for temp file (system /tmp may have user quota limits)
    tmp_dir = abspath(joinpath(@__DIR__, "..", "..", "data"))
    isdir(tmp_dir) || mkpath(tmp_dir)
    (tmp_json, tmp_io) = mktemp(tmp_dir)
    try
        write(tmp_io, json_str)
        close(tmp_io)
        
        # Ensure output directory exists
        out_dir = dirname(out_path)
        isdir(out_dir) || mkpath(out_dir)
        
        cmd = `python3 $script_path $tmp_json $out_path $lang`
        result = read(cmd, String)
        
        if !isfile(out_path)
            error("Python script ran but output file not created at: $out_path\nScript output: $result")
        end
        @info "Exported E-PSMA Word report" path=out_path size=filesize(out_path)
        return out_path
    catch e
        @error "Failed to export Word document" exception=(e, catch_backtrace()) script_path=script_path
        rethrow(e)
    finally
        rm(tmp_json, force=true)
    end
end

end # module EPSMAStructuredReport
