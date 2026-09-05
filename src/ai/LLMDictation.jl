module LLMDictation

using JSON
using Statistics
using Dates

export compile_timepoint_clinical_data,
       build_dictation_prompt,
       call_academiccloud_chat,
       generate_radiological_dictation,
       generate_report_async,
       copy_to_clipboard,
       TimePointClinicalData,
       LesionFinding,
       ResolvedLesion

# ── API Configuration ────────────────────────────────────────────────────────
const ACADEMICCLOUD_API_ENDPOINT = "https://chat-ai.academiccloud.de/v1"
const DEFAULT_API_KEY = "0ba9f0fe8f336d9c0330e3e86c0bbb15"
# Heaviest available Qwen model on AcademicCloud (397B total params MoE)
const DEFAULT_MODEL = "qwen3.5-397b-a17b"

function get_api_key()::String
    key = get(ENV, "ACADEMICCLOUD_API_KEY", "")
    !isempty(key) && return key
    key = get(ENV, "DIZ_API_KEY", "")
    !isempty(key) && return key
    return DEFAULT_API_KEY
end

function get_api_endpoint()::String
    endpoint = get(ENV, "ACADEMICCLOUD_API_ENDPOINT", "")
    !isempty(endpoint) && return rstrip(endpoint, '/')
    return ACADEMICCLOUD_API_ENDPOINT
end

# ── Data Structures for Clinical Context ─────────────────────────────────────
struct LesionFinding
    id::Int
    display_name::String
    base_anatomy::String
    side::String
    sublocation::String
    lesion_type::String
    radlex::String
    comment::String
    centroid::Vector{Int}
    slice::Int
    volume_cc::Float64
    diameter_mm::Float64
    suv_max::Float32
    suv_liver_ratio::Float32
    suv_blood_ratio::Float32
    # Longitudinal trajectory
    match_group_id::Int
    has_prior::Bool
    prior_tp_label::String
    prior_volume_cc::Float64
    prior_suv_max::Float32
    vol_delta_abs::Float64
    vol_delta_pct::Float64
    suv_delta_abs::Float32
    suv_delta_pct::Float64
    recip_status::String
    is_new::Bool
end

struct ResolvedLesion
    prior_tp_label::String
    prior_id::Int
    base_anatomy::String
    prior_volume_cc::Float64
    prior_suv_max::Float32
end

struct TimePointClinicalData
    patient_id::String
    tp_index::Int
    tp_label::String
    modality::String
    description_de::String
    description_en::String
    background_suv::Dict{String, Float32}  # "liver", "blood", "parotid"
    lesions::Vector{LesionFinding}
    resolved_lesions::Vector{ResolvedLesion}
    total_lesions::Int
    total_tmtv_cc::Float64
    prior_tmtv_cc::Float64
    tmtv_delta_pct::Float64
    overall_recip::String
    has_prior_comparisons::Bool
end

# Helper to dynamically access modules without circular compile-time locks
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

function _get_la()
    isdefined(Main, :MedEye3d) && isdefined(Main.MedEye3d, :LesionAssociation) &&
        return Main.MedEye3d.LesionAssociation
    return nothing
end

# ── Longitudinal Context Compiler ────────────────────────────────────────────
"""
    compile_timepoint_clinical_data(tp_idx::Int) -> TimePointClinicalData

Aggregates all anatomical, quantitative, background physiological, and longitudinal
comparison data for the specified time point.
"""
function compile_timepoint_clinical_data(tp_idx::Int)::TimePointClinicalData
    _MEH = _get_meh()
    LMW = _get_lmw()
    LA = _get_la()

    patient_id = _MEH !== nothing ? _MEH.patient_id[] : "PAT_UNKNOWN"
    tp_label = _MEH !== nothing ? get(_MEH.tp_labels, tp_idx, "TP $tp_idx") : "TP $tp_idx"
    modality = _MEH !== nothing ? get(_MEH.tp_modalities, tp_idx, "PET/CT") : "PET/CT"
    desc_de = _MEH !== nothing ? get(_MEH.tp_descriptions, tp_idx, "") : ""
    desc_en = _MEH !== nothing ? get(_MEH.tp_english_descriptions, tp_idx, "") : ""

    # Background SUVs
    bg = if LMW !== nothing
        try LMW.get_background_suvs(tp_idx) catch; Dict{String, Float32}("liver" => 2.0f0, "blood" => 1.5f0, "parotid" => 1.8f0) end
    else
        Dict{String, Float32}("liver" => 2.0f0, "blood" => 1.5f0, "parotid" => 1.8f0)
    end
    liver_bg = get(bg, "liver", 2.0f0)
    blood_bg = get(bg, "blood", 1.5f0)

    # Lesion IDs present at this time point
    l_ids = if LMW !== nothing
        try LMW.get_mask_ids(tp_idx) catch; Int[] end
    else
        Int[]
    end

    lesions = LesionFinding[]
    has_prior_comparisons = false

    db = if LMW !== nothing
        try LMW.get_active_lesion_db() catch; Dict{String, Any}() end
    else
        Dict{String, Any}()
    end

    all_prior_tp_indices = _MEH !== nothing ? filter(t -> t < tp_idx, collect(keys(_MEH.tp_labels))) : Int[]
    has_prior_studies = !isempty(all_prior_tp_indices)

    for lid in l_ids
        # 1. Metadata
        state = if LMW !== nothing && !isempty(db)
            try LMW.get_lesion_state(db, string(lid)) catch; Dict{String, String}() end
        else
            Dict{String, String}()
        end
        
        display_name = get(state, "_display_name", "Lesion $lid")
        base_anatomy = get(state, "BaseAnatomy", "")
        if isempty(base_anatomy) && _MEH !== nothing
            organ = get(_MEH.global_organ_mapping[], lid, "")
            if !isempty(organ) && organ != "Unknown"
                base_anatomy = organ
            end
        end
        side = get(state, "BaseAnatomySide", "")
        sublocation = get(state, "Anatomical Details", "")
        lesion_type = get(state, "LesionType", "Lesion")
        radlex = get(state, "RadLex", "")
        comment = get(state, "Comment", "")

        # 2. Centroid & Slice
        c = [0, 0, 0]
        if _MEH !== nothing
            if haskey(_MEH.lesion_centroids_cache, (tp_idx, lid))
                c = _MEH.lesion_centroids_cache[(tp_idx, lid)]
            elseif haskey(_MEH.lesion_centroids_cache, lid)
                c = _MEH.lesion_centroids_cache[lid]
            end
        end
        slice = length(c) >= 3 ? c[3] : 0

        # 3. Volume & Diameter
        vol_cc = 0.0
        diam_mm = 0.0
        if LMW !== nothing
            try
                v_res = LMW.compute_lesion_volume(lid, tp_idx)
                vol_cc = get(v_res, "volume_cc", 0.0)
                diam_mm = get(v_res, "diameter_mm", 0.0)
            catch; end
        end

        # 4. SUVmax & Ratios
        suv_max = 0.0f0
        if LMW !== nothing
            try suv_max = LMW._get_suv_max(lid, tp_idx) catch; end
        end
        suv_liver_ratio = liver_bg > 0.01f0 ? round(suv_max / liver_bg, digits=2) : 0.0f0
        suv_blood_ratio = blood_bg > 0.01f0 ? round(suv_max / blood_bg, digits=2) : 0.0f0

        # 5. Longitudinal Match Analysis
        match_group_id = 0
        has_prior = false
        prior_tp_label = ""
        prior_volume_cc = 0.0
        prior_suv_max = 0.0f0
        vol_delta_abs = 0.0
        vol_delta_pct = 0.0
        suv_delta_abs = 0.0f0
        suv_delta_pct = 0.0
        recip_status = "BASELINE"
        is_new = false

        if LMW !== nothing
            try
                m_res = LMW.compute_match_analysis(lid, tp_idx)
                if m_res !== nothing
                    match_group_id = m_res.group_id
                    if m_res.recip != "N/A (baseline)"
                        has_prior = true
                        has_prior_comparisons = true
                        prior_tp_label = m_res.baseline_node
                        prior_volume_cc = m_res.base_volume_cc
                        prior_suv_max = m_res.base_suv_max
                        vol_delta_abs = m_res.vol_delta_abs
                        vol_delta_pct = m_res.vol_delta_pct
                        suv_delta_abs = m_res.suv_delta_abs
                        suv_delta_pct = m_res.suv_delta_pct
                        recip_status = m_res.recip
                        is_new = false
                    else
                        recip_status = "BASELINE"
                        is_new = false
                    end
                else
                    if has_prior_studies
                        recip_status = "NEW LESION"
                        is_new = true
                        has_prior_comparisons = true
                    else
                        recip_status = "BASELINE"
                        is_new = false
                    end
                end
            catch; end
        end

        push!(lesions, LesionFinding(
            lid, display_name, base_anatomy, side, sublocation, lesion_type,
            radlex, comment, c, slice, vol_cc, diam_mm, suv_max,
            suv_liver_ratio, suv_blood_ratio,
            match_group_id, has_prior, prior_tp_label, prior_volume_cc, prior_suv_max,
            vol_delta_abs, vol_delta_pct, suv_delta_abs, suv_delta_pct,
            recip_status, is_new
        ))
    end

    # 6. Check for Resolved Lesions (present in prior studies but not in current)
    resolved_lesions = ResolvedLesion[]
    if LA !== nothing && LMW !== nothing && !isempty(all_prior_tp_indices)
        try
            match_groups = LA.get_match_groups()
            current_node = _MEH !== nothing ? _MEH.get_node_name_for_tp(tp_idx) : "TP$tp_idx"
            
            for p_idx in all_prior_tp_indices
                p_label = _MEH !== nothing ? get(_MEH.tp_labels, p_idx, "TP $p_idx") : "TP $p_idx"
                p_ids = LMW.get_mask_ids(p_idx)
                for pid in p_ids
                    # Check if pid is matched to current_node
                    is_present_now = false
                    for (gid, members) in match_groups
                        has_p = any(m -> m[2] == pid && LA._tp_index_from_node(m[1]) == p_idx, members)
                        has_cur = any(m -> m[1] == current_node, members)
                        if has_p && has_cur
                            is_present_now = true
                            break
                        end
                    end
                    if !is_present_now
                        # Compute prior volume & SUV
                        p_vol = 0.0
                        p_suv = 0.0f0
                        try
                            pv = LMW.compute_lesion_volume(pid, p_idx)
                            p_vol = get(pv, "volume_cc", 0.0)
                            p_suv = LMW._get_suv_max(pid, p_idx)
                        catch; end
                        p_state = try LMW.get_lesion_state(db, string(pid)) catch; Dict{String,String}() end
                        p_anat = get(p_state, "BaseAnatomy", "Lesion $pid")
                        push!(resolved_lesions, ResolvedLesion(p_label, pid, p_anat, p_vol, p_suv))
                    end
                end
            end
        catch; end
    end

    # 7. Total Tumor Burden & Overall Response
    total_tmtv_cc = round(sum(l.volume_cc for l in lesions; init=0.0), digits=2)
    prior_tmtv_cc = round(sum(l.prior_volume_cc for l in lesions if l.has_prior; init=0.0), digits=2)
    tmtv_delta_pct = prior_tmtv_cc > 0.001 ?
        round((total_tmtv_cc - prior_tmtv_cc) / prior_tmtv_cc * 100.0, digits=1) : 0.0

    overall_recip = if !has_prior_comparisons
        "BASELINE (Initial Staging)"
    elseif any(l -> l.is_new, lesions)
        "PROGRESSIVE METABOLIC DISEASE (PMD / RECIP-PD) - New Lesion Emerged"
    elseif any(l -> l.recip_status == "RECIP-PD", lesions)
        "PROGRESSIVE METABOLIC DISEASE (PMD / RECIP-PD) - Lesion Growth"
    elseif !isempty(lesions) && all(l -> l.recip_status == "RECIP-CR" || l.volume_cc < 0.001, lesions)
        "COMPLETE METABOLIC RESPONSE (CMR / RECIP-CR)"
    elseif any(l -> l.recip_status == "RECIP-PR", lesions) || tmtv_delta_pct < -30.0
        "PARTIAL METABOLIC RESPONSE (PMR / RECIP-PR)"
    else
        "STABLE METABOLIC DISEASE (SMD / RECIP-SD)"
    end

    return TimePointClinicalData(
        patient_id, tp_idx, tp_label, modality, desc_de, desc_en,
        bg, lesions, resolved_lesions, length(lesions),
        total_tmtv_cc, prior_tmtv_cc, tmtv_delta_pct,
        overall_recip, has_prior_comparisons
    )
end

# ── Radiological Dictation Prompt Builder ─────────────────────────────────────
"""
    build_dictation_prompt(data::TimePointClinicalData; lang="EN") -> Vector{Dict{String, String}}

Builds system and user prompt messages formatted for high-quality, comprehensive
clinical radiological dictation following PERCIST 1.0, RECIP 1.0, and RECIST 1.1 criteria.
"""
function build_dictation_prompt(data::TimePointClinicalData; lang="EN")::Vector{Dict{String, String}}
    is_de = uppercase(lang) == "DE"

    system_prompt = if is_de
        """Sie sind ein habilitierter Facharzt für Radiologie und Nuklearmedizin mit Schwerpunkt auf onkologischer PET/CT- und SPECT/CT-Bildgebung.
Ihre Aufgabe ist es, einen vollständigen, strukturierten und klinisch maßgeblichen radiologischen Befundbericht (Befunddiktat) zu erstellen.
Priorität hat absolute klinische Präzision, methodische Strenge und Vollständigkeit gemäß den Kriterien von PERCIST 1.0, RECIP 1.0 und RECIST 1.1.
Der Bericht muss exakt und ohne Auslassungen alle quantitativen Läsionsparameter (Volumen in ml/cc, Durchmesser in mm, SUVmax), Referenzorgane (Leber, Mediastinum/Blutpool), longitudinale Vergleiche mit Voruntersuchungen (Deltas und Prozentwerte) sowie eine definitive onkologische Gesamtbeurteilung enthalten."""
    else
        """You are an expert board-certified radiologist and nuclear medicine physician specialized in oncological PET/CT and SPECT/CT interpretation.
Your task is to generate a comprehensive, fully structured, and clinically authoritative radiological dictation / formal report.
Highest priority is placed on medical accuracy, clinical rigor, and exhaustive completeness adhering to PERCIST 1.0, RECIP 1.0, and RECIST 1.1 criteria.
The report must meticulously detail all quantitative lesion measurements (volume in cc, longest diameter in mm, SUVmax), physiological background references (liver, mediastinal blood pool), longitudinal comparisons with earlier time points (absolute & percentage deltas), explicit tracking of new or resolved lesions, and an actionable, definitive oncological impression."""
    end

    # Build User Content Payload
    io = IOBuffer()
    println(io, "================================================================================")
    println(io, "CLINICAL EXAMINATION METADATA & QUANTITATIVE MEASUREMENTS")
    println(io, "================================================================================")
    println(io, "Patient Identifier: $(data.patient_id)")
    println(io, "Current Study: $(data.tp_label) (TimePoint Index: $(data.tp_index))")
    println(io, "Imaging Modality: $(data.modality)")
    if !isempty(data.description_en)
        println(io, "Clinical History & Indication (Translated): $(data.description_en)")
    elseif !isempty(data.description_de)
        println(io, "Klinische Vorgeschichte / Originalbefund (DE): $(data.description_de)")
    end
    println(io, "")

    println(io, "--------------------------------------------------------------------------------")
    println(io, "PHYSIOLOGICAL BACKGROUND REFERENCE VALUES (SUVmean)")
    println(io, "--------------------------------------------------------------------------------")
    println(io, "Liver SUVmean: $(get(data.background_suv, "liver", 0.0f0))")
    println(io, "Mediastinal Blood Pool SUVmean: $(get(data.background_suv, "blood", 0.0f0))")
    println(io, "Parotid SUVmean: $(get(data.background_suv, "parotid", 0.0f0))")
    println(io, "")

    println(io, "--------------------------------------------------------------------------------")
    println(io, "TUMOR BURDEN SUMMARY")
    println(io, "--------------------------------------------------------------------------------")
    println(io, "Total Active Target Lesions: $(data.total_lesions)")
    println(io, "Total Metabolic Tumor Volume (TMTV): $(data.total_tmtv_cc) cc")
    if data.has_prior_comparisons
        println(io, "Prior Baseline TMTV: $(data.prior_tmtv_cc) cc")
        println(io, "TMTV Longitudinal Change: $(data.tmtv_delta_pct > 0 ? "+" : "")$(data.tmtv_delta_pct)%")
        println(io, "Calculated Global Response (PERCIST/RECIP): $(data.overall_recip)")
    else
        println(io, "Evaluation Mode: Baseline Study (Initial Staging)")
    end
    println(io, "")

    println(io, "--------------------------------------------------------------------------------")
    println(io, "TARGET LESION INVENTORY ($(data.total_lesions) LESIONS)")
    println(io, "--------------------------------------------------------------------------------")
    if isempty(data.lesions)
        println(io, "(No segmented lesions identified in this time point)")
    else
        for (i, l) in enumerate(data.lesions)
            loc_str = !isempty(l.base_anatomy) ? l.base_anatomy : "Unspecified site"
            side_str = !isempty(l.side) ? " ($(l.side))" : ""
            sub_str = !isempty(l.sublocation) ? " - $(l.sublocation)" : ""
            println(io, "• Lesion $(l.id) [$(l.display_name)]: $(loc_str)$(side_str)$(sub_str)")
            println(io, "  - Classification: $(l.lesion_type)$(isempty(l.radlex) ? "" : " | RadLex: $(l.radlex)")")
            println(io, "  - 3D Coordinates: (x=$(l.centroid[1]), y=$(l.centroid[2]), z=$(l.centroid[3])) | Axial Slice: $(l.slice)")
            println(io, "  - Current Metrics: Volume = $(l.volume_cc) cc | Longest Diameter = $(l.diameter_mm) mm | SUVmax = $(l.suv_max)")
            println(io, "  - Target/Background Ratios: SUVmax/Liver = $(l.suv_liver_ratio)x | SUVmax/BloodPool = $(l.suv_blood_ratio)x")
            
            if l.has_prior
                println(io, "  - Prior Baseline ($(l.prior_tp_label)): Volume = $(l.prior_volume_cc) cc | SUVmax = $(l.prior_suv_max)")
                v_sgn = l.vol_delta_pct >= 0 ? "+" : ""
                s_sgn = l.suv_delta_abs >= 0 ? "+" : ""
                println(io, "  - Longitudinal Delta: Volume $(v_sgn)$(round(l.vol_delta_pct, digits=1))% ($(v_sgn)$(round(l.vol_delta_abs, digits=2)) cc) | SUVmax $(s_sgn)$(round(l.suv_delta_pct, digits=1))% ($(s_sgn)$(round(l.suv_delta_abs, digits=1)))")
                println(io, "  - Response Status: $(l.recip_status)")
            elseif l.is_new
                println(io, "  - Longitudinal Trajectory: *** NEW HYPERMETABOLIC LESION *** (Not present on prior baseline studies)")
                println(io, "  - Response Status: NEW LESION (Indicates Progressive Disease by PERCIST 1.0 / RECIP 1.0)")
            else
                println(io, "  - Longitudinal Trajectory: Baseline Reference Lesion")
            end
            if !isempty(l.comment)
                println(io, "  - Radiologist Note / Morphology: $(l.comment)")
            end
            println(io, "")
        end
    end

    if !isempty(data.resolved_lesions)
        println(io, "--------------------------------------------------------------------------------")
        println(io, "RESOLVED LESIONS (COMPLETE METABOLIC CLEARANCE)")
        println(io, "--------------------------------------------------------------------------------")
        for rl in data.resolved_lesions
            println(io, "• Prior Lesion $(rl.prior_id) ($(rl.base_anatomy)) from $(rl.prior_tp_label):")
            println(io, "  - Baseline Volume: $(rl.prior_volume_cc) cc | Baseline SUVmax: $(rl.prior_suv_max)")
            println(io, "  - Current Status: No longer visible / non-avid (100% Volume and Metabolic Reduction - Complete Response)")
        end
        println(io, "")
    end

    instructions = if is_de
        """Bitte verfassen Sie auf Grundlage obiger Messwerte das vollständige, strukturierte radiologische Befunddiktat mit folgenden Standardabschnitten:
1. KLINISCHE ANGABEN & UNTERSUCHUNGSTECHNIK
2. PHYSIOLOGISCHE HINTERGRUNDAKTIVITÄT
3. DETAILLIERTER BEFUND (Anatomisch & Läsionsbasiert mit exakten Messwerten)
4. VERLAUFS- UND THERAPIEANSPRECHEN (PERCIST / RECIP / RECIST 1.1)
5. GESAMTBEURTEILUNG & EMPFEHLUNG (Nummerierte, präzise Zusammenfassung mit Staging und Verlaufskategorie)"""
    else
        """Please formulate a complete, formal, structured radiological dictation based on the quantitative measurements above.
Structure the report into the standard clinical sections:
1. CLINICAL INDICATION & EXAMINATION TECHNIQUE
2. PHYSIOLOGICAL BACKGROUND REFERENCE UPTAKE
3. DETAILED ANATOMICAL & METABOLIC FINDINGS
4. LONGITUDINAL RESPONSE EVALUATION (PERCIST 1.0 / RECIP 1.0 / RECIST 1.1)
5. IMPRESSION & CLINICAL RECOMMENDATION (Numbered, definitive conclusions, response category, and follow-up guidance)."""
    end

    println(io, instructions)
    user_prompt = String(take!(io))

    return [
        Dict("role" => "system", "content" => system_prompt),
        Dict("role" => "user", "content" => user_prompt)
    ]
end

# ── API Execution ────────────────────────────────────────────────────────────
"""
    call_academiccloud_chat(messages; model=DEFAULT_MODEL, max_tokens=6000, temperature=0.1, api_key="", api_base="") -> String

Dispatches chat completion call to AcademicCloud API using curl subprocess.
"""
function call_academiccloud_chat(
    messages::Vector{Dict{String, String}};
    model::String = DEFAULT_MODEL,
    max_tokens::Int = 6000,
    temperature::Float64 = 0.1,
    api_key::String = "",
    api_base::String = ""
)::String
    key = isempty(api_key) ? get_api_key() : api_key
    base = isempty(api_base) ? get_api_endpoint() : api_base
    url = rstrip(base, '/') * "/chat/completions"

    body_dict = Dict(
        "model" => model,
        "messages" => messages,
        "max_tokens" => max_tokens,
        "temperature" => temperature
    )
    body_json = JSON.json(body_dict)

    (tmp_path, tmp_io) = mktemp()
    try
        write(tmp_io, body_json)
        close(tmp_io)

        cmd = `curl -s -X POST $url -H "Authorization: Bearer $key" -H "Content-Type: application/json" -d @$tmp_path`
        output = read(cmd, String)
        
        resp = JSON.parse(output)
        if haskey(resp, "error")
            err_msg = get(resp["error"], "message", "Unknown API error")
            return "API Error: $err_msg"
        end

        choices = get(resp, "choices", [])
        if isempty(choices)
            return "API Error: No choices returned in response: $output"
        end

        msg = get(choices[1], "message", Dict())
        content = get(msg, "content", nothing)
        
        if content === nothing || isempty(strip(string(content)))
            # Model might have stopped while inside reasoning tokens
            reasoning = get(msg, "reasoning", "")
            if !isempty(reasoning)
                return "[Reasoning completed, content empty - check token budget]\n\nReasoning:\n$reasoning"
            end
            return "API Error: Empty content returned."
        end

        return strip(string(content))
    catch e
        return "Connection Error: $e"
    finally
        rm(tmp_path, force=true)
    end
end

"""
    generate_radiological_dictation(tp_idx::Int; lang="EN", model=DEFAULT_MODEL) -> String

High-level synchronous function: aggregates time point data, builds prompt,
and calls Qwen 3.5 397B on AcademicCloud to produce a complete radiological dictation.
"""
function generate_radiological_dictation(tp_idx::Int; lang::String="EN", model::String=DEFAULT_MODEL)::String
    data = compile_timepoint_clinical_data(tp_idx)
    messages = build_dictation_prompt(data; lang=lang)
    return call_academiccloud_chat(messages; model=model)
end

"""
    generate_report_async(tp_idx::Int; lang="EN", model=DEFAULT_MODEL, on_complete=nothing, on_error=nothing)

Asynchronously generates the radiological dictation on a background thread (`Threads.@spawn`)
so the Makie GLFW rendering loop is never blocked.
"""
function generate_report_async(
    tp_idx::Int;
    lang::String = "EN",
    model::String = DEFAULT_MODEL,
    on_complete::Union{Function, Nothing} = nothing,
    on_error::Union{Function, Nothing} = nothing
)
    Threads.@spawn begin
        t_start = time()
        try
            report = generate_radiological_dictation(tp_idx; lang=lang, model=model)
            elapsed = round(time() - t_start, digits=1)
            if on_complete !== nothing
                on_complete(report, elapsed)
            end
        catch e
            @error "Failed to generate radiological dictation: $e"
            if on_error !== nothing
                on_error(e)
            end
        end
    end
end

"""
    copy_to_clipboard(text::String) -> Bool

Cross-platform clipboard copying for Linux (xclip, wl-copy, or fallback).
"""
function copy_to_clipboard(text::String)::Bool
    # Try xclip (X11)
    try
        p = open(`xclip -selection clipboard`, "w", stdout)
        write(p, text)
        close(p)
        return true
    catch; end

    # Try wl-copy (Wayland)
    try
        p = open(`wl-copy`, "w", stdout)
        write(p, text)
        close(p)
        return true
    catch; end

    # Try InteractiveUtils if available in Main
    if isdefined(Main, :InteractiveUtils) && isdefined(Main.InteractiveUtils, :clipboard)
        try
            Main.InteractiveUtils.clipboard(text)
            return true
        catch; end
    end

    println("[Clipboard Output]:\n$text")
    return false
end

end # module LLMDictation
