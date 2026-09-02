"""
LesionMetadataWindow — Julia port of the Slicer Lesion Text Extension annotation panel.

Architecture:
  - Reads annotation schema from extension/data/def.json  (20 questions, matches Python source)
  - Reads RadLex ontology from extension/data/RadLex.csv  (~45k terms, cached to 2000 for performance)
  - Renders a GLMakie Figure with per-question comboboxes (Menu) or textboxes
  - Persists annotations as JSON in ~/medeye3d_lesion_annotations.json
  - Exposes Observables for integration with the MedEye3d GLFW viewer channel

Usage:
    win = LesionMetadataWindow.create_metadata_window(
        active_lesion_id,   # Observable{String}
        lesion_ids,         # Observable{Vector{String}}
        ui_hooks            # Dict{Symbol,Observable}
    )

ui_hooks keys:
    :scroll      => Observable{Int}           — set ±1 to scroll a slice
    :windowing   => Observable{Tuple{Float32,Float32}} — (min,max) CT window
    :paint_val   => Observable{Int}           — 1=paint 0=erase
    :sync_lesion => Observable{Bool}          — trigger lesion sync
"""
module LesionMetadataWindow

using GLMakie
using Observables
using JSON
using HDF5
using Dates
using ..MakieEvents
import ..SegmentationDisplay: synchronized_makie_renderloop, GLOBAL_OPENGL_LOCK
import ..SegmentationDisplay.MakieEventHandlers as _MEH
import ..LesionAssociation as LA

abstract type DBMessage end
struct SaveDBMessage <: DBMessage
    db::Dict
    global_app_state::Dict
    path_json::String
    path_hdf5::String
end
struct LoadDBMessage <: DBMessage
    path_json::String
    reply_channel::Channel{Dict}
    path_hdf5::String
    LoadDBMessage(path_json::String, reply_channel::Channel{Dict}, path_hdf5::String=DEFAULT_HDF5_PATH) = new(path_json, reply_channel, path_hdf5)
end

export create_metadata_window, load_annotations, save_annotations, load_annotations_hdf5, save_annotations_hdf5, get_lesion_state, display_metadata_window

# ─── Paths ────────────────────────────────────────────────────────────────────
const _PKG_ROOT      = joinpath(@__DIR__, "..", "..", "extension", "data")
const _SLICER_DATA   = "/mnt/big/project_ssd/project_ssd/slicer_lesion_text_extension/data"
const DEF_JSON_PATH  = isfile(joinpath(_SLICER_DATA, "def.json")) ? joinpath(_SLICER_DATA, "def.json") : (isfile(joinpath(_PKG_ROOT, "def.json")) ? joinpath(_PKG_ROOT, "def.json") : joinpath(@__DIR__, "..", "..", "data", "def.json"))
const RADLEX_CSV_PATH= isfile(joinpath(_SLICER_DATA, "RadLex.csv")) ? joinpath(_SLICER_DATA, "RadLex.csv") : (isfile(joinpath(_PKG_ROOT, "RadLex.csv")) ? joinpath(_PKG_ROOT, "RadLex.csv") : joinpath(@__DIR__, "..", "..", "data", "RadLex.csv"))
const DEFAULT_SAVE_PATH = joinpath(homedir(), "medeye3d_lesion_annotations.json")
const DEFAULT_HDF5_PATH = joinpath(homedir(), "medeye3d_lesion_annotations.h5")
const ANATOMY_MAPPING_PATH = joinpath(@__DIR__, "..", "..", "data", "max_anatomy_to_ontology.json")

# Persistent custom dropdown options (cross-patient and cross-run)
const GLOBAL_CUSTOM_OPTS_PATH = joinpath(homedir(), ".medeye3d_custom_options.json")
const GLOBAL_CUSTOM_FIELDS_PATH = joinpath(homedir(), ".medeye3d_custom_fields.json")
const CUSTOM_OPTS_PATH = let
    p1 = joinpath(_SLICER_DATA, "custom_options.json")
    p2 = joinpath(_PKG_ROOT, "custom_options.json")
    isfile(p1) ? p1 : p2
end

const _custom_opts_cache = Ref{Dict{String,Any}}(Dict{String,Any}())
const _custom_fields_cache = Ref{Dict{String,String}}(Dict{String,String}())

# Persistent display preferences (PET/CT blend, Label Opacity)
const GLOBAL_DISPLAY_CONFIG_PATH = joinpath(homedir(), ".medeye3d_display_config.json")

function load_display_config()::Dict{String,Any}
    if isfile(GLOBAL_DISPLAY_CONFIG_PATH)
        try
            return JSON.parse(read(GLOBAL_DISPLAY_CONFIG_PATH, String))
        catch e
            @warn "Failed to parse $GLOBAL_DISPLAY_CONFIG_PATH: $e"
        end
    end
    return Dict{String,Any}(
        "label_opacity" => 0.5,
        "pet_ct_blend" => 0.5
    )
end

function save_display_config(cfg::Dict{String,Any})
    try
        write(GLOBAL_DISPLAY_CONFIG_PATH, JSON.json(cfg, 4))
    catch e
        @warn "Failed to save $GLOBAL_DISPLAY_CONFIG_PATH: $e"
    end
end

function load_custom_options()::Dict{String,Any}
    isempty(_custom_opts_cache[]) || return _custom_opts_cache[]
    res = Dict{String,Any}()
    # Merge package/slicer default custom options
    if isfile(CUSTOM_OPTS_PATH)
        try
            merge!(res, JSON.parse(read(CUSTOM_OPTS_PATH, String)))
        catch e
            @warn "Failed to load custom_options.json: $e"
        end
    end
    # Merge user global custom options
    if isfile(GLOBAL_CUSTOM_OPTS_PATH)
        try
            g_opts = JSON.parse(read(GLOBAL_CUSTOM_OPTS_PATH, String))
            for (k, v) in g_opts
                cur = get(res, k, Any[])
                res[k] = unique(vcat(cur, v))
            end
        catch e
            @warn "Failed to load global custom options: $e"
        end
    end
    _custom_opts_cache[] = res
    return _custom_opts_cache[]
end

function save_custom_options!(db::Dict)
    _custom_opts_cache[] = copy(db)
    for p in [GLOBAL_CUSTOM_OPTS_PATH, CUSTOM_OPTS_PATH]
        try
            mkpath(dirname(p))
            open(p, "w") do f
                JSON.print(f, db, 4)
            end
        catch e
            @warn "Failed to save custom options to $p: $e"
        end
    end
end

function add_global_custom_option(category::String, option::String)
    isempty(category) || isempty(option) && return
    opts = load_custom_options()
    cur = get(opts, category, Any[])
    if !(option in cur)
        push!(cur, option)
        opts[category] = cur
        save_custom_options!(opts)
        @info "Persisted global custom option: [$category] -> '$option'"
    end
end

function load_global_custom_fields()::Dict{String,String}
    isempty(_custom_fields_cache[]) || return _custom_fields_cache[]
    if isfile(GLOBAL_CUSTOM_FIELDS_PATH)
        try
            raw = JSON.parse(read(GLOBAL_CUSTOM_FIELDS_PATH, String))
            _custom_fields_cache[] = Dict{String,String}(string(k) => string(v) for (k, v) in raw)
        catch e
            @warn "Failed to load global custom fields: $e"
            _custom_fields_cache[] = Dict{String,String}()
        end
    end
    return _custom_fields_cache[]
end

function save_global_custom_fields(fields::Dict)
    _custom_fields_cache[] = Dict{String,String}(string(k) => string(v) for (k, v) in fields)
    try
        mkpath(dirname(GLOBAL_CUSTOM_FIELDS_PATH))
        open(GLOBAL_CUSTOM_FIELDS_PATH, "w") do f
            JSON.print(f, fields, 4)
        end
    catch e
        @warn "Failed to save global custom fields: $e"
    end
end

# ─── Schema ───────────────────────────────────────────────────────────────────
struct QuestionDef
    short::String
    full::String
    options::Vector{String}
    categories::Vector{String}
    mode::String
    default_answer::String
end

const _schema_cache = Ref{Vector{QuestionDef}}(QuestionDef[])

function _builtin_schema()::Vector{QuestionDef}
    [
        QuestionDef("Radioligand Type","Radioligand used",
            ["68Ga-PSMA-11","18F-PSMA-1007","18F-DCFPyL","Other"],
            ["Technical Parameters"],"both","68Ga-PSMA-11"),
        QuestionDef("Lesion tracking name?","Anatomical descriptor",
            String[],["Identification"],"both",""),
        QuestionDef("Anatomic Location","Primary anatomical site",
            ["Prostate Gland","Axial Skeleton","Appendicular Skeleton",
             "Pelvic Lymph Node","Distant Lymph Node","Solid Organ / Viscera",
             "General Soft Tissue","Blood Vessel","Other"],
            ["Location"],"both",""),
        QuestionDef("Inner Texture / Density / Attenuation","Internal density",
            ["Sclerotic / Blastic","Lytic / Lucent","Mixed Lytic & Sclerotic",
             "Ground-Glass / Fibrous","Fluid-Filled / Cystic","Fat Density","Central Necrosis"],
            ["Morphology"],"both",""),
        QuestionDef("Border and Margin","Margin character",
            ["Smooth / Well-Defined","Spiculated / Feathered","Moth-Eaten",
             "Ill-Defined / Permeative","Reactive Sclerotic Rim"],
            ["Morphology"],"both",""),
        QuestionDef("Lesion Shape","3D morphology",
            ["Oval / Bean-Shaped","Round","Teardrop","Lobulated","Irregular"],
            ["Morphology"],"both",""),
        QuestionDef("Certainty","Diagnostic certainty",
            ["High (>90%)","Medium (50-90%)","Low (<50%)"],
            ["Final Assessment"],"both",""),
        QuestionDef("Comment","Free-text comment",String[],["Reporting"],"both",""),
    ]
end

function load_schema()::Vector{QuestionDef}
    isempty(_schema_cache[]) || return _schema_cache[]
    if !isfile(DEF_JSON_PATH)
        _schema_cache[] = _builtin_schema()
        return _schema_cache[]
    end
    raw = JSON.parse(read(DEF_JSON_PATH, String))
    result = QuestionDef[]
    for q in raw
        # JSON.parse returns String keys
        short_q = get(q, "short q", get(q, "short_q", ""))
        opts = String[string(s) for s in get(q, "allowed_answer", Any[])]
        cats = String[string(c) for c in get(q, "category",       Any[])]
        push!(result, QuestionDef(
            string(short_q),
            string(get(q, "full", "")),
            opts,
            cats,
            string(get(q, "meta_or_prostate", "both")),
            string(get(q, "default_answer", ""))
        ))
    end
    _schema_cache[] = result
    return result
end

# ─── RadLex ──────────────────────────────────────────────────────────────────
const _radlex_cache = Ref{Vector{String}}(String[])
const RADLEX_MAX_TERMS = 2000

function load_radlex()::Vector{String}
    isempty(_radlex_cache[]) || return _radlex_cache[]
    terms = String[]
    if isfile(RADLEX_CSV_PATH)
        for (i, line) in enumerate(eachline(RADLEX_CSV_PATH))
            i == 1 && continue   # skip header
            parts = split(line, ','; limit = 4)
            length(parts) >= 2 || continue
            rid   = strip(parts[1])
            label = strip(parts[2])
            (isempty(rid) || isempty(label)) && continue
            push!(terms, "$(rid) - $(label)")
            length(terms) >= RADLEX_MAX_TERMS && break
        end
        @info "Loaded $(length(terms)) RadLex terms"
    else
        @warn "RadLex CSV not found at $(RADLEX_CSV_PATH)"
    end
    _radlex_cache[] = isempty(terms) ? ["(none)"] : sort(terms)
    return _radlex_cache[]
end

# ─── Anatomy Ontology (FoundationalAnatomy.csv) for Base Anatomy autocomplete ─
const ANATOMY_CSV_PATH = isfile(joinpath(_SLICER_DATA, "FoundationalAnatomy.csv")) ? joinpath(_SLICER_DATA, "FoundationalAnatomy.csv") : joinpath(_PKG_ROOT, "FoundationalAnatomy.csv")
const _anatomy_cache = Ref{Vector{String}}(String[])

"""Load ALL FoundationalAnatomy.csv labels for Base Anatomy search/autocomplete."""
function load_anatomy_ontology()::Vector{String}
    isempty(_anatomy_cache[]) || return _anatomy_cache[]
    terms = String[]
    if isfile(ANATOMY_CSV_PATH)
        seen = Set{String}()
        for (i, line) in enumerate(eachline(ANATOMY_CSV_PATH))
            i == 1 && continue   # header: Name,ID
            parts = split(strip(line), ','; limit = 2)
            length(parts) >= 1 || continue
            label = strip(parts[1])
            isempty(label) && continue
            lbl_low = lowercase(label)
            if !(lbl_low in seen)
                push!(seen, lbl_low)
                push!(terms, label)
            end
        end
        @info "Loaded $(length(terms)) unique FoundationalAnatomy terms for Base Anatomy autocomplete"
    else
        @warn "FoundationalAnatomy CSV not found at $(ANATOMY_CSV_PATH)"
    end
    _anatomy_cache[] = isempty(terms) ? ["(none)"] : sort(terms)
    return _anatomy_cache[]
end

# ─── JSON Anatomy Mapping (max_anatomy → ontology) ──────────────────────────
const _anatomy_mapping_cache = Ref{Dict{String,Any}}(Dict{String,Any}())

"""Load the hand-crafted max_anatomy → UBERON ontology mapping from JSON."""
function load_anatomy_mapping()::Dict{String,Any}
    isempty(_anatomy_mapping_cache[]) || return _anatomy_mapping_cache[]
    if isfile(ANATOMY_MAPPING_PATH)
        try
            _anatomy_mapping_cache[] = JSON.parsefile(ANATOMY_MAPPING_PATH)
            @info "Loaded $(length(_anatomy_mapping_cache[])) max_anatomy→ontology mappings"
        catch e
            @warn "Failed to load anatomy mapping JSON: $e"
        end
    else
        @warn "Anatomy mapping JSON not found at $(ANATOMY_MAPPING_PATH)"
    end
    return _anatomy_mapping_cache[]
end

"""
    lookup_anatomy(raw_organ) → Dict or nothing

Look up a raw max_anatomy organ name in the JSON mapping.
Returns a Dict with keys: "detailed", "generalized", "side", "anatomic_location", "lesion_type".
"""
function lookup_anatomy(raw_organ::String)
    isempty(raw_organ) && return nothing
    mapping = load_anatomy_mapping()
    key = lowercase(strip(raw_organ))
    haskey(mapping, key) && return mapping[key]
    # Try case-preserving exact match (max_anatomy uses mixed case for vertebrae)
    haskey(mapping, strip(raw_organ)) && return mapping[strip(raw_organ)]
    return nothing
end

# ─── Static TotalSegmentator → Anatomy Mapping ──────────────────────────────
# Deterministic mapping from raw TS segment names to clean anatomy labels.
# Matches the Slicer extension's parseAndMapToOntology output.
const TS_TO_ANATOMY = Dict{String, String}(
    # Bones
    "femur" => "Femur", "hip" => "Hip Bone", "sacrum" => "Sacrum",
    "skull" => "Skull", "sternum" => "Sternum", "scapula" => "Scapula",
    "clavicula" => "Clavicle", "humerus" => "Humerus",
    # Vertebrae (individual)
    "vertebrae_c1" => "C1 Vertebra", "vertebrae_c2" => "C2 Vertebra",
    "vertebrae_c3" => "C3 Vertebra", "vertebrae_c4" => "C4 Vertebra",
    "vertebrae_c5" => "C5 Vertebra", "vertebrae_c6" => "C6 Vertebra",
    "vertebrae_c7" => "C7 Vertebra",
    "vertebrae_t1" => "T1 Vertebra", "vertebrae_t2" => "T2 Vertebra",
    "vertebrae_t3" => "T3 Vertebra", "vertebrae_t4" => "T4 Vertebra",
    "vertebrae_t5" => "T5 Vertebra", "vertebrae_t6" => "T6 Vertebra",
    "vertebrae_t7" => "T7 Vertebra", "vertebrae_t8" => "T8 Vertebra",
    "vertebrae_t9" => "T9 Vertebra", "vertebrae_t10" => "T10 Vertebra",
    "vertebrae_t11" => "T11 Vertebra", "vertebrae_t12" => "T12 Vertebra",
    "vertebrae_l1" => "L1 Vertebra", "vertebrae_l2" => "L2 Vertebra",
    "vertebrae_l3" => "L3 Vertebra", "vertebrae_l4" => "L4 Vertebra",
    "vertebrae_l5" => "L5 Vertebra", "vertebrae_s1" => "S1 Vertebra",
    "vertebrae" => "Vertebra",
    # Ribs (base name — number appended dynamically)
    "rib" => "Rib",
    # Organs
    "liver" => "Liver", "spleen" => "Spleen", "kidney" => "Kidney",
    "pancreas" => "Pancreas", "stomach" => "Stomach",
    "gallbladder" => "Gallbladder", "esophagus" => "Esophagus",
    "colon" => "Colon", "duodenum" => "Duodenum", "small_bowel" => "Small Bowel",
    "lung_upper_lobe" => "Lung Upper Lobe", "lung_lower_lobe" => "Lung Lower Lobe",
    "lung_middle_lobe" => "Lung Middle Lobe",
    "brain" => "Brain", "prostate" => "Prostate Gland",
    "urinary_bladder" => "Urinary Bladder", "thyroid_gland" => "Thyroid Gland",
    "adrenal_gland" => "Adrenal Gland",
    # Vasculature
    "aorta" => "Aorta", "pulmonary_artery" => "Pulmonary Artery",
    "iliac_artery" => "Iliac Artery", "iliac_vena" => "Iliac Vein",
    # Heart
    "heart_myocardium" => "Heart Myocardium",
    "heart_atrium" => "Heart Atrium", "heart_ventricle" => "Heart Ventricle",
    # Muscles
    "gluteus_maximus" => "Gluteus Maximus", "gluteus_medius" => "Gluteus Medius",
    "gluteus_minimus" => "Gluteus Minimus", "psoas_major" => "Psoas Major",
    "rectus_abdominis" => "Rectus Abdominis", "autocad_muscle" => "Muscle",
    # Other
    "trachea" => "Trachea", "face" => "Face",
)

"""
    map_ts_to_anatomy(raw_ts_name) → (anatomy_label, side)

Maps raw TotalSegmentator segment names to clean anatomy labels + side.
Handles all TS naming patterns:
- Simple: "liver" → ("Liver", "")
- Sided: "femur_left" → ("Femur", "Left")
- Numbered+Sided: "rib_left_4" → ("Rib 4", "Left")
- Vertebrae: "vertebrae_T5" → ("T5 Vertebra", "")
"""
function map_ts_to_anatomy(raw_ts_name::String)
    isempty(raw_ts_name) && return ("", "")
    name_low = lowercase(strip(raw_ts_name))
    side = ""
    
    # Extract side from anywhere in name: "rib_left_4" → side="Left", core="rib__4"
    if occursin("_left", name_low)
        side = "Left"
        name_low = replace(name_low, "_left" => ""; count=1)
    elseif occursin("_right", name_low)
        side = "Right"
        name_low = replace(name_low, "_right" => ""; count=1)
    end
    
    # Clean up doubled/trailing underscores: "rib__4" → "rib_4"
    name_low = replace(name_low, "__" => "_")
    name_low = strip(name_low, '_')
    
    # 1. Exact match in static table
    haskey(TS_TO_ANATOMY, name_low) && return (TS_TO_ANATOMY[name_low], side)
    
    # 2. Strip trailing number for numbered items: "rib_4" → base="rib", num="4"
    last_us = findlast('_', name_low)
    if last_us !== nothing
        base = name_low[1:last_us-1]
        num = name_low[last_us+1:end]
        if !isempty(num) && all(isdigit, num)
            if haskey(TS_TO_ANATOMY, base)
                return ("$(TS_TO_ANATOMY[base]) $num", side)
            end
        end
    end
    
    # 3. Fallback: title-case with underscores→spaces
    return (titlecase(replace(name_low, "_" => " ")), side)
end

# ─── Persistence ─────────────────────────────────────────────────────────────

"""
    parse_lesion_id(id_str::String) -> Union{Int, Nothing}

Extract the numeric lesion ID from a display string.
Handles formats: "1: Femur", "1 - New Lesion", "1: liver [Grp 1, 3 TPs]", "1", etc.
"""
function parse_lesion_id(id_str::String)::Union{Int, Nothing}
    cp = findfirst(':', id_str)
    dash = cp === nothing ? findfirst(" - ", id_str) : nothing
    ns = if cp !== nothing
        strip(id_str[1:cp-1])
    elseif dash !== nothing
        strip(id_str[1:first(dash)-1])
    else
        strip(id_str)
    end
    return tryparse(Int, ns)
end

"""
    get_lesion_state(db::Dict, key::String) -> Dict{String,Any}

Robust lookup of lesion metadata from database `db` by exact key match or numeric lesion ID.
Handles keys like "1", "1: Femur", "1: Femur [Grp 1, 3 TPs]", etc.
"""
function get_lesion_state(db::Dict, key::String)::Dict{String,Any}
    # 1. Exact match
    if haskey(db, key)
        v = db[key]
        return v isa AbstractDict ? Dict{String,Any}(string(ik) => iv for (ik, iv) in v) : Dict{String,Any}()
    end
    
    # 2. Extract numeric ID from key (e.g. "1: femur" -> 1, "1 - New" -> 1)
    lid = parse_lesion_id(key)
    
    if lid !== nothing
        # Check canonical integer string key (e.g. "1") — preferred since autosave now uses this
        lid_str = string(lid)
        if haskey(db, lid_str)
            v = db[lid_str]
            return v isa AbstractDict ? Dict{String,Any}(string(ik) => iv for (ik, iv) in v) : Dict{String,Any}()
        end
        # Backward compat: search all display-name keys, prefer the one with most data
        best_v = Dict{String,Any}()
        best_count = 0
        for (k, v) in db
            k_lid = parse_lesion_id(k)
            if k_lid == lid && v isa AbstractDict && length(v) > best_count
                best_v = Dict{String,Any}(string(ik) => iv for (ik, iv) in v)
                best_count = length(v)
            end
        end
        return best_v
    end
    
    return Dict{String,Any}()
end

"""
    _migrate_db(raw_db::Dict) -> Dict{String,Any}

Migrate database keys to canonical numeric IDs (e.g. "1", "2").
Merges duplicate/stale entries for the same lesion ID by retaining all fields,
preferring newer/more populated entries when keys collide.
"""
function _migrate_db(raw_db::Dict)::Dict{String,Any}
    migrated = Dict{String, Any}()
    for (k, v) in raw_db
        sk = string(k)
        if startswith(sk, "_")
            migrated[sk] = v
            continue
        end
        lid = parse_lesion_id(sk)
        canonical = lid !== nothing ? string(lid) : sk
        
        if haskey(migrated, canonical)
            existing = migrated[canonical]
            if v isa AbstractDict && existing isa AbstractDict
                if length(v) > length(existing)
                    merged = Dict{String,Any}(string(ik) => iv for (ik, iv) in v)
                    for (ik, iv) in existing
                        sik = string(ik)
                        if !haskey(merged, sik); merged[sik] = iv; end
                    end
                    migrated[canonical] = merged
                else
                    merged = Dict{String,Any}(string(ik) => iv for (ik, iv) in existing)
                    for (ik, iv) in v
                        sik = string(ik)
                        if !haskey(merged, sik); merged[sik] = iv; end
                    end
                    migrated[canonical] = merged
                end
            end
        else
            migrated[canonical] = v isa AbstractDict ? Dict{String,Any}(string(ik) => iv for (ik, iv) in v) : v
        end
    end
    return migrated
end

"""
    load_annotations_hdf5(path = DEFAULT_HDF5_PATH) -> Dict{String,Dict{String,Any}}

Load all lesion metadata, button states, and global app states from HDF5 database.
"""
function load_annotations_hdf5(path::String = DEFAULT_HDF5_PATH)::Dict{String,Dict{String,Any}}
    isfile(path) || return Dict{String,Dict{String,Any}}()
    try
        out = Dict{String,Dict{String,Any}}()
        h5open(path, "r") do file
            for id in keys(file)
                obj = file[id]
                if obj isa HDF5.Group
                    inner = Dict{String,Any}()
                    for k in keys(obj)
                        try
                            val = read(obj[k])
                            # If value is a JSON string representing a dict/array, parse it
                            if val isa AbstractString && (startswith(strip(val), "{") || startswith(strip(val), "["))
                                try
                                    parsed = JSON.parse(val)
                                    inner[k] = parsed
                                catch
                                    inner[k] = val
                                end
                            else
                                inner[k] = val
                            end
                        catch e
                            @warn "Error reading HDF5 key $k in group $id: $e"
                        end
                    end
                    out[id] = inner
                end
            end
        end
        @debug "Annotations loaded from HDF5 → $path ($(length(out)) entries)"
        if !isempty(out)
            migrated = _migrate_db(out)
            needs_rewrite = any(k -> begin
                if startswith(string(k), "_"); return false; end
                lid = parse_lesion_id(string(k))
                return lid !== nothing && string(lid) != string(k)
            end, keys(out))
            if needs_rewrite
                save_annotations_hdf5(migrated, path)
            end
            return migrated
        end
        return out
    catch e
        @warn "Cannot load annotations from HDF5 $(path): $(e)"
        return Dict{String,Dict{String,Any}}()
    end
end

function load_annotations(path::String = DEFAULT_SAVE_PATH)::Dict{String,Dict{String,Any}}
    # Try loading from JSON
    out = Dict{String,Dict{String,Any}}()
    if isfile(path)
        try
            raw = JSON.parse(read(path, String))
            for (k, v) in raw
                if v isa AbstractDict
                    inner = Dict{String,Any}()
                    for (ik, iv) in v
                        inner[string(ik)] = iv
                    end
                    out[string(k)] = inner
                end
            end
        catch e
            @warn "Cannot load annotations from JSON $(path): $(e)"
        end
    end
    
    # If JSON is empty or missing, try loading from default HDF5
    if isempty(out) && isfile(DEFAULT_HDF5_PATH)
        out = load_annotations_hdf5(DEFAULT_HDF5_PATH)
    end
    
    if !isempty(out)
        migrated = _migrate_db(out)
        needs_rewrite = any(k -> begin
            if startswith(string(k), "_"); return false; end
            lid = parse_lesion_id(string(k))
            return lid !== nothing && string(lid) != string(k)
        end, keys(out))
        if needs_rewrite
            save_annotations(migrated, path)
        end
        return migrated
    end
    return out
end

function save_annotations_hdf5(db::Dict, path::String=DEFAULT_HDF5_PATH)
    try
        h5open(path, "w") do file
            for (id, lesion_data) in db
                g = create_group(file, string(id))
                if lesion_data isa AbstractDict
                    for (k, v) in lesion_data
                        if v isa AbstractArray
                            try
                                write(g, string(k), v)
                            catch
                                write(g, string(k), JSON.json(v))
                            end
                        elseif v isa AbstractDict
                            write(g, string(k), JSON.json(v))
                        else
                            write(g, string(k), string(v))
                        end
                    end
                else
                    write(g, "value", string(lesion_data))
                end
            end
        end
        @debug "Annotations saved to HDF5 → $path ($(length(db)) entries)"
    catch e
        @error "Failed to save annotations to HDF5" exception=(e, catch_backtrace())
    end
end

function save_annotations(db::Dict,
                          path::String = DEFAULT_SAVE_PATH)
    try
        open(path, "w") do io
            JSON.print(io, db, 2)
        end
        @debug "Annotations saved → $(path)  ($(length(db)) lesions)"
    catch e
        @warn "Cannot save annotations: $(e)"
    end
end

# ─── SUV Computation ─────────────────────────────────────────────────────────
"""
    compute_suv_max_at_centroid(pet_vol, centroid) -> Float32

Compute SUVmax in a 3x3x3 voxel neighborhood around the lesion centroid.
`centroid` is [x, y, z] in voxel coordinates (1-indexed).
"""
function compute_suv_max_at_centroid(pet_vol::AbstractArray{Float32, 3}, centroid::Vector{Int})::Float32
    suv_max = 0.0f0
    cx, cy, cz = centroid
    for dz in -1:1, dy in -1:1, dx in -1:1
        nx, ny, nz = cx + dx, cy + dy, cz + dz
        if 1 <= nx <= size(pet_vol, 1) && 1 <= ny <= size(pet_vol, 2) && 1 <= nz <= size(pet_vol, 3)
            val = pet_vol[nx, ny, nz]
            if val > suv_max; suv_max = val; end
        end
    end
    return suv_max
end

"""
    compute_background_suvs(pet_vol, ts_atlas, ts_names) -> Dict{String, Float32}

Compute mean SUV in reference organs (liver, parotid, blood pool) using
TotalSegmentator atlas overlay on the PET volume.
Returns Dict("liver" => mean, "parotid" => mean, "blood" => mean).
"""
function compute_background_suvs(pet_vol::AbstractArray{Float32, 3},
                                  ts_atlas::AbstractArray,
                                  ts_names::Dict{Int, String})::Dict{String, Float32}
    result = Dict{String, Float32}("liver" => 0.0f0, "parotid" => 0.0f0, "blood" => 0.0f0)
    target_keywords = Dict("liver" => ["liver"],
                           "parotid" => ["parotid"],
                           "blood" => ["vena_cava", "aorta", "blood", "heart"])
    
    organ_labels = Dict{String, Vector{Int}}("liver" => Int[], "parotid" => Int[], "blood" => Int[])
    for (label_id, name) in ts_names
        name_low = lowercase(name)
        for (organ_key, keywords) in target_keywords
            if any(kw -> occursin(kw, name_low), keywords)
                push!(organ_labels[organ_key], label_id)
            end
        end
    end
    
    label_to_organ = Dict{Int, String}()
    for (organ_key, label_ids) in organ_labels
        for lid in label_ids
            label_to_organ[lid] = organ_key
        end
    end
    
    # 1. Collect coords in TS space in ONE fast pass
    pts_dict = Dict{String, Vector{CartesianIndex{3}}}("liver" => [], "parotid" => [], "blood" => [])
    for i in CartesianIndices(ts_atlas)
        val = Int(ts_atlas[i])
        if val > 0 && haskey(label_to_organ, val)
            push!(pts_dict[label_to_organ[val]], i)
        end
    end
    
    needs_scale = size(ts_atlas) != size(pet_vol)
    sx = needs_scale ? size(pet_vol, 1) / size(ts_atlas, 1) : 1.0
    sy = needs_scale ? size(pet_vol, 2) / size(ts_atlas, 2) : 1.0
    sz = needs_scale ? size(pet_vol, 3) / size(ts_atlas, 3) : 1.0
    
    for (organ_key, pts) in pts_dict
        isempty(pts) && continue
        kept_pts = CartesianIndex{3}[]
        
        if organ_key == "blood"
            z_vals = [p[3] for p in pts]
            z_min, z_max = minimum(z_vals), maximum(z_vals)
            # Center 1/3 of the blood vessel (descending aorta) or heart
            z_start = round(Int, z_min + (z_max - z_min)/3)
            z_end = round(Int, z_max - (z_max - z_min)/3)
            
            z_groups = Dict{Int, Vector{CartesianIndex{3}}}()
            for p in pts
                if z_start <= p[3] <= z_end
                    push!(get!(z_groups, p[3], CartesianIndex{3}[]), p)
                end
            end
            
            # For each slice, grab exactly the 1cm center (radius ~4-6mm)
            for (z, z_pts) in z_groups
                cx = sum(p[1] for p in z_pts) / length(z_pts)
                cy = sum(p[2] for p in z_pts) / length(z_pts)
                for p in z_pts
                    if (p[1] - cx)^2 + (p[2] - cy)^2 <= 16
                        push!(kept_pts, p)
                    end
                end
            end
            
        elseif organ_key == "liver"
            # Sphere radius 30mm directly in the centroid of the liver
            cx = sum(p[1] for p in pts) / length(pts)
            cy = sum(p[2] for p in pts) / length(pts)
            cz = sum(p[3] for p in pts) / length(pts)
            for p in pts
                if (p[1] - cx)^2 + (p[2] - cy)^2 + (p[3] - cz)^2 <= 625
                    push!(kept_pts, p)
                end
            end
            
        elseif organ_key == "parotid"
            # Separate left and right, take center of each
            cx_all = sum(p[1] for p in pts) / length(pts)
            left_pts = [p for p in pts if p[1] < cx_all]
            right_pts = [p for p in pts if p[1] >= cx_all]
            
            for sub_pts in (left_pts, right_pts)
                if !isempty(sub_pts)
                    cx = sum(p[1] for p in sub_pts) / length(sub_pts)
                    cy = sum(p[2] for p in sub_pts) / length(sub_pts)
                    cz = sum(p[3] for p in sub_pts) / length(sub_pts)
                    for p in sub_pts
                        if (p[1] - cx)^2 + (p[2] - cy)^2 + (p[3] - cz)^2 <= 9
                            push!(kept_pts, p)
                        end
                    end
                end
            end
        end
        
        # 2. Average in PET space
        total_val = 0.0
        for p in kept_pts
            px = needs_scale ? clamp(round(Int, p[1] * sx), 1, size(pet_vol, 1)) : p[1]
            py = needs_scale ? clamp(round(Int, p[2] * sy), 1, size(pet_vol, 2)) : p[2]
            pz = needs_scale ? clamp(round(Int, p[3] * sz), 1, size(pet_vol, 3)) : p[3]
            total_val += pet_vol[px, py, pz]
        end
        
        if !isempty(kept_pts)
            result[organ_key] = Float32(total_val / length(kept_pts))
        end
    end
    return result
end

# Background SUVs cache (computed once per TP, reused for all lesions)
const _bg_suv_cache = Dict{Int, Dict{String, Float32}}()
const _lesion_suv_cache = Dict{Tuple{Int,Int}, String}()  # (tp_idx, lesion_id) → SUV string

"""
    get_background_suvs(tp_idx) -> Dict{String, Float32}

Get cached or compute background SUVs for the given time point.
"""
function get_background_suvs(tp_idx::Int)::Dict{String, Float32}
    haskey(_bg_suv_cache, tp_idx) && return _bg_suv_cache[tp_idx]
    pet_vol = get(_MEH.pet_volumes_cache, tp_idx, nothing)
    ts_atlas = _MEH.global_ts_atlas[]
    ts_names = _MEH.global_ts_names[]
    if pet_vol === nothing || ts_atlas === nothing || isempty(ts_names)
        return Dict{String, Float32}("liver" => 0.0f0, "parotid" => 0.0f0, "blood" => 0.0f0)
    end
    bg = compute_background_suvs(pet_vol, ts_atlas, ts_names)
    _bg_suv_cache[tp_idx] = bg
    return bg
end

"""
    compute_lesion_suv_string(lesion_id, tp_idx) -> String

Auto-compute SUV max and background references, returning formatted string:
"Max: X.X ; Parotid: X.X ; Liver: X.X ; Blood: X.X"
"""
function compute_lesion_suv_string(lesion_id::Int, tp_idx::Int)::String
    pet_vol = get(_MEH.pet_volumes_cache, tp_idx, nothing)
    centroid = if haskey(_MEH.lesion_centroids_cache, (tp_idx, lesion_id))
        _MEH.lesion_centroids_cache[(tp_idx, lesion_id)]
    elseif haskey(_MEH.lesion_centroids_cache, lesion_id)
        _MEH.lesion_centroids_cache[lesion_id]
    else
        nothing
    end
    if pet_vol === nothing || centroid === nothing
        return ""
    end
    suv_max = compute_suv_max_at_centroid(pet_vol, centroid)
    bg = get_background_suvs(tp_idx)
    return "Max: $(round(suv_max, digits=1)) ; Parotid: $(round(bg["parotid"], digits=1)) ; Liver: $(round(bg["liver"], digits=1)) ; Blood: $(round(bg["blood"], digits=1))"
end

"""
    generate_tracking_name(lesion_id, organ_name, tp_idx, modality, patient_id) -> String

Auto-generate a lesion tracking name like "Liver_L1_PET_TP0_PAT123"
"""
function generate_tracking_name(lesion_id::Int, organ_name::String, tp_idx::Int,
                                 modality::String, patient_id_str::String)::String
    organ_clean = isempty(organ_name) ? "Lesion" : replace(strip(organ_name), " " => "_")
    return "$(organ_clean)_L$(lesion_id)_$(modality)_TP$(tp_idx)_$(patient_id_str)"
end
"""
    parse_suv_fields(suv_str) -> Dict{String,Float32}

Parse the formatted SUV string "Max: X.X ; Parotid: X.X ; Liver: X.X ; Blood: X.X"
into a Dict with keys "max", "parotid", "liver", "blood".
"""
function parse_suv_fields(suv_str::String)::Dict{String, Float32}
    result = Dict{String, Float32}()
    for part in split(suv_str, ";")
        sp = strip(part)
        colon = findfirst(':', sp)
        colon === nothing && continue
        key = lowercase(strip(sp[1:colon-1]))
        val = tryparse(Float32, strip(sp[colon+1:end]))
        val !== nothing && (result[key] = val)
    end
    return result
end

"""
    compute_promise_score(suv_max, bg_suvs) -> Int

PROMISE scoring: 0=<blood, 1=blood≤x<liver, 2=liver≤x<parotid, 3=≥parotid.
Returns -1 if references are missing/zero.
"""
function compute_promise_score(suv_max::Float32, bg::Dict{String, Float32})::Int
    blood = get(bg, "blood", 0.0f0)
    liver = get(bg, "liver", 0.0f0)
    parotid = get(bg, "parotid", 0.0f0)
    (blood <= 0 || liver <= 0 || parotid <= 0) && return -1
    suv_max < blood  && return 0
    suv_max < liver  && return 1
    suv_max < parotid && return 2
    return 3
end

"""
    compute_suv_comparison_string(suv_max, bg_suvs) -> String

Generate human-readable SUV comparison:
"SUVmax X.X > Liver (Y.Y) — PROMISE 2" or "SUVmax X.X < Liver (Y.Y) — PROMISE 1"
"""
function compute_suv_comparison_string(suv_max::Float32, bg::Dict{String, Float32})::String
    liver = get(bg, "liver", 0.0f0)
    parotid = get(bg, "parotid", 0.0f0)
    blood = get(bg, "blood", 0.0f0)
    promise = compute_promise_score(suv_max, bg)
    
    parts = String[]
    if liver > 0
        cmp = suv_max >= liver ? "≥" : "<"
        push!(parts, "Liver: $(cmp) ($(round(suv_max/liver, digits=2))×)")
    end
    if parotid > 0
        cmp = suv_max >= parotid ? "≥" : "<"
        push!(parts, "Parotid: $(cmp) ($(round(suv_max/parotid, digits=2))×)")
    end
    if blood > 0
        cmp = suv_max >= blood ? "≥" : "<"
        push!(parts, "Blood: $(cmp) ($(round(suv_max/blood, digits=2))×)")
    end
    promise_str = promise >= 0 ? " | PROMISE $promise" : ""
    return join(parts, " ; ") * promise_str
end

# ─── Volume Computation ──────────────────────────────────────────────────────

# Cache: (tp_idx, lid) → Dict("volume_mm3" => ..., "volume_cc" => ..., "voxel_count" => ...)
const _volume_cache = Dict{Tuple{Int,Int}, Dict{String, Float64}}()

"""
    compute_lesion_volume(lid, tp_idx) -> Dict{String, Float64}

Compute lesion volume from the cached mask at a specific time point.
Uses display-space voxel spacing from Main.first_spacing.
Returns Dict("volume_mm3", "volume_cc", "voxel_count", "diameter_mm").
"""
function compute_lesion_volume(lid::Int, tp_idx::Int)::Dict{String, Float64}
    cache_key = (tp_idx, lid)
    haskey(_volume_cache, cache_key) && return _volume_cache[cache_key]
    
    result = Dict{String, Float64}("volume_mm3" => 0.0, "volume_cc" => 0.0, 
                                     "voxel_count" => 0.0, "diameter_mm" => 0.0)
    try
        entry = _MEH.get_or_load_tp_data(tp_idx)
        entry === nothing && return result
        mask = entry.mask
        
        # Count voxels with this label
        voxel_count = count(x -> x == lid, mask)
        voxel_count == 0 && return result
        
        # Get spacing: display space = native / HIRES_FACTOR for x,y
        native_spacing = try
            Main.first_spacing
        catch
            (1.0, 1.0, 1.0)
        end
        hires_factor = try
            Main.HIRES_FACTOR
        catch
            2.0
        end
        
        # Display spacing
        sp_x = native_spacing[1] / hires_factor
        sp_y = native_spacing[2] / hires_factor
        sp_z = native_spacing[3]
        
        voxel_vol_mm3 = sp_x * sp_y * sp_z
        volume_mm3 = voxel_count * voxel_vol_mm3
        volume_cc = volume_mm3 / 1000.0
        
        # Equivalent sphere diameter
        diameter_mm = 2.0 * (3.0 * volume_mm3 / (4.0 * π))^(1.0/3.0)
        
        result["volume_mm3"] = volume_mm3
        result["volume_cc"] = volume_cc
        result["voxel_count"] = Float64(voxel_count)
        result["diameter_mm"] = diameter_mm
    catch e
        @warn "Volume computation failed for lesion $lid at TP $tp_idx: $e"
    end
    
    _volume_cache[cache_key] = result
    return result
end

"""
    precompute_all_volumes!(mask_vol, tp_idx)

Single O(N) pass over the mask volume to accumulate voxel counts for ALL lesion IDs.
Populates `_volume_cache` so that subsequent `compute_lesion_volume` calls are O(1) cache hits.
Should be called once per TP at load time (alongside `precompute_mask_centroids!`).
"""
function precompute_all_volumes!(mask_vol::AbstractArray{T, 3}, tp_idx::Int) where T
    # Single pass: count voxels per label
    voxel_counts = Dict{Int, Int}()
    @inbounds for v in mask_vol
        iv = Int(round(v))
        if iv > 0
            voxel_counts[iv] = get(voxel_counts, iv, 0) + 1
        end
    end
    
    isempty(voxel_counts) && return
    
    # Get spacing
    native_spacing = try
        Main.first_spacing
    catch
        (1.0, 1.0, 1.0)
    end
    hires_factor = try
        Main.HIRES_FACTOR
    catch
        2.0
    end
    sp_x = native_spacing[1] / hires_factor
    sp_y = native_spacing[2] / hires_factor
    sp_z = native_spacing[3]
    voxel_vol_mm3 = sp_x * sp_y * sp_z
    
    for (lid, vc) in voxel_counts
        volume_mm3 = vc * voxel_vol_mm3
        volume_cc = volume_mm3 / 1000.0
        diameter_mm = 2.0 * (3.0 * volume_mm3 / (4.0 * π))^(1.0/3.0)
        _volume_cache[(tp_idx, lid)] = Dict{String, Float64}(
            "volume_mm3" => volume_mm3,
            "volume_cc" => volume_cc,
            "voxel_count" => Float64(vc),
            "diameter_mm" => diameter_mm
        )
    end
    @info "[PERF] Precomputed volumes for $(length(voxel_counts)) lesions at TP $tp_idx"
end

# ─── Match Analysis ──────────────────────────────────────────────────────────

"""
    MatchAnalysisResult

Contains volume/SUV comparison data for a lesion tracked across time points.
"""
struct MatchAnalysisResult
    group_id::Int
    current_volume_mm3::Float64
    current_volume_cc::Float64
    current_diameter_mm::Float64
    current_suv_max::Float32
    # Baseline comparison
    baseline_volume_mm3::Float64
    baseline_volume_cc::Float64
    baseline_suv_max::Float32
    baseline_node::String
    baseline_lid::Int
    # Delta values
    volume_delta_pct::Float64    # (current - baseline) / baseline * 100
    volume_delta_abs_cc::Float64  # current - baseline in cc
    suv_delta_abs::Float32       # current - baseline SUVmax
    suv_delta_pct::Float64       # (current - baseline) / baseline * 100
    # RECIP classification
    recip_category::String       # CR, PR, SD, PD, or N/A
    n_timepoints::Int            # total TPs in this match group
end

"""
    compute_match_analysis(lid, tp_idx) -> Union{MatchAnalysisResult, Nothing}

Compute longitudinal match analysis for a lesion. Finds the earliest
time point in the same match group (baseline), computes volume and SUV
at both time points, and calculates deltas.
"""
function compute_match_analysis(lid::Int, tp_idx::Int)::Union{MatchAnalysisResult, Nothing}
    # Find match group for this lesion
    current_node = _MEH.get_node_name_for_tp(tp_idx)
    match_groups = LA.get_match_groups()
    
    group_id = nothing
    members = nothing
    for (gid, mems) in match_groups
        if any(m -> m[1] == current_node && m[2] == lid, mems)
            group_id = gid
            members = mems
            break
        end
    end
    
    group_id === nothing && return nothing
    length(members) < 2 && return nothing
    
    # Find baseline (earliest TP in group)
    sorted_members = sort(members, by = m -> LA._tp_index_from_node(m[1]))
    baseline_member = sorted_members[1]
    baseline_node = baseline_member[1]
    baseline_lid = baseline_member[2]
    baseline_tp_idx = LA._tp_index_from_node(baseline_node)
    
    # Skip if current IS the baseline
    if baseline_node == current_node && baseline_lid == lid
        # Still report volume but no delta
        cur_vol = compute_lesion_volume(lid, tp_idx)
        cur_suv = _get_suv_max(lid, tp_idx)
        return MatchAnalysisResult(
            group_id,
            cur_vol["volume_mm3"], cur_vol["volume_cc"], cur_vol["diameter_mm"], cur_suv,
            0.0, 0.0, 0.0f0, "", 0,
            0.0, 0.0, 0.0f0, 0.0,
            "N/A (baseline)", length(members)
        )
    end
    
    # Compute volumes
    cur_vol = compute_lesion_volume(lid, tp_idx)
    base_vol = compute_lesion_volume(baseline_lid, baseline_tp_idx)
    
    # Compute SUVmax
    cur_suv = _get_suv_max(lid, tp_idx)
    base_suv = _get_suv_max(baseline_lid, baseline_tp_idx)
    
    # Volume delta
    vol_delta_pct = base_vol["volume_cc"] > 0.001 ?
        (cur_vol["volume_cc"] - base_vol["volume_cc"]) / base_vol["volume_cc"] * 100.0 : 0.0
    vol_delta_abs = cur_vol["volume_cc"] - base_vol["volume_cc"]
    
    # SUV delta
    suv_delta_abs = cur_suv - base_suv
    suv_delta_pct = base_suv > 0.1f0 ?
        Float64((cur_suv - base_suv) / base_suv * 100) : 0.0
    
    # RECIP classification based on volume change
    recip = if cur_vol["volume_cc"] < 0.001 && base_vol["volume_cc"] > 0.001
        "RECIP-CR"  # Complete Response (disappeared)
    elseif vol_delta_pct < -30.0
        "RECIP-PR"  # Partial Response (>30% decrease)
    elseif vol_delta_pct > 20.0
        "RECIP-PD"  # Progressive Disease (>20% increase)
    else
        "RECIP-SD"  # Stable Disease
    end
    
    return MatchAnalysisResult(
        group_id,
        cur_vol["volume_mm3"], cur_vol["volume_cc"], cur_vol["diameter_mm"], cur_suv,
        base_vol["volume_mm3"], base_vol["volume_cc"], base_suv, baseline_node, baseline_lid,
        vol_delta_pct, vol_delta_abs, suv_delta_abs, suv_delta_pct,
        recip, length(members)
    )
end

"""
    _get_suv_max(lid, tp_idx) -> Float32

Get SUVmax for a lesion, using cache if available.
"""
function _get_suv_max(lid::Int, tp_idx::Int)::Float32
    cache_key = (tp_idx, lid)
    cached = get(_lesion_suv_cache, cache_key, "")
    if !isempty(cached)
        fields = parse_suv_fields(cached)
        return get(fields, "max", 0.0f0)
    end
    # Compute fresh
    pet_vol = get(_MEH.pet_volumes_cache, tp_idx, nothing)
    centroid = if haskey(_MEH.lesion_centroids_cache, (tp_idx, lid))
        _MEH.lesion_centroids_cache[(tp_idx, lid)]
    elseif haskey(_MEH.lesion_centroids_cache, lid)
        _MEH.lesion_centroids_cache[lid]
    else
        nothing
    end
    (pet_vol === nothing || centroid === nothing) && return 0.0f0
    return compute_suv_max_at_centroid(pet_vol, centroid)
end

"""
    format_match_analysis(result::MatchAnalysisResult) -> String

Format match analysis result for display in the metadata panel.
"""
function format_match_analysis(r::MatchAnalysisResult)::String
    parts = String[]
    push!(parts, "Vol: $(round(r.current_volume_cc, digits=2))cc ($(round(r.current_diameter_mm, digits=1))mm⌀)")
    
    if r.baseline_volume_cc > 0.001
        sign = r.volume_delta_pct >= 0 ? "+" : ""
        push!(parts, "ΔVol: $(sign)$(round(r.volume_delta_pct, digits=1))% ($(sign)$(round(r.volume_delta_abs_cc, digits=2))cc)")
    end
    
    if r.baseline_suv_max > 0.1f0
        sign = r.suv_delta_abs >= 0 ? "+" : ""
        push!(parts, "ΔSUV: $(sign)$(round(r.suv_delta_abs, digits=1)) ($(sign)$(round(r.suv_delta_pct, digits=1))%)")
    end
    
    push!(parts, "Grp $(r.group_id) [$(r.n_timepoints) TPs] $(r.recip_category)")
    
    return join(parts, " | ")
end

# ─── UI helpers ──────────────────────────────────────────────────────────────
function all_categories(schema::Vector{QuestionDef})::Vector{String}
    # Fixed order to match the Slicer extension layout
    SLICER_ORDER = [
        "Technical Parameters",
        "Identification",
        "Location",
        "Morphology",
        "Quantitative",
        "Clinical Context",
        "Prostate Analysis",
        "Final Assessment",
        "Reporting",
    ]
    cats_in_schema = Set{String}()
    for q in schema, c in q.categories; push!(cats_in_schema, c) end
    # Use Slicer order, then append any extra categories
    result = String[]
    for cat in SLICER_ORDER
        cat in cats_in_schema && push!(result, cat)
    end
    for cat in sort(collect(cats_in_schema))
        cat in result || push!(result, cat)
    end
    return result
end

# ─── SearchableMenu: Menu with type-to-filter ────────────────────────────────
# A factory that wraps a standard Menu with per-instance keyboard handlers.
# When the dropdown is open, typing filters options in real time.
# Returns the Menu itself so `.i_selected[]` / `.selection[]` code is unchanged.

"""
    searchable_menu(g, row, cols; options, default=1, fontsize=10)

Create a searchable dropdown.  Returns a standard **Menu** widget.

When the dropdown is open, typing characters filters the visible options.
Backspace removes a character from the filter. Escape resets the filter.
The Menu prompt shows the current filter text ("Filter: query").
Closing the dropdown restores all options.
"""
function searchable_menu(g, row, cols;
        options,
        default = 1,
        fontsize = 10)

    # Snapshot original full options
    opts_vec = options isa Observable ? copy(options[]) : collect(String, options)
    isempty(opts_vec) && (opts_vec = ["(none)"])
    all_opts = Ref(opts_vec)

    # Standard Menu — button respects grid column width; use type-to-filter for long options
    menu = Menu(g[row, cols];
        options  = copy(all_opts[]),
        default  = default,
        fontsize = fontsize)

    filter_buf = Ref("")

    # ── When menu opens → reset filter; when it closes → restore options ──
    on(menu.is_open) do is_open
        if is_open
            filter_buf[] = ""
        else
            # Restore full options and reset prompt
            menu.options[] = all_opts[]
            menu.prompt[]  = "Select..."
            filter_buf[]   = ""
        end
    end

    # ── Keyboard: type to filter (scoped to this menu's blockscene) ──
    bscene = menu.blockscene

    on(bscene, events(bscene).unicode_input, priority = 100) do char
        !menu.is_open[] && return Consume(false)
        filter_buf[] *= string(char)
        _apply_searchable_filter!(menu, filter_buf[], all_opts[])
        return Consume(true)
    end

    on(bscene, events(bscene).keyboardbutton, priority = 100) do event
        !menu.is_open[] && return Consume(false)
        act = event.action
        (act == Makie.Keyboard.press || act == Makie.Keyboard.repeat) || return Consume(false)

        if event.key == Makie.Keyboard.backspace && !isempty(filter_buf[])
            filter_buf[] = filter_buf[][1:prevind(filter_buf[], end)]
            _apply_searchable_filter!(menu, filter_buf[], all_opts[])
            return Consume(true)
        elseif event.key == Makie.Keyboard.escape
            filter_buf[] = ""
            _apply_searchable_filter!(menu, "", all_opts[])
            return Consume(true)
        end
        return Consume(false)
    end

    # ── Track Observable option changes ──
    if options isa Observable
        on(options) do new_opts
            all_opts[] = collect(String, new_opts)
            if !menu.is_open[]
                menu.options[] = all_opts[]
            end
        end
    end

    return menu
end

"""Apply the current filter text to a searchable menu's options."""
function _apply_searchable_filter!(menu, query::String, full_opts::Vector{String})
    if isempty(query)
        menu.options[] = full_opts
        menu.prompt[]  = "Select..."
    else
        q = lowercase(query)
        filtered = filter(s -> occursin(q, lowercase(s)), full_opts)
        if isempty(filtered)
            menu.options[] = ["(no match for '$query')"]
        else
            menu.options[] = filtered
        end
        menu.prompt[] = "Filter: $query"
    end
end

# ─── Channel Proxy for deferred connection (parallel startup) ────────────────

"""
    ChannelProxy

Wraps a `Ref{Union{Channel, Nothing}}` so that `put!(proxy, event)` silently
drops events when the channel is not yet connected (= `nothing`).
This enables creating the Makie GUI before the Vulkan display channel exists.
"""
struct ChannelProxy
    ref::Ref{Union{Base.Channel, Nothing}}
end

function Base.put!(proxy::ChannelProxy, event)
    ch = proxy.ref[]
    if ch !== nothing
        put!(ch, event)
    end
    # Silently drop if channel not yet connected
end

"""Holds references returned by create_metadata_window for later channel connection."""
mutable struct MetadataWindowResult
    fig::Any
    channel_ref::Ref{Union{Base.Channel, Nothing}}
    lesion_db::Any
    MetadataWindowResult(fig, ch_ref, ldb=nothing) = new(fig, ch_ref, ldb)
end

"""Connect a live channel to a previously created metadata window (greyed-out → active)."""
function connect_channel!(win::MetadataWindowResult, ch::Base.Channel)
    win.channel_ref[] = ch
    println("  [MAKIE] Channel connected — all controls now active"); flush(stdout)
end

export connect_channel!, MetadataWindowResult

# ─── Main window ─────────────────────────────────────────────────────────────
function create_metadata_window(
        active_lesion_id::Observable{String},
        lesion_ids::Observable{Vector{String}},
        channel_arg::Union{Base.Channel, Nothing};
        save_path::String = DEFAULT_SAVE_PATH,
        ui_hooks::Dict{Symbol, Observable} = Dict{Symbol, Observable}())
    # Wrap channel in Ref for deferred connection (parallel startup)
    channel_ref = Ref{Union{Base.Channel, Nothing}}(channel_arg)
    # Proxy that silently drops events when channel is not yet connected
    channel = ChannelProxy(channel_ref)
    local _build_match_display!
    schema   = load_schema()
    radlex   = load_radlex()
    anatomy_ontology = load_anatomy_ontology()  # FoundationalAnatomy labels for Base Anatomy autocomplete
    all_cats = all_categories(schema)

    # In-memory DB
    lesion_db = Observable{Dict}(Dict{String,Dict{String,Any}}())
    _db_dirty = Ref(false)

    db_channel = Channel{Any}(32)
    @async begin
        for msg in db_channel
            if msg isa SaveDBMessage
                try
                    db_to_save = copy(msg.db)
                    save_annotations(db_to_save, msg.path_json)
                    save_annotations_hdf5(db_to_save, msg.path_hdf5)
                catch e
                    @warn "Database save failed" e
                end
            elseif msg isa LoadDBMessage
                try
                    db = load_annotations(msg.path_json)
                    if isempty(db) && isfile(msg.path_hdf5)
                        db = load_annotations_hdf5(msg.path_hdf5)
                    end
                    put!(msg.reply_channel, db)
                catch e
                    @warn "Database load failed" e
                    put!(msg.reply_channel, Dict{String,Dict{String,Any}}())
                end
            end
        end
    end
    
    # Load initial db asynchronously with migration to canonical numeric keys
    @async begin
        reply = Channel{Dict}(1)
        put!(db_channel, LoadDBMessage(save_path, reply, DEFAULT_HDF5_PATH))
        raw_db = take!(reply)
        migrated = _migrate_db(raw_db)
        n_before = length(raw_db)
        n_after = length(migrated)
        if n_before != n_after
            println("  [DB] Migrated $n_before entries → $n_after canonical keys"); flush(stdout)
            _db_dirty[] = true
            try
                put!(db_channel, SaveDBMessage(migrated, Dict{String, Any}(), save_path, DEFAULT_HDF5_PATH))
            catch; end
        end
        lesion_db[] = migrated
    end

    # Helper for safely extracting and stripping text from Makie Textboxes
    _safe_strip(x) = x === nothing ? "" : String(strip(x))

    # Helper for safely updating Textbox displayed and stored values simultaneously
    function _set_tb_val!(tb::Textbox, val)
        v = _safe_strip(val)
        tb.displayed_string[] = v
        tb.stored_string[] = v
    end

    # ── Theme ──────────────────────────────────────────────────────────────
    BG      = RGBf(0.10, 0.12, 0.15)
    BG_PNL  = RGBf(0.13, 0.15, 0.19)
    ACCENT  = RGBf(0.20, 0.60, 1.00)
    TXT     = :white
    SUBTXT  = RGBf(0.70, 0.75, 0.80)
    GRN     = RGBf(0.15, 0.45, 0.15)
    RED_BTN = RGBf(0.50, 0.10, 0.10)
    BLU_BTN = RGBf(0.20, 0.35, 0.60)
    SEC_HDR = RGBf(0.16, 0.18, 0.22)  # slightly lighter bg for section headers
    LBL_FG  = RGBf(0.90, 0.92, 0.95)  # high-contrast label text on dark bg

    # ── Figure ──────────────────────────────────────────────────────────────
    fig = Figure(size = (920, 900), backgroundcolor = BG, figure_padding = 0)
    
    main_layout = GridLayout(fig[1,1])
    sl = Slider(main_layout[1, 2], range = 0:0.01:1, startvalue = 1, horizontal = false, tellheight = false)
    
    g = GridLayout(main_layout[1,1], tellheight = false, halign = :left, valign = sl.value)
    
    # Mouse scroll event to control slider
    on(fig.scene.events.scroll) do scroll
        # Allow scrolling more freely to avoid being locked out by layout computation glitches
        content_h = g.layoutobservables.computedbbox[].widths[2]
        window_h = size(fig.scene)[2]
        
        if content_h > window_h * 0.5  # relaxed threshold
            sl.value[] = clamp(sl.value[] + scroll[2] * 0.05, 0.0, 1.0)
        else
            sl.value[] = 1.0
        end
        return Consume(true)
    end
    
    rowgap!(g, 2)   # compact spacing: 2px gap between rows (prevents Menu/Slider overlap)
    colgap!(g, 2)   # compact columns
    colsize!(g, 1, Auto())
    r = [0]  # row counter as array for mutation in closures
    nr!() = (r[1] += 1; r[1])

    # Registry of rows with explicit Fixed heights (Menu, Slider, Textbox rows).
    # set_row_visible! uses this to restore the correct Fixed height instead of Auto()
    # when re-showing a row after it was hidden (e.g., during compare mode toggle).
    _row_fixed_heights = Dict{Int, Int}()  # row_index => pixel_height
    function register_fixed_row!(row_idx::Int, height::Int)
        _row_fixed_heights[row_idx] = height
    end

    # Header removed for compactness — title was decorative only

    # ── Section helper ──────────────────────────────────────────────────────
    function begin_section!(title; default_open=true)
        is_open = Observable(default_open)
        header_r = nr!()
        btn = Button(g[header_r, 1:4], label = @lift($is_open ? "▼  $(title)" : "▶  $(title)"),
            buttoncolor = SEC_HDR, labelcolor = ACCENT, fontsize = 11, halign = :left)
        
        start_row = r[1] + 1
        end_row_ref = Ref(start_row)  # updated by end_section!
        return (is_open, start_row, header_r, btn, end_row_ref)
    end
    
    function end_section!(sec_data)
        is_open, start_row, header_r, btn, end_row_ref = sec_data
        end_row = r[1]
        end_row_ref[] = end_row
        
        # If default_open is false, collapse immediately
        if !is_open[]
            for i in start_row:end_row
                rowsize!(g, i, Fixed(0))
            end
            for i in (start_row > 1 ? start_row - 1 : start_row):min(end_row, r[1] - 1)
                rowgap!(g, i, 0)
            end
            for c in g.content
                if c.span.rows.start >= start_row && c.span.rows.stop <= end_row
                    if hasproperty(c.content, :blockscene)
                        c.content.blockscene.visible[] = false
                    end
                end
            end
        end
        
        on(btn.clicks) do _
            is_open[] = !is_open[]
            for i in start_row:end_row
                if is_open[]
                    rowsize!(g, i, Auto())
                else
                    rowsize!(g, i, Fixed(0))
                end
            end
            
            # Zero/restore row gaps to eliminate empty space between collapsed headers
            for i in (start_row > 1 ? start_row - 1 : start_row):min(end_row, r[1] - 1)
                rowgap!(g, i, is_open[] ? 2 : 0)
            end
            
            for c in g.content
                if c.span.rows.start >= start_row && c.span.rows.stop <= end_row
                    if hasproperty(c.content, :blockscene)
                        c.content.blockscene.visible[] = is_open[]
                    end
                end
            end
        end
    end
    
    # Hide/show an entire section (header + content) for compare mode
    function hide_section!(sec_data)
        _, start_row, header_r, _, end_row_ref = sec_data
        for i in header_r:end_row_ref[]
            set_row_visible!(i, false)
        end
    end
    function show_section!(sec_data)
        is_open, start_row, header_r, _, end_row_ref = sec_data
        set_row_visible!(header_r, true)
        if is_open[]
            for i in start_row:end_row_ref[]
                set_row_visible!(i, true)
            end
        end
    end

    # ── Cursor Info (always visible, not in a section) ───────────────────────
    # Study name — compare mode shows both studies
    Label(g[nr!(), 1:4], @lift(string($(_MEH.cursor_study_text))),
        fontsize = 11, color = RGBAf(0.6, 0.8, 1.0, 1.0), halign = :left)
    # HU / SUV / Lesion / View / Slice
    Label(g[nr!(), 1:4], @lift(string($(_MEH.cursor_info_text))),
        fontsize = 11, color = RGBAf(0.95, 0.85, 0.55, 1.0), halign = :left)

    # ── Lesion Navigation ────────────────────────────────────────────────────
    sec_nav = begin_section!("Lesion Navigation")
    nav_r = nr!()
    btn_prev = Button(g[nav_r, 1], label = "<< Prev",
        buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    les_menu = Menu(g[nav_r, 2:3],
        options = @lift(isempty($lesion_ids) ? ["(none)"] : $lesion_ids),
        fontsize = 10)
    btn_next = Button(g[nav_r, 4], label = "Next >>",
        buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    rowsize!(g, nav_r, Fixed(28)); register_fixed_row!(nav_r, 28)

    # Active lesion display kept for callbacks (no visible label — shown in dropdown)
    active_lesion_display = Observable{String}("(none)")

    # Prominent Lesion & Bone Subsegments & Anatomy Layer Visibility Controls
    vis_row = nr!()
    vis_lesion_active = Ref(true)
    vis_surface_active = Ref(true)
    vis_marrow_active = Ref(true)
    vis_anatomy_active = Ref(false)
    
    btn_vis_lesion  = Button(g[vis_row, 1], label = "Lesion: ON",   buttoncolor = GRN, labelcolor = TXT, fontsize = 9)
    btn_vis_surface = Button(g[vis_row, 2], label = "Surf: ON",     buttoncolor = RGBf(0.0, 0.75, 0.75), labelcolor = TXT, fontsize = 9)
    btn_vis_marrow  = Button(g[vis_row, 3], label = "Marrow: ON",   buttoncolor = RGBf(0.75, 0.75, 0.1), labelcolor = TXT, fontsize = 9)
    btn_vis_anatomy = Button(g[vis_row, 4], label = "Anatomy: OFF", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 9)

    on(btn_vis_lesion.clicks) do _
        vis_lesion_active[] = !vis_lesion_active[]
        btn_vis_lesion.label[] = vis_lesion_active[] ? "Lesion: ON" : "Lesion: OFF"
        btn_vis_lesion.buttoncolor[] = vis_lesion_active[] ? GRN : BG_PNL
        @info "BTN_VIS_LESION clicked: $(vis_lesion_active[])"
        put!(channel, ShowMaskLayerEvent(1, vis_lesion_active[]))
    end
    
    on(btn_vis_surface.clicks) do _
        vis_surface_active[] = !vis_surface_active[]
        btn_vis_surface.label[] = vis_surface_active[] ? "Surf: ON" : "Surf: OFF"
        btn_vis_surface.buttoncolor[] = vis_surface_active[] ? RGBf(0.0, 0.75, 0.75) : BG_PNL
        @info "BTN_VIS_SURFACE clicked: $(vis_surface_active[])"
        put!(channel, ShowMaskLayerEvent(2, vis_surface_active[]))
    end
    
    on(btn_vis_marrow.clicks) do _
        vis_marrow_active[] = !vis_marrow_active[]
        btn_vis_marrow.label[] = vis_marrow_active[] ? "Marrow: ON" : "Marrow: OFF"
        btn_vis_marrow.buttoncolor[] = vis_marrow_active[] ? RGBf(0.75, 0.75, 0.1) : BG_PNL
        @info "BTN_VIS_MARROW clicked: $(vis_marrow_active[])"
        put!(channel, ShowMaskLayerEvent(3, vis_marrow_active[]))
    end

    on(btn_vis_anatomy.clicks) do _
        vis_anatomy_active[] = !vis_anatomy_active[]
        btn_vis_anatomy.label[] = vis_anatomy_active[] ? "Anatomy: ON" : "Anatomy: OFF"
        btn_vis_anatomy.buttoncolor[] = vis_anatomy_active[] ? RGBf(0.5, 0.0, 0.8) : BG_PNL
        @info "BTN_VIS_ANATOMY clicked: $(vis_anatomy_active[])"
        put!(channel, ShowMaskLayerEvent(4, vis_anatomy_active[]))
    end

    is_syncing_selection = Ref(false)

    on(les_menu.selection) do sel
        is_syncing_selection[] && return
        sel === nothing && return
        s = string(sel)
        if s != active_lesion_id[]
            # Save current lesion's state BEFORE switching (captures in-progress textbox edits)
            try trigger_autosave() catch; end
            is_syncing_selection[] = true
            try
                active_lesion_id[] = s
            finally
                is_syncing_selection[] = false
            end
        end
    end
    on(active_lesion_id) do id
        active_lesion_display[] = id
        # Sync Menu widget — set both i_selected and selection for visual update
        opts = lesion_ids[]
        isempty(opts) && return
        idx = findfirst(==(id), opts)
        idx === nothing && return
        is_syncing_selection[] = true
        try
            les_menu.i_selected[] = idx
            les_menu.selection[] = id
        finally
            is_syncing_selection[] = false
        end
    end
    on(btn_prev.clicks) do _
        t = time_ns()
        opts = lesion_ids[]; isempty(opts) && return
        idx = findfirst(==(active_lesion_id[]), opts)
        new_idx = idx === nothing ? 1 : (idx == 1 ? length(opts) : idx - 1)
        active_lesion_id[] = opts[new_idx]
        @info "[BENCH] Next/Prev Lesion (UI Update): $(round((time_ns()-t)/1e6, digits=1))ms"
    end
    on(btn_next.clicks) do _
        t = time_ns()
        opts = lesion_ids[]; isempty(opts) && return
        idx = findfirst(==(active_lesion_id[]), opts)
        new_idx = idx === nothing ? 1 : (idx == length(opts) ? 1 : idx + 1)
        active_lesion_id[] = opts[new_idx]
        @info "[BENCH] Next/Prev Lesion (UI Update): $(round((time_ns()-t)/1e6, digits=1))ms"
    end

    end_section!(sec_nav)

    # ── Viewport Controls ────────────────────────────────────────────────────
    sec_view = begin_section!("Viewport & Windowing")
    
    vc0_r = nr!()
    btn_ax  = Button(g[vc0_r, 1], label = "Axial",    buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    btn_cor = Button(g[vc0_r, 2], label = "Coronal",  buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    btn_sag = Button(g[vc0_r, 3], label = "Sagittal", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    btn_cv  = Button(g[vc0_r, 4], label = "Compare",  buttoncolor = BLU_BTN, labelcolor = TXT, fontsize = 10)

    on(btn_ax.clicks) do _; put!(channel, ChangePlaneEvent(:Axial)) end
    on(btn_cor.clicks) do _; put!(channel, ChangePlaneEvent(:Coronal)) end
    on(btn_sag.clicks) do _; put!(channel, ChangePlaneEvent(:Sagittal)) end
    
    cv_active = Ref(false)
    on(btn_cv.clicks) do _
        cv_active[] = !cv_active[]
        btn_cv.buttoncolor[] = cv_active[] ? GRN : BLU_BTN
        put!(channel, CompareTimePointsEvent(cv_active[]))
    end

    vc_r = nr!()
    btn_ps = Button(g[vc_r, 1], label = "<< Slice",  buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    btn_ns = Button(g[vc_r, 2], label = "Slice >>",  buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    btn_pt = Button(g[vc_r, 3], label = "<< TP",     buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    btn_nt = Button(g[vc_r, 4], label = "TP >>",     buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    on(btn_ps.clicks) do _; put!(channel, Int64(-1)) end
    on(btn_ns.clicks) do _; put!(channel, Int64(1)) end
    on(btn_pt.clicks) do _; put!(channel, ChangeTimePointEvent(-1)) end
    on(btn_nt.clicks) do _; put!(channel, ChangeTimePointEvent(1)) end

    # TP label removed — modality/timepoint info is in the viewer title bar
    tp_status = Observable{String}("")  # kept for internal use
    function update_tp_label()
        idx = _MEH.current_tp_index[]
        label = get(_MEH.tp_labels, idx, "TP $idx")
        if cv_active[]
            right_idx = _MEH.compare_right_tp[]
            right_label = get(_MEH.tp_labels, right_idx, "TP $right_idx")
            put!(channel, SetWindowTitleEvent("MedEye3d - Compare: L [$label]  |  R [$right_label]"))
        else
            put!(channel, SetWindowTitleEvent("MedEye3d - Viewing: $label"))
        end
    end
    on(_MEH.tp_switched) do _
        update_tp_label()
        if cv_active[]
            _build_match_display!()
        end
    end
    update_tp_label()

    # Merged overlay/single/all/refresh into one compact row
    vc2_r = nr!()
    btn_tl = Button(g[vc2_r, 1], label = "Overlay", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    btn_single = Button(g[vc2_r, 2], label = "Single", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    btn_all = Button(g[vc2_r, 3], label = "All",    buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    btn_rf = Button(g[vc2_r, 4], label = "Refresh",  buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    on(btn_tl.clicks) do _; put!(channel, ToggleLesionEvent()) end
    on(btn_rf.clicks) do _; put!(channel, RefreshListEvent()) end
    on(btn_single.clicks) do _
        id_str = active_lesion_id[]
        parsed = parse_lesion_id(id_str)
        if parsed !== nothing
            put!(channel, ShowSingleLesionEvent(parsed))
        end
    end
    on(btn_all.clicks) do _
        put!(channel, ShowSingleLesionEvent(0))
    end
    
    end_section!(sec_view)

    # ── Dedicated Windowing & Image Offsets Subpanel ─────────────────────────
    sec_win = begin_section!("Windowing & Image Offsets"; default_open=false)

    display_cfg = load_display_config()
    init_blend = Float32(get(display_cfg, "pet_ct_blend", 0.5))
    init_label_opacity = Float32(get(display_cfg, "label_opacity", 0.5))

    # PET/CT Blend slider (0.0 = CT only, 1.0 = full PET overlay)
    blend_r = nr!()
    Label(g[blend_r, 1], "PET/CT:", fontsize = 10, color = LBL_FG, halign = :right)
    slider_blend = Slider(g[blend_r, 2:3], range = 0.0f0:0.01f0:1.0f0, startvalue = init_blend)
    lbl_blend_val = Label(g[blend_r, 4], @lift(string(round($(slider_blend.value), digits=2))),
        fontsize = 10, color = TXT)
    rowsize!(g, blend_r, Fixed(28)); register_fixed_row!(blend_r, 28)
    on(slider_blend.value) do val
        v = Float32(val)
        display_cfg["pet_ct_blend"] = v
        save_display_config(display_cfg)
        put!(channel, PetBlendEvent(v))
    end

    # Label / Mask Opacity slider (0.0 = transparent, 1.0 = opaque)
    opac_r = nr!()
    Label(g[opac_r, 1], "Label Opacity:", fontsize = 10, color = LBL_FG, halign = :right)
    slider_label_opacity = Slider(g[opac_r, 2:3], range = 0.0f0:0.01f0:1.0f0, startvalue = init_label_opacity)
    lbl_opac_val = Label(g[opac_r, 4], @lift(string(round($(slider_label_opacity.value), digits=2))),
        fontsize = 10, color = TXT)
    rowsize!(g, opac_r, Fixed(28)); register_fixed_row!(opac_r, 28)
    on(slider_label_opacity.value) do val
        v = Float32(val)
        display_cfg["label_opacity"] = v
        save_display_config(display_cfg)
        put!(channel, LabelOpacityEvent(v))
    end

        # CT Windowing
    ct_lbl_r = nr!()
    Label(g[ct_lbl_r, 1:4], "-- CT Window & Offsets (HU) --", fontsize = 10, color = ACCENT, halign = :center, tellwidth = false)
    
    ct_s_r = nr!()
    islider_ct = IntervalSlider(g[ct_s_r, 1:4], range = -1500.0:10.0:3000.0, startvalues = (-150.0, 250.0))
    rowsize!(g, ct_s_r, Fixed(28)); register_fixed_row!(ct_s_r, 28)
    
    ct_p_r = nr!()
    Label(g[ct_p_r, 1], "Presets:", fontsize = 10, color = SUBTXT, halign = :right)
    btn_soft = Button(g[ct_p_r, 2], label = "Soft Tissue", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    btn_bone = Button(g[ct_p_r, 3], label = "Bone",        buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    btn_lung = Button(g[ct_p_r, 4], label = "Lung",        buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)

    ct_c_r = nr!()
    btn_ct_minus = Button(g[ct_c_r, 1], label = "- 50", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    tb_ct_min = Textbox(g[ct_c_r, 2], placeholder = "Min (-150)", stored_string = "-150.0", fontsize = 10)
    tb_ct_max = Textbox(g[ct_c_r, 3], placeholder = "Max (250)",  stored_string = "250.0",  fontsize = 10)
    btn_ct_plus  = Button(g[ct_c_r, 4], label = "+ 50", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    rowsize!(g, ct_c_r, Fixed(28)); register_fixed_row!(ct_c_r, 28)
    
    # Apply CT removed — slider drag and Enter-in-textbox already apply

    is_syncing_ct = Ref(false)
    function apply_ct_win(min_v::Real, max_v::Real)
        is_syncing_ct[] && return
        is_syncing_selection[] && return
        is_syncing_ct[] = true
        try
            tb_ct_min.stored_string[] = string(round(min_v, digits=1))
            tb_ct_min.displayed_string[] = string(round(min_v, digits=1))
            tb_ct_max.stored_string[] = string(round(max_v, digits=1))
            tb_ct_max.displayed_string[] = string(round(max_v, digits=1))
            set_close_to!(islider_ct, Float32(min_v), Float32(max_v))
            put!(channel, WindowingEvent("CT", Float32(min_v), Float32(max_v)))
        finally
            is_syncing_ct[] = false
        end
    end

    on(islider_ct.interval) do (min_v, max_v); apply_ct_win(min_v, max_v); end
    on(btn_soft.clicks) do _; apply_ct_win(-160.0, 240.0) end
    on(btn_bone.clicks) do _; apply_ct_win(-450.0, 1050.0) end
    on(btn_lung.clicks) do _; apply_ct_win(-1350.0, 150.0) end
    on(btn_ct_minus.clicks) do _
        v_min = tryparse(Float32, _safe_strip(tb_ct_min.stored_string[])); v_min = v_min === nothing ? -150.0f0 : v_min - 50.0f0
        v_max = tryparse(Float32, _safe_strip(tb_ct_max.stored_string[])); v_max = v_max === nothing ? 250.0f0 : v_max - 50.0f0
        apply_ct_win(v_min, v_max)
    end
    on(btn_ct_plus.clicks) do _
        v_min = tryparse(Float32, _safe_strip(tb_ct_min.stored_string[])); v_min = v_min === nothing ? -150.0f0 : v_min + 50.0f0
        v_max = tryparse(Float32, _safe_strip(tb_ct_max.stored_string[])); v_max = v_max === nothing ? 250.0f0 : v_max + 50.0f0
        apply_ct_win(v_min, v_max)
    end
    
    function apply_ct_from_text()
        is_syncing_selection[] && return
        s_min = !isempty(tb_ct_min.displayed_string[]) ? tb_ct_min.displayed_string[] : tb_ct_min.stored_string[]
        s_max = !isempty(tb_ct_max.displayed_string[]) ? tb_ct_max.displayed_string[] : tb_ct_max.stored_string[]
        v_min = tryparse(Float32, _safe_strip(s_min))
        v_max = tryparse(Float32, _safe_strip(s_max))
        if v_min !== nothing && v_max !== nothing
            apply_ct_win(v_min, v_max)
        end
    end
    on(tb_ct_min.stored_string) do _; apply_ct_from_text(); end
    on(tb_ct_max.stored_string) do _; apply_ct_from_text(); end

    # PET Windowing
    pet_lbl_r = nr!()
    Label(g[pet_lbl_r, 1:4], "-- PET Window & Offsets (SUV) --", fontsize = 10, color = ACCENT, halign = :center, tellwidth = false)
    
    pet_s_r = nr!()
    islider_pet = IntervalSlider(g[pet_s_r, 1:4], range = 0.0:0.1:50.0, startvalues = (0.0, 10.0))
    rowsize!(g, pet_s_r, Fixed(28)); register_fixed_row!(pet_s_r, 28)
    
    pet_p_r = nr!()
    Label(g[pet_p_r, 1], "Presets:", fontsize = 10, color = SUBTXT, halign = :right)
    btn_pet_5  = Button(g[pet_p_r, 2], label = "SUV 0-5",  buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    btn_pet_10 = Button(g[pet_p_r, 3], label = "SUV 0-10", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    btn_pet_15 = Button(g[pet_p_r, 4], label = "SUV 0-15", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)

    pet_c_r = nr!()
    btn_pet_minus = Button(g[pet_c_r, 1], label = "- 0.5", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    tb_pet_min = Textbox(g[pet_c_r, 2], placeholder = "Min (0.0)", stored_string = "0.0", fontsize = 10)
    tb_pet_max = Textbox(g[pet_c_r, 3], placeholder = "Max (10.0)", stored_string = "10.0", fontsize = 10)
    btn_pet_plus  = Button(g[pet_c_r, 4], label = "+ 0.5", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    rowsize!(g, pet_c_r, Fixed(28)); register_fixed_row!(pet_c_r, 28)
    
    # Apply PET removed — slider drag and Enter-in-textbox already apply

    is_syncing_pet = Ref(false)
    function apply_pet_win(min_v::Real, max_v::Real)
        is_syncing_pet[] && return
        is_syncing_selection[] && return
        is_syncing_pet[] = true
        try
            tb_pet_min.stored_string[] = string(round(min_v, digits=1))
            tb_pet_min.displayed_string[] = string(round(min_v, digits=1))
            tb_pet_max.stored_string[] = string(round(max_v, digits=1))
            tb_pet_max.displayed_string[] = string(round(max_v, digits=1))
            set_close_to!(islider_pet, Float32(min_v), Float32(max_v))
            put!(channel, WindowingEvent("PET", Float32(min_v), Float32(max_v)))
        finally
            is_syncing_pet[] = false
        end
    end

    on(islider_pet.interval) do (min_v, max_v); apply_pet_win(min_v, max_v); end
    on(btn_pet_5.clicks)  do _; apply_pet_win(0.0, 5.0) end
    on(btn_pet_10.clicks) do _; apply_pet_win(0.0, 10.0) end
    on(btn_pet_15.clicks) do _; apply_pet_win(0.0, 15.0) end
    on(btn_pet_minus.clicks) do _
        v_min = tryparse(Float32, _safe_strip(tb_pet_min.stored_string[])); v_min = v_min === nothing ? 0.0f0 : v_min - 0.5f0
        v_max = tryparse(Float32, _safe_strip(tb_pet_max.stored_string[])); v_max = v_max === nothing ? 10.0f0 : v_max - 0.5f0
        apply_pet_win(v_min, v_max)
    end
    on(btn_pet_plus.clicks) do _
        v_min = tryparse(Float32, _safe_strip(tb_pet_min.stored_string[])); v_min = v_min === nothing ? 0.0f0 : v_min + 0.5f0
        v_max = tryparse(Float32, _safe_strip(tb_pet_max.stored_string[])); v_max = v_max === nothing ? 10.0f0 : v_max + 0.5f0
        apply_pet_win(v_min, v_max)
    end
    
    function apply_pet_from_text()
        is_syncing_selection[] && return
        s_min = !isempty(tb_pet_min.displayed_string[]) ? tb_pet_min.displayed_string[] : tb_pet_min.stored_string[]
        s_max = !isempty(tb_pet_max.displayed_string[]) ? tb_pet_max.displayed_string[] : tb_pet_max.stored_string[]
        v_min = tryparse(Float32, _safe_strip(s_min))
        v_max = tryparse(Float32, _safe_strip(s_max))
        if v_min !== nothing && v_max !== nothing
            apply_pet_win(v_min, v_max)
        end
    end
    on(tb_pet_min.stored_string) do _; apply_pet_from_text(); end
    on(tb_pet_max.stored_string) do _; apply_pet_from_text(); end

    # SPECT Windowing
    spect_lbl_r = nr!()
    Label(g[spect_lbl_r, 1:4], "-- SPECT Window & Offsets (Counts) --", fontsize = 10, color = ACCENT, halign = :center, tellwidth = false)
    
    spect_s_r = nr!()
    islider_spect = IntervalSlider(g[spect_s_r, 1:4], range = 0.0:0.1:100.0, startvalues = (0.0, 10.0))
    rowsize!(g, spect_s_r, Fixed(28)); register_fixed_row!(spect_s_r, 28)
    
    spect_p_r = nr!()
    Label(g[spect_p_r, 1], "Presets:", fontsize = 10, color = SUBTXT, halign = :right)
    btn_spect_5  = Button(g[spect_p_r, 2], label = "SPECT 0-5",  buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    btn_spect_10 = Button(g[spect_p_r, 3], label = "SPECT 0-10", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    btn_spect_20 = Button(g[spect_p_r, 4], label = "SPECT 0-20", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)

    spect_c_r = nr!()
    btn_spect_minus = Button(g[spect_c_r, 1], label = "- 0.5", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    tb_spect_min = Textbox(g[spect_c_r, 2], placeholder = "Min (0.0)", stored_string = "0.0", fontsize = 10)
    tb_spect_max = Textbox(g[spect_c_r, 3], placeholder = "Max (10.0)", stored_string = "10.0", fontsize = 10)
    btn_spect_plus  = Button(g[spect_c_r, 4], label = "+ 0.5", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    rowsize!(g, spect_c_r, Fixed(28)); register_fixed_row!(spect_c_r, 28)
    
    # Apply SPECT removed — slider drag and Enter-in-textbox already apply

    is_syncing_spect = Ref(false)
    function apply_spect_win(min_v::Real, max_v::Real)
        is_syncing_spect[] && return
        is_syncing_selection[] && return
        is_syncing_spect[] = true
        try
            tb_spect_min.stored_string[] = string(round(min_v, digits=1))
            tb_spect_min.displayed_string[] = string(round(min_v, digits=1))
            tb_spect_max.stored_string[] = string(round(max_v, digits=1))
            tb_spect_max.displayed_string[] = string(round(max_v, digits=1))
            set_close_to!(islider_spect, Float32(min_v), Float32(max_v))
            put!(channel, WindowingEvent("SPECT", Float32(min_v), Float32(max_v)))
        finally
            is_syncing_spect[] = false
        end
    end

    on(islider_spect.interval) do (min_v, max_v); apply_spect_win(min_v, max_v); end
    on(btn_spect_5.clicks)  do _; apply_spect_win(0.0, 5.0) end
    on(btn_spect_10.clicks) do _; apply_spect_win(0.0, 10.0) end
    on(btn_spect_20.clicks) do _; apply_spect_win(0.0, 20.0) end
    on(btn_spect_minus.clicks) do _
        v_min = tryparse(Float32, _safe_strip(tb_spect_min.stored_string[])); v_min = v_min === nothing ? 0.0f0 : v_min - 0.5f0
        v_max = tryparse(Float32, _safe_strip(tb_spect_max.stored_string[])); v_max = v_max === nothing ? 10.0f0 : v_max - 0.5f0
        apply_spect_win(v_min, v_max)
    end
    on(btn_spect_plus.clicks) do _
        v_min = tryparse(Float32, _safe_strip(tb_spect_min.stored_string[])); v_min = v_min === nothing ? 0.0f0 : v_min + 0.5f0
        v_max = tryparse(Float32, _safe_strip(tb_spect_max.stored_string[])); v_max = v_max === nothing ? 10.0f0 : v_max + 0.5f0
        apply_spect_win(v_min, v_max)
    end
    
    function apply_spect_from_text()
        s_min = !isempty(tb_spect_min.displayed_string[]) ? tb_spect_min.displayed_string[] : tb_spect_min.stored_string[]
        s_max = !isempty(tb_spect_max.displayed_string[]) ? tb_spect_max.displayed_string[] : tb_spect_max.stored_string[]
        v_min = tryparse(Float32, _safe_strip(s_min))
        v_max = tryparse(Float32, _safe_strip(s_max))
        if v_min !== nothing && v_max !== nothing
            apply_spect_win(v_min, v_max)
        end
    end
    on(tb_spect_min.stored_string) do _; apply_spect_from_text(); end
    on(tb_spect_max.stored_string) do _; apply_spect_from_text(); end
end_section!(sec_win)

    # ── All Annotation Fields (single section, like Slicer extension) ─────────
    field_widgets = Dict{String, Any}()
    q_row_indices = Dict{String, Vector{Int}}()
    all_metadata_rows = Int[]
    
    schema_dict = Dict(q.short => q for q in schema)
    
    # All fields in order, grouped by visual sub-headers
    metadata_groups = [
        "Identification" => [
            "Radioligand Type",
            "Lesion tracking name?",
            "Anatomic Location",
            "Anatomical Sublocation"
        ],
        "Morphology" => [
            "Inner Texture / Density / Attenuation",
            "Border and Margin",
            "Lesion Shape",
            "Lesion Orientation",
            "Macroscopic Pattern",
            "Relation to Bone Marrow (Surrounding Changes Part A)",
            "Periosteal Reaction (Surrounding Changes Part B)",
            "Other Structural & Soft Tissue Changes (Surrounding Changes Part C)"
        ],
        "Clinical Context" => [
            "SUV max",
            "SUV Quantitative Metrics & References",
            "Clinical Context & Staging Variables"
        ],
        "PRIMARY Score" => [
            "PRIMARY score pattern?"
        ],
        "Assessment" => [
            "PSMA-RADS 2.0",
            "Alternative Hypothesis (False Positive)",
            "Certainty"
        ],
        "Reporting" => [
            "Comment"
        ]
    ]

    
    function set_row_visible!(row_idx::Int, visible::Bool)
        if visible
            if haskey(_row_fixed_heights, row_idx)
                rowsize!(g, row_idx, Fixed(_row_fixed_heights[row_idx]))
            else
                rowsize!(g, row_idx, Auto())
            end
        else
            rowsize!(g, row_idx, Fixed(0))
        end
        if row_idx < r[1]
            rowgap!(g, row_idx, visible ? 2 : 0)
        end
        for c in g.content
            if c.span.rows.start <= row_idx && c.span.rows.stop >= row_idx
                if hasproperty(c.content, :blockscene)
                    c.content.blockscene.visible[] = visible
                elseif hasproperty(c.content, :visible)
                    c.content.visible[] = visible
                end
            end
        end
    end

    section_headers = Dict{String, Int}()
    
    # Load persistent custom options
    custom_opts_db = load_custom_options()
    
    # Single collapsible section for all metadata
    sec_meta = begin_section!("Lesion Metadata"; default_open=true)
    is_meta_open, meta_start_row, meta_header_r, meta_btn, _ = sec_meta
    push!(all_metadata_rows, meta_header_r)

    # Pre-declare variables that will be created inside the "Identification" group injection
    # (Julia if-blocks create local scope, so we need outer-scope declarations)
    local btn_type_prostate, btn_type_bone, btn_type_organ, btn_type_ln
    local active_lesion_type, menu_base_anat, ba_all_opts, menu_side
    local anat_detail_label_r, btn_add_anat_rel
    local ANAT_RELATIONS, MAX_ANAT_ROWS
    local anat_row_indices, anat_rel_menus, anat_struct_menus, anat_rm_btns, anat_active_count
    local update_type_buttons
    local no_ct_toggle, lbl_suv_comparison, lbl_match_analysis

    for (group_title, q_list) in metadata_groups
        # Sub-headers removed for compactness — field names are descriptive enough

        for sq in q_list
            q = get(schema_dict, sq, nothing)
            if q === nothing
                q = QuestionDef(sq, sq, String[], ["General"], "both", "")
            end

            q_r = nr!()
            push!(all_metadata_rows, q_r)
            q_rows = Int[q_r]

            Label(g[q_r, 1], q.short * ":",
                fontsize = 10, color = LBL_FG,
                halign = :right, tellwidth = false)

            if isempty(q.options)
                if q.short == "Comment"
                    # Comment field: bigger, with visible text on dark-themed background
                    tb = Textbox(g[q_r, 2:4],
                        placeholder = "Free-text clinical notes...",
                        fontsize = 10,
                        textcolor = RGBf(0.95, 0.95, 0.95),
                        textcolor_placeholder = RGBf(0.5, 0.5, 0.5),
                        boxcolor = RGBf(0.15, 0.15, 0.18),
                        boxcolor_focused = RGBf(0.2, 0.2, 0.25),
                        boxcolor_hover = RGBf(0.18, 0.18, 0.22),
                        bordercolor = RGBf(0.3, 0.3, 0.35),
                        bordercolor_focused = RGBf(0.4, 0.5, 0.9),
                        cursorcolor = RGBf(0.9, 0.9, 0.9))
                    rowsize!(g, q_r, Fixed(80)); register_fixed_row!(q_r, 80)
                else
                    tb = Textbox(g[q_r, 2:4],
                        placeholder = isempty(q.default_answer) ? "..." : q.default_answer,
                        fontsize = 10,
                        textcolor = RGBf(0.9, 0.9, 0.9),
                        boxcolor = RGBf(0.15, 0.15, 0.18),
                        boxcolor_focused = RGBf(0.2, 0.2, 0.25),
                        bordercolor = RGBf(0.3, 0.3, 0.35),
                        bordercolor_focused = RGBf(0.4, 0.5, 0.9))
                    rowsize!(g, q_r, Fixed(28)); register_fixed_row!(q_r, 28)
                end
                field_widgets[q.short] = tb
            else
                # Inject saved custom options into dropdown
                saved_opts = String[string(s) for s in get(custom_opts_db, q.short, Any[])]
                all_opts = String["- select -"; q.options; saved_opts]
                opts_obs = Observable(all_opts)
                def_idx = 1
                if q.short == "Radioligand Type"
                    ga_idx = findfirst(==("68Ga-PSMA-11"), all_opts)
                    def_idx = ga_idx !== nothing ? ga_idx : 2
                end
                m = searchable_menu(g, q_r, 2:3, options = opts_obs, default = def_idx, fontsize = 10)
                btn_add_opt = Button(g[q_r, 4], label = "+", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
                rowsize!(g, q_r, Fixed(28)); register_fixed_row!(q_r, 28)
                field_widgets[q.short] = m
                
                let q_name = q.short, menu_w = m, obs = opts_obs
                    on(btn_add_opt.clicks) do _
                        typed_prompt = menu_w.prompt[]
                        val_to_add = startswith(typed_prompt, "Filter: ") ? strip(typed_prompt[9:end]) : ""
                        if isempty(val_to_add) || val_to_add == "Select..."
                            menu_w.is_open[] = true
                        else
                            val_str = String(val_to_add)
                            add_global_custom_option(q_name, val_str)
                            cur_opts = copy(obs[])
                            if !(val_str in cur_opts)
                                push!(cur_opts, val_str)
                                obs[] = cur_opts
                            end
                            idx = findfirst(==(val_str), menu_w.options[])
                            if idx !== nothing
                                menu_w.i_selected[] = idx
                            end
                            trigger_autosave()
                        end
                    end
                end
            end

            # Compact: skip tooltip row (tooltips are too verbose for compact layout)
            q_row_indices[q.short] = q_rows
        end
        
        # ── After "Identification" group: inject Type Buttons + Anatomy ──────
        if group_title == "Identification"
            # Type buttons row
            lt_r = nr!()
            push!(all_metadata_rows, lt_r)
            btn_type_prostate = Button(g[lt_r, 1], label = "Prostate",   buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
            btn_type_bone     = Button(g[lt_r, 2], label = "Bone Meta",  buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
            btn_type_organ    = Button(g[lt_r, 3], label = "Organ Meta", buttoncolor = ACCENT, labelcolor = TXT, fontsize = 10)
            btn_type_ln       = Button(g[lt_r, 4], label = "Lymph Node", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
            
            active_lesion_type = Observable("Organ Meta")
            
            function update_type_buttons(t)
                active_lesion_type[] = t
                btn_type_prostate.buttoncolor[] = (t == "Prostate") ? ACCENT : BG_PNL
                btn_type_bone.buttoncolor[]     = (t == "Bone Meta") ? ACCENT : BG_PNL
                btn_type_organ.buttoncolor[]    = (t == "Organ Meta") ? ACCENT : BG_PNL
                btn_type_ln.buttoncolor[]       = (t == "Lymph Node" || t == "Lymph Node Meta") ? ACCENT : BG_PNL
                
                is_bone = (t == "Bone Meta")
                @info "Lesion type set to '$t' -> ShowBoneMaskEvent($is_bone)"
                put!(channel, ShowBoneMaskEvent(is_bone))
            end
            
            on(btn_type_prostate.clicks) do _; update_type_buttons("Prostate") end
            on(btn_type_bone.clicks)     do _; update_type_buttons("Bone Meta") end
            on(btn_type_organ.clicks)    do _; update_type_buttons("Organ Meta") end
            on(btn_type_ln.clicks)       do _; update_type_buttons("Lymph Node Meta") end
            
            # Base Anatomy (filterable Menu) + Side
            ba_r = nr!()
            push!(all_metadata_rows, ba_r)
            Label(g[ba_r, 1], "Anatomy:", fontsize = 10, color = LBL_FG, halign = :right)
            ba_all_opts = Observable(String[""; anatomy_ontology])
            menu_base_anat = searchable_menu(g, ba_r, 2:3, options = ba_all_opts, fontsize = 10)
            menu_side = Menu(g[ba_r, 4], options = ["", "Right", "Left", "NA"], default = "", fontsize = 10)
            rowsize!(g, ba_r, Fixed(28)); register_fixed_row!(ba_r, 28)
            
            # Anatomical Details (OntologyBuilder-style rows)
            anat_detail_label_r = nr!()
            push!(all_metadata_rows, anat_detail_label_r)
            Label(g[anat_detail_label_r, 1:3], "Anatomical Details:", fontsize = 10, color = LBL_FG, halign = :left, tellwidth = false)
            btn_add_anat_rel = Button(g[anat_detail_label_r, 4], label = "+ Row",
                buttoncolor = GRN, labelcolor = TXT, fontsize = 10)
            
            ANAT_RELATIONS = ["", "Inside / Contained In", "Surrounded By", "Adjacent To",
                "Anterior To", "Posterior To", "Superior To", "Inferior To",
                "Deep To", "Superficial To", "Lateral To", "Medial To",
                "Proximal To", "Distal To", "Between"]
            MAX_ANAT_ROWS = 6
            
            # Pre-allocate rows (all hidden by default)
            anat_row_indices = Int[]
            anat_rel_menus = Menu[]
            anat_struct_menus = Menu[]
            anat_rm_btns = Button[]
            anat_active_count = Observable(0)
            
            for i in 1:MAX_ANAT_ROWS
                ar = nr!()
                push!(anat_row_indices, ar)
                push!(all_metadata_rows, ar)
                rel_m = searchable_menu(g, ar, 1, options = ANAT_RELATIONS, fontsize = 10)
                struct_opts_i = Observable(String[""; anatomy_ontology])
                struct_m = searchable_menu(g, ar, 2:3, options = struct_opts_i, fontsize = 10)
                rm_btn = Button(g[ar, 4], label = "-", buttoncolor = RED_BTN, labelcolor = TXT, fontsize = 10, width = 30)
                push!(anat_rel_menus, rel_m)
                push!(anat_struct_menus, struct_m)
                push!(anat_rm_btns, rm_btn)
                # Hide by default
                rowsize!(g, ar, Fixed(0))
                rel_m.blockscene.visible[] = false
                struct_m.blockscene.visible[] = false
                rm_btn.blockscene.visible[] = false
                
                # Remove button
                let idx = i
                    on(rm_btn.clicks) do _
                        n = anat_active_count[]
                        n <= 0 && return
                        for j in idx:(n-1)
                            anat_rel_menus[j].i_selected[] = anat_rel_menus[j+1].i_selected[]
                            anat_struct_menus[j].i_selected[] = anat_struct_menus[j+1].i_selected[]
                        end
                        anat_rel_menus[n].i_selected[] = 1
                        anat_struct_menus[n].i_selected[] = 1
                        anat_active_count[] = n - 1
                    end
                end
            end
            
            # Show/hide rows based on active count
            on(anat_active_count) do n
                for i in 1:MAX_ANAT_ROWS
                    visible = (i <= n) && !cv_active[]
                    ar = anat_row_indices[i]
                    rowsize!(g, ar, visible ? Auto() : Fixed(0))
                    anat_rel_menus[i].blockscene.visible[] = visible
                    anat_struct_menus[i].blockscene.visible[] = visible
                    anat_rm_btns[i].blockscene.visible[] = visible
                end
            end
            
            # Add row button
            on(btn_add_anat_rel.clicks) do _
                n = anat_active_count[]
                n < MAX_ANAT_ROWS && (anat_active_count[] = n + 1)
            end
        end  # end "Identification" injection

        # Inject "No CT Correlate" toggle + SUV comparison after Identification
        if group_title == "Identification"
            ncc_r = nr!()
            push!(all_metadata_rows, ncc_r)
            no_ct_toggle = Toggle(g[ncc_r, 1], active = false, buttoncolor = ACCENT)
            Label(g[ncc_r, 2], "No CT Correlate", fontsize = 10, color = LBL_FG, halign = :left)
            
            suv_comparison_r = nr!()
            push!(all_metadata_rows, suv_comparison_r)
            lbl_suv_comparison = Label(g[suv_comparison_r, 1:4], "", fontsize = 9, color = GRN, halign = :left, tellwidth = false)
            
            match_analysis_r = nr!()
            push!(all_metadata_rows, match_analysis_r)
            lbl_match_analysis = Label(g[match_analysis_r, 1:4], "", fontsize = 9, color = RGBf(0.6, 0.8, 1.0), halign = :left, tellwidth = false)
        end
    end
    
    # ── RadLex Multi-Value Panel ──────────────────────────────────────────────
    Label(g[nr!(), 1:4], "-- RadLex Ontology Properties --", fontsize = 11, color = ACCENT, halign = :center, tellwidth = false)
    radlex_selected = Observable(String[])

    rl_r = nr!()
    Label(g[rl_r, 1], "Search:", fontsize = 10, color = LBL_FG, halign = :right)
    rl_search = Textbox(g[rl_r, 2:3], placeholder = "type to filter RadLex terms...", fontsize = 10)
    btn_rl_add = Button(g[rl_r, 4], label = "+ Add",
        buttoncolor = GRN, labelcolor = TXT, fontsize = 10)

    radlex_opts = isempty(radlex) ? ["(none)"] : (length(radlex) > 200 ? radlex[1:200] : radlex)
    radlex_filtered = Observable(radlex_opts)
    rl_menu_r = r[1] + 1  # peek at next row number before nr!()
    rl_menu = Menu(g[nr!(), 1:4], options = radlex_filtered, fontsize = 10)
    rowsize!(g, rl_menu_r, Fixed(28)); register_fixed_row!(rl_menu_r, 28)

    on(rl_search.stored_string) do txt
        t = _safe_strip(txt)
        if isempty(t)
            radlex_filtered[] = isempty(radlex) ? ["(none)"] : (length(radlex) > 200 ? radlex[1:200] : radlex)
        else
            tl = lowercase(t)
            hits = filter(s -> occursin(tl, lowercase(s)), radlex)
            radlex_filtered[] = isempty(hits) ? ["(none)"] : (length(hits) > 200 ? hits[1:200] : hits)
        end
    end
    on(btn_rl_add.clicks) do _
        sel = rl_menu.selection[]
        sel === nothing && return
        s = string(sel)
        cur = copy(radlex_selected[])
        s in cur || push!(cur, s)
        radlex_selected[] = cur
    end

    Label(g[nr!(), 1:4], "Selected RadLex terms:",
        fontsize = 10, color = LBL_FG, halign = :left, tellwidth = false)

    RL_SLOTS = 8
    rl_slot_row_start = r[1] + 1
    rl_labels = [Label(g[rl_slot_row_start + i - 1, 1:3], "",
                    fontsize = 10, color = TXT, halign = :left, tellwidth = false)
                 for i in 1:RL_SLOTS]
    rl_rm_btns = [Button(g[rl_slot_row_start + i - 1, 4], label = "x",
                    buttoncolor = RED_BTN, labelcolor = TXT, fontsize = 10, width = 30)
                  for i in 1:RL_SLOTS]
    # Initialize all slots as hidden (zero height)
    for i in 1:RL_SLOTS
        rowsize!(g, rl_slot_row_start + i - 1, Fixed(0))
        rl_rm_btns[i].blockscene.visible[] = false
        rl_labels[i].blockscene.visible[] = false
        if (rl_slot_row_start + i - 1) < r[1]
            rowgap!(g, rl_slot_row_start + i - 1, 0)
        end
    end
    r[1] = rl_slot_row_start + RL_SLOTS - 1

    on(radlex_selected) do terms
        for i in 1:RL_SLOTS
            if i <= length(terms)
                rl_labels[i].text[] = terms[i]
                rowsize!(g, rl_slot_row_start + i - 1, Auto())
                rl_rm_btns[i].blockscene.visible[] = true
                rl_labels[i].blockscene.visible[] = true
                if (rl_slot_row_start + i - 1) < r[1]
                    rowgap!(g, rl_slot_row_start + i - 1, 1)
                end
            else
                rl_labels[i].text[] = ""
                rowsize!(g, rl_slot_row_start + i - 1, Fixed(0))
                rl_rm_btns[i].blockscene.visible[] = false
                rl_labels[i].blockscene.visible[] = false
                if (rl_slot_row_start + i - 1) < r[1]
                    rowgap!(g, rl_slot_row_start + i - 1, 0)
                end
            end
        end
    end
    for i in 1:RL_SLOTS
        on(rl_rm_btns[i].clicks) do _
            cur = copy(radlex_selected[])
            i <= length(cur) && deleteat!(cur, i)
            radlex_selected[] = cur
        end
    end

    

    # ── Custom Key-Value Fields ───────────────────────────────────────────────
    Label(g[nr!(), 1:4], "-- Custom Key-Value Fields --", fontsize = 11, color = ACCENT, halign = :center, tellwidth = false)
    custom_db = Observable(Dict{String,String}())

    ck_r = nr!()
    Label(g[ck_r, 1], "Key:", fontsize = 10, color = LBL_FG, halign = :right)
    ck_tb = Textbox(g[ck_r, 2], placeholder = "field name", fontsize = 10)
    Label(g[ck_r, 3], "Value:", fontsize = 10, color = LBL_FG, halign = :right)
    cv_tb = Textbox(g[ck_r, 4], placeholder = "value", fontsize = 10)

    btn_add_c = Button(g[nr!(), 4], label = "+ Add Custom Field",
        buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    on(btn_add_c.clicks) do _
        k = _safe_strip(ck_tb.stored_string[])
        v = _safe_strip(cv_tb.stored_string[])
        isempty(k) && return
        d = copy(custom_db[])
        d[k] = v
        custom_db[] = d
    end

    

    end_section!(sec_meta)

    # ── Radiological Report ──────────────────────────────────────────────────
    sec_report = begin_section!("Radiological Report"; default_open=true)

    dict_hdr_r = nr!()
    Label(g[dict_hdr_r, 1], "Dictation:", fontsize = 10, color = LBL_FG, halign = :right)
    lang_btn_de = Button(g[dict_hdr_r, 2], label="DE", buttoncolor=GRN, labelcolor=TXT, fontsize=9)
    lang_btn_en = Button(g[dict_hdr_r, 3], label="EN", buttoncolor=BG_PNL, labelcolor=TXT, fontsize=9)
    current_dict_lang = Observable("DE")

    dict_r = nr!()
    dict_text = Observable{String}("Radiological dictation...")
    dict_lbl = Label(g[dict_r, 1:4], dict_text, word_wrap = true, tellwidth = false, fontsize = 9,
        color = LBL_FG, halign = :left)

    on(lang_btn_de.clicks) do _
        current_dict_lang[] = "DE"
        lang_btn_de.buttoncolor[] = GRN
        lang_btn_en.buttoncolor[] = BG_PNL
        tp = _MEH.current_tp_index[]
        de_desc = get(_MEH.tp_descriptions, tp, "")
        dict_text[] = isempty(de_desc) ? "(No dictation available)" : de_desc
    end

    on(lang_btn_en.clicks) do _
        current_dict_lang[] = "EN"
        lang_btn_en.buttoncolor[] = GRN
        lang_btn_de.buttoncolor[] = BG_PNL
        tp = _MEH.current_tp_index[]
        en_desc = get(_MEH.tp_english_descriptions, tp, "")
        dict_text[] = isempty(en_desc) ? "(No English translation available)" : en_desc
    end

    rpt_r = nr!()
    Label(g[rpt_r, 1], "Report:", fontsize = 10, color = LBL_FG, halign = :right)
    rpt_tb = Textbox(g[rpt_r, 2:3], placeholder = "Generated summary...", fontsize = 10)
    btn_gen = Button(g[rpt_r, 4], label = "Gen",
        buttoncolor = BLU_BTN, labelcolor = TXT, fontsize = 10)

    end_section!(sec_report)

    # CT-specific fields hidden when "No CT Correlate" is checked
    CT_SPECIFIC_FIELDS = Set([
        "Inner Texture / Density / Attenuation",
        "Border and Margin",
        "Lesion Shape",
        "Lesion Orientation",
        "Relation to Bone Marrow (Surrounding Changes Part A)",
        "Periosteal Reaction (Surrounding Changes Part B)",
        "Other Structural & Soft Tissue Changes (Surrounding Changes Part C)"
    ])

    # ── Dynamic Visibility Engine ─────────────────────────────────────────────
    function update_dynamic_visibility!(active_type::String)
        if cv_active[]
            return # Let Compare Mode keep everything hidden
        end
        
        is_p = (active_type == "Prostate")
        is_bm = (active_type == "Bone Meta")
        no_ct = no_ct_toggle.active[]
        
        for (sq, rows) in q_row_indices
            visible = true
            
            # Hide CT-specific fields when No CT Correlate is checked
            if no_ct && sq in CT_SPECIFIC_FIELDS
                visible = false
            elseif sq == "PRIMARY score pattern?"
                visible = is_p
            elseif sq == "Relation to Bone Marrow (Surrounding Changes Part A)" || 
                   sq == "Periosteal Reaction (Surrounding Changes Part B)"
                visible = is_bm && !no_ct
            elseif sq == "PSMA-RADS 2.0"
                visible = !is_p
            elseif sq == "Alternative Hypothesis (False Positive)"
                visible = !is_p
            end

            for row_idx in rows
                set_row_visible!(row_idx, visible)
            end
        end
    end

    # Wire No CT Correlate toggle to refresh visibility
    on(no_ct_toggle.active) do _
        update_dynamic_visibility!(active_lesion_type[])
        trigger_autosave()
    end

    # Auto-defaults and presets on lesion type change
    on(btn_type_prostate.clicks) do _
        update_type_buttons("Prostate")
        if haskey(field_widgets, "Anatomic Location") && field_widgets["Anatomic Location"] isa Menu
            field_widgets["Anatomic Location"].selection[] = "Prostate Gland"
        end
        if haskey(field_widgets, "Anatomical Sublocation") && field_widgets["Anatomical Sublocation"] isa Menu
            field_widgets["Anatomical Sublocation"].selection[] = "Prostate Peripheral Zone (PZ)"
        end
        if haskey(field_widgets, "Macroscopic Pattern") && field_widgets["Macroscopic Pattern"] isa Menu
            field_widgets["Macroscopic Pattern"].selection[] = "Solitary / Isolated Focus"
        end
    end
    
    on(btn_type_bone.clicks) do _
        update_type_buttons("Bone Meta")
        if haskey(field_widgets, "Anatomic Location") && field_widgets["Anatomic Location"] isa Menu
            field_widgets["Anatomic Location"].selection[] = "Axial Skeleton (Spine, Pelvis, Ribs, Skull, Sternum, Clavicles)"
        end
        if haskey(field_widgets, "Inner Texture / Density / Attenuation") && field_widgets["Inner Texture / Density / Attenuation"] isa Menu
            field_widgets["Inner Texture / Density / Attenuation"].selection[] = "Sclerotic / Blastic / Ivory (>1000 HU)"
        end
        
        # Trigger GenManualEvent to ensure bone subsegmentation is calculated for manually painted lesions
        active_str = active_lesion_id[]
        if active_str != "" && active_str != "(none)"
            parts = split(active_str, " - ")
            lid = tryparse(Int, strip(parts[1]))
            if lid !== nothing
                put!(channel, GenManualEvent(lid))
            end
        end
    end
    
    on(btn_type_organ.clicks) do _
        update_type_buttons("Organ Meta")
        if haskey(field_widgets, "Anatomic Location") && field_widgets["Anatomic Location"] isa Menu
            field_widgets["Anatomic Location"].selection[] = "Solid Organ / Viscera"
        end
    end
    
    on(btn_type_ln.clicks) do _
        update_type_buttons("Lymph Node Meta")
        if haskey(field_widgets, "Anatomic Location") && field_widgets["Anatomic Location"] isa Menu
            field_widgets["Anatomic Location"].selection[] = "Pelvic Lymph Node"
        end
    end
    local sec_map_lesions
    # Auto-hide metadata, segmentation, radlex, custom sections in Compare mode
    on(btn_cv.clicks) do _
        t_start = time_ns()
        if cv_active[]
            # Hide entire sections
            for sec in (sec_meta, sec_seg, sec_report)
                hide_section!(sec)
            end
            # Force map section open and visible
            sec_map_lesions[1][] = true  # is_open = true
            show_section!(sec_map_lesions)
            notify(anat_active_count)
            try
                _build_match_display!()
            catch e
                @warn "Auto-building match display on compare mode toggle failed: $e"
            end
        else
            # Show sections
            for sec in (sec_meta, sec_seg, sec_report)
                show_section!(sec)
            end
            hide_section!(sec_map_lesions)
            # Explicitly clear dynamically-created elements in nested map_grid
            # (hide_section! can't reach nested GridLayout children — they persist visually)
            try for elem in contents(map_grid); delete!(elem); end catch; end
            update_dynamic_visibility!(active_lesion_type[])
            notify(anat_active_count)
        end
        @info "[BENCH] Compare Volumes Toggle UI: $(round((time_ns()-t_start)/1e6, digits=1))ms"
    end



    # ── Segmentation Mini Manager (compact) ────────────────────────────────
    sec_seg = begin_section!("Segmentation & AI")
    
    # Row 1: New Lesion + Paint/Erase/View
    seg_r1 = nr!()
    btn_new_lesion = Button(g[seg_r1, 1], label = "New", buttoncolor = BLU_BTN, labelcolor = TXT, fontsize = 10)
    btn_paint      = Button(g[seg_r1, 2], label = "Paint", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    btn_erase      = Button(g[seg_r1, 3], label = "Erase", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    btn_view_mode  = Button(g[seg_r1, 4], label = "View", buttoncolor = BLU_BTN, labelcolor = TXT, fontsize = 10)
    rowsize!(g, seg_r1, Fixed(28)); register_fixed_row!(seg_r1, 28)
    
    current_paint_mode = Observable(:view)
    
    on(btn_new_lesion.clicks) do _
        max_id = 0
        for opt in lesion_ids[]
            cp = findfirst(':', opt)
            dash = cp === nothing ? findfirst(" - ", opt) : nothing
            ns = if cp !== nothing
                strip(opt[1:cp-1])
            elseif dash !== nothing
                strip(opt[1:first(dash)-1])
            else
                opt
            end
            p = tryparse(Int, ns)
            p !== nothing && (max_id = max(max_id, p))
        end
        for k in keys(lesion_db[])
            p = parse_lesion_id(k)
            p !== nothing && (max_id = max(max_id, p))
        end
        new_id = max_id + 1
        
        # Auto-name from anatomy atlas at the CURRENT cursor position
        display_name = "New Lesion"
        try
            atlas = _MEH.global_ts_atlas[]
            ts_names = _MEH.global_ts_names[]
            if atlas !== nothing && ts_names !== nothing
                # Use the actual viewer cursor position (tracked in ReactOnMouseClickAndDrag)
                vpos = _MEH.current_viewer_position[]
                cx = clamp(vpos[1], 1, size(atlas, 1))
                cy = clamp(vpos[2], 1, size(atlas, 2))
                cz = clamp(vpos[3], 1, size(atlas, 3))
                
                # Direct lookup at cursor position
                anat_val = Int(atlas[cx, cy, cz])
                
                # If exact voxel is unlabeled, try expanding sphere search
                if anat_val <= 0
                    for radius in [1, 2, 4, 8, 16]
                        found = false
                        for dz in -radius:radius, dy in -radius:radius, dx in -radius:radius
                            dx*dx + dy*dy + dz*dz > radius*radius && continue
                            nx = clamp(cx + dx, 1, size(atlas, 1))
                            ny = clamp(cy + dy, 1, size(atlas, 2))
                            nz = clamp(cz + dz, 1, size(atlas, 3))
                            v = Int(atlas[nx, ny, nz])
                            if v > 0 && haskey(ts_names, v)
                                anat_val = v
                                found = true
                                break
                            end
                        end
                        found && break
                    end
                end
                
                if anat_val > 0
                    organ_name = get(ts_names, anat_val, "")
                    if !isempty(organ_name)
                        entry = lookup_anatomy(organ_name)
                        if entry !== nothing
                            detailed = get(entry, "detailed", "")
                            display_name = !isempty(detailed) ? detailed : organ_name
                        else
                            display_name = organ_name
                        end
                    end
                end
            end
        catch e
            @warn "Anatomy lookup for new lesion failed: $e"
        end
        
        new_name = "$(new_id): $(display_name)"
        db = copy(lesion_db[]); db[string(new_id)] = Dict{String, Any}("_display_name" => new_name); lesion_db[] = db
        opts = copy(lesion_ids[]); push!(opts, new_name)
        is_syncing_selection[] = true
        try
            lesion_ids[] = opts; les_menu.options[] = opts
            les_menu.selection[] = new_name; les_menu.i_selected[] = length(opts)
        finally; is_syncing_selection[] = false; end
        active_lesion_id[] = new_name
        current_paint_mode[] = :paint
        btn_paint.buttoncolor[] = GRN; btn_erase.buttoncolor[] = BG_PNL; btn_view_mode.buttoncolor[] = BG_PNL
        empty!(_MASK_IDS_CACHE)
        # For a NEW lesion, do NOT send SyncLesionEvent — it has no voxels yet,
        # so the centroid lookup defaults to the volume center (jumping to middle slice).
        # Instead: activate painting and show ALL lesion IDs so newly painted voxels are visible.
        put!(channel, PaintValEvent(new_id, true))
        put!(channel, ShowSingleLesionEvent(0))  # show all lesions (0 = show all)
    end
    on(btn_paint.clicks) do _
        current_paint_mode[] = :paint
        btn_paint.buttoncolor[] = GRN; btn_erase.buttoncolor[] = BG_PNL; btn_view_mode.buttoncolor[] = BG_PNL
        empty!(_MASK_IDS_CACHE)
        val = (p = parse_lesion_id(active_lesion_id[])) !== nothing ? p : 1
        put!(channel, PaintValEvent(val, true))
    end
    on(btn_erase.clicks) do _
        current_paint_mode[] = :erase
        btn_paint.buttoncolor[] = BG_PNL; btn_erase.buttoncolor[] = RED_BTN; btn_view_mode.buttoncolor[] = BG_PNL
        empty!(_MASK_IDS_CACHE)
        put!(channel, PaintValEvent(0, true))
    end
    on(btn_view_mode.clicks) do _
        current_paint_mode[] = :view
        btn_paint.buttoncolor[] = BG_PNL; btn_erase.buttoncolor[] = BG_PNL; btn_view_mode.buttoncolor[] = BLU_BTN
        put!(channel, PaintValEvent(-1, false))
    end
    
    # Row 2: Brush slider + Move button (fixed height to prevent slider overlap)
    seg_r2 = nr!()
    Label(g[seg_r2, 1], "Brush:", halign=:right, fontsize=10, color=LBL_FG)
    slider_brush = Slider(g[seg_r2, 2:3], range = 1:20, startvalue = 1)
    on(slider_brush.value) do val; put!(channel, ChangeBrushSizeEvent(val)) end
    btn_move_lesion = Button(g[seg_r2, 4], label = "Move", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    rowsize!(g, seg_r2, Fixed(30)); register_fixed_row!(seg_r2, 30)
    move_lesion_active = Ref(false)
    on(btn_move_lesion.clicks) do _
        move_lesion_active[] = !move_lesion_active[]
        btn_move_lesion.buttoncolor[] = move_lesion_active[] ? GRN : BG_PNL
        put!(channel, ToggleMoveLesionModeEvent(move_lesion_active[]))
    end
    
    # Row 3: Algorithm dropdown (fixed height for Menu dropdown clearance)
    seg_r3 = nr!()
    Label(g[seg_r3, 1], "AI:", halign=:right, fontsize=10, color=LBL_FG)
    algo_combo = Menu(g[seg_r3, 2:3], options = ["HELPNet (AI)", "NNInteractive", "Traditional (PETTumor)"], default = "HELPNet (AI)", fontsize = 10)
    btn_add_ai = Button(g[seg_r3, 4], label = "Run AI", buttoncolor = GRN, labelcolor = TXT, fontsize = 10)
    rowsize!(g, seg_r3, Fixed(30)); register_fixed_row!(seg_r3, 30)
    on(btn_add_ai.clicks) do _
        @async try
            put!(channel, AddAutoPetEvent(algo_combo.selection[], channel))
        catch e
            @warn "Failed to dispatch AddAutoPetEvent: $e"
        end
    end

    # Row 4: AI status (fixed height)
    seg_r4 = nr!()
    Label(g[seg_r4, 1:4], @lift(string($(_MEH.ai_status_text))),
        fontsize=10, color=RGBAf(0.7, 0.9, 0.7, 1.0), halign=:center)
    rowsize!(g, seg_r4, Fixed(20)); register_fixed_row!(seg_r4, 20)

    end_section!(sec_seg)

    sec_map_lesions = begin_section!("Map Lesions (Compare Mode)"; default_open=false)
    
    map_info_r = nr!()
    lbl_map_left = Label(g[map_info_r, 1:2], "Current TP: ...", fontsize=10, font=:bold, color=LBL_FG, halign=:left)
    lbl_map_right = Label(g[map_info_r, 3:4], "Compare TP: ...", fontsize=10, font=:bold, color=LBL_FG, halign=:left)
    
    btn_refresh_map = Button(g[nr!(), 1:4], label="Refresh Associations", buttoncolor = BG_PNL, labelcolor = TXT, fontsize=10)
    
    map_container_r = nr!()
    map_grid = GridLayout(g[map_container_r, 1:4])
    rowsize!(g, map_container_r, Auto())
    
    map_selected_left = Observable{Vector{Int}}(Int[])
    map_selected_right = Observable{Vector{Int}}(Int[])

    _MASK_IDS_CACHE = Dict{Int, Vector{Int}}()
    function get_mask_ids(tp)
        if haskey(_MASK_IDS_CACHE, tp) return _MASK_IDS_CACHE[tp] end
        # Fast path: derive mask IDs from precomputed _volume_cache keys (O(1))
        cached_ids = Int[lid for (tp_idx, lid) in keys(_volume_cache) if tp_idx == tp && lid > 0]
        if !isempty(cached_ids)
            ids = sort!(unique!(cached_ids))
            _MASK_IDS_CACHE[tp] = ids
            return ids
        end
        # Fallback: scan mask volume (O(N) — only if volume cache is empty)
        if !haskey(_MEH.tp_data_cache, tp) return Int[] end
        entry = _MEH.tp_data_cache[tp]
        mask = entry.mask
        ids = Int.(filter(x -> x > 0, sort(unique(mask))))
        _MASK_IDS_CACHE[tp] = ids
        return ids
    end

    function _build_match_display!()
        # Skip building map display when not in compare mode (prevents layout overlap)
        if !cv_active[]
            for elem in contents(map_grid); delete!(elem); end
            return
        end
        for elem in contents(map_grid); delete!(elem); end
        
        tp_left = _MEH.current_tp_index[]
        tp_right = _MEH.compare_right_tp[]
        if tp_right < 0
            tp_indices = sort(collect(keys(_MEH.tp_labels)))
            if !isempty(tp_indices)
                cur_pos = findfirst(==(tp_left), tp_indices)
                cur_pos = cur_pos === nothing ? 1 : cur_pos
                next_pos = mod1(cur_pos + 1, length(tp_indices))
                tp_right = tp_indices[next_pos]
            else
                tp_right = (tp_left + 1)
            end
        end
        
        left_node = _MEH.get_node_name_for_tp(tp_left)
        right_node = _MEH.get_node_name_for_tp(tp_right)
        
        left_lbl = get(_MEH.tp_labels, tp_left, "TP $tp_left")
        right_lbl = get(_MEH.tp_labels, tp_right, "TP $tp_right")
        
        lbl_map_left.text[] = "Current TP: $left_lbl ($left_node)"
        lbl_map_right.text[] = "Compare TP: $right_lbl ($right_node)"
        
        l_ids = get_mask_ids(tp_left)
        r_ids = get_mask_ids(tp_right)
        
        cur_act = active_lesion_id[]
        active_lid = parse_lesion_id(cur_act)
        active_lid = active_lid !== nothing ? active_lid : (isempty(l_ids) ? 0 : l_ids[1])
        
        cur_left_sel = copy(map_selected_left[])
        if isempty(cur_left_sel) && active_lid > 0
            cur_left_sel = Int[active_lid]
            map_selected_left[] = cur_left_sel
        end
        
        cur_right_sel = copy(map_selected_right[])
        if isempty(cur_right_sel) && !isempty(cur_left_sel)
            matched_rights = Int[]
            for lid in cur_left_sel
                append!(matched_rights, LA.find_cross_tp_lesion(left_node, lid, right_node))
            end
            cur_right_sel = unique(matched_rights)
            map_selected_right[] = cur_right_sel
        end
        
        # Back-propagate to ensure all related left lesions are shown in the mapping
        if !isempty(cur_right_sel)
            matched_lefts = Int[]
            for rid in cur_right_sel
                append!(matched_lefts, LA.find_cross_tp_lesion(right_node, rid, left_node))
            end
            if !isempty(matched_lefts)
                new_lefts = unique(vcat(cur_left_sel, matched_lefts))
                if length(new_lefts) > length(cur_left_sel)
                    cur_left_sel = new_lefts
                    map_selected_left[] = cur_left_sel
                end
            end
        end
        
        Label(map_grid[1, 1:2], "Add Left Lesion:", fontsize=9, color=LBL_FG, halign=:left)
        Label(map_grid[1, 3:4], "Add Right Lesion:", fontsize=9, color=LBL_FG, halign=:left)
        
        l_opts = String["- select lesion -"]
        for lid in l_ids
            # Prefer full name from main dropdown, else anatomy-enriched name
            display = ""
            for opt in lesion_ids[]
                p = parse_lesion_id(opt)
                if p == lid; display = opt; break; end
            end
            if isempty(display)
                organ = get(_MEH.global_organ_mapping[], lid, "")
                if !isempty(organ) && organ != "Unknown"
                    entry = lookup_anatomy(organ)
                    detailed = entry !== nothing ? get(entry, "detailed", organ) : organ
                    display = "ID $lid: $detailed"
                else
                    display = "ID $lid"
                end
            end
            push!(l_opts, display)
        end
        r_opts = String["- select lesion -"]
        for rid in r_ids
            display = ""
            for opt in lesion_ids[]
                p = parse_lesion_id(opt)
                if p == rid; display = opt; break; end
            end
            if isempty(display)
                organ = get(_MEH.global_organ_mapping[], rid, "")
                if !isempty(organ) && organ != "Unknown"
                    entry = lookup_anatomy(organ)
                    detailed = entry !== nothing ? get(entry, "detailed", organ) : organ
                    display = "ID $rid: $detailed"
                else
                    display = "ID $rid"
                end
            end
            push!(r_opts, display)
        end
        
        menu_l = searchable_menu(map_grid, 2, 1:2, options = Observable(l_opts), fontsize = 9)
        menu_r = searchable_menu(map_grid, 2, 3:4, options = Observable(r_opts), fontsize = 9)
        
        function sync_mapping_and_display!()
            h5_path = _MEH.h5_path_ref[]
            if !isempty(h5_path) && !isempty(map_selected_left[]) && !isempty(map_selected_right[])
                for l_id in map_selected_left[]
                    for r_id in map_selected_right[]
                        LA.update_match_group!(left_node, l_id, right_node, r_id, h5_path)
                    end
                end
            end
            if !isempty(map_selected_left[])
                put!(channel, SyncLesionEvent(map_selected_left[][1]))
            end
            _build_match_display!()
        end
        
        on(menu_l.selection) do sel
            _is_applying_state[] && return
            sel_str = string(sel)
            (isempty(sel_str) || sel_str == "- select lesion -") && return
            lid = parse_lesion_id(sel_str)
            if lid !== nothing && !(lid in map_selected_left[])
                push!(map_selected_left[], lid)
                notify(map_selected_left)
                sync_mapping_and_display!()
            end
        end
        
        on(menu_r.selection) do sel
            _is_applying_state[] && return
            sel_str = string(sel)
            (isempty(sel_str) || sel_str == "- select lesion -") && return
            rid = parse_lesion_id(sel_str)
            if rid !== nothing && !(rid in map_selected_right[])
                push!(map_selected_right[], rid)
                notify(map_selected_right)
                sync_mapping_and_display!()
            end
        end
        
        row_offset = 3
        max_rows = max(length(map_selected_left[]), length(map_selected_right[]))
        if max_rows == 0
            Label(map_grid[row_offset, 1:4], "(No mapped lesions selected)", fontsize=9, color=SUBTXT, halign=:center)
        else
            for i in 1:length(map_selected_left[])
                lid = map_selected_left[][i]
                # Build a rich display label: prefer full name from dropdown list, else organ mapping + anatomy
                lbl_txt = ""
                for opt in lesion_ids[]
                    p = parse_lesion_id(opt)
                    if p == lid
                        lbl_txt = opt  # e.g. "3: Left Femur [Grp 1, 2 TPs]"
                        break
                    end
                end
                if isempty(lbl_txt)
                    organ = get(_MEH.global_organ_mapping[], lid, "")
                    if !isempty(organ) && organ != "Unknown"
                        entry = lookup_anatomy(organ)
                        detailed = entry !== nothing ? get(entry, "detailed", organ) : organ
                        lbl_txt = "$lid: $detailed"
                    else
                        lbl_txt = "ID $lid"
                    end
                end
                Label(map_grid[row_offset + i - 1, 1], lbl_txt, fontsize=9, color=TXT, halign=:left)
                btn_rm_l = Button(map_grid[row_offset + i - 1, 2], label="✕", buttoncolor=RGBf(0.8, 0.2, 0.2), labelcolor=TXT, fontsize=9, width=25)
                let rm_id = lid
                    on(btn_rm_l.clicks) do _
                        filter!(x -> x != rm_id, map_selected_left[])
                        h5_path = _MEH.h5_path_ref[]
                        !isempty(h5_path) && LA.remove_from_match_group!(left_node, rm_id, h5_path)
                        notify(map_selected_left)
                        _build_match_display!()
                    end
                end
            end
            
            for j in 1:length(map_selected_right[])
                rid = map_selected_right[][j]
                # Build a rich display label: prefer from dropdown list, else organ + anatomy
                lbl_txt = ""
                for opt in lesion_ids[]
                    p = parse_lesion_id(opt)
                    if p == rid
                        lbl_txt = opt
                        break
                    end
                end
                if isempty(lbl_txt)
                    organ = get(_MEH.global_organ_mapping[], rid, "")
                    if !isempty(organ) && organ != "Unknown"
                        entry = lookup_anatomy(organ)
                        detailed = entry !== nothing ? get(entry, "detailed", organ) : organ
                        lbl_txt = "$rid: $detailed"
                    else
                        lbl_txt = "ID $rid"
                    end
                end
                Label(map_grid[row_offset + j - 1, 3], lbl_txt, fontsize=9, color=TXT, halign=:left)
                btn_rm_r = Button(map_grid[row_offset + j - 1, 4], label="✕", buttoncolor=RGBf(0.8, 0.2, 0.2), labelcolor=TXT, fontsize=9, width=25)
                let rm_id = rid
                    on(btn_rm_r.clicks) do _
                        filter!(x -> x != rm_id, map_selected_right[])
                        h5_path = _MEH.h5_path_ref[]
                        if !isempty(h5_path)
                            for (gid, members) in LA.get_match_groups()
                                idx_r = findfirst(m -> m[1] == right_node && m[2] == rm_id, members)
                                if idx_r !== nothing
                                    deleteat!(members, idx_r)
                                    length(members) <= 1 && delete!(LA.get_match_groups(), gid)
                                    LA.save_matches_to_h5(h5_path)
                                    break
                                end
                            end
                        end
                        notify(map_selected_right)
                        _build_match_display!()
                    end
                end
            end
        end
    end
    
    on(btn_refresh_map.clicks) do _
        empty!(_MASK_IDS_CACHE)
        _build_match_display!()
    end
    
    end_section!(sec_map_lesions)
    hide_section!(sec_map_lesions)

    # ── Settings & Export (merged: Active Data + Preprocessing + Save + Report) ──
    sec_settings = begin_section!("Settings & Export"; default_open=false)
    
    # Save / Load row
    sv_r = nr!()
    btn_save = Button(g[sv_r, 1:2], label = "Save Annotations",
        buttoncolor = GRN, labelcolor = TXT, fontsize = 10)
    btn_load = Button(g[sv_r, 3:4], label = "Load Annotations",
        buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    status_lbl = Label(g[nr!(), 1:4], "",
        fontsize = 10, color = RGBf(0.4, 0.9, 0.4), halign = :left, tellwidth = false)
    
    # Preprocessing
    pre_r1 = nr!()
    tog_autorun = Toggle(g[pre_r1, 1], active=false)
    Label(g[pre_r1, 2:3], "Auto-preprocess on load", fontsize = 10, color = LBL_FG, halign = :left)
    btn_save_mrb = Button(g[pre_r1, 4], label = "Save MRB", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    on(tog_autorun.active) do val; put!(channel, AutoRunPreprocessEvent(val)) end
    on(btn_save_mrb.clicks) do _; put!(channel, SaveMRBEvent()) end
    
    # Active Data Settings
    ads_r1 = nr!()
    Label(g[ads_r1, 1], "PET:", fontsize = 10, color = LBL_FG, halign = :right)
    Menu(g[ads_r1, 2], options = ["Auto", "SUV_PET_Image_0", "SUV_PET_Image_1"], fontsize = 10)
    Label(g[ads_r1, 3], "CT:", fontsize = 10, color = LBL_FG, halign = :right)
    Menu(g[ads_r1, 4], options = ["Auto", "Fixed_CT_Volume_0", "Fixed_CT_Volume_1"], fontsize = 10)
    rowsize!(g, ads_r1, Fixed(28)); register_fixed_row!(ads_r1, 28)

    ads_r2 = nr!()
    Label(g[ads_r2, 1], "Mask:", fontsize = 10, color = LBL_FG, halign = :right)
    Menu(g[ads_r2, 2], options = ["Auto", "Segmentation_0", "Segmentation_1"], fontsize = 10)
    Label(g[ads_r2, 3], "Atlas:", fontsize = 10, color = LBL_FG, halign = :right)
    Menu(g[ads_r2, 4], options = ["None", "Bone_Mask", "Organ_Mask"], fontsize = 10)
    rowsize!(g, ads_r2, Fixed(28)); register_fixed_row!(ads_r2, 28)

    ads_r3 = nr!()
    Label(g[ads_r3, 1], "Xform Fwd:", fontsize = 10, color = LBL_FG, halign = :right)
    Menu(g[ads_r3, 2], options = ["None", "Elastic_Transform_0_to_1"], fontsize = 10)
    Label(g[ads_r3, 3], "Xform Bwd:", fontsize = 10, color = LBL_FG, halign = :right)
    Menu(g[ads_r3, 4], options = ["None", "Elastic_Transform_1_to_0"], fontsize = 10)
    rowsize!(g, ads_r3, Fixed(28)); register_fixed_row!(ads_r3, 28)

    end_section!(sec_settings)

    # ── Collect / apply UI state ──────────────────────────────────────────────
    function collect_state()::Dict{String,String}
        d = Dict{String,String}()
        d["LesionType"] = active_lesion_type[]
        ba_sel = menu_base_anat.selection[]
        v_base = ba_sel === nothing ? "" : _safe_strip(string(ba_sel))
        (isempty(v_base) || v_base == "") || (d["BaseAnatomy"] = v_base)
        side_sel = menu_side.selection[]
        if side_sel !== nothing && !isempty(string(side_sel))
            d["BaseAnatomySide"] = string(side_sel)
        end
        
        # Serialize OntologyBuilder anatomical detail rows
        n = anat_active_count[]
        if n > 0
            parts = String[]
            for i in 1:n
                rel_sel = anat_rel_menus[i].selection[]
                rel_str = rel_sel === nothing ? "" : string(rel_sel)
                struct_sel = anat_struct_menus[i].selection[]
                struct_str = struct_sel === nothing ? "" : _safe_strip(string(struct_sel))
                if !isempty(struct_str)
                    push!(parts, isempty(rel_str) ? struct_str : "$(rel_str):$(struct_str)")
                end
            end
            isempty(parts) || (d["Anatomical Details"] = join(parts, " | "))
        end
        
        # No CT Correlate toggle
        if no_ct_toggle.active[]
            d["NoCTCorrelate"] = "true"
        end
        
        # Windowing values
        d["_CT_Min"] = _safe_strip(tb_ct_min.stored_string[])
        d["_CT_Max"] = _safe_strip(tb_ct_max.stored_string[])
        d["_PET_Min"] = _safe_strip(tb_pet_min.stored_string[])
        d["_PET_Max"] = _safe_strip(tb_pet_max.stored_string[])
        d["_SPECT_Min"] = _safe_strip(tb_spect_min.stored_string[])
        d["_SPECT_Max"] = _safe_strip(tb_spect_max.stored_string[])

        for q in schema
            w = get(field_widgets, q.short, nothing)
            w === nothing && continue
            if w isa Textbox
                # Prefer displayed_string (live typing) over stored_string (Enter-confirmed)
                # so in-progress edits are captured on autosave
                v = _safe_strip(w.displayed_string[])
                if isempty(v)
                    v = _safe_strip(w.stored_string[])
                end
                isempty(v) || (d[q.short] = v)
            elseif w isa Menu
                sel = w.selection[]
                sel === nothing && continue
                s = string(sel)
                (isempty(s) || s == "- select -") && continue
                d[q.short] = s
            end
        end
        rl = radlex_selected[]
        isempty(rl) || (d["RadLex"] = join(rl, " | "))
        for (k, v) in custom_db[]
            d["Custom:$(k)"] = v
        end
        if current_dict_lang[] == "EN"
            v_dict = _safe_strip(dict_text[])
            isempty(v_dict) || (d["RadiologicalDictationEN"] = v_dict)
        else
            v_dict = _safe_strip(dict_text[])
            isempty(v_dict) || (d["RadiologicalDictation"] = v_dict)
        end
        v_rpt = _safe_strip(rpt_tb.stored_string[])
        isempty(v_rpt) || (d["RadiologicalReportOutput"] = v_rpt)
        return d
    end

    _is_applying_state = Ref(false)
    function trigger_autosave()
        _is_applying_state[] && return
        display_id = active_lesion_id[]
        lid = parse_lesion_id(display_id)
        # Use canonical key: numeric ID string, or display name as fallback
        canonical_key = lid !== nothing ? string(lid) : display_id
        
        db = copy(lesion_db[])
        state = collect_state()
        # Store the current display name for reference/debugging
        state["_display_name"] = display_id
        db[canonical_key] = state
        
        db["_GLOBAL_APP_STATE"] = Dict{String,String}(
            "CT_Min" => _safe_strip(tb_ct_min.stored_string[]),
            "CT_Max" => _safe_strip(tb_ct_max.stored_string[]),
            "PET_Min" => _safe_strip(tb_pet_min.stored_string[]),
            "PET_Max" => _safe_strip(tb_pet_max.stored_string[]),
            "SPECT_Min" => _safe_strip(tb_spect_min.stored_string[]),
            "SPECT_Max" => _safe_strip(tb_spect_max.stored_string[]),
            "vis_lesion" => string(vis_lesion_active[]),
            "vis_surface" => string(vis_surface_active[]),
            "vis_marrow" => string(vis_marrow_active[]),
            "vis_anatomy" => string(vis_anatomy_active[])
        )
        
        lesion_db[] = db
        _db_dirty[] = true
        # Immediate save (db_channel consumer deduplicates rapid changes)
        try
            put!(db_channel, SaveDBMessage(db, global_app_state, save_path, DEFAULT_HDF5_PATH))
        catch; end
    end
    function apply_global_state(gst::AbstractDict)
        _is_applying_state[] = true
        is_syncing_selection[] = true
        try
            # Apply Windowing
            if haskey(gst, "CT_Min") && haskey(gst, "CT_Max")
                _set_tb_val!(tb_ct_min, gst["CT_Min"])
                _set_tb_val!(tb_ct_max, gst["CT_Max"])
                v_min = tryparse(Float32, gst["CT_Min"])
                v_max = tryparse(Float32, gst["CT_Max"])
                if v_min !== nothing && v_max !== nothing
                    put!(channel, WindowingEvent("CT", v_min, v_max))
                end
            end
            if haskey(gst, "PET_Min") && haskey(gst, "PET_Max")
                _set_tb_val!(tb_pet_min, gst["PET_Min"])
                _set_tb_val!(tb_pet_max, gst["PET_Max"])
                v_min = tryparse(Float32, gst["PET_Min"])
                v_max = tryparse(Float32, gst["PET_Max"])
                if v_min !== nothing && v_max !== nothing
                    put!(channel, WindowingEvent("PET", v_min, v_max))
                end
            end
            if haskey(gst, "SPECT_Min") && haskey(gst, "SPECT_Max")
                _set_tb_val!(tb_spect_min, gst["SPECT_Min"])
                _set_tb_val!(tb_spect_max, gst["SPECT_Max"])
                v_min = tryparse(Float32, gst["SPECT_Min"])
                v_max = tryparse(Float32, gst["SPECT_Max"])
                if v_min !== nothing && v_max !== nothing
                    put!(channel, WindowingEvent("SPECT", v_min, v_max))
                end
            end
            
            # Apply Toggles
            if haskey(gst, "vis_lesion")
                vis_lesion_active[] = (gst["vis_lesion"] == "true")
                btn_vis_lesion.label[] = vis_lesion_active[] ? "Lesion: ON" : "Lesion: OFF"
                btn_vis_lesion.buttoncolor[] = vis_lesion_active[] ? GRN : BG_PNL
                put!(channel, ShowMaskLayerEvent(1, vis_lesion_active[]))
            end
            if haskey(gst, "vis_surface")
                vis_surface_active[] = (gst["vis_surface"] == "true")
                btn_vis_surface.label[] = vis_surface_active[] ? "Surf: ON" : "Surf: OFF"
                btn_vis_surface.buttoncolor[] = vis_surface_active[] ? RGBf(0.0, 0.75, 0.75) : BG_PNL
                put!(channel, ShowMaskLayerEvent(2, vis_surface_active[]))
            end
            if haskey(gst, "vis_marrow")
                vis_marrow_active[] = (gst["vis_marrow"] == "true")
                btn_vis_marrow.label[] = vis_marrow_active[] ? "Marrow: ON" : "Marrow: OFF"
                btn_vis_marrow.buttoncolor[] = vis_marrow_active[] ? RGBf(0.75, 0.75, 0.1) : BG_PNL
                put!(channel, ShowMaskLayerEvent(3, vis_marrow_active[]))
            end
            if haskey(gst, "vis_anatomy")
                vis_anatomy_active[] = (gst["vis_anatomy"] == "true")
                btn_vis_anatomy.label[] = vis_anatomy_active[] ? "Anatomy: ON" : "Anatomy: OFF"
                btn_vis_anatomy.buttoncolor[] = vis_anatomy_active[] ? RGBf(0.5, 0.0, 0.8) : BG_PNL
                put!(channel, ShowMaskLayerEvent(4, vis_anatomy_active[]))
            end
        catch e
            @warn "Failed to apply global state: $e"
        finally
            _is_applying_state[] = false
            is_syncing_selection[] = false
        end
    end

    function apply_state(data::AbstractDict)
        _is_applying_state[] = true
        is_syncing_selection[] = true
        try
            cur_id_str = active_lesion_id[]
            db_updates = Dict{String, Any}()
            lid = (p = parse_lesion_id(cur_id_str)) !== nothing ? p : 1

        t_type = if haskey(data, "LesionType")
            data["LesionType"]
        else
            # Auto-detect lesion type: try JSON mapping first, then keyword fallback
            raw_organ_for_type = get(_MEH.global_organ_mapping[], lid, "")
            if isempty(raw_organ_for_type) || raw_organ_for_type == "Unknown"
                # Fast centroid-based atlas lookup (O(1) instead of O(N) findall scan)
                tp = _MEH.current_tp_index[]
                if _MEH.global_ts_atlas[] !== nothing
                    try
                        ts_atlas = _MEH.global_ts_atlas[]
                        ts_names = _MEH.global_ts_names[]
                        centroid_for_map = if haskey(_MEH.lesion_centroids_cache, (tp, lid))
                            _MEH.lesion_centroids_cache[(tp, lid)]
                        elseif haskey(_MEH.lesion_centroids_cache, lid)
                            _MEH.lesion_centroids_cache[lid]
                        else
                            nothing
                        end
                        if centroid_for_map !== nothing
                            # Scale centroid from mask space to atlas space
                            mask_sz = haskey(_MEH.tp_data_cache, tp) ? size(_MEH.tp_data_cache[tp].mask) : nothing
                            if mask_sz !== nothing
                                scale_x = size(ts_atlas, 1) / mask_sz[1]
                                scale_y = size(ts_atlas, 2) / mask_sz[2]
                                scale_z = size(ts_atlas, 3) / mask_sz[3]
                                sx = clamp(round(Int, centroid_for_map[1] * scale_x), 1, size(ts_atlas, 1))
                                sy = clamp(round(Int, centroid_for_map[2] * scale_y), 1, size(ts_atlas, 2))
                                sz = clamp(round(Int, centroid_for_map[3] * scale_z), 1, size(ts_atlas, 3))
                            else
                                sx = clamp(centroid_for_map[1], 1, size(ts_atlas, 1))
                                sy = clamp(centroid_for_map[2], 1, size(ts_atlas, 2))
                                sz = clamp(centroid_for_map[3], 1, size(ts_atlas, 3))
                            end
                            ts_val = Int(ts_atlas[sx, sy, sz])
                            if ts_val > 0 && haskey(ts_names, ts_val)
                                organ_name = ts_names[ts_val]
                                _MEH.global_organ_mapping[][lid] = organ_name
                                raw_organ_for_type = organ_name
                                @info "[DYNAMIC MAP] Fast centroid lookup: lesion $lid → '$organ_name' at [$sx,$sy,$sz]"
                            end
                        end
                    catch e
                        @warn "Dynamic organ mapping failed for lesion $lid: $e"
                    end
                end
            end
            json_entry = lookup_anatomy(raw_organ_for_type)
            
            if json_entry !== nothing
                get(json_entry, "lesion_type", "Organ Meta")
            else
                # Keyword fallback for organs not in JSON mapping
                base_anat = lowercase(get(data, "BaseAnatomy", ""))
                id_low = lowercase(cur_id_str)
                raw_organ = lowercase(raw_organ_for_type)
                combined = base_anat * " " * id_low * " " * raw_organ
                
                bone_kws = ["femur", "hip", "vertebra", "rib", "sacrum", "clavicula",
                            "clavicle", "humerus", "scapula", "sternum", "skull",
                            "palate", "bone", "spine", "ilium", "ischium", "pubis",
                            "tibia", "radius", "carpal", "tarsal", "costal_cartilage"]
                vascular_exclusions = ["vena", "artery", "vein", "vessel", "trunk"]
                
                is_bone_kw = any(kw -> occursin(kw, combined), bone_kws) &&
                             !any(v -> occursin(v, combined), vascular_exclusions)
                
                # Only consider bone_subsegments_cache if it actually has non-empty data
                has_real_bone_subseg = if haskey(_MEH.bone_subsegments_cache, (_MEH.current_tp_index[], lid))
                    cached_data = _MEH.bone_subsegments_cache[(_MEH.current_tp_index[], lid)]
                    cached_data isa Tuple && length(cached_data) >= 2 &&
                        (!isempty(cached_data[1]) || !isempty(cached_data[2]))
                else
                    false
                end
                
                if occursin("prostate", combined)
                    "Prostate"
                elseif is_bone_kw || has_real_bone_subseg
                    "Bone Meta"
                elseif occursin("lymph", combined) || occursin("node", combined)
                    "Lymph Node Meta"
                else
                    "Organ Meta"
                end
            end
        end
        update_type_buttons(t_type)

        # ── Edge-slice artefact detection ─────────────────────────────────
        # Lesions on the first 2 or last 2 axial slices are classified as
        # technical artifacts (partial volume / reconstruction edge effects)
        cur_tp = _MEH.current_tp_index[]
        centroid = if haskey(_MEH.lesion_centroids_cache, (cur_tp, lid))
            _MEH.lesion_centroids_cache[(cur_tp, lid)]
        elseif haskey(_MEH.lesion_centroids_cache, lid)
            _MEH.lesion_centroids_cache[lid]
        else
            nothing
        end
        if lid > 0 && centroid !== nothing && _MEH.volume_z_size[] > 0
            z_slice = centroid[3]
            total_z = _MEH.volume_z_size[]
            
            if z_slice <= 2 || z_slice >= total_z - 1
                # Set Alternative Hypothesis to "Technical Artifact"
                if haskey(field_widgets, "Alternative Hypothesis (False Positive)") && field_widgets["Alternative Hypothesis (False Positive)"] isa Menu
                    opts = field_widgets["Alternative Hypothesis (False Positive)"].options[]
                    idx = findfirst(==("Technical Artifact"), opts)
                    if idx !== nothing
                        field_widgets["Alternative Hypothesis (False Positive)"].i_selected[] = idx
                    end
                end
                # Set Certainty to 0
                if haskey(field_widgets, "Certainty") && field_widgets["Certainty"] isa Menu
                    opts = field_widgets["Certainty"].options[]
                    idx = findfirst(==("0"), opts)
                    if idx !== nothing
                        field_widgets["Certainty"].i_selected[] = idx
                    end
                end
                # Persist in db_updates
                db_updates["Alternative Hypothesis (False Positive)"] = "Technical Artifact"
                db_updates["Certainty"] = "0"
                @info "Edge-slice artefact: lesion $lid z=$z_slice/$total_z → Technical Artifact, Certainty=0"
            end
        end
        
        t_base = get(data, "BaseAnatomy", "")
        t_side = get(data, "BaseAnatomySide", "")
        
        # Resolve the raw organ name for this lesion (used for BaseAnatomy + Location auto-fill)
        raw_organ = ""
        if lid > 0
            organ_map = _MEH.global_organ_mapping[]
            raw_organ = get(organ_map, lid, "")
            # Fallback: if organ mapping is empty or "Unknown", try volume-based scan
            if (isempty(raw_organ) || raw_organ == "Unknown") && _MEH.global_ts_atlas[] !== nothing
                try
                    atlas = _MEH.global_ts_atlas[]
                    ts_nm = _MEH.global_ts_names[]
                    tp_idx = _MEH.current_tp_index[]
                    
                    # Try volume-based scan from tp_data_cache mask
                    if haskey(_MEH.tp_data_cache, tp_idx)
                        mask_vol = _MEH.tp_data_cache[tp_idx].mask
                        best = LA.classify_and_pick_best_organ(mask_vol, atlas, ts_nm, lid)
                        if !isempty(best)
                            raw_organ = best
                            organ_map[lid] = raw_organ
                            _MEH.global_organ_mapping[] = organ_map
                            @info "Auto-named lesion $lid via volume scan (bone priority): '$raw_organ'"
                        end
                    end
                    
                    # If volume scan didn't find anything, try centroid fallback
                    if isempty(raw_organ) || raw_organ == "Unknown"
                        centroid = get(_MEH.lesion_centroids_cache, (tp_idx, lid), nothing)
                        if centroid === nothing
                            centroid = get(_MEH.lesion_centroids_cache, lid, nothing)
                        end
                        if centroid !== nothing
                            cx, cy, cz = clamp(centroid[1], 1, size(atlas,1)), clamp(centroid[2], 1, size(atlas,2)), clamp(centroid[3], 1, size(atlas,3))
                            anat_val = Int(atlas[cx, cy, cz])
                            if anat_val > 0
                                raw_organ = get(ts_nm, anat_val, "")
                                if !isempty(raw_organ)
                                    organ_map[lid] = raw_organ
                                    _MEH.global_organ_mapping[] = organ_map
                                    @info "Auto-named lesion $lid from centroid at [$cx,$cy,$cz]: '$raw_organ'"
                                end
                            end
                        end
                    end
                catch e
                    @warn "Anatomy atlas lookup for lesion $lid failed: $e"
                end
            end
        end
        
        # Lookup the anatomy ontology entry (used for both BaseAnatomy and Location)
        anat_entry = (!isempty(raw_organ) && raw_organ != "Unknown") ? lookup_anatomy(raw_organ) : nothing
        println("[PREFILL] lid=$lid, raw_organ='$raw_organ', anat_entry=$(anat_entry !== nothing ? "found" : "null"), t_base='$t_base'"); flush(stdout)
        
        # Auto-detect BaseAnatomy from max_anatomy JSON mapping if not saved
        if isempty(t_base) && anat_entry !== nothing
            t_base = get(anat_entry, "detailed", "")
            auto_side = get(anat_entry, "side", "")
            if isempty(t_side) && !isempty(auto_side)
                t_side = auto_side
            end
            @info "Auto-detected BaseAnatomy for lesion $lid: '$t_base' (side='$t_side') from organ '$raw_organ'"
        elseif isempty(t_base) && !isempty(raw_organ) && raw_organ != "Unknown"
            # Fallback to old map_ts_to_anatomy for unknown organs
            t_base, auto_side = map_ts_to_anatomy(raw_organ)
            if isempty(t_side) && !isempty(auto_side)
                t_side = auto_side
            end
            @info "Auto-detected BaseAnatomy for lesion $lid: '$t_base' (side='$t_side') via keyword fallback from '$raw_organ'"
        end
        
        # ── Auto-fill Anatomic Location & Sublocation (independent of BaseAnatomy) ──
        # These run whenever the fields are empty, even if BaseAnatomy is already saved
        existing_loc = get(data, "Anatomic Location", "")
        existing_subloc = get(data, "Anatomical Sublocation", "")
        
        if anat_entry !== nothing
            anat_loc = get(anat_entry, "anatomic_location", "")
            anat_subloc = get(anat_entry, "anatomical_sublocation", "")
            println("[PREFILL] anat_entry for '$raw_organ': loc='$anat_loc', subloc='$anat_subloc', existing_loc='$existing_loc', existing_subloc='$existing_subloc'"); flush(stdout)
            
            # Auto-fill Anatomic Location if empty
            if isempty(existing_loc) && !isempty(anat_loc)
                data["Anatomic Location"] = anat_loc
                db_updates["Anatomic Location"] = anat_loc
                println("[PREFILL] Auto-filled Anatomic Location='$anat_loc'"); flush(stdout)
            end
            
            # Auto-fill Anatomical Sublocation if empty
            if isempty(existing_subloc) && !isempty(anat_subloc)
                data["Anatomical Sublocation"] = anat_subloc
                db_updates["Anatomical Sublocation"] = anat_subloc
                println("[PREFILL] Auto-filled Anatomical Sublocation='$anat_subloc'"); flush(stdout)
            end
        end
        
        # ── Auto-rename "New Lesion" entries when anatomy is determined ──
        if lid > 0 && occursin("New Lesion", cur_id_str) && !isempty(t_base)
            new_display_name = "$lid: $t_base"
            if new_display_name != cur_id_str
                # Rename in dropdown list
                opts = copy(lesion_ids[])
                idx = findfirst(==(cur_id_str), opts)
                if idx !== nothing
                    opts[idx] = new_display_name
                    is_syncing_selection[] = true
                    try
                        lesion_ids[] = opts
                        les_menu.options[] = opts
                        les_menu.selection[] = new_display_name
                        les_menu.i_selected[] = idx
                    finally
                        is_syncing_selection[] = false
                    end
                end
                # Update in lesion_db: store under canonical key string(lid)
                db = copy(lesion_db[])
                can_k = string(lid)
                if haskey(db, can_k)
                    if db[can_k] isa AbstractDict
                        db[can_k]["_display_name"] = new_display_name
                    end
                elseif haskey(db, cur_id_str)
                    d_old = pop!(db, cur_id_str)
                    if d_old isa AbstractDict
                        d_old["_display_name"] = new_display_name
                    end
                    db[can_k] = d_old
                else
                    db[can_k] = Dict{String, Any}("_display_name" => new_display_name)
                end
                lesion_db[] = db
                # Update internal reference (but do NOT set active_lesion_id[] to avoid recursive callback)
                active_lesion_display[] = new_display_name
                cur_id_str = new_display_name
                @info "Auto-renamed lesion to '$new_display_name'"
            end
        end
        
        # Set Base Anatomy menu selection
        if !isempty(t_base)
            ba_opts = menu_base_anat.options[]
            ba_idx = findfirst(==(t_base), ba_opts)
            if ba_idx !== nothing
                menu_base_anat.i_selected[] = ba_idx
            else
                # Value not in options — add it dynamically
                new_ba_opts = copy(ba_opts)
                push!(new_ba_opts, t_base)
                ba_all_opts[] = new_ba_opts
                menu_base_anat.i_selected[] = length(new_ba_opts)
            end
        else
            menu_base_anat.i_selected[] = 1  # reset to ""
        end
        
        side_opts = menu_side.options[]
        s_idx = findfirst(==(t_side), side_opts)
        menu_side.i_selected[] = s_idx !== nothing ? s_idx : 1
        
        # ── Restore OntologyBuilder anatomical detail rows ─────────────────
        anat_details_raw = get(data, "Anatomical Details", "")
        if !isempty(anat_details_raw)
            parts = split(anat_details_raw, " | ")
            count = min(length(parts), MAX_ANAT_ROWS)
            for i in 1:count
                p = strip(parts[i])
                colon = findfirst(':', p)
                if colon !== nothing
                    rel_str = strip(p[1:colon-1])
                    struct_str = strip(p[colon+1:end])
                else
                    rel_str = ""
                    struct_str = p
                end
                # Set relation menu
                rel_opts = anat_rel_menus[i].options[]
                r_idx = findfirst(==(rel_str), rel_opts)
                anat_rel_menus[i].i_selected[] = r_idx !== nothing ? r_idx : 1
                # Set structure menu
                struct_opts = anat_struct_menus[i].options[]
                st_idx = findfirst(==(struct_str), struct_opts)
                if st_idx !== nothing
                    anat_struct_menus[i].i_selected[] = st_idx
                else
                    # Value not in options — add it
                    new_sopts = copy(struct_opts)
                    push!(new_sopts, struct_str)
                    anat_struct_menus[i].options[] = new_sopts
                    anat_struct_menus[i].i_selected[] = length(new_sopts)
                end
            end
            anat_active_count[] = count
        else
            # Clear all rows
            for i in 1:MAX_ANAT_ROWS
                anat_rel_menus[i].i_selected[] = 1
                anat_struct_menus[i].i_selected[] = 1
            end
            anat_active_count[] = 0
        end
        
        # ── Auto-fill Lesion tracking name ────────────────────────────────
        tp_idx = _MEH.current_tp_index[]
        modality = get(_MEH.tp_modalities, tp_idx, "PET")
        pat_id = _MEH.patient_id[]
        
        if !haskey(data, "Lesion tracking name?") || isempty(get(data, "Lesion tracking name?", ""))
            tracking_name = generate_tracking_name(lid, t_base, tp_idx, modality, pat_id)
            if haskey(field_widgets, "Lesion tracking name?") && field_widgets["Lesion tracking name?"] isa Textbox
                field_widgets["Lesion tracking name?"].stored_string[] = tracking_name
            end
            # Also store in db_updates so it persists
            db_updates["Lesion tracking name?"] = tracking_name
        end
        
        # ── Auto-fill SUV max (always recompute - cache handles freshness) ───
        try
            cache_key = (tp_idx, lid)
            suv_str = get(_lesion_suv_cache, cache_key, "")
            if isempty(suv_str)
                suv_str = compute_lesion_suv_string(lid, tp_idx)
                if !isempty(suv_str)
                    _lesion_suv_cache[cache_key] = suv_str
                end
            end
            if !isempty(suv_str)
                if haskey(field_widgets, "SUV max") && field_widgets["SUV max"] isa Textbox
                    field_widgets["SUV max"].stored_string[] = suv_str
                end
                # Persist
                db_updates["SUV max"] = suv_str
            end
        catch e
            @warn "Auto-SUV computation failed for lesion $lid: $e"
        end
        
        # ── Auto-compute PROMISE score and SUV comparison ────────────────
        try
            suv_str = get(data, "SUV max", "")
            if isempty(suv_str) && haskey(field_widgets, "SUV max") && field_widgets["SUV max"] isa Textbox
                suv_str = _safe_strip(field_widgets["SUV max"].stored_string[])
            end
            if !isempty(suv_str)
                fields = parse_suv_fields(suv_str)
                suv_max = get(fields, "max", 0.0f0)
                if suv_max > 0
                    bg = Dict{String,Float32}(
                        "liver" => get(fields, "liver", 0.0f0),
                        "parotid" => get(fields, "parotid", 0.0f0),
                        "blood" => get(fields, "blood", 0.0f0)
                    )
                    cmp_str = compute_suv_comparison_string(suv_max, bg)
                    lbl_suv_comparison.text[] = cmp_str
                end
            end
        catch e
            @warn "PROMISE auto-computation failed: $e"
        end
        
        # ── Auto-compute Volume and Match Analysis ───────────────────────
        try
            vol = compute_lesion_volume(lid, tp_idx)
            vol_cc = vol["volume_cc"]
            vol_mm3 = vol["volume_mm3"]
            diameter = vol["diameter_mm"]
            
            # Store volume in metadata
            if vol_cc > 0
                db_updates["_Volume_mm3"] = string(round(vol_mm3, digits=1))
                db_updates["_Volume_cc"] = string(round(vol_cc, digits=3))
                db_updates["_Diameter_mm"] = string(round(diameter, digits=1))
            end
            
            # Match analysis (cross-TP comparison) — only show in Compare Volumes mode
            if cv_active[]
                analysis = compute_match_analysis(lid, tp_idx)
                if analysis !== nothing
                    analysis_str = format_match_analysis(analysis)
                    lbl_match_analysis.text[] = analysis_str
                    
                    # Persist match analysis to metadata
                    db_updates["_MatchGroup"] = string(analysis.group_id)
                    db_updates["_RECIP"] = analysis.recip_category
                    if analysis.baseline_volume_cc > 0.001
                        db_updates["_VolDelta_pct"] = string(round(analysis.volume_delta_pct, digits=1))
                        db_updates["_VolDelta_cc"] = string(round(analysis.volume_delta_abs_cc, digits=3))
                    end
                    if analysis.baseline_suv_max > 0.1f0
                        db_updates["_SUVDelta"] = string(round(analysis.suv_delta_abs, digits=1))
                        db_updates["_SUVDelta_pct"] = string(round(analysis.suv_delta_pct, digits=1))
                    end
                else
                    # Compare mode but no match group — show volume
                    if vol_cc > 0
                        lbl_match_analysis.text[] = "Vol: $(round(vol_cc, digits=2))cc ($(round(diameter, digits=1))mm⌀)"
                    else
                        lbl_match_analysis.text[] = ""
                    end
                end
            else
                # Single-TP mode — show volume only, no cross-TP comparison
                if vol_cc > 0
                    lbl_match_analysis.text[] = "Vol: $(round(vol_cc, digits=2))cc ($(round(diameter, digits=1))mm⌀)"
                else
                    lbl_match_analysis.text[] = ""
                end
            end
        catch e
            @warn "Volume/Match analysis failed for lesion $lid: $e"
            lbl_match_analysis.text[] = ""
        end
        
        # ── Restore No CT Correlate toggle ───────────────────────────────
        no_ct_val = get(data, "NoCTCorrelate", "false") == "true"
        no_ct_toggle.active[] = no_ct_val
        
        # Restore windowing if present
        if haskey(data, "_CT_Min") && haskey(data, "_CT_Max")
            _set_tb_val!(tb_ct_min, data["_CT_Min"])
            _set_tb_val!(tb_ct_max, data["_CT_Max"])
            v_min = tryparse(Float32, data["_CT_Min"])
            v_max = tryparse(Float32, data["_CT_Max"])
            if v_min !== nothing && v_max !== nothing
                put!(channel, WindowingEvent("CT", v_min, v_max))
            end
        end
        if haskey(data, "_PET_Min") && haskey(data, "_PET_Max")
            _set_tb_val!(tb_pet_min, data["_PET_Min"])
            _set_tb_val!(tb_pet_max, data["_PET_Max"])
            v_min = tryparse(Float32, data["_PET_Min"])
            v_max = tryparse(Float32, data["_PET_Max"])
            if v_min !== nothing && v_max !== nothing
                put!(channel, WindowingEvent("PET", v_min, v_max))
            end
        end
        if haskey(data, "_SPECT_Min") && haskey(data, "_SPECT_Max")
            _set_tb_val!(tb_spect_min, data["_SPECT_Min"])
            _set_tb_val!(tb_spect_max, data["_SPECT_Max"])
            v_min = tryparse(Float32, data["_SPECT_Min"])
            v_max = tryparse(Float32, data["_SPECT_Max"])
            if v_min !== nothing && v_max !== nothing
                put!(channel, WindowingEvent("SPECT", v_min, v_max))
            end
        end

        for q in schema
            w = get(field_widgets, q.short, nothing)
            w === nothing && continue
            val = get(data, q.short, nothing)
            if w isa Textbox
                target_str = val === nothing ? "" : String(val)
                _set_tb_val!(w, target_str)
            elseif w isa Menu
                if val !== nothing && !isempty(String(val))
                    val_str = String(val)
                    opts = w.options[]
                    idx = findfirst(==(val_str), opts)
                    if idx === nothing
                        new_opts = copy(opts)
                        push!(new_opts, val_str)
                        w.options[] = new_opts
                        idx = length(new_opts)
                    end
                    if w.i_selected[] != idx
                        w.i_selected[] = idx
                    end
                else
                    if q.short == "Radioligand Type"
                        opts = w.options[]
                        ga_idx = findfirst(==("68Ga-PSMA-11"), opts)
                        w.i_selected[] = ga_idx !== nothing ? ga_idx : 1
                    elseif !isempty(q.default_answer)
                        opts = w.options[]
                        d_idx = findfirst(==(q.default_answer), opts)
                        w.i_selected[] = d_idx !== nothing ? d_idx : 1
                    else
                        if w.i_selected[] != 1
                            w.i_selected[] = 1   # reset to "- select -"
                        end
                    end
                end
            end
        end
        rl_raw = get(data, "RadLex", "")
        target_radlex = isempty(rl_raw) ? String[] : String.(split(rl_raw, " | "))
        if radlex_selected[] != target_radlex
            radlex_selected[] = target_radlex
        end
        cdb = Dict{String,String}()
        for (k, v) in data
            startswith(k, "Custom:") && (cdb[k[8:end]] = v)
        end
        custom_db[] = cdb
        
        target_dict = if current_dict_lang[] == "EN"
            en_val = get(data, "RadiologicalDictationEN", get(_MEH.tp_english_descriptions, _MEH.current_tp_index[], ""))
            isempty(en_val) ? get(data, "RadiologicalDictation", get(_MEH.tp_descriptions, _MEH.current_tp_index[], "")) : en_val
        else
            get(data, "RadiologicalDictation", get(_MEH.tp_descriptions, _MEH.current_tp_index[], ""))
        end
        dict_text[] = target_dict
        
        target_rpt = get(data, "RadiologicalReportOutput", "")
        _set_tb_val!(rpt_tb, target_rpt)
        
        # ── Single batch commit for all metadata updates ──────────────────
        if !isempty(db_updates)
            display_id = active_lesion_id[]
            lid = parse_lesion_id(display_id)
            canonical_key = lid !== nothing ? string(lid) : display_id
            db = copy(lesion_db[])
            cur_ld = copy(get(db, canonical_key, Dict{String,Any}()))
            for (k, v) in db_updates
                cur_ld[k] = v
            end
            db[canonical_key] = cur_ld
            lesion_db[] = db
        end
        finally
            _is_applying_state[] = false
            is_syncing_selection[] = false
        end
    end

    # ── Wire callbacks ────────────────────────────────────────────────────────
    on(active_lesion_id) do id
        t_cb = time_ns()
        @info "WIRE_CALLBACK: active_lesion_id changed to: $id"
        db = lesion_db[]
        try
            apply_state(get_lesion_state(db, id))
        catch e
            @warn "Failed to apply state for lesion $id: $e"
        end
        @info "[TIMING] apply_state: $(round((time_ns()-t_cb)/1e6, digits=1))ms for $id"
        
        # Refresh Map Lesions lists to track the newly selected lesion
        if cv_active[] && sec_map_lesions[1][]
            try
                lid = parse_lesion_id(id)
                if lid !== nothing
                    map_selected_left[] = Int[lid]
                    map_selected_right[] = Int[]  # will be auto-populated from cross-TP match
                    _build_match_display!()
                end
            catch e
                @warn "Failed to refresh Map Lesions: $e"
            end
        end
        
        # Synchronize lesion with viewer (filters mask and jumps to slice)
        # Non-blocking: use @async so we don't freeze the Makie GUI thread if the
        # consumer is busy processing a previous event
        @async try
            lid = parse_lesion_id(id)
            if lid !== nothing
                put!(channel, SyncLesionEvent(lid))
            end
        catch e
            @warn "Failed to send SyncLesionEvent: $e"
        end
    end
    if active_lesion_id[] != "" && active_lesion_id[] != "(none)"
        notify(active_lesion_id)
    end

    on(btn_save.clicks) do _
        display_id = active_lesion_id[]
        lid = parse_lesion_id(display_id)
        canonical_key = lid !== nothing ? string(lid) : display_id
        db = copy(lesion_db[])
        state = collect_state()
        state["_display_name"] = display_id
        db[canonical_key] = state
        
        # Persist global windowing
        db["_GLOBAL_APP_STATE"] = Dict{String,String}(
            "CT_Min" => _safe_strip(tb_ct_min.stored_string[]),
            "CT_Max" => _safe_strip(tb_ct_max.stored_string[]),
            "PET_Min" => _safe_strip(tb_pet_min.stored_string[]),
            "PET_Max" => _safe_strip(tb_pet_max.stored_string[]),
            "SPECT_Min" => _safe_strip(tb_spect_min.stored_string[]),
            "SPECT_Max" => _safe_strip(tb_spect_max.stored_string[]),
            "vis_lesion" => string(vis_lesion_active[]),
            "vis_surface" => string(vis_surface_active[]),
            "vis_marrow" => string(vis_marrow_active[]),
            "vis_anatomy" => string(vis_anatomy_active[])
        )
        global_st = Dict{String, Any}(
            "windowing" => Dict(
                "CT" => [_safe_strip(tb_ct_min.stored_string[]), _safe_strip(tb_ct_max.stored_string[])],
                "PET" => [_safe_strip(tb_pet_min.stored_string[]), _safe_strip(tb_pet_max.stored_string[])],
                "SPECT" => [_safe_strip(tb_spect_min.stored_string[]), _safe_strip(tb_spect_max.stored_string[])]
            )
        )
        lesion_db[] = db
        _db_dirty[] = false
        put!(db_channel, SaveDBMessage(db, global_st, save_path, DEFAULT_HDF5_PATH))
        try
            _MEH.flush_all_dirty_masks!()
        catch; end
        status_lbl.text[] = "Saved at $(Dates.format(Dates.now(), "HH:MM:SS"))"
    end

    on(btn_load.clicks) do _
        @async begin
            reply = Channel{Dict}(1)
            put!(db_channel, LoadDBMessage(save_path, reply, DEFAULT_HDF5_PATH))
            raw_db = take!(reply)
            db = _migrate_db(raw_db)
            lesion_db[] = db
            
            # Apply global state if found
            if haskey(db, "_GLOBAL_APP_STATE")
                gst = db["_GLOBAL_APP_STATE"]
                apply_global_state(gst)
            end
            
            apply_state(get_lesion_state(db, active_lesion_id[]))
            status_lbl.text[] = "Loaded $(length(db)) lesion(s)"
        end
    end

    on(btn_gen.clicks) do _
        id   = active_lesion_id[]
        data = collect_state()
        lines = ["=== Structured Radiological Report ===",
                 "Lesion ID: $(id)", ""]
        dictation = get(data, "RadiologicalDictation", "")
        if !isempty(dictation)
            push!(lines, "Dictation: $(dictation)", "")
        end
        for q in schema
            v = get(data, q.short, ""); isempty(v) && continue
            push!(lines, "* $(q.short): $(v)")
        end
        rl = get(data, "RadLex", "")
        isempty(rl) || push!(lines, "* RadLex Properties: $(rl)")
        for (k, v) in data
            startswith(k, "Custom:") && push!(lines, "* $(k[8:end]): $(v)")
        end
        push!(lines, "", "Generated: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
        
        # In a real environment we would send 'lines' to DIZ LLM API here
        # For now, we simulate the DIZ API generated text
        generated_text = join(lines, "\n") * "\n[DIZ LLM Translated Output]"
        rpt_tb.stored_string[] = generated_text
        
        # Trigger autosave update
        trigger_autosave()
    end

    # Auto-save logic
    global_app_state = Dict{String, Any}("windowing" => (0.0f0, 0.0f0))
    if haskey(ui_hooks, :windowing)
        on(ui_hooks[:windowing]) do val
            global_app_state["windowing"] = val
        end
    end
    
    # ── Wire autosave triggers for all metadata UI elements ──────────
    on(menu_base_anat.selection) do _; trigger_autosave(); end
    # OntologyBuilder anatomical detail rows
    for i in 1:MAX_ANAT_ROWS
        on(anat_struct_menus[i].selection) do _; trigger_autosave(); end
        on(anat_rel_menus[i].selection) do _; trigger_autosave(); end
    end
    on(anat_active_count) do _; trigger_autosave(); end
    on(menu_side.selection) do _; trigger_autosave(); end
    on(active_lesion_type) do t
        update_dynamic_visibility!(t)
        trigger_autosave()
    end
    on(dict_text) do _; trigger_autosave(); end
    on(rpt_tb.stored_string) do _; trigger_autosave(); end
    on(rpt_tb.focused) do is_focused
        if !is_focused
            disp = _safe_strip(rpt_tb.displayed_string[])
            if !isempty(disp) && disp != _safe_strip(rpt_tb.stored_string[])
                rpt_tb.stored_string[] = disp
            end
        end
    end
    on(radlex_selected) do _; trigger_autosave(); end
    
    for q in schema
        w = get(field_widgets, q.short, nothing)
        if w isa Textbox
            on(w.stored_string) do _; trigger_autosave(); end
            # When textbox loses focus, commit displayed_string → stored_string
            # so in-progress typing is saved without requiring Enter
            on(w.focused) do is_focused
                if !is_focused
                    disp = _safe_strip(w.displayed_string[])
                    if !isempty(disp) && disp != _safe_strip(w.stored_string[])
                        w.stored_string[] = disp
                    end
                end
            end
        elseif w isa Menu
            on(w.selection) do _; trigger_autosave(); end
        end
    end

    @async begin
        while true
            sleep(5.0)
            if _db_dirty[]
                _db_dirty[] = false
                try
                    db = lesion_db[]
                    put!(db_channel, SaveDBMessage(db, global_app_state, save_path, DEFAULT_HDF5_PATH))
                    println("  [AUTOSAVE] Saved (dirty flag was set)"); flush(stdout)
                catch e
                    @warn "Background autosave enqueue failed" e
                end
            end
        end
    end

    # Flush-on-close: save any pending dirty state on graceful shutdown
    atexit() do
        if _db_dirty[]
            _db_dirty[] = false
            try
                db = lesion_db[]
                save_annotations(db, save_path)
                save_annotations_hdf5(db, DEFAULT_HDF5_PATH)
                println("  [AUTOSAVE] Final save on exit"); flush(stdout)
            catch e
                @warn "atexit save failed" e
            end
        end
    end

    # Initial state application
    initial_id = active_lesion_id[]
    db = lesion_db[]
    
    # Apply global state (windowing, toggles) first
    if haskey(db, "_GLOBAL_APP_STATE")
        try
            apply_global_state(db["_GLOBAL_APP_STATE"])
        catch e
            @warn "Failed initial apply_global_state: $e"
        end
    end
    
    if initial_id != "" && initial_id != "(none)"
        try
            apply_state(get_lesion_state(db, initial_id))
        catch e
            @warn "Failed initial apply_state: $e"
        end
    end
    # Ensure dictation is populated if empty
    tp_cur = _MEH.current_tp_index[]
    de_desc = get(_MEH.tp_descriptions, tp_cur, "")
    if !isempty(de_desc) && isempty(_safe_strip(dict_text[]))
        dict_text[] = de_desc
    end

    return MetadataWindowResult(fig, channel_ref, lesion_db)
end

"""
Display the metadata Figure using the thread-safe synchronized GLMakie render loop.
"""
function display_metadata_window(fig::Figure)
    screen = lock(GLOBAL_OPENGL_LOCK) do
        GLMakie.Screen(fig.scene; renderloop=synchronized_makie_renderloop)
    end
    
    # Fix: Force hasfocus=true so GLMakie's MousePositionUpdater always tracks
    # the mouse position. Without this, the Makie window ignores mouse movement
    # when it doesn't have window focus (e.g., when the GLFW viewer window is
    # focused), which prevents button click detection entirely.
    #
    # Root cause: GLMakie/src/events.jl MousePositionUpdater has:
    #   !p.hasfocus[] && return
    # This skips mouse tracking when the window isn't focused, so
    # mouse_was_inside is never set to true, and button clicks are ignored.
    events(fig.scene).hasfocus[] = true
    on(events(fig.scene).hasfocus) do focused
        # Override any GLFW focus-lost events — always keep tracking
        if !focused
            @async (events(fig.scene).hasfocus[] = true)
        end
    end
    
    lock(GLOBAL_OPENGL_LOCK) do
        display(screen, fig)
    end
    return screen
end

end # module LesionMetadataWindow
