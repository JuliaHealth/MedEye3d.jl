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
end

export create_metadata_window, load_annotations, save_annotations, display_metadata_window

# ─── Paths ────────────────────────────────────────────────────────────────────
const _PKG_ROOT      = joinpath(@__DIR__, "..", "..", "extension", "data")
const _SLICER_DATA   = "/mnt/big/project_ssd/project_ssd/slicer_lesion_text_extension/data"
const DEF_JSON_PATH  = isfile(joinpath(_SLICER_DATA, "def.json")) ? joinpath(_SLICER_DATA, "def.json") : (isfile(joinpath(_PKG_ROOT, "def.json")) ? joinpath(_PKG_ROOT, "def.json") : joinpath(@__DIR__, "..", "..", "data", "def.json"))
const RADLEX_CSV_PATH= isfile(joinpath(_SLICER_DATA, "RadLex.csv")) ? joinpath(_SLICER_DATA, "RadLex.csv") : (isfile(joinpath(_PKG_ROOT, "RadLex.csv")) ? joinpath(_PKG_ROOT, "RadLex.csv") : joinpath(@__DIR__, "..", "..", "data", "RadLex.csv"))
const DEFAULT_SAVE_PATH = joinpath(homedir(), "medeye3d_lesion_annotations.json")
const DEFAULT_HDF5_PATH = joinpath(homedir(), "medeye3d_lesion_annotations.h5")

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

function load_schema()::Vector{QuestionDef}
    isempty(_schema_cache[]) || return _schema_cache[]
    if !isfile(DEF_JSON_PATH)
        error("Strict Configuration Enforcement: def.json schema file not found at $(DEF_JSON_PATH). Built-in fallback schema has been disabled.")
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
    _radlex_cache[] = sort(terms)
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
    _anatomy_cache[] = sort(terms)
    return _anatomy_cache[]
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
    m = match(r"^(.+?)_(\d+)$", name_low)
    if m !== nothing
        base = m.captures[1]
        num = m.captures[2]
        if haskey(TS_TO_ANATOMY, base)
            return ("$(TS_TO_ANATOMY[base]) $num", side)
        end
    end
    
    # 3. Fallback: title-case with underscores→spaces
    return (titlecase(replace(name_low, "_" => " ")), side)
end

# ─── Persistence ─────────────────────────────────────────────────────────────
function load_annotations(path::String = DEFAULT_SAVE_PATH)::Dict{String,Dict{String,Any}}
    isfile(path) || return Dict{String,Dict{String,Any}}()
    try
        raw = JSON.parse(read(path, String))
        out = Dict{String,Dict{String,Any}}()
        for (k, v) in raw
            if v isa AbstractDict
                inner = Dict{String,Any}()
                for (ik, iv) in v
                    inner[string(ik)] = iv
                end
                out[string(k)] = inner
            end
        end
        return out
    catch e
        @warn "Cannot load annotations from $(path): $(e)"
        return Dict{String,Dict{String,Any}}()
    end
end

function save_annotations_hdf5(db::Dict, path::String=DEFAULT_HDF5_PATH)
    try
        h5open(path, "w") do file
            for (id, lesion_data) in db
                g = create_group(file, string(id))
                for (k, v) in lesion_data
                    write(g, string(k), string(v))
                end
            end
        end
        @debug "Annotations saved to HDF5 → $path"
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

# ─── Main window ─────────────────────────────────────────────────────────────
function create_metadata_window(
        active_lesion_id::Observable{String},
        lesion_ids::Observable{Vector{String}},
        channel::Base.Channel;
        save_path::String = DEFAULT_SAVE_PATH,
        ui_hooks::Dict{Symbol, Observable} = Dict{Symbol, Observable}())

    schema   = load_schema()
    radlex   = load_radlex()
    anatomy_ontology = load_anatomy_ontology()  # FoundationalAnatomy labels for Base Anatomy autocomplete
    all_cats = all_categories(schema)

    # In-memory DB
    lesion_db = Observable{Dict}(Dict{String,Dict{String,Any}}())

    db_channel = Channel{Any}(32)
    @async begin
        for msg in db_channel
            if msg isa SaveDBMessage
                try
                    db_to_save = copy(msg.db)
                    db_to_save["_GlobalAppState"] = msg.global_app_state
                    save_annotations(db_to_save, msg.path_json)
                    save_annotations_hdf5(msg.db, msg.path_hdf5)
                catch e
                    @warn "Database save failed" e
                end
            elseif msg isa LoadDBMessage
                try
                    db = load_annotations(msg.path_json)
                    put!(msg.reply_channel, db)
                catch e
                    @warn "Database load failed" e
                    put!(msg.reply_channel, Dict{String,Dict{String,Any}}())
                end
            end
        end
    end
    
    # Load initial db asynchronously
    @async begin
        reply = Channel{Dict}(1)
        put!(db_channel, LoadDBMessage(save_path, reply))
        lesion_db[] = take!(reply)
    end

    # Helper for safely extracting and stripping text from Makie Textboxes
    _safe_strip(x) = x === nothing ? "" : String(strip(x))

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
    fig = Figure(size = (920, 900), backgroundcolor = BG)
    
    main_layout = GridLayout(fig[1,1])
    sl = Slider(main_layout[1, 2], range = 0:0.01:1, startvalue = 1, horizontal = false, tellheight = false)
    
    g = GridLayout(main_layout[1,1], tellheight = false, halign = :left, valign = sl.value)
    
    # Mouse scroll event to control slider
    on(fig.scene.events.scroll) do scroll
        sl.value[] = clamp(sl.value[] + scroll[2] * 0.05, 0.0, 1.0)
        return Consume(true)
    end
    
    rowgap!(g, 1)   # compact: was 3
    colgap!(g, 3)   # compact: was 4
    colsize!(g, 1, Auto())
    r = [0]  # row counter as array for mutation in closures
    nr!() = (r[1] += 1; r[1])

    # ── Header ──────────────────────────────────────────────────────────────
    Label(g[nr!(), 1:4], "Lesion Annotation Panel",
        fontsize = 22, font = :bold, color = ACCENT, halign = :center, tellwidth = false)
    Label(g[nr!(), 1:4],
        "Julia port of Slicer Lesion Text Extension  |  $(length(schema)) annotation fields from def.json",
        fontsize = 10, color = LBL_FG, halign = :center, tellwidth = false)

    # ── Section helper ──────────────────────────────────────────────────────
    function begin_section!(title; default_open=true)
        is_open = Observable(default_open)
        header_r = nr!()
        btn = Button(g[header_r, 1:4], label = @lift($is_open ? "▼  $(title)" : "▶  $(title)"),
            buttoncolor = SEC_HDR, labelcolor = ACCENT, fontsize = 11, halign = :left)
        
        start_row = r[1] + 1
        return (is_open, start_row, header_r, btn)
    end
    
    function end_section!(sec_data)
        is_open, start_row, header_r, btn = sec_data
        end_row = r[1]
        
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
                rowgap!(g, i, is_open[] ? 1 : 0)
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

    # Prominent active lesion status banner
    nav_info_r = nr!()
    active_lesion_display = Observable{String}("Active Lesion: (none)")
    Label(g[nav_info_r, 1:4], active_lesion_display, fontsize = 11, color = ACCENT, halign = :center, tellwidth = false)

    # Prominent Lesion & Bone Subsegments Layer Visibility Controls
    vis_row = nr!()
    vis_lesion_active = Ref(true)
    vis_surface_active = Ref(true)
    vis_marrow_active = Ref(true)
    
    btn_vis_lesion  = Button(g[vis_row, 1:2], label = "Lesion: ON", buttoncolor = GRN, labelcolor = TXT, fontsize = 10)
    btn_vis_surface = Button(g[vis_row, 3],   label = "Bone Surf: ON", buttoncolor = RGBf(0.0, 0.75, 0.75), labelcolor = TXT, fontsize = 10)
    btn_vis_marrow  = Button(g[vis_row, 4],   label = "Marrow: ON", buttoncolor = RGBf(0.75, 0.75, 0.1), labelcolor = TXT, fontsize = 10)

    on(btn_vis_lesion.clicks) do _
        vis_lesion_active[] = !vis_lesion_active[]
        btn_vis_lesion.label[] = vis_lesion_active[] ? "Lesion: ON" : "Lesion: OFF"
        btn_vis_lesion.buttoncolor[] = vis_lesion_active[] ? GRN : BG_PNL
        @info "BTN_VIS_LESION clicked: $(vis_lesion_active[])"
        put!(channel, ShowMaskLayerEvent(1, vis_lesion_active[]))
    end
    
    on(btn_vis_surface.clicks) do _
        vis_surface_active[] = !vis_surface_active[]
        btn_vis_surface.label[] = vis_surface_active[] ? "Bone Surf: ON" : "Bone Surf: OFF"
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

    is_syncing_selection = Ref(false)

    on(les_menu.selection) do sel
        is_syncing_selection[] && return
        sel === nothing && return
        s = string(sel)
        if s != active_lesion_id[]
            is_syncing_selection[] = true
            try
                active_lesion_id[] = s
            finally
                is_syncing_selection[] = false
            end
        end
    end
    on(active_lesion_id) do id
        # Update status banner
        active_lesion_display[] = "Active Lesion: [ $id ]"
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
        @info "BTN_PREV clicked"
        opts = lesion_ids[]; isempty(opts) && return
        idx = findfirst(==(active_lesion_id[]), opts)
        new_idx = idx === nothing ? 1 : (idx == 1 ? length(opts) : idx - 1)
        @info "BTN_PREV: setting active_lesion_id from $(active_lesion_id[]) to $(opts[new_idx])"
        active_lesion_id[] = opts[new_idx]
    end
    on(btn_next.clicks) do _
        @info "BTN_NEXT clicked"
        opts = lesion_ids[]; isempty(opts) && return
        idx = findfirst(==(active_lesion_id[]), opts)
        new_idx = idx === nothing ? 1 : (idx == length(opts) ? 1 : idx + 1)
        @info "BTN_NEXT: setting active_lesion_id from $(active_lesion_id[]) to $(opts[new_idx])"
        active_lesion_id[] = opts[new_idx]
    end

    end_section!(sec_nav)

    # ── Lesion Type & Anatomic Localization ───────────────────────────────────
    sec_type = begin_section!("Lesion Type & Anatomic Localization")
    
    lt_r = nr!()
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
    
    # Base Anatomy & Side (with ontology autocomplete)
    ba_r = nr!()
    Label(g[ba_r, 1], "Base Anatomy:", fontsize = 10, color = LBL_FG, halign = :right)
    tb_base_anat = Textbox(g[ba_r, 2], placeholder = "type & Enter to search...", fontsize = 10)
    btn_ba_search = Button(g[ba_r, 3], label = "🔍 Search",
        buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    menu_side = Menu(g[ba_r, 4], options = ["", "Right", "Left", "NA"], default = "", fontsize = 10)
    
    # Ontology dropdown for Base Anatomy (filtered by search text)
    ba_menu_r = nr!()
    ba_filtered = Observable(anatomy_ontology[1:min(50, end)])
    ba_menu = Menu(g[ba_menu_r, 1:4], options = ba_filtered, fontsize = 10)
    
    # Filter function for anatomy ontology
    function _filter_anatomy!(filtered_obs, txt, ontology)
        t = _safe_strip(txt)
        if isempty(t)
            filtered_obs[] = ontology[1:min(50, end)]
        else
            tl = lowercase(t)
            hits = filter(s -> occursin(tl, lowercase(s)), ontology)
            filtered_obs[] = isempty(hits) ? ["(no matches)"] : hits[1:min(50, end)]
        end
    end
    
    # Wire search → filter dropdown (both Enter and button click)
    on(tb_base_anat.stored_string) do txt
        _filter_anatomy!(ba_filtered, txt, anatomy_ontology)
    end
    on(btn_ba_search.clicks) do _
        _filter_anatomy!(ba_filtered, tb_base_anat.stored_string[], anatomy_ontology)
    end
    # Wire dropdown selection → fill textbox
    on(ba_menu.selection) do sel
        sel === nothing && return
        s = string(sel)
        s == "(no matches)" && return
        if tb_base_anat.stored_string[] != s
            tb_base_anat.stored_string[] = s
        end
    end
    
    # Anatomical Relations (with ontology autocomplete for the target organ)
    rel_r = nr!()
    Label(g[rel_r, 1], "Relation:", fontsize = 10, color = LBL_FG, halign = :right)
    menu_rel = Menu(g[rel_r, 2], options = ["", "Surrounded By", "Lateral To", "Medial To",
        "Anterior To", "Posterior To", "Superior To", "Inferior To", "Between", "Inside"],
        default = "", fontsize = 10)
    tb_rel_base = Textbox(g[rel_r, 3], placeholder = "type & Enter...", fontsize = 10)
    btn_rel_search = Button(g[rel_r, 4], label = "🔍",
        buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    
    # Ontology dropdown for Relation target
    rel_menu_r = nr!()
    rel_filtered = Observable(anatomy_ontology[1:min(50, end)])
    rel_menu = Menu(g[rel_menu_r, 1:4], options = rel_filtered, fontsize = 10)
    
    on(tb_rel_base.stored_string) do txt
        _filter_anatomy!(rel_filtered, txt, anatomy_ontology)
    end
    on(btn_rel_search.clicks) do _
        _filter_anatomy!(rel_filtered, tb_rel_base.stored_string[], anatomy_ontology)
    end
    on(rel_menu.selection) do sel
        sel === nothing && return
        s = string(sel)
        s == "(no matches)" && return
        if tb_rel_base.stored_string[] != s
            tb_rel_base.stored_string[] = s
        end
    end
    
    end_section!(sec_type)
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

    # TP and Modality status indicator
    tp_label_r = nr!()
    tp_status = Observable{String}("Modality & Timepoint: Initializing...")
    Label(g[tp_label_r, 1:4], tp_status, fontsize = 10, color = GRN, halign = :center, tellwidth = false)
    
    # Update TP label when TP buttons or Compare view are toggled
    function update_tp_label()
        idx = _MEH.current_tp_index[]
        label = get(_MEH.tp_labels, idx, "TP $idx")
        
        if cv_active[]
            right_idx = _MEH.compare_right_tp[]
            right_label = get(_MEH.tp_labels, right_idx, "TP $right_idx")
            tp_status[] = "Active: Left [ $label ] | Right [ $right_label ]"
        else
            tp_status[] = "Visualized Modality & TP: [ $label ]"
        end
    end

    on(btn_pt.clicks) do _
        @async begin
            sleep(0.08)
            update_tp_label()
        end
    end
    on(btn_nt.clicks) do _
        @async begin
            sleep(0.08)
            update_tp_label()
        end
    end
    on(btn_cv.clicks) do _
        @async begin
            sleep(0.08)
            update_tp_label()
        end
    end
    
    # Initialize immediately and shortly after startup to capture populated labels
    update_tp_label()
    @async begin
        sleep(0.2)
        update_tp_label()
    end

    vc2_r = nr!()
    btn_tl = Button(g[vc2_r, 1:2], label = "Toggle Lesion Overlay", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    btn_rf = Button(g[vc2_r, 3:4], label = "Refresh List",          buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    on(btn_tl.clicks) do _; put!(channel, ToggleLesionEvent()) end
    on(btn_rf.clicks) do _; put!(channel, RefreshListEvent()) end

    vc3_r = nr!()
    btn_single = Button(g[vc3_r, 1:2], label = "Show Active Single", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    btn_all = Button(g[vc3_r, 3:4], label = "Show All Lesions",      buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    on(btn_single.clicks) do _
        # extract lesion id integer safely
        id_str = active_lesion_id[]
        m = match(r"\d+", id_str)
        if m !== nothing
            put!(channel, ShowSingleLesionEvent(parse(Int, m.match)))
        end
    end
    on(btn_all.clicks) do _
        put!(channel, ShowSingleLesionEvent(0))
    end
    
    end_section!(sec_view)

    # ── Dedicated Windowing & Image Offsets Subpanel ─────────────────────────
    sec_win = begin_section!("Windowing & Image Offsets"; default_open=true)

        # CT Windowing
    ct_lbl_r = nr!()
    Label(g[ct_lbl_r, 1:4], "── CT Window & Offsets (HU) ──", fontsize = 10, color = ACCENT, halign = :center, tellwidth = false)
    
    ct_s_r = nr!()
    islider_ct = IntervalSlider(g[ct_s_r, 1:4], range = -1500.0:10.0:3000.0, startvalues = (-150.0, 250.0))
    
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
    
    ct_a_r = nr!()
    btn_apply_ct = Button(g[ct_a_r, 2:3], label = "Apply CT", buttoncolor = BLU_BTN, labelcolor = TXT, fontsize = 10)

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
    on(btn_apply_ct.clicks) do _; apply_ct_from_text(); end

    # PET Windowing
    pet_lbl_r = nr!()
    Label(g[pet_lbl_r, 1:4], "── PET Window & Offsets (SUV) ──", fontsize = 10, color = ACCENT, halign = :center, tellwidth = false)
    
    pet_s_r = nr!()
    islider_pet = IntervalSlider(g[pet_s_r, 1:4], range = 0.0:0.1:50.0, startvalues = (0.0, 10.0))
    
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
    
    pet_a_r = nr!()
    btn_apply_pet = Button(g[pet_a_r, 2:3], label = "Apply PET", buttoncolor = BLU_BTN, labelcolor = TXT, fontsize = 10)

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
    on(btn_apply_pet.clicks) do _; apply_pet_from_text(); end

    # SPECT Windowing
    spect_lbl_r = nr!()
    Label(g[spect_lbl_r, 1:4], "── SPECT Window & Offsets (Counts) ──", fontsize = 10, color = ACCENT, halign = :center, tellwidth = false)
    
    spect_s_r = nr!()
    islider_spect = IntervalSlider(g[spect_s_r, 1:4], range = 0.0:0.1:100.0, startvalues = (0.0, 10.0))
    
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
    
    spect_a_r = nr!()
    btn_apply_spect = Button(g[spect_a_r, 2:3], label = "Apply SPECT", buttoncolor = BLU_BTN, labelcolor = TXT, fontsize = 10)

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
    on(btn_apply_spect.clicks) do _; apply_spect_from_text(); end
end_section!(sec_win)

    # ── All Annotation Fields (single section, like Slicer extension) ─────────
    field_widgets = Dict{String, Any}()
    q_row_indices = Dict{String, Vector{Int}}()
    all_metadata_rows = Int[]
    
    schema_dict = Dict(q.short => q for q in schema)
    
    # All fields in order, grouped by visual sub-headers
    metadata_groups = [
        "Identification" => [
            "Lesion tracking name?",
            "Anatomic Location",
            "Anatomical Sublocation",
            "Anatomical Details"
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
        rowsize!(g, row_idx, visible ? Auto() : Fixed(0))
        if row_idx < r[1]
            rowgap!(g, row_idx, visible ? 1 : 0)
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
    
    # Single collapsible section for all metadata
    sec_meta = begin_section!("Lesion Metadata"; default_open=true)
    is_meta_open, meta_start_row, meta_header_r, meta_btn = sec_meta
    push!(all_metadata_rows, meta_header_r)

    for (group_title, q_list) in metadata_groups
        # Visual sub-header (just a colored label, not collapsible)
        sub_r = nr!()
        push!(all_metadata_rows, sub_r)
        Label(g[sub_r, 1:4], "── $group_title ──",
            fontsize = 10, color = ACCENT, halign = :left, tellwidth = false)
        section_headers[group_title] = sub_r

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
                tb = Textbox(g[q_r, 2:4],
                    placeholder = isempty(q.default_answer) ? "..." : q.default_answer,
                    fontsize = 10)
                field_widgets[q.short] = tb
            else
                opts_obs = Observable(String["- select -"; q.options])
                m = Menu(g[q_r, 2:3], options = opts_obs, fontsize = 10)
                field_widgets[q.short] = m
                
                btn_add_opt = Button(g[q_r, 4], label = "+", buttoncolor=BG_PNL, labelcolor=TXT, fontsize=10)
                
                tb_new_row = nr!()
                push!(all_metadata_rows, tb_new_row)
                push!(q_rows, tb_new_row)

                tb_new = Textbox(g[tb_new_row, 2:3], placeholder="Type new & press Enter...", fontsize=10)
                rowsize!(g, tb_new_row, Fixed(0))
                tb_new.blockscene.visible[] = false
                if tb_new_row < r[1]; rowgap!(g, tb_new_row, 0); end
                
                tb_new_visible = Observable(false)
                
                on(btn_add_opt.clicks) do _
                    tb_new_visible[] = !tb_new_visible[]
                    if tb_new_visible[]
                        rowsize!(g, tb_new_row, Auto())
                        tb_new.blockscene.visible[] = true
                        if tb_new_row < r[1]; rowgap!(g, tb_new_row, 1); end
                        tb_new.stored_string[] = ""
                    else
                        rowsize!(g, tb_new_row, Fixed(0))
                        tb_new.blockscene.visible[] = false
                        if tb_new_row < r[1]; rowgap!(g, tb_new_row, 0); end
                    end
                end
                
                on(tb_new.stored_string) do val
                    val = _safe_strip(val)
                    if !isempty(val) && !(val in opts_obs[])
                        new_opts = copy(opts_obs[])
                        push!(new_opts, val)
                        opts_obs[] = new_opts
                        m.selection[] = val
                    end
                    rowsize!(g, tb_new_row, Fixed(0))
                    tb_new.blockscene.visible[] = false
                    if tb_new_row < r[1]; rowgap!(g, tb_new_row, 0); end
                    tb_new_visible[] = false
                end
            end

            # Compact: skip tooltip row (tooltips are too verbose for compact layout)
            q_row_indices[q.short] = q_rows
        end
    end
    
    end_section!(sec_meta)

    # ── Dynamic Visibility Engine ─────────────────────────────────────────────
    function update_dynamic_visibility!(active_type::String)
        is_p = (active_type == "Prostate")
        is_bm = (active_type == "Bone Meta")
        
        # Hide Prostate Primary Score sub-header if not prostate
        if haskey(section_headers, "PRIMARY Score")
            p_hdr = section_headers["PRIMARY Score"]
            set_row_visible!(p_hdr, is_p)
        end

        for (sq, rows) in q_row_indices
            visible = true
            if sq == "PRIMARY score pattern?"
                visible = is_p
            elseif sq == "Relation to Bone Marrow (Surrounding Changes Part A)" || 
                   sq == "Periosteal Reaction (Surrounding Changes Part B)"
                visible = is_bm
            elseif sq == "PSMA-RADS 2.0"
                visible = !is_p
            elseif sq == "Alternative Hypothesis (False Positive)"
                # Retained across non-prostate and all metastatic/benign mimics
                visible = !is_p
            end

            for row_idx in rows
                set_row_visible!(row_idx, visible)
            end
        end
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

    # Auto-hide metadata section in Compare Volumes mode
    on(btn_cv.clicks) do _
        for r_idx in all_metadata_rows
            if cv_active[]
                set_row_visible!(r_idx, false)
            else
                set_row_visible!(r_idx, true)
            end
        end
        if !cv_active[]
            update_dynamic_visibility!(active_lesion_type[])
        end
    end

    # ── RadLex Multi-Value Panel ──────────────────────────────────────────────
    sec_radlex = begin_section!("RadLex Ontology Properties")
    radlex_selected = Observable(String[])

    rl_r = nr!()
    Label(g[rl_r, 1], "Search:", fontsize = 10, color = LBL_FG, halign = :right)
    rl_search = Textbox(g[rl_r, 2:3], placeholder = "type to filter RadLex terms...", fontsize = 10)
    btn_rl_add = Button(g[rl_r, 4], label = "+ Add",
        buttoncolor = GRN, labelcolor = TXT, fontsize = 10)

    radlex_filtered = Observable(length(radlex) > 200 ? radlex[1:200] : radlex)
    rl_menu = Menu(g[nr!(), 1:4], options = radlex_filtered, fontsize = 10)

    on(rl_search.stored_string) do txt
        t = _safe_strip(txt)
        if isempty(t)
            radlex_filtered[] = length(radlex) > 200 ? radlex[1:200] : radlex
        else
            tl = lowercase(t)
            hits = filter(s -> occursin(tl, lowercase(s)), radlex)
            radlex_filtered[] = length(hits) > 200 ? hits[1:200] : hits
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
    r[1] = rl_slot_row_start + RL_SLOTS - 1

    on(radlex_selected) do terms
        for i in 1:RL_SLOTS
            rl_labels[i].text[] = i <= length(terms) ? terms[i] : ""
        end
    end
    for i in 1:RL_SLOTS
        on(rl_rm_btns[i].clicks) do _
            cur = copy(radlex_selected[])
            i <= length(cur) && deleteat!(cur, i)
            radlex_selected[] = cur
        end
    end

    end_section!(sec_radlex)

    # ── Custom Key-Value Fields ───────────────────────────────────────────────
    sec_custom = begin_section!("Custom Key-Value Fields")
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

    end_section!(sec_custom)

    # ── Segmentation Mini Manager ────────────────────────────────────────────
    sec_seg = begin_section!("Segmentation Mini Manager")
    btn_new_lesion = Button(g[nr!(), 1:4], label = "Create New Empty Lesion", buttoncolor = BLU_BTN, labelcolor = TXT, fontsize = 10)
    on(btn_new_lesion.clicks) do _
        max_id = 0
        for opt in lesion_ids[]
            m = match(r"^(\d+)", opt)
            if m !== nothing
                max_id = max(max_id, parse(Int, m.match))
            end
        end
        for k in keys(lesion_db[])
            m = match(r"^(\d+)", k)
            if m !== nothing
                max_id = max(max_id, parse(Int, m.match))
            end
        end
        new_id = max_id + 1
        new_name = "$(new_id) - New Lesion"
        
        db = copy(lesion_db[])
        db[new_name] = Dict{String, Any}()
        lesion_db[] = db
        
        opts = copy(lesion_ids[])
        push!(opts, new_name)
        
        is_syncing_selection[] = true
        try
            lesion_ids[] = opts
            les_menu.options[] = opts
            les_menu.selection[] = new_name
            les_menu.i_selected[] = length(opts)
        finally
            is_syncing_selection[] = false
        end
        active_lesion_id[] = new_name
        
        # Auto-switch to paint mode
        current_paint_mode[] = :paint
        btn_paint.buttoncolor[] = GRN
        btn_erase.buttoncolor[] = BG_PNL
        btn_view_mode.buttoncolor[] = BG_PNL
        put!(channel, PaintValEvent(new_id, true))
        put!(channel, SyncLesionEvent(new_id))
    end

    brush_r = nr!()
    Label(g[brush_r, 1], "Brush Size:", halign=:left, fontsize=11, color=TXT)
    slider_brush = Slider(g[brush_r, 2:4], range = 1:20, startvalue = 1)
    on(slider_brush.value) do val
        put!(channel, ChangeBrushSizeEvent(val))
    end

    pe_r = nr!()
    btn_paint     = Button(g[pe_r, 1], label = "Paint", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    btn_erase     = Button(g[pe_r, 2], label = "Erase", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    btn_view_mode = Button(g[pe_r, 3:4], label = "View (No Paint)", buttoncolor = BLU_BTN, labelcolor = TXT, fontsize = 10)
    
    move_r = nr!()
    btn_move_lesion = Button(g[move_r, 1:4], label = "Move Lesion", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    
    move_lesion_active = Ref(false)
    on(btn_move_lesion.clicks) do _
        move_lesion_active[] = !move_lesion_active[]
        btn_move_lesion.buttoncolor[] = move_lesion_active[] ? GRN : BG_PNL
        put!(channel, ToggleMoveLesionModeEvent(move_lesion_active[]))
    end
    
    current_paint_mode = Observable(:view)
    
    on(btn_paint.clicks) do _
        current_paint_mode[] = :paint
        btn_paint.buttoncolor[] = GRN
        btn_erase.buttoncolor[] = BG_PNL
        btn_view_mode.buttoncolor[] = BG_PNL
        m = match(r"^(\d+)", active_lesion_id[])
        val = m !== nothing ? parse(Int, m.match) : 1
        put!(channel, PaintValEvent(val, true))
    end
    on(btn_erase.clicks) do _
        current_paint_mode[] = :erase
        btn_paint.buttoncolor[] = BG_PNL
        btn_erase.buttoncolor[] = RED_BTN
        btn_view_mode.buttoncolor[] = BG_PNL
        put!(channel, PaintValEvent(0, true))
    end
    on(btn_view_mode.clicks) do _
        current_paint_mode[] = :view
        btn_paint.buttoncolor[] = BG_PNL
        btn_erase.buttoncolor[] = BG_PNL
        btn_view_mode.buttoncolor[] = BLU_BTN
        put!(channel, PaintValEvent(-1, false))
    end

    algo_r = nr!()
    Label(g[algo_r, 1], "Algorithm:", halign=:left, fontsize=11, color=TXT)
    algo_combo = Menu(g[algo_r, 2:4], options = ["HELPNet (AI)", "NNInteractive", "Traditional (PETTumor)"], default = "HELPNet (AI)", fontsize = 10)

    btn_add_ai = Button(g[nr!(), 1:4], label = "Run Semiauto AI (HELPNet / nnInteractive)", buttoncolor = GRN, labelcolor = TXT, fontsize = 10)
    on(btn_add_ai.clicks) do _; put!(channel, AddAutoPetEvent(algo_combo.selection[], channel)) end

    # AI status label — shows real-time processing state to the user
    Label(g[nr!(), 1:4], @lift(string($(_MEH.ai_status_text))),
        fontsize=10, color=RGBAf(0.7, 0.9, 0.7, 1.0), halign=:center)


    ai_r2 = nr!()
    tog_lesion = Toggle(g[ai_r2, 1], active=true)
    Label(g[ai_r2, 2], "Lesion", fontsize=11, color=TXT, halign=:left)
    
    tog_surface = Toggle(g[ai_r2, 3], active=true)
    Label(g[ai_r2, 4], "Bone Surface", fontsize=11, color=TXT, halign=:left)
    
    ai_r3 = nr!()
    tog_marrow = Toggle(g[ai_r3, 1], active=true)
    Label(g[ai_r3, 2], "Bone Marrow", fontsize=11, color=TXT, halign=:left)
    
    on(tog_lesion.active) do val; put!(channel, ShowMaskLayerEvent(1, val)) end
    on(tog_surface.active) do val; put!(channel, ShowMaskLayerEvent(2, val)) end
    on(tog_marrow.active) do val; put!(channel, ShowMaskLayerEvent(3, val)) end

    end_section!(sec_seg)

    # ── Active Data Settings ──────────────────────────────────────────────────
    sec_ads = begin_section!("Active Data Settings")
    ads_r1 = nr!()
    Label(g[ads_r1, 1], "PET/SUV Node:", fontsize = 10, color = LBL_FG, halign = :right)
    Menu(g[ads_r1, 2], options = ["Auto", "SUV_PET_Image_0", "SUV_PET_Image_1"], fontsize = 10)
    Label(g[ads_r1, 3], "CT Node:", fontsize = 10, color = LBL_FG, halign = :right)
    Menu(g[ads_r1, 4], options = ["Auto", "Fixed_CT_Volume_0", "Fixed_CT_Volume_1"], fontsize = 10)

    ads_r2 = nr!()
    Label(g[ads_r2, 1], "Lesion Mask:", fontsize = 10, color = LBL_FG, halign = :right)
    Menu(g[ads_r2, 2], options = ["Auto", "Segmentation_0", "Segmentation_1"], fontsize = 10)
    Label(g[ads_r2, 3], "Anatomy Atlas:", fontsize = 10, color = LBL_FG, halign = :right)
    Menu(g[ads_r2, 4], options = ["None", "Bone_Mask", "Organ_Mask"], fontsize = 10)

    ads_r3 = nr!()
    Label(g[ads_r3, 1], "To Next TP Xform:", fontsize = 10, color = LBL_FG, halign = :right)
    Menu(g[ads_r3, 2], options = ["None", "Elastic_Transform_0_to_1"], fontsize = 10)
    Label(g[ads_r3, 3], "To Prev TP Xform:", fontsize = 10, color = LBL_FG, halign = :right)
    Menu(g[ads_r3, 4], options = ["None", "Elastic_Transform_1_to_0"], fontsize = 10)

    btn_map_link = Button(g[nr!(), 1:4], label = "Explicitly Map Selected Lesions (Link)", buttoncolor = BLU_BTN, labelcolor = TXT, fontsize = 10)
    on(btn_map_link.clicks) do _; put!(channel, MapLinkEvent()) end

    end_section!(sec_ads)

    # ── Preprocessing & Automation ────────────────────────────────────────────
    sec_pre = begin_section!("Preprocessing & Scene Management")
    pre_r1 = nr!()
    tog_autorun = Toggle(g[pre_r1, 1], active=false)
    Label(g[pre_r1, 2:4], "Auto-run preprocessing on scene load", fontsize = 11, color = TXT, halign = :left)
    on(tog_autorun.active) do val; put!(channel, AutoRunPreprocessEvent(val)) end



    btn_save_mrb = Button(g[nr!(), 1:4], label = "Save Scene as Preprocessed MRB...", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    on(btn_save_mrb.clicks) do _; put!(channel, SaveMRBEvent()) end

    end_section!(sec_pre)

    # ── Persistence ──────────────────────────────────────────────────────────
    sec_persist = begin_section!("Save & Load")
    sv_r = nr!()
    btn_save = Button(g[sv_r, 1:2], label = "Save Annotations",
        buttoncolor = GRN, labelcolor = TXT, fontsize = 10)
    btn_load = Button(g[sv_r, 3:4], label = "Load Annotations",
        buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
    status_lbl = Label(g[nr!(), 1:4], "",
        fontsize = 11, color = RGBf(0.4, 0.9, 0.4), halign = :left, tellwidth = false)

    end_section!(sec_persist)

    # ── Report Panel ─────────────────────────────────────────────────────────
    sec_report = begin_section!("Radiological Report Output")
    Label(g[nr!(), 1:4], "Original Dictation:", fontsize = 10, color = LBL_FG, halign = :left)
    dict_tb = Textbox(g[nr!(), 1:4], placeholder = "Enter radiological dictation here...", fontsize = 10)
    
    Label(g[nr!(), 1:4], "Generated Description:", fontsize = 10, color = LBL_FG, halign = :left)
    rpt_tb = Textbox(g[nr!(), 1:4],
        placeholder = "Click 'Generate Report' to build structured summary...",
        fontsize = 10)
    gen_r = nr!()
    btn_gen = Button(g[gen_r, 1:2], label = "Generate Report",
        buttoncolor = BLU_BTN, labelcolor = TXT, fontsize = 10)

    end_section!(sec_report)

    # ── Collect / apply UI state ──────────────────────────────────────────────
    function collect_state()::Dict{String,String}
        d = Dict{String,String}()
        d["LesionType"] = active_lesion_type[]
        v_base = _safe_strip(tb_base_anat.stored_string[])
        isempty(v_base) || (d["BaseAnatomy"] = v_base)
        side_sel = menu_side.selection[]
        if side_sel !== nothing && !isempty(string(side_sel))
            d["BaseAnatomySide"] = string(side_sel)
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
                v = _safe_strip(w.stored_string[])
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
        v_dict = _safe_strip(dict_tb.stored_string[])
        isempty(v_dict) || (d["RadiologicalDictation"] = v_dict)
        v_rpt = _safe_strip(rpt_tb.stored_string[])
        isempty(v_rpt) || (d["RadiologicalReportOutput"] = v_rpt)
        return d
    end

    function apply_state(data::AbstractDict)
        cur_id_str = active_lesion_id[]
        m = match(r"^(\d+)", cur_id_str)
        lid = m !== nothing ? parse(Int, m.match) : 1

        t_type = if haskey(data, "LesionType")
            data["LesionType"]
        else
            # Auto-detect lesion type from available info.
            # Mirrors Slicer extension's categorization logic (LesionMetadata.py L4258-4266).
            
            # Check BaseAnatomy text if available (from previous detection or TotalSegmentator)
            base_anat = lowercase(get(data, "BaseAnatomy", ""))
            id_low = lowercase(cur_id_str)
            # Also check global_organ_mapping for the raw TS organ name
            raw_organ = lowercase(get(_MEH.global_organ_mapping[], lid, ""))
            # Combine all sources for keyword matching
            combined = base_anat * " " * id_low * " " * raw_organ
            
            # Bone keywords from TotalSegmentator segment names — must exclude
            # vascular structures that share similar names (e.g. "iliac_artery")
            bone_kws = ["femur", "hip", "vertebra", "rib", "sacrum", "clavicula",
                        "clavicle", "humerus", "scapula", "sternum", "skull",
                        "palate", "bone", "spine", "ilium", "ischium", "pubis",
                        "tibia", "radius", "carpal", "tarsal", "costal_cartilage"]
            vascular_exclusions = ["vena", "artery", "vein", "vessel", "trunk"]
            
            is_bone_kw = any(kw -> occursin(kw, combined), bone_kws) &&
                         !any(v -> occursin(v, combined), vascular_exclusions)
            
            # Only consider bone_subsegments_cache if it actually has non-empty data
            # (empty entries are cached for non-bone lesions to avoid re-computation)
            has_real_bone_subseg = if haskey(_MEH.bone_subsegments_cache, lid)
                cached_data = _MEH.bone_subsegments_cache[lid]
                cached_data isa Tuple && length(cached_data) >= 2 &&
                    !isempty(cached_data[1]) || !isempty(cached_data[2])
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
        update_type_buttons(t_type)
        
        if t_type == "Bone Meta"
            tog_surface.active[] = true
            tog_marrow.active[] = true
        else
            tog_surface.active[] = false
            tog_marrow.active[] = false
        end

        
        t_base = get(data, "BaseAnatomy", "")
        t_side = get(data, "BaseAnatomySide", "")
        
        # Auto-detect BaseAnatomy from TotalSegmentator organ mapping if not saved
        if isempty(t_base) && lid > 0
            organ_map = _MEH.global_organ_mapping[]
            raw_organ = get(organ_map, lid, "")
            if !isempty(raw_organ)
                t_base, auto_side = map_ts_to_anatomy(raw_organ)
                if isempty(t_side) && !isempty(auto_side)
                    t_side = auto_side
                end
                @info "Auto-detected BaseAnatomy for lesion $lid: '$t_base' (side='$t_side') from TS organ '$raw_organ'"
            end
        end
        
        if tb_base_anat.stored_string[] != t_base
            tb_base_anat.stored_string[] = t_base
        end
        
        side_opts = menu_side.options[]
        s_idx = findfirst(==(t_side), side_opts)
        menu_side.i_selected[] = s_idx !== nothing ? s_idx : 1
        
        # Restore windowing if present
        if haskey(data, "_CT_Min") && haskey(data, "_CT_Max")
            tb_ct_min.stored_string[] = data["_CT_Min"]
            tb_ct_max.stored_string[] = data["_CT_Max"]
            v_min = tryparse(Float32, data["_CT_Min"])
            v_max = tryparse(Float32, data["_CT_Max"])
            if v_min !== nothing && v_max !== nothing
                put!(channel, WindowingEvent("CT", v_min, v_max))
            end
        end
        if haskey(data, "_PET_Min") && haskey(data, "_PET_Max")
            tb_pet_min.stored_string[] = data["_PET_Min"]
            tb_pet_max.stored_string[] = data["_PET_Max"]
            v_min = tryparse(Float32, data["_PET_Min"])
            v_max = tryparse(Float32, data["_PET_Max"])
            if v_min !== nothing && v_max !== nothing
                put!(channel, WindowingEvent("PET", v_min, v_max))
            end
        end
        if haskey(data, "_SPECT_Min") && haskey(data, "_SPECT_Max")
            tb_spect_min.stored_string[] = data["_SPECT_Min"]
            tb_spect_max.stored_string[] = data["_SPECT_Max"]
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
                target_str = val === nothing ? "" : val
                if w.stored_string[] != target_str
                    w.stored_string[] = target_str
                end
            elseif w isa Menu
                if val !== nothing
                    opts = w.options[]
                    idx = findfirst(==(val), opts)
                    target_idx = idx !== nothing ? idx : 1
                    if w.i_selected[] != target_idx
                        w.i_selected[] = target_idx
                    end
                else
                    if w.i_selected[] != 1
                        w.i_selected[] = 1   # reset to "- select -"
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
        
        target_dict = get(data, "RadiologicalDictation", get(_MEH.tp_descriptions, _MEH.current_tp_index[], ""))
        if dict_tb.stored_string[] != target_dict
            dict_tb.stored_string[] = target_dict
        end
        
        target_rpt = get(data, "RadiologicalReportOutput", "")
        if rpt_tb.stored_string[] != target_rpt
            rpt_tb.stored_string[] = target_rpt
        end
    end

    # ── Wire callbacks ────────────────────────────────────────────────────────
    on(active_lesion_id) do id
        @info "WIRE_CALLBACK: active_lesion_id changed to: $id"
        db = lesion_db[]
        try
            apply_state(get(db, id, Dict{String,String}()))
        catch e
            @warn "Failed to apply state for lesion $id: $e"
        end
        
        # Synchronize lesion with viewer (filters mask and jumps to slice)
        try
            m = match(r"^(\d+)", id)
            if m !== nothing
                lid = parse(Int, m.match)
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
        id = active_lesion_id[]
        db = copy(lesion_db[])
        db[id] = collect_state()
        
        # Persist global windowing
        db["_GLOBAL_APP_STATE"] = Dict{String,String}(
            "CT_Min" => _safe_strip(tb_ct_min.stored_string[]),
            "CT_Max" => _safe_strip(tb_ct_max.stored_string[]),
            "PET_Min" => _safe_strip(tb_pet_min.stored_string[]),
            "PET_Max" => _safe_strip(tb_pet_max.stored_string[]),
            "SPECT_Min" => _safe_strip(tb_spect_min.stored_string[]),
            "SPECT_Max" => _safe_strip(tb_spect_max.stored_string[])
        )
        global_st = Dict{String, Any}(
            "windowing" => Dict(
                "CT" => [_safe_strip(tb_ct_min.stored_string[]), _safe_strip(tb_ct_max.stored_string[])],
                "PET" => [_safe_strip(tb_pet_min.stored_string[]), _safe_strip(tb_pet_max.stored_string[])],
                "SPECT" => [_safe_strip(tb_spect_min.stored_string[]), _safe_strip(tb_spect_max.stored_string[])]
            )
        )
        lesion_db[] = db
        put!(db_channel, SaveDBMessage(db, global_st, save_path, DEFAULT_HDF5_PATH))
        status_lbl.text[] = "Saved at $(Dates.format(Dates.now(), "HH:MM:SS"))"
    end

    on(btn_load.clicks) do _
        @async begin
            reply = Channel{Dict}(1)
            put!(db_channel, LoadDBMessage(save_path, reply))
            db = take!(reply)
            lesion_db[] = db
            
            # Apply global state if found
            if haskey(db, "_GLOBAL_APP_STATE")
                gst = db["_GLOBAL_APP_STATE"]
                apply_state(gst)
            end
            
            apply_state(get(db, active_lesion_id[], Dict{String,String}()))
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
        id = active_lesion_id[]
        db = copy(lesion_db[])
        db[id] = collect_state()
        lesion_db[] = db
    end

    # Auto-save logic
    global_app_state = Dict{String, Any}("windowing" => (0.0f0, 0.0f0))
    if haskey(ui_hooks, :windowing)
        on(ui_hooks[:windowing]) do val
            global_app_state["windowing"] = val
        end
    end
    
    # Auto-update lesion_db on dictation and custom changes to ensure background task catches it
    on(dict_tb.stored_string) do _
        id = active_lesion_id[]
        db = copy(lesion_db[])
        db[id] = collect_state()
        lesion_db[] = db
    end

    @async begin
        last_save_time = time()
        while true
            sleep(5.0)
            try
                db = lesion_db[]
                put!(db_channel, SaveDBMessage(db, global_app_state, save_path, DEFAULT_HDF5_PATH))
            catch e
                @warn "Background autosave enqueue failed" e
            end
        end
    end

    return (fig = fig, lesion_db = lesion_db)
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
