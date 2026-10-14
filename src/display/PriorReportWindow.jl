module PriorReportWindow

using GLMakie
using Observables
import Dates
import ..SegmentationDisplay: synchronized_makie_renderloop, GLOBAL_OPENGL_LOCK
import ..SegmentationDisplay.MakieEventHandlers as _MEH
import ..LLMDictation

export open_prior_report_window, close_prior_report_window, active_prior_screen, active_prior_fig, update_prior_report_tp!

# ── Global State & References ────────────────────────────────────────────────
const active_prior_screen = Ref{Any}(nothing)
const active_prior_fig = Ref{Any}(nothing)
const current_prior_tp = Ref{Int}(0)
const current_prior_lang = Observable("EN")

const report_text_obs = Observable{String}("")
const report_title_obs = Observable{String}("Original Radiological Report / Vorbefund")
const report_info_obs = Observable{String}("Study: Initializing...")
const report_status_obs = Observable{String}("Ready")
const report_status_color = Observable{RGBf}(RGBf(0.70, 0.75, 0.80))
const on_copy_to_anamnese = Ref{Any}(nothing)

"""
    update_prior_report_tp!(tp::Int; lang::String = current_prior_lang[])

Updates the displayed report content, header info, and status for the given time point.
Automatically called when switching time points or language.
"""
function update_prior_report_tp!(tp::Int; lang::String = current_prior_lang[])
    current_prior_tp[] = tp
    current_prior_lang[] = lang
    is_en = uppercase(lang) == "EN"

    lbl = (_MEH !== nothing && haskey(_MEH.tp_labels, tp)) ? _MEH.tp_labels[tp] : "TP $tp"
    modality = (_MEH !== nothing && haskey(_MEH.tp_modalities, tp)) ? _MEH.tp_modalities[tp] : "PET/CT"
    pat_id = (_MEH !== nothing && !isempty(_MEH.patient_id[])) ? _MEH.patient_id[] : "PAT_001"

    en_desc = (_MEH !== nothing && haskey(_MEH.tp_english_descriptions, tp)) ? _MEH.tp_english_descriptions[tp] : ""
    de_desc = (_MEH !== nothing && haskey(_MEH.tp_descriptions, tp)) ? _MEH.tp_descriptions[tp] : ""

    report_info_obs[] = "Patient: $pat_id  |  Study: $lbl  |  Modality: $modality  |  TP: $tp"
    report_title_obs[] = is_en ?
        "Original Radiological Report / Vorbefund — $lbl" :
        "Originalradiologischer Befund (Vorbefund) — $lbl"

    if is_en
        if !isempty(en_desc)
            report_text_obs[] = en_desc
            report_status_obs[] = "Showing English translated report ($(length(en_desc)) chars)"
            report_status_color[] = RGBf(0.18, 0.65, 0.30)
        elseif !isempty(de_desc)
            report_text_obs[] = "(English translation unavailable — showing German original):\n\n" * de_desc
            report_status_obs[] = "German original displayed (click '[T] Translate to English' to translate)"
            report_status_color[] = RGBf(0.95, 0.77, 0.20)
        else
            report_text_obs[] = "(No original radiological report found in metadata for this time point)"
            report_status_obs[] = "No report available for TP $tp in metadata.json"
            report_status_color[] = RGBf(0.70, 0.75, 0.80)
        end
    else
        if !isempty(de_desc)
            report_text_obs[] = de_desc
            report_status_obs[] = "Originaler deutscher Befund ($(length(de_desc)) Zeichen)"
            report_status_color[] = RGBf(0.18, 0.65, 0.30)
        elseif !isempty(en_desc)
            report_text_obs[] = "(Originalbefund auf Deutsch nicht verfügbar — zeige englische Übersetzung):\n\n" * en_desc
            report_status_obs[] = "Zeige englische Version"
            report_status_color[] = RGBf(0.95, 0.77, 0.20)
        else
            report_text_obs[] = "(Kein radiologischer Befund für diesen Zeitpunkt in den Metadaten vorhanden)"
            report_status_obs[] = "Kein Befund für TP $tp gefunden"
            report_status_color[] = RGBf(0.70, 0.75, 0.80)
        end
    end
end

"""
    open_prior_report_window(tp_idx::Int = (_MEH !== nothing ? _MEH.current_tp_index[] : 0); lang::String = current_prior_lang[])

Opens (or focuses/reloads) the dedicated, separate small window displaying the original / prior
radiological report for the current time point.
Features language switching (German / English), translation on demand, and clipboard copy.
"""
function open_prior_report_window(
    tp_idx::Int = (_MEH !== nothing ? _MEH.current_tp_index[] : 0);
    lang::String = current_prior_lang[]
)
    # Check if window already open; reuse screen for instant responsiveness
    screen = active_prior_screen[]
    local reuse_screen = false
    if screen !== nothing
        try
            if isopen(screen)
                reuse_screen = true
            else
                screen = nothing
            end
        catch
            screen = nothing
        end
    end

    # Theme colors
    BG        = RGBf(0.08, 0.10, 0.13)
    BG_CARD   = RGBf(0.12, 0.14, 0.18)
    BG_INSET  = RGBf(0.16, 0.18, 0.23)
    HEADER_BG = RGBf(0.11, 0.21, 0.36)
    ACCENT    = RGBf(0.20, 0.60, 1.00)
    GOLD      = RGBf(0.95, 0.77, 0.20)
    TXT       = RGBf(0.95, 0.95, 0.95)
    SUBTXT    = RGBf(0.70, 0.75, 0.80)
    GRN       = RGBf(0.18, 0.65, 0.30)
    RED       = RGBf(0.85, 0.25, 0.25)
    BORDER    = RGBf(0.25, 0.28, 0.35)

    if reuse_screen
        update_prior_report_tp!(tp_idx; lang = lang)
        lock(GLOBAL_OPENGL_LOCK) do
            try
                GLMakie.GLFW.RestoreWindow(screen.glscreen)
                GLMakie.GLFW.ShowWindow(screen.glscreen)
                GLMakie.GLFW.FocusWindow(screen.glscreen)
            catch; end
        end
        return screen
    end

    # Create compact separate Figure
    fig = Figure(size = (960, 720), backgroundcolor = BG, figure_padding = (16, 16, 12, 12))
    active_prior_fig[] = fig

    # ── Row 1: Header & Control Toolbar ──────────────────────────────────────
    tb_layout = GridLayout(fig[1, 1], tellwidth = false)
    rowsize!(fig.layout, 1, Fixed(44))

    Box(tb_layout[1, 1:8], color = HEADER_BG, cornerradius = 4, strokecolor = BORDER, strokewidth = 1)

    lbl_title = Label(tb_layout[1, 1], report_title_obs, fontsize = 13, font = :bold, color = ACCENT, halign = :left, padding = (12, 0, 0, 0))

    btn_lang_en = Button(tb_layout[1, 2], label = "[EN] English", buttoncolor = (uppercase(lang) == "EN" ? GRN : BG_CARD), labelcolor = TXT, fontsize = 11, width = 100)
    btn_lang_de = Button(tb_layout[1, 3], label = "[DE] Deutsch", buttoncolor = (uppercase(lang) == "DE" ? GRN : BG_CARD), labelcolor = TXT, fontsize = 11, width = 100)
    btn_translate = Button(tb_layout[1, 4], label = "[T] Translate to EN", buttoncolor = ACCENT, labelcolor = TXT, fontsize = 11, width = 145)
    btn_copy = Button(tb_layout[1, 5], label = "[C] Copy", buttoncolor = BG_CARD, labelcolor = TXT, fontsize = 11, width = 85)
    btn_anamnese = Button(tb_layout[1, 6], label = "[->] To Anamnese", buttoncolor = BG_CARD, labelcolor = TXT, fontsize = 11, width = 125)
    btn_close = Button(tb_layout[1, 7], label = "[X] Close", buttoncolor = RGBf(0.35, 0.15, 0.15), labelcolor = TXT, fontsize = 11, width = 75)

    colsize!(tb_layout, 1, Relative(0.40))
    colsize!(tb_layout, 2, Fixed(104))
    colsize!(tb_layout, 3, Fixed(104))
    colsize!(tb_layout, 4, Fixed(150))
    colsize!(tb_layout, 5, Fixed(90))
    colsize!(tb_layout, 6, Fixed(130))
    colsize!(tb_layout, 7, Fixed(80))

    # ── Row 2: Metadata Info & Status Bar ────────────────────────────────────
    status_layout = GridLayout(fig[2, 1], tellwidth = false)
    rowsize!(fig.layout, 2, Fixed(24))

    Label(status_layout[1, 1], report_info_obs, fontsize = 9, color = SUBTXT, halign = :left)
    Label(status_layout[1, 2], report_status_obs, fontsize = 9, color = report_status_color, halign = :right)

    # ── Row 3: Scrollable Report Text Area ───────────────────────────────────
    body_scroll = GridLayout(fig[3, 1])
    rowsize!(fig.layout, 3, Relative(1.0))

    sl = Slider(body_scroll[1, 2], range = 0.0:0.005:1.0, startvalue = 1.0, horizontal = false, tellheight = false)
    text_gl = GridLayout(body_scroll[1, 1], tellheight = false, halign = :left, valign = :top)

    colsize!(body_scroll, 1, Relative(0.975))
    colsize!(body_scroll, 2, Fixed(18))
    colsize!(text_gl, 1, Relative(1.0))

    # Background card for readability
    Box(text_gl[1, 1], color = BG_INSET, cornerradius = 6, strokecolor = BORDER, strokewidth = 1)
    
    # Read-only text display with full wrapping
    report_lbl = Label(text_gl[1, 1], report_text_obs,
        word_wrap = true,
        tellwidth = false,
        tellheight = false,
        fontsize = 11,
        color = TXT,
        halign = :left,
        valign = :top,
        padding = (16, 16, 16, 16)
    )

    # ── Direct BBox Scroll Engine ────────────────────────────────────────────
    _scroll_offset_px = Ref(0.0f0)
    _max_scroll_px = Ref(0.0f0)
    _bbox_cache = Dict{Int, Makie.GeometryBasics.HyperRectangle{2, Float32}}()
    _bbox_cache_ready = Ref(false)
    _is_syncing_slider = Ref(false)
    Rect2f_scroll = Makie.GeometryBasics.HyperRectangle{2, Float32}

    function _cache_widget_bboxes!()
        empty!(_bbox_cache)
        for (i, c) in enumerate(text_gl.content)
            w = c.content
            bb = w.layoutobservables.computedbbox[]
            _bbox_cache[i] = Rect2f_scroll(Float32[bb.origin[1], bb.origin[2]], Float32[bb.widths[1], bb.widths[2]])
        end
        if !isempty(_bbox_cache)
            # Estimate text height from line count (approx 18px per wrapped line)
            txt = report_text_obs[]
            line_count = count(==('\n'), txt) + div(length(txt), 90) + 2
            estimated_text_h = Float32(line_count * 20.0f0 + 40.0f0)

            window_h = Float32(body_scroll.layoutobservables.computedbbox[].widths[2])
            if window_h < 100.0f0
                window_h = Float32(size(fig.scene)[2] - 100)
            end
            _max_scroll_px[] = max(0.0f0, estimated_text_h - window_h + 40.0f0)
            _bbox_cache_ready[] = true
            return true
        end
        return false
    end

    function _apply_scroll!(offset_px::Float32; sync_slider::Bool = true)
        _scroll_offset_px[] = offset_px
        !_bbox_cache_ready[] && return
        for (i, c) in enumerate(text_gl.content)
            w = c.content
            orig = get(_bbox_cache, i, nothing)
            orig === nothing && continue
            new_bbox = Rect2f_scroll(Float32[orig.origin[1], orig.origin[2] + offset_px], orig.widths)
            w.layoutobservables.computedbbox[] = new_bbox
        end
        if sync_slider && _max_scroll_px[] > 1.0f0
            _is_syncing_slider[] = true
            try
                sl_val = clamp(1.0 - Float64(offset_px / _max_scroll_px[]), 0.0, 1.0)
                set_close_to!(sl, sl_val)
            finally
                _is_syncing_slider[] = false
            end
        end
    end

    on(sl.value) do val
        _is_syncing_slider[] && return
        if !_bbox_cache_ready[]
            _cache_widget_bboxes!() && (text_gl.block_updates = true)
        end
        !_bbox_cache_ready[] && return
        _max_scroll_px[] <= 1.0f0 && return
        target_offset = Float32((1.0 - val) * _max_scroll_px[])
        _apply_scroll!(target_offset; sync_slider = false)
    end

    on(fig.scene.events.scroll) do scroll
        if !_bbox_cache_ready[]
            _cache_widget_bboxes!() && (text_gl.block_updates = true)
        end
        !_bbox_cache_ready[] && return Consume(true)
        _max_scroll_px[] <= 1.0f0 && return Consume(true)
        step_px = Float32(-scroll[2] * 60.0f0)
        new_offset = clamp(_scroll_offset_px[] + step_px, 0.0f0, _max_scroll_px[])
        _apply_scroll!(new_offset; sync_slider = true)
        return Consume(true)
    end

    # ── Interactive Controls Handlers ────────────────────────────────────────
    on(btn_lang_en.clicks) do _
        btn_lang_en.buttoncolor[] = GRN
        btn_lang_de.buttoncolor[] = BG_CARD
        update_prior_report_tp!(current_prior_tp[]; lang = "EN")
        _bbox_cache_ready[] = false
        _cache_widget_bboxes!()
        _apply_scroll!(0.0f0; sync_slider = true)
    end

    on(btn_lang_de.clicks) do _
        btn_lang_en.buttoncolor[] = BG_CARD
        btn_lang_de.buttoncolor[] = GRN
        update_prior_report_tp!(current_prior_tp[]; lang = "DE")
        _bbox_cache_ready[] = false
        _cache_widget_bboxes!()
        _apply_scroll!(0.0f0; sync_slider = true)
    end

    # On-the-fly Translation via LLM
    on(btn_translate.clicks) do _
        tp = current_prior_tp[]
        de_text = (_MEH !== nothing && haskey(_MEH.tp_descriptions, tp)) ? _MEH.tp_descriptions[tp] : ""
        if isempty(strip(de_text))
            report_status_obs[] = "[!] No German text available to translate for TP $tp"
            report_status_color[] = RED
            return
        end

        report_status_obs[] = "[...] Translating German report via LLM (DIZ / AcademicCloud)..."
        report_status_color[] = GOLD
        btn_translate.buttoncolor[] = GOLD

        @async begin
            try
                t0 = time_ns()
                translated = LLMDictation.translate_report_to_english(de_text)
                elapsed = round((time_ns() - t0) / 1e9, digits = 1)
                if !isempty(translated) && !startswith(translated, "Error") && !startswith(translated, "API Error")
                    if _MEH !== nothing
                        _MEH.tp_english_descriptions[tp] = translated
                    end
                    btn_lang_en.buttoncolor[] = GRN
                    btn_lang_de.buttoncolor[] = BG_CARD
                    update_prior_report_tp!(tp; lang = "EN")
                    report_status_obs[] = "[OK] Translation completed in $(elapsed)s ($(length(translated)) chars)"
                    report_status_color[] = GRN
                    btn_translate.buttoncolor[] = ACCENT
                    _bbox_cache_ready[] = false
                    _cache_widget_bboxes!()
                    _apply_scroll!(0.0f0; sync_slider = true)
                else
                    report_status_obs[] = "[ERR] Translation returned empty or error response"
                    report_status_color[] = RED
                    btn_translate.buttoncolor[] = RED
                end
            catch e
                @warn "[PriorReportWindow] Translation failed: $e"
                report_status_obs[] = "[ERR] Translation failed: $e"
                report_status_color[] = RED
                btn_translate.buttoncolor[] = RED
            end
        end
    end

    # Copy to Clipboard
    on(btn_copy.clicks) do _
        txt = report_text_obs[]
        if !isempty(txt) && !startswith(txt, "(")
            ok = LLMDictation.copy_to_clipboard(txt)
            if ok
                report_status_obs[] = "[OK] Copied report to clipboard"
                report_status_color[] = GRN
            else
                report_status_obs[] = "Copied to clipboard (check stdout fallback)"
                report_status_color[] = SUBTXT
            end
        else
            report_status_obs[] = "[!] No content to copy"
            report_status_color[] = RED
        end
    end

    # Copy to Anamnese Notes
    on(btn_anamnese.clicks) do _
        txt = report_text_obs[]
        if !isempty(txt) && !startswith(txt, "(")
            if on_copy_to_anamnese[] !== nothing
                try
                    on_copy_to_anamnese[](txt)
                    report_status_obs[] = "[OK] Copied into Anamnese notes successfully"
                    report_status_color[] = GRN
                catch e
                    report_status_obs[] = "[ERR] Failed copying to Anamnese: $e"
                    report_status_color[] = RED
                end
            else
                # Fallback: copy to clipboard
                LLMDictation.copy_to_clipboard(txt)
                report_status_obs[] = "[OK] Copied to clipboard (paste into Anamnese)"
                report_status_color[] = GRN
            end
        end
    end

    # Close Window
    on(btn_close.clicks) do _
        close_prior_report_window()
    end

    # Populate initial content for tp_idx
    update_prior_report_tp!(tp_idx; lang = lang)

    # Display window using synchronized renderloop
    screen = lock(GLOBAL_OPENGL_LOCK) do
        s = GLMakie.Screen(;
            renderloop = synchronized_makie_renderloop,
            title = "Radiological Report / Vorbefund — TP $tp_idx",
            focus_on_show = true
        )
        display(s, fig)
        s
    end

    active_prior_screen[] = screen
    active_prior_fig[] = fig

    # Initialize bbox cache once layout settles
    @async begin
        for _ in 1:15
            sleep(0.04)
            if _cache_widget_bboxes!()
                text_gl.block_updates = true
                _apply_scroll!(0.0f0; sync_slider = true)
                break
            end
        end
    end

    println("[PriorReport] Prior report window opened for TP $tp_idx ($lang)")
    return screen
end

"""
    close_prior_report_window()

Hides or closes the prior report window if currently open.
"""
function close_prior_report_window()
    scr = active_prior_screen[]
    if scr !== nothing
        @async begin
            try
                lock(GLOBAL_OPENGL_LOCK) do
                    GLMakie.GLFW.IconifyWindow(scr.glscreen)
                end
            catch; end
        end
    end
end

# ── Dynamic Reactive Hook for Time Point Navigation ────────────────────────
# Automatically loads the report for the new time point whenever the user switches TP!
function _init_tp_switched_hook()
    if _MEH !== nothing && isdefined(_MEH, :tp_switched)
        on(_MEH.tp_switched) do _
            scr = active_prior_screen[]
            if scr !== nothing
                try
                    if isopen(scr)
                        new_tp = _MEH.current_tp_index[]
                        update_prior_report_tp!(new_tp; lang = current_prior_lang[])
                        # Update window title
                        lbl = haskey(_MEH.tp_labels, new_tp) ? _MEH.tp_labels[new_tp] : "TP $new_tp"
                        lock(GLOBAL_OPENGL_LOCK) do
                            GLMakie.GLFW.SetWindowTitle(scr.glscreen, "Radiological Report / Vorbefund — $lbl")
                        end
                    end
                catch; end
            end
        end
    end
end

try
    _init_tp_switched_hook()
catch; end

end # module PriorReportWindow
