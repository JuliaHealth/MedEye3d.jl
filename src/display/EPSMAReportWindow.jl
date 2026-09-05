module EPSMAReportWindow

using GLMakie
using Observables
using Dates
import ..SegmentationDisplay: synchronized_makie_renderloop, GLOBAL_OPENGL_LOCK
import ..SegmentationDisplay.MakieEventHandlers as _MEH
import ..EPSMAStructuredReport as ESR
import ..LLMDictation

export open_epsma_report_window, close_epsma_report_window, active_report_screen, active_report_fig

# Global reference to open report window / screen
const active_report_screen = Ref{Any}(nothing)
const active_report_fig = Ref{Any}(nothing)
const active_report_data = Ref{Any}(nothing)

"""
    _sync_edits_to_report!(report, lang, sec1, sec3, sec4p, sec4l, sec4b, sec4v, sec6)

Syncs user edits from GUI textbox Observables back into the mutable EPSMAReport
so that DOCX export and clipboard copy reflect the user's changes.
"""
function _sync_edits_to_report!(report::ESR.EPSMAReport, lang::String,
        sec1_text, sec3_text, sec4_prostate, sec4_lymph, sec4_bone, sec4_visceral, sec6_text)
    is_de = lang == "DE"
    if is_de
        report.history_text_de = sec1_text[]
        report.biodistribution_text_de = sec3_text[]
        report.findings_prostate_de = sec4_prostate[]
        report.findings_lymph_de = sec4_lymph[]
        report.findings_bone_de = sec4_bone[]
        report.findings_visceral_de = sec4_visceral[]
        report.conclusion_de = sec6_text[]
    else
        report.history_text_en = sec1_text[]
        report.biodistribution_text_en = sec3_text[]
        report.findings_prostate_en = sec4_prostate[]
        report.findings_lymph_en = sec4_lymph[]
        report.findings_bone_en = sec4_bone[]
        report.findings_visceral_en = sec4_visceral[]
        report.conclusion_en = sec6_text[]
    end
end

"""
    open_epsma_report_window(report::ESR.EPSMAReport; on_refresh::Union{Function,Nothing}=nothing)

Opens (or focuses/reopens) the dedicated E-PSMA Standardized Structured Report GLMakie window.
Adheres to: E-PSMA: The EANM standardized reporting guidelines v1.0 (Eur J Nucl Med Mol Imaging 2021).
"""
function open_epsma_report_window(report::ESR.EPSMAReport; on_refresh::Union{Function,Nothing}=nothing)
    active_report_data[] = report
    
    # Check if window already open; reuse old screen for INSTANT rebuilds
    local reuse_screen = false
    local screen = active_report_screen[]
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
    BG      = RGBf(0.08, 0.10, 0.13)
    BG_CARD = RGBf(0.12, 0.14, 0.18)
    BG_INSET= RGBf(0.16, 0.18, 0.23)
    HEADER_BG = RGBf(0.11, 0.21, 0.36) # Deep Navy #1B365D
    ACCENT  = RGBf(0.20, 0.60, 1.00)
    GOLD    = RGBf(0.95, 0.77, 0.20)
    TXT     = RGBf(0.95, 0.95, 0.95)
    SUBTXT  = RGBf(0.70, 0.75, 0.80)
    GRN     = RGBf(0.18, 0.65, 0.30)
    RED     = RGBf(0.85, 0.25, 0.25)
    ORANGE  = RGBf(0.90, 0.50, 0.15)
    BORDER  = RGBf(0.25, 0.28, 0.35)

    # Main Figure — 30% narrower for compact reading
    fig = Figure(size = (1820, 1020), backgroundcolor = BG, figure_padding = (24, 24, 16, 16))
    active_report_fig[] = fig

    # Top Toolbar Layout
    tb_layout = GridLayout(fig[1, 1], tellwidth = false)
    rowsize!(fig.layout, 1, Fixed(48))

    # Initial Language ("EN" or "DE") from report default
    init_lang = (hasproperty(report, :default_lang) && !isempty(report.default_lang)) ? report.default_lang : "EN"
    current_lang = Observable(init_lang)

    # Toolbar Controls — prominent DE / EN switch placed right on top
    Label(tb_layout[1, 1], "E-PSMA Standardized Structured Report (EANM v1.0 Guidelines)", fontsize = 15, font = :bold, color = ACCENT, halign = :left)
    
    Label(tb_layout[1, 2], "Language / Sprache:", fontsize = 11, font = :bold, color = SUBTXT, halign = :right)
    btn_lang_en = Button(tb_layout[1, 3], label = "[EN] English", buttoncolor = (init_lang == "EN" ? ACCENT : BG_CARD), labelcolor = TXT, fontsize = 12)
    btn_lang_de = Button(tb_layout[1, 4], label = "[DE] Deutsch", buttoncolor = (init_lang == "DE" ? ACCENT : BG_CARD), labelcolor = TXT, fontsize = 12)
    
    btn_export = Button(tb_layout[1, 5], label = "Export Word (.docx)", buttoncolor = GRN, labelcolor = TXT, fontsize = 11)
    btn_copy   = Button(tb_layout[1, 6], label = "Copy Text", buttoncolor = BG_CARD, labelcolor = TXT, fontsize = 11)
    btn_refresh= Button(tb_layout[1, 7], label = "Refresh", buttoncolor = BG_CARD, labelcolor = TXT, fontsize = 11)
    btn_close  = Button(tb_layout[1, 8], label = "← Return to Main Panel", buttoncolor = RGBf(0.4, 0.15, 0.15), labelcolor = TXT, fontsize = 11)

    # Status Bar
    status_text = Observable("Status: Ready (E-PSMA v1.0 Standardized Structured Report)")
    lbl_status = Label(fig[2, 1], status_text, fontsize = 10, color = SUBTXT, halign = :left)
    rowsize!(fig.layout, 2, Fixed(20))

    # Scrollable Main Body Layout — expanded width
    body_scroll = GridLayout(fig[3, 1])
    sl = Slider(body_scroll[1, 2], range = 0:0.005:1, startvalue = 1, horizontal = false, tellheight = false)

    g = GridLayout(body_scroll[1, 1], tellheight = false, halign = :left, valign = sl.value)
    rowgap!(g, 10)

    # CRITICAL: Enforce full-width expansion across all layout levels so text fields never shrinkwrap!
    colsize!(fig.layout, 1, Relative(1.0))
    colsize!(body_scroll, 1, Relative(0.99))
    colsize!(body_scroll, 2, Fixed(16))
    colsize!(g, 1, Relative(1.0))

    on(fig.scene.events.scroll) do scroll
        content_h = g.layoutobservables.computedbbox[].widths[2]
        window_h = size(fig.scene)[2]
        if content_h > window_h * 0.4
            sl.value[] = clamp(sl.value[] + scroll[2] * 0.04, 0.0, 1.0)
        else
            sl.value[] = 1.0
        end
        return Consume(true)
    end

    r = [0]
    nr!() = (r[1] += 1; r[1])

    # Clean text wrapping helper for wide display cards (> 2400px)
    function wrap_text(s::String; max_chars::Int = 220)::String
        lines = String[]
        for para in split(s, '\n')
            if length(para) <= max_chars
                push!(lines, String(para))
            else
                words = split(para, ' ')
                curr = ""
                for w in words
                    if isempty(curr)
                        curr = String(w)
                    elseif length(curr) + 1 + length(w) <= max_chars
                        curr *= " " * w
                    else
                        push!(lines, curr)
                        curr = String(w)
                    end
                end
                !isempty(curr) && push!(lines, curr)
            end
        end
        return join(lines, "\n")
    end

    # Observables for Dynamic Bilingual Translation
    hdr_patient_lbl = Observable("Patient ID: $(report.patient_id) | Exam Date: $(report.study_date) | Modality: $(report.modality)")
    hdr_mitnm_lbl   = Observable("Overall miTNM: $(report.final_mitnm)")
    
    sec1_title      = Observable("1. Patient History & Clinical Indication")
    sec1_text       = Observable(wrap_text(report.history_text_en))
    
    sec2_title      = Observable("2. Technical Information & Methodology (Synoptic Table 1)")
    
    sec3_title      = Observable("3. Physiological Background Reference Values & Visual Scale")
    sec3_text       = Observable(wrap_text(
        "• Physiological Tracer Biodistribution: $(report.biodistribution_text_en)\n" *
        "• Reference Background SUVmean Values:\n" *
        "    - Mediastinal Blood Pool SUVmean: $(round(get(report.background_suv, "blood", 1.5f0), digits=2))\n" *
        "    - Liver SUVmean: $(round(get(report.background_suv, "liver", 2.3f0), digits=2))\n" *
        "    - Parotid Gland SUVmean: $(round(get(report.background_suv, "parotid", 1.8f0), digits=2))\n\n" *
        "• E-PSMA Table 2 – 4-Point Visual Scale Criteria (PSMA Expression V):\n" *
        "    - Score 0: Below mediastinal blood pool\n" *
        "    - Score 1: Equal to or above blood pool and lower than liver\n" *
        "    - Score 2: Equal to or above liver and lower than parotid gland\n" *
        "    - Score 3: Equal to or above parotid gland"
    ))
    
    sec4_title      = Observable("4. Reporting of Findings by Anatomical Region (Delphi Consensus)")
    sec4_prostate   = Observable(wrap_text(">> Prostate / Prostate Bed (Local Tumor miT):\n$(report.findings_prostate_en)"))
    sec4_lymph      = Observable(wrap_text(">> Regional & Extra-Pelvic Lymph Nodes (miN / miM1a):\n$(report.findings_lymph_en)"))
    sec4_bone       = Observable(wrap_text(">> Osseous Disease / Skeleton (miM1b):\n$(report.findings_bone_en)"))
    sec4_visceral   = Observable(wrap_text(">> Non-Nodal Visceral Disease (miM1c) & Incidental Findings:\n$(report.findings_visceral_en)"))
    
    sec5_title      = Observable("5. Synoptic Table 2 – Reporting on PSMA PET/CT Findings (PCa Metastases)")
    sec5_t6_title   = Observable(report.default_lang == "DE" ? 
        "Tabelle 6 – Nebenbefunde & Technische Artefakte (Vom PCa-Staging/TMTV exkludiert)" : 
        "Table 6 – Incidental Findings & Technical Artifacts (Excluded from Staging / TMTV)")
    
    sec6_title      = Observable("6. Overall Conclusion & Clinical Impression")
    sec6_text       = Observable(wrap_text(report.conclusion_en))
    
    sec6_callout    = Observable(wrap_text(
        "Summary Molecular Imaging Classification: $(report.final_mitnm)\n" *
        "• Total Metabolic Tumor Volume (TMTV): $(round(report.tmtv_cc, digits=2)) cc\n" *
        "• Treatment Response Evaluation (PERCIST / RECIP): $(report.overall_recip)"
    ))

    sec7_title      = Observable("7. ✏️ Technical Parameters (User Input)")

    # Function to apply language switch
    function set_language!(lang::String)
        # Save current-language edits before switching (if textboxes are already created)
        old_lang = current_lang[]
        if old_lang != lang && @isdefined(sec1_text)
            _sync_edits_to_report!(report, old_lang, sec1_text, sec3_text, sec4_prostate, sec4_lymph, sec4_bone, sec4_visceral, sec6_text)
        end
        
        current_lang[] = lang
        is_de = lang == "DE"
        
        btn_lang_en.buttoncolor[] = is_de ? BG_CARD : ACCENT
        btn_lang_de.buttoncolor[] = is_de ? ACCENT : BG_CARD
        
        # Section 1 — mark as needing input if empty or placeholder
        history_placeholder = isempty(report.history_text_en) || occursin("Please enter", report.history_text_en)
        s1_suffix = history_placeholder ? "  ⚠️" : ""
        sec1_title[] = is_de ? "1. Klinische Angaben und Anamnese$s1_suffix" : "1. Patient History & Clinical Indication$s1_suffix"
        sec1_text[]  = wrap_text(is_de ? report.history_text_de : report.history_text_en)
        
        # Section 2 — mark as needing input if technical params are placeholder
        tech_placeholder = report.tech_params.radiotracer == "(not specified)"
        s2_suffix = tech_placeholder ? "  ⚠️" : ""
        sec2_title[] = is_de ? "2. Untersuchungstechnik (Synoptische Tabelle 1)$s2_suffix" : "2. Technical Information & Methodology (Synoptic Table 1)$s2_suffix"
        # Update Table 1 values from (possibly updated) tech_params
        if @isdefined(t1_obs)
            tp = report.tech_params
            t1_obs[1][] = tp.radiotracer
            t1_obs[2][] = tp.injected_activity
            t1_obs[3][] = tp.uptake_time
            t1_obs[4][] = tp.acquisition_type
            t1_obs[5][] = tp.ct_protocol
            t1_obs[6][] = tp.contrast
            t1_obs[7][] = tp.diuretic
        end
        
        # Section 3 — mark biodistribution if placeholder
        biodist_placeholder = occursin("Please enter", report.biodistribution_text_en) || occursin("Bitte", report.biodistribution_text_de)
        sec3_title[] = is_de ? "3. Physiologische Hintergrundaktivität & Visuelle Skala" : "3. Physiological Background Reference Values & Visual Scale"
        sec3_text[]  = wrap_text(is_de ?
            ("• Physiologische Tracer-Biodistribution: $(report.biodistribution_text_de)\n" *
             "• Referenzwerte Hintergrundaktivität (SUVmean):\n" *
             "    - Mediastinaler Blutpool SUVmean: $(round(get(report.background_suv, "blood", 1.5f0), digits=2))\n" *
             "    - Leber SUVmean: $(round(get(report.background_suv, "liver", 2.3f0), digits=2))\n" *
             "    - Glandula parotidea SUVmean: $(round(get(report.background_suv, "parotid", 1.8f0), digits=2))\n\n" *
             "• E-PSMA Tabelle 2 – 4-Punkte visuelle PSMA-Expressionsskala (PSMA Expression V):\n" *
             "    - Score 0: Unterhalb Blutpool-Niveau\n" *
             "    - Score 1: Gleich oder oberhalb Blutpool und unterhalb Leber-Niveau\n" *
             "    - Score 2: Gleich oder oberhalb Leber und unterhalb Parotis-Niveau\n" *
             "    - Score 3: Gleich oder oberhalb Parotis-Niveau") :
            ("• Physiological Tracer Biodistribution: $(report.biodistribution_text_en)\n" *
             "• Reference Background SUVmean Values:\n" *
             "    - Mediastinal Blood Pool SUVmean: $(round(get(report.background_suv, "blood", 1.5f0), digits=2))\n" *
             "    - Liver SUVmean: $(round(get(report.background_suv, "liver", 2.3f0), digits=2))\n" *
             "    - Parotid Gland SUVmean: $(round(get(report.background_suv, "parotid", 1.8f0), digits=2))\n\n" *
             "• E-PSMA Table 2 – 4-Point Visual Scale Criteria (PSMA Expression V):\n" *
             "    - Score 0: Below mediastinal blood pool\n" *
             "    - Score 1: Equal to or above blood pool and lower than liver\n" *
             "    - Score 2: Equal to or above liver and lower than parotid gland\n" *
             "    - Score 3: Equal to or above parotid gland"))
            
        sec4_title[] = is_de ? "4. Detaillierter Befund nach anatomischen Regionen (Delphi-Konsensus)" : "4. Reporting of Findings by Anatomical Region (Delphi Consensus)"
        sec4_prostate[] = wrap_text(is_de ? ">> Prostata / Prostatabett (Lokaler Tumor miT):\n$(report.findings_prostate_de)" : ">> Prostate / Prostate Bed (Local Tumor miT):\n$(report.findings_prostate_en)")
        sec4_lymph[]    = wrap_text(is_de ? ">> Regionale und extra-pelvine Lymphknoten (miN / miM1a):\n$(report.findings_lymph_de)" : ">> Regional & Extra-Pelvic Lymph Nodes (miN / miM1a):\n$(report.findings_lymph_en)")
        sec4_bone[]     = wrap_text(is_de ? ">> Skelettbefall / Knochenmetastasen (miM1b):\n$(report.findings_bone_de)" : ">> Osseous Disease / Skeleton (miM1b):\n$(report.findings_bone_en)")
        sec4_visceral[] = wrap_text(is_de ? ">> Viszerale Weichteilbefunde (miM1c) und Nebenbefunde:\n$(report.findings_visceral_de)" : ">> Non-Nodal Visceral Disease (miM1c) & Incidental Findings:\n$(report.findings_visceral_en)")
        
        sec5_title[] = is_de ? "5. Synoptische Tabelle 2 – Übersicht der PSMA-PET/CT Befunde (PCa-Metastasen)" : "5. Synoptic Table 2 – Reporting on PSMA PET/CT Findings (PCa Metastases)"
        sec5_t6_title[] = is_de ? "Tabelle 6 – Nebenbefunde & Technische Artefakte (Vom PCa-Staging/TMTV exkludiert)" : "Table 6 – Incidental Findings & Technical Artifacts (Excluded from Staging / TMTV)"
        
        sec6_title[] = is_de ? "6. Gesamtbeurteilung und klinische Empfehlung" : "6. Overall Conclusion & Clinical Impression"
        sec6_text[]  = wrap_text(is_de ? report.conclusion_de : report.conclusion_en)
        
        sec6_callout[] = wrap_text(is_de ?
            ("Zusammenfassendes molekulares Staging: $(report.final_mitnm)\n" *
             "• Gesamt-Tumorvolumen (TMTV): $(round(report.tmtv_cc, digits=2)) ml\n" *
             "• Verlauf / Therapieansprechen (PERCIST / RECIP): $(report.overall_recip)") :
            ("Summary Molecular Imaging Classification: $(report.final_mitnm)\n" *
             "• Total Metabolic Tumor Volume (TMTV): $(round(report.tmtv_cc, digits=2)) cc\n" *
             "• Treatment Response Evaluation (PERCIST / RECIP): $(report.overall_recip)"))
        
        # Section 7 title
        sec7_title[] = is_de ? "7. ✏️ Technische Parameter (Benutzereingabe)" : "7. ✏️ Technical Parameters (User Input)"
            
        status_text[] = is_de ? "Status: Sprache auf Deutsch gesetzt" : "Status: Language switched to English"
    end

    on(btn_lang_en.clicks) do _; set_language!("EN"); end
    on(btn_lang_de.clicks) do _; set_language!("DE"); end

    # Apply initial language state
    set_language!(init_lang)

    # ── Section Builder Helpers (Direct placement without nested shrinkwrap) ──
    function add_section_header!(title_obs::Observable{String})
        r_idx = nr!()
        Box(g[r_idx, 1], color = HEADER_BG, cornerradius = 4)
        Label(g[r_idx, 1], title_obs, fontsize = 13, font = :bold, color = TXT, halign = :left, padding = (20, 20, 8, 8))
        rowsize!(g, r_idx, Fixed(38))
    end

    function add_card_text!(text_obs::Observable{String})
        r_idx = nr!()
        Box(g[r_idx, 1], color = BG_CARD, cornerradius = 4, strokecolor = BORDER, strokewidth = 1)
        tb = Textbox(g[r_idx, 1],
            stored_string = text_obs[],
            placeholder = "(click to edit)",
            fontsize = 11,
            textcolor = TXT,
            textcolor_placeholder = SUBTXT,
            boxcolor = BG_INSET,
            boxcolor_focused = RGBf(0.20, 0.24, 0.32),
            boxcolor_hover = RGBf(0.17, 0.20, 0.27),
            bordercolor = BORDER,
            bordercolor_focused = ACCENT,
            cursorcolor = TXT,
            width = 1680,
            tellwidth = false)
        # Sync textbox → observable when user edits
        on(tb.stored_string) do val
            text_obs[] = val
        end
        # Sync observable → textbox when changed externally (e.g. language switch)
        on(text_obs) do val
            if tb.stored_string[] != val
                tb.stored_string[] = val
                tb.displayed_string[] = val
            end
        end
        return tb
    end

    # ── 0. Patient & Staging Top Banner ──────────────────────────────────────
    r_hdr = nr!()
    hdr_box = GridLayout(g[r_hdr, 1])
    Box(hdr_box[1, 1:2], color = BG_CARD, cornerradius = 6, strokecolor = BORDER, strokewidth = 1)
    Label(hdr_box[1, 1], hdr_patient_lbl, fontsize = 12, font = :bold, color = TXT, halign = :left, padding = (20, 0, 10, 10))
    Label(hdr_box[1, 2], hdr_mitnm_lbl, fontsize = 13, font = :bold, color = GOLD, halign = :right, padding = (0, 24, 10, 10))
    colsize!(hdr_box, 1, Relative(0.70))
    colsize!(hdr_box, 2, Relative(0.30))
    rowsize!(g, r_hdr, Fixed(46))

    # ── 1. Patient History ───────────────────────────────────────────────────
    add_section_header!(sec1_title)
    add_card_text!(sec1_text)

    # ── 2. Technical Parameters (Synoptic Table 1) ───────────────────────────
    add_section_header!(sec2_title)
    r_t1 = nr!()
    t1_grid = GridLayout(g[r_t1, 1])
    Box(t1_grid[1:2, 1:7], color = BG_CARD, cornerradius = 4, strokecolor = BORDER, strokewidth = 1)

    t1_headers = ["Radiotracer", "Injected Activity", "Uptake Time", "Acquisition", "CT Protocol", "Contrast (Oral/IV)", "Diuretic"]
    t1_obs = [Observable(report.tech_params.radiotracer),
              Observable(report.tech_params.injected_activity),
              Observable(report.tech_params.uptake_time),
              Observable(report.tech_params.acquisition_type),
              Observable(report.tech_params.ct_protocol),
              Observable(report.tech_params.contrast),
              Observable(report.tech_params.diuretic)]

    for c_i in 1:7
        Box(t1_grid[1, c_i], color = BG_INSET, cornerradius = 2)
        Label(t1_grid[1, c_i], t1_headers[c_i], fontsize = 10, font = :bold, color = ACCENT, halign = :center, padding = (6, 6, 6, 6))
        Label(t1_grid[2, c_i], t1_obs[c_i], fontsize = 10, color = TXT, halign = :center, padding = (6, 6, 6, 6))
    end
    for c_i in 1:7
        colsize!(t1_grid, c_i, Relative(1/7))
    end
    rowsize!(t1_grid, 1, Fixed(32))
    rowsize!(t1_grid, 2, Fixed(34))

    # ── 3. Physiological Background ──────────────────────────────────────────
    add_section_header!(sec3_title)
    add_card_text!(sec3_text)

    # ── 4. Detailed Findings by Region ───────────────────────────────────────
    add_section_header!(sec4_title)
    r_f_gl = nr!()
    findings_grid = GridLayout(g[r_f_gl, 1])
    rowgap!(findings_grid, 8)
    colsize!(findings_grid, 1, Relative(1.0))

    Box(findings_grid[1, 1], color = BG_CARD, cornerradius = 4, strokecolor = BORDER, strokewidth = 1)
    tb_prostate = Textbox(findings_grid[1, 1],
        stored_string = sec4_prostate[], placeholder = "(click to edit prostate findings)",
        fontsize = 11,
        textcolor = TXT, textcolor_placeholder = SUBTXT,
        boxcolor = BG_INSET, boxcolor_focused = RGBf(0.20, 0.24, 0.32), boxcolor_hover = RGBf(0.17, 0.20, 0.27),
        bordercolor = BORDER, bordercolor_focused = ACCENT, cursorcolor = TXT,
        width = 1680, tellwidth = false)
    on(tb_prostate.stored_string) do val; sec4_prostate[] = val; end

    Box(findings_grid[2, 1], color = BG_CARD, cornerradius = 4, strokecolor = BORDER, strokewidth = 1)
    tb_lymph = Textbox(findings_grid[2, 1],
        stored_string = sec4_lymph[], placeholder = "(click to edit lymph node findings)",
        fontsize = 11,
        textcolor = TXT, textcolor_placeholder = SUBTXT,
        boxcolor = BG_INSET, boxcolor_focused = RGBf(0.20, 0.24, 0.32), boxcolor_hover = RGBf(0.17, 0.20, 0.27),
        bordercolor = BORDER, bordercolor_focused = ACCENT, cursorcolor = TXT,
        width = 1680, tellwidth = false)
    on(tb_lymph.stored_string) do val; sec4_lymph[] = val; end

    Box(findings_grid[3, 1], color = BG_CARD, cornerradius = 4, strokecolor = BORDER, strokewidth = 1)
    tb_bone = Textbox(findings_grid[3, 1],
        stored_string = sec4_bone[], placeholder = "(click to edit bone findings)",
        fontsize = 11,
        textcolor = TXT, textcolor_placeholder = SUBTXT,
        boxcolor = BG_INSET, boxcolor_focused = RGBf(0.20, 0.24, 0.32), boxcolor_hover = RGBf(0.17, 0.20, 0.27),
        bordercolor = BORDER, bordercolor_focused = ACCENT, cursorcolor = TXT,
        width = 1680, tellwidth = false)
    on(tb_bone.stored_string) do val; sec4_bone[] = val; end

    Box(findings_grid[4, 1], color = BG_CARD, cornerradius = 4, strokecolor = BORDER, strokewidth = 1)
    tb_visceral = Textbox(findings_grid[4, 1],
        stored_string = sec4_visceral[], placeholder = "(click to edit visceral findings)",
        fontsize = 11,
        textcolor = TXT, textcolor_placeholder = SUBTXT,
        boxcolor = BG_INSET, boxcolor_focused = RGBf(0.20, 0.24, 0.32), boxcolor_hover = RGBf(0.17, 0.20, 0.27),
        bordercolor = BORDER, bordercolor_focused = ACCENT, cursorcolor = TXT,
        width = 1680, tellwidth = false)
    on(tb_visceral.stored_string) do val; sec4_visceral[] = val; end

    # Sync observables -> textboxes when language changes
    on(sec4_prostate) do v; if tb_prostate.stored_string[] != v; tb_prostate.stored_string[] = v; tb_prostate.displayed_string[] = v; end; end
    on(sec4_lymph)    do v; if tb_lymph.stored_string[] != v; tb_lymph.stored_string[] = v; tb_lymph.displayed_string[] = v; end; end
    on(sec4_bone)     do v; if tb_bone.stored_string[] != v; tb_bone.stored_string[] = v; tb_bone.displayed_string[] = v; end; end
    on(sec4_visceral) do v; if tb_visceral.stored_string[] != v; tb_visceral.stored_string[] = v; tb_visceral.displayed_string[] = v; end; end

    # ── 5. Synoptic Table 2 (Findings Overview) ──────────────────────────────
    add_section_header!(sec5_title)
    r_t2 = nr!()
    t2_grid = GridLayout(g[r_t2, 1])
    rowgap!(t2_grid, 3)

    t2_hdrs = ["Anatomical Location & CT Correlation", "miTNM", "Size / Volume", "#", "PSMA Expr. Q (SUVmax)", "PSMA Expr. V (0-3)", "Reader Confidence (1-5)"]
    num_cols = length(t2_hdrs)
    
    # Table 2 Header Row
    for c_i in 1:num_cols
        Box(t2_grid[1, c_i], color = HEADER_BG, cornerradius = 2)
        Label(t2_grid[1, c_i], t2_hdrs[c_i], fontsize = 10, font = :bold, color = TXT, halign = (c_i == 1 ? :left : :center), padding = (10, 10, 6, 6))
    end
    rowsize!(t2_grid, 1, Fixed(32))

    colsize!(t2_grid, 1, Relative(0.32))
    colsize!(t2_grid, 2, Relative(0.08))
    colsize!(t2_grid, 3, Relative(0.12))
    colsize!(t2_grid, 4, Relative(0.05))
    colsize!(t2_grid, 5, Relative(0.14))
    colsize!(t2_grid, 6, Relative(0.15))
    colsize!(t2_grid, 7, Relative(0.14))

    if isempty(report.synoptic_rows)
        Box(t2_grid[2, 1:num_cols], color = BG_CARD, cornerradius = 2)
        Label(t2_grid[2, 1:num_cols], "(No focal hypermetabolic lesions identified / Keine suspekten Befunde)", fontsize = 9, color = SUBTXT, halign = :center)
        rowsize!(t2_grid, 2, Fixed(28))
    else
        for (idx, row) in enumerate(report.synoptic_rows)
            r_row = idx + 1
            row_bg = idx % 2 == 1 ? BG_CARD : BG_INSET
            
            # Badge color for PSMA Expression V
            v_color = if row.psma_v_num == 3
                RED
            elseif row.psma_v_num == 2
                ORANGE
            elseif row.psma_v_num == 1
                ACCENT
            else
                SUBTXT
            end
            
            Box(t2_grid[r_row, 1:num_cols], color = row_bg, cornerradius = 2)
            
            # Col 1: Location + New Lesion badge
            loc_disp = row.is_new ? "* $(row.location) [NEW]" : row.location
            Label(t2_grid[r_row, 1], loc_disp, fontsize = 10, color = (row.is_new ? GOLD : TXT), halign = :left, padding = (10, 6, 6, 6))
            
            # Col 2: miTNM
            Label(t2_grid[r_row, 2], row.mitnm, fontsize = 10, font = :bold, color = GOLD, halign = :center)
            
            # Col 3: Size
            Label(t2_grid[r_row, 3], row.size_str, fontsize = 10, color = TXT, halign = :center)
            
            # Col 4: Count
            Label(t2_grid[r_row, 4], string(row.num_lesions), fontsize = 10, color = TXT, halign = :center)
            
            # Col 5: PSMA Expression Q
            Label(t2_grid[r_row, 5], row.psma_q, fontsize = 10, color = TXT, halign = :center)
            
            # Col 6: PSMA Expression V
            Label(t2_grid[r_row, 6], row.psma_v, fontsize = 10, font = :bold, color = v_color, halign = :center)
            
            # Col 7: Confidence
            Label(t2_grid[r_row, 7], "$(row.reader_confidence) / 5", fontsize = 10, font = :bold, color = GRN, halign = :center)
            
            rowsize!(t2_grid, r_row, Fixed(32))
        end
    end

    # ── Table 6: Incidental Findings & Technical Artifacts (Delphi consensus) ──
    if !isempty(report.artifact_rows)
        r_t6_hdr = nr!()
        Label(g[r_t6_hdr, 1], sec5_t6_title, fontsize = 12, font = :bold, color = SUBTXT, halign = :left, padding = (24, 6, 8, 4))

        r_t6 = nr!()
        t6_grid = GridLayout(g[r_t6, 1], 1, 6)
        colsize!(t6_grid, 1, Auto())
        colsize!(t6_grid, 2, Fixed(220))
        colsize!(t6_grid, 3, Fixed(130))
        colsize!(t6_grid, 4, Fixed(130))
        colsize!(t6_grid, 5, Fixed(140))
        colsize!(t6_grid, 6, Fixed(180))

        t6_headers = ["Anatomische Lokalisation / Location", "Vermutete Ätiologie / Aetiology", "Größe / Vol.", "PSMA Q (SUVmax)", "PSMA Expr. V", "Staging-Auswirkung / Impact"]
        Box(t6_grid[1, 1:6], color = BG_INSET, cornerradius = 2)
        for (c_idx, h) in enumerate(t6_headers)
            Label(t6_grid[1, c_idx], h, fontsize = 9, font = :bold, color = SUBTXT, halign = c_idx == 1 ? :left : :center, padding = (c_idx == 1 ? 10 : 0, 0, 4, 4))
        end
        rowsize!(t6_grid, 1, Fixed(26))

        for (idx, row) in enumerate(report.artifact_rows)
            r_row = idx + 1
            row_bg = idx % 2 == 1 ? BG_CARD : BG_INSET
            Box(t6_grid[r_row, 1:6], color = row_bg, cornerradius = 2)
            Label(t6_grid[r_row, 1], row.location, fontsize = 9, color = TXT, halign = :left, padding = (10, 4, 4, 4))
            Label(t6_grid[r_row, 2], row.comment, fontsize = 9, color = SUBTXT, halign = :center)
            Label(t6_grid[r_row, 3], row.size_str, fontsize = 9, color = SUBTXT, halign = :center)
            Label(t6_grid[r_row, 4], row.psma_q, fontsize = 9, color = SUBTXT, halign = :center)
            Label(t6_grid[r_row, 5], row.psma_v, fontsize = 9, color = SUBTXT, halign = :center)
            Label(t6_grid[r_row, 6], "Excluded / Exkludiert (miM0)", fontsize = 9, font = :bold, color = GOLD, halign = :center)
            rowsize!(t6_grid, r_row, Fixed(28))
        end
    end

    # ── 6. Conclusion & Recommendation ───────────────────────────────────────
    add_section_header!(sec6_title)
    add_card_text!(sec6_text)

    # Staging Callout Box (Direct placement without nested subgrid)
    r_callout = nr!()
    Box(g[r_callout, 1], color = BG_INSET, strokecolor = ACCENT, strokewidth = 1.5, cornerradius = 6)
    Label(g[r_callout, 1], sec6_callout, fontsize = 12, font = :bold, color = GOLD, halign = :left, justification = :left, padding = (24, 24, 16, 16))

    # ── 7. Technical Parameters (User Input) ────────────────────────────────
    add_section_header!(sec7_title)

    r_s7 = nr!()
    s7_grid = GridLayout(g[r_s7, 1])
    Box(s7_grid[1:8, 1:4], color = BG_CARD, cornerradius = 4, strokecolor = ACCENT, strokewidth = 1)

    # Helper for label + widget rows in Section 7
    s7_row = Ref(0)
    function s7_label!(text::String)
        s7_row[] += 1
        Label(s7_grid[s7_row[], 1], text, fontsize = 10, font = :bold, color = SUBTXT, halign = :right, padding = (10, 6, 4, 4))
    end

    # 1. Radiotracer
    s7_label!("Radiotracer:")
    tracer_opts = ["(not specified)", "[68Ga]Ga-PSMA-11", "[18F]DCFPyL", "[18F]PSMA-1007", "[68Ga]Ga-FAPI-46", "Other"]
    menu_tracer = Menu(s7_grid[s7_row[], 2], options = tracer_opts, default = report.tech_params.radiotracer in tracer_opts ? report.tech_params.radiotracer : "(not specified)", fontsize = 10, width = 250, tellwidth = false)

    # 2. Injected Activity
    s7_label!("Injected Activity:")
    tb_activity = Textbox(s7_grid[s7_row[], 2:4],
        placeholder = "e.g. 185 MBq (2.5 MBq/kg)",
        stored_string = report.tech_params.injected_activity == "(not specified)" ? "" : report.tech_params.injected_activity,
        fontsize = 11,
        textcolor = TXT, textcolor_placeholder = SUBTXT,
        boxcolor = BG_INSET, boxcolor_focused = RGBf(0.20, 0.24, 0.32), boxcolor_hover = RGBf(0.17, 0.20, 0.27),
        bordercolor = BORDER, bordercolor_focused = ACCENT, cursorcolor = TXT,
        width = 750, tellwidth = false)

    # 3. Uptake Time
    s7_label!("Uptake Time:")
    tb_uptake = Textbox(s7_grid[s7_row[], 2:4],
        placeholder = "e.g. 60 min",
        stored_string = report.tech_params.uptake_time == "(not specified)" ? "" : report.tech_params.uptake_time,
        fontsize = 11,
        textcolor = TXT, textcolor_placeholder = SUBTXT,
        boxcolor = BG_INSET, boxcolor_focused = RGBf(0.20, 0.24, 0.32), boxcolor_hover = RGBf(0.17, 0.20, 0.27),
        bordercolor = BORDER, bordercolor_focused = ACCENT, cursorcolor = TXT,
        width = 750, tellwidth = false)

    # 4. Acquisition Type
    s7_label!("Acquisition Type:")
    acq_opts = ["(not specified)", "Standard (Vertex to mid-thigh)", "Whole Body (Vertex to toes)", "Limited Field of View"]
    menu_acq = Menu(s7_grid[s7_row[], 2], options = acq_opts, default = report.tech_params.acquisition_type in acq_opts ? report.tech_params.acquisition_type : "(not specified)", fontsize = 10, width = 300, tellwidth = false)

    # 5. CT Protocol
    s7_label!("CT Protocol:")
    ct_opts = ["(not specified)", "Low-Dose (non-diagnostic)", "Diagnostic (full-dose)", "Full-Dose with Contrast"]
    menu_ct = Menu(s7_grid[s7_row[], 2], options = ct_opts, default = report.tech_params.ct_protocol in ct_opts ? report.tech_params.ct_protocol : "(not specified)", fontsize = 10, width = 300, tellwidth = false)

    # 6. Contrast Agent
    s7_label!("Contrast Agent:")
    contrast_opts = ["(not specified)", "No / Nein", "Oral", "IV", "Oral + IV"]
    menu_contrast = Menu(s7_grid[s7_row[], 2], options = contrast_opts, default = report.tech_params.contrast in contrast_opts ? report.tech_params.contrast : "(not specified)", fontsize = 10, width = 250, tellwidth = false)

    # 7. Diuretic
    s7_label!("Furosemide / Diuretic:")
    diuretic_opts = ["(not specified)", "No / Nein", "Yes (20 mg Furosemide)", "Yes (40 mg Furosemide)"]
    menu_diuretic = Menu(s7_grid[s7_row[], 2], options = diuretic_opts, default = report.tech_params.diuretic in diuretic_opts ? report.tech_params.diuretic : "(not specified)", fontsize = 10, width = 300, tellwidth = false)

    # Apply button
    s7_row[] += 1
    btn_apply_s7 = Button(s7_grid[s7_row[], 2], label = "✏️ Apply & Update Report ↻", fontsize = 11, width = 280, tellwidth = false)

    colsize!(s7_grid, 1, Fixed(200))
    colsize!(s7_grid, 2, Relative(0.35))
    colsize!(s7_grid, 3, Relative(0.20))
    colsize!(s7_grid, 4, Relative(0.15))
    for ri in 1:s7_row[]
        rowsize!(s7_grid, ri, Fixed(36))
    end

    # Section 7 Apply handler — push tech param values into report and re-render
    on(btn_apply_s7.clicks) do _
        lang = current_lang[]
        is_de = lang == "DE"

        # Update tech params (struct is immutable, so build a new one)
        tracer_val = menu_tracer.selection[]
        activity_val = let v = tb_activity.stored_string[]; isempty(v) ? "(not specified)" : v; end
        uptake_val = let v = tb_uptake.stored_string[]; isempty(v) ? "(not specified)" : v; end
        acq_val = menu_acq.selection[]
        ct_val = menu_ct.selection[]
        contrast_val = menu_contrast.selection[]
        diuretic_val = menu_diuretic.selection[]

        new_tp = ESR.EPSMATechnicalParams(
            radiotracer = tracer_val,
            injected_activity = activity_val,
            uptake_time = uptake_val,
            acquisition_type = acq_val,
            ct_protocol = ct_val,
            contrast = contrast_val,
            diuretic = diuretic_val
        )
        report.tech_params = new_tp

        # Auto-generate technical narrative from tech params
        report.tech_narrative_en = "The patient was given $(activity_val) $(tracer_val) intravenously. " *
            "Imaging was obtained after $(uptake_val) with $(ct_val) CT for attenuation correction and anatomical correlation."
        report.tech_narrative_de = "Dem Patienten wurden $(activity_val) $(tracer_val) intravenös verabreicht. " *
            "Die Bildgebung erfolgte nach $(uptake_val) mit $(ct_val) CT zur Schwächungskorrektur und anatomischen Korrelation."

        # Re-render sections with updated data
        set_language!(lang)
        status_text[] = is_de ? "Status: Technische Parameter übernommen ✓" : "Status: Technical parameters applied ✓"
    end

    # ── Button Handlers ──────────────────────────────────────────────────────
    on(btn_export.clicks) do _
        lang = current_lang[]
        reports_dir = joinpath(@__DIR__, "..", "..", "data", "reports")
        isdir(reports_dir) || mkpath(reports_dir)
        filename = "E_PSMA_Report_$(report.patient_id)_TP$(report.tp_index)_$(lang).docx"
        out_path = joinpath(reports_dir, filename)
        
        status_text[] = "[...] Generating official Word report ($lang)..."
        try
            # Sync user edits from GUI textboxes → report object before export
            _sync_edits_to_report!(report, lang, sec1_text, sec3_text, sec4_prostate, sec4_lymph, sec4_bone, sec4_visceral, sec6_text)
            ESR.export_to_docx(report, out_path; lang = lang)
            status_text[] = "[OK] Word report successfully exported: $out_path"
            @info "Exported E-PSMA Word report to $out_path"
        catch e
            status_text[] = "[ERR] Export failed: $e"
            @error "Failed to export Word document" exception=e
        end
    end

    on(btn_copy.clicks) do _
        lang = current_lang[]
        is_de = lang == "DE"
        
        # Sync user edits from GUI textboxes → report object before copy
        _sync_edits_to_report!(report, lang, sec1_text, sec3_text, sec4_prostate, sec4_lymph, sec4_bone, sec4_visceral, sec6_text)
        
        io = IOBuffer()
        println(io, is_de ? "=== E-PSMA STRUKTURIERTER BEFUNDBERICHT (EANM v1.0) ===" : "=== E-PSMA STANDARDIZED STRUCTURED REPORT (EANM v1.0) ===")
        println(io, hdr_patient_lbl[])
        println(io, hdr_mitnm_lbl[])
        println(io, "")
        println(io, sec1_title[])
        println(io, sec1_text[])
        println(io, "")
        println(io, sec3_title[])
        println(io, sec3_text[])
        println(io, "")
        println(io, sec4_title[])
        println(io, sec4_prostate[])
        println(io, sec4_lymph[])
        println(io, sec4_bone[])
        println(io, sec4_visceral[])
        println(io, "")
        println(io, sec5_title[])
        for r in report.synoptic_rows
            println(io, "• $(r.location) | $(r.mitnm) | $(r.size_str) | $(r.psma_q) | $(r.psma_v) | Conf: $(r.reader_confidence)/5")
        end
        if !isempty(report.artifact_rows)
            println(io, "")
            println(io, is_de ? "Tabelle 6 – Nebenbefunde & Technische Artefakte (Exkludiert):" : "Table 6 – Incidental Findings & Technical Artifacts (Excluded):")
            for r in report.artifact_rows
                println(io, "• $(r.location) | $(r.comment) | $(r.size_str) | $(r.psma_q) | Excluded (miM0)")
            end
        end
        println(io, "")
        println(io, sec6_title[])
        println(io, sec6_text[])
        println(io, "")
        println(io, sec6_callout[])
        
        report_str = String(take!(io))
        LLMDictation.copy_to_clipboard(report_str)
        status_text[] = "[OK] Full structured report copied to clipboard!"
    end

    on(btn_refresh.clicks) do _
        status_text[] = "[...] Re-aggregating clinical metadata & metrics..."
        try
            ESR.invalidate_report!(report.tp_index)
            new_report = ESR.get_or_build_report(report.tp_index)
            active_report_data[] = new_report
            set_language!(current_lang[])
            status_text[] = "[OK] Report refreshed with latest metadata!"
            if on_refresh !== nothing
                on_refresh()
            end
        catch e
            status_text[] = "[ERR] Refresh failed: $e"
        end
    end

    on(btn_close.clicks) do _
        close_epsma_report_window()
    end

    if reuse_screen
        lock(GLOBAL_OPENGL_LOCK) do
            empty!(screen)
            display(screen, fig)
            GLMakie.GLFW.ShowWindow(screen.glscreen)
        end
    else
        screen = lock(GLOBAL_OPENGL_LOCK) do
            s = GLMakie.Screen(fig.scene; renderloop = synchronized_makie_renderloop)
            display(s, fig)
            s
        end
    end
    
    active_report_screen[] = screen
    active_report_fig[] = fig
    return screen
end

"""
    close_epsma_report_window()

Closes the E-PSMA report window if open.
"""
function close_epsma_report_window()
    if active_report_screen[] !== nothing
        try
            screen = active_report_screen[]
            if isopen(screen)
                # Don't use GLOBAL_OPENGL_LOCK here — this callback fires
                # from inside the render loop which already holds the lock.
                GLMakie.GLFW.SetWindowShouldClose(screen.glscreen, true)
                active_report_screen[] = nothing
                @info "[E-PSMA] Report window closed (← Return to Main Panel)"
            end
        catch e
            @warn "[E-PSMA] Close failed" exception=e
            try
                active_report_screen[] = nothing
            catch; end
        end
    end
end

end # module EPSMAReportWindow
