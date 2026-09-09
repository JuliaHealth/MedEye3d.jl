#!/usr/bin/env python3
"""
generate_epsma_docx.py
Generates professional Word (.docx) structured reports adhering to:
E-PSMA: The EANM standardized reporting guidelines v1.0 for PSMA PET (Eur J Nucl Med Mol Imaging 2021 48:1626–1638).

Supports both English (EN) and German (DE) versions.
"""

import sys
import json
import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, hex_color):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tc_pr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tc_mar.append(node)
    tc_pr.append(tc_mar)

def set_table_borders(table, color="CCCCCC", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'  <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def build_epsma_docx(data, output_path, lang="EN"):
    is_de = lang.upper() == "DE"
    doc = Document()

    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    PRIMARY = RGBColor(27, 54, 93)     # Deep Navy
    MUTED = RGBColor(108, 117, 125)    # Gray
    DARK_TEXT = RGBColor(33, 37, 41)   # Near black

    # Title & Header
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(2)
    run_title = title_p.add_run(
        "E-PSMA Strukturierter Befundbericht" if is_de else "E-PSMA Standardized Structured Report"
    )
    run_title.font.size = Pt(18)
    run_title.font.bold = True
    run_title.font.color.rgb = PRIMARY

    subtitle_p = doc.add_paragraph()
    subtitle_p.paragraph_format.space_after = Pt(14)
    run_subtitle = subtitle_p.add_run(
        "Gemäß EANM Standardized Reporting Guidelines v1.0 für PSMA-PET/CT (Eur J Nucl Med Mol Imaging 2021 48:1626–1638)" if is_de else
        "According to EANM Standardized Reporting Guidelines v1.0 for PSMA-PET/CT (Eur J Nucl Med Mol Imaging 2021 48:1626–1638)"
    )
    run_subtitle.font.size = Pt(9.5)
    run_subtitle.font.italic = True
    run_subtitle.font.color.rgb = MUTED

    patient_id = data.get("patient_id", "Unknown")
    exam_date = data.get("study_date", data.get("tp_label", "Current"))
    modality = data.get("modality", "PET/CT")
    mitnm = data.get("final_mitnm", "miT0 miN0 miM0")

    # 1. Patient History & Indication
    h1 = doc.add_paragraph()
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(4)
    r_h1 = h1.add_run("Klinische Angaben und Anamnese" if is_de else "Patient History")
    r_h1.font.size = Pt(12)
    r_h1.font.bold = True
    r_h1.font.color.rgb = PRIMARY

    hist_txt = data.get("history_text_de" if is_de else "history_text_en", "")
    if not hist_txt:
        hist_txt = data.get("history_text_en", "(No prior clinical indication provided)")
    
    if is_de:
        hist_txt = hist_txt + "\n\n[@]Verlauf:\naktuell\t\t[@]biochem. Rezidiv (aPSA [@] ng/ml); Re-Staging"
    else:
        hist_txt = hist_txt + "\n\n[@]Course:\ncurrent\t\t[@]biochem. recurrence (aPSA [@] ng/ml); Re-Staging"

    p_hist = doc.add_paragraph(hist_txt)
    p_hist.paragraph_format.space_after = Pt(10)

    # 1.5 Consent
    h_consent = doc.add_paragraph()
    r_consent = h_consent.add_run("Aufklärung und Einwilligung:" if is_de else "Informed Consent:")
    r_consent.font.bold = True
    p_consent = doc.add_paragraph(
        ("Nach Erhebung der Anamnese und Indikationsstellung wurde der Patient über den Zweck "
         "sowie die Art und Weise der PSMA-PET/CT-Untersuchung aufgeklärt. Spezielle und individuelle Fragen "
         "des Patienten sind auf dem Aufklärungsbogen dokumentiert. Der Patient wurde auf spezielle "
         "Verhaltensmaßregeln vor, während und nach der Untersuchung hingewiesen. Der Nuklearmediziner hat die "
         "rechtfertigende Indikation geprüft.") if is_de else
        "After obtaining the patient's medical history and establishing the indication, the patient was informed about the purpose "
        "and nature of the PSMA-PET/CT examination [...]."
    )
    p_consent.paragraph_format.space_after = Pt(10)

    # 2. Technical information
    h2 = doc.add_paragraph()
    h2.paragraph_format.space_before = Pt(10)
    h2.paragraph_format.space_after = Pt(4)
    r_h2 = h2.add_run("Untersuchungstechnik:" if is_de else "Technical information:")
    r_h2.font.bold = True

    tech_narr = data.get("tech_narrative_de" if is_de else "tech_narrative_en", "")
    if not tech_narr:
        tech = data.get("tech_params", {})
        inj_act = tech.get("injected_activity", "155 MBq")
        tracer = tech.get("radiotracer", "[68Ga]Ga-PSMA-11")
        up_time = tech.get("uptake_time", "60 minutes")
        ct_prot = tech.get("ct_protocol", "diagnostic")
        if is_de:
            tech_narr = f"[@]Gabe von 20 ml Accupaque zur Kontrastierung der Ureteren (keine diagnostische KM-CT). Topogramm zur Untersuchungsplanung ([@]Scheitel/Schädelbasis bis oberes Femurdrittel). Nach vorangegangener Applikation von {inj_act} {tracer} erfolgte die PET-Emissionsmessung bei insgesamt komplikationslosem Untersuchungsablauf (Uptakephase {up_time})."
        else:
            tech_narr = f"[@]Administration of 20 ml Accupaque for ureter contrast. Topogram for planning. Following administration of {inj_act} {tracer}, PET acquisition was performed without complications (uptake phase {up_time})."
    
    p_tech = doc.add_paragraph(tech_narr)
    p_tech.paragraph_format.space_after = Pt(10)

    # 3. Reporting of Findings
    h4 = doc.add_paragraph()
    h4.paragraph_format.space_before = Pt(10)
    h4.paragraph_format.space_after = Pt(4)
    r_h4 = h4.add_run("Befund:" if is_de else "Findings:")
    r_h4.font.underline = True
    r_h4.font.color.rgb = DARK_TEXT
    
    r_comp = h4.add_run("\nKeine Voruntersuchungen zum Vergleich vorliegend." if is_de else "\nNo prior examinations available for comparison.")
    r_comp.font.color.rgb = DARK_TEXT
    h4.paragraph_format.space_after = Pt(10)

    bg = data.get("background_suv", {})
    ref_title = "Referenzwerte Hintergrundaktivität (SUVmean):" if is_de else "Reference background activity (SUVmean):"
    p_ref = doc.add_paragraph(ref_title)
    r_bl = p_ref.add_run(f"\n    - Mediastinaler Blutpool SUVmean: {bg.get('blood', '2.12')}")
    r_lv = p_ref.add_run(f"\n    - Leber SUVmean: {bg.get('liver', '8.64')}")
    r_pr = p_ref.add_run(f"\n    - Glandula parotidea SUVmean: {bg.get('parotid', '0.0')}")
    p_ref.paragraph_format.space_after = Pt(10)

    # Compile text for Abdomen/Becken
    abdomen_txt = []
    for k in ["findings_prostate", "findings_lymph", "findings_visceral"]:
        txt = data.get(f"{k}_de" if is_de else f"{k}_en", "")
        if txt:
            clean_lines = [line.lstrip('•>>-').strip() for line in txt.split('\n')]
            clean = "\n".join([line for line in clean_lines if line])
            abdomen_txt.append(clean)
    abdomen_txt = "\n\n".join(abdomen_txt)

    regions = [
        ("Kopf/Hals (excl. Knochen):" if is_de else "Head/Neck (excl. bone):", data.get("findings_head_neck_de" if is_de else "findings_head_neck_en", "")),
        ("Thorax (excl. Knochen):" if is_de else "Thorax (excl. bone):", data.get("findings_thorax_de" if is_de else "findings_thorax_en", "")),
        ("Abdomen/Becken (excl. Knochen):" if is_de else "Abdomen/Pelvis (excl. bone):", abdomen_txt),
        ("Ossärer Status/Weichteilmantel:" if is_de else "Osseous Status/Soft Tissue:", data.get("findings_bone_de" if is_de else "findings_bone_en", ""))
    ]

    for reg_title, reg_txt in regions:
        if not reg_txt:
            continue
        reg_txt_clean = "\n".join([line.lstrip('•>>-').strip() for line in reg_txt.split('\n') if line.strip()])
        p_reg = doc.add_paragraph()
        p_reg.paragraph_format.space_before = Pt(4)
        p_reg.paragraph_format.space_after = Pt(4)
        
        r_hdr = p_reg.add_run(reg_title)
        r_hdr.font.underline = True
        
        r_body = p_reg.add_run(f"\n{reg_txt_clean}")
        r_body.font.color.rgb = DARK_TEXT

    artifacts_txt = data.get("findings_artifacts_de" if is_de else "findings_artifacts_en", "")
    if artifacts_txt:
        artifacts_txt_clean = "\n".join([line.lstrip("•>>-").strip() for line in artifacts_txt.split("\n") if line.strip()])
        if artifacts_txt_clean:
            p_art = doc.add_paragraph(artifacts_txt_clean)
            p_art.paragraph_format.space_before = Pt(4)
            p_art.paragraph_format.space_after = Pt(4)
            for r in p_art.runs:
                r.font.color.rgb = DARK_TEXT

    # 4. Conclusion
    h6 = doc.add_paragraph()
    h6.paragraph_format.space_before = Pt(14)
    h6.paragraph_format.space_after = Pt(4)
    r_h6 = h6.add_run("Beurteilung:" if is_de else "Conclusion:")
    r_h6.font.underline = True
    r_h6.font.color.rgb = DARK_TEXT

    concl_txt = data.get("conclusion_de" if is_de else "conclusion_en", "")
    if concl_txt:
        for line in concl_txt.split("\n"):
            p_concl = doc.add_paragraph(f" {line}")
            p_concl.paragraph_format.space_after = Pt(2)

    # Final Staging
    p_stage = doc.add_paragraph()
    r_s1 = p_stage.add_run(f"Molekulares Tumorstadium gemäß ePROMISE v2.2 / miTNM: {mitnm}." if is_de else f"Molecular tumor stage according to ePROMISE v2.2 / miTNM: {mitnm}.")
    
    tmtv = data.get("tmtv_cc", 0.0)
    recip = data.get("overall_recip", "BASELINE")
    
    r_s2 = p_stage.add_run(f"\nZusammenfassend PSMA-positive Erkrankung (TMTV: {tmtv:.2f} ml). Vorstellung in der interdisziplinären Tumorkonferenz empfohlen." if is_de else f"\nIn summary, PSMA-positive disease (TMTV: {tmtv:.2f} ml). Presentation at the interdisciplinary tumor board recommended.")
    p_stage.paragraph_format.space_after = Pt(10)

    # Synoptic Tables Header
    h7 = doc.add_paragraph()
    h7.paragraph_format.space_before = Pt(14)
    h7.paragraph_format.space_after = Pt(4)
    r_h7 = h7.add_run("Synoptische Tabellen (" if is_de else "Synoptic Tables (")
    r_h7.font.size = Pt(14)
    r_h7.font.bold = True
    r_h7.font.color.rgb = PRIMARY
    r_h7_2 = h7.add_run("v1.0 der EANM für PSMA-PET/CT)" if is_de else "EANM v1.0 for PSMA-PET/CT)")

    p_st1 = doc.add_paragraph()
    r_st1 = p_st1.add_run("[Untersuchungstechnik]" if is_de else "[Technical information]")
    r_st1.font.bold = True

    p_st2 = doc.add_paragraph()
    r_st2 = p_st2.add_run("[Befunde]" if is_de else "[Findings]")
    r_st2.font.bold = True
    
    p_st3 = doc.add_paragraph()
    r_st3 = p_st3.add_run("[Tabelle 6 — Nebenbefunde]" if is_de else "[Table 6 — Incidental Findings]")
    r_st3.font.bold = True

    p_foot2 = doc.add_paragraph(
        "E-PSMA Tabelle 2 – 4-Punkte visuelle PSMA-Expressionsskala (PSMA Expression V):\n"
        "    - Score 0: Unterhalb Blutpool-Niveau\n"
        "    - Score 1: Gleich oder oberhalb Blutpool und unterhalb Leber-Niveau\n"
        "    - Score 2: Gleich oder oberhalb Leber und unterhalb Parotis-Niveau\n"
        "    - Score 3: Gleich oder oberhalb Parotis-Niveau" if is_de else 
        "E-PSMA Table 2 - 4-point visual PSMA expression scale:\n"
        "    - Score 0: Below blood pool\n"
        "    - Score 1: Equal or above blood pool and below liver\n"
        "    - Score 2: Equal or above liver and below parotid\n"
        "    - Score 3: Equal or above parotid"
    )

    p_foot = doc.add_paragraph()
    p_foot.paragraph_format.space_before = Pt(18)
    r_ft = p_foot.add_run(
        "Referenzen: (1) Ceci F, et al. E-PSMA: the EANM standardized reporting guidelines v1.0 for PSMA-PET. Eur J Nucl Med Mol Imaging 2021; 48:1626–1638. "
        "(2) Eiber M, et al. Prostate cancer molecular imaging standardized evaluation (PROMISE): proposed miTNM classification. J Nucl Med 2018; 59:469–478."
    )
    r_ft.font.size = Pt(7.5)
    r_ft.font.italic = True
    r_ft.font.color.rgb = MUTED

    def add_table_header(table, headers, bg_color="1B365D", text_color=RGBColor(255, 255, 255)):
        row = table.rows[0]
        for idx, text in enumerate(headers):
            cell = row.cells[idx]
            cell.text = text
            set_cell_background(cell, bg_color)
            set_cell_margins(cell)
            for r in cell.paragraphs[0].runs:
                r.font.bold = True
                r.font.size = Pt(9.5)
                r.font.color.rgb = text_color
                
    def add_table_row(table, row_data):
        row = table.add_row()
        for idx, text in enumerate(row_data):
            cell = row.cells[idx]
            cell.text = str(text)
            set_cell_margins(cell)
            for r in cell.paragraphs[0].runs:
                r.font.size = Pt(8.5)
                r.font.color.rgb = DARK_TEXT

    doc.add_paragraph()
    
    t0 = doc.add_table(rows=1, cols=4)
    add_table_header(t0, ["Patient ID", "Untersuchungsdatum" if is_de else "Study Date", "Modalität" if is_de else "Modality", "Gesamt miTNM" if is_de else "Overall miTNM"], bg_color="E9ECEF", text_color=PRIMARY)
    add_table_row(t0, [patient_id, exam_date, modality, mitnm])
    set_table_borders(t0)
    
    doc.add_paragraph()
    
    t1 = doc.add_table(rows=1, cols=7)
    add_table_header(t1, ["Radiopharmakon", "Aktivität", "Uptake-Zeit", "Akquisitionsbereich", "CT-Protokoll", "Kontrastmittel", "Diuretikum"] if is_de else ["Radiopharmaceutical", "Activity", "Uptake Time", "Acquisition Field", "CT Protocol", "Contrast", "Diuretic"])
    tech = data.get("tech_params", {})
    add_table_row(t1, [
        tech.get("radiotracer", ""), tech.get("injected_activity", ""), tech.get("uptake_time", ""),
        tech.get("acquisition_type", ""), tech.get("ct_protocol", ""), tech.get("contrast", ""), tech.get("diuretic", "")
    ])
    set_table_borders(t1)
    
    doc.add_paragraph()
    
    synoptic_rows = data.get("synoptic_rows", [])
    if synoptic_rows:
        t2 = doc.add_table(rows=1, cols=7)
        add_table_header(t2, ["Anatomische Lokalisation", "miTNM", "Größe / Vol.", "Anzahl", "PSMA Expr. Q (SUVmax)", "PSMA Expr. V", "Konfidenz (1-5)"] if is_de else ["Anatomical Location", "miTNM", "Size / Vol.", "Count", "PSMA Expr. Q (SUVmax)", "PSMA Expr. V", "Confidence (1-5)"])
        for r in synoptic_rows:
            add_table_row(t2, [
                r.get("location", ""), r.get("mitnm", ""), r.get("size_str", ""), str(r.get("num_lesions", 1)),
                r.get("psma_q", ""), r.get("psma_v", ""), str(r.get("reader_confidence", 5))
            ])
        set_table_borders(t2)
    
    doc.add_paragraph()
    
    artifact_rows = data.get("artifact_rows", [])
    if artifact_rows:
        t3 = doc.add_table(rows=1, cols=6)
        add_table_header(t3, ["Anatomische Lokalisation", "Vermutete Ätiologie", "Größe / Vol.", "PSMA Expr. Q (SUVmax)", "PSMA Expr. V", "Staging-Auswirkung"] if is_de else ["Anatomical Location", "Suspected Etiology", "Size / Vol.", "PSMA Expr. Q (SUVmax)", "PSMA Expr. V", "Staging Impact"], bg_color="5C6F84")
        for r in artifact_rows:
            add_table_row(t3, [
                r.get("location", ""), r.get("comment", ""), r.get("size_str", ""),
                r.get("psma_q", ""), r.get("psma_v", ""), "Exkludiert (miM0)" if is_de else "Excluded (miM0)"
            ])
        set_table_borders(t3)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc.save(output_path)
    print(f"Successfully generated E-PSMA report: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 generate_epsma_docx.py <input_json> <output_docx> [EN|DE]")
        sys.exit(1)
    json_path = sys.argv[1]
    out_docx = sys.argv[2]
    lang = sys.argv[3] if len(sys.argv) > 3 else "EN"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    build_epsma_docx(data, out_docx, lang=lang)
