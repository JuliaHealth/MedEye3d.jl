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
    """Sets background shading of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tc_pr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets inner margins (padding) of a table cell in dxa (1/20 pt)."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tc_mar.append(node)
    tc_pr.append(tc_mar)

def set_table_borders(table, color="CCCCCC", sz="4", val="single"):
    """Sets clean subtle borders on a table."""
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

    # Configure Margins: 0.75 in (1.9 cm)
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Color Palette
    PRIMARY = RGBColor(27, 54, 93)     # #1B365D Deep Navy
    SECONDARY = RGBColor(70, 130, 180) # Steel Blue
    DARK_TEXT = RGBColor(33, 37, 41)   # Near black
    MUTED = RGBColor(108, 117, 125)    # Gray
    HEADER_BG = "1B365D"               # Navy table header
    ALT_ROW_BG = "F8F9FA"              # Soft light gray zebra

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



    # Patient data (used in staging box later)
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
    p_hist = doc.add_paragraph(hist_txt)
    p_hist.paragraph_format.space_after = Pt(10)
    p_hist.style.font.size = Pt(10)

    # 2. Technical information
    h2 = doc.add_paragraph()
    h2.paragraph_format.space_before = Pt(10)
    h2.paragraph_format.space_after = Pt(4)
    r_h2 = h2.add_run("Untersuchungstechnik" if is_de else "Technical information")
    r_h2.font.size = Pt(12)
    r_h2.font.bold = True
    r_h2.font.color.rgb = PRIMARY

    tech_narr = data.get("tech_narrative_de" if is_de else "tech_narrative_en", "")
    if not tech_narr:
        tech = data.get("tech_params", {})
        inj_act = tech.get("injected_activity", "155 MBq")
        tracer = tech.get("radiotracer", "[68Ga]Ga-PSMA-11")
        up_time = tech.get("uptake_time", "60 minutes")
        ct_prot = tech.get("ct_protocol", "diagnostic")
        if is_de:
            tech_narr = f"Dem Patienten wurden {inj_act} {tracer} intravenös verabreicht. Die Bildgebung erfolgte nach {up_time} mit einer {ct_prot} CT zur Schwächungskorrektur und anatomischen Korrelation."
        else:
            tech_narr = f"The patient was given {inj_act} {tracer} intravenously. Imaging was obtained after {up_time} with {ct_prot} CT for attenuation correction and anatomical correlation."
    
    p_tech = doc.add_paragraph(tech_narr)
    p_tech.paragraph_format.space_after = Pt(10)
    p_tech.style.font.size = Pt(10)

    # 3. Reporting of Findings
    h4 = doc.add_paragraph()
    h4.paragraph_format.space_before = Pt(10)
    h4.paragraph_format.space_after = Pt(4)
    r_h4 = h4.add_run("Befundung" if is_de else "Reporting of Findings")
    r_h4.font.size = Pt(12)
    r_h4.font.bold = True
    r_h4.font.color.rgb = PRIMARY



    bio_txt = data.get("biodistribution_text_de" if is_de else "biodistribution_text_en",
                       "Die physiologische Biodistribution des Tracers war regulär." if is_de else
                       "The physiological biodistribution of the radiotracer was regular.")
    p_bio = doc.add_paragraph(bio_txt)
    p_bio.paragraph_format.space_after = Pt(10)
    p_bio.style.font.size = Pt(10)

    regions = [
        ("Prostata" if is_de else "Prostate", data.get("findings_prostate_de" if is_de else "findings_prostate_en", "")),
        ("Lymphknoten" if is_de else "Lymph nodes", data.get("findings_lymph_de" if is_de else "findings_lymph_en", "")),
        ("Skelett" if is_de else "Osseous disease", data.get("findings_bone_de" if is_de else "findings_bone_en", "")),
        ("Viszerale Weichteile" if is_de else "Visceral Disease", data.get("findings_visceral_de" if is_de else "findings_visceral_en", ""))
    ]

    for reg_title, reg_txt in regions:
        if not reg_txt:
            continue
        reg_txt_clean = " ".join([line.strip().lstrip('•>>-').strip() for line in reg_txt.split('\n') if line.strip()])
        p_reg = doc.add_paragraph()
        p_reg.paragraph_format.space_before = Pt(4)
        p_reg.paragraph_format.space_after = Pt(4)
        r_txt = p_reg.add_run(f"- {reg_title}: {reg_txt_clean}")
        r_txt.font.size = Pt(10)
        r_txt.font.color.rgb = DARK_TEXT

    # 4. Conclusion
    h6 = doc.add_paragraph()
    h6.paragraph_format.space_before = Pt(14)
    h6.paragraph_format.space_after = Pt(4)
    r_h6 = h6.add_run("Beurteilung" if is_de else "Conclusion")
    r_h6.font.size = Pt(12)
    r_h6.font.bold = True
    r_h6.font.color.rgb = PRIMARY

    concl_txt = data.get("conclusion_de" if is_de else "conclusion_en", "")
    p_concl = doc.add_paragraph(concl_txt)
    p_concl.paragraph_format.space_after = Pt(10)
    p_concl.style.font.size = Pt(10)

    # Final Staging & Response Callout Box
    tmtv = data.get("tmtv_cc", 0.0)
    recip = data.get("overall_recip", "BASELINE")
    tmtv_delta = data.get("tmtv_delta_pct", 0.0)

    box_tbl = doc.add_table(rows=1, cols=1)
    box_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_box = box_tbl.cell(0, 0)
    set_cell_background(c_box, "F0F4F8")
    set_cell_margins(c_box, top=140, bottom=140, left=200, right=200)
    set_table_borders(box_tbl, color="1B365D", sz="12")

    p_box = c_box.paragraphs[0]
    p_box.paragraph_format.space_after = Pt(0)
    r_b1 = p_box.add_run(f"Zusammenfassendes miTNM-Stadium: {mitnm}\n" if is_de else f"Summary miTNM Classification: {mitnm}\n")
    r_b1.font.bold = True
    r_b1.font.size = Pt(11)
    r_b1.font.color.rgb = PRIMARY

    r_b2 = p_box.add_run(f"• Total Metabolic Tumor Volume (TMTV): {tmtv:.2f} cc\n")
    r_b2.font.size = Pt(9.5)
    if tmtv_delta != 0.0:
        sgn = "+" if tmtv_delta > 0 else ""
        r_b2 = p_box.add_run(f"• Longitudinal TMTV Change: {sgn}{tmtv_delta:.1f}%\n")
        r_b2.font.size = Pt(9.5)

    r_b3 = p_box.add_run(f"• Response Classification (PERCIST/RECIP): {recip}")
    r_b3.font.bold = True
    r_b3.font.size = Pt(9.5)



    # Footnote with reference guidelines
    p_foot = doc.add_paragraph()
    p_foot.paragraph_format.space_before = Pt(18)
    r_ft = p_foot.add_run(
        "Referenzen: (1) Ceci F, et al. E-PSMA: the EANM standardized reporting guidelines v1.0 for PSMA-PET. Eur J Nucl Med Mol Imaging 2021; 48:1626–1638. "
        "(2) Eiber M, et al. Prostate cancer molecular imaging standardized evaluation (PROMISE): proposed miTNM classification. J Nucl Med 2018; 59:469–478."
    )
    r_ft.font.size = Pt(7.5)
    r_ft.font.italic = True
    r_ft.font.color.rgb = MUTED

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
