#!/usr/bin/env python3
"""
Populate `_meta_/metadata.json` in `Prostata_bimodal` HDF5 files from
`data/temp/Table MRT_PET october26 (1).xlsx` so MedEye3d displays the
translated PSMA PET and MRI radiological reports in the Radiological Report section.

Usage:
  /mnt/big/project_ssd/project_ssd/lymph_node_rules/for_joanna/.local/conda/envs/seg/bin/python \
      scripts/preprocessing/populate_radiological_reports.py [--dry-run] [--cases Pat22 Pat51] [--sync-cache]
"""
import argparse
import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import sys

import h5py
import openpyxl

EXCEL_PATH = "/workspaces/MedEye3d.jl/data/temp/Table MRT_PET october26 (1).xlsx"
CASES_DIR = "/workspaces/MedEye3d.jl/data/cases/Prostata_bimodal"
CACHE_DIR = os.path.expanduser("~/.medeye3d/xnat_cache")

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "januar": 1, "februar": 2, "märz": 3, "maerz": 3, "mai": 5, "juni": 6,
    "juli": 7, "oktober": 10, "dezember": 12,
}

DE_HINT_RE = re.compile(
    r"\b(und|der|die|das|mit|bei|kein|keine|einem|einer|nicht|zur|zum|im|am|Untersuchung|Befund|Beurteilung|Einrichtung)\b"
)

KNOWN_TRANSLATIONS = {
    "Pat117": (
        "Presented at our facility on 20.01.2025.\n"
        "Clinical presentation / Question or justifying indication:\n"
        "Intermediate-risk prostate carcinoma (Gleason 7a, iPSA 13.3 ng/ml)\n"
        "Currently primary staging\n"
        "Informed consent:\n"
        "After obtaining medical history and establishing the indication, the patient was informed about the purpose and procedure of the PSMA PET/CT examination. Specific and individual patient questions are documented on the consent form. The patient was instructed on specific behavioral precautions before, during, and after the examination. The nuclear medicine physician verified the justifying indication.\n"
        "Examination technique:\n"
        "Administration of 20 ml Accupaque for ureteral opacification (no diagnostic contrast-enhanced CT). Topogram for examination planning (skull base to upper third of the femora bilaterally). Following intravenous administration of 138 MBq Ga-68-PSMA-11, PET emission acquisition was performed with an overall uncomplicated course (uptake phase 110 minutes).\n"
        "Findings:\n"
        "No prior examinations available for comparison.\n"
        "Neck (excl. bones):\n"
        "Physiologically increased uptake in the salivary glands (parotid and submandibular/sublingual glands bilaterally). As far as assessable on low-dose CT, no acute cervical pathologies.\n"
        "Thorax (excl. bones):\n"
        "No evidence of suspicious thoracic tracer uptake. No pleural effusions. No pneumothorax. As far as assessable on low-dose CT, no acute thoracic pathologies.\n"
        "Abdomen/Pelvis (excl. bones):\n"
        "No evidence of suspicious uptake in the upper abdomen.\n"
        "Focal intense PSMA expression in the right prostatic lobe (SUVmax: 28.9) at approx. 7-10 o'clock lithotomy position. No evidence of other suspicious pelvic uptake.\n"
        "As far as assessable on low-dose CT, no acute abdominal or pelvic pathologies.\n"
        "Osseous status:\n"
        "No evidence of malignancy-associated osteodestruction or suspicious osseous tracer uptake.\n"
        "Impression:\n"
        "Intense PSMA expression in the prostate carcinoma in the right prostatic lobe.\n"
        "No evidence of lymph node or distant metastases.\n"
        "No evidence of other suspicious PSMA-positive findings."
    ),
}


def is_german(text: str) -> bool:
    return len(DE_HINT_RE.findall(text)) >= 3


def extract_yyyymmdd(text: str, fallback: str = "00000000") -> str:
    if not text:
        return fallback
    head = text[:400]
    m_dot = re.search(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", head)
    if m_dot:
        d, m, y = int(m_dot.group(1)), int(m_dot.group(2)), int(m_dot.group(3))
        if 1 <= m <= 12 and 1 <= d <= 31:
            return f"{y:04d}{m:02d}{d:02d}"
    m_en = re.search(
        r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December|"
        r"Januar|Februar|März|Maerz|Mai|Juni|Juli|Oktober|Dezember)\s+(\d{4})\b",
        head,
        re.IGNORECASE,
    )
    if m_en:
        d = int(m_en.group(1))
        m = MONTHS.get(m_en.group(2).lower(), 0)
        y = int(m_en.group(3))
        if 1 <= m <= 12 and 1 <= d <= 31:
            return f"{y:04d}{m:02d}{d:02d}"
    return fallback


def translate_de_to_en_via_julia(text: str) -> str:
    """Call MedEye3d.LLMDictation.translate_report_to_english for German reports."""
    jl_code = (
        "using MedEye3d\n"
        "txt = read(stdin, String)\n"
        "res = MedEye3d.LLMDictation.translate_report_to_english(txt)\n"
        "print(res)\n"
    )
    proc = subprocess.run(
        ["julia", "--project=/workspaces/MedEye3d.jl", "-e", jl_code],
        input=text,
        text=True,
        capture_output=True,
        timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Julia translation failed: {proc.stderr[-400:]}")
    out = proc.stdout.strip()
    if not out or out.startswith(("Error", "API Error", "Connection Error")):
        raise RuntimeError(f"LLM translation returned error: {out[:200]}")
    return out


def clean_report(val) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if s.lower() in ("", "keine angaben", "nan", "none"):
        return ""
    return s


def load_excel_reports(excel_path: str) -> dict:
    wb = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
    ws = wb["bimodal"]
    records = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        pid = int(str(row[0]).strip())
        dob = row[1]
        if isinstance(dob, (datetime.datetime, datetime.date)):
            dob_str = dob.strftime("%Y-%m-%d")
        else:
            dob_str = str(dob or "").strip()
        records[pid] = {
            "dob": dob_str,
            "psma": clean_report(row[2]),
            "mri": clean_report(row[3]),
        }
    return records


def build_metadata_dict(existing: dict, pat_id: str, rec: dict, translation_cache: dict) -> dict:
    meta = dict(existing) if isinstance(existing, dict) else {}
    meta["patient_id"] = pat_id
    meta["case_id"] = pat_id
    if rec["dob"]:
        meta["date_of_birth"] = rec["dob"]

    # Remove old synthetic 00000000_* / YYYYMMDD_PET / YYYYMMDD_MRI keys before rewriting cleanly
    for k in list(meta.keys()):
        if isinstance(meta[k], dict) and (k.endswith("_PET") or k.endswith("_MRI")):
            del meta[k]

    def _make_entry(text: str, mod_key: str) -> tuple:
        ymd = extract_yyyymmdd(text, fallback="00000000")
        key = f"{ymd}_{mod_key}"
        entry = {mod_key: {}, "Modality": mod_key}
        if is_german(text):
            entry["Description"] = text
            if pat_id in KNOWN_TRANSLATIONS and mod_key == "PET":
                entry["EnglishDescription"] = KNOWN_TRANSLATIONS[pat_id]
            elif text in translation_cache:
                entry["EnglishDescription"] = translation_cache[text]
            else:
                print(f"  [{pat_id}] Translating German {mod_key} report ({len(text)} chars) via LLMDictation...")
                en_text = translate_de_to_en_via_julia(text)
                translation_cache[text] = en_text
                entry["EnglishDescription"] = en_text
        else:
            entry["EnglishDescription"] = text
        return key, entry

    if rec["psma"]:
        k, e = _make_entry(rec["psma"], "PET")
        meta[k] = e
    if rec["mri"]:
        k, e = _make_entry(rec["mri"], "MRI")
        if k in meta:
            k = k + "_2"
        meta[k] = e
    return meta


def update_h5_metadata(h5_path: str, pat_id: str, rec: dict, translation_cache: dict, dry_run: bool) -> bool:
    with h5py.File(h5_path, "r") as f:
        existing = {}
        if "_meta_/metadata.json" in f:
            raw = f["_meta_/metadata.json"][()]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            try:
                existing = json.loads(raw)
            except Exception:
                existing = {}

    new_meta = build_metadata_dict(existing, pat_id, rec, translation_cache)
    if new_meta == existing:
        return False
    if dry_run:
        return True

    with h5py.File(h5_path, "a") as f:
        m_grp = f.require_group("_meta_")
        if "metadata.json" in m_grp:
            del m_grp["metadata.json"]
        m_grp.create_dataset("metadata.json", data=json.dumps(new_meta, ensure_ascii=False))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sync-cache", action="store_true")
    ap.add_argument("--cases", nargs="*", default=None)
    args = ap.parse_args()

    records = load_excel_reports(EXCEL_PATH)
    cases = sorted(
        [
            os.path.basename(p)
            for p in glob.glob(os.path.join(CASES_DIR, "Pat*"))
            if "_student" not in p and os.path.isfile(os.path.join(p, "preprocessed_volumes.h5"))
        ],
        key=lambda s: int(re.search(r"\d+", s).group()),
    )
    if args.cases:
        want = set(args.cases)
        cases = [c for c in cases if c in want]

    translation_cache = {}
    updated, skipped, missing = 0, 0, []
    for pat_id in cases:
        pid = int(re.search(r"\d+", pat_id).group())
        if pid not in records:
            missing.append(pat_id)
            continue
        h5_path = os.path.join(CASES_DIR, pat_id, "preprocessed_volumes.h5")
        changed = update_h5_metadata(h5_path, pat_id, records[pid], translation_cache, args.dry_run)
        if changed:
            updated += 1
        else:
            skipped += 1
        if args.sync_cache and not args.dry_run and os.path.isdir(CACHE_DIR):
            for cache_h5 in glob.glob(os.path.join(CACHE_DIR, "**", f"*{pat_id}_preprocessed*.h5"), recursive=True):
                if os.path.isfile(cache_h5):
                    update_h5_metadata(cache_h5, pat_id, records[pid], translation_cache, dry_run=False)

    print(f"Done: updated={updated}, unchanged={skipped}, missing_in_excel={missing}, dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
