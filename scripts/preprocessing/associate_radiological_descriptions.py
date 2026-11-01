#!/usr/bin/env python3
"""
associate_radiological_descriptions.py

Associates radiological reports (both English and German) from
`/workspaces/MedEye3d.jl/data/Table MRT_PET august26.xlsx`
into MedEye3d preprocessed_volumes.h5 datasets.

For each patient:
1. Looks up Patient ID in the Excel table.
2. Extracts Date of Birth, PSMA report, and MRI report (in English).
3. Translates English reports into German using local Helsinki-NLP/opus-mt-en-de model.
4. Formats standard multi-timepoint metadata:
   - TP 0 (PET/CT): EnglishDescription & Description (German)
   - TP 1 (MRI): EnglishDescription & Description (German, or 'keine Angaben')
5. Injects metadata into `_meta_/metadata.json` inside preprocessed_volumes.h5.
6. Writes companion metadata.json into the case directory.
"""

import os
import sys
import json
import re
import time
import openpyxl
import h5py
import torch
torch.set_num_threads(4)
from transformers import MarianTokenizer, MarianMTModel

EXCEL_PATH = "/workspaces/MedEye3d.jl/data/Table MRT_PET october26.xlsx" if os.path.isfile("/workspaces/MedEye3d.jl/data/Table MRT_PET october26.xlsx") else "/workspaces/MedEye3d.jl/data/Table MRT_PET august26.xlsx"
CASES_DIR = "/workspaces/MedEye3d.jl/data/cases/Prostata_bimodal"
CACHE_PATH = "/workspaces/MedEye3d.jl/data/translation_cache.json"

import threading
_TRANSLATOR_INSTANCE = None
_TRANSLATOR_LOCK = threading.Lock()
_REPORTS_DB_CACHE = None
_TRANSLATION_CACHE = None

def _load_cache():
    global _TRANSLATION_CACHE
    if os.path.isfile(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                _TRANSLATION_CACHE = json.load(f)
        except Exception:
            if _TRANSLATION_CACHE is None:
                _TRANSLATION_CACHE = {}
    elif _TRANSLATION_CACHE is None:
        _TRANSLATION_CACHE = {}
    return _TRANSLATION_CACHE

def _save_cache():
    if _TRANSLATION_CACHE is not None:
        try:
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(_TRANSLATION_CACHE, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

def get_translator():
    global _TRANSLATOR_INSTANCE
    with _TRANSLATOR_LOCK:
        if _TRANSLATOR_INSTANCE is not None:
            return _TRANSLATOR_INSTANCE
        model_name = "Helsinki-NLP/opus-mt-en-de"
        device = "cpu"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(device)
        
        def translate(text):
            if not text or not text.strip() or text.strip().lower() in ["keine angaben", "none", "nan"]:
                return "keine Angaben"
            cache = _load_cache()
            if text in cache:
                return cache[text]
            with _TRANSLATOR_LOCK:
                paras = text.split("\n")
                non_empty = [(i, p.strip()) for i, p in enumerate(paras) if p.strip()]
                if not non_empty:
                    return text
                indices, texts_to_trans = zip(*non_empty)
                # Chunked batch translation with limited threads to prevent contention
                torch.set_num_threads(4)
                translated_texts = []
                chunk_size = 8
                for c_i in range(0, len(texts_to_trans), chunk_size):
                    chunk = list(texts_to_trans[c_i:c_i + chunk_size])
                    inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                    outputs = model.generate(**inputs, max_new_tokens=256)
                    translated_texts.extend([tokenizer.decode(out, skip_special_tokens=True) for out in outputs])
                
                result_paras = list(paras)
                for idx, trans in zip(indices, translated_texts):
                    result_paras[idx] = trans
                result = "\n".join(result_paras)
                cache[text] = result
                _save_cache()
                return result
        _TRANSLATOR_INSTANCE = translate
        return _TRANSLATOR_INSTANCE

def parse_date(date_text, default="20230101"):
    if not date_text:
        return default
    # Search for DD Month YYYY (e.g. 31 August 2023)
    months = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12"
    }
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", date_text)
    if m:
        day, mon, year = m.groups()
        mon_num = months.get(mon.lower(), "01")
        return f"{year}{mon_num}{int(day):02d}"
    
    # Search for DD.MM.YYYY
    m2 = re.search(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", date_text)
    if m2:
        day, mon, year = m2.groups()
        return f"{year}{int(mon):02d}{int(day):02d}"
        
    return default

def load_patient_reports(excel_path):
    global _REPORTS_DB_CACHE
    with _TRANSLATOR_LOCK:
        if _REPORTS_DB_CACHE is not None:
            return _REPORTS_DB_CACHE
        wb = openpyxl.load_workbook(excel_path, data_only=True)
        sheet = wb.active
        records = {}
        for row in sheet.iter_rows(min_row=2, values_only=True):
            if row[0] is None:
                continue
            try:
                pid = int(row[0])
            except (ValueError, TypeError):
                continue
            dob = str(row[1]) if row[1] is not None else ""
            if " " in dob:
                dob = dob.split(" ")[0]
            psma_rep = str(row[2]).strip() if row[2] is not None else ""
            mri_rep = str(row[3]).strip() if row[3] is not None else ""
            records[pid] = {
                "dob": dob,
                "psma_report": psma_rep,
                "mri_report": mri_rep
            }
        _REPORTS_DB_CACHE = records
        return records

def process_case(case_dir, patient_id_int, reports_db, translator):
    h5_path = os.path.join(case_dir, "preprocessed_volumes.h5")
    if not os.path.isfile(h5_path):
        print(f"  [SKIP] {h5_path} does not exist")
        return False
        
    rec = reports_db.get(patient_id_int)
    if not rec:
        print(f"  [WARN] Patient {patient_id_int} not found in Excel table")
        return False
        
    dob = rec["dob"]
    psma_en = rec["psma_report"]
    mri_en = rec["mri_report"]
    
    # Check if MRI is empty or 'keine Angaben'
    if not mri_en or mri_en.lower() in ["none", "nan"]:
        mri_en = "keine Angaben"
        
    print(f"  Patient {patient_id_int}: PSMA len={len(psma_en)}, MRI len={len(mri_en)}")
    
    # Dates
    pet_date = parse_date(psma_en, default="20230831")
    mri_date = parse_date(mri_en, default=f"{int(pet_date[:8])+1}")
    
    print(f"  Dates: PET={pet_date}, MRI={mri_date}")
    
    # Translate to German
    t0 = time.time()
    psma_de = translator(psma_en)
    mri_de = translator(mri_en)
    print(f"  Translation completed in {round(time.time() - t0, 2)}s")
    
    # Read existing metadata to preserve other fields
    existing_meta = {}
    with h5py.File(h5_path, "r") as h5:
        if "_meta_/metadata.json" in h5:
            try:
                raw_meta = json.loads(h5["_meta_/metadata.json"][()])
                if isinstance(raw_meta, list) and len(raw_meta) > 0 and isinstance(raw_meta[0], dict):
                    existing_meta = raw_meta[0]
                elif isinstance(raw_meta, dict):
                    existing_meta = raw_meta
            except Exception:
                pass
                
    patient_info = {
        "PatientID": f"Pat{patient_id_int}",
        "SubjectID": existing_meta.get("SubjectID", ""),
        "ExperimentID": existing_meta.get("ExperimentID", ""),
        "BirthdayDate": dob,
        "Modalities": ["CT", "PET", "MRI"],
        "CreatedDate": existing_meta.get("CreatedDate", time.strftime("%Y-%m-%d %H:%M:%S")),
        "ReferenceSpacing": existing_meta.get("ReferenceSpacing", [1.0, 1.0, 3.0]),
        "ReferenceOrigin": existing_meta.get("ReferenceOrigin", [0.0, 0.0, 0.0])
    }
    
    # Date entry 1 (PET/CT)
    pet_key = f"{pet_date}_PET"
    pet_entry = {
        pet_key: {
            "Description": psma_de,
            "EnglishDescription": psma_en,
            "Modality": "PET",
            "PET": {
                "name": "SUV_PET_Image_0",
                "Modality": "PET",
                "StudyDate": pet_date,
                "SeriesDescription": "PET PSMA"
            },
            "CT": {
                "name": "Fixed_CT_Volume_0",
                "Modality": "CT",
                "StudyDate": pet_date,
                "SeriesDescription": "CT AC"
            }
        }
    }
    
    # Date entry 2 (MRI)
    mri_key = f"{mri_date}_MRI"
    mri_entry = {
        mri_key: {
            "Description": mri_de,
            "EnglishDescription": mri_en,
            "Modality": "T2",
            "MRI": {
                "name": "Fixed_CT_Volume_1",
                "Modality": "T2",
                "StudyDate": mri_date,
                "SeriesDescription": "T2 MRI Registered"
            },
            "PET": {
                "name": "SUV_PET_Image_1",
                "Modality": "PET",
                "StudyDate": mri_date
            }
        }
    }
    
    metadata = [patient_info, pet_entry, mri_entry]
    meta_json_str = json.dumps(metadata, indent=2)
    
    # Write to HDF5
    with h5py.File(h5_path, "a") as h5:
        if "_meta_/metadata.json" in h5:
            del h5["_meta_/metadata.json"]
        h5["_meta_"].create_dataset("metadata.json", data=meta_json_str)
        
    # Write companion file
    companion_path = os.path.join(case_dir, "metadata.json")
    with open(companion_path, "w", encoding="utf-8") as f:
        f.write(meta_json_str)
        
    print(f"  ✓ Successfully updated {h5_path} with English and German descriptions!")
    return True

def main():
    print("Loading patient reports from Excel...")
    reports = load_patient_reports(EXCEL_PATH)
    print(f"Loaded {len(reports)} patient reports from Excel.")
    
    print("Initializing translation model...")
    translator = get_translator()
    print("Translator ready.")
    
    target_cases = [d for d in os.listdir(CASES_DIR) if os.path.isdir(os.path.join(CASES_DIR, d))]
    target_cases.sort()
    
    for case_name in target_cases:
        m = re.search(r"Pat(\d+)", case_name)
        if not m:
            continue
        pid = int(m.group(1))
        case_dir = os.path.join(CASES_DIR, case_name)
        print(f"\nProcessing {case_name} (ID: {pid})...")
        process_case(case_dir, pid, reports, translator)
        
    print("\nAll cases processed successfully!")

if __name__ == "__main__":
    main()
