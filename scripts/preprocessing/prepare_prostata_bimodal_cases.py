#!/usr/bin/env python3
"""
prepare_prostata_bimodal_cases.py

Downloads and prepares 4 PET PSMA - MRI pair cases from XNAT:
  - Pat44 (XNATImagingPlatform_S00965 / E00128)
  - Pat33 (XNATImagingPlatform_S00966 / E00106)
  - Pat23 (XNATImagingPlatform_S00968 / E00088)
  - Pat24 (XNATImagingPlatform_S00969 / E00006)

Pipeline for each case:
1. Downloads:
   - Full body CT (Fixed_CT_Volume.nii.gz / CT_WB_*.nii.gz)
   - PET SUV volume (SUV_PET_WB_*.nii.gz)
   - Registered MRI volume (T2 Registered to CT / SWD)
   - Lesion segmentations (NNUNET_OR_Processed_Lesions_CT.seg.nrrd, MR.seg.nrrd)
   - Packed anatomical tensor (packed_anatomical_tensor.h5 with 161+ structures)
   - 3D Slicer verification scene (.mrb)
2. Resampling & Processing:
   - Resamples PET SUV, Registered MRI, and Lesion NRRD onto the full-resolution CT grid.
   - Unpacks anatomical tensor into multi-label max_anatomy volume using strict tissue layering:
       Layer 0: Muscles and soft tissue
       Layer 1: Visceral organs and blood vessels
       Layer 2: Skeletal bones (vertebrae, ribs, pelvis, femur, skull, etc.)
       Layer 3: Target structures (prostate, seminal vesicles, urethra)
   - Normalizes label names matching TotalSegmentator ontology (max_anatomy_to_ontology.json).
   - Generates binary bone_atlas and skellytour masks with corresponding _meta_/bone_label_ids.
   - Cleans physiological uptake: purges lesions from urinary bladder, parotid, submandibular,
     and lacrimal glands (both by segment name and by 3D organ mask intersection).
   - Computes champion Heat-GDT diffusivity tensor D(x) = eps + (1-eps)*exp(-gamma*E) with insulation.
   - Performs dynamic per-lesion organ mapping and generates cross-timepoint matches.
   - Pre-flips ALL volumes along axis 1 (Y) to guarantee standard radiological transverse orientation
     (anterior/belly at the top, posterior/spine at the bottom, left/right preserved) with preflipped=1.
   - Recomputes 1-based CENTROIDS from the flipped coordinates so lesion navigation lands accurately.
3. Packages into MedEye3d preprocessed_volumes.h5 with dual-timepoint quad-view support:
   - BASELINE (TP 0): CT + PET SUV + CT Lesions + max_anatomy + diffusivity + skellytour
   - TFM_Transform_FollowUp_to_Baseline_1.tfm (TP 1): MRI + PET SUV overlay + MR Lesions + mri_anatomy + diffusivity
   - ATLAS: max_anatomy, bone_atlas, skellytour
   - CENTROIDS: tp0_lidX, tp1_lidX
   - _meta_: preflipped=1, bone_label_ids, max_anatomy_labels.json, prostate_anatomy_labels.json,
             organ_mapping, segment_names.json, matches.json, scene_hierarchy.json, metadata.json
"""

import os
import sys
import io
import json
import time
import re
import shutil
import subprocess
import zipfile
import tempfile
import requests
import numpy as np
import h5py
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_credentials():
    creds_path = "/root/.medeye3d/xnat_credentials.json"
    if os.path.isfile(creds_path):
        try:
            with open(creds_path, "r") as f:
                creds = json.load(f)
            alias = creds.get("alias") or creds.get("username")
            secret = creds.get("secret") or creds.get("password")
            server = creds.get("server_url", "https://imaging-platform.diz-ag.med.ovgu.de")
            if alias and secret:
                return (alias, secret), server.rstrip("/")
        except Exception as e:
            print(f"  [WARN] Failed to load {creds_path}: {e}")
    alias = os.environ.get("XNAT_ALIAS", "0b3032a3-00b9-4555-a385-4809046779c8")
    secret = os.environ.get("XNAT_SECRET", "r0c51pSg9wP7r5eO1qC1q55k9z0fV049jGvX9X2aOQ1Z")
    server = os.environ.get("XNAT_SERVER", "https://imaging-platform.diz-ag.med.ovgu.de")
    return (alias, secret), server.rstrip("/")

AUTH, BASE = load_credentials()

TARGET_CASES = [
    {
        "patient_id": "Pat44",
        "subject_id": "XNATImagingPlatform_S00965",
        "exp_id": "XNATImagingPlatform_E00128",
        "mri_pattern": "T2_TSE_TRA_3.5mm_Registered_to_CT"
    },
    {
        "patient_id": "Pat33",
        "subject_id": "XNATImagingPlatform_S00966",
        "exp_id": "XNATImagingPlatform_E00106",
        "mri_pattern": "T2W_TSE_tra_Registered_to_CT"
    },
    {
        "patient_id": "Pat23",
        "subject_id": "XNATImagingPlatform_S00968",
        "exp_id": "XNATImagingPlatform_E00088",
        "mri_pattern": "T2_ax_3_mm_Registered_to_CT"
    },
    {
        "patient_id": "Pat24",
        "subject_id": "XNATImagingPlatform_S00969",
        "exp_id": "XNATImagingPlatform_E00006",
        "mri_pattern": "T2_ax_3_mm_Registered_to_CT"
    }
]

def classify_mri_modality(fname):
    """Classifies an MRI filename into modality (T2, ADC, DWI, T1, DCE) and plane (ax, cor, sag)."""
    fl = fname.lower()
    if any(fl.endswith(ext) for ext in [".seg.nrrd", ".nrrd", ".fcsv", ".mrb", ".json", ".tfm"]):
        return None
    if "lesion" in fl or "mask" in fl or "totalseg" in fl or "segmentation" in fl or "seg" in fl or "prostate" in fl or "mri_registered" in fl:
        return None
    if not (fl.endswith(".nii.gz") or fl.endswith(".nii")):
        return None
    if "ct" in fl and not any(k in fl for k in ["registered_to_ct", "to_ct"]):
        return None
    if "suv" in fl or "pet" in fl:
        return None

    if any(k in fl for k in ["adc", "ddadc", "eadc"]):
        mod = "ADC"
    elif any(k in fl for k in ["dwi", "dw_", "isodwi", "b1400", "b1500", "b1200", "b-50", "b50", "bval", "diff_", "trace"]):
        mod = "DWI"
    elif any(k in fl for k in ["dce", "dyn", "dynamik", "lava", "water", "perfusion", "sub."]):
        mod = "DCE"
    elif any(k in fl for k in ["t1", "t1w", "fl2d", "vibe"]):
        mod = "T1"
    elif any(k in fl for k in ["t2", "t2w", "haste", "space", "prop"]):
        mod = "T2"
    else:
        mod = "MRI"

    if any(k in fl for k in ["cor", "coronal"]):
        plane = "cor"
    elif any(k in fl for k in ["sag", "sagittal"]):
        plane = "sag"
    else:
        plane = "ax"

    return mod, plane

def select_ranked_mri_sequences(all_files_map):
    """
    Ranks available MRI sequences for multi-modal quad comparison:
    Rank 1: T2 (Axial)
    Rank 2: ADC (Axial)
    Rank 3: DWI (Axial)
    Rank 4: T1 (Axial)
    Rank 5: DCE (Axial)
    Followed by remaining axial, then coronal/sagittal planes.
    """
    parsed = []
    for fn, item in all_files_map.items():
        res = classify_mri_modality(fn)
        if res:
            mod, plane = res
            is_orig = "original" in fn.lower()
            is_swd = "swd" in fn.lower()
            is_reg = "registered" in fn.lower()
            parsed.append({"mod": mod, "plane": plane, "fn": fn, "item": item, "is_orig": is_orig, "is_swd": is_swd, "is_reg": is_reg})

    unique_groups = {}
    for p in parsed:
        key = (p["mod"], p["plane"])
        if key not in unique_groups:
            unique_groups[key] = p
        else:
            curr = unique_groups[key]
            def _score(x):
                s = 0
                # User standard: prefer raw non-registered MRI (inherently aligned in scanner space)
                if x.get("is_orig", False): s += 20
                if not x.get("is_reg", False): s += 10
                if x.get("is_swd", False): s += 5
                return s
            if _score(p) > _score(curr):
                unique_groups[key] = p

    ordered = []
    # Rank 1: T2 axial
    if ("T2", "ax") in unique_groups:
        ordered.append(unique_groups.pop(("T2", "ax")))
    elif any(k[0] in ("T2", "MRI") and k[1] == "ax" for k in unique_groups):
        k = [k for k in unique_groups if k[0] in ("T2", "MRI") and k[1] == "ax"][0]
        ordered.append(unique_groups.pop(k))

    # Rank 2: ADC axial
    if ("ADC", "ax") in unique_groups:
        ordered.append(unique_groups.pop(("ADC", "ax")))
    # Rank 3: DWI axial
    if ("DWI", "ax") in unique_groups:
        ordered.append(unique_groups.pop(("DWI", "ax")))
    # Rank 4: T1 axial
    if ("T1", "ax") in unique_groups:
        ordered.append(unique_groups.pop(("T1", "ax")))
    # Rank 5: DCE axial
    if ("DCE", "ax") in unique_groups:
        ordered.append(unique_groups.pop(("DCE", "ax")))

    # Any remaining axial sequences
    for k in list(unique_groups.keys()):
        if k[1] == "ax":
            ordered.append(unique_groups.pop(k))
    # Any remaining non-axial sequences
    for k in list(unique_groups.keys()):
        ordered.append(unique_groups.pop(k))

    return ordered

# Champion PDE parameters from DATA_AND_MODEL_ACCESS_GUIDE.md
CHAMPION_GAMMA = 10.095886
CHAMPION_EPS_D = 1.0e-5
CHAMPION_THETA = 0.00104404625
CHAMPION_DT = 0.16
CHAMPION_K = 200

# Load anatomy ontology
ONTOLOGY_PATH = "/workspaces/MedEye3d.jl/data/max_anatomy_to_ontology.json"
if os.path.isfile(ONTOLOGY_PATH):
    with open(ONTOLOGY_PATH, "r") as f:
        ONTOLOGY = json.load(f)
else:
    ONTOLOGY = {}

CANONICAL_ANAT_PATH = "/workspaces/MedEye3d.jl/data/canonical_anatomy_names.json"
if os.path.isfile(CANONICAL_ANAT_PATH):
    with open(CANONICAL_ANAT_PATH, "r") as f:
        CANONICAL_ANAT = set(json.load(f))
else:
    CANONICAL_ANAT = set()

def load_anat_registry(anat_h5_path, af=None):
    """
    Loads the anatomical tensor registry from either:
      1. The internal 'registry' dataset inside packed_anatomical_tensor.h5 (161-mask format)
      2. The sibling packed_anatomical_tensor_registry.json file (735-mask format, filtered to CANONICAL_ANAT)
    """
    raw_masks = None
    if af is not None and "registry" in af:
        reg_raw = af["registry"][()]
        if isinstance(reg_raw, bytes):
            reg_raw = reg_raw.decode("utf-8")
        data = json.loads(reg_raw)
        raw_masks = data.get("masks", data)
    else:
        reg_json_path = anat_h5_path.replace(".h5", "_registry.json")
        if not os.path.isfile(reg_json_path):
            reg_json_path = os.path.join(os.path.dirname(anat_h5_path), "packed_anatomical_tensor_registry.json")
        if os.path.isfile(reg_json_path):
            with open(reg_json_path, "r") as rf:
                data = json.load(rf)
            raw_masks = data.get("masks", data)
    if raw_masks:
        # Filter to canonical anatomical structures (with id > 0) so helper masks, temp masks & composite unions don't overwrite
        filtered = {}
        seen_norms = {}
        for k, meta in raw_masks.items():
            if not isinstance(meta, dict) or meta.get("id", 0) <= 0:
                continue
            norm = normalize_name(k)
            if not CANONICAL_ANAT or norm in CANONICAL_ANAT:
                # Prefer exact non-aliased names if duplicate normalized names exist
                if norm not in seen_norms or k == norm or k.startswith("nvseg_"):
                    if norm in seen_norms:
                        filtered.pop(seen_norms[norm], None)
                    seen_norms[norm] = k
                    filtered[k] = meta
        return filtered
    return {}

BONE_KEYWORDS = [
    'femur', 'hip', 'vertebra', 'rib', 'sacrum', 'clavicula', 'humerus',
    'scapula', 'sternum', 'skull', 'palate', 'bone', 'spine', 'mandible',
    'costal', 'ilium', 'ischium', 'pubis'
]

MUSCLE_TERMS = [
    'gluteus', 'iliopsoas', 'autochthon', 'pectoralis', 'latissimus',
    'trapezius', 'deltoid', 'infraspinatus', 'subscapularis', 'triceps',
    'teres', 'supraspinatus', 'coracobrachial', 'quadriceps', 'compartment',
    'biceps', 'sartorius', 'rectus', 'oblique', 'muscle', 'platysma', 'adductor'
]

def normalize_name(name):
    clean = name.replace("nvseg_", "").lower()
    if clean == "bladder":
        clean = "urinary_bladder"
    # Rib pattern: right_rib_6 -> rib_right_6
    m = re.match(r"(left|right)_rib_(\d+)", clean)
    if m:
        cand = f"rib_{m.group(1)}_{m.group(2)}"
        if cand in ONTOLOGY:
            return cand
    m = re.match(r"rib_(\d+)_(left|right)", clean)
    if m:
        cand = f"rib_{m.group(2)}_{m.group(1)}"
        if cand in ONTOLOGY:
            return cand
    # General left/right prefix
    m = re.match(r"(left|right)_(.*)", clean)
    if m:
        cand = f"{m.group(2)}_{m.group(1)}"
        if cand in ONTOLOGY:
            return cand
    # General left/right suffix
    m = re.match(r"(.*)_(left|right)", clean)
    if m:
        cand = f"{m.group(2)}_{m.group(1)}"
        if cand in ONTOLOGY:
            return cand
    return clean

def is_muscle(norm_name):
    if norm_name in ONTOLOGY:
        o = ONTOLOGY[norm_name]
        if o.get("is_muscle", False):
            return True
        if "Muscles" in o.get("anatomic_location", ""):
            return True
    return any(m in norm_name for m in MUSCLE_TERMS)

def is_bone(norm_name):
    if is_muscle(norm_name):
        return False
    if norm_name in ONTOLOGY:
        o = ONTOLOGY[norm_name]
        if "Skeleton" in o.get("anatomic_location", ""):
            return True
        if o.get("lesion_type") == "Bone Meta":
            return True
    return any(b in norm_name for b in BONE_KEYWORDS)

def is_prostate(norm_name):
    return any(k in norm_name for k in ["prostate", "seminal_vesicle", "urethra"])

def is_lymph_node(norm_name):
    lower = norm_name.lower()
    if norm_name in ONTOLOGY:
        o = ONTOLOGY[norm_name]
        if o.get("anatomic_location") == "Lymph Node" or o.get("lesion_type") == "Lymph Node":
            return True
    return any(lower.startswith(p) for p in [
        "neck_", "thoracic_", "axillary_", "abdominal_",
        "deep_inguinal", "superficial_inguinal", "inguinal_"
    ]) or any(k in lower for k in [
        'lymph', 'station', 'iliac', 'obturator',
        'paratracheal', 'subcarinial', 'subaortic', 'aortic_hiatus',
        'paraaortic', 'pararectal', 'presacral', 'inguinal',
        'supraclavicular', 'prepericardial', 'axillary', 'rotter',
        'submental'
    ])

def get_layer(norm_name):
    if is_prostate(norm_name):
        return 3
    if is_bone(norm_name):
        return 2
    if is_lymph_node(norm_name):
        return 1.5
    if is_muscle(norm_name):
        return 0
    return 1

def load_lymph_node_stations(case_dir, raw_dir, Z, Y, X, ct_img, exp_id=None):
    """
    Extracts lymph node stations from either:
    1. final_results_gpu.h5 in raw_dir / case_dir (recomputed with latest lymph_node_rules).
    2. All_Lymph_Node_Areas.seg.nrrd inside any .mrb archive in case_dir/raw_dir.
    3. Downloads final_results_gpu.h5 from XNAT LYMPH_NODE_RESULTS if not local.
    Returns:
      ln_masks: dict {station_name: binary_mask (Z, Y, X)}
    """
    # 1. Check for local final_results_gpu.h5 first (freshly recomputed by run_ln_worker.jl)
    h5_candidates = [
        os.path.join(raw_dir, "final_results_gpu.h5"),
        os.path.join(case_dir, "final_results_gpu.h5")
    ]
    h5_path = None
    for cand in h5_candidates:
        if os.path.isfile(cand) and os.path.getsize(cand) > 1024:
            h5_path = cand
            break

    if h5_path and os.path.isfile(h5_path):
        try:
            print(f"  [LYMPH NODES] Extracting stations from {os.path.basename(h5_path)}...")
            ln_masks = {}
            with h5py.File(h5_path, "r") as hf:
                station_keys = [
                    k for k in hf.keys() 
                    if k.startswith(('Neck_', 'Thoracic_', 'Axillary_', 'Abdominal_', 'Deep_Inguinal', 'Superficial_Inguinal')) 
                    and 'helper' not in k.lower() and not k.endswith('_Base')
                ]
                for k in station_keys:
                    arr = hf[k][:]
                    if arr.shape == (Z, Y, X):
                        m = (arr > 0)
                        if m.any():
                            ln_masks[k] = m
                    else:
                        print(f"  [WARN] Station {k} shape mismatch: {arr.shape} vs {(Z, Y, X)}")
            if ln_masks:
                print(f"  [LYMPH NODES] Extracted {len(ln_masks)} stations from HDF5.")
                return ln_masks
        except Exception as e:
            print(f"  [WARN] Failed to read stations from {h5_path}: {e}")

    # 2. Search for All_Lymph_Node_Areas.seg.nrrd inside .mrb archives
    mrb_candidates = []
    for d in [case_dir, raw_dir]:
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".mrb"):
                    mrb_candidates.append(os.path.join(d, f))
                    
    for mrb_path in mrb_candidates:
        try:
            with zipfile.ZipFile(mrb_path, "r") as z:
                ln_files = [f for f in z.namelist() if "All_Lymph_Node_Areas" in f]
                if ln_files:
                    ln_file = ln_files[0]
                    print(f"  [LYMPH NODES] Found {ln_file} inside {os.path.basename(mrb_path)}")
                    with tempfile.NamedTemporaryFile(suffix=".seg.nrrd") as tmp:
                        tmp.write(z.read(ln_file))
                        tmp.flush()
                        
                        reader = sitk.ImageFileReader()
                        reader.SetFileName(tmp.name)
                        reader.LoadPrivateTagsOn()
                        reader.ReadImageInformation()
                        
                        keys = reader.GetMetaDataKeys()
                        name_map = {}
                        for k in keys:
                            if k.endswith("_Name"):
                                pfx = k[:-len("_Name")]
                                val_key = f"{pfx}_LabelValue"
                                if reader.HasMetaDataKey(val_key):
                                    val = int(reader.GetMetaData(val_key))
                                    name_map[val] = reader.GetMetaData(k)
                                    
                        ln_img = sitk.ReadImage(tmp.name)
                        if ln_img.GetSize() != ct_img.GetSize() or ln_img.GetSpacing() != ct_img.GetSpacing():
                            ln_img = sitk.Resample(ln_img, ct_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
                            
                        ln_arr = sitk.GetArrayFromImage(ln_img)  # (Z, Y, X)
                        ln_masks = {}
                        for val, name in name_map.items():
                            m = (ln_arr == val)
                            if m.any():
                                ln_masks[name] = m
                        print(f"  [LYMPH NODES] Extracted {len(ln_masks)} stations from NRRD.")
                        return ln_masks
        except Exception as e:
            print(f"  [WARN] Could not read lymph nodes from {mrb_path}: {e}")

    # 3. Download final_results_gpu.h5 from XNAT if exp_id provided
    if not h5_path and exp_id:
        ln_url = f"{BASE}/data/experiments/{exp_id}/resources/LYMPH_NODE_RESULTS/files/final_results_gpu.h5"
        cand_dest = os.path.join(raw_dir, "final_results_gpu.h5")
        if download_file_streaming(ln_url, cand_dest, min_expected_size=10*1024*1024):
            h5_path = cand_dest
            try:
                ln_masks = {}
                with h5py.File(h5_path, "r") as hf:
                    station_keys = [
                        k for k in hf.keys() 
                        if k.startswith(('Neck_', 'Thoracic_', 'Axillary_', 'Abdominal_', 'Deep_Inguinal', 'Superficial_Inguinal')) 
                        and 'helper' not in k.lower() and not k.endswith('_Base')
                    ]
                    for k in station_keys:
                        arr = hf[k][:]
                        if arr.shape == (Z, Y, X):
                            m = (arr > 0)
                            if m.any():
                                ln_masks[k] = m
                print(f"  [LYMPH NODES] Extracted {len(ln_masks)} stations from downloaded HDF5.")
                return ln_masks
            except Exception as e:
                print(f"  [WARN] Failed to read stations from {h5_path}: {e}")
            
    return {}

def unpack_anatomy(anat_h5_path, ln_masks, Z, Y, X):
    """
    Unpacks packed_anatomical_tensor.h5 with hierarchical tissue layering:
      Layer 0: Muscles / soft tissues
      Layer 1: Visceral organs / vessels
      Layer 1.5: Lymph Node Stations (86 clinical stations)
      Layer 2: Skeletal bones
      Layer 3: Prostate / target organs
    """
    max_anatomy = np.zeros((Z, Y, X), dtype=np.uint16)
    exclusion_mask = np.zeros((Z, Y, X), dtype=np.uint8)
    anatomy_label_names = {}
    bone_ids = []
    
    with h5py.File(anat_h5_path, "r") as af:
        packed_tensor = af["packed_tensor"][:]  # (C, Z, Y, X)
        registry = load_anat_registry(anat_h5_path, af)
        
        items_by_layer = {0: [], 1: [], 2: [], 3: []}
        for name, meta in registry.items():
            norm = normalize_name(name)
            lay = int(get_layer(norm))
            items_by_layer[lay].append((name, norm, meta))
            
        lbl_idx = 1
        # Layer 0 (Muscles) and Layer 1 (Viscera & Vessels)
        for lay in [0, 1]:
            for name, norm, meta in sorted(items_by_layer[lay], key=lambda x: x[1]):
                ch = meta["channel"] - 1
                mid = meta["id"]
                if ch < packed_tensor.shape[0]:
                    mask_i = (packed_tensor[ch] == mid)
                    if mask_i.any():
                        anatomy_label_names[str(lbl_idx)] = norm
                        max_anatomy[mask_i] = lbl_idx
                        if any(t in norm for t in ["urinary_bladder", "bladder", "parotid", "submandibular", "lacrimal", "kidney"]):
                            exclusion_mask[mask_i] = 1
                        lbl_idx += 1
                        
        # Layer 1.5: Lymph Node Stations
        if ln_masks:
            print(f"  [ANATOMY] Layering {len(ln_masks)} lymph node stations (Layer 1.5)...")
            for name, mask_i in sorted(ln_masks.items(), key=lambda x: x[0]):
                anatomy_label_names[str(lbl_idx)] = name
                max_anatomy[mask_i] = lbl_idx
                lbl_idx += 1
                
        # Layer 2 (Bones) and Layer 3 (Prostate / Target Organs)
        for lay in [2, 3]:
            for name, norm, meta in sorted(items_by_layer[lay], key=lambda x: x[1]):
                ch = meta["channel"] - 1
                mid = meta["id"]
                if ch < packed_tensor.shape[0]:
                    mask_i = (packed_tensor[ch] == mid)
                    if mask_i.any():
                        anatomy_label_names[str(lbl_idx)] = norm
                        max_anatomy[mask_i] = lbl_idx
                        if is_bone(norm):
                            bone_ids.append(lbl_idx)
                        lbl_idx += 1
                        
    bone_mask = np.isin(max_anatomy, bone_ids).astype(np.float32)
    skellytour = bone_mask.copy()
    
    return max_anatomy, anatomy_label_names, bone_ids, bone_mask, skellytour, exclusion_mask

def download_file_streaming(url, dest_path, min_expected_size=1024):
    if os.path.isfile(dest_path) and os.path.getsize(dest_path) > min_expected_size:
        print(f"  [CACHE] File exists ({round(os.path.getsize(dest_path)/(1024*1024), 2)} MB): {os.path.basename(dest_path)}")
        return True
    
    print(f"  [DOWNLOAD] Fetching {os.path.basename(dest_path)}...")
    t0 = time.time()
    try:
        r = requests.get(url, auth=AUTH, stream=True, timeout=60)
        r.raise_for_status()
        tmp_dest = dest_path + ".tmp"
        with open(tmp_dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        if os.path.getsize(tmp_dest) < min_expected_size:
            print(f"  [WARN] Downloaded file smaller than expected ({os.path.getsize(tmp_dest)} bytes): {dest_path}")
            if os.path.isfile(tmp_dest):
                os.remove(tmp_dest)
            return False
        os.replace(tmp_dest, dest_path)
        sz_mb = os.path.getsize(dest_path) / (1024 * 1024)
        elapsed = max(0.1, time.time() - t0)
        print(f"  [DONE] {os.path.basename(dest_path)}: {sz_mb:.2f} MB in {elapsed:.1f}s ({sz_mb/elapsed:.2f} MB/s)")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to download {url}: {e}")
        return False

def get_resource_files_map(url):
    try:
        r = requests.get(url, auth=AUTH, timeout=30)
        if r.status_code == 200:
            data = r.json().get("ResultSet", {}).get("Result", [])
            return {f["Name"]: f["URI"] for f in data}
    except Exception as e:
        print(f"  [WARN] Could not query resource files from {url}: {e}")
    return {}

def compute_champion_diffusivity(ct_arr, pet_arr, exclusion_mask):
    """
    Computes champion diffusivity tensor D(x) = eps_D + (1 - eps_D) * exp(-gamma * E(x))
    using CT & PET finite difference edge magnitude, zeroing out exclusion organs.
    Arrays are expected in (Z, Y, X) order.
    """
    print("  [DIFFUSIVITY] Computing champion diffusivity field (gamma=10.096, eps=1e-5)...")
    # Normalize CT: (clamp(CT, -1024, 1500) - (-200)) / 400
    ct_norm = (np.clip(ct_arr, -1024.0, 1500.0) - (-200.0)) / 400.0
    # Normalize PET: clamp(PET, 0, 20) / 20
    pet_norm = np.clip(pet_arr, 0.0, 20.0) / 20.0
    
    # Finite differences for CT
    gz_ct, gy_ct, gx_ct = np.gradient(ct_norm)
    mag_ct = np.sqrt(gz_ct**2 + gy_ct**2 + gx_ct**2)
    mag_ct = gaussian_filter(mag_ct, sigma=0.8)
    
    # Finite differences for PET
    gz_pet, gy_pet, gx_pet = np.gradient(pet_norm)
    mag_pet = np.sqrt(gz_pet**2 + gy_pet**2 + gx_pet**2)
    mag_pet = gaussian_filter(mag_pet, sigma=0.8)
    
    # Sigmoidal edge probability
    edge_ct = 1.0 / (1.0 + np.exp(-3.0 * (mag_ct - 0.5)))
    edge_pet = 1.0 / (1.0 + np.exp(-4.0 * (mag_pet - 0.4)))
    E = np.maximum(edge_ct, edge_pet).astype(np.float32)
    
    # Compute diffusivity: D = eps_D + (1 - eps_D) * exp(-gamma * E)
    D = CHAMPION_EPS_D + (1.0 - CHAMPION_EPS_D) * np.exp(-CHAMPION_GAMMA * E)
    D = np.clip(D, CHAMPION_EPS_D, 1.0).astype(np.float32)
    
    # Apply insulation to exclusion organs
    if exclusion_mask is not None:
        D[exclusion_mask > 0] = 0.0
        print(f"  [INSULATION] Applied boundary insulation to {(exclusion_mask > 0).sum()} exclusion voxels.")
        
    return D

def compute_mri_diffusivity(mri_arr, pet_arr=None, anat_vol=None, exclusion_mask=None, modality="T2", companion_edges=None):
    """
    Computes genuine edge-preserving diffusivity tensor D(x) for an MRI sequence
    using normalized 3D gradient magnitude, prostate capsule boundaries,
    PET uptake edges, and exclusion organ insulation.
    """
    pos = mri_arr[mri_arr > 0]
    if len(pos) == 0:
        return np.ones_like(mri_arr, dtype=np.float32)
    p99 = np.percentile(pos, 99)
    p01 = np.percentile(pos, 1)
    norm = np.clip((mri_arr - p01) / max(p99 - p01, 1e-4), 0.0, 1.5)
    
    gz, gy, gx = np.gradient(norm)
    mag = gaussian_filter(np.sqrt(gz**2 + gy**2 + gx**2), sigma=0.8)
    
    mod_str = str(modality).upper()
    if "ADC" in mod_str:
        p_mid, slope = 0.08, 16.0
    elif "DWI" in mod_str or "BVAL" in mod_str:
        p_mid, slope = 0.08, 16.0
    elif "T1" in mod_str:
        p_mid, slope = 0.10, 15.0
    elif "DCE" in mod_str:
        p_mid, slope = 0.09, 15.0
    else:
        p_mid, slope = 0.12, 15.0
        
    edge_raw = 1.0 / (1.0 + np.exp(-slope * (mag - p_mid)))
    edge_zero = 1.0 / (1.0 + np.exp(-slope * (-p_mid)))
    E = np.clip((edge_raw - edge_zero) / (1.0 - edge_zero), 0.0, 1.0).astype(np.float32)
    
    if companion_edges:
        for ce in companion_edges:
            if ce is not None:
                E = np.maximum(E, 0.75 * ce)
                
    if pet_arr is not None and np.max(pet_arr) > 0.5:
        pet_norm = np.clip(pet_arr, 0.0, 20.0) / 20.0
        gz_p, gy_p, gx_p = np.gradient(pet_norm)
        mag_p = gaussian_filter(np.sqrt(gz_p**2 + gy_p**2 + gx_p**2), sigma=0.8)
        edge_p_raw = 1.0 / (1.0 + np.exp(-8.0 * (mag_p - 0.15)))
        edge_p_zero = 1.0 / (1.0 + np.exp(-8.0 * (-0.15)))
        edge_pet = np.clip((edge_p_raw - edge_p_zero) / (1.0 - edge_p_zero), 0.0, 1.0).astype(np.float32)
        E = np.maximum(E, edge_pet)
        
    if anat_vol is not None and np.max(anat_vol) > 0:
        pros_mask = (anat_vol > 0).astype(np.float32)
        if pros_mask.sum() > 0:
            gz_a, gy_a, gx_a = np.gradient(pros_mask)
            mag_a = gaussian_filter(np.sqrt(gz_a**2 + gy_a**2 + gx_a**2), sigma=0.6)
            edge_capsule = np.clip(mag_a * 2.0, 0.0, 1.0).astype(np.float32)
            E = np.maximum(E, edge_capsule)
            
    E = np.clip(E, 0.0, 1.0).astype(np.float32)
    D = CHAMPION_EPS_D + (1.0 - CHAMPION_EPS_D) * np.exp(-CHAMPION_GAMMA * E)
    D = np.clip(D, CHAMPION_EPS_D, 1.0).astype(np.float32)
    
    if exclusion_mask is not None:
        D[exclusion_mask > 0] = 0.0
        
    return D

def process_case(case_info, out_base_dir, offline=False):
    pat_id = case_info["patient_id"]
    subj_id = case_info["subject_id"]
    exp_id = case_info["exp_id"]
    mri_pattern = case_info.get("mri_pattern", "T2")
    
    case_dir = os.path.join(out_base_dir, pat_id)
    os.makedirs(case_dir, exist_ok=True)
    raw_dir = os.path.join(case_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print(f" PROCESSING CASE: {pat_id}")
    print(f" Destination:     {case_dir}")
    print(f" XNAT Subject:    {subj_id} | Experiment: {exp_id}")
    print("=" * 70)
    
    # 1. Query resources from XNAT (unless offline=True)
    if offline:
        suv_files, slicer_files, anat_files, reg_files, prostate_files, mrb_files = {}, {}, {}, {}, {}, {}
    else:
        subj_suv_url = f"{BASE}/data/projects/Prostata_bimodal_PIPELINE/subjects/{subj_id}/resources/scans_SUV_CT_PET/files?format=json"
        subj_slicer_url = f"{BASE}/data/projects/Prostata_bimodal_PIPELINE/subjects/{subj_id}/resources/slicer_final/files?format=json"
        exp_anat_url = f"{BASE}/data/experiments/{exp_id}/resources/ANATOMICAL_TENSOR/files?format=json"
        exp_reg_url = f"{BASE}/data/experiments/{exp_id}/resources/CT_MRI_REGISTRATION/files?format=json"
        exp_prostate_url = f"{BASE}/data/experiments/{exp_id}/resources/PROSTATE_SEGMENTATION/files?format=json"
        exp_mrb_url = f"{BASE}/data/experiments/{exp_id}/resources/SLICER_SCENES/files?format=json"
        
        suv_files = get_resource_files_map(subj_suv_url)
        slicer_files = get_resource_files_map(subj_slicer_url)
        anat_files = get_resource_files_map(exp_anat_url)
        reg_files = get_resource_files_map(exp_reg_url)
        prostate_files = get_resource_files_map(exp_prostate_url)
        mrb_files = get_resource_files_map(exp_mrb_url)
    
    # 2. Identify target file URLs
    ct_name = None
    ct_uri = None
    for name in ["Fixed_CT_Volume.nii.gz", "CT_WB_120min_Abdomen_3_0.nii.gz"]:
        if name in slicer_files:
            ct_name = name; ct_uri = slicer_files[name]; break
        elif name in suv_files:
            ct_name = name; ct_uri = suv_files[name]; break
    if not ct_name:
        for name, uri in suv_files.items():
            if "CT" in name and name.endswith(".nii.gz"):
                ct_name = name; ct_uri = uri; break
                
    suv_name = None
    suv_uri = None
    for name in ["SUV_PET_WB_120min_Uncorrected.nii.gz"]:
        if name in suv_files:
            suv_name = name; suv_uri = suv_files[name]; break
    if not suv_name:
        for name, uri in suv_files.items():
            if "SUV" in name and name.endswith(".nii.gz"):
                suv_name = name; suv_uri = uri; break
                
    # User standard: Use raw non-registered moving MRI (inherently aligned in scanner space)
    # and apply the shared registration transform (Transform_Landmark_SWD.h5) directly onto the CT grid!
    # (Pre-resampled volumes like MRI_registered_to_CT_SWD have origin/slice shifts).
    mri_name = None
    mri_uri = None
    for name, uri in slicer_files.items():
        if mri_pattern in name and "Original" in name and "cor" not in name.lower() and "sag" not in name.lower():
            mri_name = name; mri_uri = uri; break
    if not mri_name:
        for name, uri in slicer_files.items():
            if ("T2" in name or "MR" in name) and "Original" in name and "cor" not in name.lower() and "sag" not in name.lower():
                mri_name = name; mri_uri = uri; break
    if not mri_name and os.path.exists(raw_dir):
        for fn in os.listdir(raw_dir):
            if ("t2" in fn.lower() or "mr" in fn.lower()) and "original" in fn.lower() and "cor" not in fn.lower() and "sag" not in fn.lower() and (fn.endswith(".nii.gz") or fn.endswith(".nii")):
                mri_name = fn; mri_uri = ""; break
    # Fallback to pre-registered only if raw original T2 is unavailable
    if not mri_name:
        if "MRI_registered_to_CT_SWD.nii.gz" in reg_files:
            mri_name = "MRI_registered_to_CT_SWD.nii.gz"
            mri_uri = reg_files[mri_name]
        elif "MRI_registered_to_CT_Landmark_Init.nii.gz" in reg_files:
            mri_name = "MRI_registered_to_CT_Landmark_Init.nii.gz"
            mri_uri = reg_files[mri_name]
        else:
            for name, uri in slicer_files.items():
                if ("T2" in name or "MR" in name) and "Registered_to_CT" in name:
                    mri_name = name; mri_uri = uri; break
        
    tfm_name = None
    tfm_uri = None
    if "Transform_Landmark_SWD.h5" in reg_files:
        tfm_name = "Transform_Landmark_SWD.h5"; tfm_uri = reg_files[tfm_name]
    elif "Transform_Landmark_Init.h5" in reg_files:
        tfm_name = "Transform_Landmark_Init.h5"; tfm_uri = reg_files[tfm_name]
    else:
        for name, uri in reg_files.items():
            if ("transform" in name.lower() or "landmark" in name.lower() or "swd" in name.lower()) and (name.endswith(".h5") or name.endswith(".tfm")):
                tfm_name = name; tfm_uri = uri; break
        if not tfm_name:
            for name, uri in slicer_files.items():
                if ("transform" in name.lower() or "landmark" in name.lower() or "swd" in name.lower()) and (name.endswith(".h5") or name.endswith(".tfm")):
                    tfm_name = name; tfm_uri = uri; break

    lesion_ct_name = "NNUNET_OR_Processed_Lesions_CT.seg.nrrd" if "NNUNET_OR_Processed_Lesions_CT.seg.nrrd" in slicer_files else None
    lesion_ct_uri = slicer_files.get(lesion_ct_name) if lesion_ct_name else None
    lesion_mr_name = "NNUNET_OR_Processed_Lesions_MR.seg.nrrd" if "NNUNET_OR_Processed_Lesions_MR.seg.nrrd" in slicer_files else None
    lesion_mr_uri = slicer_files.get(lesion_mr_name) if lesion_mr_name else None
    
    anat_tensor_name = "packed_anatomical_tensor.h5" if "packed_anatomical_tensor.h5" in anat_files else None
    anat_tensor_uri = anat_files.get(anat_tensor_name)
    
    mrb_name = None
    mrb_uri = None
    for name, uri in mrb_files.items():
        if name.endswith(".mrb"):
            mrb_name = name; mrb_uri = uri; break
    if not mrb_name and "verification_scene.mrb" in slicer_files:
        mrb_name = "verification_scene.mrb"
        mrb_uri = slicer_files[mrb_name]
        
    prostate_name = None
    prostate_uri = None
    if "prostate_segmentation.nii.gz" in prostate_files:
        prostate_name = "prostate_segmentation.nii.gz"; prostate_uri = prostate_files[prostate_name]
    elif "MRI_Prostate_Mask_registered_SWD.nii.gz" in reg_files:
        prostate_name = "MRI_Prostate_Mask_registered_SWD.nii.gz"; prostate_uri = reg_files[prostate_name]
    elif "CT_Prostate_Mask.nii.gz" in reg_files:
        prostate_name = "CT_Prostate_Mask.nii.gz"; prostate_uri = reg_files[prostate_name]

    # Fallback to local raw cache if offline or files already downloaded
    local_files = os.listdir(raw_dir) if os.path.exists(raw_dir) else []
    if not ct_name:
        for f in ["Fixed_CT_Volume.nii.gz", "CT_WB_120min_Abdomen_3_0.nii.gz"]:
            if f in local_files:
                ct_name = f; ct_uri = None; break
        if not ct_name:
            for f in local_files:
                if "CT" in f and f.endswith(".nii.gz"):
                    ct_name = f; ct_uri = None; break
    if not suv_name:
        for f in ["SUV_PET_WB_120min_Uncorrected.nii.gz"]:
            if f in local_files:
                suv_name = f; suv_uri = None; break
        if not suv_name:
            for f in local_files:
                if "SUV" in f and f.endswith(".nii.gz"):
                    suv_name = f; suv_uri = None; break
    if not mri_name:
        for f in ["MRI_registered_to_CT_SWD.nii.gz", "MRI_registered_to_CT_Landmark_Init.nii.gz"]:
            if f in local_files:
                mri_name = f; mri_uri = None; break
        if not mri_name:
            for f in local_files:
                if ("Original" in f and (mri_pattern in f or "T2" in f or "MR" in f)):
                    mri_name = f; mri_uri = None; break
            if not mri_name:
                for f in local_files:
                    if mri_pattern in f or (("T2" in f or "MR" in f) and "Registered" in f):
                        mri_name = f; mri_uri = None; break
    if not lesion_ct_name and "NNUNET_OR_Processed_Lesions_CT.seg.nrrd" in local_files:
        lesion_ct_name = "NNUNET_OR_Processed_Lesions_CT.seg.nrrd"
        lesion_ct_uri = None
    if not lesion_mr_name and "NNUNET_OR_Processed_Lesions_MR.seg.nrrd" in local_files:
        lesion_mr_name = "NNUNET_OR_Processed_Lesions_MR.seg.nrrd"
        lesion_mr_uri = None
    if not anat_tensor_name and "packed_anatomical_tensor.h5" in local_files:
        anat_tensor_name = "packed_anatomical_tensor.h5"
        anat_tensor_uri = None
    if not prostate_name:
        for f in ["prostate_segmentation.nii.gz", "MRI_Prostate_Mask_registered_SWD.nii.gz", "CT_Prostate_Mask.nii.gz"]:
            if f in local_files:
                prostate_name = f; prostate_uri = None; break
    if not tfm_name and "Transform_Landmark_SWD.h5" in local_files:
        tfm_name = "Transform_Landmark_SWD.h5"
        tfm_uri = None

    print("▶ Target components identified:")
    print(f"  CT:         {ct_name}")
    print(f"  PET SUV:    {suv_name}")
    print(f"  MRI Reg:    {mri_name}")
    print(f"  Lesion CT:  {lesion_ct_name}")
    print(f"  Lesion MR:  {lesion_mr_name}")
    print(f"  Anat H5:    {anat_tensor_name}")
    print(f"  Slicer MRB: {mrb_name}")
    print(f"  Prostate:   {prostate_name}")
    
    # 3. Download files if URLs available
    ct_path = os.path.join(raw_dir, ct_name) if ct_name else None
    suv_path = os.path.join(raw_dir, suv_name) if suv_name else None
    mri_path = os.path.join(raw_dir, mri_name) if mri_name else None
    anat_h5_path = os.path.join(raw_dir, anat_tensor_name) if anat_tensor_name else None
    
    if ct_uri:
        download_file_streaming(f"{BASE}{ct_uri}", ct_path)
    if suv_uri:
        download_file_streaming(f"{BASE}{suv_uri}", suv_path)
    if mri_uri:
        download_file_streaming(f"{BASE}{mri_uri}", mri_path)
    if anat_tensor_uri:
        download_file_streaming(f"{BASE}{anat_tensor_uri}", anat_h5_path)
    
    if lesion_ct_name:
        lesion_ct_path = os.path.join(raw_dir, lesion_ct_name)
        if lesion_ct_uri:
            download_file_streaming(f"{BASE}{lesion_ct_uri}", lesion_ct_path)
    else:
        lesion_ct_path = None
        
    if lesion_mr_name:
        lesion_mr_path = os.path.join(raw_dir, lesion_mr_name)
        if lesion_mr_uri:
            download_file_streaming(f"{BASE}{lesion_mr_uri}", lesion_mr_path)
    else:
        lesion_mr_path = None
        
    if mrb_uri:
        mrb_path = os.path.join(case_dir, f"{pat_id}.mrb")
        download_file_streaming(f"{BASE}{mrb_uri}", mrb_path)
        
    if prostate_uri:
        prostate_path = os.path.join(raw_dir, prostate_name)
        download_file_streaming(f"{BASE}{prostate_uri}", prostate_path)
    else:
        prostate_path = os.path.join(raw_dir, prostate_name) if prostate_name else None

    tfm_reg_path = None
    if tfm_uri:
        tfm_download_path = os.path.join(raw_dir, tfm_name)
        download_file_streaming(f"{BASE}{tfm_uri}", tfm_download_path)
        tfm_reg_path = tfm_download_path

    # Check local fallback transform paths if not downloaded or file missing
    if not tfm_reg_path or not os.path.isfile(tfm_reg_path):
        for candidate_tfm in [
            os.path.join(case_dir, "registration_output", "transforms", "Transform_Landmark_SWD.h5"),
            os.path.join(case_dir, "registration_output", "transforms", "Transform_Landmark_SWD.tfm"),
            os.path.join(case_dir, "registration_output", "transforms", "Transform_Landmark_Init.h5"),
            os.path.join(case_dir, "registration_output", "transforms", "Transform_FollowUp_to_Baseline_1.tfm"),
            os.path.join(case_dir, "Transform_FollowUp_to_Baseline_1.tfm"),
            os.path.join(raw_dir, "Transform_Landmark_SWD.h5"),
            os.path.join(raw_dir, "Transform_Landmark_Init.h5"),
            os.path.join(raw_dir, "Transform_FollowUp_to_Baseline_1.tfm"),
        ]:
            if os.path.isfile(candidate_tfm):
                tfm_reg_path = candidate_tfm
                break
    if tfm_reg_path and os.path.isfile(tfm_reg_path):
        print(f"  [REGISTRATION] Active transform source: {tfm_reg_path}")
    else:
        print("  [WARN] No registration transform file found for this case.")
        
    # 4. Load CT as reference geometry
    print("▶ Loading and aligning volumes to CT spatial reference...")
    ct_img = sitk.ReadImage(ct_path)
    spacing = ct_img.GetSpacing()  # (sx, sy, sz)
    origin = ct_img.GetOrigin()
    direction = ct_img.GetDirection()
    size = ct_img.GetSize()        # (X, Y, Z)
    ct_arr = sitk.GetArrayFromImage(ct_img).astype(np.float32)  # (Z, Y, X)
    Z, Y, X = ct_arr.shape
    print(f"  CT Grid: {X}x{Y}x{Z}, Spacing: ({spacing[0]:.3f}, {spacing[1]:.3f}, {spacing[2]:.3f}) mm")
    
    # Resample PET SUV
    suv_img = sitk.ReadImage(suv_path)
    res_suv = sitk.Resample(suv_img, ct_img, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    pet_arr = sitk.GetArrayFromImage(res_suv).astype(np.float32)
    print(f"  PET SUV resampled: shape={pet_arr.shape}, max={pet_arr.max():.2f}")
    
    # Resample Registered MRI (or zero volume if CT/PET-only case like Pat61)
    if mri_name and os.path.isfile(mri_path):
        mri_img = sitk.ReadImage(mri_path)
        mri_tfm = sitk.Transform()
        if mri_img.GetSize() != ct_img.GetSize() or mri_img.GetOrigin() != ct_img.GetOrigin():
            if os.path.isfile(tfm_reg_path):
                print(f"  Applying {os.path.basename(tfm_reg_path)} to MRI moving volume...")
                mri_tfm = sitk.ReadTransform(tfm_reg_path)
        res_mri = sitk.Resample(mri_img, ct_img, mri_tfm, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        mri_arr = sitk.GetArrayFromImage(res_mri).astype(np.float32)
        mri_arr = np.maximum(0.0, mri_arr)
        pos_mri = mri_arr[mri_arr > 0]
        if len(pos_mri) > 0:
            p99 = float(np.percentile(pos_mri, 99))
            if p99 > 1200.0 or p99 < 500.0:
                scale = 1000.0 / max(p99, 1.0)
                mri_arr = mri_arr * scale
                print(f"  MRI scaled to [0, 1000] standard window: p99 {p99:.1f} -> 1000.0 (scale {scale:.4f})")
        print(f"  MRI resampled: shape={mri_arr.shape}, non-zero voxels={(mri_arr > 0).sum()}, min={mri_arr.min():.1f}, max={mri_arr.max():.1f}")
    else:
        print("  [WARN] No MRI volume available for this case (CT/PET only). Initializing empty MRI volume.")
        mri_arr = np.zeros((Z, Y, X), dtype=np.float32)

    # 4b. Identify and resample all additional MRI modalities (ADC, DWI, T1, DCE, etc.)
    extra_mri_series = []
    candidate_mri_files = {}
    if os.path.exists(raw_dir):
        for fn in os.listdir(raw_dir):
            if fn == mri_name:
                continue
            res = classify_mri_modality(fn)
            if res and res[0] != "CT":
                candidate_mri_files[fn] = os.path.join(raw_dir, fn)
    if not offline:
        # Pre-rank MRI sequences to download only top unique modalities
        ranked_slicer_seqs = select_ranked_mri_sequences(slicer_files)
        for seq in ranked_slicer_seqs:
            fn = seq["fn"]
            if fn == mri_name or fn in candidate_mri_files:
                continue
            if fn in slicer_files:
                dst = os.path.join(raw_dir, fn)
                download_file_streaming(f"{BASE}{slicer_files[fn]}", dst)
                candidate_mri_files[fn] = dst

    # Auto-fetch ADC scan from XNAT MR session if missing in slicer_final
    has_adc = any(classify_mri_modality(fn) and classify_mri_modality(fn)[0] == "ADC" for fn in candidate_mri_files)
    if not has_adc and not offline:
        try:
            mr_exps = []
            sub_mr_url = f"{BASE}/data/subjects/{subj_id}/experiments?format=json"
            r_mr = requests.get(sub_mr_url, auth=AUTH, timeout=20)
            if r_mr.status_code == 200:
                data = r_mr.json()
                items = data.get("items", []) if "items" in data else []
                if items and "children" in items[0]:
                    for child in items[0]["children"]:
                        for it in child.get("items", []):
                            fields = it.get("data_fields", {})
                            lbl = fields.get("label", "")
                            if "MRI" in lbl or "MR" in lbl or "mrSession" in it.get("meta", {}).get("xsi:type", ""):
                                mr_exps.append(fields.get("ID"))
            if not mr_exps:
                r_all_exp = requests.get(f"{BASE}/data/experiments?format=json", auth=AUTH, timeout=20)
                if r_all_exp.status_code == 200:
                    for exp in r_all_exp.json().get("ResultSet", {}).get("Result", []):
                        lbl = exp.get("label", "")
                        if pat_id in lbl and ("MRI" in lbl or "mrSession" in exp.get("xsiType", "")):
                            mr_exps.append(exp.get("ID"))
            for mr_eid in mr_exps:
                if not mr_eid:
                    continue
                r_scans = requests.get(f"{BASE}/data/experiments/{mr_eid}/scans?format=json", auth=AUTH, timeout=20)
                if r_scans.status_code == 200:
                    for s in r_scans.json().get("ResultSet", {}).get("Result", []):
                        s_desc = s.get("series_description", "")
                        if "adc" in s_desc.lower():
                            scan_id = s.get("ID")
                            print(f"  [XNAT-ADC] Found ADC scan {scan_id} ({s_desc}) in {mr_eid}. Downloading zip...")
                            zip_url = f"{BASE}/data/experiments/{mr_eid}/scans/{scan_id}/files?format=zip"
                            r_zip = requests.get(zip_url, auth=AUTH, stream=True, timeout=90)
                            if r_zip.status_code == 200:
                                with tempfile.TemporaryDirectory() as tmp_adc_dir:
                                    with zipfile.ZipFile(io.BytesIO(r_zip.content)) as z_adc:
                                        z_adc.extractall(tmp_adc_dir)
                                    dcm_target = None
                                    for root, _, files in os.walk(tmp_adc_dir):
                                        if any(f.startswith("I000") or f.endswith(".dcm") for f in files):
                                            dcm_target = root
                                            break
                                    if dcm_target:
                                        dcm_reader = sitk.ImageSeriesReader()
                                        dcm_series = dcm_reader.GetGDCMSeriesFileNames(dcm_target)
                                        if dcm_series:
                                            dcm_reader.SetFileNames(dcm_series)
                                            adc_sitk = dcm_reader.Execute()
                                            adc_fn = "IsoADC_Original_for_Comparison.nii.gz"
                                            adc_dst = os.path.join(raw_dir, adc_fn)
                                            sitk.WriteImage(adc_sitk, adc_dst)
                                            candidate_mri_files[adc_fn] = adc_dst
                                            print(f"  [XNAT-ADC] Successfully saved {adc_fn} to raw_dir!")
                                            has_adc = True
                                            break
                if has_adc:
                    break
        except Exception as e_adc:
            print(f"  [WARN] Failed to fetch ADC scan from XNAT: {e_adc}")

    ranked_extra = select_ranked_mri_sequences(candidate_mri_files)
    if mri_name and os.path.isfile(mri_path):
        ranked_extra = [x for x in ranked_extra if not (x["mod"] == "T2" and x["plane"] == "ax")]

    if ranked_extra:
        print(f"▶ Additional MRI modalities identified: {[x['mod'] + ' (' + x['fn'] + ')' for x in ranked_extra]}")
        tfm_shared = None
        if tfm_reg_path and os.path.isfile(tfm_reg_path):
            try:
                tfm_shared = sitk.ReadTransform(tfm_reg_path)
                print(f"  [REGISTRATION] Loaded {os.path.basename(tfm_reg_path)} for multi-modal MRI alignment")
            except Exception as e_tfm:
                print(f"  [WARN] Failed to read {tfm_reg_path}: {e_tfm}")

        for i, seq in enumerate(ranked_extra):
            tp_i = i + 2
            p = seq["item"]
            print(f"  Resampling TP {tp_i}: {seq['mod']} ({seq['fn']})...")
            try:
                im = sitk.ReadImage(p)
                # All MRI modalities were acquired in the same scanner session and are inherently registered.
                # Apply the single shared MRI-to-CT registration transform directly to each raw MRI modality.
                tfm_to_use = tfm_shared

                # Handle flattened 4D Dynamic/DCE acquisitions (e.g. 414 slices = 18 phases * 23 slices)
                if (seq["mod"] in ("DCE", "DYNAMIC") or "dynamik" in seq["fn"].lower()) and im.GetSize()[2] > 100:
                    raw_arr = sitk.GetArrayFromImage(im)
                    total_z = raw_arr.shape[0]
                    slices_per_phase = 23 if total_z % 23 == 0 else (18 if total_z % 18 == 0 else 23)
                    n_phases = total_z // slices_per_phase
                    target_phase = min(4, n_phases - 1)
                    phase_arr = raw_arr[target_phase * slices_per_phase : (target_phase + 1) * slices_per_phase]
                    dce_clean = sitk.GetImageFromArray(phase_arr)
                    dce_clean.SetSpacing((im.GetSpacing()[0], im.GetSpacing()[1], 3.0))
                    # In NIfTI, the origin Z is at the top slice, so the bottom slice (slice 0) origin Z is:
                    z_origin = im.GetOrigin()[2] - (slices_per_phase - 1) * 3.0
                    dce_clean.SetOrigin((im.GetOrigin()[0], im.GetOrigin()[1], z_origin))
                    dce_clean.SetDirection(im.GetDirection())
                    im = dce_clean
                    print(f"  [DCE] Extracted peak arterial phase {target_phase}/{n_phases} ({slices_per_phase} slices, 3.0mm spacing, origin Z={z_origin:.2f}mm)")

                if im.GetSize() == ct_img.GetSize() and im.GetOrigin() == ct_img.GetOrigin():
                    res_im = im
                elif tfm_to_use is not None:
                    res_im = sitk.Resample(im, ct_img, tfm_to_use, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
                else:
                    res_im = sitk.Resample(im, ct_img, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
                arr = sitk.GetArrayFromImage(res_im).astype(np.float32)
                arr = np.maximum(0.0, arr)
                pos = arr[arr > 0]
                if len(pos) > 0:
                    p99 = float(np.percentile(pos, 99))
                    if seq["mod"] == "ADC":
                        if p99 > 3000.0 or p99 < 200.0:
                            scale = 1500.0 / max(p99, 1.0)
                            arr = arr * scale
                    else:
                        if p99 > 1200.0 or p99 < 500.0:
                            scale = 1000.0 / max(p99, 1.0)
                            arr = arr * scale
                extra_mri_series.append({
                    "tp_idx": tp_i,
                    "mod": seq["mod"],
                    "plane": seq["plane"],
                    "fn": seq["fn"],
                    "arr": arr,
                    "desc": f"{seq['mod']} MRI Registered ({seq['plane'].upper()})"
                })
            except Exception as ex_seq:
                print(f"  [WARN] Failed to resample {seq['fn']}: {ex_seq}")
    
    # 5. Extract lymph node stations and unpack layered max_anatomy
    print("▶ Extracting lymph node stations (All_Lymph_Node_Areas / final_results_gpu)...")
    ln_masks = load_lymph_node_stations(case_dir, raw_dir, Z, Y, X, ct_img, exp_id)
    print(f"  Loaded {len(ln_masks)} lymph node stations.")
    
    print("▶ Unpacking hierarchical anatomical structures into multi-layer max_anatomy...")
    max_anatomy, anatomy_label_names, bone_ids, bone_mask, skellytour, exclusion_mask = unpack_anatomy(
        anat_h5_path, ln_masks, Z, Y, X
    )
    print(f"  Extracted {len(anatomy_label_names)} anatomical structures into layered max_anatomy.")
    print(f"  Identified {len(bone_ids)} bone label IDs; bone voxels: {(bone_mask > 0).sum()}.")
    print(f"  Identified {(exclusion_mask > 0).sum()} exclusion voxels for physiological uptake suppression.")
    
    # 6. Resample Lesions and filter out physiological uptake
    lesion_ct_arr = np.zeros((Z, Y, X), dtype=np.int16)
    ct_nrrd_names = {}
    if lesion_ct_path and os.path.isfile(lesion_ct_path):
        l_img = sitk.ReadImage(lesion_ct_path)
        for k in l_img.GetMetaDataKeys():
            if k.endswith("_LabelValue"):
                pfx = k[:-len("_LabelValue")]
                val = int(l_img.GetMetaData(k))
                ct_nrrd_names[val] = l_img.GetMetaData(f"{pfx}_Name") if l_img.HasMetaDataKey(f"{pfx}_Name") else ""
        res_l = sitk.Resample(l_img, ct_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
        raw_l = sitk.GetArrayFromImage(res_l).astype(np.int16)
        # Suppress physiological uptake: by NRRD name and by spatial exclusion
        for val, name in ct_nrrd_names.items():
            if any(t in name.lower() for t in ["bladder", "parotid", "submandibular", "lacrimal", "kidney"]):
                raw_l[raw_l == val] = 0
        raw_l[exclusion_mask > 0] = 0
        lesion_ct_arr = raw_l
        print(f"  CT Lesions resampled & cleaned: {(lesion_ct_arr > 0).sum()} voxels, unique labels: {np.unique(lesion_ct_arr)}")

    # 7. Resample MRI Anatomy & Native MR Segmentations
    # MRI anatomy contains the multi-zone prostate segmentation (PZ, TZ, CZ, AFS, UR, PG, SV_L, SV_R)
    # in registered CT coordinate space, strictly independent of CT whole-body anatomy.
    mri_anatomy = np.zeros((Z, Y, X), dtype=np.uint16)
    mri_mask = (mri_arr > 0)
    
    prostate_labels = {
        "1": "Peripheral Zone (PZ)",
        "2": "Transition Zone (TZ)",
        "3": "Central Zone (CZ)",
        "4": "Anterior Fibromuscular Stroma (AFS)",
        "5": "Urethra (UR)",
        "6": "Periurethral Glands (PG)",
        "7": "Seminal Vesicle Left (SV_L)",
        "8": "Seminal Vesicle Right (SV_R)"
    }
    dl_mr_lesion_mask = None
    if prostate_path and os.path.isfile(prostate_path):
        p_img = sitk.ReadImage(prostate_path)
        p_tfm = sitk.Transform()
        if p_img.GetSize() != ct_img.GetSize() or p_img.GetOrigin() != ct_img.GetOrigin():
            if os.path.isfile(tfm_reg_path):
                print(f"  Applying {os.path.basename(tfm_reg_path)} to prostate mask...")
                p_tfm = sitk.ReadTransform(tfm_reg_path)
        res_p = sitk.Resample(p_img, ct_img, p_tfm, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
        p_arr = sitk.GetArrayFromImage(res_p).astype(np.uint16)
        # Retain individual prostate anatomical zones (labels 1-8)
        for zid in [1, 2, 3, 4, 5, 6, 7, 8]:
            mri_anatomy[p_arr == zid] = zid
        print(f"  MRI Prostate Anatomy loaded from {os.path.basename(prostate_path)}: unique labels={np.unique(mri_anatomy)}, non-zero={(mri_anatomy > 0).sum()} voxels")
    else:
        # Fallback: extract nvseg_prostate or prostate from packed_tensor within MRI FOV
        p_tid = 1
        with h5py.File(anat_h5_path, "r") as af:
            packed_tensor = af["packed_tensor"][:]
            registry = load_anat_registry(anat_h5_path, af)
            for p_key in ["nvseg_prostate", "prostate"]:
                if p_key in registry:
                    meta = registry[p_key]
                    ch = meta["channel"] - 1
                    mid = meta["id"]
                    if 0 <= ch < packed_tensor.shape[0]:
                        m = (packed_tensor[ch] == mid)
                        mri_anatomy[m] = p_tid
                    break
        prostate_labels = {"1": "Prostate"}
        print(f"  MRI Prostate Anatomy extracted from nvseg_prostate: {(mri_anatomy == p_tid).sum()} voxels")

    # 7c. Build MR Lesions (Study B) — include all verified pelvic lesions in registered coordinate space
    lesion_mr_arr = lesion_ct_arr.copy()
    mr_nrrd_names = dict(ct_nrrd_names)
    print(f"  Final MR Lesions on CT grid: {(lesion_mr_arr > 0).sum()} voxels, unique labels: {np.unique(lesion_mr_arr)}")

    # Export standard ITK Transform file alongside preprocessed volumes
    if os.path.isfile(tfm_reg_path):
        tfm_out_path = os.path.join(case_dir, "Transform_FollowUp_to_Baseline_1.tfm")
        try:
            itk_tfm = sitk.ReadTransform(tfm_reg_path)
            sitk.WriteTransform(itk_tfm, tfm_out_path)
            print(f"  Exported {os.path.basename(tfm_out_path)} from {os.path.basename(tfm_reg_path)}")
        except Exception as e:
            print(f"  [WARN] Failed to export tfm: {e}")
        
    # 8. Dynamic per-lesion organ mapping
    ct_labels = [int(l) for l in np.unique(lesion_ct_arr) if l > 0]
    if not ct_labels:
        ct_labels = [1]
        
    mr_labels = [int(l) for l in np.unique(lesion_mr_arr) if l > 0]
    if not mr_labels:
        mr_labels = [1]

    def find_best_organ_for_lesion(lesion_vol, lid, nrrd_name):
        pts = np.argwhere(lesion_vol == lid)
        if len(pts) == 0:
            return "pelvic_lymph_node"
        vox_vals = max_anatomy[lesion_vol == lid]
        nz = vox_vals[vox_vals > 0]
        if len(nz) > 0:
            vals, counts = np.unique(nz, return_counts=True)
            ranked = []
            for v, c in zip(vals, counts):
                norm = anatomy_label_names.get(str(v), "")
                lay = get_layer(norm)
                ranked.append((lay, c, norm))
            ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
            return ranked[0][2]
        
        # Expanding search around centroid
        cz, cy, cx = np.round(pts.mean(axis=0)).astype(int)
        for r in [2, 4, 8, 16, 24]:
            z0, z1 = max(0, cz - r), min(Z, cz + r + 1)
            y0, y1 = max(0, cy - r), min(Y, cy + r + 1)
            x0, x1 = max(0, cx - r), min(X, cx + r + 1)
            sub = max_anatomy[z0:z1, y0:y1, x0:x1]
            nz_sub = sub[sub > 0]
            if len(nz_sub) > 0:
                vals, counts = np.unique(nz_sub, return_counts=True)
                ranked = []
                for v, c in zip(vals, counts):
                    norm = anatomy_label_names.get(str(v), "")
                    lay = get_layer(norm)
                    ranked.append((lay, c, norm))
                ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
                return ranked[0][2]
                
        # Check NRRD name
        if nrrd_name:
            m = re.search(r"NNUNET_OR_TP_(.*)_(\d+)_(CT|MR)", nrrd_name)
            if m and "unknown" not in m.group(1):
                return m.group(1).lower()
        return "pelvic_lymph_node"

    organ_mapping = {"0": {}, "1": {}}
    for lid in ct_labels:
        raw_name = ct_nrrd_names.get(lid, "")
        org = find_best_organ_for_lesion(lesion_ct_arr, lid, raw_name)
        disp = org.replace("_", " ").title()
        organ_mapping["0"][str(lid)] = disp

    for lid in mr_labels:
        raw_name = mr_nrrd_names.get(lid, "")
        org = find_best_organ_for_lesion(lesion_mr_arr, lid, raw_name)
        disp = org.replace("_", " ").title()
        organ_mapping["1"][str(lid)] = disp

    print("  Mapped CT lesions to organs:", organ_mapping["0"])
    print("  Mapped MR lesions to organs:", organ_mapping["1"])

    # 9. Compute champion diffusivity fields
    diff_arr = compute_champion_diffusivity(ct_arr, pet_arr, exclusion_mask)
    mri_diff_arr = compute_mri_diffusivity(mri_arr, pet_arr=pet_arr, anat_vol=mri_anatomy, exclusion_mask=exclusion_mask, modality="T2")
    for s in extra_mri_series:
        s["diff"] = compute_mri_diffusivity(s["arr"], pet_arr=pet_arr, anat_vol=mri_anatomy, exclusion_mask=exclusion_mask, modality=s["mod"])
    
    # 10. Pre-flip ALL volumes along Y (axis 1) for standard radiological orientation
    print("▶ Pre-flipping all volumes along Y (axis 1) for MedEye3d standard orientation...")
    ct_arr = np.flip(ct_arr, axis=1)
    pet_arr = np.flip(pet_arr, axis=1)
    mri_arr = np.flip(mri_arr, axis=1)
    lesion_ct_arr = np.flip(lesion_ct_arr, axis=1)
    lesion_mr_arr = np.flip(lesion_mr_arr, axis=1)
    max_anatomy = np.flip(max_anatomy, axis=1)
    mri_anatomy = np.flip(mri_anatomy, axis=1)
    diff_arr = np.flip(diff_arr, axis=1)
    mri_diff_arr = np.flip(mri_diff_arr, axis=1)
    bone_atlas = np.flip(bone_mask, axis=1)
    skellytour = np.flip(skellytour, axis=1)
    for s in extra_mri_series:
        s["arr"] = np.flip(s["arr"], axis=1)
        s["diff"] = np.flip(s["diff"], axis=1)
    
    # 11. Assemble HDF5 Package
    h5_target = os.path.join(case_dir, "preprocessed_volumes.h5")
    print(f"▶ Assembling final HDF5 package: {h5_target}...")
    
    ct_segment_names = [f"Lesion {lid}: {organ_mapping['0'].get(str(lid), '')}" for lid in ct_labels]
    mr_segment_names = [f"Lesion {lid}: {organ_mapping['1'].get(str(lid), '')}" for lid in mr_labels]
    
    segment_names = {
        "0": {str(lid): ct_segment_names[i] for i, lid in enumerate(ct_labels)},
        "1": {str(lid): mr_segment_names[i] for i, lid in enumerate(mr_labels)}
    }
    
    # Parse global lesion numbers from NRRD names
    def parse_global_num(name):
        m = re.search(r"_(\d+)_(?:CT|MR)$", name)
        return int(m.group(1)) if m else None

    ct_global_nums = {lid: parse_global_num(ct_nrrd_names.get(lid, "")) for lid in ct_labels}
    mr_global_nums = {lid: parse_global_num(mr_nrrd_names.get(lid, "")) for lid in mr_labels}
    num_to_ct = {num: lid for lid, num in ct_global_nums.items() if num is not None}
    
    def get_centroid(mask, lid):
        pts = np.argwhere(mask == lid)
        return pts.mean(axis=0) if len(pts) > 0 else None
        
    ct_cents = {lid: get_centroid(lesion_ct_arr, lid) for lid in ct_labels}
    mr_cents = {lid: get_centroid(lesion_mr_arr, lid) for lid in mr_labels}
    
    matches = []
    matched_mr = set()
    matched_ct = set()
    current_gid = 1
    
    # 0. Match by exact label ID (direct correspondence between CT and MRI)
    for m_id in mr_labels:
        if m_id in ct_labels and m_id not in matched_mr and m_id not in matched_ct:
            matched_ct.add(m_id)
            matched_mr.add(m_id)
            matches.append({
                "name": "PET_Lesions_0",
                "raw_lesion": f"Segment_{m_id}",
                "node": "parent",
                "group_id": current_gid,
                "lesion": organ_mapping["0"].get(str(m_id), f"Lesion {m_id}"),
                "children": [
                    {
                        "name": "PET_Lesions_1",
                        "raw_lesion": f"Segment_{m_id}",
                        "node": "child",
                        "group_id": current_gid,
                        "lesion": organ_mapping["1"].get(str(m_id), f"Lesion {m_id}"),
                        "match_type": "EXACT_ID"
                    }
                ]
            })
            current_gid += 1

    # 1. Match by exact global number
    for m_id in mr_labels:
        if m_id in matched_mr:
            continue
        g_num = mr_global_nums.get(m_id)
        if g_num is not None and g_num in num_to_ct:
            c_id = num_to_ct[g_num]
            if c_id in matched_ct:
                continue
            matched_ct.add(c_id)
            matched_mr.add(m_id)
            matches.append({
                "name": "PET_Lesions_0",
                "raw_lesion": f"Segment_{c_id}",
                "node": "parent",
                "group_id": current_gid,
                "lesion": organ_mapping["0"].get(str(c_id), f"Lesion {c_id}"),
                "children": [
                    {
                        "name": "PET_Lesions_1",
                        "raw_lesion": f"Segment_{m_id}",
                        "node": "child",
                        "group_id": current_gid,
                        "lesion": organ_mapping["1"].get(str(m_id), f"Lesion {m_id}"),
                        "match_type": "GLOBAL_ID"
                    }
                ]
            })
            current_gid += 1
            
    # 2. Proximity fallback for remaining unmapped lesions (within 15 voxels)
    for c_id in ct_labels:
        if c_id in matched_ct:
            continue
        c_cent = ct_cents.get(c_id)
        best_m = None
        best_dist = 999.0
        if c_cent is not None:
            for m_id in mr_labels:
                if m_id in matched_mr:
                    continue
                m_cent = mr_cents.get(m_id)
                if m_cent is not None:
                    dist_vox = np.linalg.norm(c_cent - m_cent)
                    if dist_vox < 15.0 and dist_vox < best_dist:
                        best_dist = dist_vox
                        best_m = m_id
                        
        if best_m is not None:
            matched_ct.add(c_id)
            matched_mr.add(best_m)
            matches.append({
                "name": "PET_Lesions_0",
                "raw_lesion": f"Segment_{c_id}",
                "node": "parent",
                "group_id": current_gid,
                "lesion": organ_mapping["0"].get(str(c_id), f"Lesion {c_id}"),
                "children": [
                    {
                        "name": "PET_Lesions_1",
                        "raw_lesion": f"Segment_{best_m}",
                        "node": "child",
                        "group_id": current_gid,
                        "lesion": organ_mapping["1"].get(str(best_m), f"Lesion {best_m}"),
                        "match_type": "PROXIMITY"
                    }
                ]
            })
            current_gid += 1
        else:
            # CT only lesion
            matches.append({
                "name": "PET_Lesions_0",
                "raw_lesion": f"Segment_{c_id}",
                "node": "parent",
                "group_id": current_gid,
                "lesion": organ_mapping["0"].get(str(c_id), f"Lesion {c_id}"),
                "children": []
            })
            current_gid += 1
            
    # 3. Any remaining MR only lesions
    for m_id in mr_labels:
        if m_id not in matched_mr:
            matches.append({
                "name": "PET_Lesions_1",
                "raw_lesion": f"Segment_{m_id}",
                "node": "parent",
                "group_id": current_gid,
                "lesion": organ_mapping["1"].get(str(m_id), f"Lesion {m_id}"),
                "children": []
            })
            current_gid += 1
    
    scene_hierarchy = [
        {
            "name": "Fixed_CT_Volume_0",
            "type": "vtkMRMLScalarVolumeNode",
            "modality": "CT"
        },
        {
            "name": "SUV_PET_Image_0",
            "type": "vtkMRMLScalarVolumeNode",
            "modality": "PET"
        },
        {
            "name": "PET_Lesions_0",
            "type": "vtkMRMLSegmentationNode",
            "segments": ct_segment_names
        },
        {
            "name": "max_anatomy_fixed_ct_0",
            "source": "max_anatomy.nii.gz",
            "type": "vtkMRMLSegmentationNode"
        },
        {
            "name": "skellytour_fixed_ct_0",
            "source": "skellytour.nii.gz",
            "type": "vtkMRMLSegmentationNode"
        },
        {
            "name": "Transform_FollowUp_to_Baseline_1",
            "type": "vtkMRMLLinearTransformNode",
            "transform_type": "Rigid/Affine (Linear)",
            "modality": "T2",
            "children": [
                {
                    "name": "Fixed_CT_Volume_1",
                    "type": "vtkMRMLScalarVolumeNode",
                    "modality": "T2"
                },
                {
                    "name": "SUV_PET_Image_1",
                    "type": "vtkMRMLScalarVolumeNode",
                    "modality": "PET"
                },
                {
                    "name": "PET_Lesions_1",
                    "type": "vtkMRMLSegmentationNode",
                    "segments": mr_segment_names
                },
                {
                    "name": "max_anatomy_fixed_ct_1",
                    "source": "max_anatomy.nii.gz",
                    "type": "vtkMRMLSegmentationNode"
                }
            ]
        }
    ]
    
    for s in extra_mri_series:
        scene_hierarchy.append({
            "name": f"Transform_FollowUp_to_Baseline_{s['tp_idx']}",
            "type": "vtkMRMLLinearTransformNode",
            "transform_type": "Rigid/Affine (Linear)",
            "modality": s["mod"],
            "children": [
                {
                    "name": f"Fixed_CT_Volume_{s['tp_idx']}",
                    "type": "vtkMRMLScalarVolumeNode",
                    "modality": s["mod"],
                    "series_description": s["desc"]
                },
                {
                    "name": f"SUV_PET_Image_{s['tp_idx']}",
                    "type": "vtkMRMLScalarVolumeNode",
                    "modality": "PET"
                },
                {
                    "name": f"PET_Lesions_{s['tp_idx']}",
                    "type": "vtkMRMLSegmentationNode",
                    "segments": mr_segment_names
                },
                {
                    "name": f"max_anatomy_fixed_ct_{s['tp_idx']}",
                    "source": "max_anatomy.nii.gz",
                    "type": "vtkMRMLSegmentationNode"
                }
            ]
        })
    
    patient_info = {
        "PatientID": pat_id,
        "SubjectID": subj_id,
        "ExperimentID": exp_id,
        "Modalities": ["CT", "PET", "MRI"],
        "CreatedDate": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ReferenceSpacing": [float(s) for s in spacing],
        "ReferenceOrigin": [float(o) for o in origin]
    }
    
    # Load and associate radiological descriptions from Excel if available
    metadata = [patient_info]
    excel_path = "/workspaces/MedEye3d.jl/data/Table MRT_PET october26.xlsx" if os.path.isfile("/workspaces/MedEye3d.jl/data/Table MRT_PET october26.xlsx") else "/workspaces/MedEye3d.jl/data/Table MRT_PET august26.xlsx"
    if os.path.isfile(excel_path):
        try:
            from associate_radiological_descriptions import load_patient_reports, get_translator, parse_date
            reports_db = load_patient_reports(excel_path)
            pid_m = re.search(r"(\d+)", pat_id)
            pid_int = int(pid_m.group(1)) if pid_m else None
            if pid_int and pid_int in reports_db:
                rec = reports_db[pid_int]
                patient_info["BirthdayDate"] = rec.get("dob", "")
                psma_en = rec.get("psma_report", "")
                mri_en = rec.get("mri_report", "")
                if not mri_en or mri_en.lower() in ["none", "nan"]:
                    mri_en = "keine Angaben"
                pet_date = parse_date(psma_en, default="20230831")
                mri_date = parse_date(mri_en, default=f"{int(pet_date[:8])+1}")
                translator = get_translator()
                psma_de = translator(psma_en)
                mri_de = translator(mri_en)
                
                pet_entry = {
                    f"{pet_date}_PET": {
                        "Description": psma_de,
                        "EnglishDescription": psma_en,
                        "Modality": "PET",
                        "PET": {"name": "SUV_PET_Image_0", "Modality": "PET", "StudyDate": pet_date, "SeriesDescription": "PET PSMA"},
                        "CT": {"name": "Fixed_CT_Volume_0", "Modality": "CT", "StudyDate": pet_date, "SeriesDescription": "CT AC"}
                    }
                }
                mri_entry = {
                    f"{mri_date}_MRI": {
                        "Description": mri_de,
                        "EnglishDescription": mri_en,
                        "Modality": "T2",
                        "MRI": {"name": "Fixed_CT_Volume_1", "Modality": "T2", "StudyDate": mri_date, "SeriesDescription": "T2 MRI Registered"},
                        "PET": {"name": "SUV_PET_Image_1", "Modality": "PET", "StudyDate": mri_date}
                    }
                }
                metadata.append(pet_entry)
                metadata.append(mri_entry)
                print(f"  [METADATA] Associated radiological reports: PSMA ({len(psma_en)} chars EN, {len(psma_de)} chars DE), MRI ({len(mri_en)} chars EN, {len(mri_de)} chars DE)")
        except Exception as ex_meta:
            print(f"  [WARN] Failed to associate radiological descriptions: {ex_meta}")
            metadata = [patient_info]
    
    # Associate multi-modal MRI metadata, segment names, organ mappings, and matches
    mri_desc_de = mri_de if 'mri_de' in locals() else "mpMRI"
    mri_desc_en = mri_en if 'mri_en' in locals() else "mpMRI"
    mri_date_str = mri_date if 'mri_date' in locals() else "20230831"
    for s in extra_mri_series:
        tp_i = s["tp_idx"]
        if isinstance(metadata, list):
            metadata.append({
                f"{mri_date_str}_MRI_{s['mod']}_{tp_i}": {
                    "Description": mri_desc_de,
                    "EnglishDescription": mri_desc_en,
                    "Modality": s["mod"],
                    "MRI": {
                        "name": f"Fixed_CT_Volume_{tp_i}",
                        "Modality": s["mod"],
                        "StudyDate": mri_date_str,
                        "SeriesDescription": s["desc"]
                    },
                    "PET": {"name": f"SUV_PET_Image_{tp_i}", "Modality": "PET"}
                }
            })
        segment_names[str(tp_i)] = segment_names.get("1", {})
        organ_mapping[str(tp_i)] = organ_mapping.get("1", {})

    for m in matches:
        if m.get("children"):
            base_c = m["children"][0]
            new_children = []
            for tp_i in [1] + [s["tp_idx"] for s in extra_mri_series]:
                c = dict(base_c)
                c["name"] = f"PET_Lesions_{tp_i}"
                new_children.append(c)
            m["children"] = new_children

    # Save HDF5
    with h5py.File(h5_target, "w") as h5:
        # BASELINE Group (TP 0)
        g_base = h5.create_group("BASELINE")
        ds_ct0 = g_base.create_dataset("Fixed_CT_Volume_0.nii.gz", data=ct_arr, compression="gzip", compression_opts=1)
        ds_ct0.attrs["spacing"] = np.array(spacing, dtype=np.float64)
        ds_ct0.attrs["origin"] = np.array(origin, dtype=np.float64)
        ds_ct0.attrs["direction"] = np.array(direction, dtype=np.float64)
        
        ds_pet0 = g_base.create_dataset("SUV_PET_Image_0.nii.gz", data=pet_arr, compression="gzip", compression_opts=1)
        ds_pet0.attrs["spacing"] = np.array(spacing, dtype=np.float64)
        
        ds_l0 = g_base.create_dataset("PET_Lesions_0.nii.gz", data=lesion_ct_arr, compression="gzip", compression_opts=1)
        ds_l0.attrs["spacing"] = np.array(spacing, dtype=np.float64)
        
        g_base.create_dataset("diffusivity", data=diff_arr, compression="gzip", compression_opts=1)
        g_base.create_dataset("max_anatomy.nii.gz", data=max_anatomy, compression="gzip", compression_opts=1)
        g_base.create_dataset("skellytour.nii.gz", data=skellytour, compression="gzip", compression_opts=1)
        
        # MRI Group (TP 1)
        g_tfm = h5.create_group("TFM_Transform_FollowUp_to_Baseline_1.tfm")
        ds_mri1 = g_tfm.create_dataset("Fixed_CT_Volume_1.nii.gz", data=mri_arr, compression="gzip", compression_opts=1)
        ds_mri1.attrs["spacing"] = np.array(spacing, dtype=np.float64)
        ds_mri1.attrs["modality"] = "T2"
        ds_mri1.attrs["series_description"] = "T2 MRI Registered"
        
        ds_pet1 = g_tfm.create_dataset("SUV_PET_Image_1.nii.gz", data=pet_arr, compression="gzip", compression_opts=1)
        ds_pet1.attrs["spacing"] = np.array(spacing, dtype=np.float64)
        
        ds_l1 = g_tfm.create_dataset("PET_Lesions_1.nii.gz", data=lesion_mr_arr, compression="gzip", compression_opts=1)
        ds_l1.attrs["spacing"] = np.array(spacing, dtype=np.float64)
        
        g_tfm.create_dataset("diffusivity", data=mri_diff_arr, compression="gzip", compression_opts=1)
        g_tfm.create_dataset("max_anatomy.nii.gz", data=mri_anatomy, compression="gzip", compression_opts=1)
        
        # Additional MRI Groups (TP 2, TP 3, TP 4, etc.)
        for s in extra_mri_series:
            tp_i = s["tp_idx"]
            g_extra = h5.create_group(f"TFM_Transform_FollowUp_to_Baseline_{tp_i}.tfm")
            ds_mri_e = g_extra.create_dataset(f"Fixed_CT_Volume_{tp_i}.nii.gz", data=s["arr"], compression="gzip", compression_opts=1)
            ds_mri_e.attrs["spacing"] = np.array(spacing, dtype=np.float64)
            ds_mri_e.attrs["modality"] = s["mod"]
            ds_mri_e.attrs["series_description"] = s["desc"]

            ds_pet_e = g_extra.create_dataset(f"SUV_PET_Image_{tp_i}.nii.gz", data=pet_arr, compression="gzip", compression_opts=1)
            ds_pet_e.attrs["spacing"] = np.array(spacing, dtype=np.float64)

            ds_l_e = g_extra.create_dataset(f"PET_Lesions_{tp_i}.nii.gz", data=lesion_mr_arr, compression="gzip", compression_opts=1)
            ds_l_e.attrs["spacing"] = np.array(spacing, dtype=np.float64)

            g_extra.create_dataset("diffusivity", data=s["diff"], compression="gzip", compression_opts=1)
            g_extra.create_dataset("max_anatomy.nii.gz", data=mri_anatomy, compression="gzip", compression_opts=1)
            print(f"  Packaged multi-modal MRI TP {tp_i}: {s['mod']} ({s['desc']})")

        # ATLAS Group
        g_atlas = h5.create_group("ATLAS")
        g_atlas.create_dataset("max_anatomy", data=max_anatomy, compression="gzip", compression_opts=1)
        g_atlas.create_dataset("bone_atlas", data=bone_atlas, compression="gzip", compression_opts=1)
        g_atlas.create_dataset("skellytour", data=skellytour, compression="gzip", compression_opts=1)
        
        # Centroids computed from the pre-flipped lesion masks
        g_cent = h5.create_group("CENTROIDS")
        for lid in ct_labels:
            pts = np.argwhere(lesion_ct_arr == lid)
            if len(pts) > 0:
                mean_pt = np.round(pts.mean(axis=0)).astype(np.int32)
                # MedEye3d 1-based indexing in [X, Y, Z]
                g_cent.create_dataset(f"tp0_lid{lid}", data=np.array([mean_pt[2]+1, mean_pt[1]+1, mean_pt[0]+1], dtype=np.int32))
                
        for lid in mr_labels:
            pts = np.argwhere(lesion_mr_arr == lid)
            if len(pts) > 0:
                mean_pt = np.round(pts.mean(axis=0)).astype(np.int32)
                g_cent.create_dataset(f"tp1_lid{lid}", data=np.array([mean_pt[2]+1, mean_pt[1]+1, mean_pt[0]+1], dtype=np.int32))
                for s in extra_mri_series:
                    g_cent.create_dataset(f"tp{s['tp_idx']}_lid{lid}", data=np.array([mean_pt[2]+1, mean_pt[1]+1, mean_pt[0]+1], dtype=np.int32))
                
        # Meta Group
        g_meta = h5.create_group("_meta_")
        g_meta.create_dataset("preflipped", data=1)
        g_meta.create_dataset("bone_label_ids", data=np.array(bone_ids, dtype=np.int32))
        g_meta.create_dataset("scene_hierarchy.json", data=json.dumps(scene_hierarchy, indent=2))
        g_meta.create_dataset("metadata.json", data=json.dumps(metadata, indent=2))
        g_meta.create_dataset("max_anatomy_labels.json", data=json.dumps(anatomy_label_names, indent=2))
        g_meta.create_dataset("prostate_anatomy_labels.json", data=json.dumps(prostate_labels, indent=2))
        g_meta.create_dataset("organ_mapping", data=json.dumps(organ_mapping, indent=2))
        g_meta.create_dataset("segment_names.json", data=json.dumps(segment_names, indent=2))
        g_meta.create_dataset("matches.json", data=json.dumps(matches, indent=2))
        
    print(f"✅ CASE {pat_id} PREPROCESSED SUCCESSFULLY!")
    print(f"  HDF5 size: {round(os.path.getsize(h5_target)/(1024*1024), 2)} MB")
    return h5_target

def main():
    out_dir = "/workspaces/MedEye3d.jl/data/cases/Prostata_bimodal"
    os.makedirs(out_dir, exist_ok=True)
    
    offline = "--offline" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--offline"]
    specific_case = args[0] if len(args) > 0 else None
    
    results = {}
    for case in TARGET_CASES:
        if specific_case and case["patient_id"] != specific_case:
            continue
        try:
            h5_path = process_case(case, out_dir, offline=offline)
            results[case["patient_id"]] = h5_path
        except Exception as e:
            print(f"❌ Error processing {case['patient_id']}: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n" + "=" * 70)
    print(" SUMMARY OF PREPARED BIMODAL CASES:")
    for pat, path in results.items():
        print(f"  • {pat}: {path} ({round(os.path.getsize(path)/(1024*1024), 1)} MB)")
    print("=" * 70)

if __name__ == "__main__":
    main()
