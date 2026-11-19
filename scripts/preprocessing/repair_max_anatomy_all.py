#!/usr/bin/env python3
"""
Repair max_anatomy in preprocessed_volumes.h5 for all Prostata bimodal cases.
Uses the FIXED layer_and_sanitize_ct_anatomy() from batch_process_cohort_dual_gpu.py
which properly handles old LN station skipping and dead label purging.

This can be run AFTER the anatomy + LN rules pipeline completes, to rebuild
the max_anatomy dataset in each case's preprocessed_volumes.h5.

HDF5 structure:
  BASELINE/max_anatomy.nii.gz  — CT whole-body anatomy
  ATLAS/max_anatomy            — copy of CT atlas
  TFM_*/max_anatomy.nii.gz    — MRI 8-zone prostate anatomy (NOT touched here)
  _meta_/max_anatomy_labels.json — label dict

Usage:
    python3 repair_max_anatomy_all.py [--cases Pat3,Pat6] [--dry-run]
"""

import os
import sys
import json
import time
import numpy as np
import h5py
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cads_anatomy_wiring import (
    merge_cads_structures, load_resampled_cads_arrays, is_lymph_label_name,
    is_excluded_non_organ_mask, is_lung_label_name, iter_authoritative_lymph_node_masks,
)

MEDEYE_DIR = "/mnt/big/project_ssd/project_ssd/MedEye3d.jl"
if not os.path.isdir(MEDEYE_DIR):
    MEDEYE_DIR = "/workspaces/MedEye3d.jl"
CASES_DIR = os.path.join(MEDEYE_DIR, "data/cases/Prostata_bimodal")


def _is_lymph_station_name(name):
    """Check if a mask name represents a lymph node station (mirrors Julia is_lymph_node_structure)."""
    ln = name.lower()
    if ln in ("thoracic_aorta", "thoracic_duct"):
        return False
    return is_lymph_label_name(ln)


def layer_and_sanitize_ct_anatomy(ct_shape, raw_dir):
    """Fixed version that skips old LN from packed tensor when fresh LN exists."""
    Z, Y, X = ct_shape
    max_anatomy = np.zeros(ct_shape, dtype=np.uint16)
    labels_dict = {}
    bone_ids = []
    lbl_idx = 1
    exclusion_mask = np.zeros(ct_shape, dtype=np.uint8)

    anat_h5 = os.path.join(raw_dir, "packed_anatomical_tensor.h5")
    reg_path = os.path.join(raw_dir, "packed_anatomical_tensor_registry.json")

    case_dir = os.path.dirname(os.path.abspath(raw_dir))
    ln_dir_exists = os.path.isdir(os.path.join(case_dir, "lymph_nodes")) or os.path.isdir(os.path.join(raw_dir, "lymph_nodes"))
    ln_h5 = os.path.join(raw_dir, "final_results_gpu.h5")
    has_fresh_ln = ln_dir_exists or os.path.isfile(ln_h5)
    skipped_old_ln = 0

    deferred_lungs = []
    if os.path.isfile(anat_h5):
        reg = None
        if os.path.isfile(reg_path):
            with open(reg_path, "r") as rf:
                reg = json.load(rf)
        with h5py.File(anat_h5, "r") as af:
            if reg is None and "registry" in af:
                raw_reg = af["registry"][()]
                s_reg = raw_reg.decode() if isinstance(raw_reg, bytes) else (bytes(raw_reg).decode() if hasattr(raw_reg, "tobytes") else str(raw_reg))
                reg = json.loads(s_reg)
            if reg is not None:
                pt = af["packed_tensor"][:]
                masks_dict = reg.get("masks", reg)
                for name, meta in masks_dict.items():
                    if not isinstance(meta, dict):
                        continue
                    clean = name.lower()
                    if is_excluded_non_organ_mask(clean):
                        continue
                    if has_fresh_ln and _is_lymph_station_name(clean):
                        skipped_old_ln += 1
                        continue
                    ch = meta.get("channel", 1) - 1
                    mid = meta.get("id", 1)
                    if 0 <= ch < pt.shape[0]:
                        mask = (pt[ch] == mid)
                        if mask.shape == ct_shape and mask.any():
                            if is_lung_label_name(clean):
                                deferred_lungs.append((clean, mask))
                                continue
                            max_anatomy[mask] = lbl_idx
                            labels_dict[str(lbl_idx)] = clean
                            if any(b in clean for b in ["bone", "femur", "hip", "sacrum", "vertebra", "rib", "sternum", "ilium", "ischium"]):
                                bone_ids.append(lbl_idx)
                            if any(u in clean for u in ["bladder", "kidney", "lacrimal", "parotid", "submandibular"]):
                                exclusion_mask[mask] = 1
                            lbl_idx += 1
        if skipped_old_ln > 0:
            print(f"    Skipped {skipped_old_ln} old LN stations from packed tensor")

    fresh_ln_count = 0
    ln_sources = set()
    if has_fresh_ln:
        for clean, m_ln, src_tag in iter_authoritative_lymph_node_masks(ct_shape, raw_dir):
            max_anatomy[m_ln] = lbl_idx
            labels_dict[str(lbl_idx)] = clean
            fresh_ln_count += 1
            ln_sources.add(src_tag.split("/")[0])
            lbl_idx += 1
        print(f"    Integrated {fresh_ln_count} authoritative LN stations (source={sorted(ln_sources)})")

    # Layer 2: Lungs ONLY — the only anatomy allowed to override lymph node areas
    for clean, mask in deferred_lungs:
        max_anatomy[mask] = lbl_idx
        labels_dict[str(lbl_idx)] = clean
        lbl_idx += 1

    # Purge dead labels
    live_ids = set(np.unique(max_anatomy))
    dead_keys = [k for k in labels_dict if int(k) not in live_ids]
    for k in dead_keys:
        del labels_dict[k]
    if dead_keys:
        print(f"    Purged {len(dead_keys)} dead label IDs")

    bone_atlas = np.isin(max_anatomy, bone_ids).astype(np.float32)
    skellytour = bone_atlas.copy()
    return max_anatomy, labels_dict, bone_ids, bone_atlas, skellytour, exclusion_mask


def repair_case(case_name, dry_run=False):
    """Repair max_anatomy in a single case's preprocessed_volumes.h5."""
    case_dir = os.path.join(CASES_DIR, case_name)
    h5_path = os.path.join(case_dir, "preprocessed_volumes.h5")
    raw_dir = os.path.join(case_dir, "raw")
    
    if not os.path.isfile(h5_path):
        return "NO_H5"
    
    anat_h5 = os.path.join(raw_dir, "packed_anatomical_tensor.h5")
    if not os.path.isfile(anat_h5):
        return "NO_ANAT"
    
    # Get CT shape from the HDF5 BASELINE group
    try:
        with h5py.File(h5_path, "r") as f:
            if "BASELINE" not in f:
                return "NO_BASELINE"
            bl = f["BASELINE"]
            # Get shape from CT volume or max_anatomy
            if "Fixed_CT_Volume_0.nii.gz" in bl:
                ct_shape = bl["Fixed_CT_Volume_0.nii.gz"].shape
            elif "max_anatomy.nii.gz" in bl:
                ct_shape = bl["max_anatomy.nii.gz"].shape
            elif "max_anatomy" in bl:
                ct_shape = bl["max_anatomy"].shape
            else:
                return "NO_CT_DATASET"
    except Exception as e:
        return f"ERROR_READ: {e}"
    
    if dry_run:
        has_ln = os.path.isfile(os.path.join(raw_dir, "final_results_gpu.h5"))
        return f"DRY_RUN (shape={ct_shape}, has_ln={has_ln})"
    
    print(f"  [{case_name}] Rebuilding max_anatomy (shape={ct_shape})...")
    
    # Run the fixed layer_and_sanitize_ct_anatomy
    max_anatomy, labels_dict, bone_ids, bone_atlas, skellytour, exclusion_mask = \
        layer_and_sanitize_ct_anatomy(ct_shape, raw_dir)

    # Always preserve/merge CADS Task 556 & 558 structures (seminal_vesicle, lacrimal glands, etc.)
    cads_556_arr, cads_558_arr = load_resampled_cads_arrays(case_dir)
    if cads_556_arr is not None or cads_558_arr is not None:
        max_anatomy, labels_dict = merge_cads_structures(
            max_anatomy,
            cads_556_arr=cads_556_arr,
            cads_558_arr=cads_558_arr,
            labels_dict=labels_dict,
            flip_y_cads=False
        )
        for lac_val in (533, 534):
            exclusion_mask[max_anatomy == lac_val] = 1
    
    n_labels = len(labels_dict)
    n_lymph = sum(1 for v in labels_dict.values() if _is_lymph_station_name(v))
    lymph_voxels = 0
    for lid, name in labels_dict.items():
        if _is_lymph_station_name(name):
            lymph_voxels += int(np.sum(max_anatomy == int(lid)))
    
    # Y-flip to match preflipped HDF5
    max_anatomy_f = np.flip(max_anatomy, axis=1).copy()
    bone_atlas_f = np.flip(bone_atlas, axis=1).copy()
    skellytour_f = np.flip(skellytour, axis=1).copy()
    
    # Write to HDF5
    try:
        with h5py.File(h5_path, "a") as f:
            # Update BASELINE/max_anatomy.nii.gz
            bl = f["BASELINE"]
            for key in ["max_anatomy.nii.gz", "max_anatomy"]:
                if key in bl:
                    # Preserve spatial attrs
                    old_ds = bl[key]
                    sp_attrs = {}
                    for attr in ["spacing", "origin", "direction"]:
                        if attr in old_ds.attrs:
                            sp_attrs[attr] = old_ds.attrs[attr]
                    del bl[key]
                    ds = bl.create_dataset(key, data=max_anatomy_f, compression="gzip", compression_opts=4, shuffle=True)
                    for attr, val in sp_attrs.items():
                        ds.attrs[attr] = val
                    break
            else:
                # Create if didn't exist
                bl.create_dataset("max_anatomy.nii.gz", data=max_anatomy_f, compression="gzip", compression_opts=4, shuffle=True)
            
            # Update ATLAS/max_anatomy
            if "ATLAS" in f:
                atl = f["ATLAS"]
                for key in ["max_anatomy", "max_anatomy.nii.gz"]:
                    if key in atl:
                        sp_attrs = {}
                        for attr in ["spacing", "origin", "direction"]:
                            if attr in atl[key].attrs:
                                sp_attrs[attr] = atl[key].attrs[attr]
                        del atl[key]
                        ds = atl.create_dataset(key, data=max_anatomy_f, compression="gzip", compression_opts=4, shuffle=True)
                        for attr, val in sp_attrs.items():
                            ds.attrs[attr] = val
                        break
                # Update bone_atlas
                if "bone_atlas" in atl:
                    sp_attrs = {}
                    for attr in ["spacing", "origin", "direction"]:
                        if attr in atl["bone_atlas"].attrs:
                            sp_attrs[attr] = atl["bone_atlas"].attrs[attr]
                    del atl["bone_atlas"]
                    ds = atl.create_dataset("bone_atlas", data=bone_atlas_f, compression="gzip", compression_opts=4, shuffle=True)
                    for attr, val in sp_attrs.items():
                        ds.attrs[attr] = val
                # Update skellytour
                if "skellytour" in atl:
                    sp_attrs = {}
                    for attr in ["spacing", "origin", "direction"]:
                        if attr in atl["skellytour"].attrs:
                            sp_attrs[attr] = atl["skellytour"].attrs[attr]
                    del atl["skellytour"]
                    ds = atl.create_dataset("skellytour", data=skellytour_f, compression="gzip", compression_opts=4, shuffle=True)
                    for attr, val in sp_attrs.items():
                        ds.attrs[attr] = val
            
            # Update BASELINE/skellytour.nii.gz
            if "skellytour.nii.gz" in bl:
                sp_attrs = {}
                for attr in ["spacing", "origin", "direction"]:
                    if attr in bl["skellytour.nii.gz"].attrs:
                        sp_attrs[attr] = bl["skellytour.nii.gz"].attrs[attr]
                del bl["skellytour.nii.gz"]
                ds = bl.create_dataset("skellytour.nii.gz", data=skellytour_f, compression="gzip", compression_opts=4, shuffle=True)
                for attr, val in sp_attrs.items():
                    ds.attrs[attr] = val
            
            # Update _meta_/max_anatomy_labels.json
            if "_meta_" in f:
                meta = f["_meta_"]
                if "max_anatomy_labels.json" in meta:
                    del meta["max_anatomy_labels.json"]
                meta.create_dataset("max_anatomy_labels.json", data=json.dumps(labels_dict))
                
                # Update bone_label_ids
                if "bone_label_ids" in meta:
                    del meta["bone_label_ids"]
                meta.create_dataset("bone_label_ids", data=np.array(bone_ids, dtype=np.int32))
    except Exception as e:
        return f"WRITE_ERROR: {e}"

    # Sync local xnat_cache copies if present
    cache_root = os.path.expanduser("~/.medeye3d/xnat_cache")
    if os.path.isdir(cache_root):
        import glob, shutil
        for cand in glob.glob(os.path.join(cache_root, "**", f"*__{case_name}_preprocessed*.h5"), recursive=True):
            if os.path.isfile(cand):
                try:
                    shutil.copyfile(h5_path, cand)
                    meta_p = cand + ".xnat_meta.json"
                    if os.path.isfile(meta_p):
                        os.remove(meta_p)
                except Exception:
                    pass

    return f"OK (labels={n_labels}, lymph_stations={n_lymph}, lymph_voxels={lymph_voxels:,})"


def main():
    parser = argparse.ArgumentParser(description="Repair max_anatomy in all preprocessed_volumes.h5")
    parser.add_argument("--cases", type=str, help="Comma-separated list of cases")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--upload-xnat", action="store_true", help="Upload repaired HDF5s to XNAT MEDEYE3D resource")
    parser.add_argument("--upload-only", action="store_true", help="Skip local rebuild and only upload existing preprocessed_volumes.h5 to XNAT")
    parser.add_argument("--skip-synced", action="store_true", help="With --upload-only/--upload-xnat, skip cases whose remote XNAT HDF5 size and status.json already match")
    args = parser.parse_args()
    
    if args.cases:
        case_names = [c.strip() for c in args.cases.split(",")]
    else:
        case_names = sorted([
            d for d in os.listdir(CASES_DIR)
            if d.startswith("Pat") and os.path.isdir(os.path.join(CASES_DIR, d))
            and "_PETCT_" not in d and "_student" not in d
            and os.path.isfile(os.path.join(CASES_DIR, d, "preprocessed_volumes.h5"))
        ], key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    
    xnat_mgr = None
    xnat_subjs = {}
    proj_id = "Prostata_bimodal_PIPELINE"
    if (args.upload_xnat or args.upload_only) and not args.dry_run:
        from batch_process_cohort_dual_gpu import XnatManager, CENSUS_PATH
        with open(CENSUS_PATH, "r") as cf:
            xnat_subjs = json.load(cf)
        xnat_mgr = XnatManager()
        print(f"XNAT upload enabled ({len(xnat_subjs)} subjects indexed from census)")

    print(f"Repairing max_anatomy for {len(case_names)} cases...")
    print("=" * 80)
    
    ok = 0
    fail = 0
    skip = 0
    uploaded = 0
    t0 = time.time()
    
    for cn in case_names:
        result = "OK (upload-only)" if args.upload_only else repair_case(cn, dry_run=args.dry_run)
        status = "OK" if result.startswith("OK") else ("SKIP" if result in ("NO_H5", "NO_ANAT", "NO_BASELINE", "NO_CT_DATASET") else "FAIL")
        if status == "OK":
            ok += 1
            if xnat_mgr is not None and cn in xnat_subjs:
                sinfo = xnat_subjs[cn]
                subj_id = sinfo.get("subj_id", "")
                subj_label = sinfo.get("subj_label", "")
                h5_p = os.path.join(CASES_DIR, cn, "preprocessed_volumes.h5")
                if subj_id and subj_label:
                    loc_sz = os.path.getsize(h5_p)
                    already_synced = False
                    if args.skip_synced:
                        rf = (xnat_mgr.get_json(f"/data/projects/{proj_id}/subjects/{subj_id}/resources/MEDEYE3D/files?format=json") or {}).get("ResultSet", {}).get("Result", [])
                        rem_sz = next((int(x.get("Size", -1)) for x in rf if x.get("Name") == f"{subj_label}_preprocessed.h5"), -1)
                        if rem_sz == loc_sz:
                            st_r = xnat_mgr.request("GET", f"/data/projects/{proj_id}/subjects/{subj_id}/resources/ANNOTATIONS/files/status.json", timeout=15)
                            if st_r and st_r.status_code == 200:
                                try:
                                    sj = st_r.json()
                                    if sj.get("manually_modified") is False and sj.get("modified_by", "") == "":
                                        already_synced = True
                                except Exception:
                                    pass
                    if already_synced:
                        print(f"  [SKIP-SYNCED] {cn}: already matches XNAT ({loc_sz} bytes)", flush=True)
                    else:
                        xnat_mgr.request("PUT", f"/data/projects/{proj_id}/subjects/{subj_id}/resources/MEDEYE3D")
                        h5_uri = f"/data/projects/{proj_id}/subjects/{subj_id}/resources/MEDEYE3D/files/{subj_label}_preprocessed.h5?overwrite=true"
                        if not xnat_mgr.upload_file_streaming(h5_uri, h5_p):
                            result = "FAIL_XNAT_UPLOAD"
                            status = "FAIL"
                            ok -= 1
                            fail += 1
                        else:
                            uploaded += 1
                if subj_id and subj_label and status == "OK" and not already_synced:
                    # Sync XNAT ANNOTATIONS/status.json from HDF5 _meta_/modification_status.json
                    try:
                        import tempfile
                        with h5py.File(h5_p, "r") as hf:
                            ms = json.loads(hf["_meta_/modification_status.json"][()]) if "_meta_/modification_status.json" in hf else {}
                        is_mod = bool(ms.get("manually_modified", False))
                        mod_by = str(ms.get("modified_by", ""))
                        last_m = str(ms.get("last_modified", ""))
                        st_payload = {
                            "project_id": proj_id,
                            "subject_id": subj_label,
                            "status": "PROCESSED" if is_mod else "PREPROCESSED",
                            "manually_modified": is_mod,
                            "modified_by": mod_by if is_mod else "",
                            "assigned_user": mod_by if is_mod else "",
                            "last_modified": last_m,
                            "summary": {"manually_modified": is_mod, "modified_by": mod_by if is_mod else ""},
                        }
                        xnat_mgr.request("PUT", f"/data/projects/{proj_id}/subjects/{subj_id}/resources/ANNOTATIONS")
                        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
                            json.dump(st_payload, tf, indent=2)
                            tmp_st = tf.name
                        try:
                            st_uri = f"/data/projects/{proj_id}/subjects/{subj_id}/resources/ANNOTATIONS/files/status.json?overwrite=true"
                            xnat_mgr.upload_file_streaming(st_uri, tmp_st)
                        finally:
                            if os.path.isfile(tmp_st):
                                os.remove(tmp_st)
                    except Exception:
                        pass
        elif status == "SKIP":
            skip += 1
        else:
            fail += 1
        
        icon = "✓" if status == "OK" else ("⊘" if status == "SKIP" else "✗")
        print(f"  {icon} {cn}: {result}", flush=True)
    
    elapsed = time.time() - t0
    print("=" * 80)
    print(f"OK: {ok}  SKIP: {skip}  FAIL: {fail}  UPLOADED: {uploaded}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
