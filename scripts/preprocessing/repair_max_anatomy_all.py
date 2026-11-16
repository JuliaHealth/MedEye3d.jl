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
    merge_cads_structures, load_resampled_cads_arrays, is_lymph_label_name
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

    ln_h5 = os.path.join(raw_dir, "final_results_gpu.h5")
    has_fresh_ln = os.path.isfile(ln_h5)
    skipped_old_ln = 0

    if os.path.isfile(anat_h5) and os.path.isfile(reg_path):
        with open(reg_path, "r") as rf:
            reg = json.load(rf)
        with h5py.File(anat_h5, "r") as af:
            pt = af["packed_tensor"][:]
            masks_dict = reg.get("masks", reg)
            for name, meta in masks_dict.items():
                if not isinstance(meta, dict):
                    continue
                clean = name.lower()
                if any(bad in clean for bad in ["temp", "helper", "dilation", "buffer"]):
                    continue
                if has_fresh_ln and _is_lymph_station_name(clean):
                    skipped_old_ln += 1
                    continue
                ch = meta.get("channel", 1) - 1
                mid = meta.get("id", 1)
                if 0 <= ch < pt.shape[0]:
                    mask = (pt[ch] == mid)
                    if mask.shape == ct_shape and mask.any():
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
    skipped_non_ln = 0
    if has_fresh_ln:
        try:
            with h5py.File(ln_h5, "r") as lf:
                for k in sorted(lf.keys()):
                    clean = k.lower()
                    if any(bad in clean for bad in ["temp", "helper", "_base", "_dilated", "bridge",
                                                     "exclusion", "cavity", "chest_wall"]):
                        continue
                    # ONLY include actual lymph node station masks
                    if not _is_lymph_station_name(clean):
                        skipped_non_ln += 1
                        continue
                    arr_ln = lf[k][:]
                    if arr_ln.ndim == 3:
                        if arr_ln.shape == (X, Y, Z):
                            arr_ln = np.transpose(arr_ln, (2, 1, 0))
                        if arr_ln.shape == (Z, Y, X):
                            m_ln = (arr_ln > 0)
                            if m_ln.any():
                                max_anatomy[m_ln] = lbl_idx
                                labels_dict[str(lbl_idx)] = clean
                                fresh_ln_count += 1
                                lbl_idx += 1
            print(f"    Integrated {fresh_ln_count} fresh LN stations (skipped {skipped_non_ln} non-LN)")
        except Exception as e_ln:
            print(f"    [WARN] Error reading {ln_h5}: {e_ln}")

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
    
    return f"OK (labels={n_labels}, lymph_stations={n_lymph}, lymph_voxels={lymph_voxels:,})"


def main():
    parser = argparse.ArgumentParser(description="Repair max_anatomy in all preprocessed_volumes.h5")
    parser.add_argument("--cases", type=str, help="Comma-separated list of cases")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    if args.cases:
        case_names = [c.strip() for c in args.cases.split(",")]
    else:
        case_names = sorted([
            d for d in os.listdir(CASES_DIR)
            if d.startswith("Pat") and os.path.isdir(os.path.join(CASES_DIR, d))
            and "_PETCT_" not in d
        ], key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    
    print(f"Repairing max_anatomy for {len(case_names)} cases...")
    print("=" * 80)
    
    ok = 0
    fail = 0
    skip = 0
    t0 = time.time()
    
    for cn in case_names:
        result = repair_case(cn, dry_run=args.dry_run)
        status = "OK" if result.startswith("OK") else ("SKIP" if result in ("NO_H5", "NO_ANAT", "NO_BASELINE", "NO_CT_DATASET") else "FAIL")
        if status == "OK":
            ok += 1
        elif status == "SKIP":
            skip += 1
        else:
            fail += 1
        
        icon = "✓" if status == "OK" else ("⊘" if status == "SKIP" else "✗")
        print(f"  {icon} {cn}: {result}")
    
    elapsed = time.time() - t0
    print("=" * 80)
    print(f"OK: {ok}  SKIP: {skip}  FAIL: {fail}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
