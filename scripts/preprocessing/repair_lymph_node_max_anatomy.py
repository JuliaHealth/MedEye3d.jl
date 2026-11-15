#!/usr/bin/env python3
"""
repair_lymph_node_max_anatomy.py - Repairs max_anatomy in cases missing lymph node stations.

This script handles three categories of issues:
  A) ZERO_LYMPH: Cases with anatomy but no lymph node stations (never rebuilt after LN rules ran)
  B) CORRUPTED: Cases where max_anatomy has unreasonable values (entire volume filled with labels)
  C) HIGH_LABELS: Cases where CADS label IDs collide with lymph node IDs

For each case, it:
  1. Rebuilds clean CT max_anatomy from raw/ data (packed_anatomical_tensor + final_results_gpu)
  2. Applies CADS structures with lymph node protection rules
  3. Restores MRI timepoint anatomy (8-zone prostate segmentation)
  4. Validates and writes back to the HDF5

Usage:
  python3 repair_lymph_node_max_anatomy.py --dry-run           # Check all without modifying
  python3 repair_lymph_node_max_anatomy.py                     # Repair all issue cases
  python3 repair_lymph_node_max_anatomy.py --cases Pat25 Pat26 # Repair specific cases
  python3 repair_lymph_node_max_anatomy.py --category zero     # Only zero-lymph cases
"""
import os
import re
import sys
import glob
import json
import argparse
import numpy as np
import h5py
import time
import traceback

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(os.path.dirname(HERE), "preprocessing") if "scratch" in HERE else HERE
sys.path.insert(0, "/mnt/big/project_ssd/project_ssd/MedEye3d.jl/scripts/preprocessing")

from cads_anatomy_wiring import (
    patch_case_h5_with_cads, CADS_STRUCTURE_MAP, lymph_label_ids, is_lymph_label_name,
)
from batch_process_cohort_dual_gpu import layer_and_sanitize_ct_anatomy

BASE_DATA_DIR = "/mnt/big/project_ssd/project_ssd/MedEye3d.jl/data/cases/Prostata_bimodal"


def flip_y(a):
    return np.flip(a, axis=1)


def rebuild_mri_anatomy(raw_dir, ct_img):
    """Rebuild 8-zone prostate labels resampled into CT space."""
    ct_shape = sitk.GetArrayFromImage(ct_img).shape
    out = np.zeros(ct_shape, dtype=np.uint16)
    pros = os.path.join(raw_dir, "prostate_segmentation.nii.gz")
    tfm = os.path.join(raw_dir, "Transform_FollowUp_to_Baseline_1.tfm")
    if not (os.path.isfile(pros) and os.path.isfile(tfm)):
        return out
    p_img = sitk.ReadImage(pros)
    p_arr = sitk.GetArrayFromImage(p_img)
    p8 = (np.isin(p_arr, list(range(1, 9)))).astype(np.uint16) * p_arr
    if p8.any():
        img = sitk.GetImageFromArray(p8)
        img.CopyInformation(p_img)
        res = sitk.Resample(img, ct_img, sitk.ReadTransform(tfm),
                            sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
        out = sitk.GetArrayFromImage(res).astype(np.uint16)
    return out


def classify_case(pat):
    """Classify a case into one of the issue categories."""
    case_dir = os.path.join(BASE_DATA_DIR, pat)
    h5_path = os.path.join(case_dir, "preprocessed_volumes.h5")
    raw_dir = os.path.join(case_dir, "raw")

    if not os.path.isfile(h5_path):
        return "NO_HDF5", {}

    info = {}
    try:
        with h5py.File(h5_path, "r") as f:
            for g in f.keys():
                if isinstance(f[g], h5py.Group) and "max_anatomy" in f[g] and not g.startswith("TFM_"):
                    arr = f[g]["max_anatomy"][:]
                    nonzero = int(np.sum(arr > 0))
                    unique = np.unique(arr[arr > 0])
                    lymph_mask = (arr >= 200) & ~((arr >= 531) & (arr <= 539))
                    lymph_voxels = int(np.sum(lymph_mask))
                    total_voxels = arr.size

                    info["shape"] = arr.shape
                    info["nonzero"] = nonzero
                    info["unique_count"] = len(unique)
                    info["lymph_voxels"] = lymph_voxels
                    info["total_voxels"] = total_voxels
                    info["max_label"] = int(unique.max()) if len(unique) > 0 else 0

                    # Check for corrupted: if lymph voxels == total voxels, it's corrupted
                    if lymph_voxels == total_voxels and total_voxels > 0:
                        return "CORRUPTED", info

                    # Check for zero lymph
                    if lymph_voxels == 0:
                        return "ZERO_LYMPH", info

                    # Check for HIGH_LABELS (> 539 excluding CADS)
                    high = [int(v) for v in unique if v > 539]
                    if high:
                        info["high_labels"] = high[:10]
                        return "HIGH_LABELS", info

                    return "OK", info

    except Exception as e:
        return "ERROR", {"error": str(e)}

    return "UNKNOWN", info


def repair_case(pat, dry_run=False, force=False):
    """Repair a single case's max_anatomy by rebuilding from raw data."""
    case_dir = os.path.join(BASE_DATA_DIR, pat)
    h5_path = os.path.join(case_dir, "preprocessed_volumes.h5")
    raw_dir = os.path.join(case_dir, "raw")
    ct_path = os.path.join(raw_dir, "Fixed_CT_Volume.nii.gz")
    pred_dir = os.path.join(case_dir, "cads", "Fixed_CT_Volume")
    p556 = os.path.join(pred_dir, "Fixed_CT_Volume_part_556.nii.gz")
    p558 = os.path.join(pred_dir, "Fixed_CT_Volume_part_558.nii.gz")

    # Check prerequisites
    for need in (h5_path, ct_path, os.path.join(raw_dir, "packed_anatomical_tensor.h5")):
        if not os.path.isfile(need):
            return f"SKIP missing {os.path.basename(need)}"

    # Check for final_results_gpu.h5 (lymph node data)
    ln_h5 = os.path.join(raw_dir, "final_results_gpu.h5")
    has_ln = os.path.isfile(ln_h5)
    if not has_ln:
        return "SKIP missing final_results_gpu.h5 (no lymph node data available)"

    # Validate final_results_gpu.h5 is not corrupted (> 1 GB is suspicious)
    ln_size = os.path.getsize(ln_h5)
    if ln_size > 1e9:  # > 1 GB
        return f"SKIP final_results_gpu.h5 too large ({ln_size/1e6:.0f} MB) - likely corrupted"

    # Read CT shape
    ct_img = sitk.ReadImage(ct_path)
    ct_shape = sitk.GetArrayFromImage(ct_img).shape

    # Rebuild clean max_anatomy from raw
    t0 = time.time()
    base, base_labels, bone_ids, bone_atlas, skellytour, excl = layer_and_sanitize_ct_anatomy(ct_shape, raw_dir)
    base_f = flip_y(base)
    rebuild_time = time.time() - t0

    # Count lymph voxels in rebuilt
    ln_ids = lymph_label_ids(base_labels)
    lymph_rebuilt = int(np.isin(base_f, ln_ids).sum())

    if lymph_rebuilt == 0:
        return f"SKIP rebuilt max_anatomy still has 0 lymph voxels (final_results_gpu.h5 may be incomplete)"

    if dry_run:
        return (f"DRY-RUN rebuild has {lymph_rebuilt:,} lymph voxels, "
                f"{len(base_labels)} labels, {len(ln_ids)} LN stations, "
                f"rebuild took {rebuild_time:.1f}s")

    # Apply CADS structures with lymph node protection
    has_cads = os.path.isfile(p556) or os.path.isfile(p558)
    ok = patch_case_h5_with_cads(
        h5_path,
        cads_556_path=p556 if os.path.isfile(p556) else None,
        cads_558_path=p558 if os.path.isfile(p558) else None,
        ref_ct_path=ct_path,
        base_anat_override=base_f,
        base_labels_override=base_labels,
    )
    if not ok:
        return "FAIL patch_case_h5_with_cads returned False"

    # Restore MRI timepoint anatomy
    with h5py.File(h5_path, "r") as f:
        tfm_groups = [k for k in f.keys() if k.startswith("TFM_")]

    mri_f = flip_y(rebuild_mri_anatomy(raw_dir, ct_img))
    mri_vox = int((mri_f > 0).sum())

    with h5py.File(h5_path, "r+") as f:
        for g in tfm_groups:
            key = f"{g}/max_anatomy.nii.gz"
            if key in f:
                f[key][:] = mri_f

    # Verify the repair
    with h5py.File(h5_path, "r") as f:
        final = f["ATLAS/max_anatomy"][:]
        final_labels = json.loads(f["_meta_/max_anatomy_labels.json"][()])
    final_ln_ids = lymph_label_ids(final_labels)
    final_lymph = int(np.isin(final, final_ln_ids).sum())

    return (f"OK lymph_voxels={final_lymph:,} labels={len(final_labels)} "
            f"ln_stations={len(final_ln_ids)} mri_vox={mri_vox:,} "
            f"tfm_groups={len(tfm_groups)} time={rebuild_time:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", help="Specific cases to repair")
    ap.add_argument("--category", choices=["zero", "corrupted", "high", "all"], default="all",
                    help="Category of issues to repair")
    ap.add_argument("--dry-run", action="store_true", help="Check without modifying")
    ap.add_argument("--force", action="store_true", help="Repair even OK cases")
    args = ap.parse_args()

    def num(s):
        m = re.search(r"\d+", s)
        return int(m.group()) if m else 9999

    cases = sorted((os.path.basename(d) for d in glob.glob(os.path.join(BASE_DATA_DIR, "Pat*"))
                    if not d.endswith("_student")), key=num)

    if args.cases:
        cases = [c for c in cases if c in args.cases]

    # First pass: classify all cases
    print("=" * 100)
    print("Phase 1: Classifying cases...")
    print("=" * 100)
    classified = {}
    for pat in cases:
        cat, info = classify_case(pat)
        classified[pat] = (cat, info)

    # Count categories
    cats = {}
    for pat, (cat, info) in classified.items():
        cats.setdefault(cat, []).append(pat)

    for cat in sorted(cats.keys()):
        print(f"  {cat}: {len(cats[cat])} cases")
        if cat != "OK" and cat != "NO_HDF5":
            for p in cats[cat][:5]:
                print(f"    - {p}: {classified[p][1]}")
            if len(cats[cat]) > 5:
                print(f"    ... and {len(cats[cat])-5} more")

    # Select cases to repair
    to_repair = []
    if args.force:
        to_repair = cases
    else:
        if args.category in ("zero", "all"):
            to_repair.extend(cats.get("ZERO_LYMPH", []))
        if args.category in ("corrupted", "all"):
            to_repair.extend(cats.get("CORRUPTED", []))
        if args.category in ("high", "all"):
            to_repair.extend(cats.get("HIGH_LABELS", []))

    to_repair = sorted(to_repair, key=num)

    if not to_repair:
        print("\nNo cases to repair.")
        return

    print(f"\n{'=' * 100}")
    print(f"Phase 2: {'DRY-RUN checking' if args.dry_run else 'Repairing'} {len(to_repair)} cases...")
    print(f"{'=' * 100}")

    results = {"OK": 0, "SKIP": 0, "FAIL": 0, "ERROR": 0}
    for i, pat in enumerate(to_repair, 1):
        try:
            msg = repair_case(pat, dry_run=args.dry_run, force=args.force)
            status = msg.split()[0] if msg else "UNKNOWN"
            if status.startswith("OK") or status.startswith("DRY"):
                results["OK"] += 1
            elif status.startswith("SKIP"):
                results["SKIP"] += 1
            else:
                results["FAIL"] += 1
        except Exception as e:
            msg = f"ERROR {type(e).__name__}: {e}"
            results["ERROR"] += 1
            traceback.print_exc()
        print(f"[{i}/{len(to_repair)}] {pat}: {msg}", flush=True)

    print(f"\n{'=' * 100}")
    print(f"RESULTS: {results}")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
