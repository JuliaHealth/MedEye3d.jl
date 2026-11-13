#!/usr/bin/env python3
"""
repair_cads_max_anatomy.py - Repairs max_anatomy in already CADS-patched cases.

The first CADS wiring overwrote lymph node areas and wrote the CT max_anatomy into
the MRI follow-up (TFM_*) groups. This script, per case:
  1. Rebuilds the clean CT max_anatomy from raw/ (same code as the cohort pipeline).
  2. Validates the rebuild against the stored atlas (outside CADS voxels).
  3. Re-merges the CADS structures with the fixed rules (lymph protected, sigmoid
     only over colon).
  4. Restores each MRI timepoint's OWN max_anatomy (8-zone prostate segmentation,
     empty if none) instead of the CT atlas.

Usage:
  repair_cads_max_anatomy.py --cases Pat9 [--dry-run]
  repair_cads_max_anatomy.py --slice 0:2        (even-indexed cases)
"""
import os
import re
import sys
import glob
import json
import argparse
import numpy as np
import h5py
import SimpleITK as sitk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cads_anatomy_wiring import (
    patch_case_h5_with_cads, CADS_STRUCTURE_MAP, lymph_label_ids,
)
from batch_process_cohort_dual_gpu import layer_and_sanitize_ct_anatomy

BASE_DATA_DIR = "/workspaces/MedEye3d.jl/data/cases/Prostata_bimodal"


def flip_y(a):
    return np.flip(a, axis=1)


def rebuild_mri_anatomy(raw_dir, ct_img):
    """Mirror of the cohort pipeline: 8-zone prostate labels resampled into CT space."""
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


def repair_case(pat, dry_run=False, min_match=0.999):
    case_dir = os.path.join(BASE_DATA_DIR, pat)
    h5_path = os.path.join(case_dir, "preprocessed_volumes.h5")
    raw_dir = os.path.join(case_dir, "raw")
    ct_path = os.path.join(raw_dir, "Fixed_CT_Volume.nii.gz")
    pred_dir = os.path.join(case_dir, "cads", "Fixed_CT_Volume")
    p556 = os.path.join(pred_dir, "Fixed_CT_Volume_part_556.nii.gz")
    p558 = os.path.join(pred_dir, "Fixed_CT_Volume_part_558.nii.gz")

    for need in (h5_path, ct_path, os.path.join(raw_dir, "packed_anatomical_tensor.h5")):
        if not os.path.isfile(need):
            return f"SKIP missing {os.path.relpath(need, case_dir)}"

    ct_img = sitk.ReadImage(ct_path)
    ct_shape = sitk.GetArrayFromImage(ct_img).shape

    base, base_labels, *_ = layer_and_sanitize_ct_anatomy(ct_shape, raw_dir)
    base_f = flip_y(base)

    with h5py.File(h5_path, "r") as f:
        cur = f["ATLAS/max_anatomy"][:]
        cur_labels = json.loads(f["_meta_/max_anatomy_labels.json"][()])
        tfm_groups = [k for k in f.keys() if k.startswith("TFM_")]

    if cur.shape != base_f.shape:
        return f"SKIP shape mismatch {cur.shape} vs rebuilt {base_f.shape}"

    # Validate: outside CADS voxels the rebuilt base must equal the stored atlas
    cads_ids = list(CADS_STRUCTURE_MAP.keys())
    outside = ~np.isin(cur, cads_ids)
    match = float((cur[outside] == base_f[outside]).mean())
    cads_names = set(CADS_STRUCTURE_MAP.values())
    # ids >= 531 may legitimately hold CADS names (or collide with real labels in
    # older cases); every other label must match the rebuilt dictionary exactly.
    labels_ok = all(cur_labels.get(k) == v or cur_labels.get(k) in cads_names
                    for k, v in base_labels.items())
    if match < min_match or not labels_ok:
        return f"SKIP validation failed (match={match:.5f}, labels_ok={labels_ok})"

    ln_ids = lymph_label_ids(base_labels)
    lost = int((np.isin(base_f, ln_ids) & ~np.isin(cur, ln_ids)).sum())

    if dry_run:
        return f"DRY-RUN ok match={match:.5f} lymph voxels to restore={lost}"

    ok = patch_case_h5_with_cads(
        h5_path,
        cads_556_path=p556 if os.path.isfile(p556) else None,
        cads_558_path=p558 if os.path.isfile(p558) else None,
        ref_ct_path=ct_path,
        base_anat_override=base_f,
        base_labels_override=base_labels,
    )
    if not ok:
        return "FAIL patch"

    mri_f = flip_y(rebuild_mri_anatomy(raw_dir, ct_img))
    with h5py.File(h5_path, "r+") as f:
        for g in tfm_groups:
            key = f"{g}/max_anatomy.nii.gz"
            if key in f:
                f[key][:] = mri_f
    return (f"OK match={match:.5f} lymph restored={lost} "
            f"mri_zone_voxels={int((mri_f > 0).sum())} tfm_groups={len(tfm_groups)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--slice", type=str, default=None, help="start:step e.g. 0:2")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    def num(s):
        m = re.search(r"\d+", s)
        return int(m.group()) if m else 9999

    cases = sorted((os.path.basename(d) for d in glob.glob(os.path.join(BASE_DATA_DIR, "Pat*"))
                    if not d.endswith("_student")), key=num)
    if args.cases:
        cases = [c for c in cases if c in args.cases]
    elif args.slice:
        start, step = [int(x) for x in args.slice.split(":")]
        cases = cases[start::step]

    for i, pat in enumerate(cases, 1):
        try:
            msg = repair_case(pat, dry_run=args.dry_run)
        except Exception as e:  # keep going across the cohort
            msg = f"ERROR {type(e).__name__}: {e}"
        print(f"[{i}/{len(cases)}] {pat}: {msg}", flush=True)


if __name__ == "__main__":
    main()
