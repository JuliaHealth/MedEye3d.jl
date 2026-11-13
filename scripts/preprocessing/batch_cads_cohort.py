#!/usr/bin/env python3
"""
batch_cads_cohort.py - Runs CADS inference (Task 556: RT Risk Organs and Task 558: Head & Neck)
across cases in the Prostata_bimodal cohort and wires the new structures directly into
preprocessed_volumes.h5 (ATLAS/max_anatomy, BASELINE/max_anatomy.nii.gz, and metadata).

Supports running on a designated GPU (cuda:0 or cuda:1) with case slicing to enable
concurrent multi-GPU execution across the entire cohort.
"""

import os
import sys
import re
import glob
import time
import json
import argparse
import numpy as np

# Ensure CADS and workspace paths are on sys.path
CADS_ROOT = "/workspaces/MedEye3d.jl/third_party/CADS"
if CADS_ROOT not in sys.path:
    sys.path.insert(0, CADS_ROOT)
sys.path.insert(0, "/workspaces/MedEye3d.jl/scripts/preprocessing")

from cads_anatomy_wiring import (
    patch_case_h5_with_cads,
    CADS_STRUCTURE_MAP,
    TASK_556_MAPPING,
    TASK_558_MAPPING
)

BASE_DATA_DIR = "/workspaces/MedEye3d.jl/data/cases/Prostata_bimodal"


def parse_slice(slice_str: str, total_len: int) -> list:
    """Parses a slice string like '0:2' (step 2 starting at 0) or '0:50'."""
    parts = [int(p) for p in slice_str.split(":")]
    if len(parts) == 1:
        return [parts[0]]
    elif len(parts) == 2:
        # start:step (e.g. 0:2 means 0, 2, 4...)
        start, step = parts[0], parts[1]
        return list(range(start, total_len, step))
    elif len(parts) == 3:
        start, stop, step = parts[0], parts[1], parts[2]
        return list(range(start, min(stop, total_len), step))
    return list(range(total_len))


def get_all_valid_cases() -> list:
    """Returns sorted list of case directories containing raw/Fixed_CT_Volume.nii.gz."""
    pat_dirs = glob.glob(os.path.join(BASE_DATA_DIR, "Pat*"))
    valid_cases = []
    for d in pat_dirs:
        bname = os.path.basename(d)
        if bname.startswith("Pat") and not bname.endswith("_student"):
            ct_path = os.path.join(d, "raw", "Fixed_CT_Volume.nii.gz")
            if os.path.isfile(ct_path):
                valid_cases.append(bname)

    # Sort numerically by patient number
    def get_num(s):
        m = re.search(r"\d+", s)
        return int(m.group()) if m else 9999

    valid_cases.sort(key=get_num)
    return valid_cases


def main():
    parser = argparse.ArgumentParser(description="Batch CADS inference and anatomy wiring")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID (0 or 1)")
    parser.add_argument("--slice", type=str, default=None, help="Index slice, e.g. '0:2' (even) or '1:2' (odd)")
    parser.add_argument("--cases", nargs="*", help="Specific case IDs to process (e.g. Pat1 Pat2 Pat9)")
    parser.add_argument("--no-patch", action="store_true", help="Skip HDF5 patching")
    parser.add_argument("--force", action="store_true", help="Force re-run inference even if output files exist")
    args = parser.parse_args()

    # Configure GPU environment before importing torch or cads
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CADS_WEIGHTS_PATH"] = "/root/.cads/model_weights"

    from cads.utils.libs import setup_nnunet_env, get_model_weights_dir, check_or_download_model_weights
    from cads.utils.inference import predict

    setup_nnunet_env("research")
    model_folder = get_model_weights_dir("research")

    # Verify weights
    for task_id in [556, 558]:
        check_or_download_model_weights(task_id, "research")

    all_cases = get_all_valid_cases()
    if args.cases:
        target_cases = [c for c in args.cases if c in all_cases]
    elif args.slice:
        indices = parse_slice(args.slice, len(all_cases))
        target_cases = [all_cases[i] for i in indices]
    else:
        target_cases = all_cases

    print(f"[CADS Worker GPU {args.gpu}] Target cases to process: {len(target_cases)} / {len(all_cases)} total")

    stats = {"processed": 0, "skipped": 0, "errors": 0}
    t_start_total = time.time()

    for idx, pat_id in enumerate(target_cases, 1):
        case_dir = os.path.join(BASE_DATA_DIR, pat_id)
        ct_path = os.path.join(case_dir, "raw", "Fixed_CT_Volume.nii.gz")
        cads_out_dir = os.path.join(case_dir, "cads")
        os.makedirs(cads_out_dir, exist_ok=True)

        pred_dir = os.path.join(cads_out_dir, "Fixed_CT_Volume")
        p556 = os.path.join(pred_dir, "Fixed_CT_Volume_part_556.nii.gz")
        p558 = os.path.join(pred_dir, "Fixed_CT_Volume_part_558.nii.gz")

        h5_path = os.path.join(case_dir, "preprocessed_volumes.h5")

        need_inference = args.force or not (os.path.isfile(p556) and os.path.isfile(p558))

        print(f"\n[{idx}/{len(target_cases)}] --- {pat_id} (GPU {args.gpu}) ---")
        t0 = time.time()

        if need_inference:
            try:
                print(f"  Running CADS Tasks 556 & 558 on {ct_path}...")
                predict(
                    [ct_path],
                    cads_out_dir,
                    model_folder,
                    task_ids=[556, 558],
                    folds="all",
                    use_cpu=False,
                    preprocess_cads=True,
                    postprocess_cads=False,
                    save_all_combined_seg=False,
                    snapshot=False,
                    save_separate_targets=False,
                    num_threads_preprocessing=4,
                    nr_threads_saving=4,
                    mode="preload-gpu",
                    verbose=False
                )
                print(f"  Inference finished in {time.time() - t0:.1f}s")
            except Exception as e:
                print(f"  [ERROR] Inference failed for {pat_id}: {e}")
                import traceback
                traceback.print_exc()
                stats["errors"] += 1
                continue
        else:
            print(f"  CADS segmentations already exist in {pred_dir}, skipping inference.")

        if not args.no_patch and os.path.isfile(h5_path):
            try:
                patch_ok = patch_case_h5_with_cads(
                    h5_path=h5_path,
                    cads_556_path=p556,
                    cads_558_path=p558,
                    ref_ct_path=ct_path
                )
                if patch_ok:
                    print(f"  Patched max_anatomy in {h5_path}")
                else:
                    print(f"  Warning: Patching returned False for {h5_path}")
            except Exception as e:
                print(f"  [ERROR] Patching failed for {pat_id}: {e}")
                stats["errors"] += 1

        stats["processed"] += 1
        elapsed = time.time() - t0
        print(f"  Case {pat_id} done in {elapsed:.1f}s (Total elapsed: {(time.time() - t_start_total)/60:.1f}m)")

    print(f"\n[CADS Worker GPU {args.gpu}] All tasks completed. Processed: {stats['processed']}, Errors: {stats['errors']}")


if __name__ == "__main__":
    main()
