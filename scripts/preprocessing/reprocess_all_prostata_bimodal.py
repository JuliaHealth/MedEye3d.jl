#!/usr/bin/env python3
"""
Comprehensive Prostata Bimodal Reprocessing Script
===================================================
Regenerates ALL anatomical segmentations + lymph node rules from scratch.
Uses both RTX 3090 GPUs concurrently.

Steps per case:
1. Run prepare_packed_case.py (TotalSeg + dental + NV-segment + packing) → packed_anatomical_tensor.h5
2. Run Julia GPU DAG VM (lymph node rules) → final_results_gpu.h5
3. Rebuild max_anatomy in preprocessed_volumes.h5 (via layer_and_sanitize_ct_anatomy fix)

Usage:
    python3 reprocess_all_prostata_bimodal.py [--dry-run] [--cases Pat3,Pat6] [--skip-anatomy] [--skip-ln]
"""

import os
import sys
import json
import time
import glob
import shutil
import subprocess
import threading
import queue
import argparse
import traceback

LN_RULES_DIR = "/mnt/big/project_ssd/project_ssd/lymph_node_rules"
MEDEYE_DIR = "/mnt/big/project_ssd/project_ssd/MedEye3d.jl"
CASES_DIR = os.path.join(MEDEYE_DIR, "data/cases/Prostata_bimodal")
PYTHON_ENV = "/mnt/big/project_ssd/project_ssd/psma_mag3/.venv/bin/python3"
JULIA_BIN = "/mnt/big/project_ssd/project_ssd/lymph_node_rules/for_joanna/.local/julia-1.12.6/bin/julia"
JULIA_PROJECT = os.path.join(LN_RULES_DIR, "src/julia_gpu")

def _load_dotenv(env_dict, dotenv_path):
    if os.path.isfile(dotenv_path):
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in env_dict:
                    env_dict[k] = v


# Environment variables
BASE_ENV = os.environ.copy()
_load_dotenv(BASE_ENV, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.env"))
_load_dotenv(BASE_ENV, os.path.join(MEDEYE_DIR, ".env"))
BASE_ENV["PYTHONPATH"] = f"{LN_RULES_DIR}:{BASE_ENV.get('PYTHONPATH', '')}"
BASE_ENV.setdefault("TOTALSEG_LICENSE_NUMBER", os.environ.get("TOTALSEG_LICENSE_NUMBER", os.environ.get("TOTALSEG_LICENSE", "")))
BASE_ENV.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", "")))
BASE_ENV.setdefault("HUGGING_FACE_HUB_TOKEN", BASE_ENV.get("HF_TOKEN", ""))
BASE_ENV["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

LOG_DIR = os.path.join(MEDEYE_DIR, "data/logs/reprocess_" + time.strftime("%Y%m%d_%H%M%S"))

results_lock = threading.Lock()
case_results = {}


def find_ct_path(case_dir):
    """Find the CT nifti file in a case's raw directory."""
    raw_dir = os.path.join(case_dir, "raw")
    if not os.path.isdir(raw_dir):
        return None
    # Primary: Fixed_CT_Volume.nii.gz
    ct_path = os.path.join(raw_dir, "Fixed_CT_Volume.nii.gz")
    if os.path.isfile(ct_path):
        return ct_path
    # Fallback: any CT nifti
    for f in os.listdir(raw_dir):
        if f.lower().startswith("ct") and f.endswith(".nii.gz"):
            return os.path.join(raw_dir, f)
        if "ct_volume" in f.lower() and f.endswith(".nii.gz"):
            return os.path.join(raw_dir, f)
    return None


def run_anatomy_pipeline(case_name, case_dir, gpu_id, log_file, force=False):
    """Step 1: Full anatomical segmentation + packing."""
    ct_path = find_ct_path(case_dir)
    if not ct_path:
        return False, f"No CT nifti found in {case_dir}/raw/"
    
    raw_dir = os.path.join(case_dir, "raw")
    
    # Clear old cached outputs if force
    if force:
        for f in ["packed_anatomical_tensor.h5", "packed_anatomical_tensor_registry.json", "primary_masks.h5"]:
            fpath = os.path.join(raw_dir, f)
            if os.path.isfile(fpath):
                os.remove(fpath)
                print(f"    [{case_name}] Removed cached {f}")
        seg_dir = os.path.join(raw_dir, "segmentations")
        if os.path.isdir(seg_dir):
            shutil.rmtree(seg_dir)
            print(f"    [{case_name}] Cleared segmentations/")
    
    env = BASE_ENV.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    cmd = [
        PYTHON_ENV,
        os.path.join(LN_RULES_DIR, "scripts/prepare_packed_case.py"),
        "--ct", ct_path,
        "--output-dir", raw_dir,
        "--gpu", str(gpu_id),
        "--skip-dag",  # We run Julia DAG separately
        "--force",
        "--retries", "5",
    ]
    
    print(f"    [{case_name}] GPU{gpu_id}: Running anatomy pipeline...", flush=True)
    with open(log_file, "a") as lf:
        lf.write(f"\n=== ANATOMY PIPELINE ({time.strftime('%H:%M:%S')}) ===\n")
        lf.write(f"CMD: {' '.join(cmd)}\n\n")
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=7200)
    
    if result.returncode != 0:
        return False, f"Anatomy pipeline failed with code {result.returncode}"
    
    # Verify outputs
    packed = os.path.join(raw_dir, "packed_anatomical_tensor.h5")
    registry = os.path.join(raw_dir, "packed_anatomical_tensor_registry.json")
    if not os.path.isfile(packed) or not os.path.isfile(registry):
        return False, "packed_anatomical_tensor.h5 or registry.json not produced"
    
    size_mb = os.path.getsize(packed) / 1e6
    if size_mb < 10:
        return False, f"packed_anatomical_tensor.h5 too small ({size_mb:.1f} MB)"
    
    print(f"    [{case_name}] GPU{gpu_id}: Anatomy pipeline done ({size_mb:.0f} MB)", flush=True)
    return True, f"OK ({size_mb:.0f} MB)"


def run_julia_ln_rules(case_name, case_dir, gpu_id, log_file, force=False):
    """Step 2: Run Julia GPU DAG VM for lymph node station rules."""
    raw_dir = os.path.join(case_dir, "raw")
    ln_h5 = os.path.join(raw_dir, "final_results_gpu.h5")
    
    # Clear old LN results if force
    if force and os.path.isfile(ln_h5):
        os.remove(ln_h5)
        print(f"    [{case_name}] Removed cached final_results_gpu.h5")
    
    # Check for required inputs
    packed = os.path.join(raw_dir, "packed_anatomical_tensor.h5")
    if not os.path.isfile(packed):
        # Check for primary_masks.h5
        primary = os.path.join(raw_dir, "primary_masks.h5")
        if not os.path.isfile(primary):
            return False, "No packed_anatomical_tensor.h5 or primary_masks.h5"
    
    env = BASE_ENV.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["JULIA_NUM_THREADS"] = "4"
    env["LN_RULES_DIR"] = LN_RULES_DIR
    
    cmd = [
        JULIA_BIN,
        f"--project={JULIA_PROJECT}",
        os.path.join(JULIA_PROJECT, "main/run_gpu_single.jl"),
        raw_dir,
    ]
    
    print(f"    [{case_name}] GPU{gpu_id}: Running Julia LN rules...", flush=True)
    t0 = time.time()
    with open(log_file, "a") as lf:
        lf.write(f"\n=== JULIA LN RULES ({time.strftime('%H:%M:%S')}) ===\n")
        lf.write(f"CMD: {' '.join(cmd)}\n\n")
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=3600)
    
    elapsed = time.time() - t0
    
    if result.returncode != 0:
        return False, f"Julia LN rules failed with code {result.returncode}"
    
    if not os.path.isfile(ln_h5):
        return False, "final_results_gpu.h5 not produced"
    
    size_mb = os.path.getsize(ln_h5) / 1e6
    if size_mb > 700:
        return False, f"final_results_gpu.h5 suspiciously large ({size_mb:.0f} MB) - likely corrupted"
    
    print(f"    [{case_name}] GPU{gpu_id}: LN rules done in {elapsed:.0f}s ({size_mb:.0f} MB)", flush=True)
    return True, f"OK ({size_mb:.0f} MB, {elapsed:.0f}s)"


def process_single_case(case_name, gpu_id, skip_anatomy=False, skip_ln=False, force=False, dry_run=False):
    """Process a single case end-to-end."""
    case_dir = os.path.join(CASES_DIR, case_name)
    if not os.path.isdir(case_dir):
        return {"status": "SKIP", "msg": f"Directory not found: {case_dir}"}
    
    log_file = os.path.join(LOG_DIR, f"{case_name}.log")
    ct_path = find_ct_path(case_dir)
    
    if not ct_path:
        return {"status": "SKIP", "msg": "No CT nifti file found"}
    
    if dry_run:
        return {"status": "DRY_RUN", "msg": f"Would process: CT={ct_path}, GPU={gpu_id}"}
    
    result = {"status": "OK", "anatomy": "SKIP", "ln_rules": "SKIP", "gpu": gpu_id}
    
    # Step 1: Anatomy pipeline
    if not skip_anatomy:
        ok, msg = run_anatomy_pipeline(case_name, case_dir, gpu_id, log_file, force=force)
        result["anatomy"] = msg
        if not ok:
            result["status"] = "FAIL"
            result["msg"] = f"Anatomy: {msg}"
            return result
    
    # Step 2: Julia LN rules
    if not skip_ln:
        ok, msg = run_julia_ln_rules(case_name, case_dir, gpu_id, log_file, force=force)
        result["ln_rules"] = msg
        if not ok:
            result["status"] = "FAIL"
            result["msg"] = f"LN rules: {msg}"
            return result
    
    result["status"] = "OK"
    return result


def worker(gpu_id, task_queue, skip_anatomy, skip_ln, force, dry_run):
    """Worker thread for a single GPU."""
    while True:
        try:
            case_name = task_queue.get_nowait()
        except queue.Empty:
            break
        
        t0 = time.time()
        print(f"\n[{time.strftime('%H:%M:%S')}] GPU{gpu_id}: === {case_name} ===", flush=True)
        
        try:
            result = process_single_case(case_name, gpu_id, skip_anatomy, skip_ln, force, dry_run)
        except Exception as e:
            result = {"status": "ERROR", "msg": str(e), "traceback": traceback.format_exc()}
        
        elapsed = time.time() - t0
        result["time_s"] = round(elapsed, 1)
        
        with results_lock:
            case_results[case_name] = result
        
        status = result["status"]
        print(f"[{time.strftime('%H:%M:%S')}] GPU{gpu_id}: {case_name} -> {status} ({elapsed:.0f}s)", flush=True)
        task_queue.task_done()


def main():
    parser = argparse.ArgumentParser(description="Reprocess all Prostata bimodal cases")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without running")
    parser.add_argument("--cases", type=str, help="Comma-separated list of specific cases (e.g., Pat3,Pat6)")
    parser.add_argument("--skip-anatomy", action="store_true", help="Skip anatomy segmentation (only run LN rules)")
    parser.add_argument("--skip-ln", action="store_true", help="Skip LN rules (only run anatomy)")
    parser.add_argument("--force", action="store_true", help="Force recomputation even if outputs exist")
    parser.add_argument("--single-gpu", type=int, help="Use only this GPU (default: use both 0 and 1)")
    args = parser.parse_args()
    
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Determine which cases to process
    if args.cases:
        case_names = [c.strip() for c in args.cases.split(",")]
    else:
        # All Pat* directories (excluding student/reference copies)
        case_names = sorted([
            d for d in os.listdir(CASES_DIR)
            if d.startswith("Pat") and os.path.isdir(os.path.join(CASES_DIR, d))
            and "_PETCT_" not in d  # Exclude student copies
        ], key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    
    # Filter to cases with CT available
    processable = []
    skipped = []
    for cn in case_names:
        ct = find_ct_path(os.path.join(CASES_DIR, cn))
        if ct:
            processable.append(cn)
        else:
            skipped.append(cn)
    
    print("=" * 80)
    print(f"PROSTATA BIMODAL FULL REPROCESSING")
    print(f"  Cases to process: {len(processable)}")
    print(f"  Cases skipped (no CT): {len(skipped)} {skipped}")
    print(f"  Skip anatomy: {args.skip_anatomy}")
    print(f"  Skip LN rules: {args.skip_ln}")
    print(f"  Force recompute: {args.force}")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Log directory: {LOG_DIR}")
    print("=" * 80)
    
    task_queue = queue.Queue()
    for cn in processable:
        task_queue.put(cn)
    
    t_start = time.time()
    
    if args.single_gpu is not None:
        # Single GPU mode
        worker(args.single_gpu, task_queue, args.skip_anatomy, args.skip_ln, args.force, args.dry_run)
    else:
        # Dual GPU mode
        t0 = threading.Thread(target=worker, args=(0, task_queue, args.skip_anatomy, args.skip_ln, args.force, args.dry_run), name="GPU0")
        t1 = threading.Thread(target=worker, args=(1, task_queue, args.skip_anatomy, args.skip_ln, args.force, args.dry_run), name="GPU1")
        t0.start()
        t1.start()
        t0.join()
        t1.join()
    
    total_time = time.time() - t_start
    
    # Summary
    print("\n" + "=" * 80)
    print(f"BATCH COMPLETE in {total_time:.0f}s ({total_time/3600:.1f}h)")
    print("=" * 80)
    
    ok_count = sum(1 for r in case_results.values() if r.get("status") == "OK")
    fail_count = sum(1 for r in case_results.values() if r.get("status") in ("FAIL", "ERROR"))
    skip_count = sum(1 for r in case_results.values() if r.get("status") in ("SKIP", "DRY_RUN"))
    
    print(f"\n{'Case':<15} {'Status':<8} {'GPU':<5} {'Time':<8} {'Anatomy':<30} {'LN Rules':<30}")
    print("-" * 100)
    for cn in processable:
        r = case_results.get(cn, {"status": "NOT_RUN"})
        gpu = r.get("gpu", "-")
        t = f"{r.get('time_s', 0):.0f}s"
        anat = r.get("anatomy", "-")[:28]
        ln = r.get("ln_rules", "-")[:28]
        status = r.get("status", "?")
        print(f"{cn:<15} {status:<8} {str(gpu):<5} {t:<8} {anat:<30} {ln:<30}")
    
    print(f"\nOK: {ok_count}  FAIL: {fail_count}  SKIP: {skip_count+len(skipped)}")
    
    # Save results
    results_file = os.path.join(LOG_DIR, "batch_results.json")
    with open(results_file, "w") as f:
        json.dump({"total_time_s": total_time, "cases": case_results, "skipped": skipped}, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    if fail_count > 0:
        print(f"\n[WARNING] {fail_count} cases FAILED. Check logs in {LOG_DIR}")
        sys.exit(1)


if __name__ == "__main__":
    main()
