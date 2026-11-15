#!/usr/bin/env python3
"""
Run TotalSegmentator + packing for cases that have incomplete segmentations.
Does NOT delete existing NVSeg/MuscleMap masks — only adds TotalSeg masks and rebuilds
packed_anatomical_tensor.h5 and primary_masks.h5.

Usage:
    python3 scripts/preprocessing/run_totalseg_incomplete_cases.py \
        --cases "Pat25,Pat26,..." --gpu 0

Or use both GPUs:
    python3 scripts/preprocessing/run_totalseg_incomplete_cases.py \
        --cases "Pat25,Pat26,..." --dual-gpu
"""
import os
import sys
import time
import shutil
import argparse
import subprocess
import threading
from queue import Queue

CASES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/cases/Prostata_bimodal")
LN_RULES_DIR = "/mnt/big/project_ssd/project_ssd/lymph_node_rules"
PYTHON_ENV = "/mnt/big/project_ssd/project_ssd/psma_mag3/.venv/bin/python3"
if not os.path.isfile(PYTHON_ENV):
    PYTHON_ENV = shutil.which("python3") or sys.executable

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


BASE_ENV = os.environ.copy()
_load_dotenv(BASE_ENV, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.env"))
VENV_LIB = os.path.join(os.path.dirname(os.path.dirname(PYTHON_ENV)), "lib")
BASE_ENV["LD_LIBRARY_PATH"] = VENV_LIB + ":" + BASE_ENV.get("LD_LIBRARY_PATH", "")
BASE_ENV["PATH"] = os.path.join(os.path.dirname(PYTHON_ENV)) + ":" + BASE_ENV.get("PATH", "")
BASE_ENV.setdefault("TOTALSEG_LICENSE", os.environ.get("TOTALSEG_LICENSE", os.environ.get("TOTALSEG_LICENSE_NUMBER", "")))
BASE_ENV.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", "")))
BASE_ENV["HF_HUB_CACHE"] = "/mnt/big/project_ssd/project_ssd/huggingface_cache"
BASE_ENV["TORCH_HOME"] = "/mnt/big/project_ssd/project_ssd/.torch"


def find_ct_path(case_dir):
    raw_dir = os.path.join(case_dir, "raw")
    ct = os.path.join(raw_dir, "Fixed_CT_Volume.nii.gz")
    if os.path.isfile(ct):
        return ct
    return None


def process_case(case_name, gpu_id, log_file):
    """Run TotalSeg + packing for a single case."""
    case_dir = os.path.join(CASES_DIR, case_name)
    raw_dir = os.path.join(case_dir, "raw")
    ct_path = find_ct_path(case_dir)
    
    if not ct_path:
        print(f"    [{case_name}] No CT found, skipping")
        return False, "No CT"
    
    # Remove markers and packed outputs to force re-run, but KEEP segmentation files
    for f in ["packed_anatomical_tensor.h5", "packed_anatomical_tensor_registry.json",
              "primary_masks.h5", ".gpu_inference_completed"]:
        fpath = os.path.join(raw_dir, f)
        if os.path.isfile(fpath):
            os.remove(fpath)
            print(f"    [{case_name}] Removed {f}")
    
    env = BASE_ENV.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    cmd = [
        PYTHON_ENV,
        os.path.join(LN_RULES_DIR, "scripts/prepare_packed_case.py"),
        "--ct", ct_path,
        "--output-dir", raw_dir,
        "--gpu", str(gpu_id),
        "--skip-dag",  # We run Julia DAG separately
        "--force",     # Force recomputation
        "--retries", "5",
    ]
    
    t0 = time.time()
    print(f"    [{case_name}] GPU{gpu_id}: Running anatomy pipeline...", flush=True)
    
    with open(log_file, "a") as lf:
        lf.write(f"\n{'='*60}\n")
        lf.write(f"=== {case_name} ({time.strftime('%H:%M:%S')}) GPU{gpu_id} ===\n")
        lf.write(f"CMD: {' '.join(cmd)}\n\n")
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=7200)
    
    elapsed = time.time() - t0
    
    if result.returncode != 0:
        print(f"    [{case_name}] GPU{gpu_id}: FAILED (code {result.returncode}, {elapsed:.0f}s)")
        return False, f"Failed (code {result.returncode})"
    
    # Verify outputs
    packed = os.path.join(raw_dir, "packed_anatomical_tensor.h5")
    pm = os.path.join(raw_dir, "primary_masks.h5")
    
    if not os.path.isfile(packed):
        print(f"    [{case_name}] GPU{gpu_id}: FAILED - no packed tensor produced")
        return False, "No packed tensor"
    
    packed_mb = os.path.getsize(packed) / 1e6
    pm_mb = os.path.getsize(pm) / 1e6 if os.path.isfile(pm) else 0
    
    # Verify primary_masks has enough masks
    import h5py
    try:
        with h5py.File(pm, "r") as f:
            n_masks = len(f["masks"].keys()) if "masks" in f else 0
            has_trachea = "trachea" in f.get("masks", {})
    except Exception:
        n_masks = 0
        has_trachea = False
    
    status = f"OK ({elapsed:.0f}s, packed={packed_mb:.0f}MB, primary={pm_mb:.0f}MB, masks={n_masks}, trachea={has_trachea})"
    print(f"    [{case_name}] GPU{gpu_id}: {status}", flush=True)
    
    if n_masks < 300:
        print(f"    [{case_name}] WARNING: Only {n_masks} masks in primary_masks.h5!")
        return False, f"Incomplete ({n_masks} masks)"
    
    return True, status


def worker(gpu_id, task_queue, log_file, results):
    while not task_queue.empty():
        try:
            case_name = task_queue.get_nowait()
        except Exception:
            break
        
        print(f"\n[{time.strftime('%H:%M:%S')}] GPU{gpu_id}: === {case_name} ===", flush=True)
        ok, msg = process_case(case_name, gpu_id, log_file)
        results.append({"case": case_name, "ok": ok, "msg": msg, "gpu": gpu_id})
        task_queue.task_done()


def main():
    parser = argparse.ArgumentParser(description="Run TotalSeg for incomplete cases")
    parser.add_argument("--cases", required=True, help="Comma-separated case names")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID (default: 0)")
    parser.add_argument("--dual-gpu", action="store_true", help="Use both GPU 0 and GPU 1")
    parser.add_argument("--dry-run", action="store_true", help="Just list cases, don't process")
    args = parser.parse_args()
    
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    
    # Filter to only cases that actually need processing
    processable = []
    for c in cases:
        case_dir = os.path.join(CASES_DIR, c)
        if not os.path.isdir(case_dir):
            print(f"  SKIP {c}: no case directory")
            continue
        ct = find_ct_path(case_dir)
        if not ct:
            print(f"  SKIP {c}: no CT")
            continue
        processable.append(c)
    
    print(f"\nProcessable cases: {len(processable)}")
    for c in processable:
        seg_dir = os.path.join(CASES_DIR, c, "raw", "segmentations")
        n = len([f for f in os.listdir(seg_dir) if f.endswith('.nii.gz')]) if os.path.isdir(seg_dir) else 0
        print(f"  {c}: {n} existing segs")
    
    if args.dry_run:
        print("\nDRY RUN - not processing")
        return
    
    log_file = "/tmp/totalseg_incomplete.log"
    task_queue = Queue()
    for c in processable:
        task_queue.put(c)
    
    results = []
    t_start = time.time()
    
    if args.dual_gpu:
        print(f"\nStarting dual-GPU processing ({len(processable)} cases)...")
        t0 = threading.Thread(target=worker, args=(0, task_queue, log_file, results), name="GPU0")
        t1 = threading.Thread(target=worker, args=(1, task_queue, log_file, results), name="GPU1")
        t0.start()
        t1.start()
        t0.join()
        t1.join()
    else:
        print(f"\nStarting single-GPU processing ({len(processable)} cases) on GPU {args.gpu}...")
        worker(args.gpu, task_queue, log_file, results)
    
    # Summary
    elapsed = time.time() - t_start
    ok_count = sum(1 for r in results if r["ok"])
    fail_count = sum(1 for r in results if not r["ok"])
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {ok_count} OK, {fail_count} FAILED out of {len(results)} processed")
    print(f"Total time: {elapsed/60:.1f} minutes")
    
    if fail_count > 0:
        print("\nFailed cases:")
        for r in results:
            if not r["ok"]:
                print(f"  {r['case']}: {r['msg']}")
    
    print(f"\nLog: {log_file}")


if __name__ == "__main__":
    main()
