#!/usr/bin/env python3
"""
orchestrate_prostata_ln_reprocess.py
====================================
End-to-end Dual-GPU Batch Orchestrator for Reprocessing Lymph Node Rules
across all 131 Prostata_bimodal CT cases.

Pipeline Per Case:
  1. Pure Julia GPU DagVm rule execution via batch_rerun_prostata_ln.jl (GPU 0 & GPU 1 in parallel)
     -> Produces <pat>/raw/final_results_gpu.h5 & pure_julia_gpu_time.json
  2. NIfTI & MRB Packaging via pack_and_mrb_single.py
     -> Exports <pat>/lymph_nodes/<station>.nii.gz, all_lymph_nodes.nii.gz, lymph_node_labels.json
     -> Generates verified 3D Slicer MRBs: <pat>/lymph_nodes/<pat>_LN_Validation.mrb, <pat>/<pat>.mrb,
        and data/ln_validation_output/mrb_files/<pat>_LN_Validation.mrb
  3. Preprocessed Volumes HDF5 Max Anatomy Update via repair_max_anatomy_all.py
     -> Updates BASELINE/max_anatomy.nii.gz & _meta_/max_anatomy_labels.json in preprocessed_volumes.h5
  4. Real-time logging & atomic progress tracking

Usage:
  python3 orchestrate_prostata_ln_reprocess.py [--force] [--postprocess-only] [--gpu 0|1]
"""

import os
import sys
import json
import time
import argparse
import subprocess
import threading
from pathlib import Path

# Paths
BASE_DIR = "/mnt/big/project_ssd/project_ssd/MedEye3d.jl"
CASES_DIR = os.path.join(BASE_DIR, "data/cases/Prostata_bimodal")
LN_RULES_DIR = "/mnt/big/project_ssd/project_ssd/lymph_node_rules"
JULIA_PROJECT = os.path.join(LN_RULES_DIR, "src/julia_gpu")
JULIA_SCRIPT = os.path.join(JULIA_PROJECT, "main/batch_rerun_prostata_ln.jl")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts/preprocessing")
SCRATCH_DIR = "/home/jm/.gemini/antigravity-cli/brain/c1ae0a4c-1bb7-4b4f-9433-0a638b84f621/scratch"

# Master progress file
MASTER_PROGRESS_FILE = os.path.join(CASES_DIR, "reprocess_ln_progress.json")

# Import packaging & repair tools
sys.path.insert(0, SCRATCH_DIR)
sys.path.insert(0, SCRIPTS_DIR)
from pack_and_mrb_single import process_case as pack_case_mrb
from repair_max_anatomy_all import repair_case as repair_case_h5


def get_candidate_cases():
    """Discover all cases with CT and primary_masks.h5."""
    all_dirs = [
        d for d in os.listdir(CASES_DIR)
        if d.startswith("Pat") and not d.startswith("Pat_")
        and "_PETCT_" not in d and "_edge_" not in d
        and os.path.isdir(os.path.join(CASES_DIR, d))
    ]
    all_dirs.sort(key=lambda x: int(x.replace("Pat", "")) if x.replace("Pat", "").isdigit() else 9999)

    valid_cases = []
    skipped = []
    for pat in all_dirs:
        raw_dir = os.path.join(CASES_DIR, pat, "raw")
        ct_file = os.path.join(raw_dir, "Fixed_CT_Volume.nii.gz")
        prim_h5 = os.path.join(raw_dir, "primary_masks.h5")
        if os.path.isfile(ct_file) and os.path.isfile(prim_h5):
            valid_cases.append(pat)
        else:
            skipped.append((pat, "No CT or primary_masks.h5"))

    return valid_cases, skipped


def load_master_progress():
    if os.path.isfile(MASTER_PROGRESS_FILE):
        try:
            with open(MASTER_PROGRESS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_master_progress(progress):
    try:
        tmp = MASTER_PROGRESS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(progress, f, indent=2)
        os.replace(tmp, MASTER_PROGRESS_FILE)
    except Exception as e:
        print(f"Warning: Failed to save master progress: {e}")


def postprocess_single_case(pat):
    """Unpack NIfTIs, build MRB, and update preprocessed_volumes.h5."""
    t0 = time.time()
    raw_dir = os.path.join(CASES_DIR, pat, "raw")
    h5_gpu = os.path.join(raw_dir, "final_results_gpu.h5")
    if not os.path.isfile(h5_gpu):
        return False, "Missing final_results_gpu.h5"

    try:
        # Step 1: Pack NIfTIs & MRB
        ok_pack = pack_case_mrb(pat)
        if not ok_pack:
            return False, "pack_and_mrb_single failed"

        # Step 2: Repair preprocessed_volumes.h5 max_anatomy
        res_h5 = repair_case_h5(pat, dry_run=False)

        dur = round(time.time() - t0, 1)
        return True, f"OK ({dur}s, h5={res_h5})"
    except Exception as e:
        return False, f"Exception: {e}"


def run_julia_worker(cases_list, gpu_id, force=False, log_file=None):
    """Run persistent Julia batch script on a designated GPU."""
    cases_json_path = os.path.join(CASES_DIR, f"cases_gpu{gpu_id}.json")
    with open(cases_json_path, "w") as f:
        json.dump(cases_list, f, indent=2)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        "julia",
        "-t4",
        f"--project={JULIA_PROJECT}",
        JULIA_SCRIPT,
        cases_json_path,
        str(gpu_id),  # Worker identifier for progress_gpu<gpu_id>.json
    ]
    if force:
        cmd.append("--force")

    if log_file:
        with open(log_file, "w") as lf:
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    else:
        p = subprocess.Popen(cmd, env=env)
    return p


def main():
    parser = argparse.ArgumentParser(description="Orchestrate Prostata Bimodal Lymph Node Reprocessing")
    parser.add_argument("--force", action="store_true", help="Force rerun already processed cases")
    parser.add_argument("--postprocess-only", action="store_true", help="Run only NIfTI/MRB/H5 packaging without GPU execution")
    parser.add_argument("--gpu", type=int, choices=[0, 1], help="Run on specific single GPU only")
    args = parser.parse_args()

    candidates, skipped = get_candidate_cases()
    print("=" * 80)
    print("PROSTATA BIMODAL LYMPH NODE REPROCESSING ORCHESTRATOR")
    print(f"Total Candidate Cases with CT: {len(candidates)}")
    if skipped:
        print(f"Skipped non-CT cases: {[s[0] for s in skipped]}")
    print("=" * 80)

    progress = load_master_progress()

    if args.postprocess_only:
        print("\n[POSTPROCESS ONLY MODE]")
        for i, pat in enumerate(candidates):
            print(f"[{i+1}/{len(candidates)}] Postprocessing {pat}...")
            ok, msg = postprocess_single_case(pat)
            status = "DONE" if ok else f"FAILED: {msg}"
            progress[pat] = {"status": status, "timestamp": time.time()}
            save_master_progress(progress)
            print(f"  Result: {status}")
        return

    # Filter cases needing GPU run
    cases_to_run = []
    for pat in candidates:
        if args.force or progress.get(pat, {}).get("status") != "COMPLETE":
            cases_to_run.append(pat)

    print(f"Cases requiring processing: {len(cases_to_run)} / {len(candidates)}")
    if not cases_to_run:
        print("All cases are already marked COMPLETE! Use --force to rerun.")
        return

    # Split cases across GPUs
    if args.gpu is not None:
        gpu_splits = {args.gpu: cases_to_run}
    else:
        # Balanced dual-GPU split
        gpu_splits = {
            0: [c for i, c in enumerate(cases_to_run) if i % 2 == 0],
            1: [c for i, c in enumerate(cases_to_run) if i % 2 == 1],
        }

    log_dir = os.path.join(CASES_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    workers = {}
    for gpu_id, cases in gpu_splits.items():
        if not cases:
            continue
        log_f = os.path.join(log_dir, f"gpu{gpu_id}_execution.log")
        print(f"Launching Worker on GPU {gpu_id}: {len(cases)} cases (Log: {log_f})")
        proc = run_julia_worker(cases, gpu_id, force=args.force, log_file=log_f)
        workers[gpu_id] = proc

    # Watcher & Post-processing loop
    print("\nStarting live watcher and post-processor...")
    completed_postprocessed = set()
    for pat, info in progress.items():
        if info.get("status") == "COMPLETE":
            completed_postprocessed.add(pat)

    while any(p.poll() is None for p in workers.values()):
        # Check progress files from each GPU
        for gpu_id in workers.keys():
            prog_file = os.path.join(CASES_DIR, f"progress_gpu{gpu_id}.json")
            if os.path.isfile(prog_file):
                try:
                    with open(prog_file, "r") as pf:
                        gpu_done = json.load(pf)
                    for pat in gpu_done:
                        if pat not in completed_postprocessed:
                            print(f"\n[GPU {gpu_id} finished {pat}] -> Running NIfTI unpack, MRB, and HDF5 max_anatomy...")
                            ok, msg = postprocess_single_case(pat)
                            if ok:
                                completed_postprocessed.add(pat)
                                progress[pat] = {
                                    "status": "COMPLETE",
                                    "gpu": gpu_id,
                                    "timestamp": time.time(),
                                    "details": msg,
                                }
                                save_master_progress(progress)
                                print(f"  ✅ {pat} fully processed ({len(completed_postprocessed)}/{len(candidates)})")
                            else:
                                print(f"  ❌ {pat} postprocess failed: {msg}")
                except Exception as e:
                    pass
        time.sleep(5)

    # Final post-processing pass for any remaining
    print("\nGPU Workers finished. Running final post-processing pass...")
    for pat in candidates:
        raw_dir = os.path.join(CASES_DIR, pat, "raw")
        h5_gpu = os.path.join(raw_dir, "final_results_gpu.h5")
        if os.path.isfile(h5_gpu) and pat not in completed_postprocessed:
            print(f"Final pass for {pat}...")
            ok, msg = postprocess_single_case(pat)
            if ok:
                completed_postprocessed.add(pat)
                progress[pat] = {"status": "COMPLETE", "timestamp": time.time(), "details": msg}
                save_master_progress(progress)

    print("\n" * 2 + "=" * 80)
    print("REPROCESSING COMPLETE SUMMARY")
    print(f"  Total Cases:       {len(candidates)}")
    print(f"  Completed:         {len(completed_postprocessed)} / {len(candidates)}")
    print(f"  Progress Log:      {MASTER_PROGRESS_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
