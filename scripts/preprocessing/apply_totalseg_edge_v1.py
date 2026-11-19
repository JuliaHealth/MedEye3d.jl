#!/usr/bin/env python3
"""
Recompute CT edge & diffusivity (`BASELINE/edge_ct`, `BASELINE/diffusivity_ct`)
using the repository's default TotalSegEdgeNet v1 (High LR) model while keeping
`BASELINE/diffusivity` (Both) and `BASELINE/diffusivity_pet` (PET) intact.

Supports single-file (--src/--dst) or cohort-wide (--cohort / --cases) runs with
xnat_cache syncing (--sync-cache) and optional async/batch XNAT upload (--upload-xnat).
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

import h5py
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CASES_DIR = os.path.join(REPO_ROOT, "data", "cases", "Prostata_bimodal")
CACHE_ROOT = os.path.expanduser("~/.medeye3d/xnat_cache/Prostata_bimodal_PIPELINE")
PROJ_ID = "Prostata_bimodal_PIPELINE"

sys.path.insert(0, SCRIPT_DIR)
from totalseg_edge_v1_model import (
    DEFAULT_V1_CKPT,
    infer_ct_edge_and_diffusivity_v1,
    load_v1_model,
)


def process_h5_ct_edges(h5_path: str, model, device: torch.device) -> tuple:
    with h5py.File(h5_path, "a") as f:
        bl = f["BASELINE"]
        ct_arr = bl["Fixed_CT_Volume_0.nii.gz"][:]
        old_diff = bl["diffusivity"][:] if "diffusivity" in bl else None
        excl = (old_diff == 0.0) if old_diff is not None else None

        edge_ct, diff_ct = infer_ct_edge_and_diffusivity_v1(
            ct_arr, device, model=model, exclusion_mask=excl,
            patch_size=64, step=48, batch_size=8,
        )

        # Preserve spatial attributes from Fixed_CT_Volume_0.nii.gz or existing diffusivity
        ref_ds = bl["diffusivity"] if "diffusivity" in bl else bl["Fixed_CT_Volume_0.nii.gz"]
        sp_attrs = {a: ref_ds.attrs[a] for a in ("spacing", "origin", "direction") if a in ref_ds.attrs}

        for k, arr in (("edge_ct", edge_ct), ("diffusivity_ct", diff_ct)):
            if k in bl:
                del bl[k]
            ds = bl.create_dataset(k, data=arr, compression="gzip", compression_opts=4, shuffle=True)
            for attr, val in sp_attrs.items():
                ds.attrs[attr] = val

    return float(edge_ct.mean()), float(diff_ct.mean())


def sync_to_xnat_cache(pat_id: str, src_h5: str, census: dict):
    info = census.get(pat_id)
    if not info:
        return
    subj_label = info.get("subj_label", "")
    if not subj_label:
        return
    dst_dir = os.path.join(CACHE_ROOT, subj_label)
    dst_h5 = os.path.join(dst_dir, f"{subj_label}_preprocessed.h5")
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copyfile(src_h5, dst_h5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="Single source preprocessed_volumes.h5")
    ap.add_argument("--dst", help="Single destination preprocessed_volumes.h5")
    ap.add_argument("--cohort", action="store_true", help="Process all Pat* cases in Prostata_bimodal")
    ap.add_argument("--cases", nargs="*", help="Specific Pat IDs (e.g. Pat27 Pat51)")
    ap.add_argument("--ckpt", default=DEFAULT_V1_CKPT)
    ap.add_argument("--sync-cache", action="store_true", help="Sync updated HDF5 into ~/.medeye3d/xnat_cache")
    ap.add_argument("--upload-xnat", action="store_true", help="Upload updated HDF5 to XNAT MEDEYE3D resource")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading TotalSegEdgeNet v1 on {dev} ({args.ckpt})...", flush=True)
    model = load_v1_model(dev, args.ckpt)

    if args.src and args.dst:
        os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
        if os.path.abspath(args.src) != os.path.abspath(args.dst):
            shutil.copyfile(args.src, args.dst)
        em, dm = process_h5_ct_edges(args.dst, model, dev)
        print(f"Done: {args.dst} (edge_ct_mean={em:.4f}, diff_ct_mean={dm:.4f})")
        return

    if args.cohort or args.cases:
        all_cases = sorted(
            [
                os.path.basename(p)
                for p in glob.glob(os.path.join(CASES_DIR, "Pat*"))
                if "_student" not in p and "_PETCT_" not in p and "_totalseg_edge" not in p
                and os.path.isfile(os.path.join(p, "preprocessed_volumes.h5"))
            ],
            key=lambda s: int(re.search(r"\d+", s).group()),
        )
        targets = [c for c in all_cases if c in set(args.cases)] if args.cases else all_cases

        census = {}
        xnat_mgr = None
        if args.sync_cache or args.upload_xnat:
            from batch_process_cohort_dual_gpu import CENSUS_PATH, XnatManager
            with open(CENSUS_PATH, "r") as cf:
                census = json.load(cf)
            if args.upload_xnat:
                xnat_mgr = XnatManager()

        t_all = time.time()
        ok, uploaded = 0, 0
        for idx, pat in enumerate(targets, 1):
            t0 = time.time()
            h5_p = os.path.join(CASES_DIR, pat, "preprocessed_volumes.h5")
            em, dm = process_h5_ct_edges(h5_p, model, dev)
            ok += 1
            if args.sync_cache:
                sync_to_xnat_cache(pat, h5_p, census)
            if xnat_mgr is not None and pat in census:
                info = census[pat]
                sid, slbl = info.get("subj_id", ""), info.get("subj_label", "")
                if sid and slbl:
                    xnat_mgr.request("PUT", f"/data/projects/{PROJ_ID}/subjects/{sid}/resources/MEDEYE3D")
                    uri = f"/data/projects/{PROJ_ID}/subjects/{sid}/resources/MEDEYE3D/files/{slbl}_preprocessed.h5?overwrite=true"
                    if xnat_mgr.upload_file_streaming(uri, h5_p):
                        uploaded += 1
            print(f"[{idx}/{len(targets)}] {pat}: edge_ct={em:.4f} diff_ct={dm:.4f} ({time.time()-t0:.1f}s)", flush=True)
        print(f"Cohort CT edge v1 complete: ok={ok}, uploaded={uploaded}, total={time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
