#!/usr/bin/env python3
"""
Compute and add missing `BASELINE/diffusivity_pet` (and `BASELINE/edge_pet`) to
Prostata_bimodal `preprocessed_volumes.h5` files using the TotalSegEdgePDE_PET
model (`checkpoints/totalseg_edge_pde_pet/best_model.pth`).
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
SEMIAUTO_DIR = "/mnt/big/project_ssd/semiautomatic"
_REPO_PET_CKPT = os.path.join(REPO_ROOT, "assets", "models", "totalseg_edge_pde_pet", "best_model.pth")
PET_CKPT = _REPO_PET_CKPT if os.path.isfile(_REPO_PET_CKPT) else os.path.join(SEMIAUTO_DIR, "checkpoints/totalseg_edge_pde_pet/best_model.pth")
CHAMPION_EPS_D = 1.0e-5


def run_sliding_pet(model, pet_arr: np.ndarray, dev: torch.device,
                    patch_size: int = 64, step: int = 48, batch_size: int = 8) -> tuple:
    pet_norm = (np.clip(pet_arr.astype(np.float32), 0.0, 20.0) / 20.0).astype(np.float32)
    D, H, W = pet_norm.shape
    out_edge = torch.zeros((D, H, W), dtype=torch.float32, device=dev)
    weight_map = torch.zeros((D, H, W), dtype=torch.float32, device=dev)

    z = np.sin(np.linspace(0, np.pi, patch_size))
    y = np.sin(np.linspace(0, np.pi, patch_size))
    x = np.sin(np.linspace(0, np.pi, patch_size))
    w3d = torch.from_numpy((z[:, None, None] * y[None, :, None] * x[None, None, :]).astype(np.float32)).to(dev)
    w3d = torch.clamp(w3d, min=1e-4)

    d_steps = list(range(0, max(1, D - patch_size + 1), step))
    if not d_steps or d_steps[-1] != max(0, D - patch_size):
        d_steps.append(max(0, D - patch_size))
    h_steps = list(range(0, max(1, H - patch_size + 1), step))
    if not h_steps or h_steps[-1] != max(0, H - patch_size):
        h_steps.append(max(0, H - patch_size))
    w_steps = list(range(0, max(1, W - patch_size + 1), step))
    if not w_steps or w_steps[-1] != max(0, W - patch_size):
        w_steps.append(max(0, W - patch_size))

    patches = [(d, h, w) for d in d_steps for h in h_steps for w in w_steps]
    pet_t = torch.from_numpy(pet_norm).to(dev)

    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            coords = patches[i:i + batch_size]
            b_inp = torch.empty((len(coords), 1, patch_size, patch_size, patch_size), dtype=torch.float32, device=dev)
            for j, (d, h, w) in enumerate(coords):
                b_inp[j, 0] = pet_t[d:d + patch_size, h:h + patch_size, w:w + patch_size]
            e_batch, _, _ = model.forward_edge(b_inp)
            for j, (d, h, w) in enumerate(coords):
                out_edge[d:d + patch_size, h:h + patch_size, w:w + patch_size] += e_batch[j] * w3d
                weight_map[d:d + patch_size, h:h + patch_size, w:w + patch_size] += w3d

        out_edge /= torch.clamp(weight_map, min=1e-6)
        edge_pet = torch.clamp(out_edge, 0.0, 1.0).cpu().numpy().astype(np.float32)

    diff_pet = np.clip(
        CHAMPION_EPS_D + (1.0 - CHAMPION_EPS_D) * np.exp(-9.4069 * edge_pet),
        CHAMPION_EPS_D, 1.0
    ).astype(np.float32)
    del pet_t, out_edge, weight_map, w3d
    torch.cuda.empty_cache()
    return edge_pet, diff_pet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--cohort", action="store_true")
    ap.add_argument("--sync-cache", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    _model_dir = SCRIPT_DIR if os.path.isfile(os.path.join(SCRIPT_DIR, "totalseg_edge_pde_pet_model.py")) else SEMIAUTO_DIR
    sys.path.insert(0, _model_dir)
    from totalseg_edge_pde_pet_model import TotalSegEdgePDE_PET

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m_pet = TotalSegEdgePDE_PET().to(dev)
    p_st = torch.load(PET_CKPT, map_location=dev, weights_only=False)
    m_pet.load_state_dict(p_st.get("model_state_dict", p_st))
    m_pet.eval()

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
    if args.sync_cache:
        from batch_process_cohort_dual_gpu import CENSUS_PATH
        with open(CENSUS_PATH, "r") as cf:
            census = json.load(cf)

    t_all = time.time()
    updated = 0
    for idx, pat in enumerate(targets, 1):
        t0 = time.time()
        h5_p = os.path.join(CASES_DIR, pat, "preprocessed_volumes.h5")
        with h5py.File(h5_p, "a") as f:
            bl = f["BASELINE"]
            if not args.force and "diffusivity_pet" in bl and "edge_pet" in bl:
                continue
            pet_arr = bl["SUV_PET_Image_0.nii.gz"][:]
            edge_pet, diff_pet = run_sliding_pet(m_pet, pet_arr, dev)
            ref_ds = bl["diffusivity"] if "diffusivity" in bl else bl["SUV_PET_Image_0.nii.gz"]
            sp_attrs = {a: ref_ds.attrs[a] for a in ("spacing", "origin", "direction") if a in ref_ds.attrs}
            for k, arr in (("edge_pet", edge_pet), ("diffusivity_pet", diff_pet)):
                if k in bl:
                    del bl[k]
                ds = bl.create_dataset(k, data=arr, compression="gzip", compression_opts=4, shuffle=True)
                for a, v in sp_attrs.items():
                    ds.attrs[a] = v
            updated += 1
        if args.sync_cache and pat in census:
            slbl = census[pat].get("subj_label", "")
            if slbl:
                dst_dir = os.path.join(CACHE_ROOT, slbl)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copyfile(h5_p, os.path.join(dst_dir, f"{slbl}_preprocessed.h5"))
        print(f"[{idx}/{len(targets)}] {pat}: edge_pet={edge_pet.mean():.4f} diff_pet={diff_pet.mean():.4f} ({time.time()-t0:.1f}s)", flush=True)
    print(f"PET edge complete: updated={updated}/{len(targets)} in {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
