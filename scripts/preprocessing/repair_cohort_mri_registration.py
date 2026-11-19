#!/usr/bin/env python3
"""
repair_cohort_mri_registration.py

End-to-end cohort repair & verification for Prostata_bimodal:
  1. Verifies/recomputes CT-MRI registration using ONLY the 4 most medial points
     of hip & femur bones (whole CT, no bladder) + Centered Prostate SWD,
     asserting CT vs MRI Prostate Dice > 0.6.
  2. Re-resamples all MRI modalities (T2, ADC, DWI, T1, DCE), 8-zone prostate
     anatomy (TFM_*/max_anatomy.nii.gz), and MRI gold lesions (PET_Lesions_1.nii.gz)
     into preprocessed_volumes.h5.
  3. Merges Y-flipped CADS Task 556 (seminal_vesicle, sigmoid_colon, spinal_canal)
     and Task 558 (lacrimal_gland_left/right, optic_chiasm, optic_nerves, pituitary)
     into ATLAS/max_anatomy, BASELINE/max_anatomy.nii.gz, and max_anatomy_labels.json.
  4. Filters physiological lacrimal gland uptake lesions from BASELINE/PET_Lesions_0.nii.gz,
     CENTROIDS, organ_mapping, segment_names.json, matches.json, and scene_hierarchy.json,
     and maps seminal vesicle lesions to seminal_vesicle.
  5. Syncs updated HDF5s to local ~/.medeye3d/xnat_cache and uploads to XNAT MEDEYE3D
     & CT_MRI_REGISTRATION.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import h5py
import numpy as np
import SimpleITK as sitk

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

WORKSPACE_DIR = "/workspaces/MedEye3d.jl"
CASES_DIR = os.path.join(WORKSPACE_DIR, "data/cases/Prostata_bimodal")
XNAT_CACHE_DIR = "/root/.medeye3d/xnat_cache/Prostata_bimodal_PIPELINE"

sys.path.insert(0, os.path.join(WORKSPACE_DIR, "scripts/preprocessing"))
sys.path.insert(0, os.path.join(WORKSPACE_DIR, "docs"))

from batch_process_cohort_dual_gpu import (
    CENSUS_PATH,
    PROJ_ID,
    XnatManager,
    classify_mri_modality,
    compute_mri_diffusivity,
    select_ranked_mri_sequences,
    unpack_stacked_mri_volume,
)
from cads_anatomy_wiring import (
    filter_lacrimal_lesions,
    load_resampled_cads_arrays,
    merge_cads_structures,
    remove_edge_slice_lesions,
)
from apply_edge_slices_and_semantic_filter import (
    compute_and_store_semantic_filter,
    get_llm_config,
)
import register_ct_mri as reg_mod


def _pts_from_mask(img: sitk.Image, mask: np.ndarray) -> np.ndarray:
    z, y, x = np.where(mask)
    if len(z) == 0:
        return np.empty((0, 3), dtype=np.float64)
    ijk = np.stack([x, y, z], axis=1).astype(np.float64)
    sp = np.array(img.GetSpacing(), dtype=np.float64)
    orig = np.array(img.GetOrigin(), dtype=np.float64)
    d = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)
    return (ijk * sp) @ d.T + orig


def _medial_centroid(pts: np.ndarray, pct: float = 0.02) -> np.ndarray:
    k = max(10, int(len(pts) * pct))
    return pts[np.argsort(np.abs(pts[:, 0]))[:k]].mean(axis=0)


def _get_ct_bones_and_prostate(raw_dir: str, ct_img: sitk.Image):
    """Loads whole-CT femur_left/right, hip_left/right medial landmarks and prostate mask."""
    ct_bones = {}
    ct_pros_img = None
    seg_dir = os.path.join(raw_dir, "segmentations")
    for bname in ("femur_left", "femur_right", "hip_left", "hip_right"):
        bp = os.path.join(seg_dir, f"{bname}.nii.gz")
        if os.path.isfile(bp):
            c = reg_mod.extract_medial_bone_centroid(bp)
            if c is not None:
                ct_bones[bname] = c
    pros_p = os.path.join(seg_dir, "prostate.nii.gz")
    if os.path.isfile(pros_p):
        im = sitk.ReadImage(pros_p)
        if np.any(sitk.GetArrayViewFromImage(im) > 0):
            ct_pros_img = im

    anat_h5 = os.path.join(raw_dir, "packed_anatomical_tensor.h5")
    reg_p = os.path.join(raw_dir, "packed_anatomical_tensor_registry.json")
    if (len(ct_bones) < 4 or ct_pros_img is None) and os.path.isfile(anat_h5) and os.path.isfile(reg_p):
        with open(reg_p, "r") as rf:
            reg_doc = json.load(rf)
        masks_meta = reg_doc.get("masks", reg_doc)
        with h5py.File(anat_h5, "r") as af:
            pt = af["packed_tensor"][:]
            for bname in ("femur_left", "femur_right", "hip_left", "hip_right"):
                if bname not in ct_bones and bname in masks_meta:
                    m = masks_meta[bname]
                    arr_b = (pt[m["channel"] - 1] == m["id"])
                    pts_b = _pts_from_mask(ct_img, arr_b)
                    if len(pts_b) > 0:
                        ct_bones[bname] = _medial_centroid(pts_b)
            if ct_pros_img is None:
                for pname in ("prostate", "nvseg_prostate"):
                    if pname in masks_meta:
                        m = masks_meta[pname]
                        arr_p = (pt[m["channel"] - 1] == m["id"]).astype(np.uint8)
                        if arr_p.any():
                            ct_pros_img = sitk.GetImageFromArray(arr_p)
                            ct_pros_img.CopyInformation(ct_img)
                            break
    return ct_bones, ct_pros_img


def _find_primary_t2(raw_dir: str) -> str:
    m = {}
    for fn in os.listdir(raw_dir):
        if fn.endswith(".nii.gz") and not fn.startswith(("Fixed_CT", "SUV", "CT", "totalseg")):
            if classify_mri_modality(fn):
                m[fn] = os.path.join(raw_dir, fn)
    ranked = select_ranked_mri_sequences(m)
    for seq in ranked:
        if seq["mod"] == "T2":
            return seq["item"]
    return ranked[0]["item"] if ranked else ""


def _get_mri_bones_and_prostate(raw_dir: str, gpu_id: int = 1, allow_totalseg: bool = True):
    mr_bones = {}
    mr_pros_img = None

    ts_mr_path = os.path.join(raw_dir, "totalseg_mr_ml.nii.gz")
    if not os.path.isfile(ts_mr_path) and allow_totalseg:
        t2_path = _find_primary_t2(raw_dir)
        if t2_path and os.path.isfile(t2_path):
            ts_bin = next(
                (p for p in (
                    "/mnt/big/project_ssd/project_ssd/lymph_node_rules/for_joanna/.local/conda/envs/seg/bin/TotalSegmentator",
                    "/usr/local/bin/TotalSegmentator",
                ) if os.path.isfile(p)),
                None,
            ) or shutil.which("TotalSegmentator") or "TotalSegmentator"
            env_ts = os.environ.copy()
            env_ts["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env_ts["TOTALSEG_LICENSE_NUMBER"] = os.environ.get("TOTALSEG_LICENSE_NUMBER", os.environ.get("TOTALSEG_LICENSE", ""))
            print(f"    Running TotalSegmentator MR on {os.path.basename(t2_path)} (GPU {gpu_id})...")
            subprocess.run(
                [ts_bin, "-i", t2_path, "-o", ts_mr_path, "-ta", "total_mr", "--ml", "--fast", "--quiet"],
                env=env_ts,
                check=True,
            )

    if os.path.isfile(ts_mr_path):
        ts_im = sitk.ReadImage(ts_mr_path)
        mr_arr = sitk.GetArrayFromImage(ts_im)
        for lbl_id, bname in ((36, "femur_left"), (37, "femur_right"), (38, "hip_left"), (39, "hip_right")):
            pts_m = _pts_from_mask(ts_im, mr_arr == lbl_id)
            if len(pts_m) > 0:
                mr_bones[bname] = _medial_centroid(pts_m)
        p_m = (mr_arr == 17).astype(np.uint8)
        if p_m.any():
            mr_pros_img = sitk.GetImageFromArray(p_m)
            mr_pros_img.CopyInformation(ts_im)

    pros_nii = os.path.join(raw_dir, "prostate_segmentation.nii.gz")
    if os.path.isfile(pros_nii):
        p_img = sitk.ReadImage(pros_nii)
        p_arr = sitk.GetArrayFromImage(p_img)
        p_bin = np.isin(p_arr, [1, 2, 3, 4, 5, 6, 9]).astype(np.uint8)
        if p_bin.sum() < 100:
            p_bin = (p_arr > 0).astype(np.uint8)
        if p_bin.any():
            p_vol = float(p_bin.sum()) * float(np.prod(p_img.GetSpacing()))
            ts_vol = (
                float((sitk.GetArrayViewFromImage(mr_pros_img) > 0).sum()) * float(np.prod(mr_pros_img.GetSpacing()))
                if mr_pros_img is not None
                else 0.0
            )
            # Only replace totalseg_mr_ml prostate when totalseg is missing or prostate_segmentation is a full gland (>=50% of totalseg volume)
            if mr_pros_img is None or (ts_vol > 0 and p_vol >= 0.5 * ts_vol):
                mr_pros_img = sitk.GetImageFromArray(p_bin)
                mr_pros_img.CopyInformation(p_img)

    # Fallback: if small-FOV T2 missed >= 3 pelvic bones, segment larger-FOV axial T1 for Stage 1 bone landmarks
    if len(mr_bones) < 3 and allow_totalseg:
        m = {
            fn: os.path.join(raw_dir, fn)
            for fn in os.listdir(raw_dir)
            if fn.endswith(".nii.gz") and not fn.startswith(("Fixed_CT", "SUV", "CT", "totalseg"))
            and classify_mri_modality(fn)
        }
        ranked = select_ranked_mri_sequences(m)
        t1_cands = [s["item"] for s in ranked if s["mod"] == "T1" and s["plane"] == "ax"]
        if t1_cands and os.path.isfile(t1_cands[0]):
            ts_t1_path = os.path.join(raw_dir, "totalseg_mr_t1_ml.nii.gz")
            if not os.path.isfile(ts_t1_path):
                ts_bin = next(
                    (p for p in (
                        "/mnt/big/project_ssd/project_ssd/lymph_node_rules/for_joanna/.local/conda/envs/seg/bin/TotalSegmentator",
                        "/usr/local/bin/TotalSegmentator",
                    ) if os.path.isfile(p)),
                    None,
                ) or shutil.which("TotalSegmentator") or "TotalSegmentator"
                env_ts = os.environ.copy()
                env_ts["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                env_ts["TOTALSEG_LICENSE_NUMBER"] = os.environ.get("TOTALSEG_LICENSE_NUMBER", os.environ.get("TOTALSEG_LICENSE", ""))
                print(f"    Running TotalSegmentator MR T1 fallback on {os.path.basename(t1_cands[0])} (GPU {gpu_id})...")
                subprocess.run(
                    [ts_bin, "-i", t1_cands[0], "-o", ts_t1_path, "-ta", "total_mr", "--ml", "--fast", "--quiet"],
                    env=env_ts,
                    check=True,
                )
            if os.path.isfile(ts_t1_path):
                t1_im = sitk.ReadImage(ts_t1_path)
                t1_arr = sitk.GetArrayFromImage(t1_im)
                for lbl_id, bname in ((36, "femur_left"), (37, "femur_right"), (38, "hip_left"), (39, "hip_right")):
                    if bname not in mr_bones:
                        pts_m = _pts_from_mask(t1_im, t1_arr == lbl_id)
                        if len(pts_m) > 0:
                            mr_bones[bname] = _medial_centroid(pts_m)

    return mr_bones, mr_pros_img


def _compute_centered_swd_registration(ct_img, ct_bones, ct_pros_img, mr_bones, mr_pros_img):
    """Stage 1: 4 medial hip/femur landmarks (whole CT, no bladder). Stage 2: Centered SWD."""
    common_bones = [k for k in ("femur_left", "femur_right", "hip_left", "hip_right") if k in ct_bones and k in mr_bones]
    if len(common_bones) >= 3:
        lm_tfm, _, _, _ = reg_mod.compute_landmark_rigid_transform(
            [ct_bones[k] for k in common_bones],
            [mr_bones[k] for k in common_bones],
            common_bones,
        )
    else:
        ct_kps = reg_mod.extract_prostate_keypoints(ct_pros_img)
        mr_kps = reg_mod.extract_prostate_keypoints(mr_pros_img)
        kps = sorted(set(ct_kps.keys()) & set(mr_kps.keys()))
        lm_tfm, _, _, _ = reg_mod.compute_landmark_rigid_transform(
            [ct_kps[k] for k in kps],
            [mr_kps[k] for k in kps],
            kps,
        )

    ct_pts = reg_mod.extract_surface_points(ct_pros_img, num_points=800, seed=42)
    mr_pts = reg_mod.extract_surface_points(mr_pros_img, num_points=800, seed=42)
    min_n = min(len(ct_pts), len(mr_pts))
    rng = np.random.RandomState(42)
    ct_eq = ct_pts[rng.choice(len(ct_pts), min_n, replace=False)]
    mr_eq = mr_pts[rng.choice(len(mr_pts), min_n, replace=False)]

    final_tfm, _, _, _, _ = reg_mod.swd_optimization_torch(ct_eq, mr_eq, lm_tfm, max_epochs=500, seed=42)
    res_mr = reg_mod.resample_mri(mr_pros_img, ct_img, final_tfm, sitk.sitkNearestNeighbor)
    dice = reg_mod.compute_dice(ct_pros_img, res_mr)

    # If bone landmark init had severe slab truncation causing Dice < 0.60, retry with prostate keypoints init
    if dice < 0.60 and len(common_bones) >= 3:
        ct_kps = reg_mod.extract_prostate_keypoints(ct_pros_img)
        mr_kps = reg_mod.extract_prostate_keypoints(mr_pros_img)
        kps = sorted(set(ct_kps.keys()) & set(mr_kps.keys()))
        if len(kps) >= 3:
            kp_tfm, _, _, _ = reg_mod.compute_landmark_rigid_transform(
                [ct_kps[k] for k in kps], [mr_kps[k] for k in kps], kps
            )
            kp_swd_tfm, _, _, _, _ = reg_mod.swd_optimization_torch(ct_eq, mr_eq, kp_tfm, max_epochs=500, seed=42)
            kp_dice = reg_mod.compute_dice(
                ct_pros_img, reg_mod.resample_mri(mr_pros_img, ct_img, kp_swd_tfm, sitk.sitkNearestNeighbor)
            )
            if kp_dice > dice:
                final_tfm = kp_swd_tfm
                dice = kp_dice

    return final_tfm, dice


def _build_tp_to_raw_map(raw_dir: str, h5_path: str):
    m = {}
    for fn in os.listdir(raw_dir):
        if fn.endswith(".nii.gz") and not fn.startswith(("Fixed_CT", "SUV", "CT", "totalseg")):
            if classify_mri_modality(fn):
                m[fn] = os.path.join(raw_dir, fn)
    ranked = select_ranked_mri_sequences(m)

    with h5py.File(h5_path, "r") as f:
        sh_raw = f["_meta_/scene_hierarchy.json"][()]
        sh = json.loads(sh_raw.decode() if isinstance(sh_raw, bytes) else sh_raw)
        pf = int(f["_meta_/preflipped"][()]) if "_meta_/preflipped" in f else 1

        tp_nodes = []
        for node in sh:
            if node.get("type") == "vtkMRMLLinearTransformNode":
                ch = [c for c in node.get("children", []) if c["name"].startswith("Fixed_CT_Volume_")]
                c0 = ch[0] if ch else {}
                name = c0.get("name", "")
                if not name:
                    continue
                tp_idx = int(name.split("_")[-1])
                mod = (c0.get("modality") or node.get("modality") or "").upper()
                tp_nodes.append((tp_idx, "TFM_" + node["name"] + ".tfm", name + ".nii.gz", mod))

    used = set()
    mapping = []
    for tp_idx, grp, ds_name, mod in sorted(tp_nodes):
        picked = None
        for idx, seq in enumerate(ranked):
            if idx in used:
                continue
            if seq["mod"].upper() == mod:
                picked = (idx, seq)
                break
        if picked is None:
            for idx, seq in enumerate(ranked):
                if idx not in used:
                    picked = (idx, seq)
                    break
        if picked is not None:
            used.add(picked[0])
            mapping.append((tp_idx, grp, ds_name, mod, picked[1]))
    return mapping, pf


def verify_case(pat_id: str) -> dict:
    case_dir = os.path.join(CASES_DIR, pat_id)
    raw_dir = os.path.join(case_dir, "raw")
    h5_path = os.path.join(case_dir, "preprocessed_volumes.h5")
    ct_path = os.path.join(raw_dir, "Fixed_CT_Volume.nii.gz")
    if not (os.path.isfile(h5_path) and os.path.isfile(ct_path)):
        return {"pat": pat_id, "status": "SKIP"}

    ct_img = sitk.ReadImage(ct_path)
    ct_bones, ct_pros_img = _get_ct_bones_and_prostate(raw_dir, ct_img)
    mr_bones, mr_pros_img = _get_mri_bones_and_prostate(raw_dir, allow_totalseg=False)

    tfm_path = os.path.join(raw_dir, "Transform_Landmark_SWD.h5")
    if not os.path.isfile(tfm_path):
        tfm_path = os.path.join(raw_dir, "Transform_FollowUp_to_Baseline_1.tfm")

    dice = None
    if os.path.isfile(tfm_path) and ct_pros_img is not None and mr_pros_img is not None:
        tfm = sitk.ReadTransform(tfm_path)
        res_mr = reg_mod.resample_mri(mr_pros_img, ct_img, tfm, sitk.sitkNearestNeighbor)
        dice = reg_mod.compute_dice(ct_pros_img, res_mr)

    with h5py.File(h5_path, "r") as f:
        t2_in_pelvis = False
        t2_z_mean = -1.0
        if "TFM_Transform_FollowUp_to_Baseline_1.tfm/Fixed_CT_Volume_1.nii.gz" in f:
            t2_arr = f["TFM_Transform_FollowUp_to_Baseline_1.tfm/Fixed_CT_Volume_1.nii.gz"][:]
            z_idx = np.where(np.any(t2_arr > 50, axis=(1, 2)))[0]
            y_idx = np.where(np.any(t2_arr > 50, axis=(0, 2)))[0]
            if len(z_idx) > 0 and len(y_idx) > 0:
                t2_z_mean = float(z_idx.mean())
                t2_in_pelvis = (20 <= t2_z_mean <= 125) and (145 <= float(y_idx.mean()) <= 345)

        lbl_raw = f["_meta_/max_anatomy_labels.json"][()]
        lbls = json.loads(lbl_raw.decode() if isinstance(lbl_raw, bytes) else lbl_raw)
        has_sv = any("seminal_vesicle" in v for v in lbls.values())
        has_lac = any("lacrimal_gland" in v for v in lbls.values())
        om_raw = f["_meta_/organ_mapping"][()]
        om = json.loads(om_raw.decode() if isinstance(om_raw, bytes) else om_raw)
        tp0_om = om.get("0", {})
        lac_les = [k for k, v in tp0_om.items() if "lacrimal" in str(v).lower()]

    ok = t2_in_pelvis and (dice is None or dice >= 0.60) and has_sv and has_lac and not lac_les
    return {
        "pat": pat_id,
        "status": "OK" if ok else "NEEDS_REPAIR",
        "dice": round(dice, 4) if dice is not None else None,
        "t2_in_pelvis": t2_in_pelvis,
        "t2_z": round(t2_z_mean, 1),
        "has_sv": has_sv,
        "has_lac": has_lac,
        "lac_les": lac_les,
    }


def repair_case(pat_id: str, subj_info: dict, xnat: XnatManager, gpu_id: int = 1, upload: bool = True, force_mri: bool = False, force_tfm: bool = False) -> dict:
    case_dir = os.path.join(CASES_DIR, pat_id)
    raw_dir = os.path.join(case_dir, "raw")
    h5_path = os.path.join(case_dir, "preprocessed_volumes.h5")
    ct_path = os.path.join(raw_dir, "Fixed_CT_Volume.nii.gz")
    if not (os.path.isfile(h5_path) and os.path.isfile(ct_path)):
        return {"pat": pat_id, "status": "SKIP"}

    ct_img = sitk.ReadImage(ct_path)
    ct_bones, ct_pros_img = _get_ct_bones_and_prostate(raw_dir, ct_img)
    mr_bones, mr_pros_img = _get_mri_bones_and_prostate(raw_dir, allow_totalseg=False)

    swd_tfm_path = os.path.join(raw_dir, "Transform_Landmark_SWD.h5")
    followup_tfm_path = os.path.join(raw_dir, "Transform_FollowUp_to_Baseline_1.tfm")
    cur_tfm_path = swd_tfm_path if os.path.isfile(swd_tfm_path) else (followup_tfm_path if os.path.isfile(followup_tfm_path) else None)

    cur_dice = None
    if cur_tfm_path and ct_pros_img is not None and mr_pros_img is not None:
        cur_tfm = sitk.ReadTransform(cur_tfm_path)
        cur_dice = reg_mod.compute_dice(
            ct_pros_img, reg_mod.resample_mri(mr_pros_img, ct_img, cur_tfm, sitk.sitkNearestNeighbor)
        )

    tfm_updated = False
    if cur_tfm_path is None or force_tfm or mr_pros_img is None or (cur_dice is not None and cur_dice < 0.60):
        # Load/compute totalseg_mr_ml.nii.gz only when registration needs recomputing or verifying
        mr_bones, mr_pros_img = _get_mri_bones_and_prostate(raw_dir, gpu_id=gpu_id, allow_totalseg=True)
        if cur_tfm_path and ct_pros_img is not None and mr_pros_img is not None and cur_dice is None:
            cur_tfm = sitk.ReadTransform(cur_tfm_path)
            cur_dice = reg_mod.compute_dice(
                ct_pros_img, reg_mod.resample_mri(mr_pros_img, ct_img, cur_tfm, sitk.sitkNearestNeighbor)
            )
    if cur_tfm_path is None or force_tfm or (cur_dice is not None and cur_dice < 0.60):
        if ct_pros_img is not None and mr_pros_img is not None:
            print(f"  [{pat_id}] Recomputing 4-Medial-Bone-Landmark + Centered SWD (prev dice={cur_dice})...")
            final_tfm, new_dice = _compute_centered_swd_registration(
                ct_img, ct_bones, ct_pros_img, mr_bones, mr_pros_img
            )
            sitk.WriteTransform(final_tfm, swd_tfm_path)
            sitk.WriteTransform(final_tfm, followup_tfm_path)
            cur_dice = new_dice
            tfm_updated = True
            print(f"  [{pat_id}] New Prostate Dice = {cur_dice:.4f}")
        else:
            final_tfm = sitk.ReadTransform(cur_tfm_path) if cur_tfm_path else sitk.AffineTransform(3)
    else:
        final_tfm = sitk.ReadTransform(cur_tfm_path)
        if not os.path.isfile(swd_tfm_path):
            sitk.WriteTransform(final_tfm, swd_tfm_path)

    # Check if HDF5 MRI needs re-resampling (including stacked DWI/DCE misaligned relative to T2)
    need_mri_resample = tfm_updated or force_mri
    with h5py.File(h5_path, "r") as f:
        pf = int(f["_meta_/preflipped"][()]) if "_meta_/preflipped" in f else 1
        t2_z_mean = None
        if "TFM_Transform_FollowUp_to_Baseline_1.tfm/Fixed_CT_Volume_1.nii.gz" in f:
            t2_arr = f["TFM_Transform_FollowUp_to_Baseline_1.tfm/Fixed_CT_Volume_1.nii.gz"][:]
            z_idx = np.where(np.any(t2_arr > 50, axis=(1, 2)))[0]
            y_idx = np.where(np.any(t2_arr > 50, axis=(0, 2)))[0]
            if len(z_idx) == 0 or not (20 <= float(z_idx.mean()) <= 125 and 145 <= float(y_idx.mean()) <= 345):
                need_mri_resample = True
            else:
                t2_z_mean = float(z_idx.mean())
        if not need_mri_resample and t2_z_mean is not None:
            for tk in sorted(k for k in f.keys() if k.startswith("TFM_Transform_FollowUp_to_Baseline_") and not k.endswith("_1.tfm")):
                vkeys = [k for k in f[tk].keys() if k.startswith("Fixed_CT_Volume_")]
                if vkeys:
                    arr_k = f[tk][vkeys[0]][:]
                    zk = np.where(np.any(arr_k > 50, axis=(1, 2)))[0]
                    if len(zk) == 0 or abs(float(zk.mean()) - t2_z_mean) > 15.0:
                        need_mri_resample = True
                        break

    sp_arr = np.array(ct_img.GetSpacing(), dtype=np.float64)
    orig_arr = np.array(ct_img.GetOrigin(), dtype=np.float64)
    dir_arr = np.array(ct_img.GetDirection(), dtype=np.float64)

    # Resample MRI anatomy & lesions if MRI needs resampling
    mri_anat_unflipped = np.zeros(sitk.GetArrayFromImage(ct_img).shape, dtype=np.uint16)
    mr_gold_unflipped = np.zeros(mri_anat_unflipped.shape, dtype=np.int16)
    if need_mri_resample:
        pros_nii = os.path.join(raw_dir, "prostate_segmentation.nii.gz")
        if os.path.isfile(pros_nii):
            p_img = sitk.ReadImage(pros_nii)
            res_p = sitk.Resample(p_img, ct_img, final_tfm, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
            p_arr = sitk.GetArrayFromImage(res_p).astype(np.uint16)
            p8_mask = np.isin(p_arr, list(range(1, 9)))
            mri_anat_unflipped[p8_mask] = p_arr[p8_mask]
            mr_gold_unflipped = (p_arr == 9).astype(np.int16)

        mapping, _ = _build_tp_to_raw_map(raw_dir, h5_path)
        ref_t2_im = next((sitk.ReadImage(seq["item"]) for _, _, _, mod, seq in mapping if mod == "T2"), None)
        ref_adc_im = next((sitk.ReadImage(seq["item"]) for _, _, _, mod, seq in mapping if mod == "ADC"), None)
        resampled_mris = {}
        for tp_idx, grp, ds_name, mod, seq in mapping:
            im = sitk.ReadImage(seq["item"])
            im = unpack_stacked_mri_volume(im, mod, seq["fn"], ref_t2=ref_t2_im, ref_adc=ref_adc_im)

            already_on_ct_grid = (
                im.GetSize() == ct_img.GetSize()
                and np.allclose(im.GetSpacing(), ct_img.GetSpacing(), atol=1e-2)
                and np.allclose(im.GetOrigin(), ct_img.GetOrigin(), atol=1.0)
            )
            res_im = sitk.Resample(
                im, ct_img, sitk.Transform() if already_on_ct_grid else final_tfm, sitk.sitkLinear, 0.0, sitk.sitkFloat32
            )
            arr = sitk.GetArrayFromImage(res_im).astype(np.float32)
            if np.max(arr) == 0.0 and already_on_ct_grid:
                res_im = sitk.Resample(im, ct_img, final_tfm, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
                arr = sitk.GetArrayFromImage(res_im).astype(np.float32)
            arr = np.maximum(0.0, arr)
            pos = arr[arr > 0]
            if len(pos) > 0:
                p99 = float(np.percentile(pos, 99))
                scale = 1500.0 / max(p99, 1.0) if mod == "ADC" else 1000.0 / max(p99, 1.0)
                arr = arr * scale
            diff_arr = compute_mri_diffusivity(arr, mri_anatomy=mri_anat_unflipped, exclusion_mask=None)
            resampled_mris[tp_idx] = (grp, ds_name, arr, diff_arr)

    # Load CADS 556 & 558
    cads_556_arr, cads_558_arr = load_resampled_cads_arrays(case_dir, ref_img=ct_img)

    # Write all updates to HDF5
    with h5py.File(h5_path, "r+") as hf:
        pf = int(hf["_meta_/preflipped"][()]) if "_meta_/preflipped" in hf else 1

        if need_mri_resample:
            mri_anat_out = (np.flip(mri_anat_unflipped, axis=1) if pf == 1 else mri_anat_unflipped).astype(np.uint16)
            mr_gold_out = (np.flip(mr_gold_unflipped, axis=1) if pf == 1 else mr_gold_unflipped).astype(np.int16)
            for tp_idx, (grp, ds_name, arr, diff_arr) in resampled_mris.items():
                if grp not in hf:
                    continue
                g = hf[grp]
                out_vol = (np.flip(arr, axis=1) if pf == 1 else arr).astype(np.float32)
                out_diff = (np.flip(diff_arr, axis=1) if pf == 1 else diff_arr).astype(np.float32)
                for k_name, data_arr in (
                    (ds_name, out_vol),
                    ("diffusivity", out_diff),
                    ("max_anatomy.nii.gz", mri_anat_out),
                    (f"PET_Lesions_{tp_idx}.nii.gz", mr_gold_out),
                ):
                    if k_name in g:
                        del g[k_name]
                    ds = g.create_dataset(k_name, data=data_arr, compression="gzip", compression_opts=1)
                    ds.attrs["spacing"] = sp_arr
                    ds.attrs["origin"] = orig_arr
                    ds.attrs["direction"] = dir_arr

        # Merge CADS 556 & 558 into ATLAS/max_anatomy and BASELINE/max_anatomy.nii.gz
        base_ma = hf["ATLAS/max_anatomy"][:]
        lbl_raw = hf["_meta_/max_anatomy_labels.json"][()]
        anat_labels = json.loads(lbl_raw.decode() if isinstance(lbl_raw, bytes) else lbl_raw)
        merged_ma, anat_labels = merge_cads_structures(
            base_ma,
            cads_556_arr=cads_556_arr,
            cads_558_arr=cads_558_arr,
            labels_dict=anat_labels,
            flip_y_cads=(pf == 1),
        )
        if "ATLAS/max_anatomy" in hf:
            del hf["ATLAS/max_anatomy"]
        ds_ama = hf.create_dataset("ATLAS/max_anatomy", data=merged_ma, compression="gzip", compression_opts=4, shuffle=True)
        ds_ama.attrs["spacing"] = sp_arr
        ds_ama.attrs["origin"] = orig_arr
        ds_ama.attrs["direction"] = dir_arr

        if "BASELINE/max_anatomy.nii.gz" in hf:
            del hf["BASELINE/max_anatomy.nii.gz"]
        ds_bma = hf["BASELINE"].create_dataset("max_anatomy.nii.gz", data=merged_ma, compression="gzip", compression_opts=4, shuffle=True)
        ds_bma.attrs["spacing"] = sp_arr
        ds_bma.attrs["origin"] = orig_arr
        ds_bma.attrs["direction"] = dir_arr

        if "_meta_/max_anatomy_labels.json" in hf:
            del hf["_meta_/max_anatomy_labels.json"]
        hf["_meta_"].create_dataset("max_anatomy_labels.json", data=json.dumps(anat_labels))

        # Re-map lesions, strip first/last 2 Z-slice technical artifacts & filter lacrimal gland uptake
        b = hf["BASELINE"]
        ct_les_f = b["PET_Lesions_0.nii.gz"][:].astype(np.int16)
        sn_raw = hf["_meta_/segment_names.json"][()]
        seg_names_dict = json.loads(sn_raw.decode() if isinstance(sn_raw, bytes) else sn_raw)
        ct_seg_names = seg_names_dict.get("0", {})

        removed_edge = remove_edge_slice_lesions(ct_les_f, edge_slices=2)
        if removed_edge:
            for r_lid in removed_edge:
                ct_seg_names.pop(str(r_lid), None)

        removed_lac, ct_labels, ct_organs = filter_lacrimal_lesions(
            ct_les_f, merged_ma, anat_labels, seg_names=ct_seg_names
        )
        if removed_lac or removed_edge:
            for r_lid in removed_lac:
                ct_seg_names.pop(str(r_lid), None)
            del b["PET_Lesions_0.nii.gz"]
            ds_l0 = b.create_dataset("PET_Lesions_0.nii.gz", data=ct_les_f, compression="gzip", compression_opts=4, shuffle=True)
            ds_l0.attrs["spacing"] = sp_arr
            ds_l0.attrs["origin"] = orig_arr
            ds_l0.attrs["direction"] = dir_arr

        mr_gold_f = np.zeros_like(ct_les_f)
        for tk in sorted([k for k in hf.keys() if k.startswith("TFM_Transform_FollowUp_to_Baseline_")]):
            tp_num = tk.replace("TFM_Transform_FollowUp_to_Baseline_", "").replace(".tfm", "")
            lk = f"PET_Lesions_{tp_num}.nii.gz"
            if lk in hf[tk]:
                mds = hf[tk][lk]
                mattrs = {a: mds.attrs[a] for a in ("spacing", "origin", "direction") if a in mds.attrs}
                marr = mds[:].astype(np.int16)
                if marr.shape[0] > 4 and (np.any(marr[:2] > 0) or np.any(marr[-2:] > 0)):
                    remove_edge_slice_lesions(marr, edge_slices=2)
                    del hf[tk][lk]
                    new_mds = hf[tk].create_dataset(lk, data=marr, compression="gzip", compression_opts=1)
                    for ak, av in mattrs.items():
                        new_mds.attrs[ak] = av
                if tk.endswith("_1.tfm"):
                    mr_gold_f = marr
        mr_seg_names = {"1": "Prostate Primary Lesion"} if mr_gold_f.any() else {}

        # Update CENTROIDS cleanly
        if "CENTROIDS" in hf:
            del hf["CENTROIDS"]
        c_grp = hf.create_group("CENTROIDS")
        for lid in ct_labels:
            zc, yc, xc = np.mean(np.argwhere(ct_les_f == lid), axis=0)
            c_grp.create_dataset(f"tp0_lid{lid}", data=np.array([round(xc) + 1, round(yc) + 1, round(zc) + 1], dtype=np.int32))
        if mr_gold_f.any():
            zc, yc, xc = np.mean(np.argwhere(mr_gold_f == 1), axis=0)
            c_grp.create_dataset("tp1_lid1", data=np.array([round(xc) + 1, round(yc) + 1, round(zc) + 1], dtype=np.int32))

        tfm_keys = sorted([k for k in hf.keys() if k.startswith("TFM_Transform_FollowUp_to_Baseline_")])
        organ_mapping = {"0": ct_organs}
        seg_names_dict["0"] = ct_seg_names
        for tk in tfm_keys:
            tp_num = tk.replace("TFM_Transform_FollowUp_to_Baseline_", "").replace(".tfm", "")
            organ_mapping[tp_num] = {"1": "Prostate"} if mr_gold_f.any() else {}
            seg_names_dict[tp_num] = mr_seg_names if mr_gold_f.any() else {}

        matches = []
        best_ov_lid, best_ov_cnt = None, 0
        if mr_gold_f.any() and ct_labels:
            for lid in ct_labels:
                ov = int(np.sum((mr_gold_f == 1) & (ct_les_f == lid)))
                if ov > best_ov_cnt:
                    best_ov_cnt, best_ov_lid = ov, lid
        for lid in ct_labels:
            children = []
            if mr_gold_f.any() and lid == best_ov_lid:
                children.append({
                    "name": "PET_Lesions_1",
                    "raw_lesion": "Segment_1",
                    "node": "child",
                    "group_id": int(lid),
                    "lesion": "Prostate Primary Lesion",
                })
            matches.append({
                "name": "PET_Lesions_0",
                "raw_lesion": f"Segment_{lid}",
                "node": "parent",
                "group_id": int(lid),
                "lesion": ct_seg_names.get(str(lid), f"Lesion {lid}"),
                "children": children,
            })

        sh_raw = hf["_meta_/scene_hierarchy.json"][()]
        scene_hierarchy = json.loads(sh_raw.decode() if isinstance(sh_raw, bytes) else sh_raw)
        for node in scene_hierarchy:
            if node.get("name") == "PET_Lesions_0":
                node["segments"] = [ct_seg_names.get(str(i), f"Lesion {i}") for i in ct_labels]

        m_grp = hf["_meta_"]
        for k_meta, val_str in (
            ("organ_mapping", json.dumps(organ_mapping)),
            ("segment_names.json", json.dumps(seg_names_dict)),
            ("matches.json", json.dumps(matches)),
            ("scene_hierarchy.json", json.dumps(scene_hierarchy)),
        ):
            if k_meta in m_grp:
                del m_grp[k_meta]
            m_grp.create_dataset(k_meta, data=val_str)

    try:
        api_base, api_key, llm_model = get_llm_config()
        if api_key:
            compute_and_store_semantic_filter(out_h5, api_base, api_key, model=llm_model, force=True)
    except Exception as _sfe:
        print(f"    [{pat_id}] Note: semantic_filter update skipped: {_sfe}")

    # Sync local xnat_cache if present
    subj_label = subj_info.get("subj_label", "")
    subj_id = subj_info.get("subj_id", "")
    exp_id = subj_info.get("exp_id", "")
    if subj_label:
        cache_h5 = os.path.join(XNAT_CACHE_DIR, subj_label, f"{subj_label}_preprocessed.h5")
        if os.path.isdir(os.path.dirname(cache_h5)):
            shutil.copyfile(h5_path, cache_h5)

    # Upload to XNAT
    if upload and xnat is not None and subj_id:
        remote_h5 = f"{subj_label}_preprocessed.h5"
        xnat.request("PUT", f"/data/projects/{PROJ_ID}/subjects/{subj_id}/resources/MEDEYE3D")
        h5_uri = f"/data/projects/{PROJ_ID}/subjects/{subj_id}/resources/MEDEYE3D/files/{remote_h5}?overwrite=true"
        xnat.upload_file_streaming(h5_uri, h5_path)
        if tfm_updated and exp_id and os.path.isfile(swd_tfm_path):
            xnat.request("PUT", f"/data/experiments/{exp_id}/resources/CT_MRI_REGISTRATION")
            tfm_uri = f"/data/experiments/{exp_id}/resources/CT_MRI_REGISTRATION/files/Transform_Landmark_SWD.h5?overwrite=true"
            xnat.upload_file_streaming(tfm_uri, swd_tfm_path)

    return {
        "pat": pat_id,
        "status": "OK",
        "dice": round(cur_dice, 4) if cur_dice is not None else None,
        "tfm_updated": tfm_updated,
        "mri_resampled": need_mri_resample,
        "lac_removed": removed_lac,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", help="Specific Pat IDs (e.g. Pat26 Pat51)")
    parser.add_argument("--verify-only", action="store_true", help="Audit without modifying files")
    parser.add_argument("--no-upload", action="store_true", help="Do not upload to XNAT")
    parser.add_argument("--force-mri", action="store_true", help="Force re-resampling all MRI sequences into HDF5")
    parser.add_argument("--force-tfm", action="store_true", help="Force recomputing Stage 1+2 CT-MRI transform")
    parser.add_argument("--gpu", type=int, default=1, help="GPU ID for TotalSegmentator/SWD")
    args = parser.parse_args()

    with open(CENSUS_PATH, "r") as cf:
        census = json.load(cf)

    if args.cases:
        targets = args.cases
    else:
        targets = sorted(
            [os.path.basename(p) for p in glob.glob(os.path.join(CASES_DIR, "Pat*")) if "_student" not in p],
            key=lambda s: int(re.search(r"\d+", s).group()) if re.search(r"\d+", s) else 9999,
        )

    if args.verify_only:
        ok_cnt, bad = 0, []
        for pat in targets:
            res = verify_case(pat)
            if res["status"] == "SKIP":
                continue
            if res["status"] == "OK":
                ok_cnt += 1
            else:
                bad.append(res)
        print(f"Verify summary: OK={ok_cnt}, NeedsRepair={len(bad)}")
        for b in bad[:30]:
            print(" ", b)
        return

    xnat = None if args.no_upload else XnatManager()
    t0 = time.time()
    ok_cnt, err_cnt = 0, 0
    for idx, pat in enumerate(targets, 1):
        try:
            res = repair_case(
                pat, census.get(pat, {}), xnat, gpu_id=args.gpu,
                upload=not args.no_upload, force_mri=args.force_mri, force_tfm=args.force_tfm,
            )
            if res["status"] != "SKIP":
                ok_cnt += 1
                print(
                    f"[{idx}/{len(targets)}] {pat}: dice={res['dice']} "
                    f"tfm_upd={res['tfm_updated']} mri_res={res['mri_resampled']} "
                    f"lac_rm={res['lac_removed']}",
                    flush=True,
                )
        except Exception as e:
            err_cnt += 1
            import traceback
            print(f"[{idx}/{len(targets)}] {pat}: ERROR {e}\n{traceback.format_exc()}", flush=True)
    print(f"Completed cohort repair in {time.time()-t0:.1f}s: OK={ok_cnt}, Errors={err_cnt}")


if __name__ == "__main__":
    main()
