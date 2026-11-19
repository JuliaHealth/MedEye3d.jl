#!/usr/bin/env python3
"""
Full-cohort & XNAT audit for Prostata_bimodal_PIPELINE.

Verifies every processed case across all pipeline requirements:
  1. Anatomy: zero excluded non-organ labels (torso_fat, oblique_hyoid_plane,
     helper/temp masks), lung-only override of lymph node areas, CADS 531..539.
  2. MRI Registration & Unpacked Series: valid TFM_* groups with CT-grid shapes,
     CT vs MRI prostate Dice >= 0.60 (or high CT-in-MR containment for huge BPH),
     no sagittal T2_TSE_FH masquerading as axial T2 when T2_TSE_RL exists.
  3. Radiological Description: `_meta_/metadata.json` has `EnglishDescription`.
  4. Lesions & Modification State: `BASELINE/PET_Lesions_0.nii.gz`, `organ_mapping`,
     `segment_names.json`, `matches.json`, `CENTROIDS`, zero lacrimal lesions
     (533/534), zero `*_expert` datasets, `_meta_/modification_status.json` with
     `manually_modified == False`.
  5. All 3 Edge/Diffusivity Models: `diffusivity` (CT+PET Both),
     `diffusivity_ct` (v1_high_lr default), `diffusivity_pet` (PET-only).
  6. Local cache & optional live XNAT (`--check-xnat`) size + `status.json` match.
"""
import argparse
import glob
import json
import os
import re
import sys

import h5py
import numpy as np
import SimpleITK as sitk

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CASES_DIR = os.path.join(REPO_ROOT, "data", "cases", "Prostata_bimodal")
CACHE_ROOT = os.path.expanduser("~/.medeye3d/xnat_cache/Prostata_bimodal_PIPELINE")

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "docs"))

from cads_anatomy_wiring import (
    is_excluded_non_organ_mask,
    is_lung_label_name,
    is_lymph_label_name,
    iter_authoritative_lymph_node_masks,
)
from batch_process_cohort_dual_gpu import CENSUS_PATH, PROJ_ID, XnatManager
import register_ct_mri as reg_mod


def _load_json_ds(ds):
    raw = ds[()]
    return json.loads(raw.decode() if isinstance(raw, bytes) else raw)


def _check_registration_dice(case_dir: str) -> dict:
    raw_dir = os.path.join(case_dir, "raw")
    tfm_p = os.path.join(raw_dir, "Transform_Landmark_SWD.h5")
    if not os.path.isfile(tfm_p):
        tfm_p = os.path.join(raw_dir, "Transform_FollowUp_to_Baseline_1.tfm")
    if not os.path.isfile(tfm_p):
        tfm_p = os.path.join(case_dir, "Transform_FollowUp_to_Baseline_1.tfm")
    ct_p = os.path.join(raw_dir, "Fixed_CT_Volume.nii.gz")
    ct_pros_p = os.path.join(raw_dir, "segmentations", "prostate.nii.gz")
    if not (os.path.isfile(tfm_p) and os.path.isfile(ct_p) and os.path.isfile(ct_pros_p)):
        return {"dice": None, "ct_in_mr": None, "status": "missing_inputs"}

    ct_img = sitk.ReadImage(ct_p)
    ct_pros = sitk.ReadImage(ct_pros_p)
    ct_arr = sitk.GetArrayFromImage(ct_pros) > 0
    if not ct_arr.any():
        return {"dice": None, "ct_in_mr": None, "status": "empty_ct_prostate"}

    mr_pros = None
    ts_mr_p = os.path.join(raw_dir, "totalseg_mr_ml.nii.gz")
    if os.path.isfile(ts_mr_p):
        ts_im = sitk.ReadImage(ts_mr_p)
        m = (sitk.GetArrayFromImage(ts_im) == 17).astype(np.uint8)
        if m.any():
            mr_pros = sitk.GetImageFromArray(m)
            mr_pros.CopyInformation(ts_im)
    pros_nii = os.path.join(raw_dir, "prostate_segmentation.nii.gz")
    if os.path.isfile(pros_nii):
        pi = sitk.ReadImage(pros_nii)
        pa = sitk.GetArrayFromImage(pi)
        m = np.isin(pa, [1, 2, 3, 4, 5, 6, 9]).astype(np.uint8)
        if m.sum() < 100:
            m = (pa > 0).astype(np.uint8)
        if m.any():
            p_vol = float(m.sum()) * float(np.prod(pi.GetSpacing()))
            ts_vol = (
                float((sitk.GetArrayViewFromImage(mr_pros) > 0).sum()) * float(np.prod(mr_pros.GetSpacing()))
                if mr_pros is not None
                else 0.0
            )
            if mr_pros is None or (ts_vol > 0 and p_vol >= 0.5 * ts_vol):
                mr_pros = sitk.GetImageFromArray(m)
                mr_pros.CopyInformation(pi)

    if mr_pros is None:
        return {"dice": None, "ct_in_mr": None, "status": "empty_mr_prostate"}

    tfm = sitk.ReadTransform(tfm_p)
    res_mr = reg_mod.resample_mri(mr_pros, ct_img, tfm, sitk.sitkNearestNeighbor)
    dice = float(reg_mod.compute_dice(ct_pros, res_mr))
    mr_arr = sitk.GetArrayFromImage(res_mr) > 0
    ct_in_mr = float(np.sum(ct_arr & mr_arr) / max(1, int(ct_arr.sum())))
    return {"dice": round(dice, 4), "ct_in_mr": round(ct_in_mr, 4), "status": "ok"}


def audit_case(pat_id: str, check_dice: bool = True) -> dict:
    case_dir = os.path.join(CASES_DIR, pat_id)
    h5_path = os.path.join(case_dir, "preprocessed_volumes.h5")
    issues = []
    info = {"patient": pat_id, "h5_size": os.path.getsize(h5_path)}

    with h5py.File(h5_path, "r") as f:
        if "BASELINE" not in f or "ATLAS" not in f or "_meta_" not in f:
            issues.append("missing_core_groups")
            return {"patient": pat_id, "issues": issues}

        bl = f["BASELINE"]
        ct_shape = bl["Fixed_CT_Volume_0.nii.gz"].shape

        # 1. Anatomy checks
        labels = _load_json_ds(f["_meta_/max_anatomy_labels.json"])
        bad_excl = [
            f"{k}:{v}" for k, v in labels.items()
            if is_excluded_non_organ_mask(str(v).lower())
        ]
        if bad_excl:
            issues.append(f"excluded_anatomy:{bad_excl[:5]}")

        ma = f["ATLAS/max_anatomy"][:]
        ln_ids = {int(k) for k, v in labels.items() if is_lymph_label_name(str(v).lower())}
        if len(ln_ids) == 0:
            issues.append("no_ln_labels")

        # Check lung-only LN override when authoritative LN source (lymph_nodes/ or final_results_gpu.h5) exists
        raw_dir = os.path.join(case_dir, "raw")
        has_ln_src = os.path.isdir(os.path.join(case_dir, "lymph_nodes")) or os.path.isfile(os.path.join(raw_dir, "final_results_gpu.h5"))
        if has_ln_src:
            pf = int(f["_meta_/preflipped"][()]) if "_meta_/preflipped" in f else 1
            lung_ids = {int(k) for k, v in labels.items() if is_lung_label_name(str(v).lower())}
            # Verify authoritative LN voxels are either kept in ln_ids or covered only by lungs / higher-priority CADS 556/558
            expected_ln = np.zeros(ct_shape, dtype=bool)
            for _, m_ln, _ in iter_authoritative_lymph_node_masks(ct_shape, raw_dir):
                expected_ln |= m_ln
            if pf == 1:
                expected_ln = np.flip(expected_ln, axis=1)
            allowed_override_ids = ln_ids | lung_ids | set(range(531, 540))
            actual_vals = ma[expected_ln]
            illegal = set(np.unique(actual_vals).tolist()) - allowed_override_ids
            if illegal:
                bad_names = {int(x): labels.get(str(int(x)), "BG") for x in illegal}
                issues.append(f"illegal_ln_override:{bad_names}")

        # 2. Edges (Both, CT v1, PET)
        for edge_key in ("diffusivity", "diffusivity_ct", "diffusivity_pet"):
            if edge_key not in bl:
                issues.append(f"missing_{edge_key}")
            else:
                ds = bl[edge_key]
                if ds.shape != ct_shape:
                    issues.append(f"shape_mismatch_{edge_key}")

        # 3. MRI groups
        sh = _load_json_ds(f["_meta_/scene_hierarchy.json"])
        tfm_groups = sorted([k for k in f.keys() if k.startswith("TFM_")])
        if not tfm_groups:
            issues.append("no_mri_tfm_groups")
        for tg in tfm_groups:
            g = f[tg]
            vols = [k for k in g.keys() if k.startswith("Fixed_CT_Volume_")]
            if not vols:
                issues.append(f"empty_{tg}")
            elif g[vols[0]].shape != ct_shape:
                issues.append(f"mri_shape_mismatch_{tg}")
            if "diffusivity" not in g:
                issues.append(f"missing_mri_diff_{tg}")
        for node in sh:
            if node.get("type") == "vtkMRMLLinearTransformNode" and node.get("modality") == "T2":
                pass
        # Flag if raw dir still had T2_TSE_FH without T2_TSE_RL downloaded
        raw_files = set(os.listdir(os.path.join(case_dir, "raw")))
        if any("t2_tse_fh" in fn.lower() for fn in raw_files) and not any("t2_tse_rl_registered" in fn.lower() for fn in raw_files):
            issues.append("sagittal_t2_tse_fh_used")

        # 4. Radiological description (stored under *_PET / *_MRI dict entries or top-level)
        md = _load_json_ds(f["_meta_/metadata.json"])
        has_desc = bool(str(md.get("EnglishDescription", "")).strip()) or any(
            isinstance(v, dict) and bool(str(v.get("EnglishDescription", "")).strip())
            for v in md.values()
        )
        if not has_desc:
            issues.append("missing_english_description")

        # 5. Lesions, edge-slice Z trim, semantic filter, unmodified flag & zero *_expert
        les0 = bl["PET_Lesions_0.nii.gz"][:]
        if int((les0 > 0).sum()) == 0:
            issues.append("zero_pet_lesions")
        if les0.shape[0] > 4 and (int((les0[:2] > 0).sum()) > 0 or int((les0[-2:] > 0).sum()) > 0):
            issues.append("edge_slice_lesions_present_tp0")
        for tg in tfm_groups:
            tp_num = tg.replace("TFM_Transform_FollowUp_to_Baseline_", "").replace(".tfm", "")
            lk = f"PET_Lesions_{tp_num}.nii.gz"
            if lk in f[tg]:
                mles = f[tg][lk][:]
                if mles.shape[0] > 4 and (int((mles[:2] > 0).sum()) > 0 or int((mles[-2:] > 0).sum()) > 0):
                    issues.append(f"edge_slice_lesions_present_{tg}")
        om = _load_json_ds(f["_meta_/organ_mapping"])
        sn = _load_json_ds(f["_meta_/segment_names.json"])
        if "0" not in om or not om["0"]:
            issues.append("empty_organ_mapping_0")
        for lid_s, org_name in om.get("0", {}).items():
            if "lacrimal" in str(org_name).lower():
                issues.append(f"lacrimal_lesion_{lid_s}")
        if not sn.get("0"):
            issues.append("empty_segment_names_0")
        if "_meta_/semantic_filter.json" not in f:
            issues.append("missing_semantic_filter_json")
        else:
            sf = _load_json_ds(f["_meta_/semantic_filter.json"])
            if "mentioned_max_anatomy" not in sf or "allowed_lesion_ids_by_tp" not in sf:
                issues.append("malformed_semantic_filter_json")
            elif has_desc and len(sf.get("mentioned_max_anatomy", [])) == 0:
                issues.append("empty_semantic_filter_mentions")
        if "_meta_/matches.json" not in f:
            issues.append("missing_matches_json")
        if "CENTROIDS" not in f or len(f["CENTROIDS"].keys()) == 0:
            issues.append("missing_centroids")

        if "_meta_/modification_status.json" not in f:
            issues.append("missing_modification_status")
        else:
            ms = _load_json_ds(f["_meta_/modification_status.json"])
            if ms.get("manually_modified") is not False or ms.get("modified_by", "") != "":
                issues.append(f"not_clean_unmodified:{ms}")

        expert_hits = []
        f.visit(lambda name: expert_hits.append(name) if name.endswith("_expert") or "_expert/" in name else None)
        if expert_hits:
            issues.append(f"has_expert_datasets:{expert_hits[:4]}")

    if check_dice:
        reg_res = _check_registration_dice(case_dir)
        info["reg"] = reg_res
        d = reg_res.get("dice")
        c = reg_res.get("ct_in_mr")
        if d is not None and d < 0.60 and (c is None or c < 0.80):
            issues.append(f"low_mri_dice:{d}_ct_in_mr:{c}")

    info["issues"] = issues
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dice", action="store_true", help="Skip 3D CT-MRI prostate Dice calculation")
    ap.add_argument("--check-xnat", action="store_true", help="Verify local cache and live XNAT sizes + status.json")
    ap.add_argument("--json-out", help="Write audit JSON report to file")
    args = ap.parse_args()

    all_cases = sorted(
        [
            os.path.basename(p)
            for p in glob.glob(os.path.join(CASES_DIR, "Pat*"))
            if re.fullmatch(r"Pat\d+", os.path.basename(p))
            and os.path.isfile(os.path.join(p, "preprocessed_volumes.h5"))
        ],
        key=lambda s: int(re.search(r"\d+", s).group()),
    )

    with open(CENSUS_PATH, "r") as cf:
        census = json.load(cf)

    results = []
    failing = []
    cache_mismatches = []
    xnat_mismatches = []

    xnat = XnatManager() if args.check_xnat else None

    for pat in all_cases:
        res = audit_case(pat, check_dice=not args.no_dice)
        slbl = census.get(pat, {}).get("subj_label", "")
        if slbl:
            c_h5 = os.path.join(CACHE_ROOT, slbl, f"{slbl}_preprocessed.h5")
            if not os.path.isfile(c_h5) or os.path.getsize(c_h5) != res["h5_size"]:
                cache_mismatches.append(pat)
                res["issues"].append("xnat_cache_mismatch")

        if xnat and pat in census:
            sid = census[pat]["subj_id"]
            rf = (xnat.get_json(f"/data/projects/{PROJ_ID}/subjects/{sid}/resources/MEDEYE3D/files?format=json") or {}).get("ResultSet", {}).get("Result", [])
            remote_name = f"{slbl}_preprocessed.h5"
            match_f = [x for x in rf if x.get("Name") == remote_name]
            if not match_f or int(match_f[0].get("Size", -1)) != res["h5_size"]:
                xnat_mismatches.append((pat, match_f[0].get("Size") if match_f else None, res["h5_size"]))
                res["issues"].append("xnat_h5_size_mismatch")
            st_r = xnat.request("GET", f"/data/projects/{PROJ_ID}/subjects/{sid}/resources/ANNOTATIONS/files/status.json", timeout=15)
            if not st_r or st_r.status_code != 200:
                res["issues"].append("xnat_missing_status_json")
            else:
                try:
                    st_j = st_r.json()
                    if st_j.get("manually_modified") is not False or st_j.get("modified_by", "") != "":
                        res["issues"].append(f"xnat_status_modified:{st_j}")
                except Exception:
                    res["issues"].append("xnat_bad_status_json")

        results.append(res)
        if res["issues"]:
            failing.append(res)

    summary = {
        "total_local_cases": len(all_cases),
        "clean_cases": len(all_cases) - len(failing),
        "failing_cases": failing,
        "cache_mismatches": cache_mismatches,
        "xnat_mismatches": xnat_mismatches,
    }
    print(f"Audited {len(all_cases)} local cases: clean={summary['clean_cases']}/{len(all_cases)}, failing={len(failing)}")
    for r in failing:
        print(f"  FAIL {r['patient']}: {r['issues']} reg={r.get('reg')}")

    if args.json_out:
        with open(args.json_out, "w") as jf:
            json.dump(summary, jf, indent=2)

    sys.exit(0 if not failing else 1)


if __name__ == "__main__":
    main()
