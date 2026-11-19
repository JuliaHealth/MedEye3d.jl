#!/usr/bin/env python3
"""
Run native MRI prostate zone + tumor segmentation (Dataset501_ProstateZonesLesions: T2W + ADC + HBV/DWI)
on all Prostata_bimodal cases missing `raw/prostate_segmentation.nii.gz` (or re-running cases that previously
lacked quantitative ADC / unpacked DWI), then update `TFM_*/max_anatomy.nii.gz` (8 prostate zones) and
`TFM_*/PET_Lesions_*` (Label 9 -> 1 Prostate Primary Lesion) + metadata + xnat_cache in `preprocessed_volumes.h5`.

Also cleans up any cases whose `TFM_*/PET_Lesions_*` had been polluted by `NNUNET_OR_Processed_Lesions_MR.seg.nrrd`
instead of native `prostate_segmentation.nii.gz` Label 9.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
import SimpleITK as sitk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CASES_DIR = os.path.join(REPO_ROOT, "data", "cases", "Prostata_bimodal")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
CACHE_ROOT = os.path.expanduser("~/.medeye3d/xnat_cache/Prostata_bimodal_PIPELINE")

sys.path.insert(0, SCRIPT_DIR)
from apply_edge_slices_and_semantic_filter import organ_matches_mentioned
from batch_process_cohort_dual_gpu import (
    CENSUS_PATH,
    PROJ_ID,
    RAW_PROJ_ID,
    XnatManager,
    classify_mri_modality,
    select_ranked_mri_sequences,
    unpack_stacked_mri_volume,
)
from cads_anatomy_wiring import remove_edge_slice_lesions
from repair_cohort_adc_and_dwi import _is_quant_adc_fn, upload_h5_and_clean_status

PROSTATE_ZONE_LABELS = {
    "1": "PZ",
    "2": "TZ",
    "3": "CZ",
    "4": "AFS",
    "5": "UR",
    "6": "PG",
    "7": "SV_L",
    "8": "SV_R",
}


def _find_t2_adc_dwi(raw_dir: str) -> tuple:
    mri_map = {}
    for fn in os.listdir(raw_dir):
        if fn.endswith(".nii.gz") and not fn.startswith(("Fixed_CT", "SUV", "CT", "totalseg", "prostate_segmentation")):
            if classify_mri_modality(fn):
                mri_map[fn] = os.path.join(raw_dir, fn)
    ranked = select_ranked_mri_sequences(mri_map)
    t2 = next((s for s in ranked if s["mod"] == "T2" and s["plane"] == "ax"), None) or next((s for s in ranked if s["mod"] == "T2"), None)
    adc = next((s for s in ranked if s["mod"] == "ADC" and _is_quant_adc_fn(s["fn"])), None)
    dwi = next((s for s in ranked if s["mod"] == "DWI" and s["plane"] == "ax"), None) or next((s for s in ranked if s["mod"] == "DWI"), None)
    return t2, adc, dwi


def _prepare_nnunet_inputs(pat_id: str, in_dir: str) -> str:
    raw_dir = os.path.join(CASES_DIR, pat_id, "raw")
    t2_seq, adc_seq, dwi_seq = _find_t2_adc_dwi(raw_dir)
    if not (t2_seq and adc_seq and dwi_seq):
        return f"missing_seq(t2={bool(t2_seq)},adc={bool(adc_seq)},dwi={bool(dwi_seq)})"

    t2_im = sitk.ReadImage(t2_seq["item"])
    adc_im = unpack_stacked_mri_volume(sitk.ReadImage(adc_seq["item"]), "ADC", adc_seq["fn"], ref_t2=t2_im)
    dwi_im = unpack_stacked_mri_volume(sitk.ReadImage(dwi_seq["item"]), "DWI", dwi_seq["fn"], ref_t2=t2_im, ref_adc=adc_im)

    adc_t2 = sitk.Resample(adc_im, t2_im, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    dwi_t2 = sitk.Resample(dwi_im, t2_im, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)

    sitk.WriteImage(sitk.Cast(t2_im, sitk.sitkFloat32), os.path.join(in_dir, f"{pat_id}_0000.nii.gz"))
    sitk.WriteImage(adc_t2, os.path.join(in_dir, f"{pat_id}_0001.nii.gz"))
    sitk.WriteImage(dwi_t2, os.path.join(in_dir, f"{pat_id}_0002.nii.gz"))
    return "OK"


def _apply_prostate_seg_to_h5(pat_id: str, census: dict, xnat: XnatManager = None,
                              sync_cache: bool = False, upload_h5: bool = False,
                              upload_raw: bool = False) -> dict:
    case_dir = os.path.join(CASES_DIR, pat_id)
    raw_dir = os.path.join(case_dir, "raw")
    h5_path = os.path.join(case_dir, "preprocessed_volumes.h5")
    ps_path = os.path.join(raw_dir, "prostate_segmentation.nii.gz")
    ct_path = os.path.join(raw_dir, "Fixed_CT_Volume.nii.gz")
    if not (os.path.isfile(h5_path) and os.path.isfile(ct_path)):
        return {"patient": pat_id, "status": "SKIP"}

    ct_img = sitk.ReadImage(ct_path)
    tfm_path = os.path.join(raw_dir, "Transform_Landmark_SWD.h5")
    if not os.path.isfile(tfm_path):
        tfm_path = os.path.join(raw_dir, "Transform_FollowUp_to_Baseline_1.tfm")
    tfm_obj = sitk.ReadTransform(tfm_path) if os.path.isfile(tfm_path) else sitk.AffineTransform(3)

    if os.path.isfile(ps_path):
        ps_im = sitk.ReadImage(ps_path)
        on_ct = (
            ps_im.GetSize() == ct_img.GetSize()
            and np.allclose(ps_im.GetSpacing(), ct_img.GetSpacing(), atol=1e-2)
            and np.allclose(ps_im.GetOrigin(), ct_img.GetOrigin(), atol=1.0)
        )
        res_ps = sitk.Resample(ps_im, ct_img, sitk.Transform() if on_ct else tfm_obj, sitk.sitkNearestNeighbor, 0, sitk.sitkInt16)
        ps_arr = sitk.GetArrayFromImage(res_ps).astype(np.int16)
        if not ps_arr.any() and on_ct:
            res_ps = sitk.Resample(ps_im, ct_img, tfm_obj, sitk.sitkNearestNeighbor, 0, sitk.sitkInt16)
            ps_arr = sitk.GetArrayFromImage(res_ps).astype(np.int16)
    else:
        ps_arr = np.zeros(sitk.GetArrayFromImage(ct_img).shape, dtype=np.int16)

    zones_arr = np.where((ps_arr >= 1) & (ps_arr <= 8), ps_arr, 0).astype(np.uint16)
    les_arr = (ps_arr == 9).astype(np.int16)

    with h5py.File(h5_path, "r+") as hf:
        ms = json.loads(hf["_meta_/modification_status.json"][()]) if "_meta_/modification_status.json" in hf else {}
        if ms.get("manually_modified") is True:
            return {"patient": pat_id, "status": "SKIP_MODIFIED"}
        pf = int(hf["_meta_/preflipped"][()]) if "_meta_/preflipped" in hf else 1
        zones_f = (np.flip(zones_arr, axis=1) if pf == 1 else zones_arr).astype(np.uint16).copy()
        les_f = (np.flip(les_arr, axis=1) if pf == 1 else les_arr).astype(np.int16).copy()
        if les_f.any():
            remove_edge_slice_lesions(les_f, edge_slices=2)
        has_mr_les = bool(les_f.any())

        ct_les_f = hf["BASELINE/PET_Lesions_0.nii.gz"][:].astype(np.int16)
        ct_labels = sorted([int(x) for x in np.unique(ct_les_f) if x > 0])

        tfm_keys = sorted(
            [k for k in hf.keys() if k.startswith("TFM_Transform_FollowUp_to_Baseline_")],
            key=lambda s: int(re.search(r"_(\d+)\.tfm$", s).group(1)),
        )
        tp_nums = []
        for tk in tfm_keys:
            tp_num = tk.replace("TFM_Transform_FollowUp_to_Baseline_", "").replace(".tfm", "")
            tp_nums.append(tp_num)
            for ds_k, arr_d, comp in (
                (f"PET_Lesions_{tp_num}.nii.gz", les_f, 1),
                ("max_anatomy.nii.gz", zones_f, 1),
            ):
                sp_attrs = {}
                if ds_k in hf[tk]:
                    for ak, av in hf[tk][ds_k].attrs.items():
                        sp_attrs[ak] = av
                    del hf[tk][ds_k]
                ds = hf[tk].create_dataset(ds_k, data=arr_d, compression="gzip", compression_opts=comp)
                for ak, av in sp_attrs.items():
                    ds.attrs[ak] = av

        # Sync CENTROIDS (tp1_lid1 only if Label 9 lesion exists)
        c_grp = hf["CENTROIDS"] if "CENTROIDS" in hf else hf.create_group("CENTROIDS")
        for k in list(c_grp.keys()):
            if k.startswith("tp1_lid"):
                del c_grp[k]
        if has_mr_les:
            zc, yc, xc = np.mean(np.argwhere(les_f == 1), axis=0)
            c_grp.create_dataset("tp1_lid1", data=np.array([round(xc) + 1, round(yc) + 1, round(zc) + 1], dtype=np.int32))

        m_grp = hf["_meta_"]
        if "prostate_anatomy_labels.json" in m_grp:
            del m_grp["prostate_anatomy_labels.json"]
        m_grp.create_dataset("prostate_anatomy_labels.json", data=json.dumps(PROSTATE_ZONE_LABELS))

        om = json.loads(m_grp["organ_mapping"][()]) if "organ_mapping" in m_grp else {"0": {}}
        sn = json.loads(m_grp["segment_names.json"][()]) if "segment_names.json" in m_grp else {"0": {}}
        for tp_k in list(om.keys()):
            if tp_k != "0":
                del om[tp_k]
        for tp_k in list(sn.keys()):
            if tp_k != "0":
                del sn[tp_k]
        for tp_num in tp_nums:
            om[tp_num] = {"1": "Prostate"} if has_mr_les else {}
            sn[tp_num] = {"1": "Prostate Primary Lesion"} if has_mr_les else {}

        best_ov_lid, best_ov_cnt = None, 0
        if has_mr_les and ct_labels:
            for lid in ct_labels:
                ov = int(np.sum((les_f == 1) & (ct_les_f == lid)))
                if ov > best_ov_cnt:
                    best_ov_cnt, best_ov_lid = ov, lid
        ct_sn0 = sn.get("0", {})
        matches = []
        for lid in ct_labels:
            children = []
            if has_mr_les and lid == best_ov_lid:
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
                "lesion": ct_sn0.get(str(lid), f"Lesion {lid}"),
                "children": children,
            })

        if "scene_hierarchy.json" in m_grp:
            sh = json.loads(m_grp["scene_hierarchy.json"][()])
            segs = ["Prostate Primary Lesion"] if has_mr_les else []
            for node in sh:
                if node.get("type") == "vtkMRMLLinearTransformNode":
                    for ch in node.get("children", []):
                        if str(ch.get("name", "")).startswith("PET_Lesions_"):
                            ch["segments"] = list(segs)
            del m_grp["scene_hierarchy.json"]
            m_grp.create_dataset("scene_hierarchy.json", data=json.dumps(sh))

        if "semantic_filter.json" in m_grp:
            sf = json.loads(m_grp["semantic_filter.json"][()])
            mentioned = sf.get("mentioned_max_anatomy", [])
            allowed_by_tp = {"0": sf.get("allowed_lesion_ids_by_tp", {}).get("0", [])}
            for tp_num in tp_nums:
                allowed_by_tp[tp_num] = [1] if (has_mr_les and organ_matches_mentioned("Prostate", mentioned)) else []
            sf["allowed_lesion_ids_by_tp"] = allowed_by_tp
            del m_grp["semantic_filter.json"]
            m_grp.create_dataset("semantic_filter.json", data=json.dumps(sf))

        for mk, mv in (("organ_mapping", json.dumps(om)), ("segment_names.json", json.dumps(sn)), ("matches.json", json.dumps(matches))):
            if mk in m_grp:
                del m_grp[mk]
            m_grp.create_dataset(mk, data=mv)

    if sync_cache and pat_id in census:
        slbl = census[pat_id].get("subj_label", "")
        if slbl:
            c_dir = os.path.join(CACHE_ROOT, slbl)
            os.makedirs(c_dir, exist_ok=True)
            dst_h5 = os.path.join(c_dir, f"{slbl}_preprocessed.h5")
            shutil.copyfile(h5_path, dst_h5)
            meta_p = dst_h5 + ".xnat_meta.json"
            if os.path.isfile(meta_p):
                os.remove(meta_p)

    if upload_raw and os.path.isfile(ps_path) and pat_id in census and xnat is not None:
        exp_id = census[pat_id].get("exp_id", "")
        if exp_id:
            xnat.request("PUT", f"/data/projects/{RAW_PROJ_ID}/experiments/{exp_id}/resources/PROSTATE_SEGMENTATION")
            ps_uri = f"/data/projects/{RAW_PROJ_ID}/experiments/{exp_id}/resources/PROSTATE_SEGMENTATION/files/prostate_segmentation.nii.gz?overwrite=true"
            xnat.upload_file_streaming(ps_uri, ps_path)

    if upload_h5 and pat_id in census and xnat is not None:
        if not upload_h5_and_clean_status(pat_id, h5_path, census, xnat):
            return {"patient": pat_id, "status": "UPLOAD_FAIL"}

    print(
        f"  ✓ {pat_id}: Dataset501 zones={int((zones_f > 0).sum()):,} vox, "
        f"Label9 lesion={int(les_f.sum()):,} vox",
        flush=True,
    )
    return {"patient": pat_id, "status": "OK", "les_vox": int(les_f.sum()), "zone_vox": int((zones_f > 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", help="Comma-separated Pat IDs to run Dataset501 on (or apply existing prostate_segmentation.nii.gz)")
    ap.add_argument("--apply-only", action="store_true", help="Skip nnUNetv2_predict and apply existing raw/prostate_segmentation.nii.gz (or clean zeros)")
    ap.add_argument("--force-predict", action="store_true", help="Run nnUNetv2_predict even if raw/prostate_segmentation.nii.gz already exists")
    ap.add_argument("--chunk-size", type=int, default=12)
    ap.add_argument("--sync-cache", action="store_true")
    ap.add_argument("--upload-h5", action="store_true")
    ap.add_argument("--upload-raw", action="store_true")
    args = ap.parse_args()

    with open(CENSUS_PATH, "r") as cf:
        census = json.load(cf)

    if args.cases:
        targets = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        targets = sorted(
            [os.path.basename(p) for p in glob.glob(os.path.join(CASES_DIR, "Pat*"))
             if re.fullmatch(r"Pat\d+", os.path.basename(p))
             and os.path.isfile(os.path.join(p, "preprocessed_volumes.h5"))
             and not os.path.isfile(os.path.join(p, "raw", "prostate_segmentation.nii.gz"))],
            key=lambda s: int(s[3:]),
        )

    print(f"Targets ({len(targets)}): {targets}", flush=True)
    xnat = XnatManager() if (args.upload_h5 or args.upload_raw) else None

    if not args.apply_only:
        need_pred = [
            p for p in targets
            if args.force_predict or not os.path.isfile(os.path.join(CASES_DIR, p, "raw", "prostate_segmentation.nii.gz"))
        ]
        for start in range(0, len(need_pred), args.chunk_size):
            batch = need_pred[start:start + args.chunk_size]
            with tempfile.TemporaryDirectory() as td:
                in_d = os.path.join(td, "in")
                out_d = os.path.join(td, "out")
                os.makedirs(in_d)
                os.makedirs(out_d)
                prepped = []
                for p in batch:
                    st = _prepare_nnunet_inputs(p, in_d)
                    if st == "OK":
                        prepped.append(p)
                    else:
                        print(f"  [SKIP-PRED] {p}: {st}", flush=True)
                if prepped:
                    env = os.environ.copy()
                    env["nnUNet_results"] = MODELS_DIR
                    env["nnUNet_raw"] = td
                    env["nnUNet_preprocessed"] = td
                    cmd = [
                        "nnUNetv2_predict", "-i", in_d, "-o", out_d,
                        "-d", "501", "-c", "3d_fullres",
                        "-f", "0", "1", "2", "3", "4",
                        "-chk", "checkpoint_final.pth",
                        "--disable_progress_bar",
                    ]
                    print(f"[Dataset501] Predicting batch ({len(prepped)} cases): {prepped}", flush=True)
                    subprocess.run(cmd, env=env, check=True)
                    for p in prepped:
                        src = os.path.join(out_d, f"{p}.nii.gz")
                        dst = os.path.join(CASES_DIR, p, "raw", "prostate_segmentation.nii.gz")
                        if os.path.isfile(src):
                            shutil.copyfile(src, dst)

    ok = 0
    for p in targets:
        res = _apply_prostate_seg_to_h5(
            p, census, xnat=xnat, sync_cache=args.sync_cache,
            upload_h5=args.upload_h5, upload_raw=args.upload_raw,
        )
        if res["status"] == "OK":
            ok += 1
    print(f"Completed Dataset501 integration: {ok}/{len(targets)} cases.", flush=True)


if __name__ == "__main__":
    main()
