#!/usr/bin/env python3
"""
Repair missing ADC, mislabeled truncated `s)_*` ADC, slice-interleaved / stacked DWI,
and omitted axial DCE MRI timepoints in Prostata_bimodal `preprocessed_volumes.h5`.

For any case missing an axial ADC NIfTI in `raw/`:
  - Queries XNAT MRI experiments (`*_MRI_*__<PatX>`) for `dDWI` / `ADC` / `IsoADC` / `ddADC`
  - Downloads DICOMs once, reconstructs the 3D NIfTI via SimpleITK ImageSeriesReader
  - Validates spatial overlap against T2 on the CT grid using `Transform_Landmark_SWD.h5`

Then re-ranks available MRI series (`T2`, `ADC`, `DWI` [de-interleaved/unpacked], `T1`, `DCE`),
writes clean `TFM_Transform_FollowUp_to_Baseline_<1..N>.tfm` groups and `_meta_/scene_hierarchy.json`,
preserving `BASELINE/*`, `ATLAS/*`, CT lesions, organ mapping, semantic filter, and CADS intact.
"""
import argparse
import glob
import io
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile

import h5py
import numpy as np
import SimpleITK as sitk

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CASES_DIR = os.path.join(REPO_ROOT, "data", "cases", "Prostata_bimodal")
CACHE_ROOT = os.path.expanduser("~/.medeye3d/xnat_cache/Prostata_bimodal_PIPELINE")

sys.path.insert(0, SCRIPT_DIR)
from batch_process_cohort_dual_gpu import (
    CENSUS_PATH,
    PROJ_ID,
    RAW_PROJ_ID,
    XnatManager,
    classify_mri_modality,
    compute_mri_diffusivity,
    select_ranked_mri_sequences,
    unpack_stacked_mri_volume,
)


def upload_h5_and_clean_status(pat_id: str, h5_path: str, census: dict, xnat: XnatManager) -> bool:
    """Upload updated preprocessed_volumes.h5 to XNAT MEDEYE3D and sync ANNOTATIONS/status.json."""
    sinfo = census.get(pat_id, {})
    subj_id = sinfo.get("subj_id", "")
    subj_label = sinfo.get("subj_label", "")
    if not (subj_id and subj_label):
        return False

    xnat.request("PUT", f"/data/projects/{PROJ_ID}/subjects/{subj_id}/resources/MEDEYE3D")
    h5_uri = f"/data/projects/{PROJ_ID}/subjects/{subj_id}/resources/MEDEYE3D/files/{subj_label}_preprocessed.h5?overwrite=true"
    if not xnat.upload_file_streaming(h5_uri, h5_path):
        return False

    try:
        with h5py.File(h5_path, "r") as hf:
            ms = json.loads(hf["_meta_/modification_status.json"][()]) if "_meta_/modification_status.json" in hf else {}
        is_mod = bool(ms.get("manually_modified", False))
        mod_by = str(ms.get("modified_by", ""))
        last_m = str(ms.get("last_modified", ""))
        st_payload = {
            "project_id": PROJ_ID,
            "subject_id": subj_label,
            "status": "PROCESSED" if is_mod else "PREPROCESSED",
            "manually_modified": is_mod,
            "modified_by": mod_by if is_mod else "",
            "assigned_user": mod_by if is_mod else "",
            "last_modified": last_m,
            "summary": {"manually_modified": is_mod, "modified_by": mod_by if is_mod else ""},
        }
        xnat.request("PUT", f"/data/projects/{PROJ_ID}/subjects/{subj_id}/resources/ANNOTATIONS")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(st_payload, tf, indent=2)
            tmp_st = tf.name
        try:
            st_uri = f"/data/projects/{PROJ_ID}/subjects/{subj_id}/resources/ANNOTATIONS/files/status.json?overwrite=true"
            xnat.upload_file_streaming(st_uri, tmp_st)
        finally:
            if os.path.isfile(tmp_st):
                os.remove(tmp_st)
    except Exception:
        pass
    return True


def _index_xnat_mri_experiments(xnat: XnatManager) -> dict:
    """Map PatID -> [(exp_id, exp_label), ...] across Prostata_bimodal_PIPELINE and Prostata_bimodal."""
    mri_exps = {}
    for proj in (PROJ_ID, RAW_PROJ_ID):
        data = xnat.get_json(f"/data/projects/{proj}/experiments?format=json")
        if not data:
            continue
        for e in data.get("ResultSet", {}).get("Result", []):
            lbl = e.get("label", "")
            m = re.search(r"__MRI_\d+__(Pat\d+)$", lbl)
            if m:
                mri_exps.setdefault(m.group(1), []).append((e["ID"], lbl))
    return mri_exps


def _is_quant_adc_fn(fn: str) -> bool:
    fl = fn.lower()
    res = classify_mri_modality(fn)
    return res == ("ADC", "ax") and "eadc" not in fl and "edwi" not in fl


def _pick_best_adc_scan(scans: list):
    """Pick the best quantitative ADC DICOM scan from XNAT MRI scan list."""
    scored = []
    for sc in scans:
        sid = str(sc.get("ID", ""))
        desc = (sc.get("series_description") or sc.get("type") or "").strip()
        dl = desc.lower()
        if not dl:
            continue
        # Exclude exponential eADC/eDWI when quantitative ADC/dDWI exists, and exclude non-ADC DWI trace scans
        score = 0
        if "ddadc" in dl or "ddwi" in dl or "isoadc" in dl:
            score = 100
        elif "dwip" in dl:
            score = 98
        elif "adc" in dl and "eadc" not in dl and "edwi" not in dl:
            score = 95
        else:
            continue
        if "mix" in dl or "cal" in dl or "tra" in dl or "sense" in dl or "ax" in dl:
            score += 5
        scored.append((score, sid, desc))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][1], scored[0][2]


def _find_adc_scan_by_dicom_imagetype(xnat: XnatManager, exp_id: str, scans: list):
    """Fallback for series described generically (e.g. 'DWI Prostata b1400') whose DICOM ImageType is ADC."""
    import pydicom
    for sc in scans:
        sid = str(sc.get("ID", ""))
        desc = (sc.get("series_description") or sc.get("type") or "").strip()
        dl = desc.lower()
        if not any(k in dl for k in ("dwi", "diff", "ep2d")):
            continue
        flist = (xnat.get_json(f"/data/experiments/{exp_id}/scans/{sid}/files?format=json") or {}).get("ResultSet", {}).get("Result", [])
        dcm_entry = next((f for f in flist if not str(f.get("Name", "")).lower().endswith((".xml", ".txt", ".json"))), None)
        if not dcm_entry:
            continue
        fn = dcm_entry.get("Name", "")
        r = xnat.request("GET", f"/data/experiments/{exp_id}/scans/{sid}/files/{fn}", timeout=30)
        if not r or r.status_code != 200:
            continue
        try:
            ds = pydicom.dcmread(io.BytesIO(r.content), stop_before_pixels=True)
            it = "/".join(str(x).upper() for x in getattr(ds, "ImageType", []))
            if "ADC" in it and "EADC" not in it:
                return sid, f"{desc} [ImageType=ADC]"
        except Exception:
            continue
    return None


def _recover_adc_from_xnat(pat_id: str, raw_dir: str, xnat: XnatManager, mri_exps: dict) -> str:
    """Download & convert missing quantitative ADC DICOM scan from XNAT MRI experiment."""
    out_nii = os.path.join(raw_dir, "ADC_Recovered_Registered_to_CT.nii.gz")
    if os.path.isfile(out_nii) and os.path.getsize(out_nii) > 10_000:
        return out_nii

    for exp_id, _ in mri_exps.get(pat_id, []):
        sc_json = xnat.get_json(f"/data/experiments/{exp_id}/scans?format=json")
        if not sc_json:
            continue
        scans = sc_json.get("ResultSet", {}).get("Result", [])
        picked = _pick_best_adc_scan(scans) or _find_adc_scan_by_dicom_imagetype(xnat, exp_id, scans)
        if not picked:
            continue
        scan_id, scan_desc = picked
        print(f"  [{pat_id}] Downloading ADC scan {scan_id} ('{scan_desc}') from {exp_id}...", flush=True)
        try:
            r = xnat.request("GET", f"/data/experiments/{exp_id}/scans/{scan_id}/files?format=zip", timeout=120)
            if not r or r.status_code != 200:
                continue
            with tempfile.TemporaryDirectory() as tmpd:
                with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                    dcm_count = 0
                    for member in zf.namelist():
                        if member.endswith("/"):
                            continue
                        bn = os.path.basename(member)
                        if not bn or bn.endswith((".xml", ".txt", ".json")):
                            continue
                        with open(os.path.join(tmpd, bn), "wb") as df:
                            df.write(zf.read(member))
                        dcm_count += 1
                if dcm_count < 10:
                    continue
                reader = sitk.ImageSeriesReader()
                series_ids = reader.GetGDCMSeriesIDs(tmpd)
                best_fns = []
                for s_uid in series_ids:
                    fns = reader.GetGDCMSeriesFileNames(tmpd, s_uid)
                    if len(fns) > len(best_fns):
                        best_fns = fns
                if len(best_fns) < 10:
                    continue
                reader.SetFileNames(best_fns)
                im = reader.Execute()
                sitk.WriteImage(im, out_nii)
                print(f"  [{pat_id}] Reconstructed ADC NIfTI {im.GetSize()} sp={[round(x,2) for x in im.GetSpacing()]}", flush=True)
                return out_nii
        except Exception as exc:
            print(f"  [{pat_id}] WARN failed recovering ADC scan {scan_id}: {exc}", flush=True)
    return ""


def _needs_repair(pat_id: str, fetch_adc_if_missing: bool = False) -> tuple:
    """Return (needs_fix: bool, reasons: list[str])."""
    case_dir = os.path.join(CASES_DIR, pat_id)
    h5_path = os.path.join(case_dir, "preprocessed_volumes.h5")
    raw_dir = os.path.join(case_dir, "raw")
    if not (os.path.isfile(h5_path) and os.path.isdir(raw_dir)):
        return False, []

    reasons = []
    with h5py.File(h5_path, "r") as f:
        sh_raw = f["_meta_/scene_hierarchy.json"][()]
        sh = json.loads(sh_raw.decode() if isinstance(sh_raw, bytes) else sh_raw)
        mods = [n.get("modality") for n in sh if n.get("type") == "vtkMRMLLinearTransformNode"]

    mri_map = {}
    for fn in os.listdir(raw_dir):
        if fn.endswith(".nii.gz") and not fn.startswith(("Fixed_CT", "SUV", "CT", "totalseg")):
            if classify_mri_modality(fn):
                mri_map[fn] = os.path.join(raw_dir, fn)
    ranked = select_ranked_mri_sequences(mri_map)[:5]
    ranked_mods = [s["mod"] for s in ranked]
    has_quant_adc_raw = any(_is_quant_adc_fn(fn) for fn in mri_map)

    if ("ADC" not in mods and ("ADC" in ranked_mods or fetch_adc_if_missing)) or (
        fetch_adc_if_missing and not has_quant_adc_raw
    ):
        reasons.append("missing_adc")
    if "DCE" not in mods and "DCE" in ranked_mods:
        reasons.append("missing_dce")
    # Check interleaved or stacked DWI in raw_dir
    for s in ranked:
        if s["mod"] == "DWI" and os.path.isfile(s["item"]):
            im = sitk.ReadImage(s["item"])
            z = im.GetSize()[2]
            spz = im.GetSpacing()[2]
            if z >= 40 and z % 2 == 0:
                arr = sitk.GetArrayFromImage(im)
                me, mo = float(arr[0::2].mean()), float(arr[1::2].mean())
                if min(me, mo) > 0 and max(me, mo) / min(me, mo) >= 1.5:
                    reasons.append("interleaved_dwi")
                    break
            if z >= 60 and spz < 1.5:
                # Check if existing H5 DWI looks stacked/corrupted compared to unpacked
                reasons.append("stacked_dwi")
                break
    return len(reasons) > 0, reasons


def repair_case_mri(pat_id: str, xnat: XnatManager, mri_exps: dict, census: dict,
                    sync_cache: bool = False, upload: bool = False, fetch_adc: bool = True) -> dict:
    case_dir = os.path.join(CASES_DIR, pat_id)
    raw_dir = os.path.join(case_dir, "raw")
    h5_path = os.path.join(case_dir, "preprocessed_volumes.h5")
    ct_path = os.path.join(raw_dir, "Fixed_CT_Volume.nii.gz")
    if not (os.path.isfile(h5_path) and os.path.isfile(ct_path)):
        return {"patient": pat_id, "status": "SKIP"}

    # Step 1: Collect raw MRI files
    mri_map = {}
    for fn in os.listdir(raw_dir):
        if fn.endswith(".nii.gz") and not fn.startswith(("Fixed_CT", "SUV", "CT", "totalseg")):
            if classify_mri_modality(fn):
                mri_map[fn] = os.path.join(raw_dir, fn)

    has_adc_raw = any(_is_quant_adc_fn(fn) for fn in mri_map)
    if not has_adc_raw and fetch_adc and xnat is not None:
        rec_p = _recover_adc_from_xnat(pat_id, raw_dir, xnat, mri_exps)
        if rec_p and os.path.isfile(rec_p):
            mri_map[os.path.basename(rec_p)] = rec_p

    ranked = select_ranked_mri_sequences(mri_map)[:5]
    if not ranked:
        return {"patient": pat_id, "status": "NO_MRI"}

    ct_img = sitk.ReadImage(ct_path)
    tfm_path = os.path.join(raw_dir, "Transform_Landmark_SWD.h5")
    if not os.path.isfile(tfm_path):
        tfm_path = os.path.join(raw_dir, "Transform_FollowUp_to_Baseline_1.tfm")
    if not os.path.isfile(tfm_path):
        tfm_path = os.path.join(case_dir, "Transform_FollowUp_to_Baseline_1.tfm")
    tfm_obj = sitk.ReadTransform(tfm_path) if os.path.isfile(tfm_path) else sitk.AffineTransform(3)

    sp_arr = np.array(ct_img.GetSpacing(), dtype=np.float64)
    orig_arr = np.array(ct_img.GetOrigin(), dtype=np.float64)
    dir_arr = np.array(ct_img.GetDirection(), dtype=np.float64)

    with h5py.File(h5_path, "r") as hf:
        pf = int(hf["_meta_/preflipped"][()]) if "_meta_/preflipped" in hf else 1
        suv0_f = hf["BASELINE/SUV_PET_Image_0.nii.gz"][:]
        ct_shape = suv0_f.shape
        if "TFM_Transform_FollowUp_to_Baseline_1.tfm/max_anatomy.nii.gz" in hf:
            mri_anat_f = hf["TFM_Transform_FollowUp_to_Baseline_1.tfm/max_anatomy.nii.gz"][:].astype(np.uint16)
        else:
            mri_anat_f = np.zeros(ct_shape, dtype=np.uint16)
        if "TFM_Transform_FollowUp_to_Baseline_1.tfm/PET_Lesions_1.nii.gz" in hf:
            mr_gold_f = hf["TFM_Transform_FollowUp_to_Baseline_1.tfm/PET_Lesions_1.nii.gz"][:].astype(np.int16)
        else:
            mr_gold_f = np.zeros(ct_shape, dtype=np.int16)

    mri_anat_unflipped = np.flip(mri_anat_f, axis=1) if pf == 1 else mri_anat_f
    ref_t2_im = next((sitk.ReadImage(s["item"]) for s in ranked if s["mod"] == "T2"), None)
    ref_adc_im = next((sitk.ReadImage(s["item"]) for s in ranked if s["mod"] == "ADC"), None)

    rebuilt_series = []
    t2_fov_mask = None

    for seq in ranked:
        im = sitk.ReadImage(seq["item"])
        if im.GetNumberOfComponentsPerPixel() > 1:
            print(f"  [{pat_id}] WARN skipping multi-component/RGB image ({seq['fn']}, comp={im.GetNumberOfComponentsPerPixel()})", flush=True)
            continue
        im = unpack_stacked_mri_volume(im, seq["mod"], seq["fn"], ref_t2=ref_t2_im, ref_adc=ref_adc_im)
        already_on_ct_grid = (
            im.GetSize() == ct_img.GetSize()
            and np.allclose(im.GetSpacing(), ct_img.GetSpacing(), atol=1e-2)
            and np.allclose(im.GetOrigin(), ct_img.GetOrigin(), atol=1.0)
        )
        res_im = sitk.Resample(
            im, ct_img, sitk.Transform() if already_on_ct_grid else tfm_obj, sitk.sitkLinear, 0.0, sitk.sitkFloat32
        )
        arr = sitk.GetArrayFromImage(res_im).astype(np.float32)
        if np.max(arr) == 0.0 and already_on_ct_grid:
            res_im = sitk.Resample(im, ct_img, tfm_obj, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
            arr = sitk.GetArrayFromImage(res_im).astype(np.float32)
        arr = np.maximum(0.0, arr)

        if seq["mod"] == "T2" and t2_fov_mask is None:
            t2_fov_mask = (arr > 0)
        elif seq["mod"] == "ADC" and t2_fov_mask is not None and t2_fov_mask.any():
            nz = (arr > 0)
            inter = float(np.sum(nz & t2_fov_mask))
            ov = max(inter / max(1, int(nz.sum())), inter / max(1, int(t2_fov_mask.sum())))
            if ov < 0.50:
                print(f"  [{pat_id}] WARN skipping misaligned recovered ADC ({seq['fn']}, overlap={ov:.3f} < 0.50)", flush=True)
                continue

        pos = arr[arr > 0]
        if len(pos) > 0:
            p99 = float(np.percentile(pos, 99))
            scale = 1500.0 / max(p99, 1.0) if seq["mod"] == "ADC" else 1000.0 / max(p99, 1.0)
            arr = arr * scale
        diff_arr = compute_mri_diffusivity(arr, mri_anatomy=mri_anat_unflipped, exclusion_mask=None)
        rebuilt_series.append({
            "mod": seq["mod"],
            "fn": seq["fn"],
            "arr_f": (np.flip(arr, axis=1) if pf == 1 else arr).astype(np.float32),
            "diff_f": (np.flip(diff_arr, axis=1) if pf == 1 else diff_arr).astype(np.float32),
        })

    if not rebuilt_series:
        return {"patient": pat_id, "status": "EMPTY"}

    has_mr_les = bool(mr_gold_f.any())
    with h5py.File(h5_path, "r+") as hf:
        # Remove existing TFM_* groups so sequence indices 1..N match scene_hierarchy cleanly
        for k in list(hf.keys()):
            if k.startswith("TFM_"):
                del hf[k]

        for idx, s in enumerate(rebuilt_series, start=1):
            grp_name = f"TFM_Transform_FollowUp_to_Baseline_{idx}.tfm"
            g = hf.create_group(grp_name)
            for ds_name, data_arr in (
                (f"Fixed_CT_Volume_{idx}.nii.gz", s["arr_f"]),
                (f"SUV_PET_Image_{idx}.nii.gz", suv0_f),
                (f"PET_Lesions_{idx}.nii.gz", mr_gold_f),
                ("max_anatomy.nii.gz", mri_anat_f),
                ("diffusivity", s["diff_f"]),
            ):
                ds = g.create_dataset(ds_name, data=data_arr, compression="gzip", compression_opts=1)
                ds.attrs["spacing"] = sp_arr
                ds.attrs["origin"] = orig_arr
                ds.attrs["direction"] = dir_arr

        # Update scene_hierarchy.json non-transform root nodes + rebuilt TFM nodes
        sh_raw = hf["_meta_/scene_hierarchy.json"][()]
        old_sh = json.loads(sh_raw.decode() if isinstance(sh_raw, bytes) else sh_raw)
        old_tp1_segs = next(
            (ch.get("segments", []) for n in old_sh if n.get("type") == "vtkMRMLLinearTransformNode"
             for ch in n.get("children", []) if ch.get("name", "").startswith("PET_Lesions_")),
            None,
        )
        new_sh = [n for n in old_sh if n.get("type") != "vtkMRMLLinearTransformNode"]
        mr_segs_for_sh = list(old_tp1_segs) if (old_tp1_segs and has_mr_les) else (["Prostate Primary Lesion"] if has_mr_les else [])
        for idx, s in enumerate(rebuilt_series, start=1):
            new_sh.append({
                "name": f"Transform_FollowUp_to_Baseline_{idx}",
                "type": "vtkMRMLLinearTransformNode",
                "transform_type": "Rigid/Affine (Linear)",
                "modality": s["mod"],
                "children": [
                    {
                        "name": f"Fixed_CT_Volume_{idx}",
                        "type": "vtkMRMLScalarVolumeNode",
                        "modality": s["mod"],
                        "series_description": f"{s['mod']} MRI Registered",
                    },
                    {
                        "name": f"SUV_PET_Image_{idx}",
                        "type": "vtkMRMLScalarVolumeNode",
                        "modality": "PET",
                    },
                    {
                        "name": f"PET_Lesions_{idx}",
                        "type": "vtkMRMLSegmentationNode",
                        "segments": list(mr_segs_for_sh),
                    },
                    {
                        "name": f"max_anatomy_fixed_ct_{idx}",
                        "source": "max_anatomy.nii.gz",
                        "type": "vtkMRMLSegmentationNode",
                    },
                ],
            })
        del hf["_meta_/scene_hierarchy.json"]
        hf["_meta_"].create_dataset("scene_hierarchy.json", data=json.dumps(new_sh))

        # Sync organ_mapping & segment_names.json for TP 1..N while leaving TP 0 untouched
        for meta_key, default_tp_val in (
            ("organ_mapping", {"1": "Prostate"} if has_mr_les else {}),
            ("segment_names.json", {"1": "Prostate Primary Lesion"} if has_mr_les else {}),
        ):
            raw_m = hf[f"_meta_/{meta_key}"][()]
            mdict = json.loads(raw_m.decode() if isinstance(raw_m, bytes) else raw_m)
            existing_tp1 = mdict.get("1")
            tp_val = dict(existing_tp1) if (isinstance(existing_tp1, dict) and existing_tp1 and has_mr_les) else default_tp_val
            for k in list(mdict.keys()):
                if k != "0":
                    del mdict[k]
            for idx in range(1, len(rebuilt_series) + 1):
                mdict[str(idx)] = dict(tp_val)
            del hf[f"_meta_/{meta_key}"]
            hf["_meta_"].create_dataset(meta_key, data=json.dumps(mdict))

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

    if upload and pat_id in census:
        ok_up = upload_h5_and_clean_status(pat_id, h5_path, census, xnat)
        if not ok_up:
            return {"patient": pat_id, "status": "UPLOAD_FAIL", "mods": [s["mod"] for s in rebuilt_series]}

    mods_out = [s["mod"] for s in rebuilt_series]
    print(f"  ✓ {pat_id}: rebuilt MRIs -> {mods_out}", flush=True)
    return {"patient": pat_id, "status": "OK", "mods": mods_out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", help="Comma-separated Pat IDs (default: auto-detect all affected Pat* cases)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sync-cache", action="store_true")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--no-xnat-fetch", action="store_true", help="Do not download missing ADC DICOMs from XNAT")
    args = ap.parse_args()

    with open(CENSUS_PATH, "r") as cf:
        census = json.load(cf)

    if args.cases:
        target_cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        all_cases = sorted(
            [os.path.basename(p) for p in glob.glob(os.path.join(CASES_DIR, "Pat*"))
             if re.fullmatch(r"Pat\d+", os.path.basename(p)) and os.path.isfile(os.path.join(p, "preprocessed_volumes.h5"))],
            key=lambda s: int(s[3:]),
        )
        target_cases = []
        for pat in all_cases:
            need, why = _needs_repair(pat, fetch_adc_if_missing=not args.no_xnat_fetch)
            if need:
                target_cases.append((pat, why))
        print(f"Auto-detected {len(target_cases)} cases needing MRI/ADC/DWI repair:")
        for pat, why in target_cases:
            print(f"  {pat}: {why}")
        if args.dry_run:
            return
        target_cases = [pat for pat, _ in target_cases]

    if args.dry_run:
        return

    need_xnat = (not args.no_xnat_fetch) or args.upload
    xnat = XnatManager() if need_xnat else None
    mri_exps = _index_xnat_mri_experiments(xnat) if (not args.no_xnat_fetch and xnat is not None) else {}

    ok = 0
    for pat in target_cases:
        res = repair_case_mri(
            pat, xnat, mri_exps, census,
            sync_cache=args.sync_cache, upload=args.upload, fetch_adc=not args.no_xnat_fetch
        )
        if res["status"] == "OK":
            ok += 1
    print(f"Completed: {ok}/{len(target_cases)} cases repaired.")


if __name__ == "__main__":
    main()
