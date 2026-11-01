#!/usr/bin/env python3
"""
batch_process_xnat_bimodal.py

Dual-GPU batch orchestrator for FDM_DPI-2024-7-KRN_Prostata_bimodal_PIPELINE.
Processes all ready patients from XNAT concurrently across GPU 0 and GPU 1:
  1. Reuses local CT/SUV cache from /mnt/big/project_ssd/project_ssd/lymph_node_rules/data/xnat_data/prostata_Pat{X}/
     or downloads from XNAT using a single cookie-authenticated JSESSION.
  2. Downloads packed_anatomical_tensor.h5 (+ registry), primary_masks.h5, prostate_segmentation.nii.gz,
     Lesions NRRDs, and original moving T2 MRI (handling slicer_final, slicer, and slicer_2.2/.mrb fallbacks).
  3. Reprocesses lymph node rules using the current version of /mnt/big/project_ssd/project_ssd/lymph_node_rules
     via a persistent Julia GPU worker daemon on the assigned GPU.
  4. Runs TotalSegmentator MR (-ta total_mr --ml --fast) and executes proper CT-MRI registration
     via docs/register_ct_mri.py (3D canonical prostate keypoints + PyTorch CUDA SWD refinement).
  5. Packages the MedEye3d preprocessed_volumes.h5 via prepare_prostata_bimodal_cases.process_case.
  6. Uploads {subj_label}_preprocessed.h5 to XNAT MEDEYE3D resource, updates ANNOTATIONS/status.json,
     uploads updated CT_MRI_REGISTRATION transforms, and cleans up raw temporary files.
  7. Polls remaining waiting cases until all 133 patients are processed.
"""

import os
import sys
import json
import time
import re
import glob
import shutil
import zipfile
import tempfile
import threading
import subprocess
import argparse
import requests
import numpy as np
import h5py
import SimpleITK as sitk

sys.path.insert(0, "/workspaces/MedEye3d.jl/docs")
sys.path.insert(0, "/workspaces/MedEye3d.jl/scripts/preprocessing")

import register_ct_mri
import prepare_prostata_bimodal_cases as prep

CENSUS_PATH = "/workspaces/MedEye3d.jl/data/xnat_bimodal_census.json"
OUT_BASE_DIR = "/workspaces/MedEye3d.jl/data/cases/Prostata_bimodal"
LOCAL_XNAT_CACHE = "/mnt/big/project_ssd/project_ssd/lymph_node_rules/data/xnat_data"
LOCAL_RESTORED_CACHE = "/mnt/big/project_ssd/project_ssd/lymph_node_rules/data/processed_cases_restored"
PROGRESS_JSON = "/workspaces/MedEye3d.jl/data/batch_progress.json"


class XnatClient:
    """Thread-safe XNAT client that authenticates ONCE via /data/JSESSION and reuses the cookie."""
    def __init__(self):
        self.auth, self.base = prep.load_credentials()
        self.lock = threading.Lock()
        self.session = requests.Session()
        self._authenticate()

    def _authenticate(self):
        with self.lock:
            r = self.session.get(f"{self.base}/data/JSESSION", auth=self.auth, timeout=30)
            r.raise_for_status()
            jsess = r.text.strip()
            self.session.cookies.set("JSESSIONID", jsess)
            print(f"[XNAT] Authenticated JSESSIONID={jsess[:10]}...")

    def request(self, method, url, **kwargs):
        if not url.startswith("http"):
            url = f"{self.base}{url}"
        for attempt in range(4):
            try:
                r = self.session.request(method, url, **kwargs)
                if r.status_code == 401:
                    print(f"[XNAT] Session expired (401) on {url}, re-authenticating...")
                    self._authenticate()
                    r = self.session.request(method, url, **kwargs)
                return r
            except Exception as e:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        return None

    def get_json(self, url):
        r = self.request("GET", url, timeout=30)
        if r is not None and r.status_code == 200:
            try:
                return r.json()
            except Exception:
                return {}
        return {}

    def get_files_map(self, url):
        data = self.get_json(url)
        res = data.get("ResultSet", {}).get("Result", [])
        return {f["Name"]: f["URI"] for f in res}

    def download_file(self, uri_or_url, dest_path, min_size=512):
        if os.path.isfile(dest_path) and os.path.getsize(dest_path) >= min_size:
            return True
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        tmp_path = f"{dest_path}.tmp.{threading.get_ident()}"
        try:
            r = self.request("GET", uri_or_url, stream=True, timeout=120)
            if r is None or r.status_code != 200:
                return False
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            if os.path.getsize(tmp_path) < min_size:
                os.remove(tmp_path)
                return False
            os.replace(tmp_path, dest_path)
            return True
        except Exception as e:
            print(f"[XNAT] Download error {uri_or_url}: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False

    def upload_file(self, uri, local_path):
        with open(local_path, "rb") as f:
            r = self.request(
                "PUT",
                uri,
                data=f,
                headers={"Content-Type": "application/octet-stream"},
                timeout=600
            )
        return r is not None and r.status_code in (200, 201)


def pick_best_moving_t2(files_list):
    orig = [f for f in files_list if f.endswith("_Original_for_Comparison.nii.gz") or (f.endswith(".nii.gz") and "Original" in f)]
    if not orig:
        return None
    def score(fname):
        s = 0
        fl = fname.lower()
        if "t2" in fl:
            s += 100
        if any(k in fl for k in ["tra", "ax", "_ax_", "trans"]):
            s += 50
        if any(k in fl for k in ["cor", "sag"]):
            s -= 40
        if any(k in fl for k in ["dwi", "dw_", "adc", "dce", "t1", "lava", "water", "sub.", "dyn", "ktrans", "b-"]):
            s -= 80
        return s
    orig.sort(key=lambda x: (score(x), len(x)), reverse=True)
    return orig[0]


def extract_ct_prostate_mask(anat_h5_path, ct_img_path, out_mask_path):
    """Extracts binary CT prostate mask from packed_anatomical_tensor.h5 and saves with CT NIfTI header."""
    if os.path.isfile(out_mask_path) and os.path.getsize(out_mask_path) > 1000:
        return True
    if not os.path.isfile(anat_h5_path) or not os.path.isfile(ct_img_path):
        return False
    try:
        ct_img = sitk.ReadImage(ct_img_path)
        with h5py.File(anat_h5_path, "r") as af:
            pt = af["packed_tensor"][:]
            reg = prep.load_anat_registry(anat_h5_path, af)
            for p_key in ["nvseg_prostate", "prostate"]:
                if p_key in reg:
                    ch = reg[p_key]["channel"] - 1
                    mid = reg[p_key]["id"]
                    if 0 <= ch < pt.shape[0]:
                        m = (pt[ch] == mid).astype(np.uint8)
                        if m.sum() > 0:
                            m_img = sitk.GetImageFromArray(m)
                            m_img.CopyInformation(ct_img)
                            sitk.WriteImage(m_img, out_mask_path)
                            return True
    except Exception as e:
        print(f"  [WARN] Failed to extract CT prostate mask from {anat_h5_path}: {e}")
    return False


def extract_mri_masks_from_totalseg_ml(ts_mr_ml_path, mri_seg_dir):
    """Unpacks multi-label totalseg_mr_ml.nii.gz into individual binary masks for register_ct_mri.py."""
    if not os.path.isfile(ts_mr_ml_path):
        return False
    os.makedirs(mri_seg_dir, exist_ok=True)
    try:
        img = sitk.ReadImage(ts_mr_ml_path)
        arr = sitk.GetArrayFromImage(img)
        # TotalSegmentator total_mr class IDs
        cid_map = {
            17: "prostate",
            16: "urinary_bladder",
            18: "sacrum",
            36: "femur_left",
            37: "femur_right",
            38: "hip_left",
            39: "hip_right",
        }
        for cid, name in cid_map.items():
            m = (arr == cid).astype(np.uint8)
            if m.sum() > 0:
                out_p = os.path.join(mri_seg_dir, f"{name}.nii.gz")
                m_img = sitk.GetImageFromArray(m)
                m_img.CopyInformation(img)
                sitk.WriteImage(m_img, out_p)
        return os.path.isfile(os.path.join(mri_seg_dir, "prostate.nii.gz"))
    except Exception as e:
        print(f"  [WARN] Failed to unpack totalseg_mr_ml: {e}")
        return False


def run_ln_job_via_daemon(queue_dir, pat_id, primary_h5_path, out_h5_path, timeout=1800):
    """Submits a job to the persistent Julia GPU worker daemon and waits for completion."""
    if os.path.isfile(out_h5_path) and os.path.getsize(out_h5_path) > 100 * 1024:
        print(f"  [LN_DAEMON] Reusing existing {os.path.basename(out_h5_path)}")
        return True
    job_id = f"{pat_id}_{int(time.time()*1000)}"
    req_file = os.path.join(queue_dir, f"{job_id}.req.json")
    done_file = os.path.join(queue_dir, f"{job_id}.done.json")
    with open(req_file + ".tmp", "w") as f:
        json.dump({"primary_h5": primary_h5_path, "out_h5": out_h5_path}, f)
    os.replace(req_file + ".tmp", req_file)

    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.isfile(done_file):
            try:
                with open(done_file, "r") as f:
                    resp = json.load(f)
                os.remove(done_file)
                if resp.get("status") == "ok" and os.path.isfile(out_h5_path):
                    return True
                else:
                    print(f"  [LN_DAEMON] Error on {pat_id}: {resp.get('error')}")
                    return False
            except Exception as e:
                time.sleep(0.5)
        time.sleep(1.0)
    print(f"  [LN_DAEMON] Timeout waiting for {pat_id} after {timeout}s")
    return False


def process_single_patient(pat_info, xnat, gpu_id, queue_dir, force_reprocess=False):
    pat_id = pat_info["pat_id"]
    subj_id = pat_info["subj_id"]
    subj_label = pat_info["subj_label"]
    exp_id = pat_info["exp_id"]
    all_subj_ids = pat_info.get("all_subj_ids", [subj_id])

    case_dir = os.path.join(OUT_BASE_DIR, pat_id)
    raw_dir = os.path.join(case_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    h5_target = os.path.join(case_dir, "preprocessed_volumes.h5")

    print(f"\n[GPU {gpu_id}] >>> Starting {pat_id} (Subj: {subj_id}, Exp: {exp_id}) <<<")
    t_start = time.time()

    # 1. Stage CT and PET SUV from local NVMe cache if available
    ct_dest = os.path.join(raw_dir, "Fixed_CT_Volume.nii.gz")
    suv_dest = os.path.join(raw_dir, "SUV_PET_WB_120min_Uncorrected.nii.gz")

    local_xnat_dir = os.path.join(LOCAL_XNAT_CACHE, f"prostata_{pat_id}")
    if not os.path.isfile(ct_dest):
        for cand in [
            os.path.join(local_xnat_dir, "Fixed_CT_Volume.nii.gz"),
            os.path.join(local_xnat_dir, "CT.nii.gz"),
            os.path.join(LOCAL_RESTORED_CACHE, f"FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__{pat_id}", "Fixed_CT_Volume.nii.gz")
        ]:
            if os.path.isfile(cand) and os.path.getsize(cand) > 100000:
                shutil.copyfile(cand, ct_dest)
                print(f"  [GPU {gpu_id}] {pat_id}: Copied CT from local cache {cand}")
                break

    if not os.path.isfile(suv_dest):
        for cand in [
            os.path.join(local_xnat_dir, "SUV.nii.gz"),
            os.path.join(LOCAL_RESTORED_CACHE, f"FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__{pat_id}", "SUV_PET_Image.nii.gz")
        ]:
            if os.path.isfile(cand) and os.path.getsize(cand) > 100000:
                shutil.copyfile(cand, suv_dest)
                print(f"  [GPU {gpu_id}] {pat_id}: Copied SUV from local cache {cand}")
                break

    # Query XNAT resources
    slicer_files = {}
    for sid in all_subj_ids:
        sf = xnat.get_files_map(f"/data/projects/Prostata_bimodal_PIPELINE/subjects/{sid}/resources/slicer_final/files?format=json")
        if len(sf) > len(slicer_files):
            slicer_files = sf
    if len(slicer_files) <= 1:
        # Check fallback 'slicer' resource (e.g. Pat13)
        for sid in all_subj_ids:
            sf_alt = xnat.get_files_map(f"/data/projects/Prostata_bimodal_PIPELINE/subjects/{sid}/resources/slicer/files?format=json")
            if len(sf_alt) > len(slicer_files):
                slicer_files = sf_alt

    suv_files = {}
    for sid in all_subj_ids:
        sv = xnat.get_files_map(f"/data/projects/Prostata_bimodal_PIPELINE/subjects/{sid}/resources/scans_SUV_CT_PET/files?format=json")
        if sv:
            suv_files.update(sv)

    anat_files = xnat.get_files_map(f"/data/experiments/{exp_id}/resources/ANATOMICAL_TENSOR/files?format=json")
    ln_files = xnat.get_files_map(f"/data/experiments/{exp_id}/resources/LYMPH_NODE_RESULTS/files?format=json")
    prostate_files = xnat.get_files_map(f"/data/experiments/{exp_id}/resources/PROSTATE_SEGMENTATION/files?format=json")
    reg_files = xnat.get_files_map(f"/data/experiments/{exp_id}/resources/CT_MRI_REGISTRATION/files?format=json")

    # Download CT / SUV if not in local cache
    if not os.path.isfile(ct_dest):
        for k in ["Fixed_CT_Volume.nii.gz", "CT_WB_120min_Abdomen_3_0.nii.gz"]:
            uri = slicer_files.get(k) or suv_files.get(k)
            if uri and xnat.download_file(uri, ct_dest, min_size=100000):
                break
        if not os.path.isfile(ct_dest):
            for k, uri in suv_files.items():
                if "CT" in k and k.endswith(".nii.gz"):
                    if xnat.download_file(uri, ct_dest, min_size=100000):
                        break

    if not os.path.isfile(suv_dest):
        for k, uri in suv_files.items():
            if "SUV" in k and k.endswith(".nii.gz"):
                if xnat.download_file(uri, suv_dest, min_size=100000):
                    break

    # Download packed_anatomical_tensor.h5 & registry
    anat_h5_path = os.path.join(raw_dir, "packed_anatomical_tensor.h5")
    anat_reg_path = os.path.join(raw_dir, "packed_anatomical_tensor_registry.json")
    restored_case_dir = os.path.join(LOCAL_RESTORED_CACHE, f"FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__{pat_id}")
    if not os.path.isfile(anat_h5_path):
        local_anat = os.path.join(restored_case_dir, "packed_anatomical_tensor.h5")
        if os.path.isfile(local_anat) and os.path.getsize(local_anat) > 1000000:
            shutil.copyfile(local_anat, anat_h5_path)
        elif "packed_anatomical_tensor.h5" in anat_files:
            xnat.download_file(anat_files["packed_anatomical_tensor.h5"], anat_h5_path, min_size=1000000)

    if not os.path.isfile(anat_reg_path):
        local_reg = os.path.join(restored_case_dir, "packed_anatomical_tensor_registry.json")
        if os.path.isfile(local_reg):
            shutil.copyfile(local_reg, anat_reg_path)
        elif "packed_anatomical_tensor_registry.json" in anat_files:
            xnat.download_file(anat_files["packed_anatomical_tensor_registry.json"], anat_reg_path, min_size=100)

    # Recompute lymph node rules via persistent Julia GPU worker (unpacks directly in RAM from packed_anatomical_tensor.h5)
    primary_h5_path = os.path.join(raw_dir, "primary_masks.h5")
    final_ln_h5_path = os.path.join(raw_dir, "final_results_gpu.h5")
    if force_reprocess and os.path.isfile(final_ln_h5_path):
        os.remove(final_ln_h5_path)

    if not os.path.isfile(final_ln_h5_path):
        if os.path.isfile(anat_h5_path) and os.path.isfile(ct_dest):
            print(f"  [GPU {gpu_id}] {pat_id}: Reprocessing lymph node rules via Julia GPU worker...")
            ok_ln = run_ln_job_via_daemon(queue_dir, pat_id, primary_h5_path, final_ln_h5_path)
            if not ok_ln and "final_results_gpu.h5" in ln_files:
                print(f"  [GPU {gpu_id}] {pat_id}: Fallback to XNAT final_results_gpu.h5...")
                xnat.download_file(ln_files["final_results_gpu.h5"], final_ln_h5_path, min_size=100000)
        elif "final_results_gpu.h5" in ln_files:
            xnat.download_file(ln_files["final_results_gpu.h5"], final_ln_h5_path, min_size=100000)

    # Download Lesions NRRD and Moving MRI
    lesion_ct_dest = os.path.join(raw_dir, "NNUNET_OR_Processed_Lesions_CT.seg.nrrd")
    lesion_mr_dest = os.path.join(raw_dir, "NNUNET_OR_Processed_Lesions_MR.seg.nrrd")

    for k in ["NNUNET_OR_Processed_Lesions_CT.seg.nrrd", "Processed_Lesions_CT.seg.nrrd"]:
        if k in slicer_files and not os.path.isfile(lesion_ct_dest):
            xnat.download_file(slicer_files[k], lesion_ct_dest, min_size=100)
            break

    for k in ["NNUNET_OR_Processed_Lesions_MR.seg.nrrd", "Processed_Lesions_MR.seg.nrrd"]:
        if k in slicer_files and not os.path.isfile(lesion_mr_dest):
            xnat.download_file(slicer_files[k], lesion_mr_dest, min_size=100)
            break

    moving_mri_name = pick_best_moving_t2(list(slicer_files.keys()))
    moving_mri_path = os.path.join(raw_dir, moving_mri_name) if moving_mri_name else None
    if moving_mri_name and not os.path.isfile(moving_mri_path):
        xnat.download_file(slicer_files[moving_mri_name], moving_mri_path, min_size=10000)

    # Download all available ranked MRI modalities (ADC, DWI, T1, DCE, etc.) for multi-modal packaging
    ranked_mri_seqs = prep.select_ranked_mri_sequences(slicer_files)
    for seq in ranked_mri_seqs:
        m_name = seq["fn"]
        m_dest = os.path.join(raw_dir, m_name)
        if not os.path.isfile(m_dest) and m_name in slicer_files:
            xnat.download_file(slicer_files[m_name], m_dest, min_size=10000)

    # Fallback for Pat2: extract from slicer_2.2/verification_scene.mrb if slicer_final had no lesions/MRI
    if not moving_mri_path or not os.path.isfile(moving_mri_path):
        for sid in all_subj_ids:
            s22 = xnat.get_files_map(f"/data/projects/Prostata_bimodal_PIPELINE/subjects/{sid}/resources/slicer_2.2/files?format=json")
            if "verification_scene.mrb" in s22:
                mrb_tmp = os.path.join(raw_dir, "slicer_2_2_scene.mrb")
                if xnat.download_file(s22["verification_scene.mrb"], mrb_tmp, min_size=100000):
                    try:
                        with zipfile.ZipFile(mrb_tmp, "r") as zf:
                            for zn in zf.namelist():
                                bname = os.path.basename(zn)
                                if bname == "NNUNET_OR_Processed_Lesions_CT.seg.nrrd" and not os.path.isfile(lesion_ct_dest):
                                    with open(lesion_ct_dest, "wb") as out_f:
                                        out_f.write(zf.read(zn))
                                elif bname == "NNUNET_OR_Processed_Lesions_MR.seg.nrrd" and not os.path.isfile(lesion_mr_dest):
                                    with open(lesion_mr_dest, "wb") as out_f:
                                        out_f.write(zf.read(zn))
                                elif "Original" in bname and (bname.endswith(".nrrd") or bname.endswith(".nii.gz")):
                                    extr_p = os.path.join(raw_dir, bname)
                                    with open(extr_p, "wb") as out_f:
                                        out_f.write(zf.read(zn))
                                    # Convert to .nii.gz for TotalSegmentator compatibility
                                    nii_p = os.path.join(raw_dir, "MR_Original_for_Comparison.nii.gz")
                                    im_tmp = sitk.ReadImage(extr_p)
                                    sitk.WriteImage(im_tmp, nii_p)
                                    moving_mri_path = nii_p
                                    moving_mri_name = "MR_Original_for_Comparison.nii.gz"
                    except Exception as e_mrb:
                        print(f"  [WARN] {pat_id}: Failed to extract slicer_2.2 mrb: {e_mrb}")
                    if os.path.isfile(mrb_tmp):
                        os.remove(mrb_tmp)
                break

    # Download prostate_segmentation.nii.gz if available
    prostate_seg_path = os.path.join(raw_dir, "prostate_segmentation.nii.gz")
    if "prostate_segmentation.nii.gz" in prostate_files and not os.path.isfile(prostate_seg_path):
        xnat.download_file(prostate_files["prostate_segmentation.nii.gz"], prostate_seg_path, min_size=500)

    # 4. Run TotalSegmentator MR + CT-MRI Registration via docs/register_ct_mri.py
    swd_mri_dest = os.path.join(raw_dir, "MRI_registered_to_CT_SWD.nii.gz")
    swd_tfm_dest = os.path.join(raw_dir, "Transform_Landmark_SWD.h5")

    if moving_mri_path and os.path.isfile(moving_mri_path):
        env_gpu = os.environ.copy()
        env_gpu["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # 4a. Run TotalSegmentator MR multi-label once (gives both pelvic MRI anatomy and MRI prostate/bone masks!)
        ts_mr_ml_path = os.path.join(raw_dir, "totalseg_mr_ml.nii.gz")
        if not os.path.isfile(ts_mr_ml_path):
            totalseg_bin = shutil.which("TotalSegmentator") or "/usr/local/bin/TotalSegmentator"
            print(f"  [GPU {gpu_id}] {pat_id}: Running TotalSegmentator MR on {os.path.basename(moving_mri_path)}...")
            try:
                subprocess.run(
                    [totalseg_bin, "-i", moving_mri_path, "-o", ts_mr_ml_path, "-ta", "total_mr", "--ml", "--fast", "--quiet"],
                    env=env_gpu,
                    check=True
                )
            except Exception as e_ts:
                print(f"  [WARN] {pat_id}: TotalSegmentator MR failed: {e_ts}")

        # 4b. Prepare seg_cache_dir for register_ct_mri.py so it doesn't need to re-invoke TotalSegmentator!
        reg_out_dir = os.path.join(raw_dir, "reg_swd")
        seg_cache_dir = os.path.join(reg_out_dir, "segmentations")
        ct_seg_dir = os.path.join(seg_cache_dir, "ct")
        mri_seg_dir = os.path.join(seg_cache_dir, "mri")
        os.makedirs(ct_seg_dir, exist_ok=True)
        os.makedirs(mri_seg_dir, exist_ok=True)

        # Extract CT prostate mask from packed_anatomical_tensor.h5 (or download CT_Prostate_Mask.nii.gz)
        ct_pro_mask_path = os.path.join(ct_seg_dir, "prostate.nii.gz")
        if not extract_ct_prostate_mask(anat_h5_path, ct_dest, ct_pro_mask_path):
            if "CT_Prostate_Mask.nii.gz" in reg_files:
                xnat.download_file(reg_files["CT_Prostate_Mask.nii.gz"], ct_pro_mask_path, min_size=500)

        # Extract MRI prostate mask from prostate_segmentation.nii.gz (if in moving MRI space) or totalseg_mr_ml.nii.gz
        mri_pro_mask_path = os.path.join(mri_seg_dir, "prostate.nii.gz")
        extract_mri_masks_from_totalseg_ml(ts_mr_ml_path, mri_seg_dir)
        if os.path.isfile(prostate_seg_path):
            try:
                p_im = sitk.ReadImage(prostate_seg_path)
                mri_im = sitk.ReadImage(moving_mri_path)
                # If prostate_segmentation.nii.gz is in moving MRI space, prefer its high-precision DL prostate mask!
                if p_im.GetSize() == mri_im.GetSize():
                    p_ar = sitk.GetArrayFromImage(p_im)
                    m_bin = ((p_ar >= 1) & (p_ar <= 8)).astype(np.uint8) if p_ar.max() > 2 else (p_ar > 0).astype(np.uint8)
                    if m_bin.sum() > 0:
                        m_im = sitk.GetImageFromArray(m_bin)
                        m_im.CopyInformation(p_im)
                        sitk.WriteImage(m_im, mri_pro_mask_path)
            except Exception:
                pass

        # 4c. Run register_ct_mri.py (3D canonical prostate keypoints + PyTorch CUDA SWD refinement)
        print(f"  [GPU {gpu_id}] {pat_id}: Running register_ct_mri.py (3D Prostate Keypoints + SWD)...")
        try:
            # Ensure PyTorch uses the assigned GPU inside this process/subprocess
            reg_cmd = [
                sys.executable, "-c",
                f"import sys; sys.path.insert(0, '/workspaces/MedEye3d.jl/docs'); "
                f"import register_ct_mri; "
                f"register_ct_mri.run_registration("
                f"ct_path={repr(ct_dest)}, mri_path={repr(moving_mri_path)}, output_dir={repr(reg_out_dir)}, "
                f"seg_cache_dir={repr(seg_cache_dir)}, swd_epochs=800, skip_mrb=True)"
            ]
            subprocess.run(reg_cmd, env=env_gpu, check=True)
            gen_swd_nii = os.path.join(reg_out_dir, "nifti", "MRI_registered_to_CT_SWD.nii.gz")
            gen_swd_tfm = os.path.join(reg_out_dir, "transforms", "Transform_Landmark_SWD.h5")
            if os.path.isfile(gen_swd_nii):
                shutil.move(gen_swd_nii, swd_mri_dest)
            if os.path.isfile(gen_swd_tfm):
                shutil.move(gen_swd_tfm, swd_tfm_dest)
        except Exception as e_reg:
            print(f"  [ERROR] {pat_id}: Registration failed: {e_reg}")

    # 5. Run process_case (offline=True since all required files are staged in raw_dir!)
    case_cfg = {
        "patient_id": pat_id,
        "subject_id": subj_id,
        "exp_id": exp_id,
        "mri_pattern": "T2"
    }
    import importlib
    importlib.reload(prep)
    h5_out = prep.process_case(case_cfg, OUT_BASE_DIR, offline=True)
    if not os.path.isfile(h5_out) or os.path.getsize(h5_out) < 5 * 1024 * 1024:
        raise RuntimeError(f"Failed to build valid HDF5 for {pat_id}")

    # 6. Upload to XNAT:
    # 6a. Upload {subj_label}_preprocessed.h5 to MEDEYE3D resource on Prostata_bimodal_PIPELINE
    remote_h5_name = f"{subj_label}_preprocessed.h5"
    medeye_uri = f"/data/projects/Prostata_bimodal_PIPELINE/subjects/{subj_id}/resources/MEDEYE3D/files/{remote_h5_name}?overwrite=true&inbody=true"
    print(f"  [GPU {gpu_id}] {pat_id}: Uploading {remote_h5_name} ({os.path.getsize(h5_out)/1e6:.1f} MB) to XNAT...")
    up_ok = xnat.upload_file(medeye_uri, h5_out)
    if not up_ok:
        raise RuntimeError(f"XNAT upload failed for {pat_id} -> {medeye_uri}")

    # 6b. Update status.json in ANNOTATIONS
    status_tmp = os.path.join(raw_dir, "status.json")
    with open(status_tmp, "w") as sf:
        json.dump({
            "status": "PREPROCESSED",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "patient_id": pat_id,
            "subject_id": subj_id,
            "experiment_id": exp_id
        }, sf, indent=2)
    annot_uri = f"/data/projects/Prostata_bimodal_PIPELINE/subjects/{subj_id}/resources/ANNOTATIONS/files/status.json?overwrite=true&inbody=true"
    xnat.upload_file(annot_uri, status_tmp)

    # 6c. Upload updated Transform_Landmark_SWD.h5 and registration_summary.json to CT_MRI_REGISTRATION on exp_id
    if os.path.isfile(swd_tfm_dest):
        xnat.upload_file(f"/data/experiments/{exp_id}/resources/CT_MRI_REGISTRATION/files/Transform_Landmark_SWD.h5?overwrite=true&inbody=true", swd_tfm_dest)
    reg_sum_path = os.path.join(raw_dir, "reg_swd", "registration_summary.json")
    if os.path.isfile(reg_sum_path):
        xnat.upload_file(f"/data/experiments/{exp_id}/resources/CT_MRI_REGISTRATION/files/registration_summary.json?overwrite=true&inbody=true", reg_sum_path)
    if os.path.isfile(final_ln_h5_path):
        xnat.upload_file(f"/data/experiments/{exp_id}/resources/LYMPH_NODE_RESULTS/files/final_results_gpu.h5?overwrite=true&inbody=true", final_ln_h5_path)

    # 7. Clean up raw_dir and local preprocessed_volumes.h5 (keep disk usage low across 133 patients; keep Pat23/24/33/44 local if desired)
    shutil.rmtree(raw_dir, ignore_errors=True)
    if pat_id not in ("Pat23", "Pat24", "Pat33", "Pat44"):
        if os.path.isfile(h5_out):
            os.remove(h5_out)

    elapsed = time.time() - t_start
    print(f"[GPU {gpu_id}] ✅ COMPLETED & UPLOADED {pat_id} in {elapsed:.1f}s!")
    return True


def start_julia_worker_daemon(gpu_id, queue_dir, log_path):
    """Spawns the persistent Julia GPU worker daemon on the specified GPU."""
    shutil.rmtree(queue_dir, ignore_errors=True)
    os.makedirs(queue_dir, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["JULIA_NUM_THREADS"] = "8"
    julia_bin = shutil.which("julia") or "/usr/local/bin/julia"
    worker_script = "/workspaces/MedEye3d.jl/scripts/preprocessing/run_ln_worker.jl"
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [julia_bin, "--project=/mnt/big/project_ssd/project_ssd/lymph_node_rules/src/julia_gpu", worker_script, queue_dir],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT
    )
    print(f"[ORCHESTRATOR] Spawned Julia GPU Worker for GPU {gpu_id} (PID {proc.pid}, log: {log_path})")
    return proc, log_file


def save_progress(completed_set, failed_dict, waiting_list):
    tmp = PROGRESS_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump({
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "completed_count": len(completed_set),
            "completed": sorted(completed_set, key=lambda x: int(x.replace("Pat", ""))),
            "failed": failed_dict,
            "waiting_on_server": waiting_list
        }, f, indent=2)
    os.replace(tmp, PROGRESS_JSON)


def refresh_waiting_cases(xnat, census, waiting_pids):
    """Checks XNAT to see if any waiting cases now have ANATOMICAL_TENSOR and LYMPH_NODE_RESULTS."""
    newly_ready = []
    still_waiting = []
    exps_bm = xnat.get_json("/data/projects/Prostata_bimodal/experiments?format=json").get("ResultSet", {}).get("Result", [])
    subjs_pl = xnat.get_json("/data/projects/Prostata_bimodal_PIPELINE/subjects?format=json").get("ResultSet", {}).get("Result", [])

    pat_exps = {}
    for e in exps_bm:
        m = re.search(r"Pat(\d+)$", e["label"])
        if m:
            pat_exps.setdefault(f"Pat{m.group(1)}", []).append(e)

    pat_subjs = {}
    for s in subjs_pl:
        m = re.search(r"Pat(\d+)$", s["label"])
        if m:
            pat_subjs.setdefault(f"Pat{m.group(1)}", []).append(s)

    for pid in waiting_pids:
        best_e = None
        best_res = []
        for e in pat_exps.get(pid, []):
            eid = e["ID"]
            res_list = xnat.get_json(f"/data/experiments/{eid}/resources?format=json").get("ResultSet", {}).get("Result", [])
            res = [x["label"] for x in res_list]
            if "ANATOMICAL_TENSOR" in res or best_e is None:
                best_e = e
                best_res = res
                if "ANATOMICAL_TENSOR" in res and "PETPSMA_0" in e["label"]:
                    break

        best_s = None
        best_s_res = []
        s_files = []
        all_sids = []
        for s in pat_subjs.get(pid, []):
            sid = s["ID"]
            all_sids.append(sid)
            sres_list = xnat.get_json(f"/data/projects/Prostata_bimodal_PIPELINE/subjects/{sid}/resources?format=json").get("ResultSet", {}).get("Result", [])
            sres = [x["label"] for x in sres_list]
            if "slicer_final" in sres or best_s is None:
                best_s = s
                best_s_res = sres
                if "slicer_final" in sres:
                    sf_list = xnat.get_json(f"/data/projects/Prostata_bimodal_PIPELINE/subjects/{sid}/resources/slicer_final/files?format=json").get("ResultSet", {}).get("Result", [])
                    s_files = [x["Name"] for x in sf_list]
                    if s["label"].startswith("FDM_DPI"):
                        break

        if best_e and best_s:
            has_anat = "ANATOMICAL_TENSOR" in best_res
            has_ln = "LYMPH_NODE_RESULTS" in best_res
            census[pid] = {
                "pat_id": pid,
                "subj_id": best_s["ID"],
                "all_subj_ids": all_sids,
                "subj_label": best_s["label"],
                "exp_id": best_e["ID"],
                "exp_label": best_e["label"],
                "exp_res": best_res,
                "subj_res": best_s_res,
                "slicer_final_files": s_files,
                "has_anat": has_anat,
                "has_ln": has_ln,
                "has_reg": "CT_MRI_REGISTRATION" in best_res,
                "has_prostate_seg": "PROSTATE_SEGMENTATION" in best_res,
                "has_medeye3d": "MEDEYE3D" in best_s_res
            }
            if has_anat and has_ln:
                print(f"[POLL] 🎉 {pid} is now READY on XNAT! Queueing for processing.")
                newly_ready.append(pid)
            else:
                still_waiting.append(pid)
        else:
            still_waiting.append(pid)

    with open(CENSUS_PATH, "w") as f:
        json.dump(census, f, indent=2)
    return newly_ready, still_waiting


def main():
    parser = argparse.ArgumentParser(description="Batch Orchestrator for Prostata_bimodal_PIPELINE")
    parser.add_argument("--patients", nargs="*", help="Optional subset of PatIDs to process (default: all 133)")
    parser.add_argument("--gpus", nargs="*", type=int, default=[1], help="GPU IDs to use (default: [1] to leave GPU 0 free)")
    parser.add_argument("--no-poll", action="store_true", help="Exit once current ready queue is empty instead of polling waiting cases")
    args = parser.parse_args()

    os.makedirs(OUT_BASE_DIR, exist_ok=True)
    with open(CENSUS_PATH, "r") as f:
        census = json.load(f)

    xnat = XnatClient()

    # Load previous progress if restarting
    completed = set()
    failed = {}
    if os.path.isfile(PROGRESS_JSON):
        try:
            with open(PROGRESS_JSON, "r") as f:
                prog = json.load(f)
                completed = set(prog.get("completed", []))
        except Exception:
            pass

    all_pids = sorted(census.keys(), key=lambda x: int(x.replace("Pat", "")))
    if args.patients:
        all_pids = [p for p in all_pids if p in args.patients]
        for p in args.patients:
            completed.discard(p)

    ready_queue = []
    waiting_list = []
    for pid in all_pids:
        if pid in completed:
            continue
        info = census[pid]
        if info.get("has_anat") and info.get("has_ln"):
            ready_queue.append(pid)
        else:
            waiting_list.append(pid)

    gpu_ids = args.gpus if args.gpus else [1]
    print("=" * 80)
    print(f" PROSTATA BIMODAL BATCH ORCHESTRATOR (GPUs: {gpu_ids})")
    print(f" Total Target Cases: {len(all_pids)} | Ready Now: {len(ready_queue)} | Already Completed: {len(completed)} | Waiting on Server: {len(waiting_list)}")
    print("=" * 80)

    # Start persistent Julia GPU workers only on requested GPUs
    worker_procs = {}
    q_lock = threading.Lock()
    stop_flag = threading.Event()
    active_workers = {gid: False for gid in gpu_ids}

    for gid in gpu_ids:
        q_dir = f"/tmp/ln_worker_gpu{gid}"
        log_p = f"/workspaces/MedEye3d.jl/data/ln_worker_gpu{gid}.log"
        proc, log_f = start_julia_worker_daemon(gid, q_dir, log_p)
        worker_procs[gid] = {"proc": proc, "log": log_f, "q_dir": q_dir}

    def ensure_daemon_alive(gpu_id):
        wp = worker_procs[gpu_id]
        if wp["proc"].poll() is not None:
            print(f"[GPU {gpu_id}] Respawning Julia GPU worker daemon...")
            try:
                wp["log"].close()
            except Exception:
                pass
            proc, log_f = start_julia_worker_daemon(
                gpu_id, wp["q_dir"], f"/workspaces/MedEye3d.jl/data/ln_worker_gpu{gpu_id}.log"
            )
            wp["proc"] = proc
            wp["log"] = log_f
            ready_marker = os.path.join(wp["q_dir"], "worker_ready")
            t_w = time.time()
            while not os.path.isfile(ready_marker) and time.time() - t_w < 180:
                time.sleep(1.0)

    def gpu_worker_loop(gpu_id, queue_dir):
        # Wait for Julia daemon to finish precompilation and write worker_ready
        ready_marker = os.path.join(queue_dir, "worker_ready")
        t_w = time.time()
        while not os.path.isfile(ready_marker) and time.time() - t_w < 180:
            time.sleep(1.0)
        print(f"[GPU {gpu_id}] Julia worker daemon ready!")

        while not stop_flag.is_set():
            pid = None
            with q_lock:
                if ready_queue:
                    pid = ready_queue.pop(0)
                    active_workers[gpu_id] = True
                else:
                    active_workers[gpu_id] = False
            if pid is None:
                time.sleep(2.0)
                continue

            ensure_daemon_alive(gpu_id)
            pat_info = census[pid]
            success = False
            for attempt in range(2):
                try:
                    success = process_single_patient(
                        pat_info, xnat, gpu_id, queue_dir,
                        force_reprocess=(attempt == 0)
                    )
                    if success:
                        break
                except Exception as e:
                    import traceback
                    err_str = f"{e}\n{traceback.format_exc()}"
                    print(f"[GPU {gpu_id}] ❌ Attempt {attempt+1} failed on {pid}: {e}")
                    with q_lock:
                        failed[pid] = str(e)

            with q_lock:
                active_workers[gpu_id] = False
                if success:
                    completed.add(pid)
                    failed.pop(pid, None)
                save_progress(completed, failed, waiting_list)
                done_in_target = sum(1 for p in all_pids if p in completed)
                print(f"[PROGRESS] Completed: {done_in_target}/{len(all_pids)} (Total: {len(completed)}) | Remaining in queue: {len(ready_queue)} | Waiting on server: {len(waiting_list)} | Failed: {len(failed)}")

    threads = []
    for gid in gpu_ids:
        t = threading.Thread(target=gpu_worker_loop, args=(gid, worker_procs[gid]["q_dir"]), daemon=True)
        t.start()
        threads.append(t)

    try:
        while True:
            with q_lock:
                rem_q = len(ready_queue)
                rem_w = len(waiting_list)
                done_in_target = sum(1 for p in all_pids if p in completed)
                fail_in_target = sum(1 for p in all_pids if p in failed)
                any_active = any(active_workers.values())

            # Check if workers are idle and ready_queue is empty
            if rem_q == 0 and not any_active and (done_in_target + fail_in_target + rem_w >= len(all_pids)):
                if rem_w == 0 or args.no_poll:
                    break
                print(f"[ORCHESTRATOR] All ready cases processed ({done_in_target} done, {fail_in_target} failed). Polling {rem_w} server-waiting cases ({waiting_list}) in 120s...")
                time.sleep(120)
                with q_lock:
                    new_r, still_w = refresh_waiting_cases(xnat, census, waiting_list)
                    ready_queue.extend(new_r)
                    waiting_list[:] = still_w
                    save_progress(completed, failed, waiting_list)
            else:
                time.sleep(5.0)
    finally:
        stop_flag.set()
        for gid, wp in worker_procs.items():
            try:
                with open(os.path.join(wp["q_dir"], "shutdown.req.json"), "w") as sf:
                    json.dump({"command": "shutdown"}, sf)
            except Exception:
                pass
        time.sleep(2.0)
        for gid, wp in worker_procs.items():
            try:
                wp["proc"].terminate()
                wp["log"].close()
            except Exception:
                pass

    print("\n" + "=" * 80)
    print(f" BATCH ORCHESTRATOR FINISHED: {len(completed)}/{len(all_pids)} completed, {len(failed)} failed, {len(waiting_list)} waiting.")
    print("=" * 80)


if __name__ == "__main__":
    main()
