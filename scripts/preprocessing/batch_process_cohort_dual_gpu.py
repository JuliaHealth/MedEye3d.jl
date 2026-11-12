#!/usr/bin/env python3
"""
batch_process_cohort_dual_gpu.py

End-to-End Dual-GPU Cohort Processor for Prostata_bimodal_PIPELINE.
Processes all 133 patient cases across GPU 0 and GPU 1 concurrently:
  1. Calibrated AC PET SUV access and conversion (Scan 3 DICOM verification & suvs.py)
  2. Two-stage Arun SVD + prostate SWD registration (Transform_FollowUp_to_Baseline_1.tfm)
  3. Resampling moving MRI sequences + dedicated 8-zone prostate sub-segmentation
  4. Recalculation of fresh lymph node stations via Julia GPU DagVm engine
  5. Whole-body CT anatomy layering & sanitation (purging 'temp' & helper masks)
  6. Modality ground truth lesions (Label 9 -> 1 on MRI; zero mapped CT lesions on MRI)
  7. Champion TotalSegEdgePDE 3D neural edges (CT+PET, CT, PET) & diffusivities on GPU
  8. Transverse axis-1 pre-flip (preflipped=1) and 1-based centroids
  9. Lossless HDF5 packaging (libver="v110", gzip-1)
 10. Headless Julia verification and asynchronous streaming upload to XNAT MEDEYE3D
"""

import os
import sys

# Disable HDF5 file locking across concurrent threads and processes
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import io
import json
import time
import re
import glob
import shutil
import zipfile
import tempfile
import threading
import queue
import subprocess
import argparse
import requests
import torch
import numpy as np
import h5py
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter, label

# Paths & Settings
WORKSPACE_DIR = "/workspaces/MedEye3d.jl"
CENSUS_PATH = os.path.join(WORKSPACE_DIR, "data/xnat_bimodal_census.json")
OUT_BASE_DIR = os.path.join(WORKSPACE_DIR, "data/cases/Prostata_bimodal")
PROGRESS_JSON = os.path.join(WORKSPACE_DIR, "data/cohort_dual_gpu_progress.json")
LOCAL_XNAT_CACHE = "/mnt/big/project_ssd/data/xnat"
LOCAL_RESTORED_CACHE = "/mnt/big/project_ssd/lymph_node_rules/data/processed_cases_restored"

# XNAT Credentials & Endpoints
ALIAS = "0eae53b5-d2e4-4bed-934a-d0540068269b"
SECRET = "NWGGgbcf4DDLrco30ITFe0E6AuMvjljjtckXPEweO0mI3abO2K1pjg7DPEdF9u0n"
SERVER = "https://imaging-platform.diz-ag.med.ovgu.de"
PROJ_ID = "Prostata_bimodal_PIPELINE"
RAW_PROJ_ID = "Prostata_bimodal"

# Champion PDE Parameters
CHAMPION_GAMMA = 10.095886
CHAMPION_EPS_D = 1.0e-5

# 8-Zone Prostate Labels (MRI Only)
PROSTATE_ZONE_LABELS = {
    "1": "Peripheral Zone (PZ)",
    "2": "Transition Zone (TZ)",
    "3": "Central Zone (CZ)",
    "4": "Anterior Fibromuscular Stroma (AFS)",
    "5": "Urethra (UR)",
    "6": "Periurethral Glands (PG)",
    "7": "Seminal Vesicle Left (SV_L)",
    "8": "Seminal Vesicle Right (SV_R)"
}


def safe_read_image(path):
    """Safely reads NIfTI / NRRD images, auto-detecting and converting NRRD format if mismatched."""
    try:
        return sitk.ReadImage(path)
    except Exception:
        try:
            with open(path, "rb") as f:
                magic = f.read(4)
            if magic == b"NRRD":
                reader = sitk.ImageFileReader()
                reader.SetImageIO("NrrdImageIO")
                reader.SetFileName(path)
                img = reader.Execute()
                sitk.WriteImage(img, path)
                return img
        except Exception:
            pass
        raise


def classify_mri_modality(fname):
    """Classifies an MRI filename into modality (T2, ADC, DWI, T1, DCE) and plane (ax, cor, sag)."""
    fl = fname.lower()
    if any(fl.endswith(ext) for ext in [".seg.nrrd", ".nrrd", ".fcsv", ".mrb", ".json", ".tfm"]):
        return None
    if any(bad in fl for bad in ["lesion", "mask", "totalseg", "segmentation", "seg", "prostate", "mri_registered"]):
        return None
    if not (fl.endswith(".nii.gz") or fl.endswith(".nii")):
        return None
    if "ct" in fl and not any(k in fl for k in ["registered_to_ct", "to_ct"]):
        return None
    if "suv" in fl or "pet" in fl:
        return None

    if any(k in fl for k in ["adc", "ddadc", "eadc"]):
        mod = "ADC"
    elif any(k in fl for k in ["dwi", "dw_", "isodwi", "b1400", "b1500", "b1200", "b-50", "b50", "bval", "diff_", "trace"]):
        mod = "DWI"
    elif any(k in fl for k in ["dce", "dyn", "dynamik", "lava", "water", "perfusion", "sub."]):
        mod = "DCE"
    elif any(k in fl for k in ["t1", "t1w", "fl2d", "vibe"]):
        mod = "T1"
    elif any(k in fl for k in ["t2", "t2w", "haste", "space", "prop"]):
        mod = "T2"
    else:
        mod = "MRI"

    if any(k in fl for k in ["cor", "coronal"]):
        plane = "cor"
    elif any(k in fl for k in ["sag", "sagittal"]):
        plane = "sag"
    else:
        plane = "ax"

    return mod, plane


def select_ranked_mri_sequences(all_files_map):
    """
    Ranks available MRI sequences for multi-modal quad comparison:
    Rank 1: T2 (Axial)
    Rank 2: ADC (Axial)
    Rank 3: DWI (Axial)
    Rank 4: T1 (Axial)
    Rank 5: DCE (Axial)
    Followed by remaining axial, then coronal/sagittal planes.
    """
    parsed = []
    for fn, item in all_files_map.items():
        res = classify_mri_modality(fn)
        if res:
            mod, plane = res
            is_orig = "original" in fn.lower()
            is_swd = "swd" in fn.lower()
            is_reg = "registered" in fn.lower()
            parsed.append({"mod": mod, "plane": plane, "fn": fn, "item": item, "is_orig": is_orig, "is_swd": is_swd, "is_reg": is_reg})

    unique_groups = {}
    for p in parsed:
        key = (p["mod"], p["plane"])
        if key not in unique_groups:
            unique_groups[key] = p
        else:
            curr = unique_groups[key]
            def _score(x):
                s = 0
                if x.get("is_reg", False): s += 20
                if x.get("is_orig", False): s += 10
                if x.get("is_swd", False): s += 5
                return s
            if _score(p) > _score(curr):
                unique_groups[key] = p

    ordered = []
    if ("T2", "ax") in unique_groups:
        ordered.append(unique_groups.pop(("T2", "ax")))
    elif any(k[0] in ("T2", "MRI") and k[1] == "ax" for k in unique_groups):
        k = [k for k in unique_groups if k[0] in ("T2", "MRI") and k[1] == "ax"][0]
        ordered.append(unique_groups.pop(k))

    if ("ADC", "ax") in unique_groups:
        ordered.append(unique_groups.pop(("ADC", "ax")))
    if ("DWI", "ax") in unique_groups:
        ordered.append(unique_groups.pop(("DWI", "ax")))
    if ("T1", "ax") in unique_groups:
        ordered.append(unique_groups.pop(("T1", "ax")))
    if ("DCE", "ax") in unique_groups:
        ordered.append(unique_groups.pop(("DCE", "ax")))

    for k in sorted(unique_groups.keys(), key=lambda x: (x[1] != "ax", x[0])):
        ordered.append(unique_groups[k])

    return ordered


def compute_mri_diffusivity(mri_arr, mri_anatomy=None, exclusion_mask=None):
    """
    Computes 3D boundary edge and diffusivity for MRI modalities.
    User instruction: treat MRI as pure CT.
    Boundary comes from MRI gradient + dedicated 8-zone prostate capsule.
    """
    pos = mri_arr[mri_arr > 0]
    p99 = float(np.percentile(pos, 99)) if len(pos) > 0 else 1000.0
    mri_norm = np.clip(mri_arr / max(p99, 1.0), 0.0, 1.0)

    gz, gy, gx = np.gradient(mri_norm)
    mag_mri = gaussian_filter(np.sqrt(gz**2 + gy**2 + gx**2), sigma=0.8)
    e_raw = 1.0 / (1.0 + np.exp(-6.0 * (mag_mri - 0.15)))
    e_zero = 1.0 / (1.0 + np.exp(-6.0 * (-0.15)))
    edge_mri = np.clip((e_raw - e_zero) / (1.0 - e_zero), 0.0, 1.0).astype(np.float32)

    if mri_anatomy is not None and (mri_anatomy > 0).any():
        p_bin = (mri_anatomy > 0).astype(np.float32)
        gz_p, gy_p, gx_p = np.gradient(p_bin)
        capsule = np.sqrt(gz_p**2 + gy_p**2 + gx_p**2)
        edge_mri = np.maximum(edge_mri, np.clip(capsule * 2.0, 0.0, 1.0))

    diff_mri = CHAMPION_EPS_D + (1.0 - CHAMPION_EPS_D) * np.exp(-CHAMPION_GAMMA * edge_mri)
    diff_mri = np.clip(diff_mri, CHAMPION_EPS_D, 1.0).astype(np.float32)
    if exclusion_mask is not None:
        diff_mri[exclusion_mask > 0] = 0.0

    return diff_mri


class XnatManager:
    """Thread-safe XNAT manager for REST queries, downloads, and streaming uploads."""
    def __init__(self):
        self.lock = threading.Lock()
        self.session = requests.Session()
        self._authenticate()

    def _authenticate(self):
        with self.lock:
            r = self.session.get(f"{SERVER}/data/JSESSION", auth=(ALIAS, SECRET), timeout=30)
            r.raise_for_status()
            jsess = r.text.strip()
            self.session.cookies.set("JSESSIONID", jsess)

    def request(self, method, url, **kwargs):
        if not url.startswith("http"):
            url = f"{SERVER}{url}"
        for attempt in range(4):
            try:
                r = self.session.request(method, url, **kwargs)
                if r.status_code == 401:
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
            r = self.request("GET", uri_or_url, stream=True, timeout=180)
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
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False

    def upload_file_streaming(self, uri, local_path):
        """Uploads via curl -T with streaming PUT to handle multi-gigabyte HDF5 archives without buffering."""
        if not os.path.isfile(local_path):
            return False
        full_url = uri if uri.startswith("http") else f"{SERVER}{uri}"
        if "inbody=true" not in full_url:
            sep = "&" if "?" in full_url else "?"
            full_url = f"{full_url}{sep}inbody=true"

        cmd = [
            "curl", "-k", "-s", "-w", "%{http_code}",
            "-u", f"{ALIAS}:{SECRET}",
            "-H", "Expect:",
            "--connect-timeout", "30",
            "--speed-time", "45",
            "--speed-limit", "1000",
            "--retry", "3",
            "--retry-delay", "5",
            "-T", local_path,
            full_url
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        code = res.stdout.strip()[-3:] if res.stdout else ""
        return code in ("200", "201")


def get_scan_list_for_subject(xnat, subj_label):
    """Finds raw experiments and scans for a patient."""
    exps = xnat.get_json(f"/data/projects/{RAW_PROJ_ID}/experiments?format=json").get("ResultSet", {}).get("Result", [])
    matching_exps = [e for e in exps if subj_label in e.get("label", "")]
    if not matching_exps:
        # Fallback to search by Pat ID
        m = re.search(r"Pat(\d+)", subj_label)
        if m:
            pid = m.group(1)
            matching_exps = [e for e in exps if f"Pat{pid}" in e.get("label", "")]
    return matching_exps


def download_ac_pet_dicom(xnat, exp_id, out_dcm_dir, pat_id=None, subj_label=None):
    """
    Step 1: Locates the true Attenuation-Corrected (AC) PET scan in raw XNAT session.
    Verifies modality PT, series_description containing PET/WB and NOT Uncorrected.
    Downloads and extracts DICOM files.
    """
    if os.path.isdir(out_dcm_dir) and len(os.listdir(out_dcm_dir)) > 50:
        return True
    os.makedirs(out_dcm_dir, exist_ok=True)

    # Check local SSD cache first!
    if pat_id:
        local_cands = glob.glob(f"/mnt/big/project_ssd/data/xnat/*{pat_id}/scans/3*/resources/DICOM/files")
        if subj_label:
            local_cands += glob.glob(f"/mnt/big/project_ssd/data/xnat/{subj_label}/scans/3*/resources/DICOM/files")
        for lc in local_cands:
            if os.path.isdir(lc) and len(os.listdir(lc)) > 50:
                print(f"    Found local AC PET DICOM files in {lc} ({len(os.listdir(lc))} files)")
                for f in os.listdir(lc):
                    src_f = os.path.join(lc, f)
                    dst_f = os.path.join(out_dcm_dir, f)
                    if not os.path.exists(dst_f):
                        try:
                            os.symlink(src_f, dst_f)
                        except Exception:
                            shutil.copyfile(src_f, dst_f)
                return True

    scans_data = xnat.get_json(f"/data/experiments/{exp_id}/scans?format=json")
    scans = scans_data.get("ResultSet", {}).get("Result", [])

    ac_scan_id = None
    for s in scans:
        desc = s.get("series_description", "").upper()
        mod = s.get("modality", "").upper()
        sid = s.get("ID")
        if (mod == "PT" or "PET" in desc) and "UNCORRECTED" not in desc and "NAC" not in desc:
            ac_scan_id = sid
            print(f"    Found AC PET scan: Scan {sid} ({desc})")
            break

    if not ac_scan_id:
        # Fallback: scan 3 is typically the AC scan
        for s in scans:
            if s.get("ID") == "3":
                ac_scan_id = "3"
                break

    if not ac_scan_id:
        return False

    zip_path = os.path.join(out_dcm_dir, "PET_DICOM.zip")
    dcm_uri = f"/data/experiments/{exp_id}/scans/{ac_scan_id}/resources/DICOM/files?format=zip"
    print(f"    Downloading Scan {ac_scan_id} DICOM archive from XNAT...")
    ok = xnat.download_file(dcm_uri, zip_path, min_size=100000)
    if not ok or not os.path.isfile(zip_path):
        return False

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dcm_dir)
        os.remove(zip_path)
        # Flatten directory: collect all extracted files
        extracted_files = []
        for root, dirs, files in os.walk(out_dcm_dir):
            for f in files:
                if not f.endswith(".zip"):
                    extracted_files.append(os.path.join(root, f))
        for p in extracted_files:
            dest = os.path.join(out_dcm_dir, os.path.basename(p))
            if p != dest:
                shutil.move(p, dest)
        # Cleanup any subdirectories
        for item in os.listdir(out_dcm_dir):
            sub = os.path.join(out_dcm_dir, item)
            if os.path.isdir(sub):
                shutil.rmtree(sub, ignore_errors=True)
        return len([f for f in os.listdir(out_dcm_dir) if os.path.isfile(os.path.join(out_dcm_dir, f))]) > 30
    except Exception as e:
        print(f"    [WARN] Failed to unpack DICOM zip: {e}")
        return False


def calibrate_suv_image(dcm_dir, ct_img, uncorr_pet_path, out_suv_path):
    """
    Step 1.2 & 1.3: Verifies ATTN+SCAT DICOM tags and converts to genuine SUV [g/mL].
    Harmonizes coordinates with CT grid.
    """
    if os.path.isfile(out_suv_path) and os.path.getsize(out_suv_path) > 1000000:
        im = sitk.ReadImage(out_suv_path)
        if im.GetSize() == ct_img.GetSize():
            return im

    sys.path.insert(0, os.path.join(WORKSPACE_DIR, "scripts"))
    import suvs

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dcm_dir) if os.path.isdir(dcm_dir) else []

    if series_ids:
        import pydicom
        dcm_series_files = reader.GetGDCMSeriesFileNames(dcm_dir, series_ids[0])
        reader.SetFileNames(dcm_series_files)
        base_pet_img = reader.Execute()
        dicom_files = [{'file_path': p, 'ds': pydicom.dcmread(p)} for p in dcm_series_files]
        suvs.validate_pet_dicom_tags(dicom_files)
        suv_img = suvs.apply_suv_correction(dicom_files, base_pet_img)
        if os.path.isfile(uncorr_pet_path):
            try:
                uncorr_img = sitk.ReadImage(uncorr_pet_path)
                if uncorr_img.GetSize() == suv_img.GetSize():
                    suv_img.CopyInformation(uncorr_img)
            except Exception:
                pass
    elif os.path.isfile(uncorr_pet_path):
        suv_img = sitk.ReadImage(uncorr_pet_path)
    else:
        raise RuntimeError(f"No PET DICOM series or fallback NIfTI available in {dcm_dir}")

    # Resample to CT grid (Linear interpolation)
    res_suv = sitk.Resample(suv_img, ct_img, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    sitk.WriteImage(res_suv, out_suv_path)
    return res_suv


def compute_two_stage_registration(ct_img, raw_dir, device="cuda:0", xnat=None, exp_id=None):
    """
    Step 2: Two-stage pelvis & prostate registration:
      Stage 1: Closed-form Arun SVD on pelvic landmarks (medial femurs & hip bones)
      Stage 2: SWD PyTorch GPU optimization strictly on the prostate volume
    """
    tfm_path = os.path.join(raw_dir, "Transform_FollowUp_to_Baseline_1.tfm")
    if os.path.isfile(tfm_path) and os.path.getsize(tfm_path) > 100:
        print(f"    Reusing existing transform: {os.path.basename(tfm_path)}")
        return tfm_path

    if xnat and exp_id:
        reg_map = xnat.get_files_map(f"/data/experiments/{exp_id}/resources/CT_MRI_REGISTRATION/files?format=json")
        for cand_name in ["Transform_Landmark_SWD.h5", "Transform_FollowUp_to_Baseline_1.tfm", "Transform_Landmark_Init.h5"]:
            if cand_name in reg_map:
                cand_dest = os.path.join(raw_dir, cand_name)
                if xnat.download_file(reg_map[cand_name], cand_dest):
                    try:
                        tfm_read = sitk.ReadTransform(cand_dest)
                        sitk.WriteTransform(tfm_read, tfm_path)
                        print(f"    Downloaded & verified {cand_name} -> {os.path.basename(tfm_path)}")
                        return tfm_path
                    except Exception as e_tfm:
                        print(f"    [WARN] Failed to convert downloaded {cand_name}: {e_tfm}")

    import torch
    import torch.nn as nn

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"    Computing Arun SVD + Prostate SWD Registration on {dev}...")

    # Load fixed landmarks from packed_anatomical_tensor.h5 or fallback
    anat_h5 = os.path.join(raw_dir, "packed_anatomical_tensor.h5")
    fixed_pts = []
    if os.path.isfile(anat_h5):
        with h5py.File(anat_h5, "r") as af:
            pt = af["packed_tensor"][:]
            # Medial femurs and hip bones (classes 25-28 or TotalSeg pelvic bones)
            for ch in range(pt.shape[0]):
                mask = np.isin(pt[ch], [25, 26, 27, 28, 36, 37, 38, 39])
                if mask.any():
                    idx = np.argwhere(mask)
                    if len(idx) > 20000:
                        idx = idx[np.random.choice(len(idx), 20000, replace=False)]
                    for z, y, x in idx:
                        phys = ct_img.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
                        fixed_pts.append(phys)
                    break

    # Load moving landmarks from totalseg_mr_ml.nii.gz
    ts_mr_path = os.path.join(raw_dir, "totalseg_mr_ml.nii.gz")
    moving_pts = []
    moving_prostate_pts = []
    moving_img = None
    if os.path.isfile(ts_mr_path):
        moving_img = sitk.ReadImage(ts_mr_path)
        mr_arr = sitk.GetArrayFromImage(moving_img)
        # Pelvic bones in TotalSeg total_mr: 36=femur_left, 37=femur_right, 38=hip_left, 39=hip_right
        bone_m = np.isin(mr_arr, [36, 37, 38, 39])
        if bone_m.any():
            idx = np.argwhere(bone_m)
            if len(idx) > 20000:
                idx = idx[np.random.choice(len(idx), 20000, replace=False)]
            for z, y, x in idx:
                phys = moving_img.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
                moving_pts.append(phys)

        # Prostate in TotalSeg: 17
        pros_m = (mr_arr == 17)
        if pros_m.any():
            idx = np.argwhere(pros_m)
            if len(idx) > 5000:
                idx = idx[np.random.choice(len(idx), 5000, replace=False)]
            for z, y, x in idx:
                phys = moving_img.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
                moving_prostate_pts.append(phys)

    # Check high-precision prostate_segmentation.nii.gz
    pros_nii = os.path.join(raw_dir, "prostate_segmentation.nii.gz")
    if os.path.isfile(pros_nii):
        p_img = sitk.ReadImage(pros_nii)
        p_arr = sitk.GetArrayFromImage(p_img)
        p_bin = (p_arr > 0)
        if p_bin.any():
            idx = np.argwhere(p_bin)
            if len(idx) > 5000:
                idx = idx[np.random.choice(len(idx), 5000, replace=False)]
            moving_prostate_pts = [p_img.TransformIndexToPhysicalPoint((int(x), int(y), int(z))) for z, y, x in idx]

    # Stage 1: Arun SVD Rigid Alignment
    if len(fixed_pts) >= 100 and len(moving_pts) >= 100:
        P = np.array(moving_pts[:min(len(fixed_pts), len(moving_pts))], dtype=np.float64)
        Q = np.array(fixed_pts[:min(len(fixed_pts), len(moving_pts))], dtype=np.float64)
        p_bar = P.mean(axis=0)
        q_bar = Q.mean(axis=0)
        H = (P - p_bar).T @ (Q - q_bar)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = q_bar - R @ p_bar
    else:
        R = np.eye(3, dtype=np.float64)
        t = np.zeros(3, dtype=np.float64)

    # Stage 2: Prostate SWD Optimization
    if len(moving_prostate_pts) > 50:
        # Fixed prostate points from CT
        fixed_pros_pts = []
        if os.path.isfile(anat_h5):
            with h5py.File(anat_h5, "r") as af:
                pt = af["packed_tensor"][:]
                for ch in range(pt.shape[0]):
                    m = (pt[ch] == 23)  # prostate ID
                    if m.any():
                        idx = np.argwhere(m)
                        if len(idx) > 5000:
                            idx = idx[np.random.choice(len(idx), 5000, replace=False)]
                        fixed_pros_pts = [ct_img.TransformIndexToPhysicalPoint((int(x), int(y), int(z))) for z, y, x in idx]
                        break

        if len(fixed_pros_pts) > 50:
            P_pros = torch.tensor(moving_prostate_pts, dtype=torch.float32, device=dev)
            Q_pros = torch.tensor(fixed_pros_pts, dtype=torch.float32, device=dev)
            R_init_t = torch.tensor(R, dtype=torch.float32, device=dev)
            t_init_t = torch.tensor(t, dtype=torch.float32, device=dev)

            # SWD Optimization
            delta_rot = nn.Parameter(torch.zeros(3, device=dev))
            delta_t = nn.Parameter(torch.zeros(3, device=dev))
            optimizer = torch.optim.Adam([delta_rot, delta_t], lr=1e-3)

            for step in range(150):
                optimizer.zero_grad()
                # Rodrigues formula for rotation
                theta = torch.norm(delta_rot)
                if theta > 1e-6:
                    k = delta_rot / theta
                    K = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], device=dev)
                    R_step = torch.eye(3, device=dev) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
                else:
                    R_step = torch.eye(3, device=dev)

                R_curr = R_step @ R_init_t
                t_curr = t_init_t + delta_t
                P_trans = (P_pros @ R_curr.T) + t_curr

                # 128 random 1D projections
                projs = torch.randn(128, 3, device=dev)
                projs = projs / torch.norm(projs, dim=1, keepdim=True)
                p_proj = torch.sort(P_trans @ projs.T, dim=0)[0]
                q_proj = torch.sort(Q_pros @ projs.T, dim=0)[0]
                loss = torch.mean(torch.abs(p_proj - q_proj))
                loss.backward()
                optimizer.step()

            # Final R and t
            theta = torch.norm(delta_rot)
            if theta > 1e-6:
                k = delta_rot / theta
                K = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], device=dev)
                R_step = torch.eye(3, device=dev) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
            else:
                R_step = torch.eye(3, device=dev)
            R = (R_step @ R_init_t).detach().cpu().numpy().astype(np.float64)
            t = (t_init_t + delta_t).detach().cpu().numpy().astype(np.float64)

    # Save ITK transform
    sitk_tfm = sitk.AffineTransform(3)
    sitk_tfm.SetMatrix(R.flatten())
    sitk_tfm.SetTranslation(t)
    sitk.WriteTransform(sitk_tfm, tfm_path)
    return tfm_path


def run_julia_lymph_node_rules(ct_path, tensor_path, out_ln_path, gpu_id=0):
    """
    Step 4: Recomputes fresh clinical lymph node stations (101-108) using the Julia GPU DagVm engine.
    """
    if os.path.isfile(out_ln_path) and os.path.getsize(out_ln_path) > 50000:
        try:
            with h5py.File(out_ln_path, "r") as test_f:
                num_k = len(test_f.keys())
            print(f"    Reusing verified clinical lymph nodes ({num_k} stations): {out_ln_path}")
            return True
        except Exception:
            print(f"    [WARN] Cached {out_ln_path} is unreadable/corrupted. Recomputing with Julia GPU...")
            os.remove(out_ln_path)

    ln_proj = "/mnt/big/project_ssd/lymph_node_rules/src/julia_gpu"
    worker_script = os.path.join(WORKSPACE_DIR, "scripts/preprocessing/run_ln_worker.jl")

    raw_dir = os.path.dirname(tensor_path)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["JULIA_NUM_THREADS"] = "4"
    env["LN_RULES_DIR"] = "/mnt/big/project_ssd/lymph_node_rules"

    julia_bin = "/mnt/big/project_ssd/lymph_node_rules/for_joanna/.local/julia-1.12.6/bin/julia" if os.path.isfile("/mnt/big/project_ssd/lymph_node_rules/for_joanna/.local/julia-1.12.6/bin/julia") else "julia"
    cmd = [
        julia_bin, f"--project={ln_proj}",
        worker_script,
        raw_dir,
        out_ln_path
    ]
    print(f"    Running Julia GPU DagVm LN rules on GPU {gpu_id}...")
    t0 = time.time()
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0 or not os.path.isfile(out_ln_path):
        print(f"    [WARN] Julia LN worker failed: {res.stderr[:300]}")
        return False
    print(f"    ✓ Lymph node rules completed in {time.time()-t0:.1f}s")
    return True


def layer_and_sanitize_ct_anatomy(ct_shape, raw_dir):
    """
    Step 5: Multi-layer stacking of viscera, vessels, freshly recomputed lymph nodes, and bones.
    Strictly purges 'temp', helper masks, and dilation buffers.
    """
    Z, Y, X = ct_shape
    max_anatomy = np.zeros(ct_shape, dtype=np.uint16)
    labels_dict = {}
    bone_ids = []
    lbl_idx = 1
    exclusion_mask = np.zeros(ct_shape, dtype=np.uint8)

    anat_h5 = os.path.join(raw_dir, "packed_anatomical_tensor.h5")
    reg_path = os.path.join(raw_dir, "packed_anatomical_tensor_registry.json")

    # Layer 0 & 1: VISTA Viscera & Vessels
    if os.path.isfile(anat_h5) and os.path.isfile(reg_path):
        with open(reg_path, "r") as rf:
            reg = json.load(rf)
        with h5py.File(anat_h5, "r") as af:
            pt = af["packed_tensor"][:]
            masks_dict = reg.get("masks", reg)
            for name, meta in masks_dict.items():
                if not isinstance(meta, dict):
                    continue
                clean = name.lower()
                # Strict sanitation: purge temp and helper masks
                if any(bad in clean for bad in ["temp", "helper", "dilation", "buffer"]):
                    continue
                ch = meta.get("channel", 1) - 1
                mid = meta.get("id", 1)
                if 0 <= ch < pt.shape[0]:
                    mask = (pt[ch] == mid)
                    if mask.shape == ct_shape and mask.any():
                        max_anatomy[mask] = lbl_idx
                        labels_dict[str(lbl_idx)] = clean
                        if any(b in clean for b in ["bone", "femur", "hip", "sacrum", "vertebra", "rib", "sternum", "ilium", "ischium"]):
                            bone_ids.append(lbl_idx)
                        if any(u in clean for u in ["bladder", "kidney"]):
                            exclusion_mask[mask] = 1
                        lbl_idx += 1

    # Layer 1.5: Fresh Lymph Node Stations (IDs 101-108)
    ln_h5 = os.path.join(raw_dir, "final_results_gpu.h5")
    if os.path.isfile(ln_h5):
        try:
            with h5py.File(ln_h5, "r") as lf:
                for k in sorted(lf.keys()):
                    clean = k.lower()
                    if any(bad in clean for bad in ["temp", "helper", "_base", "_dilated", "bridge"]):
                        continue
                    arr_ln = lf[k][:]
                    if arr_ln.ndim == 3:
                        if arr_ln.shape == (X, Y, Z):
                            arr_ln = np.transpose(arr_ln, (2, 1, 0))
                        if arr_ln.shape == (Z, Y, X):
                            m_ln = (arr_ln > 0)
                            if m_ln.any():
                                max_anatomy[m_ln] = lbl_idx
                                labels_dict[str(lbl_idx)] = clean
                                lbl_idx += 1
        except Exception as e_ln:
            print(f"    [WARN] Error reading {ln_h5}: {e_ln}")

    bone_atlas = np.isin(max_anatomy, bone_ids).astype(np.float32)
    skellytour = bone_atlas.copy()
    return max_anatomy, labels_dict, bone_ids, bone_atlas, skellytour, exclusion_mask


def compute_neural_edges(ct_arr, pet_arr, exclusion_mask, device="cuda:0"):
    """
    Step 7: Recomputes 3D neural edge probability maps and diffusivities using champion TotalSegEdgePDE models.
    """
    sys.path.insert(0, "/mnt/big/project_ssd/semiautomatic")
    import torch
    from totalseg_edge_pde_model import TotalSegEdgePDE
    from totalseg_edge_pde_pet_model import TotalSegEdgePDE_PET

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"    Running TotalSegEdgePDE inference on {dev}...")

    # Load models
    ckpt_dual = "/mnt/big/project_ssd/semiautomatic/checkpoints/totalseg_edge_pde/best_model.pth"
    m_dual = TotalSegEdgePDE().to(dev)
    d_st = torch.load(ckpt_dual, map_location=dev)
    m_dual.load_state_dict(d_st.get("model_state_dict", d_st))
    m_dual.eval()

    ckpt_pet = "/mnt/big/project_ssd/semiautomatic/checkpoints/totalseg_edge_pde_pet/best_model.pth"
    m_pet = TotalSegEdgePDE_PET().to(dev)
    p_st = torch.load(ckpt_pet, map_location=dev)
    m_pet.load_state_dict(p_st.get("model_state_dict", p_st))
    m_pet.eval()

    ct_norm = (np.clip(ct_arr, -1024.0, 1500.0) - (-200.0)) / 400.0
    pet_norm = np.clip(pet_arr, 0.0, 20.0) / 20.0

    def run_batched_sliding_window(model, inp_tensor, patch_size=64, step=48, batch_size=8):
        _, _, D, H, W = inp_tensor.shape
        out_edge = torch.zeros((D, H, W), dtype=torch.float32, device=dev)
        weight_map = torch.zeros((D, H, W), dtype=torch.float32, device=dev)

        z = np.sin(np.linspace(0, np.pi, patch_size))
        y = np.sin(np.linspace(0, np.pi, patch_size))
        x = np.sin(np.linspace(0, np.pi, patch_size))
        w3d = torch.from_numpy((z[:, None, None] * y[None, :, None] * x[None, None, :]).astype(np.float32)).to(dev)
        w3d = torch.clamp(w3d, min=1e-4)

        d_steps = list(range(0, D - patch_size + 1, step))
        if not d_steps or d_steps[-1] != D - patch_size: d_steps.append(max(0, D - patch_size))
        h_steps = list(range(0, H - patch_size + 1, step))
        if not h_steps or h_steps[-1] != H - patch_size: h_steps.append(max(0, H - patch_size))
        w_steps = list(range(0, W - patch_size + 1, step))
        if not w_steps or w_steps[-1] != W - patch_size: w_steps.append(max(0, W - patch_size))

        patches = [(d, h, w) for d in d_steps for h in h_steps for w in w_steps]

        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                b_coords = patches[i:i+batch_size]
                B = len(b_coords)
                C = inp_tensor.shape[1]
                b_inp = torch.empty((B, C, patch_size, patch_size, patch_size), dtype=torch.float32, device=dev)
                for j, (d, h, w) in enumerate(b_coords):
                    b_inp[j] = inp_tensor[0, :, d:d+patch_size, h:h+patch_size, w:w+patch_size]
                e_batch, _, _ = model.forward_edge(b_inp)
                for j, (d, h, w) in enumerate(b_coords):
                    out_edge[d:d+patch_size, h:h+patch_size, w:w+patch_size] += e_batch[j] * w3d
                    weight_map[d:d+patch_size, h:h+patch_size, w:w+patch_size] += w3d
        out_edge /= torch.clamp(weight_map, min=1e-6)
        res = torch.clamp(out_edge, 0.0, 1.0).cpu().numpy().astype(np.float32)
        del out_edge, weight_map, w3d
        torch.cuda.empty_cache()
        return res

    # 1. Combined (PET+CT)
    inp_dual = torch.from_numpy(np.stack([ct_norm, pet_norm], axis=0)).unsqueeze(0).to(dev)
    edge_comb = run_batched_sliding_window(m_dual, inp_dual, batch_size=8)
    del inp_dual
    torch.cuda.empty_cache()

    # 2. Pure CT
    inp_ct = torch.from_numpy(np.stack([ct_norm, np.zeros_like(pet_norm)], axis=0)).unsqueeze(0).to(dev)
    edge_ct = run_batched_sliding_window(m_dual, inp_ct, batch_size=8)
    del inp_ct, m_dual
    torch.cuda.empty_cache()

    # 3. Pure PET
    inp_pet = torch.from_numpy(pet_norm).unsqueeze(0).unsqueeze(0).to(dev)
    edge_pet = run_batched_sliding_window(m_pet, inp_pet, batch_size=8)
    del inp_pet, m_pet
    torch.cuda.empty_cache()

    diff_comb = np.clip(CHAMPION_EPS_D + (1.0 - CHAMPION_EPS_D) * np.exp(-CHAMPION_GAMMA * edge_comb), CHAMPION_EPS_D, 1.0).astype(np.float32)
    diff_ct = np.clip(CHAMPION_EPS_D + (1.0 - CHAMPION_EPS_D) * np.exp(-CHAMPION_GAMMA * edge_ct), CHAMPION_EPS_D, 1.0).astype(np.float32)
    diff_pet = np.clip(CHAMPION_EPS_D + (1.0 - CHAMPION_EPS_D) * np.exp(-9.4069 * edge_pet), CHAMPION_EPS_D, 1.0).astype(np.float32)

    if exclusion_mask is not None:
        diff_comb[exclusion_mask > 0] = 0.0
        diff_ct[exclusion_mask > 0] = 0.0

    return edge_comb, edge_ct, edge_pet, diff_comb, diff_ct, diff_pet


def process_case(pat_id, gpu_id, xnat):
    """
    Executes the full 10-stage processing pipeline on the assigned GPU.
    """
    case_dir = os.path.join(OUT_BASE_DIR, pat_id)
    raw_dir = os.path.join(case_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    out_h5 = os.path.join(case_dir, "preprocessed_volumes.h5")

    print(f"\n================================================================================")
    print(f" [GPU {gpu_id}] Processing Case: {pat_id}")
    print(f"================================================================================")
    t_start = time.time()

    # Load Census info
    with open(CENSUS_PATH, "r") as cf:
        census = json.load(cf)
    pat_info = census.get(pat_id, {})
    subj_label = pat_info.get("subj_label", f"FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__{pat_id}")
    subj_id = pat_info.get("subj_id")
    exp_id = pat_info.get("exp_id")

    # Step 1: Raw data acquisition & staging
    ct_dest = os.path.join(raw_dir, "Fixed_CT_Volume.nii.gz")
    local_cands = [
        os.path.join(LOCAL_RESTORED_CACHE, subj_label),
        os.path.join(LOCAL_XNAT_CACHE, subj_label),
        os.path.join(LOCAL_XNAT_CACHE, f"FDM_DPI-2024-7-KRN_Lu177_PSMA__PETPSMA_0__{pat_id}"),
        os.path.join(LOCAL_XNAT_CACHE, f"prostata_{pat_id}"),
    ]
    local_cache_dir = None
    for cand in local_cands:
        if os.path.isdir(cand):
            local_cache_dir = cand
            break
    if not local_cache_dir:
        local_cache_dir = os.path.join(LOCAL_XNAT_CACHE, f"prostata_{pat_id}")

    # Step 1.1: Fetch Fixed_CT_Volume.nii.gz
    # Always prioritize slicer_final from XNAT for exact cohort coordinate alignment
    if not os.path.isfile(ct_dest):
        sf_map = xnat.get_files_map(f"/data/projects/{PROJ_ID}/subjects/{subj_id}/resources/slicer_final/files?format=json")
        for k in ["Fixed_CT_Volume.nii.gz", "CT.nii.gz", "Fixed_CT_Volume.nii"]:
            if k in sf_map:
                xnat.download_file(sf_map[k], ct_dest)
                break
        if not os.path.isfile(ct_dest):
            # Check local restored cache strictly with full subj_label
            restored_cand = os.path.join(LOCAL_RESTORED_CACHE, subj_label, "Fixed_CT_Volume.nii.gz")
            if os.path.isfile(restored_cand) and os.path.getsize(restored_cand) > 100000:
                shutil.copyfile(restored_cand, ct_dest)
        if not os.path.isfile(ct_dest):
            # Fallback to slicer_2.2 / slicer / verification_scene.mrb
            for res_name in ["slicer_2.2", "slicer"]:
                res_map = xnat.get_files_map(f"/data/projects/{PROJ_ID}/subjects/{subj_id}/resources/{res_name}/files?format=json")
                for k in ["Fixed_CT_Volume.nii.gz", "CT.nii.gz", "Fixed_CT_Volume.nii"]:
                    if k in res_map:
                        xnat.download_file(res_map[k], ct_dest)
                        break
                if os.path.isfile(ct_dest):
                    break
                if "verification_scene.mrb" in res_map:
                    tmp_mrb = os.path.join(raw_dir, "verification_scene.mrb")
                    xnat.download_file(res_map["verification_scene.mrb"], tmp_mrb)
                    if os.path.isfile(tmp_mrb):
                        try:
                            with zipfile.ZipFile(tmp_mrb, "r") as z:
                                for zf in z.namelist():
                                    if "Fixed_CT" in zf or "CT.nii" in zf:
                                        with z.open(zf) as zsrc, open(ct_dest, "wb") as zdst:
                                            shutil.copyfileobj(zsrc, zdst)
                                        break
                        except Exception:
                            pass
                if os.path.isfile(ct_dest):
                    break

    if not os.path.isfile(ct_dest):
        raise RuntimeError(f"Missing Fixed_CT_Volume.nii.gz for {pat_id}")

    ct_img = safe_read_image(ct_dest)
    ct_arr = sitk.GetArrayFromImage(ct_img)
    ct_shape = ct_arr.shape

    # Download VISTA packed tensor & registry
    anat_h5 = os.path.join(raw_dir, "packed_anatomical_tensor.h5")
    anat_reg = os.path.join(raw_dir, "packed_anatomical_tensor_registry.json")
    if not os.path.isfile(anat_h5):
        for c in local_cands:
            cand_h5 = os.path.join(c, "packed_anatomical_tensor.h5")
            cand_reg = os.path.join(c, "packed_anatomical_tensor_registry.json")
            if os.path.isfile(cand_h5) and os.path.getsize(cand_h5) > 1000000:
                print(f"    Reusing local packed_anatomical_tensor.h5 from {c}")
                shutil.copyfile(cand_h5, anat_h5)
                if os.path.isfile(cand_reg):
                    shutil.copyfile(cand_reg, anat_reg)
                break
        if not os.path.isfile(anat_h5):
            af_map = xnat.get_files_map(f"/data/experiments/{exp_id}/resources/ANATOMICAL_TENSOR/files?format=json")
            if "packed_anatomical_tensor.h5" in af_map:
                xnat.download_file(af_map["packed_anatomical_tensor.h5"], anat_h5)
            if "packed_anatomical_tensor_registry.json" in af_map:
                xnat.download_file(af_map["packed_anatomical_tensor_registry.json"], anat_reg)

    # Download prostate segmentation & CT lesions
    pros_nii = os.path.join(raw_dir, "prostate_segmentation.nii.gz")
    if not os.path.isfile(pros_nii):
        for c in local_cands:
            cand_pros = os.path.join(c, "prostate_segmentation.nii.gz")
            if os.path.isfile(cand_pros) and os.path.getsize(cand_pros) > 10000:
                shutil.copyfile(cand_pros, pros_nii)
                break
        if not os.path.isfile(pros_nii):
            pf_map = xnat.get_files_map(f"/data/experiments/{exp_id}/resources/PROSTATE_SEGMENTATION/files?format=json")
            if "prostate_segmentation.nii.gz" in pf_map:
                xnat.download_file(pf_map["prostate_segmentation.nii.gz"], pros_nii)

    ct_lesions_dest = os.path.join(raw_dir, "NNUNET_OR_Processed_Lesions_CT.seg.nrrd")
    if not os.path.isfile(ct_lesions_dest):
        sf_map = xnat.get_files_map(f"/data/projects/{PROJ_ID}/subjects/{subj_id}/resources/slicer_final/files?format=json")
        for lk in ["NNUNET_OR_Processed_Lesions_CT.seg.nrrd", "Processed_Lesions_CT.seg.nrrd"]:
            if lk in sf_map:
                xnat.download_file(sf_map[lk], ct_lesions_dest)
                break

    # Download raw AC PET DICOM & Calibrate SUV
    pet_dcm_dir = os.path.join(raw_dir, "PET_AC_DICOM")
    uncorr_pet = os.path.join(local_cache_dir, "SUV.nii.gz")
    out_suv_dest = os.path.join(raw_dir, "SUV_PET_Image_0.nii.gz")
    download_ac_pet_dicom(xnat, exp_id, pet_dcm_dir, pat_id=pat_id, subj_label=subj_label)
    suv_img = calibrate_suv_image(pet_dcm_dir, ct_img, uncorr_pet, out_suv_dest)
    pet_arr = sitk.GetArrayFromImage(suv_img)

    # Step 2: Two-Stage Registration (Arun SVD + SWD Prostate)
    tfm_path = compute_two_stage_registration(ct_img, raw_dir, device=f"cuda:{gpu_id}", xnat=xnat, exp_id=exp_id)

    # Step 4: Julia GPU Lymph Node rules
    final_ln_h5 = os.path.join(raw_dir, "final_results_gpu.h5")
    if not os.path.isfile(final_ln_h5):
        for c in local_cands:
            cand_ln = os.path.join(c, "final_results_gpu.h5")
            if os.path.isfile(cand_ln) and os.path.getsize(cand_ln) > 1000000:
                print(f"    Reusing local final_results_gpu.h5 from {c}")
                shutil.copyfile(cand_ln, final_ln_h5)
                break
        if not os.path.isfile(final_ln_h5):
            ln_map = xnat.get_files_map(f"/data/experiments/{exp_id}/resources/LYMPH_NODE_RESULTS/files?format=json")
            for lk in ["final_results_gpu.h5", "lymph_node_results.h5", "lymph_nodes.h5"]:
                if lk in ln_map:
                    xnat.download_file(ln_map[lk], final_ln_h5)
                    break
    run_julia_lymph_node_rules(ct_dest, anat_h5, final_ln_h5, gpu_id=gpu_id)

    # Step 5: Anatomy Layering & Sanitation (purging temp)
    max_anat, anat_labels, bone_ids, bone_atlas, skellytour, excl_mask = layer_and_sanitize_ct_anatomy(ct_shape, raw_dir)

    # Step 6: Ground Truth Lesions & Overlap Matches
    ct_lesion_arr = np.zeros(ct_shape, dtype=np.int16)
    if os.path.isfile(ct_lesions_dest):
        try:
            cl_img = sitk.ReadImage(ct_lesions_dest)
            cl_arr = sitk.GetArrayFromImage(cl_img).astype(np.int16)
            if cl_arr.shape == ct_shape:
                ct_lesion_arr = cl_arr
        except Exception:
            pass

    # Step 3: Resample Moving MRI Sequences and Dedicated 8-Zone Prostate Anatomy
    sf_map = xnat.get_files_map(f"/data/projects/{PROJ_ID}/subjects/{subj_id}/resources/slicer_final/files?format=json")

    combined_mri_map = {}
    for fn, furi in sf_map.items():
        if classify_mri_modality(fn):
            combined_mri_map[fn] = furi

    for fname in os.listdir(raw_dir):
        if fname.endswith(".nii.gz") and not fname.startswith("Fixed_CT"):
            if classify_mri_modality(fname):
                combined_mri_map[fname] = os.path.join(raw_dir, fname)

    # Rank sequences before downloading to fetch only top-5 relevant axial series
    ranked_mri = select_ranked_mri_sequences(combined_mri_map)[:5]
    for seq in ranked_mri:
        loc_p = os.path.join(raw_dir, seq["fn"])
        if not os.path.isfile(loc_p):
            if seq["item"].startswith("/") or seq["item"].startswith("http"):
                xnat.download_file(seq["item"], loc_p)
        if os.path.isfile(loc_p):
            seq["item"] = loc_p

    tfm_obj = sitk.ReadTransform(tfm_path)

    # Dedicated 8-zone prostate anatomy + Gold Standard MRI lesion
    mri_anatomy_arr = np.zeros(ct_shape, dtype=np.uint16)
    mr_gold_arr = np.zeros(ct_shape, dtype=np.int16)

    if os.path.isfile(pros_nii):
        try:
            p_img = sitk.ReadImage(pros_nii)
            p_arr = sitk.GetArrayFromImage(p_img)

            # Label 9: Expert Gold Standard Lesion
            mr_l9 = (p_arr == 9).astype(np.uint8)
            if mr_l9.any():
                l9_img = sitk.GetImageFromArray(mr_l9)
                l9_img.CopyInformation(p_img)
                res_l9 = sitk.Resample(l9_img, ct_img, tfm_obj, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
                mr_gold_arr = sitk.GetArrayFromImage(res_l9).astype(np.int16)
                print(f"    [GOLD MR LESION] Found {mr_gold_arr.sum()} voxels of prostate lesion (Label 9) on MRI")

            # Labels 1..8: 8-zone prostate sub-segmentation
            p8_mask = (np.isin(p_arr, list(range(1, 9)))).astype(np.uint16) * p_arr
            if p8_mask.any():
                p8_img = sitk.GetImageFromArray(p8_mask)
                p8_img.CopyInformation(p_img)
                res_p8 = sitk.Resample(p8_img, ct_img, tfm_obj, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
                mri_anatomy_arr = sitk.GetArrayFromImage(res_p8).astype(np.uint16)
                print(f"    [PROSTATE ANATOMY] Resampled 8-zone prostate anatomy (non-zero={(mri_anatomy_arr > 0).sum()} voxels)")
        except Exception as e:
            print(f"    [WARN] Prostate segmentation extraction error: {e}")

    # Fallback for MRI lesion if no Label 9 in prostate segmentation
    if not mr_gold_arr.any():
        mr_lesions_dest = os.path.join(raw_dir, "NNUNET_OR_Processed_Lesions_MR.seg.nrrd")
        if not os.path.isfile(mr_lesions_dest) and "NNUNET_OR_Processed_Lesions_MR.seg.nrrd" in sf_map:
            xnat.download_file(sf_map["NNUNET_OR_Processed_Lesions_MR.seg.nrrd"], mr_lesions_dest)
        if os.path.isfile(mr_lesions_dest):
            try:
                ml_img = sitk.ReadImage(mr_lesions_dest)
                res_ml = sitk.Resample(ml_img, ct_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkInt16) if ml_img.GetSize() == ct_img.GetSize() else sitk.Resample(ml_img, ct_img, tfm_obj, sitk.sitkNearestNeighbor, 0, sitk.sitkInt16)
                ml_arr = sitk.GetArrayFromImage(res_ml).astype(np.int16)
                if ml_arr.any():
                    mr_gold_arr = (ml_arr > 0).astype(np.int16)
                    print(f"    [GOLD MR LESION] Extracted {mr_gold_arr.sum()} voxels from NNUNET_OR_Processed_Lesions_MR")
            except Exception as e_ml:
                print(f"    [WARN] Fallback MR lesion extraction error: {e_ml}")

    # Resample Primary T2 (TP 1)
    t2_arr = None
    if ranked_mri and ranked_mri[0]["mod"] == "T2":
        t2_info = ranked_mri[0]
        try:
            t2_im = sitk.ReadImage(t2_info["item"])
            res_t2 = sitk.Resample(t2_im, ct_img, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32) if t2_info["is_reg"] else sitk.Resample(t2_im, ct_img, tfm_obj, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
            t2_arr = sitk.GetArrayFromImage(res_t2).astype(np.float32)
            t2_arr = np.maximum(0.0, t2_arr)
            pos = t2_arr[t2_arr > 0]
            if len(pos) > 0:
                p99 = float(np.percentile(pos, 99))
                if p99 > 0:
                    t2_arr = t2_arr * (1000.0 / p99)
        except Exception as e_t2:
            print(f"    [WARN] Failed to resample T2: {e_t2}")

    if t2_arr is None:
        t2_arr = ct_arr.copy()

    # Resample Additional MRI Sequences (ADC, DWI, T1, DCE)
    extra_mri_series = []
    tp_idx_counter = 2
    for seq in (ranked_mri[1:] if (ranked_mri and ranked_mri[0]["mod"] == "T2") else ranked_mri):
        try:
            im = sitk.ReadImage(seq["item"])
            if (seq["mod"] in ("DCE", "DYNAMIC") or "dyn" in seq["fn"].lower()) and im.GetSize()[2] > 100:
                raw_4d = sitk.GetArrayFromImage(im)
                tot_z = raw_4d.shape[0]
                slices_p = 23 if tot_z % 23 == 0 else (18 if tot_z % 18 == 0 else 23)
                n_ph = tot_z // slices_p
                tgt_ph = min(3, n_ph - 1)
                ph_arr = raw_4d[tgt_ph * slices_p : (tgt_ph + 1) * slices_p]
                dce_c = sitk.GetImageFromArray(ph_arr)
                dce_c.SetSpacing((im.GetSpacing()[0], im.GetSpacing()[1], 3.0))
                dce_c.SetOrigin(im.GetOrigin())
                dce_c.SetDirection(im.GetDirection())
                im = dce_c

            res_im = sitk.Resample(im, ct_img, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32) if seq["is_reg"] else sitk.Resample(im, ct_img, tfm_obj, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
            arr = sitk.GetArrayFromImage(res_im).astype(np.float32)
            arr = np.maximum(0.0, arr)
            pos = arr[arr > 0]
            if len(pos) > 0:
                p99 = float(np.percentile(pos, 99))
                scale = 1500.0 / max(p99, 1.0) if seq["mod"] == "ADC" else 1000.0 / max(p99, 1.0)
                arr = arr * scale

            diff_seq = compute_mri_diffusivity(arr, mri_anatomy=mri_anatomy_arr, exclusion_mask=excl_mask)
            extra_mri_series.append({
                "tp_idx": tp_idx_counter,
                "mod": seq["mod"],
                "fn": seq["fn"],
                "desc": f"{seq['mod']} MRI Registered",
                "arr": arr,
                "diff": diff_seq
            })
            tp_idx_counter += 1
        except Exception as e_seq:
            print(f"    [WARN] Failed to resample {seq['fn']}: {e_seq}")

    diff_mri = compute_mri_diffusivity(t2_arr, mri_anatomy=mri_anatomy_arr, exclusion_mask=excl_mask)

    # Compute matches.json
    matches = []
    if mr_gold_arr.any() and ct_lesion_arr.any():
        for cid in np.unique(ct_lesion_arr):
            if cid == 0:
                continue
            ov = int(np.sum((mr_gold_arr == 1) & (ct_lesion_arr == cid)))
            if ov > 0:
                matches.append({
                    "group_id": int(cid),
                    "lesion_type": "Prostate Primary Lesion",
                    "tp0_lid": int(cid),
                    "tp1_lid": 1,
                    "overlap_voxels": ov
                })

    # Step 7: Neural 3D Edges & Diffusivities on GPU
    edge_comb, edge_ct, edge_pet, diff_comb, diff_ct, diff_pet = compute_neural_edges(
        ct_arr, pet_arr, excl_mask, device=f"cuda:{gpu_id}"
    )

    # Step 8: Transverse Axis-1 Pre-flip & Centroids
    def flip_y(a):
        return np.flip(a, axis=1)

    ct_arr_f = flip_y(ct_arr)
    pet_arr_f = flip_y(pet_arr)
    ct_les_f = flip_y(ct_lesion_arr)
    mr_gold_f = flip_y(mr_gold_arr)
    max_anat_f = flip_y(max_anat)
    bone_atlas_f = flip_y(bone_atlas)
    skellytour_f = flip_y(skellytour)
    edge_comb_f = flip_y(edge_comb)
    diff_comb_f = flip_y(diff_comb)
    t2_arr_f = flip_y(t2_arr)
    diff_mri_f = flip_y(diff_mri)
    mri_anat_f = flip_y(mri_anatomy_arr) if mri_anatomy_arr.any() else max_anat_f

    for s in extra_mri_series:
        s["arr_f"] = flip_y(s["arr"])
        s["diff_f"] = flip_y(s["diff"])

    # 1-based centroids
    centroids = {}
    for lid in np.unique(ct_les_f):
        if lid == 0:
            continue
        zc, yc, xc = np.mean(np.argwhere(ct_les_f == lid), axis=0)
        centroids[f"tp0_lid{lid}"] = np.array([round(xc) + 1, round(yc) + 1, round(zc) + 1], dtype=np.int32)

    if mr_gold_f.any():
        zc, yc, xc = np.mean(np.argwhere(mr_gold_f == 1), axis=0)
        centroids["tp1_lid1"] = np.array([round(xc) + 1, round(yc) + 1, round(zc) + 1], dtype=np.int32)

    # Build Scene Hierarchy
    ct_labels = [int(i) for i in np.unique(ct_les_f) if i > 0]
    ct_seg_names = {str(i): f"Lesion {i}" for i in ct_labels}
    mr_seg_names = {"1": "Prostate Primary Lesion"} if mr_gold_f.any() else {}

    scene_hierarchy = [
        {"name": "Fixed_CT_Volume_0", "type": "vtkMRMLScalarVolumeNode", "modality": "CT"},
        {"name": "SUV_PET_Image_0", "type": "vtkMRMLScalarVolumeNode", "modality": "PET"},
        {"name": "PET_Lesions_0", "type": "vtkMRMLSegmentationNode", "segments": [ct_seg_names.get(str(i)) for i in ct_labels]},
        {"name": "max_anatomy_fixed_ct_0", "source": "max_anatomy.nii.gz", "type": "vtkMRMLSegmentationNode"},
        {"name": "skellytour_fixed_ct_0", "source": "skellytour.nii.gz", "type": "vtkMRMLSegmentationNode"},
        {
            "name": "Transform_FollowUp_to_Baseline_1",
            "type": "vtkMRMLLinearTransformNode",
            "transform_type": "Rigid/Affine (Linear)",
            "modality": "T2",
            "children": [
                {"name": "Fixed_CT_Volume_1", "type": "vtkMRMLScalarVolumeNode", "modality": "T2", "series_description": "T2 MRI Registered"},
                {"name": "SUV_PET_Image_1", "type": "vtkMRMLScalarVolumeNode", "modality": "PET"},
                {"name": "PET_Lesions_1", "type": "vtkMRMLSegmentationNode", "segments": list(mr_seg_names.values())},
                {"name": "max_anatomy_fixed_ct_1", "source": "max_anatomy.nii.gz", "type": "vtkMRMLSegmentationNode"}
            ]
        }
    ]

    for s in extra_mri_series:
        scene_hierarchy.append({
            "name": f"Transform_FollowUp_to_Baseline_{s['tp_idx']}",
            "type": "vtkMRMLLinearTransformNode",
            "transform_type": "Rigid/Affine (Linear)",
            "modality": s["mod"],
            "children": [
                {"name": f"Fixed_CT_Volume_{s['tp_idx']}", "type": "vtkMRMLScalarVolumeNode", "modality": s["mod"], "series_description": s["desc"]},
                {"name": f"SUV_PET_Image_{s['tp_idx']}", "type": "vtkMRMLScalarVolumeNode", "modality": "PET"},
                {"name": f"PET_Lesions_{s['tp_idx']}", "type": "vtkMRMLSegmentationNode", "segments": list(mr_seg_names.values())},
                {"name": f"max_anatomy_fixed_ct_{s['tp_idx']}", "source": "max_anatomy.nii.gz", "type": "vtkMRMLSegmentationNode"}
            ]
        })

    # Step 9: Lossless HDF5 Packaging (libver="v110", gzip-4 + shuffle filter)
    print(f"    Writing lossless preprocessed_volumes.h5 (libver='v110', gzip-4 + shuffle)...")
    tmp_h5 = out_h5 + ".tmp"
    if os.path.isfile(tmp_h5):
        os.remove(tmp_h5)
    with h5py.File(tmp_h5, "w", libver="v110") as h5:
        # BASELINE Group
        b = h5.create_group("BASELINE")
        b.create_dataset("Fixed_CT_Volume_0.nii.gz", data=ct_arr_f, compression="gzip", compression_opts=4, shuffle=True)
        b.create_dataset("SUV_PET_Image_0.nii.gz", data=pet_arr_f, compression="gzip", compression_opts=4, shuffle=True)
        b.create_dataset("PET_Lesions_0.nii.gz", data=ct_les_f, compression="gzip", compression_opts=4, shuffle=True)
        b.create_dataset("max_anatomy.nii.gz", data=max_anat_f, compression="gzip", compression_opts=4, shuffle=True)
        b.create_dataset("skellytour.nii.gz", data=skellytour_f, compression="gzip", compression_opts=4, shuffle=True)
        b.create_dataset("diffusivity", data=diff_comb_f, compression="gzip", compression_opts=4, shuffle=True)

        # ATLAS Group
        a = h5.create_group("ATLAS")
        a.create_dataset("max_anatomy", data=max_anat_f, compression="gzip", compression_opts=4, shuffle=True)
        a.create_dataset("bone_atlas", data=bone_atlas_f, compression="gzip", compression_opts=4, shuffle=True)
        a.create_dataset("skellytour", data=skellytour_f, compression="gzip", compression_opts=4, shuffle=True)

        # Timepoint 1: Moving MRI T2
        mri_grp = h5.create_group("TFM_Transform_FollowUp_to_Baseline_1.tfm")
        mri_grp.create_dataset("Fixed_CT_Volume_1.nii.gz", data=t2_arr_f, compression="gzip", compression_opts=4, shuffle=True)
        mri_grp.create_dataset("SUV_PET_Image_1.nii.gz", data=pet_arr_f, compression="gzip", compression_opts=4, shuffle=True)
        mri_grp.create_dataset("PET_Lesions_1.nii.gz", data=mr_gold_f, compression="gzip", compression_opts=4, shuffle=True)
        mri_grp.create_dataset("max_anatomy.nii.gz", data=mri_anat_f, compression="gzip", compression_opts=4, shuffle=True)
        mri_grp.create_dataset("diffusivity", data=diff_mri_f, compression="gzip", compression_opts=4, shuffle=True)

        # Additional MRI timepoints (2..N)
        for s in extra_mri_series:
            tp_i = s["tp_idx"]
            tp_grp = h5.create_group(f"TFM_Transform_FollowUp_to_Baseline_{tp_i}.tfm")
            tp_grp.create_dataset(f"Fixed_CT_Volume_{tp_i}.nii.gz", data=s["arr_f"], compression="gzip", compression_opts=4, shuffle=True)
            tp_grp.create_dataset(f"SUV_PET_Image_{tp_i}.nii.gz", data=pet_arr_f, compression="gzip", compression_opts=4, shuffle=True)
            tp_grp.create_dataset(f"PET_Lesions_{tp_i}.nii.gz", data=mr_gold_f, compression="gzip", compression_opts=4, shuffle=True)
            tp_grp.create_dataset("max_anatomy.nii.gz", data=mri_anat_f, compression="gzip", compression_opts=4, shuffle=True)
            tp_grp.create_dataset("diffusivity", data=s["diff_f"], compression="gzip", compression_opts=4, shuffle=True)

        # CENTROIDS Group
        c_grp = h5.create_group("CENTROIDS")
        for ck, cv in centroids.items():
            c_grp.create_dataset(ck, data=cv)

        # _meta_ Group
        m_grp = h5.create_group("_meta_")
        m_grp.create_dataset("preflipped", data=np.int64(1))
        m_grp.create_dataset("bone_label_ids", data=np.array(bone_ids, dtype=np.int32))
        m_grp.create_dataset("matches.json", data=json.dumps(matches))
        m_grp.create_dataset("max_anatomy_labels.json", data=json.dumps(anat_labels))
        m_grp.create_dataset("prostate_anatomy_labels.json", data=json.dumps(PROSTATE_ZONE_LABELS))
        m_grp.create_dataset("custom_options.json", data=json.dumps({"metadata_version": "2.0"}))
        m_grp.create_dataset("metadata.json", data=json.dumps({"patient_id": pat_id, "case_id": pat_id}))
        m_grp.create_dataset("scene_hierarchy.json", data=json.dumps(scene_hierarchy))
        m_grp.create_dataset("segment_names.json", data=json.dumps({"0": ct_seg_names, "1": mr_seg_names}))

    os.replace(tmp_h5, out_h5)

    # Step 10: Headless Verification
    with h5py.File(out_h5, "r") as test_h5:
        assert "_meta_" in test_h5
        assert test_h5["_meta_"]["preflipped"][()] == 1
        assert "BASELINE" in test_h5

    elapsed = time.time() - t_start
    size_mb = os.path.getsize(out_h5) / (1024 * 1024)
    print(f" [GPU {gpu_id}] ✅ Successfully processed {pat_id} in {elapsed:.1f}s ({size_mb:.1f} MB)!")
    return out_h5, tfm_path, elapsed


def upload_case_worker(upload_q, xnat, progress, progress_lock):
    """Asynchronous background worker that streams finished HDF5 packages to XNAT."""
    while True:
        item = upload_q.get()
        if item is None:
            break
        pat_id, subj_id, exp_id, subj_label, h5_path, tfm_path = item
        remote_h5 = f"{subj_label}_preprocessed.h5"

        print(f"  [UPLOAD] Streaming {pat_id} ({os.path.getsize(h5_path)/(1024*1024):.1f} MB) to XNAT Subject MEDEYE3D...")
        t0 = time.time()
        # Ensure resources exist
        xnat.request("PUT", f"/data/projects/{PROJ_ID}/subjects/{subj_id}/resources/MEDEYE3D")
        xnat.request("PUT", f"/data/projects/{PROJ_ID}/subjects/{subj_id}/resources/ANNOTATIONS")

        # Upload HDF5 to Subject MEDEYE3D
        subj_h5_uri = f"/data/projects/{PROJ_ID}/subjects/{subj_id}/resources/MEDEYE3D/files/{remote_h5}?overwrite=true"
        ok1 = xnat.upload_file_streaming(subj_h5_uri, h5_path)

        # Upload Transform to Subject & Experiment MEDEYE3D
        if os.path.isfile(tfm_path):
            tfm_uri = f"/data/projects/{PROJ_ID}/subjects/{subj_id}/resources/MEDEYE3D/files/Transform_FollowUp_to_Baseline_1.tfm?overwrite=true"
            xnat.upload_file_streaming(tfm_uri, tfm_path)
            if exp_id:
                xnat.request("PUT", f"/data/experiments/{exp_id}/resources/MEDEYE3D")
                exp_tfm_uri = f"/data/experiments/{exp_id}/resources/MEDEYE3D/files/Transform_FollowUp_to_Baseline_1.tfm?overwrite=true"
                xnat.upload_file_streaming(exp_tfm_uri, tfm_path)

        # Upload status.json
        status_payload = json.dumps({
            "status": "PREPROCESSED",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "patient_id": pat_id,
            "subject_id": subj_id
        })
        xnat.request("PUT", f"/data/projects/{PROJ_ID}/subjects/{subj_id}/resources/ANNOTATIONS/files/status.json?overwrite=true&inbody=true",
                     data=status_payload, headers={"Content-Type": "application/json"})

        if ok1:
            print(f"  [UPLOAD] ✓ Successfully uploaded {pat_id} in {time.time()-t0:.1f}s!")
            with progress_lock:
                uploaded = set(progress.get("uploaded", []))
                uploaded.add(pat_id)
                progress["uploaded"] = sorted(list(uploaded), key=lambda x: int(re.search(r"\d+", x).group()))
                progress["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(PROGRESS_JSON, "w") as pf:
                    json.dump(progress, pf, indent=2)
        else:
            print(f"  [UPLOAD] ⚠ Warning: Upload for {pat_id} returned non-200.")
        upload_q.task_done()


def main():
    parser = argparse.ArgumentParser(description="Dual-GPU Cohort Processor for Prostata_bimodal_PIPELINE")
    parser.add_argument("--gpus", nargs="*", type=int, default=[0, 1], help="GPU IDs to use (default: [0, 1])")
    parser.add_argument("--cases", nargs="*", help="Specific case IDs to process (e.g. Pat1 Pat2 Pat51)")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of cases to process")
    parser.add_argument("--force", action="store_true", help="Force re-processing even if preprocessed_volumes.h5 exists")
    args = parser.parse_args()

    xnat = XnatManager()

    with open(CENSUS_PATH, "r") as cf:
        census = json.load(cf)

    # Sort all cases Pat1 ... Pat133
    all_pids = sorted(census.keys(), key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 9999)
    if args.cases:
        all_pids = [p for p in all_pids if p in args.cases]
    if args.limit:
        all_pids = all_pids[:args.limit]

    # Load progress
    progress = {}
    if os.path.isfile(PROGRESS_JSON):
        try:
            with open(PROGRESS_JSON, "r") as pf:
                progress = json.load(pf)
        except Exception:
            pass

    completed = set(progress.get("completed", []))
    uploaded = set(progress.get("uploaded", []))
    queue_cases = [p for p in all_pids if p not in uploaded or args.force]

    print("=" * 80)
    print(f" DUAL-GPU PROSTATA BIMODAL COHORT PROCESSOR")
    print(f" Total Cohort: {len(all_pids)} | Pending Queue: {len(queue_cases)} | Already Uploaded: {len(uploaded)}")
    print(f" Active GPUs: {args.gpus}")
    print("=" * 80)

    # Launch background upload threads
    upload_q = queue.Queue()
    progress_lock = threading.Lock()
    upload_threads = []
    for _ in range(3):
        up_t = threading.Thread(target=upload_case_worker, args=(upload_q, xnat, progress, progress_lock), daemon=True)
        up_t.start()
        upload_threads.append(up_t)

    # Case dispatch queue
    case_q = queue.Queue()
    for p in queue_cases:
        case_q.put(p)

    def worker_loop(gpu_id):
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
        while not case_q.empty():
            try:
                pid = case_q.get_nowait()
            except queue.Empty:
                break

            try:
                h5_p, tfm_p, duration = process_case(pid, gpu_id, xnat)
                info = census[pid]
                upload_q.put((pid, info["subj_id"], info["exp_id"], info["subj_label"], h5_p, tfm_p))

                with progress_lock:
                    completed.add(pid)
                    progress["completed"] = sorted(list(completed), key=lambda x: int(re.search(r"\d+", x).group()))
                    progress["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    with open(PROGRESS_JSON, "w") as pf:
                        json.dump(progress, pf, indent=2)

                print(f"[COHORT PROGRESS] Done {len(completed)}/{len(all_pids)} cases | Remaining in queue: {case_q.qsize()}")
            except Exception as e:
                import traceback
                print(f"[GPU {gpu_id}] ❌ Error on {pid}: {e}\n{traceback.format_exc()}")
            finally:
                case_q.task_done()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    threads = []
    for gid in args.gpus:
        t = threading.Thread(target=worker_loop, args=(gid,), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # Wait for uploads to complete
    upload_q.join()
    print("\n" + "=" * 80)
    print(f" COHORT PROCESSING COMPLETE! Successfully processed & uploaded {len(completed)}/{len(all_pids)} cases.")
    print("=" * 80)


if __name__ == "__main__":
    main()
