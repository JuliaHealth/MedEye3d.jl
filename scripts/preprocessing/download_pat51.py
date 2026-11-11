#!/usr/bin/env python3
"""
download_pat51.py
Downloads the required files for Pat51 from XNAT:
- packed_anatomical_tensor.h5 & registry
- prostate_segmentation.nii.gz
- Raw MRI DICOM series and converts them to NIfTI with SimpleITK
"""

import os
import sys
import io
import zipfile
import tempfile
import requests
import SimpleITK as sitk

RAW_DIR = "/workspaces/MedEye3d.jl/data/cases/Prostata_bimodal/Pat51/raw"
os.makedirs(RAW_DIR, exist_ok=True)

ALIAS = "755f5d1e-2a2a-45da-88a0-8dd543f4a594"
SECRET = "nn6Y9kDWyRuzlwVMrxH3RM2ooMOKBBM2grcqrhDgfNGd8oWJrjTZkPD27NgwZY0n"
SERVER = "https://imaging-platform.diz-ag.med.ovgu.de"

PET_EXP = "XNATImagingPlatform_E00142"
MRI_EXP = "XNATImagingPlatform_E00141"

sess = requests.Session()
r = sess.get(f"{SERVER}/data/JSESSION", auth=(ALIAS, SECRET), timeout=20)
r.raise_for_status()
sess.cookies.set("JSESSIONID", r.text.strip())
print("[XNAT] Authenticated JSESSIONID successfully.")

def download_file(uri, dest_path, min_bytes=512):
    if os.path.isfile(dest_path) and os.path.getsize(dest_path) >= min_bytes:
        print(f"  [CACHE] {os.path.basename(dest_path)} exists ({os.path.getsize(dest_path)} bytes)")
        return True
    print(f"  [DOWNLOAD] Fetching {os.path.basename(dest_path)}...")
    url = f"{SERVER}{uri}" if not uri.startswith("http") else uri
    r = sess.get(url, stream=True, timeout=120)
    r.raise_for_status()
    tmp_path = dest_path + ".tmp"
    with open(tmp_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    if os.path.getsize(tmp_path) < min_bytes:
        os.remove(tmp_path)
        raise RuntimeError(f"File {dest_path} smaller than min_bytes")
    os.replace(tmp_path, dest_path)
    print(f"  [DONE] Saved {os.path.basename(dest_path)} ({os.path.getsize(dest_path)/(1024*1024):.2f} MB)")
    return True

# 1. Download packed_anatomical_tensor.h5 & registry
download_file(
    f"/data/experiments/{PET_EXP}/resources/ANATOMICAL_TENSOR/files/packed_anatomical_tensor.h5",
    os.path.join(RAW_DIR, "packed_anatomical_tensor.h5"),
    min_bytes=1000000
)
download_file(
    f"/data/experiments/{PET_EXP}/resources/ANATOMICAL_TENSOR/files/packed_anatomical_tensor_registry.json",
    os.path.join(RAW_DIR, "packed_anatomical_tensor_registry.json"),
    min_bytes=100
)

# 2. Download prostate_segmentation.nii.gz
download_file(
    f"/data/experiments/{PET_EXP}/resources/PROSTATE_SEGMENTATION/files/prostate_segmentation.nii.gz",
    os.path.join(RAW_DIR, "prostate_segmentation.nii.gz"),
    min_bytes=1000
)

# 3. Download & Convert Raw MRI DICOMs
SCANS_TO_FETCH = [
    ("8001", "T2_ax_3mm.nii.gz", "T2 ax  3 mm"),
    ("10004", "IsoADC.nii.gz", "IsoADC"),
    ("10002", "IsoDWI.nii.gz", "IsoDWI"),
    ("7001", "T1_ax.nii.gz", "T1 FSE ax 6mm"),
    ("11001", "Dynamik_3D.nii.gz", "Dynamik 3D"),
]

for scan_id, out_name, desc in SCANS_TO_FETCH:
    out_path = os.path.join(RAW_DIR, out_name)
    if os.path.isfile(out_path) and os.path.getsize(out_path) > 10000:
        print(f"  [CACHE] {out_name} already exists ({os.path.getsize(out_path)/(1024*1024):.2f} MB)")
        continue
    print(f"  [FETCH-DICOM] Scan {scan_id} ({desc})...")
    url = f"{SERVER}/data/experiments/{MRI_EXP}/scans/{scan_id}/resources/DICOM/files?format=zip"
    rz = sess.get(url, stream=True, timeout=180)
    rz.raise_for_status()
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(io.BytesIO(rz.content)) as zf:
            zf.extractall(td)
        # Find directory containing DICOM slices
        dcm_dir = None
        for root, _, files in os.walk(td):
            if any(f.endswith(".dcm") or f.startswith("I00") or f.startswith("MR") for f in files):
                dcm_dir = root
                break
        if not dcm_dir:
            dcm_dir = td
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(dcm_dir)
        if not series_ids:
            # Fallback to direct file search
            fnames = reader.GetGDCMSeriesFileNames(dcm_dir)
        else:
            fnames = reader.GetGDCMSeriesFileNames(dcm_dir, series_ids[0])
        reader.SetFileNames(fnames)
        img = reader.Execute()
        sitk.WriteImage(img, out_path)
        print(f"  [CONVERTED] Scan {scan_id} -> {out_name} (Size: {img.GetSize()}, Spacing: {img.GetSpacing()})")

print("✅ Stage 1 Downloads & Conversions complete!")
