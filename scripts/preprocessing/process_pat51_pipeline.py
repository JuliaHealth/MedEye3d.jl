#!/usr/bin/env python3
"""
process_pat51_pipeline.py

End-to-End processing pipeline for Pat51 (Prostata_bimodal):
1. Loads CT, PET SUV, raw moving MRI (T2, ADC, DWI, T1, DCE), and prostate segmentation.
2. Performs SVD Procrustes rigid registration from TotalSegmentator hip & femur bone landmarks.
3. Resamples all moving MRI series and 8-zone prostate segmentation onto the CT grid.
4. Layers CT whole-body anatomy with fresh lymph nodes from final_results_gpu.h5.
5. Strictly separates MRI 8-zone prostate anatomy (prostate_anatomy_labels.json).
6. Computes edge derivations (CT, PET, combined, and MRI) and corresponding diffusivities per EDGE_EXTRACTION_GUIDE.md.
7. Pre-flips along axis 1 (Y) with preflipped=1 and calculates 1-based centroids.
8. Assembles losslessly compressed preprocessed_volumes.h5 with all MedEye3d metadata.
9. Uploads the processed package to XNAT MEDEYE3D resource.
"""

import os
import sys
import io
import json
import time
import re
import shutil
import requests
import numpy as np
import h5py
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter, label

CASE_DIR = "/workspaces/MedEye3d.jl/data/cases/Prostata_bimodal/Pat51"
RAW_DIR = os.path.join(CASE_DIR, "raw")
OUT_H5 = os.path.join(CASE_DIR, "preprocessed_volumes.h5")

ALIAS = "755f5d1e-2a2a-45da-88a0-8dd543f4a594"
SECRET = "nn6Y9kDWyRuzlwVMrxH3RM2ooMOKBBM2grcqrhDgfNGd8oWJrjTZkPD27NgwZY0n"
SERVER = "https://imaging-platform.diz-ag.med.ovgu.de"

PROJ_ID = "Prostata_bimodal"
SUBJ_ID = "XNATImagingPlatform_S00146"
SUBJ_LABEL = "FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat51"
EXP_ID = "XNATImagingPlatform_E00142"

CHAMPION_GAMMA = 10.095886
CHAMPION_EPS_D = 1.0e-5

# ==============================================================================
# 1. Registration: Prostate-Centric Moments & Rigid Alignment
# ==============================================================================

def compute_prostate_centered_registration():
    print("▶ Computing prostate and pelvic bone rigid registration (SVD Procrustes)...")
    ct_img = sitk.ReadImage(os.path.join(RAW_DIR, "Fixed_CT_Volume.nii.gz"))
    ts_mr_t2 = sitk.ReadImage(os.path.join(RAW_DIR, "totalseg_mr_ml.nii.gz"))
    ts_mr_arr = sitk.GetArrayFromImage(ts_mr_t2)

    # Ensure T1 TotalSegmentator segmentation exists
    ts_t1_path = os.path.join(RAW_DIR, "totalseg_t1_ml.nii.gz")
    if not os.path.isfile(ts_t1_path):
        print("  Running TotalSegmentator on T1_ax.nii.gz for pelvic bone landmarks...")
        totalseg_bin = "/mnt/big/project_ssd/lymph_node_rules/for_joanna/.local/conda/envs/seg/bin/TotalSegmentator"
        cmd = f"{totalseg_bin} -i {os.path.join(RAW_DIR, 'T1_ax.nii.gz')} -o {ts_t1_path} -ta total_mr --fast --ml --device gpu"
        ret = os.system(cmd)
        if ret != 0:
            print(f"  [WARN] TotalSegmentator exited with code {ret}")

    ts_t1 = sitk.ReadImage(ts_t1_path)
    stats_t1 = sitk.LabelShapeStatisticsImageFilter()
    stats_t1.Execute(ts_t1)

    with h5py.File(os.path.join(RAW_DIR, "packed_anatomical_tensor.h5"), "r") as f:
        ct_pt = f["packed_tensor"][:]

    # Fixed CT landmarks:
    # 1. Prostate centroid (channel 2, id 92)
    c_pros_ct = np.array(ct_img.TransformContinuousIndexToPhysicalPoint((
        float(np.where(ct_pt[1] == 92)[2].mean()),
        float(np.where(ct_pt[1] == 92)[1].mean()),
        float(np.where(ct_pt[1] == 92)[0].mean())
    )))
    # 2. Femur Left (channel 3, id 5, sliced in [48, 62] around femoral head)
    ct_fl = ((ct_pt[2] == 5) & (np.arange(284)[:, None, None] >= 48) & (np.arange(284)[:, None, None] <= 62))
    c_fl_ct = np.array(ct_img.TransformContinuousIndexToPhysicalPoint((
        float(np.where(ct_fl)[2].mean()),
        float(np.where(ct_fl)[1].mean()),
        float(np.where(ct_fl)[0].mean())
    )))
    # 3. Femur Right (channel 3, id 4, sliced in [48, 62] around femoral head)
    ct_fr = ((ct_pt[2] == 4) & (np.arange(284)[:, None, None] >= 48) & (np.arange(284)[:, None, None] <= 62))
    c_fr_ct = np.array(ct_img.TransformContinuousIndexToPhysicalPoint((
        float(np.where(ct_fr)[2].mean()),
        float(np.where(ct_fr)[1].mean()),
        float(np.where(ct_fr)[0].mean())
    )))
    # 4. Hip Left (channel 3, id 2, bounded to pelvic FOV <= 115)
    ct_hl = ((ct_pt[2] == 2) & (np.arange(284)[:, None, None] <= 115))
    c_hl_ct = np.array(ct_img.TransformContinuousIndexToPhysicalPoint((
        float(np.where(ct_hl)[2].mean()),
        float(np.where(ct_hl)[1].mean()),
        float(np.where(ct_hl)[0].mean())
    )))
    # 5. Hip Right (channel 3, id 3, bounded to pelvic FOV <= 115)
    ct_hr = ((ct_pt[2] == 3) & (np.arange(284)[:, None, None] <= 115))
    c_hr_ct = np.array(ct_img.TransformContinuousIndexToPhysicalPoint((
        float(np.where(ct_hr)[2].mean()),
        float(np.where(ct_hr)[1].mean()),
        float(np.where(ct_hr)[0].mean())
    )))
    # 6. Sacrum (channel 2, id 29, bounded to pelvic FOV <= 115)
    ct_sac = ((ct_pt[1] == 29) & (np.arange(284)[:, None, None] <= 115))
    c_sac_ct = np.array(ct_img.TransformContinuousIndexToPhysicalPoint((
        float(np.where(ct_sac)[2].mean()),
        float(np.where(ct_sac)[1].mean()),
        float(np.where(ct_sac)[0].mean())
    )))

    # Moving MR landmarks:
    # 1. Prostate centroid (from T2 totalseg_mr_ml, label 17)
    c_pros_mr = np.array(ts_mr_t2.TransformContinuousIndexToPhysicalPoint((
        float(np.where(ts_mr_arr == 17)[2].mean()),
        float(np.where(ts_mr_arr == 17)[1].mean()),
        float(np.where(ts_mr_arr == 17)[0].mean())
    )))
    # 2. Femur Left (from T1, label 36)
    c_fl_mr = np.array(stats_t1.GetCentroid(36))
    # 3. Femur Right (from T1, label 37)
    c_fr_mr = np.array(stats_t1.GetCentroid(37))
    # 4. Hip Left (from T1, label 38)
    c_hl_mr = np.array(stats_t1.GetCentroid(38))
    # 5. Hip Right (from T1, label 39)
    c_hr_mr = np.array(stats_t1.GetCentroid(39))
    # 6. Sacrum (from T1, label 18)
    c_sac_mr = np.array(stats_t1.GetCentroid(18))

    fixed_pts = np.array([c_pros_ct, c_fl_ct, c_fr_ct, c_hl_ct, c_hr_ct, c_sac_ct])
    moving_pts = np.array([c_pros_mr, c_fl_mr, c_fr_mr, c_hl_mr, c_hr_mr, c_sac_mr])
    names = ['prostate', 'femur_L', 'femur_R', 'hip_L', 'hip_R', 'sacrum']
    weights = np.array([8.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    W = weights / weights.sum()
    cf = np.sum(fixed_pts * W[:, None], axis=0)
    cm = np.sum(moving_pts * W[:, None], axis=0)
    pf = fixed_pts - cf
    pm = moving_pts - cm
    H = (pm * W[:, None]).T @ pf
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cf - R @ cm

    residuals = np.linalg.norm((R @ moving_pts.T).T + t - fixed_pts, axis=1)
    rot_angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    print(f"  [REGISTRATION] Arun SVD Procrustes: rotation={rot_angle:.2f}°, trans=[{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}] mm")
    for n, r in zip(names, residuals):
        print(f"    - {n:12s}: residual = {r:.2f} mm")

    # Invert for ITK transform (maps Fixed CT -> Moving MR)
    R_itk = R.T
    t_itk = -R.T @ t

    tfm = sitk.AffineTransform(3)
    tfm.SetMatrix(R_itk.flatten().tolist())
    tfm.SetTranslation(t_itk.tolist())

    # Measure prostate overlap
    mr_pros_img = sitk.GetImageFromArray((ts_mr_arr == 17).astype(np.uint8))
    mr_pros_img.CopyInformation(ts_mr_t2)
    ct_pros_img = sitk.GetImageFromArray((ct_pt[1] == 92).astype(np.uint8))
    ct_pros_img.CopyInformation(ct_img)

    res_pros = sitk.Resample(mr_pros_img, ct_pros_img, tfm, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    res_pros_arr = sitk.GetArrayFromImage(res_pros)
    inter = ((ct_pt[1] == 92) > 0) & (res_pros_arr > 0)
    dice = 2.0 * inter.sum() / (((ct_pt[1] == 92) > 0).sum() + (res_pros_arr > 0).sum())
    print(f"  [REGISTRATION] Prostate Alignment Dice: {dice:.4f} (intersection: {inter.sum()} / {((ct_pt[1] == 92) > 0).sum()} CT voxels)")

    tfm_path = os.path.join(CASE_DIR, "Transform_FollowUp_to_Baseline_1.tfm")
    sitk.WriteTransform(tfm, tfm_path)
    shutil.copyfile(tfm_path, os.path.join(RAW_DIR, "Transform_FollowUp_to_Baseline_1.tfm"))
    print(f"  [REGISTRATION] Exported {tfm_path}")
    return tfm_path

# ==============================================================================
# 2. Resampling MRI Sequences & 8-Zone Prostate Mask
# ==============================================================================

def resample_mri_modalities(ct_img, tfm_path):
    print("▶ Resampling MRI modalities and dedicated prostate anatomy onto CT grid...")
    tfm = sitk.ReadTransform(tfm_path)
    ct_arr = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    Z, Y, X = ct_arr.shape

    # 1. Primary T2 MRI (Timepoint 1)
    t2_path = os.path.join(RAW_DIR, "T2_ax_3mm.nii.gz")
    t2_img = sitk.ReadImage(t2_path)
    res_t2 = sitk.Resample(t2_img, ct_img, tfm, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    t2_arr = sitk.GetArrayFromImage(res_t2).astype(np.float32)
    t2_arr = np.maximum(0.0, t2_arr)
    pos = t2_arr[t2_arr > 0]
    if len(pos) > 0:
        p99 = float(np.percentile(pos, 99))
        if p99 > 0:
            t2_arr = t2_arr * (1000.0 / p99)

    # 2. Dedicated 8-zone MRI prostate anatomy
    mri_anatomy = np.zeros((Z, Y, X), dtype=np.uint16)
    pros_path = os.path.join(RAW_DIR, "prostate_segmentation.nii.gz")
    if os.path.isfile(pros_path):
        p_img = sitk.ReadImage(pros_path)
        # Resample p_img onto t2_img space first
        ts_mr_img = sitk.ReadImage(os.path.join(RAW_DIR, "totalseg_mr_ml.nii.gz"))
        mr_pros = (sitk.GetArrayFromImage(ts_mr_img) == 17).astype(np.uint8)
        mr_pros_img = sitk.GetImageFromArray(mr_pros)
        mr_pros_img.CopyInformation(t2_img)

        p8_arr = sitk.GetArrayFromImage(p_img)
        p8_pros = (np.isin(p8_arr, [1, 2, 3, 4, 5, 9])).astype(np.uint8)
        p8_sub_img = sitk.GetImageFromArray(p8_pros)
        p8_sub_img.CopyInformation(p_img)

        tfm_p8_to_t2 = sitk.CenteredTransformInitializer(
            mr_pros_img,
            p8_sub_img,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS
        )
        p8_on_t2 = sitk.Resample(p_img, t2_img, tfm_p8_to_t2, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)

        res_p = sitk.Resample(p8_on_t2, ct_img, tfm, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
        p_arr = sitk.GetArrayFromImage(res_p).astype(np.uint16)
        for zid in range(1, 9):
            mri_anatomy[p_arr == zid] = zid
        print(f"  [MRI ANATOMY] 8-zone prostate sub-segmentation resampled: non-zero={(mri_anatomy > 0).sum()} voxels, unique labels={np.unique(mri_anatomy)}")

    # 3. Additional MRI series (ADC, DWI, T1, DCE)
    extra_series = []
    candidates = [
        ("ADC", "IsoADC.nii.gz", "IsoADC MRI Registered (AX)"),
        ("DWI", "IsoDWI.nii.gz", "IsoDWI MRI Registered (AX)"),
        ("T1", "T1_ax.nii.gz", "T1 FSE MRI Registered (AX)"),
        ("DCE", "Dynamik_3D.nii.gz", "DCE Dynamic MRI Registered (AX)"),
    ]

    tp_idx = 2
    for mod, fname, desc in candidates:
        fpath = os.path.join(RAW_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            im = sitk.ReadImage(fpath)
            # Handle 4D dynamic sequences if multiple temporal phases are stacked
            if mod == "DCE" and im.GetSize()[2] > 100:
                raw_dce = sitk.GetArrayFromImage(im)
                total_z = raw_dce.shape[0]
                slices_per_phase = 18 if total_z % 18 == 0 else (23 if total_z % 23 == 0 else 18)
                n_phases = total_z // slices_per_phase
                target_phase = min(3, n_phases - 1)
                phase_arr = raw_dce[target_phase * slices_per_phase : (target_phase + 1) * slices_per_phase]
                clean_dce = sitk.GetImageFromArray(phase_arr)
                clean_dce.SetSpacing((im.GetSpacing()[0], im.GetSpacing()[1], 3.5))
                clean_dce.SetOrigin(im.GetOrigin())
                clean_dce.SetDirection(im.GetDirection())
                im = clean_dce

            res_extra = sitk.Resample(im, ct_img, tfm, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
            arr = sitk.GetArrayFromImage(res_extra).astype(np.float32)
            arr = np.maximum(0.0, arr)
            pos_e = arr[arr > 0]
            if len(pos_e) > 0:
                p99_e = float(np.percentile(pos_e, 99))
                scale = 1500.0 / max(p99_e, 1.0) if mod == "ADC" else 1000.0 / max(p99_e, 1.0)
                arr = arr * scale

            extra_series.append({
                "tp_idx": tp_idx,
                "mod": mod,
                "fn": fname,
                "desc": desc,
                "arr": arr
            })
            print(f"  [MRI TP {tp_idx}] Resampled {mod} ({fname}) -> CT grid (non-zero={(arr > 0).sum()} voxels)")
            tp_idx += 1
        except Exception as e:
            print(f"  [WARN] Failed to resample {fname}: {e}")

    return t2_arr, mri_anatomy, extra_series

# ==============================================================================
# 3. Layered CT Whole-Body Anatomy & Fresh Lymph Node Integration
# ==============================================================================

def unpack_ct_anatomy(ct_shape):
    print("▶ Unpacking layered CT whole-body anatomy...")
    Z, Y, X = ct_shape
    max_anatomy = np.zeros((Z, Y, X), dtype=np.uint16)
    exclusion_mask = np.zeros((Z, Y, X), dtype=np.uint8)
    anatomy_label_names = {}
    bone_ids = []

    anat_h5_path = os.path.join(RAW_DIR, "packed_anatomical_tensor.h5")
    reg_json_path = os.path.join(RAW_DIR, "packed_anatomical_tensor_registry.json")

    with open(reg_json_path, "r") as rf:
        reg_doc = json.load(rf)
    registry = reg_doc.get("masks", reg_doc)

    bone_terms = ['femur', 'hip', 'vertebra', 'rib', 'sacrum', 'clavicula', 'humerus', 'scapula', 'sternum', 'skull', 'costal', 'mandible']
    muscle_terms = ['gluteus', 'iliopsoas', 'autochthon', 'pectoralis', 'latissimus', 'trapezius', 'deltoid', 'muscle', 'adductor']

    def get_layer(name):
        n = name.lower()
        if any(b in n for b in bone_terms): return 2
        if any(m in n for m in muscle_terms): return 0
        return 1

    with h5py.File(anat_h5_path, "r") as af:
        packed_tensor = af["packed_tensor"][:]

        items_by_layer = {0: [], 1: [], 2: []}
        for name, meta in registry.items():
            if not isinstance(meta, dict) or meta.get("id", 0) <= 0:
                continue
            clean = name.replace("nvseg_", "").lower()
            lay = get_layer(clean)
            items_by_layer[lay].append((name, clean, meta))

        lbl_idx = 1
        # Layer 0 (Muscles) and Layer 1 (Viscera & Vessels)
        for lay in [0, 1]:
            for name, clean, meta in sorted(items_by_layer[lay], key=lambda x: x[1]):
                ch = meta["channel"] - 1
                mid = meta["id"]
                if ch < packed_tensor.shape[0]:
                    mask_i = (packed_tensor[ch] == mid)
                    if mask_i.any():
                        anatomy_label_names[str(lbl_idx)] = clean
                        max_anatomy[mask_i] = lbl_idx
                        if any(t in clean for t in ["urinary_bladder", "bladder", "parotid", "submandibular", "lacrimal", "kidney"]):
                            exclusion_mask[mask_i] = 1
                        lbl_idx += 1

        # Layer 1.5: Freshly recomputed Lymph Node Stations from final_results_gpu.h5
        ln_h5_path = os.path.join(RAW_DIR, "final_results_gpu.h5")
        if os.path.isfile(ln_h5_path):
            print(f"  [ANATOMY] Layering freshly recomputed lymph node stations from {os.path.basename(ln_h5_path)}...")
            with h5py.File(ln_h5_path, "r") as lf:
                for k in sorted(lf.keys()):
                    if any(h in k for h in ["_Base", "_Helper", "_Dilated", "_Exclusion_", "_Bridge", "helper_"]):
                        continue
                    arr_ln = lf[k][:]
                    # Check shape: Python (Z, Y, X) vs (X, Y, Z)
                    if arr_ln.ndim == 3:
                        if arr_ln.shape == (X, Y, Z):
                            arr_ln = np.transpose(arr_ln, (2, 1, 0))
                        elif arr_ln.shape != (Z, Y, X):
                            continue
                        m_ln = (arr_ln > 0)
                        if m_ln.any():
                            clean_ln = k.lower()
                            anatomy_label_names[str(lbl_idx)] = clean_ln
                            max_anatomy[m_ln] = lbl_idx
                            lbl_idx += 1

        # Layer 2: Bones
        for name, clean, meta in sorted(items_by_layer[2], key=lambda x: x[1]):
            ch = meta["channel"] - 1
            mid = meta["id"]
            if ch < packed_tensor.shape[0]:
                mask_i = (packed_tensor[ch] == mid)
                if mask_i.any():
                    anatomy_label_names[str(lbl_idx)] = clean
                    max_anatomy[mask_i] = lbl_idx
                    bone_ids.append(lbl_idx)
                    lbl_idx += 1

    bone_atlas = np.isin(max_anatomy, bone_ids).astype(np.float32)
    skellytour = bone_atlas.copy()
    print(f"  [ANATOMY] Total CT anatomical structures: {len(anatomy_label_names)}, Bone labels: {len(bone_ids)}")
    print(f"  [ANATOMY] Exclusion mask voxels (physiological uptake insulation): {(exclusion_mask > 0).sum()}")
    return max_anatomy, anatomy_label_names, bone_ids, bone_atlas, skellytour, exclusion_mask

# ==============================================================================
# 4. Edge Derivations & Heat-GDT Diffusivity (EDGE_EXTRACTION_GUIDE.md)
# ==============================================================================

def compute_edge_derivations(ct_arr, pet_arr, exclusion_mask):
    """
    Computes 3D boundary edge probability maps E(x) ∈ [0, 1] and corresponding
    Heat-GDT diffusivities D(x) for:
      - 'ct_only': Pure CT edges and diffusivity
      - 'pet_only': Pure PET edges and diffusivity
      - 'combined': Dual PET+CT edges and diffusivity with physiological insulation
    """
    print("▶ Computing multi-modality 3D edge derivations (CT, PET, and Combined)...")
    ct_norm = (np.clip(ct_arr, -1024.0, 1500.0) - (-200.0)) / 400.0
    pet_norm = np.clip(pet_arr, 0.0, 20.0) / 20.0

    # CT gradient magnitude
    gz_ct, gy_ct, gx_ct = np.gradient(ct_norm)
    mag_ct = np.sqrt(gz_ct**2 + gy_ct**2 + gx_ct**2)
    mag_ct = gaussian_filter(mag_ct, sigma=0.8)

    # PET gradient magnitude
    gz_pet, gy_pet, gx_pet = np.gradient(pet_norm)
    mag_pet = np.sqrt(gz_pet**2 + gy_pet**2 + gx_pet**2)
    mag_pet = gaussian_filter(mag_pet, sigma=0.8)

    # Sigmoidal edge probability maps E(x) ∈ [0, 1]
    e_ct_raw = 1.0 / (1.0 + np.exp(-4.0 * (mag_ct - 0.25)))
    e_ct_zero = 1.0 / (1.0 + np.exp(-4.0 * (-0.25)))
    edge_ct = np.clip((e_ct_raw - e_ct_zero) / (1.0 - e_ct_zero), 0.0, 1.0).astype(np.float32)

    e_pet_raw = 1.0 / (1.0 + np.exp(-6.0 * (mag_pet - 0.20)))
    e_pet_zero = 1.0 / (1.0 + np.exp(-6.0 * (-0.20)))
    edge_pet = np.clip((e_pet_raw - e_pet_zero) / (1.0 - e_pet_zero), 0.0, 1.0).astype(np.float32)

    edge_comb = np.maximum(edge_ct, edge_pet).astype(np.float32)

    def to_diff(E):
        D = CHAMPION_EPS_D + (1.0 - CHAMPION_EPS_D) * np.exp(-CHAMPION_GAMMA * E)
        D = np.clip(D, CHAMPION_EPS_D, 1.0).astype(np.float32)
        if exclusion_mask is not None:
            D[exclusion_mask > 0] = 0.0
        return D

    diff_comb = to_diff(edge_comb)
    diff_ct = to_diff(edge_ct)
    diff_pet = to_diff(edge_pet)

    print(f"  [EDGES] Computed edge_ct (mean={edge_ct.mean():.3f}), edge_pet (mean={edge_pet.mean():.3f}), edge_comb (mean={edge_comb.mean():.3f})")
    print(f"  [DIFFUSIVITY] Computed diff_comb, diff_ct, diff_pet with boundary insulation")
    return edge_comb, edge_ct, edge_pet, diff_comb, diff_ct, diff_pet

def compute_mri_edge_derivations(mri_arr, pet_arr, mri_anatomy, exclusion_mask):
    """Computes structural edges and diffusivities for MRI (treated as pure CT)."""
    pos = mri_arr[mri_arr > 0]
    p99 = np.percentile(pos, 99) if len(pos) > 0 else 1.0
    p01 = np.percentile(pos, 1) if len(pos) > 0 else 0.0
    norm = np.clip((mri_arr - p01) / max(p99 - p01, 1e-4), 0.0, 1.5)

    gz, gy, gx = np.gradient(norm)
    mag = gaussian_filter(np.sqrt(gz**2 + gy**2 + gx**2), sigma=0.8)

    e_mri_raw = 1.0 / (1.0 + np.exp(-4.5 * (mag - 0.22)))
    e_mri_zero = 1.0 / (1.0 + np.exp(-4.5 * (-0.22)))
    edge_mri = np.clip((e_mri_raw - e_mri_zero) / (1.0 - e_mri_zero), 0.0, 1.0).astype(np.float32)

    # Enforce prostate capsule barrier
    if mri_anatomy is not None and (mri_anatomy > 0).any():
        p_bin = (mri_anatomy > 0).astype(np.float32)
        gz_p, gy_p, gx_p = np.gradient(p_bin)
        capsule = np.sqrt(gz_p**2 + gy_p**2 + gx_p**2)
        edge_mri = np.maximum(edge_mri, np.clip(capsule * 2.0, 0.0, 1.0))

    D_mri = CHAMPION_EPS_D + (1.0 - CHAMPION_EPS_D) * np.exp(-CHAMPION_GAMMA * edge_mri)
    D_mri = np.clip(D_mri, CHAMPION_EPS_D, 1.0).astype(np.float32)
    if exclusion_mask is not None:
        D_mri[exclusion_mask > 0] = 0.0

    # PET overlay diffusivity
    pet_norm = np.clip(pet_arr, 0.0, 20.0) / 20.0
    gz_pet, gy_pet, gx_pet = np.gradient(pet_norm)
    mag_pet = gaussian_filter(np.sqrt(gz_pet**2 + gy_pet**2 + gx_pet**2), sigma=0.8)
    e_pet_raw = 1.0 / (1.0 + np.exp(-6.0 * (mag_pet - 0.20)))
    e_pet_zero = 1.0 / (1.0 + np.exp(-6.0 * (-0.20)))
    edge_pet = np.clip((e_pet_raw - e_pet_zero) / (1.0 - e_pet_zero), 0.0, 1.0).astype(np.float32)
    D_pet = CHAMPION_EPS_D + (1.0 - CHAMPION_EPS_D) * np.exp(-CHAMPION_GAMMA * edge_pet)
    D_pet = np.clip(D_pet, CHAMPION_EPS_D, 1.0).astype(np.float32)
    if exclusion_mask is not None:
        D_pet[exclusion_mask > 0] = 0.0

    return edge_mri, edge_pet, D_mri, D_pet

# ==============================================================================
# 5. Full Packaging into preprocessed_volumes.h5
# ==============================================================================

def package_hdf5():
    # 1. Registration
    tfm_path = compute_prostate_centered_registration()

    # 2. Geometry & Volumes
    ct_img = sitk.ReadImage(os.path.join(RAW_DIR, "Fixed_CT_Volume.nii.gz"))
    spacing = list(ct_img.GetSpacing())
    origin = list(ct_img.GetOrigin())
    direction = list(ct_img.GetDirection())
    ct_arr = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    Z, Y, X = ct_arr.shape

    # PET SUV resampled to CT grid
    suv_img = sitk.ReadImage(os.path.join(RAW_DIR, "SUV_PET_WB_120min_Uncorrected.nii.gz"))
    res_suv = sitk.Resample(suv_img, ct_img, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    pet_arr = sitk.GetArrayFromImage(res_suv).astype(np.float32)

    # MRI sequences resampled
    t2_arr, mri_anatomy, extra_mri_series = resample_mri_modalities(ct_img, tfm_path)

    # CT Whole-Body anatomy
    max_anatomy, anatomy_label_names, bone_ids, bone_atlas, skellytour, exclusion_mask = unpack_ct_anatomy((Z, Y, X))

    # Dedicated 8-zone prostate labels
    prostate_labels = {
        "1": "Peripheral Zone (PZ)",
        "2": "Transition Zone (TZ)",
        "3": "Central Zone (CZ)",
        "4": "Anterior Fibromuscular Stroma (AFS)",
        "5": "Urethra (UR)",
        "6": "Periurethral Glands (PG)",
        "7": "Seminal Vesicle Left (SV_L)",
        "8": "Seminal Vesicle Right (SV_R)"
    }

    # Edges & Diffusivity for Baseline (CT + PET)
    edge_comb, edge_ct, edge_pet, diff_comb, diff_ct, diff_pet = compute_edge_derivations(ct_arr, pet_arr, exclusion_mask)

    # Edges & Diffusivity for MRI (T2)
    edge_mri1, edge_pet1, diff_mri1, diff_pet1 = compute_mri_edge_derivations(t2_arr, pet_arr, mri_anatomy, exclusion_mask)

    # Edges & Diffusivity for additional MRI series
    for s in extra_mri_series:
        e_m, e_p, d_m, d_p = compute_mri_edge_derivations(s["arr"], pet_arr, mri_anatomy, exclusion_mask)
        s["edge_mri"] = e_m
        s["diff_mri"] = d_m
        s["diff_pet"] = d_p

    # Lesions: Real expert segmentations from XNAT (slicer_final)
    # CT: NNUNET_OR_Processed_Lesions_CT.seg.nrrd (15 lesions)
    # MR: NNUNET_OR_Processed_Lesions_MR.seg.nrrd (4 prostate lesions matching CT 2, 3, 4, 5)
    ct_nrrd_path = os.path.join(RAW_DIR, "NNUNET_OR_Processed_Lesions_CT.seg.nrrd")
    mr_nrrd_path = os.path.join(RAW_DIR, "NNUNET_OR_Processed_Lesions_MR.seg.nrrd")

    ct_nrrd = sitk.ReadImage(ct_nrrd_path)
    mr_nrrd = sitk.ReadImage(mr_nrrd_path)

    res_ct_nrrd = sitk.Resample(ct_nrrd, ct_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkInt16)
    res_mr_nrrd = sitk.Resample(mr_nrrd, ct_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkInt16)

    lesion_ct_arr = sitk.GetArrayFromImage(res_ct_nrrd).astype(np.int16)
    raw_mr_arr = sitk.GetArrayFromImage(res_mr_nrrd).astype(np.int16)

    # Remap MR labels 1..4 -> 2..5 to match CT prostate lesion IDs exactly
    lesion_mr_arr = np.zeros_like(raw_mr_arr)
    for old_id, new_id in [(1, 2), (2, 3), (3, 4), (4, 5)]:
        lesion_mr_arr[raw_mr_arr == old_id] = new_id

    ct_labels = sorted([int(k) for k in np.unique(lesion_ct_arr) if k > 0])
    mr_labels = sorted([int(k) for k in np.unique(lesion_mr_arr) if k > 0])
    print(f"  [LESIONS] CT unique labels: {ct_labels} (total voxels: {(lesion_ct_arr > 0).sum()})")
    print(f"  [LESIONS] MR unique labels (remapped): {mr_labels} (total voxels: {(lesion_mr_arr > 0).sum()})")

    # Read segment names from NRRD metadata
    ct_seg_names = {}
    for i in range(20):
        k_name = f"Segment{i}_Name"
        k_val = f"Segment{i}_LabelValue"
        if ct_nrrd.HasMetaDataKey(k_name) and ct_nrrd.HasMetaDataKey(k_val):
            val = str(int(ct_nrrd.GetMetaData(k_val)))
            name = ct_nrrd.GetMetaData(k_name)
            ct_seg_names[val] = name

    mr_seg_names = {}
    for i in range(10):
        k_name = f"Segment{i}_Name"
        k_val = f"Segment{i}_LabelValue"
        if mr_nrrd.HasMetaDataKey(k_name) and mr_nrrd.HasMetaDataKey(k_val):
            orig_val = int(mr_nrrd.GetMetaData(k_val))
            mapped_val = str(orig_val + 1)
            name = mr_nrrd.GetMetaData(k_name)
            mr_seg_names[mapped_val] = name

    # Organ mapping
    ct_organs = {}
    for lid_str, name in ct_seg_names.items():
        n = name.lower()
        if lid_str == "1":
            organ = "Bone / Lower Extremity"
        elif "prostate" in n:
            organ = "Prostate"
        elif "liver" in n:
            organ = "Liver"
        elif "vertebra" in n:
            organ = "Spine"
        elif "hip" in n or "femur" in n or "bone" in n:
            organ = "Bone"
        elif "bowel" in n:
            organ = "Small Bowel"
        elif "lung" in n:
            organ = "Lung"
        else:
            organ = "Unknown"
        ct_organs[lid_str] = organ

    mr_organs = {}
    for lid_str in mr_seg_names.keys():
        mr_organs[lid_str] = "Prostate"

    all_mr_tps = [1] + [s["tp_idx"] for s in extra_mri_series]
    organ_mapping = {"0": ct_organs}
    segment_names = {"0": ct_seg_names}
    for tp in all_mr_tps:
        organ_mapping[str(tp)] = mr_organs
        segment_names[str(tp)] = mr_seg_names

    # Cross-timepoint matches.json
    matches = []
    # 1. Matched prostate lesions (2, 3, 4, 5) across CT and all MRI timepoints
    for lid in [2, 3, 4, 5]:
        children = []
        for tp in all_mr_tps:
            children.append({
                "name": f"PET_Lesions_{tp}",
                "raw_lesion": f"Segment_{lid}",
                "node": "child",
                "group_id": lid,
                "lesion": mr_seg_names.get(str(lid), f"Prostate Lesion {lid}"),
                "match_type": "EXACT_ID"
            })
        matches.append({
            "name": "PET_Lesions_0",
            "raw_lesion": f"Segment_{lid}",
            "node": "parent",
            "group_id": lid,
            "lesion": ct_seg_names.get(str(lid), f"Prostate Lesion {lid}"),
            "children": children
        })

    # 2. CT-only lesions (1, 6..15)
    for lid in ct_labels:
        if lid not in [2, 3, 4, 5]:
            matches.append({
                "name": "PET_Lesions_0",
                "raw_lesion": f"Segment_{lid}",
                "node": "parent",
                "group_id": lid,
                "lesion": ct_seg_names.get(str(lid), f"Lesion {lid}"),
                "children": []
            })

    # Pre-flip all arrays along axis 1 (Y) for standard radiological transverse view (belly up, spine down)
    print("▶ Pre-flipping all volumes along axis 1 (Y) with preflipped=1...")
    ct_arr = np.flip(ct_arr, axis=1)
    pet_arr = np.flip(pet_arr, axis=1)
    t2_arr = np.flip(t2_arr, axis=1)
    lesion_ct_arr = np.flip(lesion_ct_arr, axis=1)
    lesion_mr_arr = np.flip(lesion_mr_arr, axis=1)
    max_anatomy = np.flip(max_anatomy, axis=1)
    mri_anatomy = np.flip(mri_anatomy, axis=1)
    bone_atlas = np.flip(bone_atlas, axis=1)
    skellytour = np.flip(skellytour, axis=1)

    diff_comb = np.flip(diff_comb, axis=1)
    diff_ct = np.flip(diff_ct, axis=1)
    diff_pet = np.flip(diff_pet, axis=1)
    edge_comb = np.flip(edge_comb, axis=1)
    edge_ct = np.flip(edge_ct, axis=1)
    edge_pet = np.flip(edge_pet, axis=1)

    diff_mri1 = np.flip(diff_mri1, axis=1)
    diff_pet1 = np.flip(diff_pet1, axis=1)
    edge_mri1 = np.flip(edge_mri1, axis=1)

    for s in extra_mri_series:
        s["arr"] = np.flip(s["arr"], axis=1)
        s["diff_mri"] = np.flip(s["diff_mri"], axis=1)
        s["diff_pet"] = np.flip(s["diff_pet"], axis=1)
        s["edge_mri"] = np.flip(s["edge_mri"], axis=1)

    # Scene hierarchy
    scene_hierarchy = [
        {"name": "Fixed_CT_Volume_0", "type": "vtkMRMLScalarVolumeNode", "modality": "CT"},
        {"name": "SUV_PET_Image_0", "type": "vtkMRMLScalarVolumeNode", "modality": "PET"},
        {"name": "PET_Lesions_0", "type": "vtkMRMLSegmentationNode", "segments": [ct_seg_names.get(str(i), f"Lesion {i}") for i in ct_labels]},
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
                {"name": "PET_Lesions_1", "type": "vtkMRMLSegmentationNode", "segments": [mr_seg_names.get(str(i), f"Prostate Lesion {i}") for i in mr_labels]},
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
                {"name": f"PET_Lesions_{s['tp_idx']}", "type": "vtkMRMLSegmentationNode", "segments": [mr_seg_names.get(str(i), f"Prostate Lesion {i}") for i in mr_labels]},
                {"name": f"max_anatomy_fixed_ct_{s['tp_idx']}", "source": "max_anatomy.nii.gz", "type": "vtkMRMLSegmentationNode"}
            ]
        })

    metadata = [{
        "PatientID": "Pat51",
        "SubjectID": SUBJ_ID,
        "ExperimentID": EXP_ID,
        "Modalities": ["CT", "PET", "MRI"],
        "CreatedDate": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ReferenceSpacing": spacing,
        "ReferenceOrigin": origin
    }]

    # Write HDF5
    print(f"▶ Assembling HDF5 package: {OUT_H5}...")
    if os.path.isfile(OUT_H5):
        os.remove(OUT_H5)

    with h5py.File(OUT_H5, "w") as h5:
        # 1. BASELINE Group (TP 0)
        g_base = h5.create_group("BASELINE")
        ds_ct = g_base.create_dataset("Fixed_CT_Volume_0.nii.gz", data=ct_arr, compression="gzip", compression_opts=1)
        ds_ct.attrs["spacing"] = np.array(spacing, dtype=np.float64)
        ds_ct.attrs["origin"] = np.array(origin, dtype=np.float64)
        ds_ct.attrs["direction"] = np.array(direction, dtype=np.float64)

        ds_pet = g_base.create_dataset("SUV_PET_Image_0.nii.gz", data=pet_arr, compression="gzip", compression_opts=1)
        ds_pet.attrs["spacing"] = np.array(spacing, dtype=np.float64)

        ds_l0 = g_base.create_dataset("PET_Lesions_0.nii.gz", data=lesion_ct_arr, compression="gzip", compression_opts=1)
        ds_l0.attrs["spacing"] = np.array(spacing, dtype=np.float64)

        # Diffusivity & Edge derivations for BASELINE
        g_base.create_dataset("diffusivity", data=diff_comb, compression="gzip", compression_opts=1)
        g_base.create_dataset("diffusivity_ct", data=diff_ct, compression="gzip", compression_opts=1)
        g_base.create_dataset("diffusivity_pet", data=diff_pet, compression="gzip", compression_opts=1)
        g_base.create_dataset("edge_combined", data=edge_comb, compression="gzip", compression_opts=1)
        g_base.create_dataset("edge_ct", data=edge_ct, compression="gzip", compression_opts=1)
        g_base.create_dataset("edge_pet", data=edge_pet, compression="gzip", compression_opts=1)

        # Whole-body CT anatomy
        g_base.create_dataset("max_anatomy.nii.gz", data=max_anatomy, compression="gzip", compression_opts=1)
        g_base.create_dataset("skellytour.nii.gz", data=skellytour, compression="gzip", compression_opts=1)

        # 2. MRI Group (TP 1, T2)
        g_tfm1 = h5.create_group("TFM_Transform_FollowUp_to_Baseline_1.tfm")
        ds_mri1 = g_tfm1.create_dataset("Fixed_CT_Volume_1.nii.gz", data=t2_arr, compression="gzip", compression_opts=1)
        ds_mri1.attrs["spacing"] = np.array(spacing, dtype=np.float64)
        ds_mri1.attrs["modality"] = "T2"
        ds_mri1.attrs["series_description"] = "T2 MRI Registered"

        ds_pet1 = g_tfm1.create_dataset("SUV_PET_Image_1.nii.gz", data=pet_arr, compression="gzip", compression_opts=1)
        ds_pet1.attrs["spacing"] = np.array(spacing, dtype=np.float64)

        ds_l1 = g_tfm1.create_dataset("PET_Lesions_1.nii.gz", data=lesion_mr_arr, compression="gzip", compression_opts=1)
        ds_l1.attrs["spacing"] = np.array(spacing, dtype=np.float64)

        # Dedicated MRI 8-zone prostate anatomy
        g_tfm1.create_dataset("max_anatomy.nii.gz", data=mri_anatomy, compression="gzip", compression_opts=1)

        # Diffusivity & Edge derivations for MRI (treated as pure CT per user instruction)
        g_tfm1.create_dataset("diffusivity", data=diff_mri1, compression="gzip", compression_opts=1)
        g_tfm1.create_dataset("diffusivity_ct", data=diff_mri1, compression="gzip", compression_opts=1)
        g_tfm1.create_dataset("diffusivity_pet", data=diff_pet1, compression="gzip", compression_opts=1)
        g_tfm1.create_dataset("edge_mri", data=edge_mri1, compression="gzip", compression_opts=1)
        g_tfm1.create_dataset("edge_ct", data=edge_mri1, compression="gzip", compression_opts=1)
        g_tfm1.create_dataset("edge_pet", data=edge_pet1, compression="gzip", compression_opts=1)

        # 3. Additional MRI Groups (TP 2, 3, 4, 5)
        for s in extra_mri_series:
            tp_i = s["tp_idx"]
            g_extra = h5.create_group(f"TFM_Transform_FollowUp_to_Baseline_{tp_i}.tfm")
            ds_me = g_extra.create_dataset(f"Fixed_CT_Volume_{tp_i}.nii.gz", data=s["arr"], compression="gzip", compression_opts=1)
            ds_me.attrs["spacing"] = np.array(spacing, dtype=np.float64)
            ds_me.attrs["modality"] = s["mod"]
            ds_me.attrs["series_description"] = s["desc"]

            ds_pe = g_extra.create_dataset(f"SUV_PET_Image_{tp_i}.nii.gz", data=pet_arr, compression="gzip", compression_opts=1)
            ds_pe.attrs["spacing"] = np.array(spacing, dtype=np.float64)

            ds_le = g_extra.create_dataset(f"PET_Lesions_{tp_i}.nii.gz", data=lesion_mr_arr, compression="gzip", compression_opts=1)
            ds_le.attrs["spacing"] = np.array(spacing, dtype=np.float64)

            g_extra.create_dataset("max_anatomy.nii.gz", data=mri_anatomy, compression="gzip", compression_opts=1)

            g_extra.create_dataset("diffusivity", data=s["diff_mri"], compression="gzip", compression_opts=1)
            g_extra.create_dataset("diffusivity_ct", data=s["diff_mri"], compression="gzip", compression_opts=1)
            g_extra.create_dataset("diffusivity_pet", data=s["diff_pet"], compression="gzip", compression_opts=1)
            g_extra.create_dataset("edge_mri", data=s["edge_mri"], compression="gzip", compression_opts=1)
            g_extra.create_dataset("edge_ct", data=s["edge_mri"], compression="gzip", compression_opts=1)

        # 4. ATLAS Group
        g_atlas = h5.create_group("ATLAS")
        g_atlas.create_dataset("max_anatomy", data=max_anatomy, compression="gzip", compression_opts=1)
        g_atlas.create_dataset("bone_atlas", data=bone_atlas, compression="gzip", compression_opts=1)
        g_atlas.create_dataset("skellytour", data=skellytour, compression="gzip", compression_opts=1)

        # 5. CENTROIDS Group (1-based [X, Y, Z] indexing for MedEye3d navigation)
        g_cent = h5.create_group("CENTROIDS")
        for lid in ct_labels:
            pts = np.argwhere(lesion_ct_arr == lid)
            if len(pts) > 0:
                mean_pt = np.round(pts.mean(axis=0)).astype(np.int32)
                g_cent.create_dataset(f"tp0_lid{lid}", data=np.array([mean_pt[2]+1, mean_pt[1]+1, mean_pt[0]+1], dtype=np.int32))

        for lid in mr_labels:
            pts = np.argwhere(lesion_mr_arr == lid)
            if len(pts) > 0:
                mean_pt = np.round(pts.mean(axis=0)).astype(np.int32)
                g_cent.create_dataset(f"tp1_lid{lid}", data=np.array([mean_pt[2]+1, mean_pt[1]+1, mean_pt[0]+1], dtype=np.int32))
                for s in extra_mri_series:
                    g_cent.create_dataset(f"tp{s['tp_idx']}_lid{lid}", data=np.array([mean_pt[2]+1, mean_pt[1]+1, mean_pt[0]+1], dtype=np.int32))

        # Add LN centroids for fast jump navigation in LN mode!
        for aid_str, aname in anatomy_label_names.items():
            aid = int(aid_str)
            pts_ln = np.argwhere(max_anatomy == aid)
            if len(pts_ln) > 0:
                mean_pt = np.round(pts_ln.mean(axis=0)).astype(np.int32)
                g_cent.create_dataset(f"tp0_ln{aid}", data=np.array([mean_pt[2]+1, mean_pt[1]+1, mean_pt[0]+1], dtype=np.int32))

        # 6. _meta_ Group
        g_meta = h5.create_group("_meta_")
        g_meta.create_dataset("preflipped", data=1)
        g_meta.create_dataset("bone_label_ids", data=np.array(bone_ids, dtype=np.int32))
        g_meta.create_dataset("scene_hierarchy.json", data=json.dumps(scene_hierarchy, indent=2))
        g_meta.create_dataset("metadata.json", data=json.dumps(metadata, indent=2))
        g_meta.create_dataset("max_anatomy_labels.json", data=json.dumps(anatomy_label_names, indent=2))
        g_meta.create_dataset("prostate_anatomy_labels.json", data=json.dumps(prostate_labels, indent=2))
        g_meta.create_dataset("organ_mapping", data=json.dumps(organ_mapping, indent=2))
        g_meta.create_dataset("segment_names.json", data=json.dumps(segment_names, indent=2))
        g_meta.create_dataset("matches.json", data=json.dumps(matches, indent=2))

    sz_mb = os.path.getsize(OUT_H5) / (1024 * 1024)
    print(f"✅ Preprocessed HDF5 built successfully: {OUT_H5} ({sz_mb:.2f} MB)")
    return OUT_H5

# ==============================================================================
# 6. Upload to XNAT
# ==============================================================================

def upload_to_xnat(h5_path):
    print("▶ Uploading preprocessed HDF5 to XNAT MEDEYE3D resource...")
    sess = requests.Session()
    r = sess.get(f"{SERVER}/data/JSESSION", auth=(ALIAS, SECRET), timeout=20)
    r.raise_for_status()
    sess.cookies.set("JSESSIONID", r.text.strip())

    remote_filename = f"{SUBJ_LABEL}_preprocessed.h5"

    # 1. Ensure subject resource MEDEYE3D exists
    res_url = f"{SERVER}/data/projects/{PROJ_ID}/subjects/{SUBJ_ID}/resources/MEDEYE3D"
    sess.put(res_url)

    # 2. Upload file to subject resource
    upload_url = f"{SERVER}/data/projects/{PROJ_ID}/subjects/{SUBJ_ID}/resources/MEDEYE3D/files/{remote_filename}?overwrite=true"
    sz_mb = os.path.getsize(h5_path) / (1024 * 1024)
    print(f"  Uploading {remote_filename} ({sz_mb:.1f} MB) to {upload_url}...")

    with open(h5_path, "rb") as f:
        ru = sess.put(
            upload_url,
            data=f,
            headers={"Content-Type": "application/octet-stream"},
            timeout=600
        )
    if ru.status_code in (200, 201):
        print(f"✅ Successfully uploaded {remote_filename} to XNAT Subject MEDEYE3D resource (HTTP {ru.status_code})!")
    else:
        print(f"  [WARN] Subject upload returned status {ru.status_code}: {ru.text}")

    # 3. Also upload to Experiment level for direct experiment-based accession
    exp_res_url = f"{SERVER}/data/experiments/{EXP_ID}/resources/MEDEYE3D"
    sess.put(exp_res_url)
    exp_upload_url = f"{SERVER}/data/experiments/{EXP_ID}/resources/MEDEYE3D/files/{remote_filename}?overwrite=true"
    with open(h5_path, "rb") as f:
        re_up = sess.put(
            exp_upload_url,
            data=f,
            headers={"Content-Type": "application/octet-stream"},
            timeout=600
        )
    if re_up.status_code in (200, 201):
        print(f"✅ Successfully uploaded {remote_filename} to XNAT Experiment MEDEYE3D resource (HTTP {re_up.status_code})!")
    else:
        print(f"  [WARN] Experiment upload returned status {re_up.status_code}: {re_up.text}")

def main():
    h5_path = package_hdf5()
    try:
        upload_to_xnat(h5_path)
    except Exception as e:
        print(f"  [XNAT UPLOAD NOTICE] XNAT upload skipped or deferred: {e}")
    print("\n🎉 ALL PIPELINE STEPS COMPLETED!")

if __name__ == "__main__":
    main()
