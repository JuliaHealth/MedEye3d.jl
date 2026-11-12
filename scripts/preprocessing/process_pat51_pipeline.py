#!/usr/bin/env python3
"""
process_pat51_pipeline.py

End-to-End processing pipeline for Pat51 (Prostata_bimodal):
1. Loads CT, PET SUV, raw moving MRI (T2, ADC, DWI, T1, DCE), and prostate segmentation.
2. Performs two-stage pelvic & prostate registration: Stage 1 initial rigid alignment based on medial femurs & hip bones (Arun SVD), followed by Stage 2 Sliced Wasserstein Distance (SWD) optimization strictly on the prostate.
3. Resamples all moving MRI series and dedicated 8-zone prostate segmentation onto the CT grid.
4. Layers CT whole-body anatomy with fresh TotalSegmentator v2 base anatomy and fresh lymph nodes from final_results_gpu.h5.
5. Strictly separates MRI 8-zone prostate anatomy (prostate_anatomy_labels.json) with NO artificial MRI lesions.
6. Computes neural 3D edge derivations (Dual PET+CT, Pure CT, Pure PET) using champion TotalSegEdgePDE models on cuda:1.
7. Pre-flips along axis 1 (Y) with preflipped=1 and calculates 1-based centroids (including LN centroids).
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
# 1. Registration: Two-Stage (Medial Femurs & Hip Bones Init -> Prostate SWD Refinement)
# ==============================================================================

def compute_bone_init_prostate_swd_registration(device="cuda:1"):
    """
    Two-stage Pelvis & Prostate Registration:
    Stage 1: Initial rigid alignment based on anatomical pelvic landmarks:
             medial femurs and hip bones from NVIDIA VISTA (packed_anatomical_tensor.h5)
             or TotalSegmentator fallback. Uses Arun et al. 1987 SVD closed-form rigid alignment.
    Stage 2: Rigid refinement via Sliced Wasserstein Distance (SWD) strictly on the prostate
             using PyTorch on GPU.
    """
    import torch
    import torch.nn as nn

    print("▶ Stage 1: Pelvic Landmark Alignment (Medial Femurs & Hip Bones)...")
    ct_img = sitk.ReadImage(os.path.join(RAW_DIR, "Fixed_CT_Volume.nii.gz"))
    ts_mr_img = sitk.ReadImage(os.path.join(RAW_DIR, "totalseg_mr_ml.nii.gz"))
    mr_arr = sitk.GetArrayFromImage(ts_mr_img)

    def get_voxel_physical_pts(img, mask):
        idx = np.argwhere(mask)
        pts = [img.TransformIndexToPhysicalPoint((int(c[2]), int(c[1]), int(c[0]))) for c in idx]
        return np.array(pts)

    def get_medial_centroid(pts, pct=0.02):
        k = max(10, int(len(pts) * pct))
        return pts[np.argsort(np.abs(pts[:, 0]))[:k]].mean(axis=0)

    # Check if XNAT packed_anatomical_tensor.h5 is present (NVIDIA VISTA models)
    anat_h5_path = os.path.join(RAW_DIR, "packed_anatomical_tensor.h5")
    reg_json_path = os.path.join(RAW_DIR, "packed_anatomical_tensor_registry.json")

    if os.path.isfile(anat_h5_path) and os.path.isfile(reg_json_path):
        print("  [STAGE 1] Loading pelvic landmarks from XNAT NVIDIA VISTA packed_anatomical_tensor.h5...")
        with open(reg_json_path, "r") as rf:
            reg_doc = json.load(rf)
        reg = reg_doc.get("masks", reg_doc)
        with h5py.File(anat_h5_path, "r") as af:
            pt = af["packed_tensor"][:]

        def get_pt_mask(name):
            meta = reg[name]
            ch = meta["channel"] - 1
            mid = meta["id"]
            return (pt[ch] == mid)

        fl_ct = get_voxel_physical_pts(ct_img, get_pt_mask("nvseg_left_femur"))
        fr_ct = get_voxel_physical_pts(ct_img, get_pt_mask("nvseg_right_femur"))
        hl_ct = get_voxel_physical_pts(ct_img, get_pt_mask("nvseg_left_hip"))
        hr_ct = get_voxel_physical_pts(ct_img, get_pt_mask("nvseg_right_hip"))
        p_ct_vox = get_voxel_physical_pts(ct_img, get_pt_mask("nvseg_prostate"))
    else:
        print("  [STAGE 1] Falling back to totalseg_ct_fast.nii.gz...")
        ts_ct_img = sitk.ReadImage(os.path.join(RAW_DIR, "totalseg_ct_fast.nii.gz"))
        ct_arr = sitk.GetArrayFromImage(ts_ct_img)
        hl_ct = get_voxel_physical_pts(ts_ct_img, ct_arr == 77)
        hr_ct = get_voxel_physical_pts(ts_ct_img, ct_arr == 78)
        fl_ct = get_voxel_physical_pts(ts_ct_img, ct_arr == 75)
        fr_ct = get_voxel_physical_pts(ts_ct_img, ct_arr == 76)
        p_ct_vox = get_voxel_physical_pts(ts_ct_img, ct_arr == 22)

    hl_mr = get_voxel_physical_pts(ts_mr_img, mr_arr == 38)
    hr_mr = get_voxel_physical_pts(ts_mr_img, mr_arr == 39)
    fl_mr = get_voxel_physical_pts(ts_mr_img, mr_arr == 36)
    fr_mr = get_voxel_physical_pts(ts_mr_img, mr_arr == 37)

    ct_landmarks = np.array([
        get_medial_centroid(fl_ct),
        get_medial_centroid(fr_ct),
        get_medial_centroid(hl_ct),
        get_medial_centroid(hr_ct),
    ])
    mr_landmarks = np.array([
        get_medial_centroid(fl_mr),
        get_medial_centroid(fr_mr),
        get_medial_centroid(hl_mr),
        get_medial_centroid(hr_mr),
    ])

    c_mr = mr_landmarks.mean(axis=0)
    c_ct = ct_landmarks.mean(axis=0)

    H = (mr_landmarks - c_mr).T @ (ct_landmarks - c_ct)
    U, S, Vt = np.linalg.svd(H)
    R_init = Vt.T @ U.T
    if np.linalg.det(R_init) < 0:
        Vt[2, :] *= -1
        R_init = Vt.T @ U.T
    t_init = c_ct - R_init @ c_mr

    pred_init = mr_landmarks @ R_init.T + t_init
    rmse_init = np.sqrt(np.mean(np.sum((ct_landmarks - pred_init)**2, axis=1)))
    print(f"  [STAGE 1] Pelvic bones rigid fit RMSE: {rmse_init:.3f} mm")
    print(f"  [STAGE 1] R_init:\n{R_init}")
    print(f"  [STAGE 1] t_init: {t_init}")

    print("▶ Stage 2: Sliced Wasserstein Distance (SWD) Optimization on Prostate...")
    p_mr_vox = get_voxel_physical_pts(ts_mr_img, mr_arr == 17)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    p_ct_t = torch.tensor(p_ct_vox, dtype=torch.float32, device=dev)
    p_mr_t = torch.tensor(p_mr_vox, dtype=torch.float32, device=dev)

    # Transform MR prostate with Stage 1 initial bone alignment
    p_mr_init_t = p_mr_t @ torch.tensor(R_init.T, dtype=torch.float32, device=dev) + torch.tensor(t_init, dtype=torch.float32, device=dev)

    center = torch.tensor(c_ct, dtype=torch.float32, device=dev)
    rot_angles = nn.Parameter(torch.zeros(3, device=device))
    delta_trans = nn.Parameter(torch.zeros(3, device=device))

    def build_R(rx, ry, rz):
        cx, sx = torch.cos(rx), torch.sin(rx)
        cy, sy = torch.cos(ry), torch.sin(ry)
        cz, sz = torch.cos(rz), torch.sin(rz)
        Rx = torch.stack([
            torch.stack([torch.tensor(1.0, device=rx.device), torch.tensor(0.0, device=rx.device), torch.tensor(0.0, device=rx.device)]),
            torch.stack([torch.tensor(0.0, device=rx.device), cx, -sx]),
            torch.stack([torch.tensor(0.0, device=rx.device), sx, cx])
        ])
        Ry = torch.stack([
            torch.stack([cy, torch.tensor(0.0, device=ry.device), sy]),
            torch.stack([torch.tensor(0.0, device=ry.device), torch.tensor(1.0, device=ry.device), torch.tensor(0.0, device=ry.device)]),
            torch.stack([-sy, torch.tensor(0.0, device=ry.device), cy])
        ])
        Rz = torch.stack([
            torch.stack([cz, -sz, torch.tensor(0.0, device=rz.device)]),
            torch.stack([sz, cz, torch.tensor(0.0, device=rz.device)]),
            torch.stack([torch.tensor(0.0, device=rz.device), torch.tensor(0.0, device=rz.device), torch.tensor(1.0, device=rz.device)])
        ])
        return Rz @ Ry @ Rx

    def swd_loss(p1, p2, num_projections=256):
        projections = torch.randn(num_projections, 3, device=p1.device)
        projections = projections / torch.norm(projections, dim=1, keepdim=True)
        proj1 = p1 @ projections.T
        proj2 = p2 @ projections.T
        s1, _ = torch.sort(proj1, dim=0)
        s2, _ = torch.sort(proj2, dim=0)
        K = 4096
        s1_interp = torch.nn.functional.interpolate(s1.T.unsqueeze(1), size=K, mode='linear', align_corners=True)
        s2_interp = torch.nn.functional.interpolate(s2.T.unsqueeze(1), size=K, mode='linear', align_corners=True)
        return torch.mean((s1_interp - s2_interp)**2)

    optimizer = torch.optim.Adam([
        {'params': [rot_angles], 'lr': 1e-3},
        {'params': [delta_trans], 'lr': 0.1}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-4)

    init_swd = swd_loss(p_mr_init_t, p_ct_t).item()
    print(f"  [STAGE 2] Initial SWD Loss (after bone alignment): {init_swd:.4f}")

    for epoch in range(1, 501):
        optimizer.zero_grad()
        R_ref = build_R(rot_angles[0], rot_angles[1], rot_angles[2])
        p_transformed = (p_mr_init_t - center) @ R_ref.T + center + delta_trans
        loss = swd_loss(p_transformed, p_ct_t)
        loss.backward()
        optimizer.step()
        scheduler.step()

    print(f"  [STAGE 2] Final SWD Loss (after prostate refinement): {loss.item():.4f}")

    R_ref_np = build_R(rot_angles[0], rot_angles[1], rot_angles[2]).detach().cpu().numpy()
    d_trans_np = delta_trans.detach().cpu().numpy()
    c_ct_np = c_ct

    R_total_m2c = R_ref_np @ R_init
    t_total_m2c = R_ref_np @ (t_init - c_ct_np) + c_ct_np + d_trans_np

    # Invert to map CT -> MR (Fixed -> Moving) for SimpleITK resampling
    R_c2m = R_total_m2c.T
    t_c2m = - R_c2m @ t_total_m2c

    itk_tfm = sitk.AffineTransform(3)
    itk_tfm.SetMatrix(R_c2m.flatten().tolist())
    itk_tfm.SetTranslation(t_c2m.tolist())

    # Evaluate Prostate Dice
    mr_p_img = sitk.GetImageFromArray((mr_arr == 17).astype(np.uint8))
    mr_p_img.CopyInformation(ts_mr_img)
    res_mr_p = sitk.Resample(mr_p_img, ct_img, itk_tfm, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    res_mr_arr = sitk.GetArrayFromImage(res_mr_p)
    if 'get_pt_mask' in locals():
        ct_p_arr = get_pt_mask("nvseg_prostate").astype(np.uint8)
    else:
        ct_p_arr = (ct_arr == 22).astype(np.uint8)

    inter = (ct_p_arr > 0) & (res_mr_arr > 0)
    dice = 2.0 * inter.sum() / (ct_p_arr.sum() + res_mr_arr.sum())
    print(f"  [REGISTRATION EVAL] Prostate Dice: {dice:.4f} (intersection: {inter.sum()} voxels)")

    tfm_path = os.path.join(CASE_DIR, "Transform_FollowUp_to_Baseline_1.tfm")
    sitk.WriteTransform(itk_tfm, tfm_path)
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

        mri_gold_lesion = (p_arr == 9).astype(np.uint8)
        if mri_gold_lesion.any():
            print(f"  [GOLD STANDARD MR LESION] Found {mri_gold_lesion.sum()} voxels of prostate lesion (Label 9) on MRI")
        else:
            mri_gold_lesion = None
    else:
        mri_gold_lesion = None

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

    return t2_arr, mri_anatomy, extra_series, mri_gold_lesion

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

    bone_terms = ['femur', 'hip', 'vertebra', 'rib', 'sacrum', 'clavicula', 'humerus', 'scapula', 'sternum', 'skull', 'costal', 'mandible']
    muscle_terms = ['gluteus', 'iliopsoas', 'autochthon', 'pectoralis', 'latissimus', 'trapezius', 'deltoid', 'muscle', 'adductor']

    def get_layer(name):
        n = name.lower()
        if any(b in n for b in bone_terms): return 2
        if any(m in n for m in muscle_terms): return 0
        return 1

    # Prefer XNAT NVIDIA VISTA packed_anatomical_tensor.h5 directly (eliminating need to rerun anatomical models)
    anat_h5_path = os.path.join(RAW_DIR, "packed_anatomical_tensor.h5")
    reg_json_path = os.path.join(RAW_DIR, "packed_anatomical_tensor_registry.json")

    lbl_idx = 1
    if os.path.isfile(anat_h5_path) and os.path.isfile(reg_json_path):
        print(f"  [ANATOMY] Unpacking NVIDIA VISTA segmentations from XNAT {os.path.basename(anat_h5_path)}...")
        with open(reg_json_path, "r") as rf:
            reg_doc = json.load(rf)
        registry = reg_doc.get("masks", reg_doc)

        with h5py.File(anat_h5_path, "r") as af:
            packed_tensor = af["packed_tensor"][:]
            items_by_layer = {0: [], 1: [], 2: []}
            for name, meta in registry.items():
                if not isinstance(meta, dict) or meta.get("id", 0) <= 0:
                    continue
                clean = name.replace("nvseg_", "").lower()
                if "temp" in clean or clean == "temp":
                    continue
                if any(h in clean for h in ["helper_", "_base", "_dilated", "_bridge", "_exclusion"]):
                    continue
                lay = get_layer(clean)
                items_by_layer[lay].append((name, clean, meta))

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
                        if any(h in k.lower() for h in ["_base", "_helper", "_dilated", "_exclusion_", "_bridge", "helper_", "temp"]):
                            continue
                        arr_ln = lf[k][:]
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

    else:
        # Fallback to TotalSegmentator if packed_tensor is not available
        ts_path = os.path.join(RAW_DIR, "totalseg_ct_fast.nii.gz")
        ts_map_path = "/workspaces/MedEye3d.jl/data/totalseg_class_map.json"
        if os.path.isfile(ts_path) and os.path.isfile(ts_map_path):
            print(f"  [ANATOMY] Loading TotalSegmentator base anatomy from {os.path.basename(ts_path)}...")
            ts_img = sitk.ReadImage(ts_path)
            ts_arr = sitk.GetArrayFromImage(ts_img)
            with open(ts_map_path, "r") as mf:
                ts_map = json.load(mf)

            items_by_layer = {0: [], 1: [], 2: []}
            for cid_str, cname in ts_map.items():
                cid = int(cid_str)
                clean = cname.lower()
                lay = get_layer(clean)
                items_by_layer[lay].append((cid, clean))

            for lay in [0, 1]:
                for cid, clean in sorted(items_by_layer[lay], key=lambda x: x[1]):
                    mask_i = (ts_arr == cid)
                    if mask_i.any():
                        anatomy_label_names[str(lbl_idx)] = clean
                        max_anatomy[mask_i] = lbl_idx
                        if any(t in clean for t in ["urinary_bladder", "bladder", "parotid", "submandibular", "lacrimal", "kidney"]):
                            exclusion_mask[mask_i] = 1
                        lbl_idx += 1

            ln_h5_path = os.path.join(RAW_DIR, "final_results_gpu.h5")
            if os.path.isfile(ln_h5_path):
                print(f"  [ANATOMY] Layering freshly recomputed lymph node stations from {os.path.basename(ln_h5_path)}...")
                with h5py.File(ln_h5_path, "r") as lf:
                    for k in sorted(lf.keys()):
                        if any(h in k.lower() for h in ["_base", "_helper", "_dilated", "_exclusion_", "_bridge", "helper_", "temp"]):
                            continue
                        arr_ln = lf[k][:]
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

            for cid, clean in sorted(items_by_layer[2], key=lambda x: x[1]):
                mask_i = (ts_arr == cid)
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
# 4. Neural Edge Derivations & Heat-GDT Diffusivity (EDGE_EXTRACTION_GUIDE.md)
# ==============================================================================

def compute_neural_edge_derivations(ct_arr, pet_arr, exclusion_mask, device="cuda:1"):
    """
    Computes 3D neural edge probability maps E(x) ∈ [0, 1] and corresponding
    Heat-GDT diffusivities D(x) using champion checkpoints from EDGE_EXTRACTION_GUIDE.md:
      - 'ct_only': TotalSegEdgePDE with zero-PET fallback
      - 'pet_only': TotalSegEdgePDE_PET single-channel model
      - 'combined': TotalSegEdgePDE dual-channel (PET+CT)
    """
    import sys
    sys.path.insert(0, '/mnt/big/project_ssd/semiautomatic')
    import torch
    from totalseg_edge_pde_model import TotalSegEdgePDE
    from totalseg_edge_pde_pet_model import TotalSegEdgePDE_PET

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"▶ Computing neural 3D edge derivations on {dev} (TotalSegEdgePDE & TotalSegEdgePDE_PET)...")

    ckpt_dual_path = "/mnt/big/project_ssd/semiautomatic/checkpoints/totalseg_edge_pde/best_model.pth"
    m_dual = TotalSegEdgePDE().to(dev)
    ckpt_dual = torch.load(ckpt_dual_path, map_location=dev)
    m_dual.load_state_dict(ckpt_dual.get("model_state_dict", ckpt_dual))
    m_dual.eval()

    ckpt_pet_path = "/mnt/big/project_ssd/semiautomatic/checkpoints/totalseg_edge_pde_pet/best_model.pth"
    m_pet = TotalSegEdgePDE_PET().to(dev)
    ckpt_pet = torch.load(ckpt_pet_path, map_location=dev)
    m_pet.load_state_dict(ckpt_pet.get("model_state_dict", ckpt_pet))
    m_pet.eval()

    ct_norm = (np.clip(ct_arr, -1024.0, 1500.0) - (-200.0)) / 400.0
    pet_norm = np.clip(pet_arr, 0.0, 20.0) / 20.0

    def run_batched_sliding_window(model, inp_tensor, patch_size=64, step=48, batch_size=16):
        _, _, D, H, W = inp_tensor.shape
        out_edge = torch.zeros((D, H, W), dtype=torch.float32, device=dev)
        weight_map = torch.zeros((D, H, W), dtype=torch.float32, device=dev)

        z = np.sin(np.linspace(0, np.pi, patch_size))
        y = np.sin(np.linspace(0, np.pi, patch_size))
        x = np.sin(np.linspace(0, np.pi, patch_size))
        w3d = torch.from_numpy((z[:, None, None] * y[None, :, None] * x[None, None, :]).astype(np.float32)).to(dev)
        w3d = torch.clamp(w3d, min=1e-3)

        d_coords = list(range(0, max(1, D - patch_size + 1), step))
        if d_coords[-1] + patch_size < D: d_coords.append(D - patch_size)
        h_coords = list(range(0, max(1, H - patch_size + 1), step))
        if h_coords[-1] + patch_size < H: h_coords.append(H - patch_size)
        w_coords = list(range(0, max(1, W - patch_size + 1), step))
        if w_coords[-1] + patch_size < W: w_coords.append(W - patch_size)

        patches = [(d, h, w) for d in d_coords for h in h_coords for w in w_coords]

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
        return torch.clamp(out_edge, 0.0, 1.0).cpu().numpy().astype(np.float32)

    # 1. Combined (PET+CT)
    print("  [1/3] Running TotalSegEdgePDE (Dual PET+CT)...")
    inp_dual = torch.from_numpy(np.stack([ct_norm, pet_norm], axis=0)).unsqueeze(0).to(dev)
    edge_comb = run_batched_sliding_window(m_dual, inp_dual)

    # 2. Pure CT
    print("  [2/3] Running TotalSegEdgePDE (Pure CT with Zero-PET)...")
    inp_ct = torch.from_numpy(np.stack([ct_norm, np.zeros_like(pet_norm)], axis=0)).unsqueeze(0).to(dev)
    edge_ct = run_batched_sliding_window(m_dual, inp_ct)

    # 3. Pure PET
    print("  [3/3] Running TotalSegEdgePDE_PET (Pure Single-Channel PET)...")
    inp_pet = torch.from_numpy(pet_norm).unsqueeze(0).unsqueeze(0).to(dev)
    edge_pet = run_batched_sliding_window(m_pet, inp_pet)

    del m_dual, m_pet, inp_dual, inp_ct, inp_pet
    torch.cuda.empty_cache()

    # Convert to diffusivities per EDGE_EXTRACTION_GUIDE.md:
    diff_comb = np.clip(CHAMPION_EPS_D + (1.0 - CHAMPION_EPS_D) * np.exp(-CHAMPION_GAMMA * edge_comb), CHAMPION_EPS_D, 1.0).astype(np.float32)
    diff_ct = np.clip(CHAMPION_EPS_D + (1.0 - CHAMPION_EPS_D) * np.exp(-CHAMPION_GAMMA * edge_ct), CHAMPION_EPS_D, 1.0).astype(np.float32)
    diff_pet = np.clip(CHAMPION_EPS_D + (1.0 - CHAMPION_EPS_D) * np.exp(-9.4069 * edge_pet), CHAMPION_EPS_D, 1.0).astype(np.float32)

    # Physiological insulation
    if exclusion_mask is not None:
        diff_comb[exclusion_mask > 0] = 0.0
        diff_ct[exclusion_mask > 0] = 0.0
        diff_pet[exclusion_mask > 0] = 0.0

    print(f"  ✓ Edges and Diffusivities successfully computed:")
    print(f"    - edge_comb: min={edge_comb.min():.4f}, max={edge_comb.max():.4f}, mean={edge_comb.mean():.4f}")
    print(f"    - edge_ct:   min={edge_ct.min():.4f}, max={edge_ct.max():.4f}, mean={edge_ct.mean():.4f}")
    print(f"    - edge_pet:  min={edge_pet.min():.4f}, max={edge_pet.max():.4f}, mean={edge_pet.mean():.4f}")

    return edge_comb, edge_ct, edge_pet, diff_comb, diff_ct, diff_pet

def compute_mri_edge_derivations(mri_arr, pet_arr, mri_anatomy, exclusion_mask):
    """
    Computes 3D boundary edge and diffusivity for MRI modalities.
    User instruction: treat MRI as pure CT.
    Boundary comes from MRI gradient + dedicated 8-zone prostate capsule.
    """
    print("▶ Computing MRI edge derivations and Heat-GDT diffusivity...")
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

    D_mri = CHAMPION_EPS_D + (1.0 - CHAMPION_EPS_D) * np.exp(-CHAMPION_GAMMA * edge_mri)
    D_mri = np.clip(D_mri, CHAMPION_EPS_D, 1.0).astype(np.float32)
    if exclusion_mask is not None:
        D_mri[exclusion_mask > 0] = 0.0

    # PET overlay diffusivity
    pet_norm = np.clip(pet_arr, 0.0, 20.0) / 20.0
    gz_pet, gy_pet, gx_pet = np.gradient(pet_norm)
    mag_pet = gaussian_filter(np.sqrt(gz_pet**2 + gy_pet**2 + gx_pet**2), sigma=0.8)
    e_pet_raw = 1.0 / (1.0 + np.exp(-6.0 * (mag_pet - 0.10)))
    e_pet_zero = 1.0 / (1.0 + np.exp(-6.0 * (-0.10)))
    edge_pet = np.clip((e_pet_raw - e_pet_zero) / (1.0 - e_pet_zero), 0.0, 1.0).astype(np.float32)
    D_pet = CHAMPION_EPS_D + (1.0 - CHAMPION_EPS_D) * np.exp(-9.4069 * edge_pet)
    D_pet = np.clip(D_pet, CHAMPION_EPS_D, 1.0).astype(np.float32)
    if exclusion_mask is not None:
        D_pet[exclusion_mask > 0] = 0.0

    return edge_mri, edge_pet, D_mri, D_pet

# ==============================================================================
# 5. Full Packaging into preprocessed_volumes.h5
# ==============================================================================

def package_hdf5():
    # 1. Registration (Stage 1: Medial femurs & hip bones -> Stage 2: Prostate SWD refinement)
    tfm_path = compute_bone_init_prostate_swd_registration(device="cuda:1")

    # 2. Geometry & Volumes
    ct_img = sitk.ReadImage(os.path.join(RAW_DIR, "Fixed_CT_Volume.nii.gz"))
    spacing = list(ct_img.GetSpacing())
    origin = list(ct_img.GetOrigin())
    direction = list(ct_img.GetDirection())
    ct_arr = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    Z, Y, X = ct_arr.shape

    # PET SUV (Attenuation-Corrected) via scripts/suvs.py resampled to CT grid
    corr_suv_path = os.path.join(RAW_DIR, "SUV_PET_WB_120min_Corrected.nii.gz")
    dcm_dir = "/mnt/big/project_ssd/data/xnat/FDM_DPI-2024-7-KRN_Lu177_PSMA__PETPSMA_0__Pat51/scans/3-PET_WB120min/resources/DICOM/files"
    uncorr_path = os.path.join(RAW_DIR, "SUV_PET_WB_120min_Uncorrected.nii.gz")
    uncorr_img = sitk.ReadImage(uncorr_path)

    if not os.path.exists(corr_suv_path) and os.path.exists(dcm_dir):
        import glob
        import pydicom
        sys.path.insert(0, '/workspaces/MedEye3d.jl/scripts')
        import suvs

        print(f"▶ Computing attenuation-corrected SUV using scripts/suvs.py from DICOM {dcm_dir}...")
        dcm_paths = glob.glob(os.path.join(dcm_dir, "*.dcm"))
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(dcm_dir))
        base_img = reader.Execute()
        dicom_files = [{'file_path': p, 'ds': pydicom.dcmread(p)} for p in dcm_paths]
        suvs.validate_pet_dicom_tags(dicom_files)
        suv_img = suvs.apply_suv_correction(dicom_files, base_img)
        suv_img.CopyInformation(uncorr_img)
        sitk.WriteImage(suv_img, corr_suv_path)
    else:
        print(f"▶ Loading corrected SUV from {corr_suv_path}...")
        suv_img = sitk.ReadImage(corr_suv_path)
        suv_img.CopyInformation(uncorr_img)

    res_suv = sitk.Resample(suv_img, ct_img, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    pet_arr = sitk.GetArrayFromImage(res_suv).astype(np.float32)
    print(f"  [PET SUV AC] shape={pet_arr.shape}, min={pet_arr.min():.2f}, max={pet_arr.max():.2f}, mean={pet_arr.mean():.4f}, p99={np.percentile(pet_arr, 99):.2f}")

    # MRI sequences resampled
    t2_arr, mri_anatomy, extra_mri_series, mri_gold_lesion = resample_mri_modalities(ct_img, tfm_path)

    # CT Whole-Body anatomy with fresh TotalSegmentator v2 + LN stations
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

    # Edges & Diffusivity for Baseline (CT + PET) using champion neural models
    edge_comb, edge_ct, edge_pet, diff_comb, diff_ct, diff_pet = compute_neural_edge_derivations(ct_arr, pet_arr, exclusion_mask, device="cuda:1")

    # Edges & Diffusivity for MRI (T2)
    edge_mri1, edge_pet1, diff_mri1, diff_pet1 = compute_mri_edge_derivations(t2_arr, pet_arr, mri_anatomy, exclusion_mask)

    # Edges & Diffusivity for additional MRI series
    for s in extra_mri_series:
        e_m, e_p, d_m, d_p = compute_mri_edge_derivations(s["arr"], pet_arr, mri_anatomy, exclusion_mask)
        s["edge_mri"] = e_m
        s["diff_mri"] = d_m
        s["diff_pet"] = d_p

    # Lesions: Real expert segmentations from XNAT (slicer_final) on CT
    ct_nrrd_path = os.path.join(RAW_DIR, "NNUNET_OR_Processed_Lesions_CT.seg.nrrd")
    ct_nrrd = sitk.ReadImage(ct_nrrd_path)
    res_ct_nrrd = sitk.Resample(ct_nrrd, ct_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkInt16)
    lesion_ct_arr = sitk.GetArrayFromImage(res_ct_nrrd).astype(np.int16)

    # MRI Lesions: strictly gold standard MRI segmentation only (no mapped CT lesions!)
    if mri_gold_lesion is not None and mri_gold_lesion.any():
        lesion_mr_arr = mri_gold_lesion.astype(np.int16)
        mr_labels = [1]
        mr_seg_names = {"1": "Prostate_Lesion_MR"}
        print(f"  [LESIONS] Embedded gold standard MRI prostate lesion: {mri_gold_lesion.sum()} voxels")
    else:
        lesion_mr_arr = np.zeros_like(lesion_ct_arr, dtype=np.int16)
        mr_labels = []
        mr_seg_names = {}
        print("  [LESIONS] No gold standard MRI lesions found in XNAT/local data (zero mapped lesions on MRI).")

    ct_labels = sorted([int(k) for k in np.unique(lesion_ct_arr) if k > 0])
    print(f"  [LESIONS] CT unique labels: {ct_labels} (total voxels: {(lesion_ct_arr > 0).sum()})")
    print(f"  [LESIONS] MR unique labels: {mr_labels} (gold standard MRI segmentations)")

    # Read segment names from NRRD metadata
    ct_seg_names = {}
    for i in range(25):
        k_name = f"Segment{i}_Name"
        k_val = f"Segment{i}_LabelValue"
        if ct_nrrd.HasMetaDataKey(k_name) and ct_nrrd.HasMetaDataKey(k_val):
            val = str(int(ct_nrrd.GetMetaData(k_val)))
            name = ct_nrrd.GetMetaData(k_name)
            ct_seg_names[val] = name

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

    all_mr_tps = [1] + [s["tp_idx"] for s in extra_mri_series]
    organ_mapping = {"0": ct_organs}
    segment_names = {"0": ct_seg_names}
    for tp in all_mr_tps:
        organ_mapping[str(tp)] = {"1": "Prostate"} if mr_labels else {}
        segment_names[str(tp)] = mr_seg_names if mr_labels else {}

    # Cross-timepoint matches.json:
    matches = []
    best_overlap_ct_lid = None
    best_overlap_count = 0
    if mr_labels:
        for lid in ct_labels:
            inter = ((lesion_ct_arr == lid) & (lesion_mr_arr == 1)).sum()
            if inter > best_overlap_count:
                best_overlap_count = inter
                best_overlap_ct_lid = lid
        if best_overlap_ct_lid is not None:
            print(f"  [MATCHES] Linked MR Prostate Lesion to CT Lesion {best_overlap_ct_lid} ({best_overlap_count} voxels overlap)")

    for lid in ct_labels:
        children = []
        if mr_labels and lid == best_overlap_ct_lid:
            children.append({
                "name": "PET_Lesions_1",
                "raw_lesion": "Segment_1",
                "node": "child",
                "group_id": lid,
                "lesion": "Prostate_Lesion_MR"
            })
        matches.append({
            "name": "PET_Lesions_0",
            "raw_lesion": f"Segment_{lid}",
            "node": "parent",
            "group_id": lid,
            "lesion": ct_seg_names.get(str(lid), f"Lesion {lid}"),
            "children": children
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
                {"name": "PET_Lesions_1", "type": "vtkMRMLSegmentationNode", "segments": list(mr_seg_names.values()) if mr_labels else []},
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
                {"name": f"PET_Lesions_{s['tp_idx']}", "type": "vtkMRMLSegmentationNode", "segments": []},
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

    with h5py.File(OUT_H5, "w", libver="v110") as h5:
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

        # Add MRI gold standard lesion centroid
        if mr_labels:
            pts_mr = np.argwhere(lesion_mr_arr == 1)
            if len(pts_mr) > 0:
                mean_pt = np.round(pts_mr.mean(axis=0)).astype(np.int32)
                g_cent.create_dataset("tp1_lid1", data=np.array([mean_pt[2]+1, mean_pt[1]+1, mean_pt[0]+1], dtype=np.int32))
                print(f"  [CENTROID] Added tp1_lid1 (MRI Gold Standard Prostate Lesion): {[mean_pt[2]+1, mean_pt[1]+1, mean_pt[0]+1]}")

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

    # 2. Upload file
    upload_url = f"{res_url}/files/{remote_filename}?overwrite=true"
    print(f"  Uploading {h5_path} ({os.path.getsize(h5_path)/(1024*1024):.1f} MB) -> {upload_url}...")
    with open(h5_path, "rb") as f:
        resp = sess.put(upload_url, data=f, timeout=600)
    if resp.status_code in (200, 201):
        print(f"  ✓ Successfully uploaded to Subject resource: {resp.status_code}")
    else:
        print(f"  [WARN] Subject upload returned status {resp.status_code}: {resp.text[:200]}")

    # 3. Ensure experiment resource MEDEYE3D exists
    exp_res_url = f"{SERVER}/data/projects/{PROJ_ID}/subjects/{SUBJ_ID}/experiments/{EXP_ID}/resources/MEDEYE3D"
    sess.put(exp_res_url)
    exp_upload_url = f"{exp_res_url}/files/{remote_filename}?overwrite=true"
    print(f"  Uploading to Experiment resource -> {exp_upload_url}...")
    with open(h5_path, "rb") as f:
        resp2 = sess.put(exp_upload_url, data=f, timeout=600)
    if resp2.status_code in (200, 201):
        print(f"  ✓ Successfully uploaded to Experiment resource: {resp2.status_code}")
    else:
        print(f"  [WARN] Experiment upload returned status {resp2.status_code}: {resp2.text[:200]}")

if __name__ == "__main__":
    t_start = time.time()
    h5_path = package_hdf5()
    try:
        upload_to_xnat(h5_path)
    except Exception as e:
        print(f"  [WARN] XNAT upload skipped or token expired: {e}")
    print(f"✨ Pat51 Full End-to-End Pipeline finished in {time.time() - t_start:.1f}s")
