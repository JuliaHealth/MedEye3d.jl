#!/usr/bin/env python3
"""
landmark_registration.py — Landmark-based rigid registration via SVD.

Extracts anatomical landmarks (extremal points) from TotalSegmentator masks
(sacrum, hip bones, prostate) and computes the MSE-optimal rigid transform
using Arun et al. 1987 SVD closed-form solution.

Usage:
    python3 landmark_registration.py \
        --fixed-labels-dir anatomy_out_fixed_ct_0/ \
        --moving-labels-dir anatomy_out_fixed_ct_3/ \
        --fixed-ref Fixed_CT_Volume_0.nii.gz \
        --moving-ref Fixed_CT_Volume_3.nii.gz \
        --output-tfm Transform_FollowUp_to_Baseline_3.tfm

Or run the full pipeline for all timepoints:
    python3 landmark_registration.py --case-dir data/cases/psma_patient_all_tp
"""

import argparse
import os
import sys
import shutil
import numpy as np

try:
    import nibabel as nib
except ImportError:
    print("ERROR: nibabel required. Install with: pip install nibabel")
    sys.exit(1)


def load_mask(path):
    """Load a NIfTI mask, return (data, affine) or (None, None) if missing/empty."""
    if not os.path.isfile(path):
        return None, None
    img = nib.load(path)
    data = np.asanyarray(img.dataobj)
    if not np.any(data > 0):
        return None, None
    return data, img.affine


def voxel_to_physical(voxel_coords, affine):
    """Convert Nx3 voxel coordinates to Nx3 physical coordinates in LPS space.
    
    NIfTI affine maps voxels to RAS physical space. ITK/MedImages uses LPS
    convention (negate X and Y). We convert to LPS here so the computed
    transform is directly compatible with ITK .tfm files.
    """
    ones = np.ones((voxel_coords.shape[0], 1))
    vox_homo = np.hstack([voxel_coords, ones])  # Nx4
    phys_ras = (affine @ vox_homo.T).T[:, :3]   # Nx3 in RAS
    # Convert RAS -> LPS: negate X and Y
    phys_lps = phys_ras.copy()
    phys_lps[:, 0] *= -1  # L = -R
    phys_lps[:, 1] *= -1  # P = -A
    return phys_lps


def extract_landmarks(labels_dir, ref_path=None):
    """
    Extract anatomical landmarks in physical (mm) space from TotalSegmentator masks.
    
    Returns dict of landmark_name -> np.array([x, y, z]) in physical space.
    """
    landmarks = {}
    
    # Determine affine from reference image or from any mask
    ref_affine = None
    if ref_path and os.path.isfile(ref_path):
        ref_affine = nib.load(ref_path).affine
    
    def _get_affine(mask_path):
        """Get affine from mask file, fall back to reference."""
        if os.path.isfile(mask_path):
            return nib.load(mask_path).affine
        return ref_affine
    
    # --- Sacrum: most inferior point ---
    sacrum_path = os.path.join(labels_dir, "sacrum.nii.gz")
    data, affine = load_mask(sacrum_path)
    if data is not None:
        coords = np.argwhere(data > 0)  # Nx3 (i, j, k)
        phys = voxel_to_physical(coords, affine)
        # Inferior = most negative Z (or smallest Z in physical)
        idx = phys[:, 2].argmin()
        landmarks["sacrum_inferior"] = phys[idx]
        print(f"    sacrum_inferior: {phys[idx]}")
    
    # --- Hip left: most medial point ---
    # In LPS: left side = positive X. Medial = closest to X=0 = smallest positive X
    hip_l_path = os.path.join(labels_dir, "hip_left.nii.gz")
    data, affine = load_mask(hip_l_path)
    if data is not None:
        coords = np.argwhere(data > 0)
        phys = voxel_to_physical(coords, affine)
        # Medial = closest to midline (X=0)
        idx = np.abs(phys[:, 0]).argmin()
        landmarks["hip_left_medial"] = phys[idx]
        print(f"    hip_left_medial: {phys[idx]}")
    
    # --- Hip right: most medial point ---
    # In LPS: right side = negative X. Medial = closest to X=0 = largest (least negative) X
    hip_r_path = os.path.join(labels_dir, "hip_right.nii.gz")
    data, affine = load_mask(hip_r_path)
    if data is not None:
        coords = np.argwhere(data > 0)
        phys = voxel_to_physical(coords, affine)
        # Medial = closest to midline (X=0)
        idx = np.abs(phys[:, 0]).argmin()
        landmarks["hip_right_medial"] = phys[idx]
        print(f"    hip_right_medial: {phys[idx]}")
    
    # --- Prostate: 6 extremal points ---
    prostate_path = os.path.join(labels_dir, "prostate.nii.gz")
    data, affine = load_mask(prostate_path)
    if data is not None:
        coords = np.argwhere(data > 0)
        phys = voxel_to_physical(coords, affine)
        
        extremals = {
            "prostate_superior": phys[phys[:, 2].argmax()],
            "prostate_inferior": phys[phys[:, 2].argmin()],
            "prostate_left":     phys[phys[:, 0].argmin()],
            "prostate_right":    phys[phys[:, 0].argmax()],
            "prostate_anterior": phys[phys[:, 1].argmin()],
            "prostate_posterior": phys[phys[:, 1].argmax()],
        }
        for name, pt in extremals.items():
            landmarks[name] = pt
            print(f"    {name}: {pt}")
    
    return landmarks


def rigid_registration_svd(fixed_pts, moving_pts):
    """
    Compute the MSE-optimal rigid transform (rotation + translation) mapping
    moving points to fixed points, using the SVD method (Arun et al. 1987).
    
    Args:
        fixed_pts:  Nx3 numpy array of target (fixed) landmark coordinates
        moving_pts: Nx3 numpy array of source (moving) landmark coordinates
    
    Returns:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        rmse: root mean squared error of the registration
    """
    assert fixed_pts.shape == moving_pts.shape
    assert fixed_pts.shape[0] >= 3, f"Need >= 3 landmarks, got {fixed_pts.shape[0]}"
    
    # 1. Compute centroids
    cf = fixed_pts.mean(axis=0)
    cm = moving_pts.mean(axis=0)
    
    # 2. Center point sets
    pf = fixed_pts - cf
    pm = moving_pts - cm
    
    # 3. Cross-covariance matrix
    H = pm.T @ pf
    
    # 4. SVD
    U, S, Vt = np.linalg.svd(H)
    
    # 5. Optimal rotation (handle reflection case)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])  # Ensures proper rotation (det = +1)
    R = Vt.T @ D @ U.T
    
    # 6. Optimal translation
    t = cf - R @ cm
    
    # 7. Compute RMSE
    transformed = (R @ moving_pts.T).T + t
    residuals = np.linalg.norm(transformed - fixed_pts, axis=1)
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    return R, t, rmse


def write_itk_tfm(R, t, output_path):
    """
    Write a rigid transform in ITK .tfm format.
    
    ITK convention: the transform maps Fixed -> Moving space.
    Our R, t map Moving -> Fixed. So we need to invert:
        ITK_R = R^T,  ITK_t = -R^T @ t
    """
    # ITK stores the inverse: T maps fixed to moving
    R_itk = R.T
    t_itk = -R.T @ t
    
    params = []
    # Row-major flattening of rotation
    for i in range(3):
        for j in range(3):
            params.append(f"{R_itk[i, j]}")
    # Translation
    for i in range(3):
        params.append(f"{t_itk[i]}")
    
    with open(output_path, "w") as f:
        f.write("#Insight Transform File V1.0\n")
        f.write("#Transform 0\n")
        f.write("Transform: AffineTransform_double_3_3\n")
        f.write(f"Parameters: {' '.join(params)}\n")
        f.write("FixedParameters: 0 0 0\n")
    
    print(f"  Written ITK transform to {output_path}")


def register_pair(fixed_labels_dir, moving_labels_dir, output_tfm,
                  fixed_ref=None, moving_ref=None):
    """Register a single pair of timepoints using landmark matching."""
    print(f"\n  Extracting fixed landmarks from {os.path.basename(fixed_labels_dir)}...")
    fixed_lm = extract_landmarks(fixed_labels_dir, fixed_ref)
    
    print(f"  Extracting moving landmarks from {os.path.basename(moving_labels_dir)}...")
    moving_lm = extract_landmarks(moving_labels_dir, moving_ref)
    
    # Match landmarks by name
    common = sorted(set(fixed_lm.keys()) & set(moving_lm.keys()))
    print(f"  Matched {len(common)} landmarks: {common}")
    
    if len(common) < 3:
        print(f"  WARNING: Only {len(common)} common landmarks — need >= 3. Writing identity.")
        R = np.eye(3)
        t = np.zeros(3)
        write_itk_tfm(R, t, output_tfm)
        return 0.0
    
    fixed_pts = np.array([fixed_lm[k] for k in common])
    moving_pts = np.array([moving_lm[k] for k in common])
    
    R, t, rmse = rigid_registration_svd(fixed_pts, moving_pts)
    
    print(f"  Registration RMSE: {rmse:.2f} mm ({len(common)} landmarks)")
    print(f"  Translation magnitude: {np.linalg.norm(t):.1f} mm")
    
    # Rotation angle
    angle_rad = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    print(f"  Rotation angle: {np.degrees(angle_rad):.2f}°")
    
    write_itk_tfm(R, t, output_tfm)
    return rmse


def run_full_pipeline(case_dir):
    """Run landmark registration for all timepoints in the case directory."""
    print("=" * 60)
    print("  Landmark-Based Rigid Registration (SVD)")
    print("=" * 60)
    
    # CT0 is the fixed (baseline) reference
    ct0_labels = os.path.join(case_dir, "anatomy_out_fixed_ct_0")
    ct0_ref = os.path.join(case_dir, "Fixed_CT_Volume_0.nii.gz")
    
    results = {}
    
    # --- CT1 -> CT0 ---
    ct1_labels = os.path.join(case_dir, "anatomy_out_fixed_ct_1")
    ct1_ref = os.path.join(case_dir, "Fixed_CT_Volume_1.nii.gz")
    tfm1 = os.path.join(case_dir, "Transform_FollowUp_to_Baseline_1.tfm")
    
    if os.path.isdir(ct1_labels):
        print("\n>>> [1/3] CT1 -> CT0")
        rmse = register_pair(ct0_labels, ct1_labels, tfm1, ct0_ref, ct1_ref)
        results["CT1->CT0"] = rmse
    else:
        print(f"\n>>> [1/3] CT1 labels not found ({ct1_labels}), writing identity")
        write_itk_tfm(np.eye(3), np.zeros(3), tfm1)
    
    # --- CT2 -> CT0 ---
    ct2_labels = os.path.join(case_dir, "anatomy_out_fixed_ct_2")
    ct2_ref = os.path.join(case_dir, "Fixed_CT_Volume_2.nii.gz")
    tfm2 = os.path.join(case_dir, "Transform_FollowUp_to_Baseline_2.tfm")
    
    if os.path.isdir(ct2_labels):
        print("\n>>> [2/3] CT2 -> CT0")
        rmse = register_pair(ct0_labels, ct2_labels, tfm2, ct0_ref, ct2_ref)
        results["CT2->CT0"] = rmse
    else:
        print(f"\n>>> [2/3] CT2 labels not found ({ct2_labels}), writing identity")
        write_itk_tfm(np.eye(3), np.zeros(3), tfm2)
    
    # --- T2W MRI -> CT0 (TP3) ---
    mr3_labels = os.path.join(case_dir, "anatomy_out_fixed_ct_3")
    mr3_ref = os.path.join(case_dir, "Fixed_CT_Volume_3.nii.gz")
    tfm3 = os.path.join(case_dir, "Transform_FollowUp_to_Baseline_3.tfm")
    
    print("\n>>> [3/3] T2W MRI -> CT0")
    rmse = register_pair(ct0_labels, mr3_labels, tfm3, ct0_ref, mr3_ref)
    results["MRI->CT0"] = rmse
    
    # --- Copy T2W transform to TP 4, 5, 6 (same MRI session, co-registered) ---
    for tp in [4, 5, 6]:
        dst = os.path.join(case_dir, f"Transform_FollowUp_to_Baseline_{tp}.tfm")
        shutil.copy2(tfm3, dst)
        print(f"  Copied Transform 3 -> Transform {tp}")
    
    # --- Summary ---
    print("\n" + "=" * 60)
    print("  Registration Summary")
    print("=" * 60)
    for pair, rmse in results.items():
        print(f"  {pair:20s}  RMSE = {rmse:.2f} mm")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Landmark-based rigid registration via SVD")
    parser.add_argument("--case-dir", type=str, help="Case directory (runs full pipeline)")
    parser.add_argument("--fixed-labels-dir", type=str, help="Fixed (baseline) labels directory")
    parser.add_argument("--moving-labels-dir", type=str, help="Moving (follow-up) labels directory")
    parser.add_argument("--fixed-ref", type=str, help="Fixed reference NIfTI (for affine)")
    parser.add_argument("--moving-ref", type=str, help="Moving reference NIfTI (for affine)")
    parser.add_argument("--output-tfm", type=str, help="Output ITK .tfm file path")
    parser.add_argument("--report", action="store_true", help="Print registration report only")
    args = parser.parse_args()
    
    if args.case_dir:
        run_full_pipeline(args.case_dir)
    elif args.fixed_labels_dir and args.moving_labels_dir and args.output_tfm:
        register_pair(args.fixed_labels_dir, args.moving_labels_dir, args.output_tfm,
                      args.fixed_ref, args.moving_ref)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
