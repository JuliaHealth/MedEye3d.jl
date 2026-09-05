#!/usr/bin/env python3
"""
run_pytorch_surface_registration.py

Runs the PyTorch-based surface registration from:
/mnt/big/project_ssd/project_ssd/registration_comparison/surface_registration

For all 3 follow-up timepoints:
1. CT 1 -> CT 0: using hip_left, hip_right, prostate, sternum
2. CT 2 -> CT 0: using hip_left, hip_right, prostate, sternum
3. MRI 3 -> CT 0: using hip_left, hip_right, prostate (from TotalSegmentator total_mr)

Writes the resulting ITK transforms to:
- Transform_FollowUp_to_Baseline_1.tfm
- Transform_FollowUp_to_Baseline_2.tfm
- Transform_FollowUp_to_Baseline_3.tfm
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def main():
    base_dir = "/mnt/big/project_ssd/project_ssd/MedEye3d.jl"
    case_dir = os.path.join(base_dir, "data", "cases", "psma_patient_all_tp")
    reg_code_dir = "/mnt/big/project_ssd/project_ssd/registration_comparison/surface_registration"
    register_script = os.path.join(reg_code_dir, "register.py")

    ct0_path = os.path.join(case_dir, "Fixed_CT_Volume_0.nii.gz")
    ct1_path = os.path.join(case_dir, "Fixed_CT_Volume_1.nii.gz")
    ct2_path = os.path.join(case_dir, "Fixed_CT_Volume_2.nii.gz")
    mr3_path = os.path.join(case_dir, "Fixed_CT_Volume_3.nii.gz")

    env = os.environ.copy()
    env["PYTHONPATH"] = reg_code_dir + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = "0"

    print("==========================================================")
    print("  PyTorch Surface Registration via TotalSegmentator Masks")
    print("==========================================================")

    # -------------------------------------------------------------------------
    # 1. Prepare Label Folders for CT-to-CT (CT 1 & CT 2)
    # Structures: hip_left, hip_right, prostate, sternum
    # -------------------------------------------------------------------------
    ct0_labels_dir = os.path.join(case_dir, "anatomy_out_fixed_ct_0")
    ct1_labels_dir = os.path.join(case_dir, "anatomy_out_fixed_ct_1")
    ct2_labels_dir = os.path.join(case_dir, "anatomy_out_fixed_ct_2")

    # Filter CT0 labels to only registration targets
    ct0_ct_reg_dir = os.path.join(case_dir, "reg_labels_ct0_ct")
    os.makedirs(ct0_ct_reg_dir, exist_ok=True)
    for s in ["hip_left.nii.gz", "hip_right.nii.gz", "prostate.nii.gz", "sternum.nii.gz"]:
        src = os.path.join(ct0_labels_dir, s)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(ct0_ct_reg_dir, s))

    # Expanded MRI registration organs (11 structures for accurate pelvic alignment)
    # Excludes bladder (shape changes between studies) per user direction
    MRI_REG_ORGANS = [
        "hip_left.nii.gz", "hip_right.nii.gz", "prostate.nii.gz",
        # Muscles (stable shape, good pelvic landmarks)
        "gluteus_maximus_left.nii.gz", "gluteus_maximus_right.nii.gz",
        "iliopsoas_left.nii.gz", "iliopsoas_right.nii.gz",
        # Vessels (stable position, fine alignment)
        "iliac_vena_left.nii.gz", "iliac_vena_right.nii.gz",
        "iliac_artery_left.nii.gz", "iliac_artery_right.nii.gz",
    ]

    # Filter CT0 labels for MRI registration (all MRI_REG_ORGANS)
    ct0_mri_reg_dir = os.path.join(case_dir, "reg_labels_ct0_mri")
    os.makedirs(ct0_mri_reg_dir, exist_ok=True)
    for s in MRI_REG_ORGANS:
        src = os.path.join(ct0_labels_dir, s)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(ct0_mri_reg_dir, s))

    # MRI labels (from TotalSegmentator total_mr output)
    mr3_labels_dir = os.path.join(case_dir, "anatomy_out_fixed_ct_3")
    mr3_reg_dir = os.path.join(case_dir, "reg_labels_mr3")
    os.makedirs(mr3_reg_dir, exist_ok=True)
    for s in MRI_REG_ORGANS:
        src = os.path.join(mr3_labels_dir, s)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(mr3_reg_dir, s))
        else:
            print(f"  ⚠️  MRI organ not found: {s}")

    # -------------------------------------------------------------------------
    # PAIR 1: CT 1 -> CT 0 (hip_left, hip_right, prostate, sternum)
    # -------------------------------------------------------------------------
    print("\n>>> [1/3] Registering Follow-up CT 1 to Baseline CT 0...")
    out_dir_1 = os.path.join(case_dir, "reg_output_tp1")
    shutil.rmtree(out_dir_1, ignore_errors=True)
    cmd_1 = [
        "python3", register_script,
        "--ct", ct0_path,
        "--mri", ct1_path,
        "--ct-labels", ct0_ct_reg_dir,
        "--mri-labels", ct1_labels_dir,
        "--output", out_dir_1
    ]
    subprocess.run(cmd_1, check=True, env=env)
    
    tfm_resample_1 = os.path.join(out_dir_1, "final_transform_CT_to_MRI_resample.tfm")
    target_tfm_1 = os.path.join(case_dir, "Transform_FollowUp_to_Baseline_1.tfm")
    shutil.copy2(tfm_resample_1, target_tfm_1)
    print(f"  ✅ Saved {target_tfm_1}")

    # -------------------------------------------------------------------------
    # PAIR 2: CT 2 -> CT 0 (hip_left, hip_right, prostate, sternum)
    # -------------------------------------------------------------------------
    print("\n>>> [2/3] Registering Follow-up CT 2 to Baseline CT 0...")
    out_dir_2 = os.path.join(case_dir, "reg_output_tp2")
    shutil.rmtree(out_dir_2, ignore_errors=True)
    cmd_2 = [
        "python3", register_script,
        "--ct", ct0_path,
        "--mri", ct2_path,
        "--ct-labels", ct0_ct_reg_dir,
        "--mri-labels", ct2_labels_dir,
        "--output", out_dir_2
    ]
    subprocess.run(cmd_2, check=True, env=env)
    
    tfm_resample_2 = os.path.join(out_dir_2, "final_transform_CT_to_MRI_resample.tfm")
    target_tfm_2 = os.path.join(case_dir, "Transform_FollowUp_to_Baseline_2.tfm")
    shutil.copy2(tfm_resample_2, target_tfm_2)
    print(f"  ✅ Saved {target_tfm_2}")

    # -------------------------------------------------------------------------
    # PAIR 3: MRI 3 -> CT 0 (hip_left, hip_right, prostate)
    # -------------------------------------------------------------------------
    print("\n>>> [3/3] Registering Follow-up MRI 3 to Baseline CT 0...")
    out_dir_3 = os.path.join(case_dir, "reg_output_tp3")
    shutil.rmtree(out_dir_3, ignore_errors=True)
    cmd_3 = [
        "python3", register_script,
        "--ct", ct0_path,
        "--mri", mr3_path,
        "--ct-labels", ct0_mri_reg_dir,
        "--mri-labels", mr3_reg_dir,
        "--output", out_dir_3
    ]
    subprocess.run(cmd_3, check=True, env=env)
    
    tfm_resample_3 = os.path.join(out_dir_3, "final_transform_CT_to_MRI_resample.tfm")
    target_tfm_3 = os.path.join(case_dir, "Transform_FollowUp_to_Baseline_3.tfm")
    shutil.copy2(tfm_resample_3, target_tfm_3)
    print(f"  ✅ Saved {target_tfm_3}")

    print("\n==========================================================")
    print("  All PyTorch surface registrations completed successfully!")
    print("==========================================================")

if __name__ == "__main__":
    main()
