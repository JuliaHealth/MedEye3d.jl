#!/usr/bin/env python3
"""
rerun_mri_registration.py

Re-runs only the MRI→CT surface registration (PAIR 3) with expanded organ set.
Does NOT re-run CT-CT registrations (PAIR 1 & 2).
"""

import os
import sys
import shutil
import subprocess
import nibabel as nib
import numpy as np
from pathlib import Path

def main():
    base_dir = "/mnt/big/project_ssd/project_ssd/MedEye3d.jl"
    case_dir = os.path.join(base_dir, "data", "cases", "psma_patient_all_tp")
    reg_code_dir = "/mnt/big/project_ssd/project_ssd/registration_comparison/surface_registration"
    register_script = os.path.join(reg_code_dir, "register.py")

    ct0_path = os.path.join(case_dir, "Fixed_CT_Volume_0.nii.gz")
    mr3_path = os.path.join(case_dir, "Fixed_CT_Volume_3.nii.gz")

    ct0_labels_dir = os.path.join(case_dir, "anatomy_out_fixed_ct_0")
    mr3_labels_dir = os.path.join(case_dir, "anatomy_out_fixed_ct_3")

    # Expanded MRI registration organs (11 structures)
    # Excludes bladder (shape changes between studies)
    MRI_REG_ORGANS = [
        "hip_left.nii.gz", "hip_right.nii.gz", "prostate.nii.gz",
        "gluteus_maximus_left.nii.gz", "gluteus_maximus_right.nii.gz",
        "iliopsoas_left.nii.gz", "iliopsoas_right.nii.gz",
        "iliac_vena_left.nii.gz", "iliac_vena_right.nii.gz",
        "iliac_artery_left.nii.gz", "iliac_artery_right.nii.gz",
    ]

    print("==========================================================")
    print("  MRI→CT Registration with Expanded Organ Set (11 organs)")
    print("==========================================================")

    # Prepare CT0 labels for MRI registration
    ct0_mri_reg_dir = os.path.join(case_dir, "reg_labels_ct0_mri")
    if os.path.exists(ct0_mri_reg_dir):
        shutil.rmtree(ct0_mri_reg_dir)
    os.makedirs(ct0_mri_reg_dir, exist_ok=True)

    ct_count = 0
    for s in MRI_REG_ORGANS:
        src = os.path.join(ct0_labels_dir, s)
        if os.path.exists(src):
            nii = nib.load(src)
            nz = int((np.asanyarray(nii.dataobj) > 0).sum())
            shutil.copy2(src, os.path.join(ct0_mri_reg_dir, s))
            print(f"  CT0 {s}: {nz} voxels")
            ct_count += 1
        else:
            print(f"  CT0 MISSING: {s}")

    # Prepare MRI labels
    mr3_reg_dir = os.path.join(case_dir, "reg_labels_mr3")
    if os.path.exists(mr3_reg_dir):
        shutil.rmtree(mr3_reg_dir)
    os.makedirs(mr3_reg_dir, exist_ok=True)

    mri_count = 0
    for s in MRI_REG_ORGANS:
        src = os.path.join(mr3_labels_dir, s)
        if os.path.exists(src):
            nii = nib.load(src)
            nz = int((np.asanyarray(nii.dataobj) > 0).sum())
            if nz > 0:
                shutil.copy2(src, os.path.join(mr3_reg_dir, s))
                print(f"  MRI {s}: {nz} voxels")
                mri_count += 1
            else:
                print(f"  MRI {s}: 0 voxels (SKIPPED)")
        else:
            print(f"  MRI MISSING: {s}")

    # Only keep organs that exist in BOTH CT and MRI label dirs
    ct_files = set(os.listdir(ct0_mri_reg_dir))
    mri_files = set(os.listdir(mr3_reg_dir))
    matched = ct_files & mri_files
    print(f"\n  Matched organs: {len(matched)}")
    for f in sorted(matched):
        print(f"    ✅ {f}")

    # Remove unmatched from both dirs
    for f in ct_files - matched:
        os.remove(os.path.join(ct0_mri_reg_dir, f))
        print(f"    Removed unmatched CT: {f}")
    for f in mri_files - matched:
        os.remove(os.path.join(mr3_reg_dir, f))
        print(f"    Removed unmatched MRI: {f}")

    # Backup old transform
    old_tfm = os.path.join(case_dir, "Transform_FollowUp_to_Baseline_3.tfm")
    if os.path.exists(old_tfm):
        backup = old_tfm + ".bak_3organs"
        shutil.copy2(old_tfm, backup)
        print(f"\n  Backed up old transform to {backup}")

    # Run registration
    env = os.environ.copy()
    env["PYTHONPATH"] = reg_code_dir + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = "0"

    out_dir_3 = os.path.join(case_dir, "reg_output_tp3_v2")
    if os.path.exists(out_dir_3):
        shutil.rmtree(out_dir_3)

    print(f"\n>>> Running MRI→CT registration ({len(matched)} organs)...")
    cmd_3 = [
        "python3", register_script,
        "--ct", ct0_path,
        "--mri", mr3_path,
        "--ct-labels", ct0_mri_reg_dir,
        "--mri-labels", mr3_reg_dir,
        "--output", out_dir_3
    ]
    print(f"  CMD: {' '.join(cmd_3)}")
    result = subprocess.run(cmd_3, env=env, capture_output=False)

    if result.returncode != 0:
        print(f"\n  ❌ Registration failed with code {result.returncode}")
        sys.exit(1)

    # Copy new transform
    tfm_resample_3 = os.path.join(out_dir_3, "final_transform_CT_to_MRI_resample.tfm")
    if not os.path.exists(tfm_resample_3):
        # Try alternate name
        for candidate in ["final_transform.tfm", "transform.tfm"]:
            alt = os.path.join(out_dir_3, candidate)
            if os.path.exists(alt):
                tfm_resample_3 = alt
                break

    if os.path.exists(tfm_resample_3):
        target_tfm_3 = os.path.join(case_dir, "Transform_FollowUp_to_Baseline_3.tfm")
        shutil.copy2(tfm_resample_3, target_tfm_3)
        print(f"\n  ✅ New transform saved to {target_tfm_3}")
        # Show transform content
        with open(target_tfm_3) as f:
            print(f"  {f.read().strip()}")
    else:
        print(f"\n  ❌ Transform output not found in {out_dir_3}")
        print(f"  Available files: {os.listdir(out_dir_3)}")
        sys.exit(1)

    print("\n==========================================================")
    print("  MRI registration completed with expanded organ set!")
    print("==========================================================")

if __name__ == "__main__":
    main()
