#!/usr/bin/env python3
"""
Unpack packed_anatomical_tensor.h5 into individual NIfTI segmentation files.
This allows the Julia GPU DAG pipeline to build primary_masks.h5 from existing data
without re-running TotalSegmentator.

Usage:
    python3 unpack_anatomy_to_niftis.py <case_raw_dir>
    python3 unpack_anatomy_to_niftis.py --batch  # Process all cases
"""

import os
import sys
import json
import numpy as np
import h5py
import SimpleITK as sitk
import argparse


def unpack_case(raw_dir):
    """Unpack packed_anatomical_tensor.h5 into segmentations/*.nii.gz."""
    anat_h5 = os.path.join(raw_dir, "packed_anatomical_tensor.h5")
    reg_path = os.path.join(raw_dir, "packed_anatomical_tensor_registry.json")
    ct_path = os.path.join(raw_dir, "Fixed_CT_Volume.nii.gz")
    
    if not os.path.isfile(anat_h5):
        return "NO_PACKED_TENSOR"
    if not os.path.isfile(reg_path):
        return "NO_REGISTRY"
    if not os.path.isfile(ct_path):
        return "NO_CT"
    
    seg_dir = os.path.join(raw_dir, "segmentations")
    os.makedirs(seg_dir, exist_ok=True)
    
    # Load CT for reference geometry
    ct_img = sitk.ReadImage(ct_path)
    
    with open(reg_path, "r") as f:
        reg = json.load(f)
    
    masks_dict = reg.get("masks", reg)
    
    count = 0
    skipped = 0
    
    with h5py.File(anat_h5, "r") as af:
        pt = af["packed_tensor"][:]
        
        for name, meta in masks_dict.items():
            if not isinstance(meta, dict):
                continue
            
            clean = name.lower()
            # Skip temp/helper masks
            if any(bad in clean for bad in ["temp", "helper", "dilation", "buffer"]):
                skipped += 1
                continue
            
            ch = meta.get("channel", 1) - 1
            mid = meta.get("id", 1)
            
            if 0 <= ch < pt.shape[0]:
                mask = (pt[ch] == mid).astype(np.uint8)
                if mask.any():
                    # Save as NIfTI with same geometry as CT
                    mask_img = sitk.GetImageFromArray(mask)
                    mask_img.SetSpacing(ct_img.GetSpacing())
                    mask_img.SetOrigin(ct_img.GetOrigin())
                    mask_img.SetDirection(ct_img.GetDirection())
                    
                    out_path = os.path.join(seg_dir, f"{clean}.nii.gz")
                    sitk.WriteImage(mask_img, out_path)
                    count += 1
                else:
                    skipped += 1
    
    # Create completion marker
    with open(os.path.join(seg_dir, ".ts_task_total_completed"), "w") as f:
        f.write("unpacked_from_packed_tensor")
    
    return f"OK ({count} masks, {skipped} skipped)"


def main():
    parser = argparse.ArgumentParser(description="Unpack anatomy tensor to NIfTIs")
    parser.add_argument("raw_dir", nargs="?", help="Case raw directory")
    parser.add_argument("--batch", action="store_true", help="Process all cases")
    args = parser.parse_args()
    
    if args.batch:
        cases_dir = "/mnt/big/project_ssd/project_ssd/MedEye3d.jl/data/cases/Prostata_bimodal"
        case_dirs = sorted([
            d for d in os.listdir(cases_dir)
            if d.startswith("Pat") and os.path.isdir(os.path.join(cases_dir, d))
            and "_PETCT_" not in d
        ])
        
        ok = 0
        fail = 0
        for cn in case_dirs:
            raw_dir = os.path.join(cases_dir, cn, "raw")
            seg_dir = os.path.join(raw_dir, "segmentations")
            
            # Skip if segmentations already exist
            if os.path.isdir(seg_dir) and len(os.listdir(seg_dir)) > 20:
                print(f"  ⊘ {cn}: segmentations/ already exists ({len(os.listdir(seg_dir))} files)")
                ok += 1
                continue
            
            result = unpack_case(raw_dir)
            icon = "✓" if result.startswith("OK") else "✗"
            print(f"  {icon} {cn}: {result}")
            if result.startswith("OK"):
                ok += 1
            else:
                fail += 1
        
        print(f"\nOK: {ok}  FAIL: {fail}")
    elif args.raw_dir:
        result = unpack_case(args.raw_dir)
        print(result)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
