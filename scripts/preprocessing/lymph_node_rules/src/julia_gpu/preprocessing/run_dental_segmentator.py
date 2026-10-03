#!/usr/bin/env python3
"""
run_dental_segmentator.py
Runs the SlicerDentalSegmentator (nnU-Net) on a single patient's CT,
cropped to the skull bounding box, to produce a mandible segmentation.

Usage:
    python3 run_dental_segmentator.py <case_dir>
"""
import os
import sys
import subprocess
import shutil
import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "SimpleITK"], check=True)
    import SimpleITK as sitk

def run_dental(case_dir):
    seg_dir = os.path.join(case_dir, "segmentations")
    ct_path = os.path.join(case_dir, "Fixed_CT_Volume.nii.gz")
    mandible_file = os.path.join(seg_dir, "mandible.nii.gz")
    
    # Check if mandible already exists
    if os.path.exists(mandible_file):
        img = sitk.ReadImage(mandible_file)
        arr = sitk.GetArrayFromImage(img)
        if arr.max() > 0:
            print(f"[DENTAL] mandible.nii.gz already exists and is non-empty ({arr.sum()} voxels). Skipping.")
            return mandible_file
    
    # Check for skull (needed for bounding box)
    skull_path = os.path.join(seg_dir, "skull.nii.gz")
    if not os.path.exists(skull_path):
        print(f"[DENTAL] WARNING: skull.nii.gz not found. Cannot run dental segmentator.")
        # Try to use TS mandible directly
        ts_mandible = os.path.join(seg_dir, "mandible_primary_totalsegmentator.nii.gz")
        if os.path.exists(ts_mandible):
            print(f"[DENTAL] Using TotalSegmentator mandible as fallback.")
            shutil.copy2(ts_mandible, mandible_file)
            return mandible_file
        return None
    
    # Check dental model weights
    model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "old", "src_v2", "anatomic_segmentation", "models", "dental")
    if not os.path.exists(model_dir):
        # Try alternate path
        model_dir = "old/src_v2/anatomic_segmentation/models/dental"
    
    if not os.path.exists(model_dir):
        print(f"[DENTAL] WARNING: Dental model not found at {model_dir}")
        # Fallback: check if TotalSegmentator produced a mandible
        ts_mandible_files = [
            os.path.join(seg_dir, "mandible_primary_totalsegmentator.nii.gz"),
        ]
        for ts_f in ts_mandible_files:
            if os.path.exists(ts_f):
                print(f"[DENTAL] Using TotalSegmentator mandible from {ts_f}")
                shutil.copy2(ts_f, mandible_file)
                return mandible_file
        return None
    
    print(f"[DENTAL] Running SlicerDentalSegmentator...")
    
    # Load CT and skull for cropping
    orig_img = sitk.ReadImage(ct_path)
    skull_img = sitk.ReadImage(skull_path)
    
    # Resample skull if size mismatch
    if skull_img.GetSize() != orig_img.GetSize():
        skull_img = sitk.Resample(skull_img, orig_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0, skull_img.GetPixelID())
    
    skull_arr = sitk.GetArrayFromImage(skull_img)
    z_indices, y_indices, x_indices = np.where(skull_arr > 0)
    
    if len(z_indices) == 0:
        print("[DENTAL] Skull mask is empty. Skipping.")
        return None
    
    # Crop CT to skull bounding box with margin
    margin = 20
    start_z = max(0, z_indices.min() - margin)
    end_z = min(orig_img.GetSize()[2], z_indices.max() + 1 + margin)
    start_y = max(0, y_indices.min() - margin)
    end_y = min(orig_img.GetSize()[1], y_indices.max() + 1 + margin)
    start_x = max(0, x_indices.min() - margin)
    end_x = min(orig_img.GetSize()[0], x_indices.max() + 1 + margin)
    
    cropped_img = orig_img[start_x:end_x, start_y:end_y, start_z:end_z]
    
    # Set up temp dirs
    tmp_in_dir = os.path.join(seg_dir, "dental_tmp_in")
    tmp_out_dir = os.path.join(seg_dir, "dental_tmp_out")
    os.makedirs(tmp_in_dir, exist_ok=True)
    os.makedirs(tmp_out_dir, exist_ok=True)
    
    tmp_in_file = os.path.join(tmp_in_dir, "dental_0000.nii.gz")
    sitk.WriteImage(cropped_img, tmp_in_file)
    
    # Set env for nnU-Net
    os.environ["nnUNet_results"] = model_dir
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    # Run nnU-Net inference
    script_content = f"""
import sys
from nnunetv2.inference.predict_from_raw_data import predict_entry_point
sys.argv = [
    "nnUNetv2_predict",
    "-i", "{tmp_in_dir}",
    "-o", "{tmp_out_dir}",
    "-d", "111",
    "-c", "3d_fullres",
    "-f", "0",
    "--disable_tta",
    "-step_size", "0.8",
    "-npp", "0",
    "-nps", "0"
]
if __name__ == '__main__':
    predict_entry_point()
"""
    script_file = os.path.join(tmp_in_dir, "run_nnunet.py")
    with open(script_file, "w") as f:
        f.write(script_content)
    
    try:
        subprocess.run([sys.executable, script_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[DENTAL] nnU-Net inference failed: {e}")
        shutil.rmtree(tmp_in_dir, ignore_errors=True)
        shutil.rmtree(tmp_out_dir, ignore_errors=True)
        return None
    
    out_file_raw = os.path.join(tmp_out_dir, "dental.nii.gz")
    if not os.path.exists(out_file_raw):
        print(f"[DENTAL] No output file produced")
        shutil.rmtree(tmp_in_dir, ignore_errors=True)
        shutil.rmtree(tmp_out_dir, ignore_errors=True)
        return None
    
    # Extract label 2 (Mandible) and pad back to original size
    pred_img = sitk.ReadImage(out_file_raw)
    pred_arr = sitk.GetArrayFromImage(pred_img)
    mandible_arr_cropped = (pred_arr == 2).astype(np.uint8)
    
    full_arr = np.zeros((orig_img.GetSize()[2], orig_img.GetSize()[1], orig_img.GetSize()[0]), dtype=np.uint8)
    full_arr[start_z:end_z, start_y:end_y, start_x:end_x] = mandible_arr_cropped
    
    mandible_img = sitk.GetImageFromArray(full_arr)
    mandible_img.CopyInformation(orig_img)
    sitk.WriteImage(mandible_img, mandible_file)
    
    # Also save as slicer dental fallback name (for preprocessing compatibility)
    fallback_path = os.path.join(seg_dir, "mandible_fallback_2_slicerdentalsegmentator.nii.gz")
    shutil.copy2(mandible_file, fallback_path)
    
    print(f"[DENTAL] Saved mandible to {mandible_file} ({mandible_arr_cropped.sum()} voxels)")
    
    # Cleanup
    shutil.rmtree(tmp_in_dir, ignore_errors=True)
    shutil.rmtree(tmp_out_dir, ignore_errors=True)
    
    return mandible_file


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <case_dir>")
        sys.exit(1)
    run_dental(sys.argv[1])
