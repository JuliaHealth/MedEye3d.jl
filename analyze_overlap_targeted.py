import os
import sys
import glob
import SimpleITK as sitk
import numpy as np

def main(seg_dir):
    target_files = ["brain.nii.gz", "liver.nii.gz", "aorta.nii.gz"]
    for target in target_files:
        target_path = os.path.join(seg_dir, target)
        if not os.path.exists(target_path):
            continue
            
        target_img = sitk.ReadImage(target_path)
        target_arr = sitk.GetArrayFromImage(target_img) > 0
        vol_per_voxel = target_img.GetSpacing()[0] * target_img.GetSpacing()[1] * target_img.GetSpacing()[2] / 1000.0
        target_vol = np.sum(target_arr) * vol_per_voxel
        
        print(f"\n--- {target} ({target_vol:.2f} cc) ---")
        
        # Check overlap with ALL other NIfTI files
        files = glob.glob(os.path.join(seg_dir, "*.nii.gz"))
        for f in files:
            name = os.path.basename(f)
            if name == target: continue
            try:
                arr = sitk.GetArrayFromImage(sitk.ReadImage(f)) > 0
                intersection = np.sum(target_arr & arr)
                if intersection > 0:
                    overlap_cc = intersection * vol_per_voxel
                    pct_target = (overlap_cc / target_vol) * 100
                    arr_vol = np.sum(arr) * vol_per_voxel
                    pct_other = (overlap_cc / arr_vol) * 100
                    if overlap_cc > 0.5:
                        print(f"Overlap with {name}: {overlap_cc:.2f} cc ( {pct_target:.1f}% of {target}, {pct_other:.1f}% of {name} )")
            except Exception as e:
                pass

if __name__ == "__main__":
    main(sys.argv[1])
