import os
import sys
import glob
import SimpleITK as sitk
import numpy as np
from itertools import combinations

def get_bbox(arr):
    if not np.any(arr): return None
    z, y, x = np.where(arr > 0)
    return (z.min(), z.max(), y.min(), y.max(), x.min(), x.max())

def bboxes_intersect(b1, b2):
    if b1 is None or b2 is None: return False
    return not (b1[1] < b2[0] or b1[0] > b2[1] or
                b1[3] < b2[2] or b1[2] > b2[3] or
                b1[5] < b2[4] or b1[4] > b2[5])

def main(seg_dir):
    files = sorted(glob.glob(os.path.join(seg_dir, "*.nii.gz")))
    print(f"Found {len(files)} NIfTI files.")
    
    arrays = {}
    bboxes = {}
    vols = {}
    spacing = None
    
    for f in files:
        name = os.path.basename(f)
        try:
            img = sitk.ReadImage(f)
            if spacing is None: spacing = img.GetSpacing()
            arr = sitk.GetArrayFromImage(img) > 0
            if not np.any(arr): continue
            
            arrays[name] = arr
            bboxes[name] = get_bbox(arr)
            vols[name] = np.sum(arr)
        except Exception as e:
            print(f"Error loading {name}: {e}")
            
    if spacing is None: return
    vol_per_voxel = spacing[0] * spacing[1] * spacing[2] / 1000.0 # in cc

    print("\nCalculating overlaps...")
    for (n1, a1), (n2, a2) in combinations(arrays.items(), 2):
        if bboxes_intersect(bboxes[n1], bboxes[n2]):
            intersection = np.sum(a1 & a2)
            if intersection > 0:
                overlap_cc = intersection * vol_per_voxel
                pct_n1 = (intersection / vols[n1]) * 100
                pct_n2 = (intersection / vols[n2]) * 100
                if overlap_cc > 0.5: # Ignore trivial overlaps less than 0.5cc
                    print(f"[{overlap_cc:.2f} cc] {n1} ({pct_n1:.1f}%) <-> {n2} ({pct_n2:.1f}%)")

if __name__ == "__main__":
    main(sys.argv[1])
