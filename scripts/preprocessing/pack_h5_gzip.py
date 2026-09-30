import os
import glob
import h5py
import nibabel as nib
import sys
import numpy as np

tp_dir = sys.argv[1]
h5_path = os.path.join(tp_dir, "primary_masks.h5")

print(f"Packing NIfTIs in {tp_dir} to {h5_path} with GZIP...")
niftis = glob.glob(os.path.join(tp_dir, "*.nii.gz"))
if not niftis:
    sys.exit(0)

with h5py.File(h5_path, 'w') as f:
    grp = f.create_group("masks")
    for nii in niftis:
        name = os.path.basename(nii).replace(".nii.gz", "")
        if name == "max_anatomy": continue
        img = nib.load(nii)
        arr = img.get_fdata().astype(np.uint8)
        arr = np.transpose(arr)
        grp.create_dataset(name, data=arr, compression="gzip", compression_opts=4)

print("Packed with GZIP!")
