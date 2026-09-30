import os
import glob
import h5py
import nibabel as nib
import sys
import numpy as np

tp_dir = sys.argv[1]
h5_path = os.path.join(tp_dir, "primary_masks.h5")
if os.path.exists(h5_path):
    print("Already exists")
    sys.exit(0)

print(f"Packing NIfTIs in {tp_dir} to {h5_path}...")
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
        # Python loads (284, 512, 512) if original was (512, 512, 284)
        # So we transpose it back to (512, 512, 284) for Julia
        arr = np.transpose(arr)
        grp.create_dataset(name, data=arr, compression="lzf")

print("Packed!")
