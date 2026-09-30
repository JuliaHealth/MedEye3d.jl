import os
import glob
import h5py
import nibabel as nib
import sys
import numpy as np

tp_dir = sys.argv[1]
h5_path = os.path.join(tp_dir, "primary_masks.h5")

print(f"Packing NIfTIs in {tp_dir} to {h5_path} with metadata...")
niftis = glob.glob(os.path.join(tp_dir, "*.nii.gz"))
if not niftis:
    sys.exit(0)

# get meta from max_anatomy or first nifti
first = niftis[0]
img = nib.load(first)
header = img.header
dims = header.get_data_shape() # usually (512, 512, 326)
spacing = header.get_zooms() # e.g. (1.5, 1.5, 2.0)
qoffset_x = header['qoffset_x']
qoffset_y = header['qoffset_y']
qoffset_z = header['qoffset_z']
origin = (float(qoffset_x), float(qoffset_y), float(qoffset_z))

with h5py.File(h5_path, 'a') as f:
    if "metadata" not in f:
        grp_meta = f.create_group("metadata")
    else:
        grp_meta = f["metadata"]
    
    # Python NIfTI reads it as (W, H, D). In Julia we want (W, H, D).
    # Wait, nibabel returns (512, 512, 326) for my files.
    print(f"Dims: {dims}")
    grp_meta.attrs["dimensions"] = np.array(dims, dtype=np.int64)
    grp_meta.attrs["spacing"] = np.array(spacing, dtype=np.float64)
    grp_meta.attrs["origin"] = np.array(origin, dtype=np.float64)

print("Added metadata!")
