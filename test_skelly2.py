import numpy as np, nibabel as nib
img = nib.load('data/pat_6_files/Skellytour_0.nii.gz').get_fdata()
mask = nib.load('data/pat_6_files/PET_Lesions_0.nii.gz').get_fdata()
pts = np.argwhere(mask == 7.0)
print(f"Lesion 7 has {len(pts)} points")
cx, cy, cz = np.mean(pts, axis=0).astype(int)
print(f"Centroid: {cx, cy, cz}")
box = img[cx-20:cx+20, cy-20:cy+20, cz-20:cz+20]
print("Unique in box:", np.unique(box))
from scipy import ndimage as ndi
bone_mask = box > 0
holes = ndi.binary_fill_holes(bone_mask)
print(f"Bone voxels: {np.sum(bone_mask)}, Filled voxels: {np.sum(holes)}")
