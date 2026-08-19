import nibabel as nib
import numpy as np
img = nib.load('data/pat_6_files/Skellytour_0.nii.gz').get_fdata()
# Check lesion 12
mask = nib.load('data/pat_6_files/PET_Lesions_0.nii.gz').get_fdata()
cx, cy, cz = 85, 84, 60
box = img[cx-20:cx+20, cy-20:cy+20, cz-20:cz+20]
print("Unique in box:", np.unique(box))
