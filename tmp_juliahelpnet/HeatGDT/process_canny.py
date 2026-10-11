import json
import os
import glob
import numpy as np
import nibabel as nib
from skimage.feature import canny
from tqdm import tqdm

def canny_3d(vol):
    out = np.zeros_like(vol)
    for z in range(vol.shape[2]):
        slice = vol[:, :, z]
        mi, ma = slice.min(), slice.max()
        if ma > mi:
            slice = (slice - mi) / (ma - mi)
        out[:, :, z] = canny(slice, sigma=1.0, low_threshold=0.1, high_threshold=0.3)
    return out

def process_manifest(manifest_path, out_manifest_path):
    with open(manifest_path, 'r') as f:
        items = json.load(f)
    
    new_items = []
    for item in tqdm(items):
        ct_path = item["ct_path"]
        pet_path = item["pet_path"]
        
        canny_ct_path = ct_path.replace(".nii.gz", "_canny.nii.gz")
        canny_pet_path = pet_path.replace(".nii.gz", "_canny.nii.gz")
        
        # Process CT
        if not os.path.exists(canny_ct_path):
            ct_img = nib.load(ct_path)
            ct_data = ct_img.get_fdata(dtype=np.float32)
            c_ct = canny_3d(ct_data).astype(np.float32)
            nib.save(nib.Nifti1Image(c_ct, ct_img.affine), canny_ct_path)
        
        # Process PET
        if not os.path.exists(canny_pet_path):
            pet_img = nib.load(pet_path)
            pet_data = pet_img.get_fdata(dtype=np.float32)
            c_pet = canny_3d(pet_data).astype(np.float32)
            nib.save(nib.Nifti1Image(c_pet, pet_img.affine), canny_pet_path)
        
        new_item = item.copy()
        new_item["canny_ct_path"] = canny_ct_path
        new_item["canny_pet_path"] = canny_pet_path
        new_items.append(new_item)
    
    with open(out_manifest_path, 'w') as f:
        json.dump(new_items, f, indent=4)

if __name__ == "__main__":
    print("Processing val manifest...")
    process_manifest("/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/val_single_lesion_manifest.json", "/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/canny_val_manifest.json")
    print("Processing train manifest...")
    process_manifest("/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/train_combined_manifest.json", "/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/canny_train_manifest.json")
    print("Done")
