import json
import numpy as np
import nibabel as nib
from scipy.ndimage import label
import os

def process_manifest(input_json, output_json):
    with open(input_json, 'r') as f:
        data = json.load(f)
        
    new_manifest = []
    
    for idx, item in enumerate(data):
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(data)}")
        gt_path = item["gt_path"]
        if not os.path.exists(gt_path):
            continue
            
        gt_data = nib.load(gt_path).get_fdata() > 0
        
        # 3D 26-connectivity
        structure = np.ones((3,3,3), dtype=int)
        labeled_array, num_features = label(gt_data, structure=structure)
        
        for i in range(1, num_features + 1):
            comp_mask = labeled_array == i
            coords = np.argwhere(comp_mask)
            if len(coords) > 15:
                # Calculate medoid
                centroid = coords.mean(axis=0)
                distances = np.sum((coords - centroid)**2, axis=1)
                medoid_idx = np.argmin(distances)
                medoid = coords[medoid_idx]
                
                # NIfTI in nibabel is X, Y, Z (usually)
                # Convert to 1-based indexing for Julia
                seed_xyz = [int(medoid[0] + 1), int(medoid[1] + 1), int(medoid[2] + 1)]
                
                new_item = item.copy()
                new_item["seed_xyz"] = seed_xyz
                new_item["component_volume"] = int(len(coords))
                new_manifest.append(new_item)
                
    with open(output_json, 'w') as f:
        json.dump(new_manifest, f, indent=2)

print("Processing train...")
process_manifest('/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/train_manifest.json', '/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/train_single_lesion_manifest.json')
print("Processing val...")
process_manifest('/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/val_manifest.json', '/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/val_single_lesion_manifest.json')
print("Done!")
