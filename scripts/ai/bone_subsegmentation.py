#!/usr/bin/env python3
"""
bone_subsegmentation.py
Replicates the bone marrow and bone surface subsegmentation logic from slicer_lesion_text_extension.
"""

import sys
import os
import argparse
import numpy as np
import nibabel as nib
from scipy import ndimage
from collections import Counter

def generate_bone_subsegments_pt(crop_lesion_np, crop_bone_np, spacing, max_surface_dist_mm=25.0):
    import torch
    import torch.nn.functional as F
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    crop_lesion = torch.from_numpy(crop_lesion_np).bool().to(device)
    crop_bone = torch.from_numpy(crop_bone_np > 0).bool().to(device)
    
    union = crop_lesion | crop_bone
    
    union_float = union.float().unsqueeze(0).unsqueeze(0)
    dilated = F.max_pool3d(union_float, kernel_size=3, stride=1, padding=1)
    dilated_bool = dilated.squeeze(0).squeeze(0) > 0
    
    dilated_inv = (~dilated_bool).float().unsqueeze(0).unsqueeze(0)
    eroded_inv = F.max_pool3d(dilated_inv, kernel_size=3, stride=1, padding=1)
    closed_bone = ~(eroded_inv.squeeze(0).squeeze(0) > 0)
    
    padded = F.pad(closed_bone.float().unsqueeze(0).unsqueeze(0), (1,1,1,1,1,1), mode='constant', value=0.0)
    
    kernel = torch.zeros(1, 1, 3, 3, 3, device=device)
    kernel[0, 0, 1, 1, 0] = 1
    kernel[0, 0, 1, 1, 2] = 1
    kernel[0, 0, 1, 0, 1] = 1
    kernel[0, 0, 1, 2, 1] = 1
    kernel[0, 0, 0, 1, 1] = 1
    kernel[0, 0, 2, 1, 1] = 1
    
    neighbor_sum = F.conv3d(padded, kernel, stride=1, padding=0).squeeze(0).squeeze(0)
    crop_cortical = closed_bone & (neighbor_sum < 6)
    
    lesion_idx = torch.nonzero(crop_lesion).float()
    if len(lesion_idx) == 0:
        return np.zeros_like(crop_lesion_np, dtype=bool)
        
    cortical_idx = torch.nonzero(crop_cortical).float()
    sp = torch.tensor(spacing, device=device, dtype=torch.float32)
    lesion_phys = lesion_idx * sp
    cortical_phys = cortical_idx * sp
    
    step = max(1, len(lesion_phys) // 500)
    lesion_sampled = lesion_phys[::step]
    
    crop_surface = torch.zeros_like(crop_cortical)
    if len(cortical_phys) > 0 and len(lesion_sampled) > 0:
        dists = torch.cdist(cortical_phys.unsqueeze(0), lesion_sampled.unsqueeze(0)).squeeze(0)
        min_dists, _ = torch.min(dists, dim=1)
        valid_cortical_mask = min_dists <= max_surface_dist_mm
        valid_cortical_idx = cortical_idx[valid_cortical_mask].long()
        if len(valid_cortical_idx) > 0:
            crop_surface[valid_cortical_idx[:, 0], valid_cortical_idx[:, 1], valid_cortical_idx[:, 2]] = True
            
    crop_surface = crop_surface & crop_bone
    
    return crop_surface.cpu().numpy().astype(bool)

def extract_bone_fragments(lesion_path, bone_path, out_surface_path, out_marrow_path, ct_path=None):
    lesion_img = nib.load(lesion_path)
    np_lesion = lesion_img.get_fdata() > 0
    affine = lesion_img.affine
    header = lesion_img.header
    spacing = header.get_zooms()[:3] # (sx, sy, sz)

    bone_img = nib.load(bone_path)
    np_bone = bone_img.get_fdata()

    bone_surface_fragment = np.zeros_like(np_lesion, dtype=np.uint8)
    bone_marrow_fragment = np.zeros_like(np_lesion, dtype=np.uint8)

    coords = np.where(np_lesion > 0)
    if len(coords[0]) == 0:
        print("No lesion voxels found. Writing empty masks.")
        nib.save(nib.Nifti1Image(bone_surface_fragment, affine, header), out_surface_path)
        nib.save(nib.Nifti1Image(bone_marrow_fragment, affine, header), out_marrow_path)
        return

    # In nibabel array, indices are (x, y, z)
    x_min, x_max = np.min(coords[0]), np.max(coords[0])
    y_min, y_max = np.min(coords[1]), np.max(coords[1])
    z_min, z_max = np.min(coords[2]), np.max(coords[2])

    margin_x = int(np.ceil(30.0 / spacing[0]))
    margin_y = int(np.ceil(30.0 / spacing[1]))
    margin_z = int(np.ceil(30.0 / spacing[2]))

    x_min = max(0, x_min - margin_x); x_max = min(np_lesion.shape[0], x_max + margin_x + 1)
    y_min = max(0, y_min - margin_y); y_max = min(np_lesion.shape[1], y_max + margin_y + 1)
    z_min = max(0, z_min - margin_z); z_max = min(np_lesion.shape[2], z_max + margin_z + 1)

    crop_lesion_mask = np_lesion[x_min:x_max, y_min:y_max, z_min:z_max]
    crop_skelly_mask = np_bone[x_min:x_max, y_min:y_max, z_min:z_max]

    bone_mask = (crop_skelly_mask > 0)
    labeled_bones, num_features = ndimage.label(bone_mask)

    target_bone_label = 0
    if num_features > 0:
        lesion_overlap = labeled_bones[crop_lesion_mask & (labeled_bones > 0)]
        if len(lesion_overlap) > 0:
            target_bone_label = Counter(lesion_overlap).most_common(1)[0][0]

    if target_bone_label == 0 and num_features > 0:
        # Distance to nearest bone
        dist, nearest_idx = ndimage.distance_transform_edt(labeled_bones == 0, return_indices=True, sampling=spacing)
        lesion_coords = np.where(crop_lesion_mask)
        if len(lesion_coords[0]) > 0:
            lesion_dists = dist[lesion_coords]
            min_dist_idx = np.argmin(lesion_dists)
            best_x = nearest_idx[0, lesion_coords[0][min_dist_idx], lesion_coords[1][min_dist_idx], lesion_coords[2][min_dist_idx]]
            best_y = nearest_idx[1, lesion_coords[0][min_dist_idx], lesion_coords[1][min_dist_idx], lesion_coords[2][min_dist_idx]]
            best_z = nearest_idx[2, lesion_coords[0][min_dist_idx], lesion_coords[1][min_dist_idx], lesion_coords[2][min_dist_idx]]
            target_bone_label = labeled_bones[best_x, best_y, best_z]

    if target_bone_label > 0:
        target_bone_mask = (labeled_bones == target_bone_label)
        
        if ct_path is not None:
            ct_img = nib.load(ct_path)
            np_ct = ct_img.get_fdata()
            crop_ct = np_ct[x_min:x_max, y_min:y_max, z_min:z_max]
            
            # Dilate the target bone mask by a few mm to account for rigid deformation mismatches
            # spacing is typically [1.5, 1.5, 2.0]. 5mm is ~3 voxels
            dilated_bone_mask = ndimage.binary_dilation(target_bone_mask, iterations=3)
            
            # Threshold CT to find the actual bone (HU > 150) inside the dilated region
            refined_bone_mask = (crop_ct > 150) & dilated_bone_mask
            
            # Update target_bone_mask and crop_skelly_mask
            target_bone_mask = refined_bone_mask
            crop_skelly_mask = np.where(target_bone_mask, 1, 0)
        else:
            crop_skelly_mask = np.where(target_bone_mask, crop_skelly_mask, 0)

        # Check if skellytour cortical (2) and marrow (1) labels exist
        # Instead of using Skellytour label 1, compute marrow using morphological erosion (Slicer fallback_to_ts behavior)
        crop_marrow = ndimage.binary_erosion(target_bone_mask, iterations=2)
        crop_marrow = crop_marrow & (crop_skelly_mask > 0) # ensure it is within the bone mask
        
        # Morphological Surface Extraction (1-voxel thick) based on the full skellytour bone mask
        bone_surface_fragment[x_min:x_max, y_min:y_max, z_min:z_max] = generate_bone_subsegments_pt(
            crop_lesion_mask, (crop_skelly_mask > 0).astype(np.uint8), spacing
        ).astype(np.uint8)

        # Skellytour Marrow Extraction (based on label 1)
        if np.any(crop_marrow):
            crop_vicinity_dist = ndimage.distance_transform_edt(~crop_lesion_mask, sampling=spacing)
            marrow_dist_from_lesion = np.where(crop_marrow, crop_vicinity_dist, np.inf)
            min_idx_marr = np.argmin(marrow_dist_from_lesion)
            if marrow_dist_from_lesion[np.unravel_index(min_idx_marr, marrow_dist_from_lesion.shape)] != np.inf:
                marrow_centroid_vox = np.unravel_index(min_idx_marr, marrow_dist_from_lesion.shape)

                voxel_vol = spacing[0] * spacing[1] * spacing[2]
                lesion_vol_vox = np.sum(crop_lesion_mask)
                lesion_vol_mm3 = lesion_vol_vox * voxel_vol
                R_L = max(3.0, (3.0 * lesion_vol_mm3 / (4.0 * np.pi))**(1.0/3.0))

                crop_x, crop_y, crop_z = crop_lesion_mask.shape
                x_indices, y_indices, z_indices = np.indices((crop_x, crop_y, crop_z))
                x_dist_m = (x_indices - marrow_centroid_vox[0]) * spacing[0]
                y_dist_m = (y_indices - marrow_centroid_vox[1]) * spacing[1]
                z_dist_m = (z_indices - marrow_centroid_vox[2]) * spacing[2]
                dist_from_marrow = np.sqrt(x_dist_m**2 + y_dist_m**2 + z_dist_m**2)
                marrow_sphere = (dist_from_marrow <= R_L)

                bone_marrow_fragment[x_min:x_max, y_min:y_max, z_min:z_max] = ((crop_marrow & marrow_sphere) & ~crop_lesion_mask).astype(np.uint8)

    # Discard fragments smaller than 8 voxels
    if np.sum(bone_surface_fragment) < 8:
        bone_surface_fragment = np.zeros_like(np_lesion, dtype=np.uint8)
    if np.sum(bone_marrow_fragment) < 8:
        bone_marrow_fragment = np.zeros_like(np_lesion, dtype=np.uint8)

    print(f"Extraction complete: bone_surface voxels={np.sum(bone_surface_fragment)}, bone_marrow voxels={np.sum(bone_marrow_fragment)}")
    nib.save(nib.Nifti1Image(bone_surface_fragment, affine, header), out_surface_path)
    nib.save(nib.Nifti1Image(bone_marrow_fragment, affine, header), out_marrow_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract bone surface and marrow subsegments around a bone lesion.")
    parser.add_argument("--lesion", required=True, help="Path to lesion NIfTI binary mask")
    parser.add_argument("--bone", required=True, help="Path to bone / skellytour NIfTI segmentation")
    parser.add_argument("--out-surface", required=True, help="Output path for bone surface mask")
    parser.add_argument("--out-marrow", required=True, help="Output path for bone marrow mask")
    parser.add_argument("--ct", required=False, default=None, help="Optional path to the CT volume for refining the bone boundaries")
    args = parser.parse_args()

    extract_bone_fragments(args.lesion, args.bone, args.out_surface, args.out_marrow, args.ct)
