#!/usr/bin/env python3
"""
bone_subsegmentation.py
Extracts bone surface and bone marrow subsegments around bone lesions.

Sources:
  - Bone surface: from max_anatomy solid bones (TotalSegmentator) — outer shell
  - Bone marrow:  from Skellytour label 1 (trabecula/spongy bone) — interior
"""

import sys
import os
import argparse
import numpy as np
import nibabel as nib
from scipy import ndimage
from collections import Counter

def generate_bone_surface_shell(solid_bone_mask, lesion_mask, spacing, max_surface_dist_mm=25.0):
    """
    Compute bone surface as the outermost 1-voxel layer of a solid bone mask,
    restricted to within max_surface_dist_mm of the lesion.
    
    solid_bone_mask: bool array — full solid bone extent (from max_anatomy / TS)
    lesion_mask: bool array — lesion binary mask
    spacing: tuple — voxel spacing in mm
    """
    import torch
    import torch.nn.functional as F
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    solid = torch.from_numpy(solid_bone_mask).bool().to(device)
    lesion = torch.from_numpy(lesion_mask).bool().to(device)
    
    # Surface = bone voxels that have at least one non-bone neighbor (6-connected)
    padded = F.pad(solid.float().unsqueeze(0).unsqueeze(0), (1,1,1,1,1,1), mode='constant', value=0.0)
    
    kernel = torch.zeros(1, 1, 3, 3, 3, device=device)
    kernel[0, 0, 1, 1, 0] = 1
    kernel[0, 0, 1, 1, 2] = 1
    kernel[0, 0, 1, 0, 1] = 1
    kernel[0, 0, 1, 2, 1] = 1
    kernel[0, 0, 0, 1, 1] = 1
    kernel[0, 0, 2, 1, 1] = 1
    
    neighbor_sum = F.conv3d(padded, kernel, stride=1, padding=0).squeeze(0).squeeze(0)
    # Surface = bone voxels where at least one of 6 neighbors is NOT bone
    surface = solid & (neighbor_sum < 6)
    
    # Restrict surface to within max_surface_dist_mm of the lesion
    lesion_idx = torch.nonzero(lesion).float()
    if len(lesion_idx) == 0:
        return np.zeros_like(solid_bone_mask, dtype=bool)
        
    surface_idx = torch.nonzero(surface).float()
    if len(surface_idx) == 0:
        return np.zeros_like(solid_bone_mask, dtype=bool)
    
    sp = torch.tensor(spacing, device=device, dtype=torch.float32)
    lesion_phys = lesion_idx * sp
    surface_phys = surface_idx * sp
    
    # Sample lesion points to keep distance computation manageable
    step = max(1, len(lesion_phys) // 500)
    lesion_sampled = lesion_phys[::step]
    
    result = torch.zeros_like(surface)
    dists = torch.cdist(surface_phys.unsqueeze(0), lesion_sampled.unsqueeze(0)).squeeze(0)
    min_dists, _ = torch.min(dists, dim=1)
    valid_mask = min_dists <= max_surface_dist_mm
    valid_idx = surface_idx[valid_mask].long()
    if len(valid_idx) > 0:
        result[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]] = True
    
    return result.cpu().numpy().astype(bool)


def extract_bone_fragments(lesion_path, bone_path, out_surface_path, out_marrow_path, 
                           ct_path=None, max_anatomy_path=None, bone_label_ids_str=None):
    """
    Extract bone surface and bone marrow subsegments around a bone metastasis.
    
    Parameters:
      lesion_path:        Path to binary lesion mask (NIfTI)
      bone_path:          Path to Skellytour subseg (label 1=trabecula, 2=cortex)
      out_surface_path:   Output path for bone surface mask
      out_marrow_path:    Output path for bone marrow mask
      ct_path:            Optional CT volume for HU-based refinement
      max_anatomy_path:   Path to max_anatomy.nii.gz (solid bones from TS for surface)
      bone_label_ids_str: Comma-separated list of bone label IDs in max_anatomy
    """
    lesion_img = nib.load(lesion_path)
    np_lesion = lesion_img.get_fdata() > 0
    affine = lesion_img.affine
    header = lesion_img.header
    spacing = header.get_zooms()[:3]

    bone_img = nib.load(bone_path)
    np_bone = bone_img.get_fdata()  # Skellytour: 0=bg, 1=trabecula, 2=cortex

    bone_surface_fragment = np.zeros_like(np_lesion, dtype=np.uint8)
    bone_marrow_fragment = np.zeros_like(np_lesion, dtype=np.uint8)

    coords = np.where(np_lesion > 0)
    if len(coords[0]) == 0:
        print("No lesion voxels found. Writing empty masks.")
        nib.save(nib.Nifti1Image(bone_surface_fragment, affine, header), out_surface_path)
        nib.save(nib.Nifti1Image(bone_marrow_fragment, affine, header), out_marrow_path)
        return

    # Bounding box with 30mm margin
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
    crop_skelly = np_bone[x_min:x_max, y_min:y_max, z_min:z_max]

    # Identify which bone the lesion overlaps in Skellytour
    bone_mask = (crop_skelly > 0)
    labeled_bones, num_features = ndimage.label(bone_mask)

    target_bone_label = 0
    if num_features > 0:
        overlap_region = crop_lesion_mask & (labeled_bones > 0)
        if np.any(overlap_region):
            lesion_overlap = labeled_bones[overlap_region]
            target_bone_label = Counter(lesion_overlap).most_common(1)[0][0]

    if target_bone_label == 0 and num_features > 0:
        # Find nearest bone via distance transform
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
        target_bone_mask_skelly = (labeled_bones == target_bone_label)
        
        # === BONE SURFACE: from max_anatomy solid bones (preferred) ===
        if max_anatomy_path is not None and bone_label_ids_str is not None:
            bone_label_ids = [int(x) for x in bone_label_ids_str.split(",") if x.strip()]
            max_anat_img = nib.load(max_anatomy_path)
            np_max_anat = max_anat_img.get_fdata()
            
            # Ensure dimensions match
            if np_max_anat.shape == np_lesion.shape:
                solid_bone_full = np.isin(np_max_anat, bone_label_ids)
                crop_solid_bone = solid_bone_full[x_min:x_max, y_min:y_max, z_min:z_max]
                
                # Only use the connected component that overlaps with the target Skellytour bone
                labeled_solid, n_solid = ndimage.label(crop_solid_bone)
                target_solid_label = 0
                if n_solid > 0:
                    overlap_solid = labeled_solid[target_bone_mask_skelly & (labeled_solid > 0)]
                    if len(overlap_solid) > 0:
                        target_solid_label = Counter(overlap_solid).most_common(1)[0][0]
                
                if target_solid_label > 0:
                    bone_for_surface = (labeled_solid == target_solid_label)
                else:
                    # Fallback: use all solid bone in crop
                    bone_for_surface = crop_solid_bone
                
                bone_surface_fragment[x_min:x_max, y_min:y_max, z_min:z_max] = generate_bone_surface_shell(
                    bone_for_surface.astype(np.uint8), crop_lesion_mask.astype(np.uint8), spacing
                ).astype(np.uint8)
            else:
                print(f"  WARNING: max_anatomy shape {np_max_anat.shape} != lesion shape {np_lesion.shape}, falling back to Skellytour cortex")
                bone_surface_fragment[x_min:x_max, y_min:y_max, z_min:z_max] = generate_bone_surface_shell(
                    target_bone_mask_skelly.astype(np.uint8), crop_lesion_mask.astype(np.uint8), spacing
                ).astype(np.uint8)
        else:
            # Fallback: use Skellytour as the bone mask for surface
            crop_bone_for_surface = target_bone_mask_skelly
            if ct_path is not None:
                ct_img = nib.load(ct_path)
                np_ct = ct_img.get_fdata()
                crop_ct = np_ct[x_min:x_max, y_min:y_max, z_min:z_max]
                dilated_bone = ndimage.binary_dilation(target_bone_mask_skelly, iterations=3)
                crop_bone_for_surface = (crop_ct > 150) & dilated_bone
            
            bone_surface_fragment[x_min:x_max, y_min:y_max, z_min:z_max] = generate_bone_surface_shell(
                crop_bone_for_surface.astype(np.uint8), crop_lesion_mask.astype(np.uint8), spacing
            ).astype(np.uint8)

        # === BONE MARROW: from Skellytour label 1 (trabecula) ===
        crop_trabecula = (crop_skelly == 1) & target_bone_mask_skelly
        
        if np.any(crop_trabecula):
            # Find marrow region near the lesion (within equivalent sphere radius)
            crop_vicinity_dist = ndimage.distance_transform_edt(~crop_lesion_mask, sampling=spacing)
            marrow_dist_from_lesion = np.where(crop_trabecula, crop_vicinity_dist, np.inf)
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

                bone_marrow_fragment[x_min:x_max, y_min:y_max, z_min:z_max] = (
                    (crop_trabecula & marrow_sphere) & ~crop_lesion_mask
                ).astype(np.uint8)

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
    parser.add_argument("--bone", required=True, help="Path to Skellytour subseg NIfTI (label 1=trabecula, 2=cortex)")
    parser.add_argument("--out-surface", required=True, help="Output path for bone surface mask")
    parser.add_argument("--out-marrow", required=True, help="Output path for bone marrow mask")
    parser.add_argument("--ct", required=False, default=None, help="Optional CT volume for HU-based fallback")
    parser.add_argument("--max-anatomy", required=False, default=None, help="Path to max_anatomy.nii.gz (solid bones for surface)")
    parser.add_argument("--bone-labels", required=False, default=None, help="Comma-separated bone label IDs from max_anatomy")
    args = parser.parse_args()

    extract_bone_fragments(args.lesion, args.bone, args.out_surface, args.out_marrow, 
                           args.ct, getattr(args, 'max_anatomy'), args.bone_labels)
