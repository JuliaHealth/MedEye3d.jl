#!/usr/bin/env python3
"""
preprocess_edge_gt.py — Generate soft edge ground truth from TotalSegmentator masks.

For each patch with TotalSegmentator data:
1. Load TotalSeg mask (64×64×64, integer labels 0-117)
2. Compute binary edge map (6-connected neighbor differences)
3. Apply soft distance decay (Gaussian, sigma=0.7)
4. Save as Float32 NIfTI: <patch_id>_edge_gt.nii.gz

Skips patches without TotalSeg (has_totalseg == False).
Updates manifests with edge_gt_path field.
"""

import json
import os
import sys
import time
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt

TRAIN_MANIFEST = "/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/train_single_lesion_manifest.json"
VAL_MANIFEST = "/mnt/big/project_ssd/project_ssd/semiautomatic/nifti_patches_clean/val_single_lesion_manifest.json"
SIGMA = 0.7  # Gaussian decay sigma for soft edges


def compute_soft_edge_gt(totalseg: np.ndarray, sigma: float = 0.7) -> np.ndarray:
    """
    Compute soft edge ground truth from TotalSegmentator integer label volume.
    
    Args:
        totalseg: (X, Y, Z) integer array with labels 0-117
        sigma: Gaussian decay sigma for soft edges
    
    Returns:
        soft_edge: (X, Y, Z) float32 array in [0, 1]
    """
    labels = totalseg.astype(np.int32)
    
    # Binary edge: voxel where any 6-connected neighbor has a different label
    edges = np.zeros(labels.shape, dtype=bool)
    for axis in range(3):
        # Compare with +1 neighbor along each axis
        slc_a = [slice(None)] * 3
        slc_b = [slice(None)] * 3
        slc_a[axis] = slice(0, -1)
        slc_b[axis] = slice(1, None)
        diff = labels[tuple(slc_a)] != labels[tuple(slc_b)]
        # Mark both sides of the edge
        edges[tuple(slc_a)] |= diff
        edges[tuple(slc_b)] |= diff
    
    if not edges.any():
        return np.zeros(labels.shape, dtype=np.float32)
    
    # Distance transform from edge voxels (inverted: distance FROM edges)
    # distance_transform_edt computes distance from zeros, so we pass ~edges
    dist = distance_transform_edt(~edges)
    
    # Soft edge: Gaussian decay from boundary
    soft_edge = np.exp(-dist**2 / (2 * sigma**2))
    
    # Normalize to [0, 1] (should already be, but clamp for safety)
    soft_edge = np.clip(soft_edge, 0.0, 1.0).astype(np.float32)
    
    return soft_edge


def process_manifest(manifest_path: str, sigma: float = 0.7) -> int:
    """Process all patches in a manifest, generate edge GT, update manifest."""
    with open(manifest_path, 'r') as f:
        items = json.load(f)
    
    total = len(items)
    processed = 0
    skipped = 0
    already_exists = 0
    
    t0 = time.time()
    
    for i, item in enumerate(items):
        # Determine output path
        patch_dir = os.path.dirname(item["ct_path"])
        patch_id = item["patch_id"]
        edge_gt_path = os.path.join(patch_dir, f"{patch_id}_edge_gt.nii.gz")
        
        if not item.get("has_totalseg", False):
            item["edge_gt_path"] = ""
            skipped += 1
            continue
        
        # Check if already processed
        if os.path.exists(edge_gt_path):
            item["edge_gt_path"] = edge_gt_path
            already_exists += 1
            continue
        
        # Load TotalSegmentator mask
        try:
            ts_nii = nib.load(item["totalseg_path"])
            ts_data = ts_nii.get_fdata().astype(np.int32)
        except Exception as e:
            print(f"  Error loading {item['totalseg_path']}: {e}")
            item["edge_gt_path"] = ""
            skipped += 1
            continue
        
        # Check if mask is actually empty (all zeros = missing TotalSeg)
        if ts_data.max() == 0:
            item["edge_gt_path"] = ""
            skipped += 1
            continue
        
        # Compute soft edge GT
        soft_edge = compute_soft_edge_gt(ts_data, sigma=sigma)
        
        # Save as NIfTI with same affine as source
        edge_nii = nib.Nifti1Image(soft_edge, ts_nii.affine, ts_nii.header)
        edge_nii.header.set_data_dtype(np.float32)
        nib.save(edge_nii, edge_gt_path)
        
        item["edge_gt_path"] = edge_gt_path
        processed += 1
        
        if (processed + already_exists) % 100 == 0:
            elapsed = time.time() - t0
            rate = (processed + already_exists) / elapsed
            print(f"  [{processed + already_exists}/{total}] {rate:.1f} patches/sec, "
                  f"processed={processed}, existed={already_exists}, skipped={skipped}")
    
    # Save updated manifest
    with open(manifest_path, 'w') as f:
        json.dump(items, f, indent=2)
    
    elapsed = time.time() - t0
    print(f"  Done: {processed} new + {already_exists} existing + {skipped} skipped "
          f"= {total} total in {elapsed:.1f}s")
    
    return processed + already_exists


def verify_output():
    """Verify the generated edge GT files."""
    print("\n=== Verification ===")
    
    for name, path in [("Train", TRAIN_MANIFEST), ("Val", VAL_MANIFEST)]:
        with open(path, 'r') as f:
            items = json.load(f)
        
        has_edge = sum(1 for x in items if x.get("edge_gt_path", ""))
        print(f"{name}: {has_edge}/{len(items)} have edge_gt_path")
        
        # Check first valid sample
        for item in items:
            edge_path = item.get("edge_gt_path", "")
            if edge_path and os.path.exists(edge_path):
                d = nib.load(edge_path).get_fdata()
                print(f"  Sample: {os.path.basename(edge_path)}")
                print(f"  Shape: {d.shape}, dtype: {d.dtype}")
                print(f"  Min: {d.min():.4f}, Max: {d.max():.4f}, Mean: {d.mean():.4f}")
                print(f"  Edge fraction (>0.5): {(d > 0.5).mean():.4f}")
                print(f"  Soft edge fraction (>0.01): {(d > 0.01).mean():.4f}")
                break


if __name__ == "__main__":
    print("=" * 60)
    print("  Edge GT Preprocessing from TotalSegmentator")
    print(f"  Sigma: {SIGMA}")
    print("=" * 60)
    
    print(f"\nProcessing TRAIN manifest: {TRAIN_MANIFEST}")
    process_manifest(TRAIN_MANIFEST, sigma=SIGMA)
    
    print(f"\nProcessing VAL manifest: {VAL_MANIFEST}")
    process_manifest(VAL_MANIFEST, sigma=SIGMA)
    
    verify_output()
    print("\n✅ Edge GT preprocessing complete!")
