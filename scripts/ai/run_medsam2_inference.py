#!/usr/bin/env python3
"""
run_medsam2_inference.py

Run MedSAM2 on MRI for lesion segmentation using bounding box prompts.
Uses existing prostate mask or nnU-Net lesion as bbox prompt source.

MedSAM2 processes 3D volumes as "video" — slice by slice with temporal propagation.
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
from pathlib import Path
from PIL import Image

# Add MedSAM2 repo to path
MEDSAM2_DIR = "/workspaces/MedEye3d.jl/models/medsam2/MedSAM2"
sys.path.insert(0, MEDSAM2_DIR)

torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)


def resize_grayscale_to_rgb(array, image_size=512):
    """Convert 3D grayscale (D,H,W) to (D,3,image_size,image_size)."""
    d, h, w = array.shape
    resized = np.zeros((d, 3, image_size, image_size), dtype=np.float32)
    for i in range(d):
        img = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        resized[i] = np.array(img_resized).transpose(2, 0, 1)
    return resized


def mask3d_to_bbox(mask_3d, margin=5):
    """Get 3D bounding box from binary mask with margin.
    
    Returns bbox as (dim0_min, dim1_min, dim2_min, dim0_max, dim1_max, dim2_max)
    matching np.where axis ordering for the NIfTI array.
    """
    indices = np.where(mask_3d > 0)
    if len(indices[0]) == 0:
        return None
    shape = mask_3d.shape
    bbox = []
    for i in range(3):
        lo = max(0, np.min(indices[i]) - margin)
        hi = min(shape[i] - 1, np.max(indices[i]) + margin)
        bbox.extend([lo, hi])
    # Returns: dim0_min, dim0_max, dim1_min, dim1_max, dim2_min, dim2_max
    return np.array([bbox[0], bbox[2], bbox[4], bbox[1], bbox[3], bbox[5]])


def normalize_volume(vol_data, clip_min=None, clip_max=None):
    """Normalize volume to 0-255 uint8."""
    if clip_min is None:
        clip_min = np.percentile(vol_data[vol_data > 0], 0.5) if np.any(vol_data > 0) else vol_data.min()
    if clip_max is None:
        clip_max = np.percentile(vol_data[vol_data > 0], 99.5) if np.any(vol_data > 0) else vol_data.max()
    vol = np.clip(vol_data, clip_min, clip_max)
    vol = ((vol - clip_min) / (clip_max - clip_min + 1e-8) * 255).astype(np.uint8)
    return vol


def run_medsam2_3d(checkpoint_path, config_path, volume_path, bbox_3d, 
                   output_path, device="cuda:0", image_size=512):
    """
    Run MedSAM2 on a 3D volume with a bounding box prompt.
    
    bbox_3d: [x_min, y_min, z_min, x_max, y_max, z_max] in original voxel coords
    """
    from sam2.build_sam import build_sam2_video_predictor_npz
    
    print(f"[MedSAM2] Loading model from {checkpoint_path}...")
    predictor = build_sam2_video_predictor_npz(config_path, checkpoint_path, device=device)
    
    # Load volume
    print(f"[MedSAM2] Loading volume {volume_path}...")
    nii = nib.load(str(volume_path))
    vol_data = nii.get_fdata()
    
    # Normalize to uint8
    vol_uint8 = normalize_volume(vol_data)
    print(f"  Volume shape: {vol_uint8.shape}")
    
    # Transpose to (D, H, W) - depth first
    # NIfTI is (X, Y, Z), we need (Z, Y, X) for slice-by-slice
    vol_dhw = vol_uint8.transpose(2, 1, 0)  # (Z, Y, X)
    D, H, W = vol_dhw.shape
    print(f"  Transposed to (D,H,W): {vol_dhw.shape}")
    
    # Resize to MedSAM2 input format: (D, 3, 512, 512)
    vol_rgb = resize_grayscale_to_rgb(vol_dhw, image_size)
    print(f"  RGB resized: {vol_rgb.shape}")
    
    # Map bbox from NIfTI (dim0, dim1, dim2) to transposed (D=dim2, H=dim1, W=dim0)
    # bbox = (dim0_min, dim1_min, dim2_min, dim0_max, dim1_max, dim2_max)
    d0_min, d1_min, d2_min, d0_max, d1_max, d2_max = bbox_3d
    # After transpose(2,1,0): D=dim2, H=dim1, W=dim0
    d_min, d_max = int(d2_min), int(d2_max)
    h_min, h_max = int(d1_min), int(d1_max)
    w_min, w_max = int(d0_min), int(d0_max)
    
    # Clamp to volume dimensions
    d_min = max(0, min(d_min, D-1))
    d_max = max(0, min(d_max, D-1))
    
    # Scale bbox to image_size for the 2D prompt (in H,W space)
    scale_w = image_size / W
    scale_h = image_size / H
    
    bbox_2d_scaled = np.array([
        w_min * scale_w, h_min * scale_h,
        w_max * scale_w, h_max * scale_h
    ]).astype(np.float32)
    
    print(f"  NIfTI bbox: dim0=[{d0_min},{d0_max}], dim1=[{d1_min},{d1_max}], dim2=[{d2_min},{d2_max}]")
    print(f"  Transposed: d=[{d_min},{d_max}], h=[{h_min},{h_max}], w=[{w_min},{w_max}]")
    print(f"  Scaled 2D bbox (for 512x512): {bbox_2d_scaled}")
    
    # Middle slice for initial prompt
    mid_d = (d_min + d_max) // 2
    print(f"  Prompt on slice: {mid_d} (of {D} total)")
    
    # Initialize with video predictor
    vol_tensor = torch.from_numpy(vol_rgb).float()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = predictor.init_state(
            vol_tensor, 
            video_height=image_size,
            video_width=image_size,
            offload_video_to_cpu=True
        )
        
        # Add bbox prompt on the middle slice
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=mid_d,
            obj_id=1,
            box=bbox_2d_scaled,
        )
        print(f"  Initial mask logits shape: {out_mask_logits.shape}")
        
        # Propagate through all slices
        print(f"  Propagating through {D} slices...")
        seg_results = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            mask = (out_mask_logits[0, 0] > 0).cpu().numpy().astype(np.uint8)
            seg_results[out_frame_idx] = mask
        
        # Reset for reverse propagation
        predictor.reset_state(state)
        state = predictor.init_state(
            vol_tensor, 
            video_height=image_size,
            video_width=image_size,
            offload_video_to_cpu=True
        )
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=mid_d,
            obj_id=1,
            box=bbox_2d_scaled,
        )
        
    print(f"  Got segmentation for {len(seg_results)} slices")
    
    # Reconstruct 3D mask
    mask_3d_small = np.zeros((D, image_size, image_size), dtype=np.uint8)
    for z_idx, mask_2d in seg_results.items():
        if mask_2d.shape == (image_size, image_size):
            mask_3d_small[z_idx] = mask_2d
        else:
            # Resize
            m = Image.fromarray(mask_2d)
            m = m.resize((image_size, image_size), Image.NEAREST)
            mask_3d_small[z_idx] = np.array(m)
    
    # Resize mask back to original resolution
    mask_3d_orig = np.zeros((D, H, W), dtype=np.uint8)
    for z in range(D):
        if np.any(mask_3d_small[z]):
            m = Image.fromarray(mask_3d_small[z])
            m = m.resize((W, H), Image.NEAREST)
            mask_3d_orig[z] = np.array(m)
    
    # Transpose back to NIfTI order (X, Y, Z)
    mask_nifti = mask_3d_orig.transpose(2, 1, 0)  # (X, Y, Z)
    
    n_lesion = np.sum(mask_nifti > 0)
    vol_cc = n_lesion * np.prod(nii.header.get_zooms()[:3]) / 1000.0
    print(f"  Lesion voxels: {n_lesion} ({vol_cc:.3f} cc)")
    
    # Save
    out_nii = nib.Nifti1Image(mask_nifti.astype(np.uint8), nii.affine, nii.header)
    nib.save(out_nii, str(output_path))
    print(f"  Saved to: {output_path}")
    
    return mask_nifti


def main():
    parser = argparse.ArgumentParser(description="MedSAM2 MRI Lesion Inference")
    parser.add_argument("--checkpoint", 
                       default="/workspaces/MedEye3d.jl/models/medsam2/MedSAM2_latest.pt")
    parser.add_argument("--config",
                       default="sam2.1_hiera_t512.yaml")
    parser.add_argument("--input", required=True, help="Input MRI NIfTI")
    parser.add_argument("--prompt-mask", help="Mask to generate bbox prompt from (e.g., nnU-Net lesion)")
    parser.add_argument("--prostate-mask", help="Prostate gland mask for auto-bbox")
    parser.add_argument("--bbox", type=str, help="Manual bbox: x1,y1,z1,x2,y2,z2")
    parser.add_argument("--output", required=True, help="Output NIfTI mask")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    
    # Determine bounding box
    bbox_3d = None
    
    if args.bbox:
        bbox_3d = np.array([int(x) for x in args.bbox.split(",")])
        print(f"Using manual bbox: {bbox_3d}")
    elif args.prompt_mask:
        print(f"Generating bbox from prompt mask: {args.prompt_mask}")
        mask = nib.load(args.prompt_mask).get_fdata()
        bbox_3d = mask3d_to_bbox(mask, margin=10)
        if bbox_3d is None:
            print("ERROR: Prompt mask is empty!")
            sys.exit(1)
    elif args.prostate_mask:
        print(f"Generating bbox from prostate mask: {args.prostate_mask}")
        mask = nib.load(args.prostate_mask).get_fdata()
        bbox_3d = mask3d_to_bbox(mask, margin=5)
        if bbox_3d is None:
            print("ERROR: Prostate mask is empty!")
            sys.exit(1)
    else:
        print("ERROR: Must provide --bbox, --prompt-mask, or --prostate-mask")
        sys.exit(1)
    
    run_medsam2_3d(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        volume_path=args.input,
        bbox_3d=bbox_3d,
        output_path=args.output,
        device=args.device,
    )
    
    print("\n[SUCCESS] MedSAM2 inference complete!")


if __name__ == "__main__":
    main()
