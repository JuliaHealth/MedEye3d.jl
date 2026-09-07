#!/usr/bin/env python3
"""
run_prostate158_inference.py

Run Prostate158 community model (MONAI UNet) for MRI lesion segmentation.
Weights: https://zenodo.org/records/7040585
Paper:   Adams et al., "Prostate158", Comp Bio Med 2022

Tumor model:
  - Input: 3 channels (T2W + ADC + DWI)
  - Preprocessing: 0.5mm isotropic, RAS orientation, ScaleIntensity(0,1), NormalizeIntensity
  - Architecture: MONAI UNet 3D, channels=(32,64,128,256,512), strides=(2,2,2,2), num_res_units=2
  - Output: 2 classes (background, tumor)

Anatomy model:
  - Input: 1 channel (T2W)
  - Output: 3 classes (background, TZ, PZ)
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import torch
from pathlib import Path


def build_unet(in_channels, out_channels, device="cuda:0"):
    """Build UNet matching Prostate158 config.
    
    Architecture inferred from checkpoint weight shapes:
    - Encoder: 16 → 32 → 64 → 128 → 256 → 512
    - 5 downsample steps, stride=2 each
    - 2 residual units per block (unit0..unit3 = 2 conv pairs)
    """
    from monai.networks.nets import UNet
    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2, 2),
        num_res_units=2,
    )
    model.to(device)
    model.eval()
    return model


def load_weights(model, weights_path, device="cuda:0"):
    """Load checkpoint, trying various key patterns."""
    state = torch.load(weights_path, map_location=device, weights_only=False)
    
    # Try different checkpoint formats
    for key in ["model_state_dict", "state_dict", "model", "net"]:
        if key in state:
            print(f"  Loading from key: '{key}'")
            model.load_state_dict(state[key], strict=False)
            return model
    
    # Try direct state_dict
    try:
        model.load_state_dict(state, strict=False)
        print(f"  Loaded directly as state_dict")
        return model
    except Exception as e:
        print(f"  Direct load failed: {e}")
    
    # Inspect checkpoint structure
    if isinstance(state, dict):
        print(f"  Checkpoint keys: {list(state.keys())[:10]}")
        # Check if first value looks like a tensor
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                print(f"  Looks like raw state_dict, trying again...")
                model.load_state_dict(state, strict=False)
                return model
            break
    
    raise RuntimeError(f"Cannot load weights from {weights_path}")


def preprocess_volume(nifti_path, target_spacing=(0.5, 0.5, 0.5)):
    """Load and preprocess a NIfTI volume for Prostate158."""
    from monai.transforms import (
        Compose, LoadImage, EnsureChannelFirst, Orientation,
        Spacing, ScaleIntensity, NormalizeIntensity, EnsureType
    )
    
    transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        Spacing(pixdim=target_spacing, mode="bilinear"),
        ScaleIntensity(minv=0, maxv=1),
        NormalizeIntensity(nonzero=True),
        EnsureType(),
    ])
    
    return transforms(str(nifti_path))


def run_tumor_inference(weights_path, t2_path, adc_path, dwi_path, output_path, device="cuda:0"):
    """Run Prostate158 TUMOR model (3-channel: T2+ADC+DWI)."""
    print(f"\n[Prostate158 TUMOR] Loading model...")
    model = build_unet(in_channels=3, out_channels=2, device=device)
    model = load_weights(model, weights_path, device)
    
    print(f"[Prostate158 TUMOR] Preprocessing volumes...")
    t2_vol = preprocess_volume(t2_path)
    adc_vol = preprocess_volume(adc_path)
    dwi_vol = preprocess_volume(dwi_path)
    
    print(f"  T2:  {t2_vol.shape}")
    print(f"  ADC: {adc_vol.shape}")
    print(f"  DWI: {dwi_vol.shape}")
    
    # Resample ADC/DWI to match T2 grid if shapes differ
    if adc_vol.shape != t2_vol.shape:
        print(f"  Resampling ADC to T2 grid...")
        adc_vol = torch.nn.functional.interpolate(
            adc_vol.unsqueeze(0), size=t2_vol.shape[1:], mode='trilinear', align_corners=False
        ).squeeze(0)
    if dwi_vol.shape != t2_vol.shape:
        print(f"  Resampling DWI to T2 grid...")
        dwi_vol = torch.nn.functional.interpolate(
            dwi_vol.unsqueeze(0), size=t2_vol.shape[1:], mode='trilinear', align_corners=False
        ).squeeze(0)
    
    # Stack channels: [1, 3, D, H, W]
    vol = torch.cat([t2_vol, adc_vol, dwi_vol], dim=0).unsqueeze(0).to(device)
    print(f"  Combined input: {vol.shape}")
    
    print(f"[Prostate158 TUMOR] Running sliding window inference...")
    from monai.inferers import sliding_window_inference
    
    with torch.no_grad():
        # Use roi_size that fits in GPU memory
        roi = (96, 96, 96)
        pred = sliding_window_inference(
            vol, roi_size=roi, sw_batch_size=4,
            predictor=model, overlap=0.25, mode="gaussian"
        )
        mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    
    print(f"  Prediction shape: {mask.shape}")
    print(f"  Tumor voxels: {np.sum(mask > 0)}")
    
    # Save — use original T2 affine for reference
    ref = nib.load(str(t2_path))
    # Note: mask is in 0.5mm isotropic RAS space, not original T2 space
    # Create appropriate affine for 0.5mm isotropic
    affine = np.eye(4)
    affine[0, 0] = 0.5
    affine[1, 1] = 0.5
    affine[2, 2] = 0.5
    # Copy origin from reference
    ref_aff = ref.affine
    affine[:3, 3] = ref_aff[:3, 3]
    
    out_nii = nib.Nifti1Image(mask, affine)
    nib.save(out_nii, str(output_path))
    print(f"  Saved to: {output_path}")
    
    return mask


def run_anatomy_inference(weights_path, t2_path, output_path, device="cuda:0"):
    """Run Prostate158 ANATOMY model (1-channel: T2 only)."""
    print(f"\n[Prostate158 ANATOMY] Loading model...")
    model = build_unet(in_channels=1, out_channels=3, device=device)
    model = load_weights(model, weights_path, device)
    
    print(f"[Prostate158 ANATOMY] Preprocessing T2 volume...")
    t2_vol = preprocess_volume(t2_path)
    vol = t2_vol.unsqueeze(0).to(device)
    print(f"  Input: {vol.shape}")
    
    print(f"[Prostate158 ANATOMY] Running sliding window inference...")
    from monai.inferers import sliding_window_inference
    
    with torch.no_grad():
        pred = sliding_window_inference(
            vol, roi_size=(96, 96, 96), sw_batch_size=4,
            predictor=model, overlap=0.25, mode="gaussian"
        )
        mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    
    labels = {0: "Background", 1: "TZ", 2: "PZ"}
    unique, counts = np.unique(mask, return_counts=True)
    for u, c in zip(unique, counts):
        vol_cc = c * 0.5 * 0.5 * 0.5 / 1000.0
        print(f"  Label {u} ({labels.get(u, '?')}): {c} voxels ({vol_cc:.2f} cc)")
    
    ref = nib.load(str(t2_path))
    affine = np.eye(4)
    affine[0, 0] = 0.5
    affine[1, 1] = 0.5
    affine[2, 2] = 0.5
    affine[:3, 3] = ref.affine[:3, 3]
    
    out_nii = nib.Nifti1Image(mask, affine)
    nib.save(out_nii, str(output_path))
    print(f"  Saved to: {output_path}")
    
    return mask


def main():
    parser = argparse.ArgumentParser(description="Prostate158 MRI Inference")
    parser.add_argument("--mode", choices=["tumor", "anatomy", "both"], default="both")
    parser.add_argument("--tumor-weights", default="/workspaces/MedEye3d.jl/models/prostate158/tumor.pt")
    parser.add_argument("--anatomy-weights", default="/workspaces/MedEye3d.jl/models/prostate158/anatomy.pt")
    parser.add_argument("--t2", required=True, help="T2-weighted MRI NIfTI")
    parser.add_argument("--adc", help="ADC map NIfTI (required for tumor)")
    parser.add_argument("--dwi", help="DWI b2000 NIfTI (required for tumor)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode in ("tumor", "both"):
        if not args.adc or not args.dwi:
            print("ERROR: --adc and --dwi required for tumor mode")
            sys.exit(1)
        run_tumor_inference(
            args.tumor_weights, args.t2, args.adc, args.dwi,
            os.path.join(args.output_dir, "prostate158_tumor.nii.gz"),
            args.device
        )
    
    if args.mode in ("anatomy", "both"):
        run_anatomy_inference(
            args.anatomy_weights, args.t2,
            os.path.join(args.output_dir, "prostate158_anatomy.nii.gz"),
            args.device
        )
    
    print("\n[SUCCESS] Prostate158 inference complete!")


if __name__ == "__main__":
    main()
