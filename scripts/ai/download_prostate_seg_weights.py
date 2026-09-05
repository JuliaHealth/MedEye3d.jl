#!/usr/bin/env python3
"""
download_prostate_seg_weights.py
Downloads the pre-trained nnU-Net weights from Hugging Face:
zshojaei/prostate_seg_weights -> models/Dataset501_ProstateZonesLesions
"""

import os
import sys
from huggingface_hub import snapshot_download

def main():
    target_dir = "/mnt/big/project_ssd/project_ssd/MedEye3d.jl/models/Dataset501_ProstateZonesLesions"
    os.makedirs(target_dir, exist_ok=True)
    print(f"Downloading prostate_seg_weights to {target_dir}...")
    
    # Download dataset.json, plans, and folds
    snapshot_download(
        repo_id="zshojaei/prostate_seg_weights",
        local_dir=target_dir,
        repo_type="model",
        ignore_patterns=["*.png", "*.ipynb_checkpoints*", "*validation*"]
    )
    print("✅ Weights download complete!")

if __name__ == "__main__":
    main()
