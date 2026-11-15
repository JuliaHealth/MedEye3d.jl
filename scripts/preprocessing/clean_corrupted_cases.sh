#!/bin/bash
# Clean corrupted anatomy data for 6 cases that have oversized packed tensors
# After cleanup, these cases need full anatomy re-run via prepare_packed_case.py

CASES_DIR="/mnt/big/project_ssd/project_ssd/MedEye3d.jl/data/cases/Prostata_bimodal"
CORRUPTED_CASES="Pat6 Pat44 Pat79 Pat114 Pat117 Pat122"

echo "=============================================="
echo "Cleaning corrupted anatomy data"
echo "=============================================="

for pat in $CORRUPTED_CASES; do
    raw="$CASES_DIR/$pat/raw"
    echo ""
    echo "--- $pat ---"
    
    # Show sizes before cleanup
    for f in packed_anatomical_tensor.h5 packed_anatomical_tensor_registry.json final_results_gpu.h5 primary_masks.h5; do
        if [ -f "$raw/$f" ]; then
            sz=$(du -sh "$raw/$f" | cut -f1)
            echo "  DELETE: $f ($sz)"
            rm -f "$raw/$f"
        fi
    done
    
    # Remove segmentations dir (if unpacked from corrupted tensor)
    if [ -d "$raw/segmentations" ]; then
        n=$(ls "$raw/segmentations"/*.nii.gz 2>/dev/null | wc -l)
        echo "  DELETE: segmentations/ ($n files)"
        rm -rf "$raw/segmentations"
    fi
    
    echo "  ✓ Cleaned"
done

echo ""
echo "=============================================="
echo "Cleanup complete. Run full anatomy pipeline on these cases."
echo "=============================================="
