#!/bin/bash
CT_PATH="/mnt/big/project_ssd/project_ssd/MedEye3d.jl/data/pat_6_files/Fixed_CT_Volume_0.nii.gz"
OUT_DIR="/mnt/big/project_ssd/project_ssd/MedEye3d.jl/data/pat_6_files/anatomy_out"

mkdir -p $OUT_DIR
docker exec -e PYTHONPATH=/mnt/big/project_ssd/project_ssd/lymph_node_rules medeye3d-ai python3 /mnt/big/project_ssd/project_ssd/lymph_node_rules/src/anatomy_segmentation/run_segmentation.py "$CT_PATH" "$OUT_DIR" --fast --models nv_segment
