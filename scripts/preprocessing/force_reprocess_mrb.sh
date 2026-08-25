#!/bin/bash
set -e

DIR="/mnt/big/project_ssd/project_ssd/MedEye3d.jl/data/pat_6_files"
MRB_FILE=$(ls $DIR/*.mrb | head -n 1)

if [ -z "$MRB_FILE" ]; then
    echo "No .mrb file found in $DIR"
    exit 1
fi

echo "Found MRB: $MRB_FILE"
TEMP_DIR=$(mktemp -d)
echo "Extracting MRB to $TEMP_DIR..."
unzip -q "$MRB_FILE" -d "$TEMP_DIR"

MRML_FILE=$(find "$TEMP_DIR" -name "*.mrml" | head -n 1)
if [ -z "$MRML_FILE" ]; then
    echo "No .mrml file found inside MRB"
    exit 1
fi

echo "Running convert_mrb_to_nifti.py on $MRML_FILE..."
source /home/jm/project_ssd/superVoxelJuliaCode_lin_sampl/venv/bin/activate || true
python3 scripts/ai/convert_mrb_to_nifti.py "$MRML_FILE" "$DIR"

echo "Running preprocess_dataset.jl..."
julia --project=. scripts/preprocessing/preprocess_dataset.jl "$DIR"

echo "Cleaning up..."
rm -rf "$TEMP_DIR"
echo "Done!"
