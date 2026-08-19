#!/bin/bash
MASTER_DIR=$1
if [ -z "$MASTER_DIR" ]; then
    echo "Usage: $0 <master_data_dir>"
    exit 1
fi

for pat_dir in "$MASTER_DIR"/*/; do
    if [ ! -f "$pat_dir/scene_hierarchy.json" ]; then
        echo "Skipping $pat_dir (no scene_hierarchy.json)"
        continue
    fi
    echo "========================================"
    echo "Preprocessing case: $pat_dir"
    echo "========================================"
    julia --project=. scripts/preprocessing/preprocess_dataset.jl "$pat_dir"
done
