cp train_champion.jl train_canny_inputs.jl
sed -i 's/include("dataset.jl")/include("dataset_canny.jl")/g' train_canny_inputs.jl
sed -i 's/val_single_lesion_manifest.json/canny_val_manifest.json/g' train_canny_inputs.jl
sed -i 's/train_combined_manifest.json/canny_train_manifest.json/g' train_canny_inputs.jl
sed -i 's/champion_checkpoints/canny_checkpoints/g' train_canny_inputs.jl
