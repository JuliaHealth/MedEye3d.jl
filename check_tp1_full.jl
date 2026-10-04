using HDF5, Statistics
h5_path = "data/pat_6_files/preprocessed_volumes.h5"
h5open(h5_path, "r") do f
    pet_key = first(filter(k -> occursin("PET", k), keys(f["TFM_Transform_FollowUp_to_Baseline_1.tfm"])))
    pet = read(f["TFM_Transform_FollowUp_to_Baseline_1.tfm/$pet_key"])
    println("TP1 PET min: ", minimum(pet))
    println("TP1 PET max: ", maximum(pet))
    println("TP1 PET zeros: ", sum(pet .== 0))
    println("TP1 PET total: ", length(pet))
end
