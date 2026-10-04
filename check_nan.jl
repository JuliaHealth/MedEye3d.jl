using HDF5, Statistics
h5_path = "data/pat_6_files/preprocessed_volumes.h5"
h5open(h5_path, "r") do f
    pet_key = first(filter(k -> occursin("PET", k), keys(f["BASELINE"])))
    pet = read(f["BASELINE/$pet_key"])
    println("NaNs in PET: ", sum(isnan.(pet)))
    println("Zeros in PET: ", sum(pet .== 0))
    println("Total voxels: ", length(pet))
end
