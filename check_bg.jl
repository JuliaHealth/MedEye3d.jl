using HDF5, Statistics
h5_path = "data/pat_6_files/preprocessed_volumes.h5"
h5open(h5_path, "r") do f
    pet_key = first(filter(k -> occursin("PET", k), keys(f["BASELINE"])))
    pet = read(f["BASELINE/$pet_key"])
    min_val = minimum(pet)
    max_val = maximum(pet)
    median_val = median(pet[1:10, 1:10, 1:10]) # sample background corner
    
    # Let us compute SUV
    scale_factor = haskey(attributes(f["BASELINE/$pet_key"]), "suv_scale_factor") ? read(attributes(f["BASELINE/$pet_key"])["suv_scale_factor"]) : 1.0f0
    pet_suv = pet .* scale_factor
    
    println("PET array min: ", min_val, ", max: ", max_val)
    println("PET SUV min: ", minimum(pet_suv), ", max: ", maximum(pet_suv))
    println("PET background corner median SUV: ", median(pet_suv[1:10, 1:10, 1:10]))
end
