using HDF5
h5_path = "data/pat_6_files/preprocessed_volumes.h5"
h5open(h5_path, "r") do f
    skelly = read(f["ATLAS/skellytour"])
    println("Skelly min: ", minimum(skelly))
    println("Skelly max: ", maximum(skelly))
    println("Skelly type: ", eltype(skelly))
    println("Skelly > 0 count: ", count(skelly .> 0))
    println("Skelly == 1 count: ", count(skelly .== 1))
    println("Skelly == 2 count: ", count(skelly .== 2))
end
