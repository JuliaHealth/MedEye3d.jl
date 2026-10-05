using HDF5
h5_path = "data/pat_6_files/preprocessed_volumes.h5"
h5open(h5_path, "r") do f
    println("Keys in root: ", keys(f))
    if haskey(f, "BASELINE")
        println("Keys in BASELINE: ", keys(f["BASELINE"]))
    end
    if haskey(f, "ATLAS")
        println("Keys in ATLAS: ", keys(f["ATLAS"]))
    end
end
