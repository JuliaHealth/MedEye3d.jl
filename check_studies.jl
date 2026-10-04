using HDF5
h5_path = "data/pat_6_files/preprocessed_volumes.h5"
h5open(h5_path, "r") do f
    if haskey(f, "_meta_/studies")
        println(read(f["_meta_/studies"]))
    end
end
