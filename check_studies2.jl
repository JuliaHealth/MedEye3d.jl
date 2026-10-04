using HDF5
h5_path = "data/pat_6_files/preprocessed_volumes.h5"
h5open(h5_path, "r") do f
    if haskey(f, "_meta_/studies")
        data = read(f["_meta_/studies"])
        for (i, row) in enumerate(eachrow(data))
            println("TP$(i-1): ", join(row, ", "))
        end
    else
        println("No studies metadata!")
    end
end
