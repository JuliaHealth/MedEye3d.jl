using HDF5, JSON
h5_path = "data/pat_6_files/preprocessed_volumes.h5"
h5open(h5_path, "r") do f
    if haskey(f, "Annotations") && haskey(f["Annotations"], "_GLOBAL_APP_STATE")
        println("_GLOBAL_APP_STATE: ", JSON.parse(read(f["Annotations"]["_GLOBAL_APP_STATE"])))
    else
        println("No _GLOBAL_APP_STATE in HDF5")
    end
end
