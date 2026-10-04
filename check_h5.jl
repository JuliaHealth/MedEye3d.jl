using HDF5, JSON
h5_path = "data/pat_6_files/medeye3d_lesion_annotations.h5"
if isfile(h5_path)
    h5open(h5_path, "r") do f
        if haskey(f, "Annotations") && haskey(f["Annotations"], "_GLOBAL_APP_STATE")
            println("_GLOBAL_APP_STATE in HDF5: ", JSON.parse(read(f["Annotations"]["_GLOBAL_APP_STATE"])))
        else
            println("No _GLOBAL_APP_STATE in HDF5")
        end
    end
else
    println("No h5 file")
end
