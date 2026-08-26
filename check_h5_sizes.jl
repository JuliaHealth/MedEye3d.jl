using HDF5
h5open("data/pat_6_files/preprocessed_volumes.h5", "r") do f
    for g in keys(f)
        if endswith(g, "_DISPLAY")
            println("Group: $g")
            for d in keys(f[g])
                println("  $d: $(size(f[g][d]))")
            end
        end
    end
end
