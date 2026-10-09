using HDF5
h5 = h5open("../MedEye3d.jl/data/pat_6_files/preprocessed_volumes.h5", "r")
println("Has BONE_SUBSEG: ", haskey(h5, "BONE_SUBSEG"))
if haskey(h5, "BONE_SUBSEG")
    bone_grp = h5["BONE_SUBSEG"]
    println("Num keys: ", length(keys(bone_grp)))
    println("Sample keys: ", first(keys(bone_grp), 10))
    for k in first(keys(bone_grp), 5)
        d = read(bone_grp[k])
        println("  $k: type=$(typeof(d)), size=$(size(d)), num_nonzero=$(sum(d .> 0))")
    end
end
close(h5)
