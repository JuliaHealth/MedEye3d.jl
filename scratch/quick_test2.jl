using JSON
h5_path = "/workspaces/MedEye3d.jl/data/pat_6_files/preprocessed_volumes.h5"
dict_path = joinpath(dirname(h5_path), "medeye3d_lesion_annotations.json")
ldb = JSON.parse(read(dict_path, String))
println("Anatomic Location in JSON = '", get(ldb["1"], "Anatomic Location", "MISSING"), "'")
