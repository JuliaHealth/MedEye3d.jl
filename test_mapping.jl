using JSON
include("src/LesionAssociation.jl")
using NIfTI

data_dir_pat6 = "/mnt/big/project_ssd/project_ssd/MedEye3d.jl/data/pat_6_files"
real_labels_path = joinpath(data_dir_pat6, "anatomy_out", "max_anatomy_labels.json")
raw_labels = JSON.parsefile(real_labels_path)
ts_names = Dict{Int,String}(parse(Int, k) => v for (k, v) in raw_labels)

# We need a mask and ts_atlas_aligned
# Since that takes too much memory/time, let's just grep the map_lesions_to_organs function
