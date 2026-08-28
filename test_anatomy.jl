using JSON
# Load the real labels
data_dir_pat6 = "/mnt/big/project_ssd/project_ssd/MedEye3d.jl/data/pat_6_files"
real_labels_path = joinpath(data_dir_pat6, "anatomy_out", "max_anatomy_labels.json")
raw_labels = JSON.parsefile(real_labels_path)
ts_names = Dict{Int,String}(parse(Int, k) => v for (k, v) in raw_labels)
println("Loaded $(length(ts_names)) real labels")
println("18 -> ", get(ts_names, 18, "NOT FOUND"))

# Check TP1 labels
tp1_path = joinpath(data_dir_pat6, "anatomy_out_fixed_ct_1", "max_anatomy_labels.json")
tp1_labels = JSON.parsefile(tp1_path)
println("TP1 18 -> ", get(tp1_labels, "18", "NOT FOUND"))
println("13 -> ", get(ts_names, 13, "NOT FOUND"))
println("TP1 13 -> ", get(tp1_labels, "13", "NOT FOUND"))
