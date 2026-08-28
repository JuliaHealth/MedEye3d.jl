using JSON

data_dir_pat6 = "/mnt/big/project_ssd/project_ssd/MedEye3d.jl/data/pat_6_files"
real_labels_path = joinpath(data_dir_pat6, "anatomy_out", "max_anatomy_labels.json")
raw_labels = JSON.parsefile(real_labels_path)
global_ts_names = Dict{Int,String}(parse(Int, k) => v for (k, v) in raw_labels)

tp1_path = joinpath(data_dir_pat6, "anatomy_out_fixed_ct_1", "max_anatomy_labels.json")
tp1_labels = JSON.parsefile(tp1_path)
cache_tp1 = Dict{Int,String}(parse(Int, k) => v for (k, v) in tp1_labels)

anat_val = 18

lbl = get(cache_tp1, anat_val, "")
println("lbl from cache: ", lbl)

if isempty(lbl) || occursin("class_", lbl)
    anat_name = get(global_ts_names, anat_val, "")
    println("Fallback hit! anat_name: ", anat_name)
else
    anat_name = lbl
    println("No fallback. anat_name: ", anat_name)
end
