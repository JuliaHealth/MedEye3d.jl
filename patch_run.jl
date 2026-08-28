using JSON
data_dir = "/mnt/big/project_ssd/project_ssd/pat_6"
p1 = joinpath(data_dir, "anatomy_out", "max_anatomy_labels.json")
p2 = joinpath(data_dir, "anatomy_out_fixed_ct_1", "max_anatomy_labels.json")
println("p1 exists? ", isfile(p1))
println("p2 exists? ", isfile(p2))
