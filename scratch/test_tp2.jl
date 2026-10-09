using HDF5, JSON
h5_path = "/workspaces/MedEye3d.jl/data/pat_6_files/preprocessed_volumes.h5"
h5 = h5open(h5_path, "r")
h5_meta = h5["_meta_"]
org_str = read(h5_meta["organ_mapping"])
organ_mapping_raw = JSON.parse(org_str)

tp_organ_mapping = Dict{Int, Dict{Int, String}}()
for (tp_str, mapping) in organ_mapping_raw
    tp_idx = parse(Int, tp_str)
    tp_dict = Dict{Int, String}()
    for (lid_str, organ_name) in mapping
        tp_dict[parse(Int, lid_str)] = organ_name
    end
    tp_organ_mapping[tp_idx] = tp_dict
end

println("tp_organ_mapping keys: ", collect(keys(tp_organ_mapping)))
tp_0 = tp_organ_mapping[0]
println("Lid 1 in TP 0: ", get(tp_0, 1, "NOT_FOUND"))
