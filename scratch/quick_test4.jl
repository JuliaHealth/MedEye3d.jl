using JSON, HDF5
h5_path = "/workspaces/MedEye3d.jl/data/pat_6_files/preprocessed_volumes.h5"
h5 = h5open(h5_path, "r")
h5_meta = h5["_meta_"]
tp_segment_names = Dict()
if haskey(h5_meta, "segment_names")
    sn_str = read(h5_meta["segment_names"])
    tp_segment_names = JSON.parse(sn_str)
end
println("Segment name for TP 0 Lid 1: ", get(get(tp_segment_names, "0", Dict()), "1", "MISSING"))
