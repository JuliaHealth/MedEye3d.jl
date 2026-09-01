#!/usr/bin/env julia
# Validation script: verify anatomy mapping fix on patient 6 data
# CRITICAL: Uses per-TP labels (not baseline) since the atlas uses per-TP numbering

using Pkg
Pkg.activate("/workspaces/MedEye3d.jl")

using HDF5, JSON, Statistics
using MedEye3d.LesionAssociation

h5 = h5open("/workspaces/MedEye3d.jl/data/pat_6_files/preprocessed_volumes.h5", "r")

# Load per-TP labels FIRST (these match the atlas numbering, NOT the baseline labels)
ts_names = Dict{Int,String}()
for k in sort(collect(keys(h5["_meta_"])))
    if startswith(k, "anatomy_labels_tp_")
        tp_raw = JSON.parse(read(h5["_meta_/$k"]))
        for (id_str, name) in tp_raw
            id = parse(Int, id_str)
            if !occursin("_class_", name) && !haskey(ts_names, id)
                ts_names[id] = name
            end
        end
        println("Loaded labels from $k: $(length(ts_names)) real labels")
        break
    end
end

# Load atlas and mask
ts_atlas = read(h5["ATLAS/max_anatomy"])
mask_raw = read(h5["BASELINE/PET_Lesions_0.nii.gz"])
mask_f32 = Float32.(mask_raw)

onto = JSON.parsefile("/workspaces/MedEye3d.jl/data/max_anatomy_to_ontology.json")
println("Ontology: $(length(onto)) entries")

close(h5)

println("\n=== Testing map_lesions_to_organs (volume-based, bone priority) ===")
result = map_lesions_to_organs(mask_f32, ts_atlas, ts_names)

total = length(filter(x -> x > 0, unique(mask_f32)))
mapped = length(result)
unknown = count(v -> v == "Unknown", values(result))

println("\nResults:")
for lid in 1:total
    name = get(result, lid, "NOT MAPPED")
    has_onto = haskey(onto, name)
    loc = has_onto ? onto[name]["anatomic_location"] : "?"
    priority = name != "NOT MAPPED" && name != "Unknown" ? classify_tissue_priority(name) : 0
    p_str = priority == 1 ? "BONE" : priority == 2 ? "ORGAN" : priority == 3 ? "LYMPH" : priority == 4 ? "VESSEL" : priority == 5 ? "MUSCLE" : "?"
    println("  Lesion $lid: $name ($p_str) -> loc=$loc")
end

println("\n=== Summary ===")
println("Total: $total, Mapped: $mapped, Unknown: $unknown, Unmapped: $(total - mapped)")
bone_count = count(v -> classify_tissue_priority(v) == 1, values(result))
muscle_count = count(v -> classify_tissue_priority(v) == 5, values(result))
organ_count = count(v -> classify_tissue_priority(v) == 2, values(result))
println("Bone: $bone_count, Organ: $organ_count, Muscle: $muscle_count")
println("All mapped? $(mapped == total ? "YES" : "NO")")
println("Zero unknown? $(unknown == 0 ? "YES" : "NO")")
