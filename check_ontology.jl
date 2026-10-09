using JSON
d = JSON.parsefile("data/max_anatomy_to_ontology.json")
bones = []
for (k, v) in d
    if get(v, "lesion_type", "") == "Bone Meta"
        push!(bones, (k, get(v, "anatomic_location", "")))
    end
end
for (k, loc) in sort(bones)
    println(rpad(k, 30), " => ", loc)
end
