using JSON

labels = JSON.parsefile("data/pat_6_files/anatomy_out_fixed_ct_0/max_anatomy_labels.json")

appendicular_kws = ["femur", "humerus", "tibia", "scapula", "radius", "carpal", "tarsal"]

bone_kws = ["femur", "hip", "vertebra", "rib", "sacrum", "clavicula", "clavicle",
            "humerus", "scapula", "sternum", "skull", "palate", "bone", "spine",
            "ilium", "ischium", "pubis", "tibia", "radius", "carpal", "tarsal",
            "costal_cartilage", "mandible", "hyoid", "styloid", "zygomatic",
            "cricoid", "thyroid_cartilage"]

results = []
for (id, name) in labels
    name_lc = lowercase(name)
    is_bone = any(kw -> occursin(kw, name_lc), bone_kws)
    if is_bone
        is_app = any(kw -> occursin(kw, name_lc), appendicular_kws)
        push!(results, (name, is_app ? "APPENDICULAR" : "AXIAL"))
    end
end

for (n, loc) in sort(results)
    println(rpad(n, 30), " => ", loc)
end
