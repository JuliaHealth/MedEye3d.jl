using JSON
include("src/display/LesionAssociation.jl")
using .LesionAssociation

d = JSON.parsefile("data/max_anatomy_to_ontology.json")
for (k, v) in d
    new_type = LesionAssociation.classify_organ_to_lesion_type(k)
    # patch
    if v["lesion_type"] == "Lymph Node" && !LesionAssociation.is_lymph_node_structure(k)
        println("Fixing ", k, " from Lymph Node to ", new_type)
        v["lesion_type"] = new_type
    end
end
