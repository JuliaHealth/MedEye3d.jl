include("scripts/lib/SceneHierarchy.jl")
using .SceneHierarchy
studies = parse_studies_from_hierarchy("data/pat_6_files")
for s in studies
    println(s)
end
