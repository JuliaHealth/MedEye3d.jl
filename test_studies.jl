include("scripts/lib/SceneHierarchy.jl")
using .SceneHierarchy
studies = parse_studies_from_hierarchy("/mnt/big/project_ssd/project_ssd/MedEye3d.jl/data/pat_6_files")
for (i, s) in enumerate(studies)
    println("s_idx=$i: node_name=", s[7], " orig_tp=", s[2])
end
