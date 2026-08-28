using JSON
include("scripts/lib/SceneHierarchy.jl")
println("Testing parse_studies_from_hierarchy...")
data_dir = "/mnt/big/project_ssd/project_ssd/pat_6"
if isdir(data_dir)
    studies = parse_studies_from_hierarchy(data_dir)
    println(studies)
end
