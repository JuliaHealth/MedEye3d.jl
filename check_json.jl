using JSON
json_path = "data/pat_6_files/medeye3d_lesion_annotations.json"
if isfile(json_path)
    d = JSON.parsefile(json_path)
    if haskey(d, "_GLOBAL_APP_STATE")
        println("_GLOBAL_APP_STATE: ", d["_GLOBAL_APP_STATE"])
    else
        println("No _GLOBAL_APP_STATE in JSON")
    end
else
    println("No json")
end
