using HDF5
using JSON

h5_path = joinpath(homedir(), "medeye3d_lesion_annotations.h5")
println("Checking HDF5: ", h5_path)
h5open(h5_path, "r") do file
    has_mod = 0
    for lid in keys(file)
        if lid == "_GLOBAL_APP_STATE" continue end
        lesion = file[lid]
        if haskey(lesion, "_last_modified_by")
            mod_by = read(lesion["_last_modified_by"])
            mod_at = haskey(lesion, "_last_modified_at") ? read(lesion["_last_modified_at"]) : "unknown time"
            
            println("- Lesion ", lid, " was modified by user '", mod_by, "'")
            
            if haskey(lesion, "expert_edit_audit_trail")
                trail = read(lesion["expert_edit_audit_trail"])
                println("  Audit trail: ", trail)
            end
            has_mod += 1
        end
    end
    println("\nTotal lesions modified and tracked: ", has_mod)
end
