using HDF5
using JSON

h5_path = joinpath(homedir(), "medeye3d_lesion_annotations.h5")

h5open(h5_path, "r") do file
    has_audit = 0
    has_da = 0
    for lid in keys(file)
        if lid == "_GLOBAL_APP_STATE" continue end
        lesion = file[lid]
        if haskey(lesion, "expert_edit_audit_trail")
            audit_json = read(lesion["expert_edit_audit_trail"])
            println("Lesion $lid - Audit trail: ", audit_json)
            has_audit += 1
        end
        if haskey(lesion, "Anatomical Details")
            da_json = read(lesion["Anatomical Details"])
            println("Lesion $lid - Anatomical Details: ", da_json)
            has_da += 1
        end
        # Print all keys for the first lesion to see what's actually there
        if lid == first(keys(file))
            println("Keys for first lesion ($lid): ", join(keys(lesion), ", "))
        end
    end
    println("Total with audit: $has_audit")
    println("Total with Anatomical Details: $has_da")
end
