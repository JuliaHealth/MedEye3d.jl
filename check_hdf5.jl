using HDF5
using JSON

h5_path = "/workspaces/MedEye3d.jl/data/pat_6_files/preprocessed_volumes.h5"

println("Checking HDF5 file: ", h5_path)
h5open(h5_path, "r") do file
    # Check for lesions
    if haskey(file, "lesions")
        lesions_group = file["lesions"]
        println("Found lesions group.")
        for lid in keys(lesions_group)
            lesion = lesions_group[lid]
            println("  Lesion $lid:")
            if haskey(lesion, "expert_edit_audit_trail")
                audit_json = read(lesion["expert_edit_audit_trail"])
                println("    Audit trail: ", audit_json)
            else
                println("    No audit trail.")
            end
            if haskey(lesion, "detailed_anatomy")
                da_json = read(lesion["detailed_anatomy"])
                println("    Detailed anatomy: ", da_json)
            end
            if haskey(lesion, "clinical_context")
                cc_json = read(lesion["clinical_context"])
                println("    Clinical context (keys): ", join(keys(JSON.parse(cc_json)), ", "))
            end
        end
    else
        println("No 'lesions' group found in HDF5.")
    end
end
