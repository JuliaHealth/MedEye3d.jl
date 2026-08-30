using Pkg; Pkg.activate(".")
using MedEye3d
using MedEye3d.LesionTracker

data_dir = joinpath(@__DIR__, "..", "data", "pat_6_files")
entries = track_lesions(data_dir)

println("Tracked $(length(entries)) lesion observations across $(length(unique(e.group_id for e in entries))) groups")
println("Report saved to: $(joinpath(data_dir, "lesion_tracking_report.json"))")
