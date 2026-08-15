using Pkg; Pkg.activate(".")
using MedEye3d
using MedEye3d.LesionTracker

data_dir = joinpath(@__DIR__, "..", "data", "pat_6_files")
output_path = joinpath(@__DIR__, "lesion_tracking_report.json")
entries = track_lesions(data_dir; output_path=output_path)

println("Tracked $(length(entries)) lesion observations across $(length(unique(e.group_id for e in entries))) groups")
println("Report saved to: $output_path")
