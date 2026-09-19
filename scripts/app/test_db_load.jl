using MedEye3d
using GLMakie
using HDF5
using JSON

db = Dict{String, Any}(
    "_GLOBAL_APP_STATE" => Dict{String, Any}("ClinicalProfile" => "POST_RLT"),
    "1" => Dict{String, Any}("KeyImage" => "true", "ObservationState" => "CORRECTED", "ClinicalProfile" => "POST_RLT", "SegmentName" => "Right Iliac Bone")
)
open("/tmp/dummy.json", "w") do f; JSON.print(f, db); end

active = Observable("1")
ids = Observable(["1", "2"]) 

res = MedEye3d.LesionMetadataWindow.create_metadata_window(active, ids, nothing; save_path="/tmp/dummy.json")

for i in 1:50
    if !isempty(MedEye3d.LesionMetadataWindow._active_lesion_db[][])
        break
    end
    sleep(0.1)
end

println("Loaded DB: ", MedEye3d.LesionMetadataWindow._active_lesion_db[][])

active[] = "1"
sleep(1.0)
