using MedEye3d
using GLMakie

active = Observable("1")
ids = Observable(["1", "2"]) 

# We can intercept ui_hooks!
hooks = Dict{Symbol, Observable}()

res = MedEye3d.LesionMetadataWindow.create_metadata_window(active, ids, nothing; ui_hooks=hooks)

# Find btn_star. It's a local variable, we can't easily get it unless we traverse the scene.
# BUT we can check if it worked by just printing _active_lesion_db
sleep(1.0)
println("Before injection: ", MedEye3d.LesionMetadataWindow._active_lesion_db[][])

db = Dict{String, Any}("1" => Dict{String, Any}("KeyImage" => "true", "ObservationState" => "CORRECTED", "ClinicalProfile" => "POST_RLT", "SegmentName" => "Right Iliac Bone"))

MedEye3d.LesionMetadataWindow._active_lesion_db[][] = db
active[] = "2"; sleep(0.2)
active[] = "1"; sleep(0.2)

println("After trigger: ", MedEye3d.LesionMetadataWindow._active_lesion_db[][])

