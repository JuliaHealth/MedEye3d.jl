using MedEye3d
using GLMakie
active = Observable("1")
ids = Observable(["1", "2"]) 
res = MedEye3d.LesionMetadataWindow.create_metadata_window(active, ids, nothing)
sleep(1.0)
db = Dict{String, Any}("1" => Dict{String, Any}("KeyImage" => "true", "ObservationState" => "CORRECTED", "ClinicalProfile" => "POST_RLT", "SegmentName" => "Right Iliac Bone"))
MedEye3d.LesionMetadataWindow._active_lesion_db[][] = db
active[] = "2"; sleep(0.2)
active[] = "1"; sleep(0.2)

# Now we need to print the button text. The button is deep in the UI.
# Let's just extract it via the figure
for block in res.fig.content
    println("Content block: ", typeof(block))
end
