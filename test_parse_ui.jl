using MedEye3d

MedEye3d.start_medeye3d_events()
fig = MedEye3d.Figure()
res = MedEye3d.create_metadata_window(fig)

sleep(1)

# Inject data into lesion_db
db = copy(MedEye3d.LesionMetadataWindow.lesion_db[])
db["TEST"] = Dict("Anatomical Details" => "Inside / Contained In:Prostate")
MedEye3d.LesionMetadataWindow.lesion_db[] = db

# Simulate clicking a lesion with TEST id
MedEye3d.LesionMetadataWindow.active_lesion_id[] = "TEST"

sleep(1)
# Collect state and see if it was restored
state = MedEye3d.LesionMetadataWindow.get_lesion_state(MedEye3d.LesionMetadataWindow.lesion_db[], "TEST")
println("Lesion TEST from DB: ", get(state, "Anatomical Details", "MISSING"))
