using MedEye3d

db = copy(MedEye3d.LesionMetadataWindow.lesion_db[])
db["TEST_LID"] = Dict("Anatomical Details" => "Inside:Prostate", "_last_modified_by" => "jm")
MedEye3d.LesionMetadataWindow.lesion_db[] = db

# Manually call save
ch = MedEye3d.LesionMetadataWindow.db_channel
put!(ch, MedEye3d.LesionMetadataWindow.SaveDBMessage(db, Dict{String,Any}(), "test.json", "test.h5"))
sleep(2)
println("Saved!")
