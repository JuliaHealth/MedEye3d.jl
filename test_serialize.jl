using MedEye3d

# Create mock data
db = Dict{String, Any}("1" => Dict{String,Any}("Anatomical Details" => "Inside / Contained In:Prostate"))

json_path = "test_out.json"
h5_path = "test_out.h5"

MedEye3d.LesionMetadataWindow.save_annotations(db, json_path)
MedEye3d.LesionMetadataWindow.save_annotations_hdf5(db, h5_path)

db_h5 = MedEye3d.LesionMetadataWindow.load_annotations_hdf5(h5_path)
println("Loaded from H5: ", db_h5["1"]["Anatomical Details"])
