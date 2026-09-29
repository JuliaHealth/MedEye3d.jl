using MedEye3d

# Load existing DB
db = MedEye3d.LesionMetadataWindow.load_annotations_hdf5(MedEye3d.LesionMetadataWindow.DEFAULT_HDF5_PATH)
migrated = MedEye3d.LesionMetadataWindow._migrate_db(db)

# Create a test lesion
lid = "999"
migrated[lid] = Dict{String,Any}("Anatomical Details" => "Inside / Contained In:Prostate")

# Save it
MedEye3d.LesionMetadataWindow.save_annotations_hdf5(migrated, MedEye3d.LesionMetadataWindow.DEFAULT_HDF5_PATH)

# Load it back
db2 = MedEye3d.LesionMetadataWindow.load_annotations_hdf5(MedEye3d.LesionMetadataWindow.DEFAULT_HDF5_PATH)
println("Loaded back: ", get(db2[lid], "Anatomical Details", "MISSING"))
