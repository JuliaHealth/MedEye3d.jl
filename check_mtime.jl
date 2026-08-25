mrb_path = "/mnt/big/project_ssd/project_ssd/MedEye3d.jl/data/pat_6_files/verification_scene.mrb"
h5_path = "/mnt/big/project_ssd/project_ssd/MedEye3d.jl/data/pat_6_files/preprocessed_volumes.h5"
if isfile(mrb_path) && isfile(h5_path)
    println("MRB is newer: ", mtime(mrb_path) > mtime(h5_path))
end
