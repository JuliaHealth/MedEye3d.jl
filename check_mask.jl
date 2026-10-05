using HDF5
h5_path = "data/pat_6_files/preprocessed_volumes.h5"
h5open(h5_path, "r") do f
    mask = read(f["BASELINE/PET_Lesions_0.nii.gz"])
    println("Mask min: ", minimum(mask))
    println("Mask max: ", maximum(mask))
    println("Mask unique: ", unique(mask))
end
