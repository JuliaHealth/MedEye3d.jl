using MedEye3d
using HDF5
import MedEye3d.SegmentationDisplay.MakieEventHandlers as MEH

h5_path = "scripts/app/../../data/pat_6_files/preprocessed_volumes.h5"
h5 = h5open(h5_path, "r")
mask = read(h5["BASELINE/PET_Lesions_0.nii.gz"])
atlas = read(h5["BASELINE/max_anatomy.nii.gz"])
close(h5)

println("Mask shape: ", size(mask))
println("Atlas shape: ", size(atlas))

# Fast bone subseg!
res = MEH.compute_bone_subsegments_fast(mask, atlas, 1)
println("Target 1: surf=", length(res[1]), " marr=", length(res[2]))
