using MedEye3d
using MedEye3d.AppMain
using HDF5, JSON

win = MedEye3d.AppMain.launch_from_h5("/workspaces/MedEye3d.jl/data/pat_6_files/preprocessed_volumes.h5"; quad=false)
sleep(5)
meh = MedEye3d.LesionMetadataWindow._MEH
fw = MedEye3d.LesionMetadataWindow.field_widgets

println("Initial selection: ", fw["Anatomic Location"].selection[])
# simulate paint event for lid 2 on femur_right
meh.organ_mapping_updated[](2, "femur_right")
sleep(1)
println("After paint selection: ", fw["Anatomic Location"].selection[])
exit()
