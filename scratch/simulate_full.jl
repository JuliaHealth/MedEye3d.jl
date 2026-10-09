using MedEye3d
using MedEye3d.AppMain

win = MedEye3d.AppMain.launch_from_h5("/workspaces/MedEye3d.jl/data/pat_6_files/preprocessed_volumes.h5"; quad=false)
sleep(25)

fw = MedEye3d.LesionMetadataWindow.field_widgets
w1 = fw["Anatomic Location"]
w2 = fw["Anatomical Sublocation"]
println("INIT SELECTION: ", w1.selection[])

# Trigger new lesion paint
meh = MedEye3d.LesionMetadataWindow._MEH
meh.organ_mapping_updated[](2, "femur_right")
sleep(1)

println("AFTER NEW PAINT SELECTION: ", w1.selection[])
exit()
