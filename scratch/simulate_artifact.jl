using MedEye3d
using MedEye3d.AppMain
win = MedEye3d.AppMain.launch_from_h5("/workspaces/MedEye3d.jl/data/pat_6_files/preprocessed_volumes.h5"; quad=false)
sleep(25)
exit()
