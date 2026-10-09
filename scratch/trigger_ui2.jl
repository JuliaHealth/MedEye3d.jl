using MedEye3d
using MedEye3d.AppMain
h5_path = "/workspaces/MedEye3d.jl/data/pat_6_files/preprocessed_volumes.h5"
win = MedEye3d.AppMain.launch_from_h5(h5_path; quad=false)
sleep(20)
exit()
