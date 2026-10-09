using MedEye3d
using MedEye3d.AppMain

h5_path = "/workspaces/MedEye3d.jl/data/pat_6_files/preprocessed_volumes.h5"
using HDF5, JSON

function test_ui()
    println("Initializing UI...")
    win = MedEye3d.AppMain.launch_from_h5(h5_path; quad=false)
    sleep(15) # Wait for it to fully render and apply states
    
    meh = MedEye3d.LesionMetadataWindow._MEH
    db = MedEye3d.LesionMetadataWindow.lesion_db[]
    
    data_1 = get(db, "1", Dict())
    println("data_1['Anatomic Location'] = ", get(data_1, "Anatomic Location", "MISSING"))
    
    fw = MedEye3d.LesionMetadataWindow.field_widgets
    w = fw["Anatomic Location"]
    println("w.selection[] = ", w.selection[])
    println("w.i_selected[] = ", w.i_selected[])
    println("w.options[i] = ", w.options[][w.i_selected[]])
    
    exit()
end
test_ui()
