using MedEye3d
include("/workspaces/MedEye3d.jl/src/packaging/AppMain.jl")

@eval MedEye3dApp function julia_main()
    append!(empty!(ARGS), ["/workspaces/MedEye3d.jl/data/cases/psma_patient_all_tp/study_all_tp.h5"])
    MedEye3dApp.main_entry()
end

julia_main()
