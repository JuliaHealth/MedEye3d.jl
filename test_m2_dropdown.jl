using MedEye3d
include("/workspaces/MedEye3d.jl/src/packaging/AppMain.jl")
@eval MedEye3dApp function julia_main()
    MedEye3dApp._test_mode_logic = function(makie_win, mainViewer)
        println(">>> Waiting for startup..."); flush(stdout)
        for _ in 1:200; yield(); sleep(0.01); end
        
        println(">>> Clicking Compare Volumes button..."); flush(stdout)
        makie_win.set_compare_mode[] = true
        
        println(">>> Waiting to see if CompareTimePointsEvent is executed..."); flush(stdout)
        for _ in 1:200; yield(); sleep(0.01); end
        
        state1 = mainViewer.states[1]
        state5 = mainViewer.states[5]
        println("State 1 mode: ", state1.displayMode)
        println("State 5 mode: ", state5.displayMode)
        
        println(">>> Done!"); flush(stdout)
        exit(0)
    end
    
    append!(empty!(ARGS), ["/workspaces/MedEye3d.jl/data/cases/psma_patient_all_tp/study_all_tp.h5"])
    MedEye3dApp.main_entry()
end
julia_main()
