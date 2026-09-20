using MedEye3d
include("/workspaces/MedEye3d.jl/src/packaging/AppMain.jl")
@eval MedEye3d.AppMain begin
    function julia_main3()
        global _test_mode_logic = function(makie_win, mainViewer)
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
        
        launch_from_h5("/workspaces/MedEye3d.jl/data/cases/psma_patient_all_tp/study_all_tp.h5"; quad=true)
    end
end
MedEye3d.AppMain.julia_main3()
