using MedEye3d
include("/workspaces/MedEye3d.jl/src/packaging/AppMain.jl")

@eval MedEye3dApp function julia_main()
    MedEye3dApp._test_mode_logic = function(makie_win, mainViewer)
        println(">>> Waiting for startup..."); flush(stdout)
        for _ in 1:100; GLFW.PollEvents(); sleep(0.01); end
        
        println(">>> M1 Compare mode..."); flush(stdout)
        makie_win.set_compare_mode[] = true
        for _ in 1:50; GLFW.PollEvents(); sleep(0.01); end
        
        state1 = mainViewer.states[1]
        state5 = mainViewer.states[5]
        println("State 1 mode: ", state1.displayMode)
        println("State 5 mode: ", state5.displayMode)
        println("State 1 imagePos: ", state1.calcDimsStruct.imagePos)
        println("State 5 imagePos: ", state5.calcDimsStruct.imagePos)
        
        println(">>> Launching M2..."); flush(stdout)
        makie_win.trigger_m2[] = true
        for _ in 1:50; GLFW.PollEvents(); sleep(0.01); end
        
        println(">>> M2 Compare mode..."); flush(stdout)
        makie_win.m2_mode[] = "Compare Curr/Next TP"
        for _ in 1:50; GLFW.PollEvents(); sleep(0.01); end
        
        state6 = mainViewer.states[6]
        state10 = mainViewer.states[10]
        println("State 6 mode: ", state6.displayMode)
        println("State 10 mode: ", state10.displayMode)
        
        println(">>> Done!"); flush(stdout)
        exit(0)
    end
    
    append!(empty!(ARGS), ["/workspaces/MedEye3d.jl/data/cases/psma_patient_all_tp/study_all_tp.h5"])
    MedEye3dApp.main_entry()
end

julia_main()
