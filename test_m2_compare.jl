using MedEye3d
include("/workspaces/MedEye3d.jl/src/packaging/AppMain.jl")

@eval MedEye3dApp function julia_main()
    MedEye3dApp._test_mode_logic = function(makie_win, mainViewer)
        println(">>> Waiting for startup..."); flush(stdout)
        for _ in 1:100; GLFW.PollEvents(); sleep(0.01); end
        
        println(">>> Launching M2..."); flush(stdout)
        makie_win.trigger_m2[] = true
        for _ in 1:50; GLFW.PollEvents(); sleep(0.01); end
        
        println(">>> Selecting Compare Mode..."); flush(stdout)
        makie_win.m2_mode[] = "Compare Curr/Next TP"
        
        for _ in 1:50; GLFW.PollEvents(); sleep(0.01); end
        println(">>> Finished!"); flush(stdout)
        
        try
            if isopen(mainViewer.channel)
                close(mainViewer.channel)
            end
        catch; end
    end
    
    append!(empty!(ARGS), ["/workspaces/MedEye3d.jl/data/cases/psma_patient_all_tp/study_all_tp.h5"])
    MedEye3dApp.main_entry()
end

julia_main()
