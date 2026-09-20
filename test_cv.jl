using MedEye3d
include("/workspaces/MedEye3d.jl/src/packaging/AppMain.jl")

@eval MedEye3dApp function julia_main()
    MedEye3dApp._test_mode_logic = function(makie_win, mainViewer)
        println(">>> Waiting for startup..."); flush(stdout)
        for _ in 1:100; GLFW.PollEvents(); sleep(0.01); end
        
        println(">>> Clicking Compare Volumes button..."); flush(stdout)
        makie_win.set_compare_mode[] = true
        
        println(">>> Wait a bit..."); flush(stdout)
        for _ in 1:200; GLFW.PollEvents(); sleep(0.01); end
        
        println(">>> Done!"); flush(stdout)
        exit(0)
    end
    
    append!(empty!(ARGS), ["/workspaces/MedEye3d.jl/data/cases/psma_patient_all_tp/study_all_tp.h5"])
    MedEye3dApp.main_entry()
end

julia_main()
