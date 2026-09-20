using MedEye3d
include("/workspaces/MedEye3d.jl/src/packaging/AppMain.jl")
append!(empty!(ARGS), ["/workspaces/MedEye3d.jl/data/cases/psma_patient_all_tp/study_all_tp.h5"])

@eval MedEye3dApp function julia_main()
    MedEye3dApp._test_mode_logic = function(makie_win, mainViewer)
        println(">>> Waiting for startup...")
        for _ in 1:100; GLFW.PollEvents(); sleep(0.01); end
        
        println(">>> Clicking Compare button...")
        makie_win.cv_active[] = true
        put!(mainViewer.channel, MedEye3d.SegmentationDisplay.MakieEventHandlers.CompareTimePointsEvent(true))
        
        for _ in 1:100; GLFW.PollEvents(); sleep(0.01); end
        println(">>> Clicking Coronal plane button...")
        put!(mainViewer.channel, MedEye3d.SegmentationDisplay.MakieEventHandlers.ChangePlaneEvent(:Coronal))
        for _ in 1:100; GLFW.PollEvents(); sleep(0.01); end
        
        println(">>> Finished waiting after compare mode ON")
        
        try
            if isopen(mainViewer.channel)
                close(mainViewer.channel)
            end
        catch; end
    end
    MedEye3dApp.main_entry()
end

julia_main()
