path = "src/packaging/AppMain.jl"
content = read(path, String)

code = """
    if !MedEye3d.InferenceClient.is_ai_enabled()
        MEH.set_ai_status!("[AI Disabled] (Viewer Mode)")
    end

    put!(mainViewer.channel, MedEye3d.MakieEvents.SetWindowTitleEvent("MedEye3d - LOADING APPLICATION... PLEASE WAIT"))

    @async begin
        sleep(6.0) # Wait for JIT warmup and bone subsegments
        try
            put!(mainViewer.channel, MedEye3d.MakieEvents.SetWindowTitleEvent("MedEye3d - Ready"))
            if makie_win !== nothing
                obs = getfield(makie_win, :_lmw_observables)
                if haskey(obs, :loading_overlay_bg)
                    obs[:loading_overlay_bg].visible = false
                    obs[:loading_overlay_txt].visible = false
                end
            end
        catch e
            @warn "Failed to hide loading screen: \$e"
        end
    end

    println("MedEye3D interactive clinical workflow initialized.")
"""

content = replace(content, """
    if !MedEye3d.InferenceClient.is_ai_enabled()
        MEH.set_ai_status!("[AI Disabled] (Viewer Mode)")
    end

    println("MedEye3D interactive clinical workflow initialized.")
""" => code)

write(path, content)
