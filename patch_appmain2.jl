path = "src/packaging/AppMain.jl"
content = read(path, String)

code = """
                    obs[:loading_overlay_bg].visible = false
                    obs[:loading_overlay_txt].visible = false
                end
            end
            
            MEH.app_is_loading[] = false
            put!(mainViewer.channel, MedEye3d.MakieEvents.RenderRequestEvent())
        catch e
"""

content = replace(content, r"                    obs\[:loading_overlay_bg\]\.visible = false\n                    obs\[:loading_overlay_txt\]\.visible = false\n                end\n                MEH\.app_is_loading\[\] = false\n                for win in \[mainViewer\.states\[1\]\.mainForDisplayObjects\.window, m2_window_cache\[\]\]\n                    if win !== nothing\n                        put!\(mainViewer\.channel, MedEye3d\.MakieEvents\.RenderRequestEvent\(\)\)\n                    end\n                end\n            end\n        catch e" => code)
write(path, content)
