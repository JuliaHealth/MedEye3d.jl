path_meh = "src/display/GLFW/MakieEventHandlers.jl"
content_meh = read(path_meh, String)
content_meh = replace(content_meh, "const current_viewer_position = Ref((0,0,0))" => "const current_viewer_position = Ref((0,0,0))\nconst app_is_loading = Ref(true)")
write(path_meh, content_meh)

path_seg = "src/display/GLFW/SegmentationDisplay.jl"
content_seg = read(path_seg, String)
content_seg = replace(content_seg, "push_consts[11] = show_crosshair; push_consts[12] = 0.0f0" => "push_consts[11] = show_crosshair; push_consts[12] = MakieEventHandlers.app_is_loading[] ? 0.3f0 : 1.0f0")
write(path_seg, content_seg)

path_sh = "src/display/Vulkan/VulkanShaders.jl"
content_sh = read(path_sh, String)
content_sh = replace(content_sh, "int padding;" => "float loadingFade;")
content_sh = replace(content_sh, "FragColor = vec4(finalColor, 1.0);" => "FragColor = vec4(finalColor * pc.loadingFade, 1.0);")
write(path_sh, content_sh)

path_app = "src/packaging/AppMain.jl"
content_app = read(path_app, String)
content_app = replace(content_app, "obs[:loading_overlay_txt].visible = false" => "obs[:loading_overlay_txt].visible = false\n                end\n                MEH.app_is_loading[] = false\n                for win in [mainViewer.states[1].mainForDisplayObjects.window, m2_window_cache[]]\n                    if win !== nothing\n                        put!(mainViewer.channel, MedEye3d.MakieEvents.RenderRequestEvent())\n                    end")
write(path_app, content_app)
