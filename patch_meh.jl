path = "src/display/GLFW/MakieEventHandlers.jl"
content = read(path, String)
content = replace(content, "const current_viewer_position = Ref((0, 0, 0))" => "const current_viewer_position = Ref((0, 0, 0))\nconst app_is_loading = Ref(true)")
write(path, content)
