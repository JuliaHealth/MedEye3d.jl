content = read("src/display/LesionMetadataWindow.jl", String)
content = replace(content, "GLOBAL_custom_opts_path()" => "global_custom_opts_path()")
write("src/display/LesionMetadataWindow.jl", content)
