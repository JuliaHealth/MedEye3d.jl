content = read("src/display/LesionMetadataWindow.jl", String)
content = replace(content, "if w.i_selected[] != idx\n                        w.i_selected[] = idx\n                    end" => "if w.i_selected[] != idx\n                        w.i_selected[] = idx\n                    end\n                    w.selection[] = val_str")
write("src/display/LesionMetadataWindow.jl", content)
