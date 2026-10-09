content = read("src/display/LesionMetadataWindow.jl", String)

# Patch 1: Makie Menu prefill for existing lesions
old1 = "if w.i_selected[] != idx\n                        w.i_selected[] = idx\n                    end\n                    w.selection[] = val_str"
new1 = "if w.i_selected[] != idx\n                        w.i_selected[] = idx\n                    end\n                    if w.selection[] != val_str\n                        w.selection[] = val_str\n                    end\n                    notify(w.selection)"
content = replace(content, old1 => new1)

write("src/display/LesionMetadataWindow.jl", content)
