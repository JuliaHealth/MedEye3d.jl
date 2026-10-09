content = read("src/display/LesionMetadataWindow.jl", String)
old = """
                    if w.i_selected[] != idx
                        w.i_selected[] = idx
                    end
                    w.selection[] = val_str
"""
new = """
                    if w.i_selected[] != idx
                        w.i_selected[] = idx
                    end
                    w.selection[] = val_str
                    notify(w.selection)
"""
content = replace(content, old => new)
write("src/display/LesionMetadataWindow.jl", content)
