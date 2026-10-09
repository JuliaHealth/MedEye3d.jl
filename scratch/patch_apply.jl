content = read("src/display/LesionMetadataWindow.jl", String)
old = """
                    if w.i_selected[] != idx
                        w.i_selected[] = idx
                    end
                    w.selection[] = val_str
                    notify(w.selection)
"""
new = """
                    if w.i_selected[] != idx
                        w.i_selected[] = idx
                    end
                    if w.selection[] != val_str
                        w.selection[] = val_str
                    end
                    notify(w.selection)
                    try notify(w.i_selected) catch; end
"""
content = replace(content, old => new)
write("src/display/LesionMetadataWindow.jl", content)
