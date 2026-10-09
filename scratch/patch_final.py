import re

with open("src/display/LesionMetadataWindow.jl", "r") as f:
    content = f.read()

old1 = r'''                    if w\.i_selected\[\] != idx
                        w\.i_selected\[\] = idx
                    end
                    w\.selection\[\] = val_str'''

new1 = r'''                    if w.i_selected[] != idx
                        w.i_selected[] = idx
                    end
                    if w.selection[] != val_str
                        w.selection[] = val_str
                    end
                    notify(w.selection)'''
                    
content = re.sub(old1, new1, content)

old2 = r'''                                    if haskey\(field_widgets, "Anatomic Location"\) && field_widgets\["Anatomic Location"\] isa Menu
                                        opts = field_widgets\["Anatomic Location"\]\.options\[\]
                                        if !\(loc in opts\)
                                            field_widgets\["Anatomic Location"\]\.options\[\] = vcat\(opts, \[loc\]\)
                                        end
                                        field_widgets\["Anatomic Location"\]\.selection\[\] = loc
                                    end
                                    if !isempty\(subloc\) && haskey\(field_widgets, "Anatomical Sublocation"\) && field_widgets\["Anatomical Sublocation"\] isa Menu
                                        opts = field_widgets\["Anatomical Sublocation"\]\.options\[\]
                                        if !\(subloc in opts\)
                                            field_widgets\["Anatomical Sublocation"\]\.options\[\] = vcat\(opts, \[subloc\]\)
                                        end
                                        field_widgets\["Anatomical Sublocation"\]\.selection\[\] = subloc
                                    end'''

new2 = r'''                                    if haskey(field_widgets, "Anatomic Location") && field_widgets["Anatomic Location"] isa Menu
                                        opts = field_widgets["Anatomic Location"].options[]
                                        idx = findfirst(==(loc), opts)
                                        if idx === nothing
                                            opts = vcat(opts, [loc])
                                            field_widgets["Anatomic Location"].options[] = opts
                                            idx = length(opts)
                                        end
                                        _set_menu_idx!(field_widgets["Anatomic Location"], idx)
                                        if field_widgets["Anatomic Location"].selection[] != loc
                                            field_widgets["Anatomic Location"].selection[] = loc
                                        end
                                    end
                                    if !isempty(subloc) && haskey(field_widgets, "Anatomical Sublocation") && field_widgets["Anatomical Sublocation"] isa Menu
                                        opts = field_widgets["Anatomical Sublocation"].options[]
                                        idx = findfirst(==(subloc), opts)
                                        if idx === nothing
                                            opts = vcat(opts, [subloc])
                                            field_widgets["Anatomical Sublocation"].options[] = opts
                                            idx = length(opts)
                                        end
                                        _set_menu_idx!(field_widgets["Anatomical Sublocation"], idx)
                                        if field_widgets["Anatomical Sublocation"].selection[] != subloc
                                            field_widgets["Anatomical Sublocation"].selection[] = subloc
                                        end
                                    end'''

content = re.sub(old2, new2, content)

with open("src/display/LesionMetadataWindow.jl", "w") as f:
    f.write(content)

print("Done")
