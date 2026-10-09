content = read("src/display/LesionMetadataWindow.jl", String)
old = """
                                    if haskey(field_widgets, "Anatomic Location") && field_widgets["Anatomic Location"] isa Menu
                                        opts = field_widgets["Anatomic Location"].options[]
                                        if !(loc in opts)
                                            field_widgets["Anatomic Location"].options[] = vcat(opts, [loc])
                                        end
                                        field_widgets["Anatomic Location"].selection[] = loc
                                    end
                                    if !isempty(subloc) && haskey(field_widgets, "Anatomical Sublocation") && field_widgets["Anatomical Sublocation"] isa Menu
                                        opts = field_widgets["Anatomical Sublocation"].options[]
                                        if !(subloc in opts)
                                            field_widgets["Anatomical Sublocation"].options[] = vcat(opts, [subloc])
                                        end
                                        field_widgets["Anatomical Sublocation"].selection[] = subloc
                                    end
"""
new = """
                                    if haskey(field_widgets, "Anatomic Location") && field_widgets["Anatomic Location"] isa Menu
                                        opts = field_widgets["Anatomic Location"].options[]
                                        idx = findfirst(==(loc), opts)
                                        if idx === nothing
                                            opts = vcat(opts, [loc])
                                            field_widgets["Anatomic Location"].options[] = opts
                                            idx = length(opts)
                                        end
                                        _set_menu_idx!(field_widgets["Anatomic Location"], idx)
                                        field_widgets["Anatomic Location"].selection[] = loc
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
                                        field_widgets["Anatomical Sublocation"].selection[] = subloc
                                    end
"""
content = replace(content, old => new)
write("src/display/LesionMetadataWindow.jl", content)
