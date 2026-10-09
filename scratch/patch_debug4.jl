content = read("src/display/LesionMetadataWindow.jl", String)
old = """
        println("[APPLY] type=\$(_t_type)ms anat=\$(_t_anatomy)ms suv=\$(_t_suv)ms fields=\$(_t_fields)ms TOTAL=\$(_t_total)ms"); flush(stdout)
        finally
"""
new = """
        println("[APPLY] type=\$(_t_type)ms anat=\$(_t_anatomy)ms suv=\$(_t_suv)ms fields=\$(_t_fields)ms TOTAL=\$(_t_total)ms"); flush(stdout)
        if lid == 1
            w1 = get(field_widgets, "Anatomic Location", nothing)
            w2 = get(field_widgets, "Anatomical Sublocation", nothing)
            if w1 !== nothing
                open("/workspaces/MedEye3d.jl/scratch/ui_state_1.txt", "w") do io
                    println(io, "Anatomic Location: selection=", w1.selection[], ", i_selected=", w1.i_selected[])
                    println(io, "Anatomical Sublocation: selection=", w2.selection[], ", i_selected=", w2.i_selected[])
                    println(io, "Data dict Anatomic Location: ", get(data, "Anatomic Location", "MISSING"))
                    println(io, "raw_organ: ", get(get(_MEH.tp_organ_mapping, _MEH.current_tp_index[], Dict()), lid, "MISSING"))
                end
            end
        end
        finally
"""
content = replace(content, old => new)
write("src/display/LesionMetadataWindow.jl", content)
