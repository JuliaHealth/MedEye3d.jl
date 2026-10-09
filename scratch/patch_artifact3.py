import re

with open("src/display/LesionMetadataWindow.jl", "r") as f:
    content = f.read()

old_block = r'''    on\(btn_type_ln\.clicks\) do _
        update_type_buttons\("Lymph Node Meta"\)
        if haskey\(field_widgets, "Anatomic Location"\) && field_widgets\["Anatomic Location"\] isa Menu
            field_widgets\["Anatomic Location"\]\.selection\[\] = "Pelvic Lymph Node"
        end
    end'''

new_block = r'''    on(btn_type_ln.clicks) do _
        update_type_buttons("Lymph Node Meta")
        if haskey(field_widgets, "Anatomic Location") && field_widgets["Anatomic Location"] isa Menu
            field_widgets["Anatomic Location"].selection[] = "Pelvic Lymph Node"
        end
    end

    on(btn_type_artifact.clicks) do _
        update_type_buttons("Technical Artifact")
        no_ct_toggle.active[] = true
        if haskey(field_widgets, "Certainty")
            cert_w = field_widgets["Certainty"]
            if cert_w isa Slider
                cert_w.selected_index[] = 1
            end
        end
        if haskey(field_widgets, "Alternative Hypothesis (False Positive)") && field_widgets["Alternative Hypothesis (False Positive)"] isa Menu
            w = field_widgets["Alternative Hypothesis (False Positive)"]
            opts = w.options[]
            idx = findfirst(==("Technical Artifact"), opts)
            if idx !== nothing
                _set_menu_idx!(w, idx)
                if w.selection[] != "Technical Artifact"
                    w.selection[] = "Technical Artifact"
                end
            end
        end
        trigger_autosave()
    end'''

content = re.sub(old_block, new_block, content)

with open("src/display/LesionMetadataWindow.jl", "w") as f:
    f.write(content)

print("Done")
