import re

with open("src/display/LesionMetadataWindow.jl", "r") as f:
    content = f.read()

old_block = r'''            btn_type_prostate = Button\(g\[lt_r, 1\], label = "Prostate",   buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10\)
            btn_type_bone     = Button\(g\[lt_r, 2\], label = "Bone Meta",  buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10\)
            btn_type_organ    = Button\(g\[lt_r, 3\], label = "Organ Meta", buttoncolor = ACCENT, labelcolor = TXT, fontsize = 10\)
            btn_type_ln       = Button\(g\[lt_r, 4\], label = "Lymph Node", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10\)
            
            active_lesion_type = Observable\("Organ Meta"\)
            _last_bone_state = Ref\(false\)
            
            function update_type_buttons\(t\)
                # Skip if type unchanged \(saves 4 button color updates \+ bone event\)
                active_lesion_type\[\] == t && return
                active_lesion_type\[\] = t
                btn_type_prostate\.buttoncolor\[\] = \(t == "Prostate"\) \? ACCENT : BG_PNL
                btn_type_bone\.buttoncolor\[\]     = \(t == "Bone Meta"\) \? ACCENT : BG_PNL
                btn_type_organ\.buttoncolor\[\]    = \(t == "Organ Meta"\) \? ACCENT : BG_PNL
                btn_type_ln\.buttoncolor\[\]       = \(t == "Lymph Node" \|\| t == "Lymph Node Meta"\) \? ACCENT : BG_PNL'''

new_block = r'''            btn_type_prostate = Button(g[lt_r, 1], label = "Prostate",   buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
            btn_type_bone     = Button(g[lt_r, 2], label = "Bone Meta",  buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
            btn_type_organ    = Button(g[lt_r, 3], label = "Organ Meta", buttoncolor = ACCENT, labelcolor = TXT, fontsize = 10)
            btn_type_ln       = Button(g[lt_r, 4], label = "Lymph Node", buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
            btn_type_artifact = Button(g[lt_r, 5], label = "Artifact",   buttoncolor = BG_PNL, labelcolor = TXT, fontsize = 10)
            
            active_lesion_type = Observable("Organ Meta")
            _last_bone_state = Ref(false)
            
            function update_type_buttons(t)
                # Skip if type unchanged (saves button color updates + bone event)
                active_lesion_type[] == t && return
                active_lesion_type[] = t
                btn_type_prostate.buttoncolor[] = (t == "Prostate") ? ACCENT : BG_PNL
                btn_type_bone.buttoncolor[]     = (t == "Bone Meta") ? ACCENT : BG_PNL
                btn_type_organ.buttoncolor[]    = (t == "Organ Meta") ? ACCENT : BG_PNL
                btn_type_ln.buttoncolor[]       = (t == "Lymph Node" || t == "Lymph Node Meta") ? ACCENT : BG_PNL
                btn_type_artifact.buttoncolor[] = (t == "Technical Artifact") ? ACCENT : BG_PNL'''

content = re.sub(old_block, new_block, content)

with open("src/display/LesionMetadataWindow.jl", "w") as f:
    f.write(content)

print("Done")
