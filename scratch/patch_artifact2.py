import re

with open("src/display/LesionMetadataWindow.jl", "r") as f:
    content = f.read()

old_block = r'''            on\(btn_type_prostate\.clicks\) do _; update_type_buttons\("Prostate"\) end
            on\(btn_type_bone\.clicks\)     do _; update_type_buttons\("Bone Meta"\) end
            on\(btn_type_organ\.clicks\)    do _; update_type_buttons\("Organ Meta"\) end
            on\(btn_type_ln\.clicks\)       do _; update_type_buttons\("Lymph Node Meta"\) end'''

new_block = r'''            on(btn_type_prostate.clicks) do _; update_type_buttons("Prostate") end
            on(btn_type_bone.clicks)     do _; update_type_buttons("Bone Meta") end
            on(btn_type_organ.clicks)    do _; update_type_buttons("Organ Meta") end
            on(btn_type_ln.clicks)       do _; update_type_buttons("Lymph Node Meta") end
            on(btn_type_artifact.clicks) do _; update_type_buttons("Technical Artifact") end'''

content = re.sub(old_block, new_block, content)

with open("src/display/LesionMetadataWindow.jl", "w") as f:
    f.write(content)

print("Done")
