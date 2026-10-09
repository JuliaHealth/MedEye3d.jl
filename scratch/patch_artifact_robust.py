import re

with open("src/display/LesionMetadataWindow.jl", "r") as f:
    content = f.read()

old_block = r'''            idx = findfirst\(==\("Technical Artifact"\), opts\)
            if idx !== nothing
                _set_menu_idx!\(w, idx\)
                if w\.selection\[\] != "Technical Artifact"
                    w\.selection\[\] = "Technical Artifact"
                end
            end'''

new_block = r'''            idx = findfirst(==("Technical Artifact"), opts)
            if idx === nothing
                opts = vcat(opts, ["Technical Artifact"])
                w.options[] = opts
                idx = length(opts)
            end
            _set_menu_idx!(w, idx)
            if w.selection[] != "Technical Artifact"
                w.selection[] = "Technical Artifact"
            end'''

content = re.sub(old_block, new_block, content)

with open("src/display/LesionMetadataWindow.jl", "w") as f:
    f.write(content)

print("Done")
