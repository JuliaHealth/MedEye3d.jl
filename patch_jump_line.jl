path = "src/display/GLFW/MakieEventHandlers.jl"
content = read(path, String)

new_code = "    idx = findfirst(m -> m.id == data.id, obj.measurements)\n    if idx !== nothing\n        m = obj.measurements[idx]\n        cx, cy, cz = round(Int, m.center_idx[1]), round(Int, m.center_idx[2]), round(Int, m.center_idx[3])\n    else\n        idx_line = findfirst(m -> m.id == data.id, obj.line_measurements)\n        if idx_line !== nothing\n            m = obj.line_measurements[idx_line]\n            cx, cy, cz = round(Int, (m.start_idx[1] + m.end_idx[1])/2), round(Int, (m.start_idx[2] + m.end_idx[2])/2), round(Int, (m.start_idx[3] + m.end_idx[3])/2)\n        else\n            @warn \"Measurement #\$(data.id) not found\"\n            return\n        end\n    end"

content = replace(content, r"    idx = findfirst\(m -> m\.id == data\.id, obj\.measurements\)\n    if idx === nothing\n        @warn \"Measurement #\$\(data\.id\) not found\"\n        return\n    end\n    \n    m = obj\.measurements\[idx\]\n    cx, cy, cz = round\(Int, m\.center_idx\[1\]\), round\(Int, m\.center_idx\[2\]\), round\(Int, m\.center_idx\[3\]\)"s => new_code)
write(path, content)
