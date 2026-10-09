content = read("src/display/LesionMetadataWindow.jl", String)
old = """
            if w isa Textbox
                target_str = val === nothing ? "" : String(val)
                _set_tb_val!(w, target_str)
            elseif w isa Menu
"""
new = """
            if w isa Textbox
                target_str = val === nothing ? "" : String(val)
                _set_tb_val!(w, target_str)
            elseif w isa Menu
                if q.short == "Anatomic Location"
                    println("[DEBUG-PREFILL] Found Anatomic Location Menu! val='\$val'")
                end
"""
content = replace(content, old => new)
write("src/display/LesionMetadataWindow.jl", content)
