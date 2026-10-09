content = read("src/display/LesionMetadataWindow.jl", String)

old_code = """
        # g.block_updates = true
        try
            if cv_active[]
"""

new_code = """
        was_blocked = g.block_updates
        g.block_updates = false
        try
            if cv_active[]
"""
content = replace(content, old_code => new_code)

old_code2 = """
            end
        finally
            @async begin
"""

new_code2 = """
            end
            Makie.GridLayoutBase.update!(g)
        finally
            @async begin
"""
content = replace(content, old_code2 => new_code2)

write("src/display/LesionMetadataWindow.jl", content)
