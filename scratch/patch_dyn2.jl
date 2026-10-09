content = read("src/display/LesionMetadataWindow.jl", String)

old_code = """
        # g.block_updates = true
        try
            for (sq, rows) in q_row_indices
"""

new_code = """
        was_blocked = g.block_updates
        g.block_updates = false
        try
            for (sq, rows) in q_row_indices
"""
content = replace(content, old_code => new_code)
write("src/display/LesionMetadataWindow.jl", content)
