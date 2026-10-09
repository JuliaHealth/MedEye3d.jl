content = read("src/display/LesionMetadataWindow.jl", String)

old_code = """
                for row_idx in rows
                    set_row_visible!(row_idx, visible)
                end
            end
        finally
"""

new_code = """
                for row_idx in rows
                    set_row_visible!(row_idx, visible)
                end
            end
            Makie.GridLayoutBase.update!(g)
        finally
"""
content = replace(content, old_code => new_code)
write("src/display/LesionMetadataWindow.jl", content)
